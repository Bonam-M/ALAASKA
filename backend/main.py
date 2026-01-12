from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, field_validator
from collections import defaultdict
from typing import Dict, List
import time
from datetime import datetime, timezone
from jose import jwt
from contextlib import asynccontextmanager
import logging
from openai import AsyncOpenAI
from backend.db_mongo import users_collection, conversations_collection, messages_collection, initialize_database, close_connection
import httpx
import html
import re
import os
import uuid
import requests
from dotenv import load_dotenv

# ========== ENV SETUP ==========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
MODEL_ID = os.getenv("MODEL_ID")
SUMMARIZE_MODEL_ID = os.getenv("SUMMARIZE_MODEL_ID")
ACCESS_TOKEN_EXPIRE_MINUTES = 60
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_API_AUDIENCE = os.getenv("AUTH0_API_AUDIENCE")
ALGORITHM = os.getenv("ALGORITHM")
FRONTEND_URL = os.getenv("FRONTEND_URL")

def validate_environment():
    required_vars = [
        "OPENAI_API_KEY", "SECRET_KEY", "ALGORITHM", "MODEL_ID", "SUMMARIZE_MODEL_ID",
         "AUTH0_DOMAIN","AUTH0_API_AUDIENCE", "ALGORITHM", "FRONTEND_URL"
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
# Call validation before creating app
validate_environment()


# ========== Set up logging ========== Chage WARNING TO INFO for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== STARTUP & SHUTDOWN EVENTS ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await initialize_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    yield  # This is where the app runs
    
    # Shutdown
    try:
        await close_connection()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")

app = FastAPI(lifespan=lifespan)

origins = [
    FRONTEND_URL,
]


# ========== MIDDLEWARES ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if request.headers.get("content-length"):
        content_length = int(request.headers.get("content-length"))
        if content_length > 1_000_000:  # 1MB limit
            raise HTTPException(status_code=413, detail="Request too large")
    response = await call_next(request)
    return response

# ========== MODELS ==========
class User(BaseModel):
    username: str
    email: EmailStr
    password: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=10000, min_length=1)
    chat_id: str | None = None
    
    @field_validator('message')
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        return v.strip()

# ========== UTILS ==========
def get_auth0_jwks():
    jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    jwks = requests.get(jwks_url, timeout=60).json()
    return jwks

def verify_token(token: str):
    jwks = get_auth0_jwks()
    unverified_header = jwt.get_unverified_header(token)
    rsa_key = {}
    for key in jwks["keys"]:
        if key["kid"] == unverified_header["kid"]:
            rsa_key = {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"]
            }
    if not rsa_key:
        raise HTTPException(status_code=401, detail="Unable to find appropriate key")
    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=ALGORITHM,
            audience=AUTH0_API_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/"
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTClaimsError:
        raise HTTPException(status_code=401, detail="Incorrect claims")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload

http_bearer = HTTPBearer()

async def get_userinfo(access_token: str):
    userinfo_url = f"https://{AUTH0_DOMAIN}/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(userinfo_url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to fetch user info")
        return response.json()

async def get_current_user(auth: HTTPAuthorizationCredentials = Depends(http_bearer)):
    token = auth.credentials
    payload = verify_token(token)
    user_info = await get_userinfo(token)

    user_id = payload["sub"]
    email = user_info.get("email", "")
    name = user_info.get("name") or user_info.get("nickname") or user_id

    existing = await users_collection.find_one({"auth0_id": user_id})
    if not existing:
        user_doc = {
            "auth0_id": user_id,
            "username": name,
            "email": email,
            "created_at": datetime.now(timezone.utc)
        }
        await users_collection.insert_one(user_doc)
    else:
        user_doc = existing
    
    return {"auth0_id": user_id, "username": name, "email": email}

def validate_chat_id(chat_id: str):
    if not re.match(r'^[a-f0-9]{8}$', chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID format")

# ========== RATE LIMITING STORAGE ==========
rate_limit_storage: Dict[str, List[float]] = defaultdict(list)

def cleanup_old_requests(user_key: str, window_seconds: int = 60):
    """Remove requests older than the time window"""
    current_time = time.time()
    cutoff_time = current_time - window_seconds
    rate_limit_storage[user_key] = [
        timestamp for timestamp in rate_limit_storage[user_key] 
        if timestamp > cutoff_time
    ]

def check_rate_limit(user_key: str, max_requests: int, window_seconds: int = 60) -> bool:
    """Check if user is within rate limit"""
    cleanup_old_requests(user_key, window_seconds)
    current_requests = len(rate_limit_storage[user_key])
    return current_requests < max_requests

def record_request(user_key: str):
    """Record a successful request"""
    rate_limit_storage[user_key].append(time.time())

async def rate_limit_dependency(request: Request, user: dict = Depends(get_current_user)):
    """Dependency to check rate limits for different endpoints"""
    auth0_id = user["auth0_id"]
    user_key = auth0_id
    
    # Check which endpoint is being called
    path = request.url.path
    
    if path == "/chat/start":
        # 5 new chats per minute
        if not check_rate_limit(f"start:{user_key}", 5):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded for starting new chats. Try again in a minute.",
                headers={"Retry-After": "60"}
            )
    elif path == "/chat":
        # 20 chat messages per minute
        if not check_rate_limit(f"chat:{user_key}", 20):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded for chat messages. Try again in a minute.",
                headers={"Retry-After": "60"}
            )
    
    return user_key

# ========== CHAT ROUTES ==========
@app.post("/chat/start")
async def start_chat(user: dict = Depends(get_current_user), user_key: str = Depends(rate_limit_dependency)):
    chat_id = f"{uuid.uuid4().hex[:8]}"
    username = user["username"]
    auth0_id = user["auth0_id"]
    email = user["email"]

    system_message = {
        "role": "system",
        "content": (
            "You are ALAASKA, a supportive teaching assistant. Your job is to guide the user to think critically and find the solution on their own."
            "Keep the conversation going with a question at the end your replies."
            "Identify the student's level with guiding questions about the topic they are inquiring about."
            "When appropriate throughout the conversation use these: flashcards, mini quizzes, scenarios, hints."
            "Discuss only academic topics and nothing else."
        )
    }

    assistant_welcome = {"role": "assistant", "content": "Hi! Welcome to ALAASKA. How can I help you today?"}
    messages = [system_message, assistant_welcome]

    await conversations_collection.insert_one({
        "chat_id": chat_id,
        "auth0_id": auth0_id,
        "username": username,
        "email": email,
        "summary": "",
        "status": "active", #Unless a conversation is deleted it is active
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    })

    await messages_collection.insert_many([
        {**system_message, "chat_id": chat_id, "timestamp": datetime.now(timezone.utc), "auth0_id": auth0_id, "username": username, "email": email},
        {**assistant_welcome, "chat_id": chat_id, "timestamp": datetime.now(timezone.utc), "auth0_id": auth0_id, "username": username, "email": email},
    ])

    # Record successful request AFTER everything succeeds
    record_request(f"start:{user_key}")
    
    return {"response": assistant_welcome["content"], "chat_id": chat_id, "history": messages}

@app.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user), user_key: str = Depends(rate_limit_dependency)):
    username = user["username"]
    auth0_id = user["auth0_id"]
    email = user["email"]

    # Input validation block
    if not req.message or len(req.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(req.message) > 3000:
        raise HTTPException(status_code=400, detail="Message exceeds maximum length of 3000 characters")
    # End of Input validation block
    try:
        convo = await conversations_collection.find_one({"chat_id": req.chat_id, "auth0_id": auth0_id}) if req.chat_id else None

        if convo:
            chat_id = req.chat_id
            summary = convo.get("summary", "")
        else:
            chat_id = f"{uuid.uuid4().hex[:8]}"
            await conversations_collection.insert_one({
                "chat_id": chat_id,
                "auth0_id": auth0_id,
                "username": username,
                "email": email,
                "summary": "",
                "status": "active",  #Unless deleted, every conversation is active
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            })
            summary = ""

        user_message = {
            "chat_id": chat_id,
            "auth0_id": auth0_id,
            "username": username,
            "email": email,
            "role": "user",
            "content": req.message,
            "timestamp": datetime.now(timezone.utc)
        }
        await messages_collection.insert_one(user_message)

        chat_cursor = messages_collection.find({"chat_id": chat_id}).sort("timestamp", 1)
        chat_messages = [{"role": m["role"], "content": m["content"]} async for m in chat_cursor]

        if not summary or summary == "" or summary == "New Chat":
            summary = await summarize_prompt(req.message)
            await conversations_collection.update_one(
                {"chat_id": chat_id, "auth0_id": auth0_id},
                {"$set": {"summary": summary}}
            )

        response = await client.chat.completions.create(
            model=MODEL_ID,
            messages=chat_messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content

        assistant_message = {
            "chat_id": chat_id,
            "auth0_id": auth0_id,
            "username": username,
            "email": email,
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now(timezone.utc)
        }
        await messages_collection.insert_one(assistant_message)

        await conversations_collection.update_one(
            {"chat_id": chat_id, "auth0_id": auth0_id},
            {"$set": {"updated_at": datetime.now(timezone.utc)}}
        )

        history_cursor = messages_collection.find({"chat_id": chat_id}).sort("timestamp", 1)
        history = [{"role": m["role"], "content": m["content"]} async for m in history_cursor]

        # Record successful request ONLY after everything succeeds
        record_request(f"chat:{user_key}")

        return {"response": reply, "chat_id": chat_id, "history": history}

    except Exception as e:
        import logging
        logging.error(f"Chat error for user {user['auth0_id']}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request")



# ========== CONVERSATION MANAGEMENT ==========
@app.get("/conversations") # Include docs without status field or not deleted
async def list_conversations(user: dict = Depends(get_current_user)):
    auth0_id = user["auth0_id"]
    cursor = conversations_collection.find({"auth0_id": auth0_id,
                                            "$or": [ {"status": {"$exists": False}}, {"status": {"$ne": "deleted"}}] 
                                           })
    conversations = []
    async for convo in cursor:
        conversations.append({
            "chat_id": convo["chat_id"],
            "summary": convo.get("summary", convo["chat_id"][:8]),
            "created_at": convo.get("created_at", "").isoformat() if "created_at" in convo else "",
            "updated_at": convo.get("updated_at", "").isoformat() if "updated_at" in convo else "",
        })
    conversations.sort(key=lambda c: c["updated_at"], reverse=True)
    return conversations

@app.get("/conversation/{chat_id}")
async def get_conversation(chat_id: str, user: dict = Depends(get_current_user)):
    validate_chat_id(chat_id)
    auth0_id = user["auth0_id"]
    
    convo = await conversations_collection.find_one({"chat_id": chat_id, "auth0_id": auth0_id, 
                                                     "$or": [{"status": {"$exists": False}}, {"status": {"$ne": "deleted"}}]
                                                     })
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    
    messages_cursor = messages_collection.find({"chat_id": chat_id, "auth0_id": auth0_id}).sort("timestamp", 1)
    return [{"role": m["role"], "content": m["content"]} async for m in messages_cursor]

@app.put("/conversation/{chat_id}/delete") # Add status deleted but keep conversation in DB
async def delete_conversation(chat_id: str, user: dict = Depends(get_current_user)):
    validate_chat_id(chat_id)
    auth0_id = user["auth0_id"]
    
    result = await conversations_collection.update_one(
        {"chat_id": chat_id, "auth0_id": auth0_id},
        {"$set": {"status": "deleted", "updated_at": datetime.now(timezone.utc)}})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized.")
    
    return {"message": "Conversation marked as deleted successfully."}

async def summarize_prompt(text):
    response = await client.chat.completions.create(
        model=SUMMARIZE_MODEL_ID,
        messages=[
            {"role": "system", "content": "Give a 4-word title to this message"},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip().strip('"')







