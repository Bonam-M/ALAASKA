from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from backend.auth import get_current_user, http_bearer
from backend.models import ChatRequest
from backend.db_mongo import conversations_collection
from backend.db_assignments import student_assignments_collection
from backend.chroma_query import query_homework
from backend.config import OPENAI_API_KEY, MODEL_ID, ASSIGNMENT_MODEL_ID, SUMMARIZE_MODEL_ID
from openai import AsyncOpenAI
from datetime import datetime, timezone
import uuid
import logging
import re

logger = logging.getLogger(__name__)
router = APIRouter()

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are ALAASKA, a supportive teaching assistant. Your job is to guide the user to think critically and find the solution on their own."
    "- When appropriate throughout the conversation use either flashcards, yes or no mini quizzes, scenarios, or hints."
    "- If student asks about multiple questions at once, encourage starting a new chat for each question."
    "- Discuss only academic topics and nothing else."
)
#    "Keep the conversation going with a question at the end your replies."
#    "Identify the student's level with guiding questions about the exact topic they are inquiring about."

async def summarize_title(text: str) -> str:
    try:
        resp = await client.chat.completions.create(
            model=SUMMARIZE_MODEL_ID,
            messages=[
                {"role": "system", "content": "Give a 4-word title to this message. Do not reveal the answer in this summary."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return (resp.choices[0].message.content or "").strip().strip('"')
    except Exception as e:
        logger.warning(f"Title summarization failed: {e}")
        s = (text or "").strip()
        return s if len(s) <= 50 else s[:50] + "..."

def now_utc():
    return datetime.now(timezone.utc)

def now_utc_iso():
    """Return current UTC time as ISO string for message timestamps"""
    return datetime.now(timezone.utc).isoformat()


def parse_iso_timestamp(timestamp: str):
    """Parse ISO timestamp safely for metadata calculations."""
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:
        return None


def lexical_richness(text: str) -> float:
    """Simple type-token ratio used as a lightweight style marker."""
    if not text:
        return 0.0
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)

def create_message(role: str, content: str, extra: dict = None) -> dict:
    """
    Create a message with ML analysis metadata.
    Includes timestamp, char_count, word_count for behavioral analysis.
    """
    msg = {
        "role": role,
        "content": content,
        "timestamp": now_utc_iso(),
        "char_count": len(content),
        "word_count": len(content.split()) if content else 0
    }
    if extra:
        msg.update(extra)
    return msg


# ========== GET ALL CONVERSATIONS ==========

@router.get("/conversations")
async def get_conversations(auth: HTTPAuthorizationCredentials = Depends(http_bearer)):
    """Get all conversations for the current user"""
    user = await get_current_user(auth)
    user_id = user["auth0_id"]
    
    cursor = conversations_collection.find(
        {"user_id": user_id, "is_deleted": False},
        {"_id": 0}
    ).sort("updated_at", -1)
    
    conversations = []
    async for conv in cursor:
        conversations.append({
            "chat_id": conv["chat_id"],
            "summary": conv.get("summary", "New Chat"),
            "created_at": conv["created_at"].isoformat() if "created_at" in conv else None,
            "updated_at": conv["updated_at"].isoformat() if "updated_at" in conv else None,
            "is_assignment_chat": conv.get("is_assignment_chat", False)
        })
    
    return conversations


# ========== GET SINGLE CONVERSATION WITH METADATA ==========

@router.get("/conversation/{chat_id}")
async def get_conversation(
    chat_id: str,
    auth: HTTPAuthorizationCredentials = Depends(http_bearer)
):
    """Get conversation messages with metadata"""
    user = await get_current_user(auth)
    user_id = user["auth0_id"]
    user_email = user["email"].lower()
    is_grader = user.get("is_grader", False)  #  Changed from is_admin

    #  Graders can view any conversation
    if is_grader:
        conversation = await conversations_collection.find_one({
            "chat_id": chat_id,
            "is_deleted": False
        })
    else:
        conversation = await conversations_collection.find_one({
            "chat_id": chat_id,
            "user_id": user_id,
            "is_deleted": False
        })

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Build metadata
    metadata = {
        "is_assignment_chat": conversation.get("is_assignment_chat", False),
        "assignment_id": None,
        "assignment_title": None,
        "question_id": None,
        "question_number": None,
        "has_submitted": False,
        "submitted_message_index": None,
        "attempts": 0,
        "is_grader_view": is_grader and conversation.get("user_id") != user_id  #  Changed
    }

    # If assignment chat, fetch question details
    if conversation.get("is_assignment_chat"):
        assignment_id = conversation.get("assignment_id")
        question_id = conversation.get("question_id")
        
        metadata["assignment_id"] = assignment_id
        metadata["question_id"] = question_id

        #  For grader view, find the student who owns this conversation
        if is_grader and conversation.get("user_id") != user_id:
            from backend.db_mongo import users_collection
            conversation_owner = await users_collection.find_one({"auth0_id": conversation.get("user_id")})
            if conversation_owner:
                user_email = conversation_owner["email"].lower()

        # Get student assignment
        student_assignment = await student_assignments_collection.find_one({
            "assignment_id": assignment_id,
            "student_email": user_email
        })

        if student_assignment:
            metadata["assignment_title"] = student_assignment.get("title", "Assignment")
            
            for q in student_assignment.get("questions", []):
                if q["question_id"] == question_id:
                    metadata["question_number"] = q.get("number", "")
                    metadata["has_submitted"] = q.get("student_solution") is not None
                    metadata["submitted_message_index"] = q.get("submitted_message_index")
                    metadata["attempts"] = q.get("attempts", 0)
                    break

    # Return messages with metadata
    messages = conversation.get("messages", [])
    
    return {
        "messages": messages,
        "metadata": metadata
    }

# ========== DELETE CONVERSATION ==========

@router.put("/conversation/{chat_id}/delete")
async def delete_conversation(chat_id: str, auth: HTTPAuthorizationCredentials = Depends(http_bearer)):
    """Soft delete a conversation"""
    user = await get_current_user(auth)
    user_id = user["auth0_id"]
    
    result = await conversations_collection.update_one(
        {"chat_id": chat_id, "user_id": user_id},
        {"$set": {"is_deleted": True}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation deleted successfully"}


# ========== START NEW CHAT ==========

@router.post("/chat/start")
async def start_chat(auth: HTTPAuthorizationCredentials = Depends(http_bearer)):
    user = await get_current_user(auth)
    user_id = user["auth0_id"]

    chat_id = uuid.uuid4().hex
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Hi! Welcome to ALAASKA. How can I help you today?"}
    ]

    conversation_doc = {
        "chat_id": chat_id,
        "user_id": user_id,
        "messages": initial_messages,
        "summary": "New Chat",
        "created_at": now_utc(),
        "updated_at": now_utc(),
        "is_deleted": False,
        "is_assignment_chat": False,
        "assignment_id": None,
        "question_id": None,
    }

    await conversations_collection.insert_one(conversation_doc)

    return {
        "response": initial_messages[1]["content"],
        "chat_id": chat_id,
        "history": initial_messages
    }


# ========== SEND CHAT MESSAGE ==========

@router.post("/chat")
async def chat(request: ChatRequest, auth: HTTPAuthorizationCredentials = Depends(http_bearer)):
    user = await get_current_user(auth)
    user_id = user["auth0_id"]

    msg_text = (request.message or "").strip()
    if not msg_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(msg_text) > 3000:
        raise HTTPException(status_code=400, detail="Message exceeds maximum length of 3000 characters")

    chat_id = request.chat_id or uuid.uuid4().hex
    existing = None

    if request.chat_id:
        existing = await conversations_collection.find_one({
            "chat_id": request.chat_id,
            "user_id": user_id,
            "is_deleted": False
        })

    if existing:
        messages = existing.get("messages", [])
        summary = existing.get("summary", "New Chat")
        rag_done = existing.get("rag_done", False)
        rag_homework_answers = existing.get("rag_homework_answers", [])
        is_assignment_chat = existing.get("is_assignment_chat", False)
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        summary = "New Chat"
        rag_done = False
        rag_homework_answers = []
        is_assignment_chat = False

    # Check if this is first user message and RAG hasn't been done
    # Skip RAG for assignment chats - they have question-specific context already
    is_first_user_message = len([m for m in messages if m["role"] == "user"]) == 0
    
    if is_first_user_message and not rag_done and not is_assignment_chat:
        try:
            rag_results = await query_homework(msg_text)
            if rag_results:
                rag_homework_answers = rag_results
                logger.info(f"RAG query successful for chat {chat_id}")
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}")
        rag_done = True

    # Append user message (with metadata only for assignment chats)
    if is_assignment_chat:
        inter_turn_response_ms = None
        if messages and messages[-1].get("role") == "assistant":
            prev_assistant_ts = parse_iso_timestamp(messages[-1].get("timestamp"))
            if prev_assistant_ts:
                inter_turn_response_ms = int((now_utc() - prev_assistant_ts).total_seconds() * 1000)

        user_turn_index = len([m for m in messages if m.get("role") == "user"]) + 1
        messages.append(create_message("user", msg_text, {
            "user_turn_index": user_turn_index,
            "inter_turn_response_ms": inter_turn_response_ms,
            "lexical_richness": lexical_richness(msg_text),
        }))
    else:
        messages.append({"role": "user", "content": msg_text})

    # Update SYSTEM_PROMPT if RAG answers exist (only for non-assignment chats)
    # Assignment chats already have their question-specific system prompt
    if rag_homework_answers and not is_assignment_chat:
        SYSTEM_PROMPT_WITH_RAG = (
            "You are ALAASKA, a Socratic teaching assistant. Guide students to think critically and find the solution on their own."
            "Discuss only academic topics and nothing else.\n"
            "[REFERENCE CONTEXT - DO NOT REVEAL]\n"
             f"Reference Answer 1 (ID: {rag_homework_answers[0]['chunk_id']}): "
            f"{rag_homework_answers[0]['answer_text']}\n"
        )
        for i, answer in enumerate(rag_homework_answers[:3], start=1):
            SYSTEM_PROMPT_WITH_RAG += (f"Reference Answer {i} (ID: {answer['chunk_id']}): {answer['answer_text']}\n"
            )
           
        SYSTEM_PROMPT_WITH_RAG += (
            "[END REFERENCE CONTEXT]\n"
            "CORE RULE: NEVER directly give, reveal, or confirm the final answer.\n"
            "If asked directly for the answer, say you can't provide the answer itself and ask them a guiding question to help them find it on their own."
            "ASSESS RELEVANCE: Determine if the reference answers are relevant to the student's question.\n"
            "- If RELEVANT: Use it to guide the student, but don't directly reveal or confirm it. \n"
            "- If NOT RELEVANT: Disregard it completely and rely on your own knowledge. \n"
            "TEACHING METHOD:\n"
            "Guide the student using EXACTLY ONE of these methods:\n"
            "FLASHCARD, MINI QUIZ, SCENARIO or HINT\n"
            "- If student provides the CORRECT ANSWER: Do not reveal or confirm it. Pose a brief question that extends or deepens their understanding.\n"
            "- If student provides an INCORRECT or PARTIAL answer: Continue guiding using one of the three teaching methods.\n"
            "- Maintain an encouraging tone."
        )
        messages[0] = {"role": "system", "content": SYSTEM_PROMPT_WITH_RAG}
        logger.info(
            f"chunk_ids={[r['chunk_id'] for r in rag_homework_answers]}"
        )
    elif not is_assignment_chat:
        # Only update system prompt for non-assignment chats
        messages[0] = {"role": "system", "content": SYSTEM_PROMPT}
    # For assignment chats: keep the original system prompt with question details

    # Strip metadata for OpenAI API call (only send role and content)
    messages_for_api = [{"role": m["role"], "content": m["content"]} for m in messages]
    
    # Use different model based on chat type
    selected_model = ASSIGNMENT_MODEL_ID if is_assignment_chat else MODEL_ID
    
    try:
        # Some assignment models (e.g., gpt-5-mini) don't support temperature parameter
        create_kwargs = {
            "model": selected_model,
            "messages": messages_for_api
        }
        if selected_model != ASSIGNMENT_MODEL_ID:  # Only add temperature for non-assignment models
            create_kwargs["temperature"] = 0.7
        
        resp = await client.chat.completions.create(**create_kwargs)
        reply = resp.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"OpenAI chat error for model {selected_model}: {e}")
        raise HTTPException(status_code=500, detail="LLM error")

    # Append assistant message (with metadata only for assignment chats)
    if is_assignment_chat:
        messages.append(create_message("assistant", reply))
    else:
        messages.append({"role": "assistant", "content": reply})

    if not summary or summary == "" or summary == "New Chat":
        summary = await summarize_title(msg_text)

    conversation_doc = {
        "chat_id": chat_id,
        "user_id": user_id,
        "messages": messages,
        "summary": summary,
        "created_at": existing.get("created_at") if existing else now_utc(),
        "updated_at": now_utc(),
        "is_deleted": existing.get("is_deleted", False) if existing else False,
        "is_assignment_chat": existing.get("is_assignment_chat", False) if existing else False,
        "assignment_id": existing.get("assignment_id") if existing else None,
        "question_id": existing.get("question_id") if existing else None,
        "rag_homework_answers": rag_homework_answers,
        "rag_done": rag_done
    }

    await conversations_collection.replace_one(
        {"chat_id": chat_id, "user_id": user_id},
        conversation_doc,
        upsert=True
    )

    return {
        "response": reply,
        "chat_id": chat_id,
        "messages": messages,
        "summary": summary
    }