import os
from dotenv import load_dotenv

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
MODEL_ID = os.getenv("MODEL_ID")
ASSIGNMENT_MODEL_ID = os.getenv("ASSIGNMENT_MODEL_ID")
SUMMARIZE_MODEL_ID = os.getenv("SUMMARIZE_MODEL_ID")
ACCESS_TOKEN_EXPIRE_MINUTES = 60
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_API_AUDIENCE = os.getenv("AUTH0_API_AUDIENCE")
FRONTEND_URL = os.getenv("FRONTEND_URL","http://localhost:3000")

def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = [
        "OPENAI_API_KEY", "SECRET_KEY", "ALGORITHM", "MODEL_ID", "SUMMARIZE_MODEL_ID",
        "AUTH0_DOMAIN", "AUTH0_API_AUDIENCE", "FRONTEND_URL"
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")