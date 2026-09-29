import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.chat_service import ChatService
from app.schemas import ChatRequest, ChatResponse


app = FastAPI(
    title="MINDO Chat Backend",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CHAT SERVICE
# ============================================================

chat_service = ChatService()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "MINDO Chat Backend",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
async def chat_endpoint(payload: ChatRequest):
    return await chat_service.chat(
        message=payload.message,
        conversation=[
            {
                "role": message.role,
                "content": message.content,
            }
            for message in payload.conversation
        ],
    )