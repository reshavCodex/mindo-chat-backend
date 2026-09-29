from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
    )

    conversation: List[ChatMessage] = []


class Source(BaseModel):
    source: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []