"""Request/response shapes. They follow OpenAI's chat format so OpenAI clients work unchanged."""
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None  # accepted for compatibility; the server uses its configured model
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(256, ge=1)
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(0.9, gt=0, le=1)
    stream: bool = False


class GenerateRequest(BaseModel):
    """Simpler endpoint: a single prompt instead of a message list."""
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(256, ge=1)
    temperature: float = Field(0.7, ge=0, le=2)
    stream: bool = False
