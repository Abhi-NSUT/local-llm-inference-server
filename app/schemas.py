"""Request/response shapes. They follow OpenAI's chat format so OpenAI clients work unchanged."""
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class JsonSchemaSpec(BaseModel):
    schema_: dict = Field(alias="schema")


class ResponseFormat(BaseModel):
    """OpenAI-style: {"type": "json_object"} or {"type": "json_schema", "json_schema": {"schema": {...}}}"""
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JsonSchemaSpec | None = None

    def to_backend_format(self) -> str | dict | None:
        if self.type == "json_object":
            return "json"
        if self.type == "json_schema" and self.json_schema:
            return self.json_schema.schema_
        return None


class ChatRequest(BaseModel):
    model: str | None = None  # must be one of the server's allowed models; None = default model
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(256, ge=1)
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(0.9, gt=0, le=1)
    stream: bool = False
    response_format: ResponseFormat | None = None


class GenerateRequest(BaseModel):
    """Simpler endpoint: a single prompt instead of a message list."""
    model: str | None = None
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(256, ge=1)
    temperature: float = Field(0.7, ge=0, le=2)
    stream: bool = False
