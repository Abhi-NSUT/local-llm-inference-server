"""The contract every model backend must follow. Routes only talk to this interface."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class GenerationParams:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    model: str | None = None              # None = the backend's default model
    format: str | dict | None = None      # "json" or a JSON schema -> output must be valid JSON


@dataclass
class Usage:
    """Filled in by the backend while streaming (one object per request, so it is concurrency-safe)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class GenerationResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"


class Backend(ABC):
    name: str
    model_id: str

    async def startup(self) -> None:
        """Open connections / load the model once when the server starts."""

    async def shutdown(self) -> None:
        """Release resources when the server stops."""

    @abstractmethod
    async def health(self) -> bool: ...

    @abstractmethod
    def stream(self, messages: list[dict], params: GenerationParams, usage: Usage) -> AsyncIterator[str]:
        """Yield pieces of text as the model produces them, and fill `usage` at the end."""

    async def generate(self, messages: list[dict], params: GenerationParams) -> GenerationResult:
        """Non-streaming answer = collect the whole stream."""
        usage = Usage()
        text = "".join([chunk async for chunk in self.stream(messages, params, usage)])
        return GenerationResult(text=text, usage=usage)
