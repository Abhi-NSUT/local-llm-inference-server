"""A pretend model for tests: instant, predictable, no Ollama needed."""
import asyncio

from .base import Backend, GenerationParams, Usage


class FakeBackend(Backend):
    name = "fake"
    model_id = "fake-model"

    def __init__(self, words: list[str] | None = None, delay: float = 0.01):
        self.words = words or ["Hello", " from", " the", " fake", " model", "."]
        self.delay = delay  # seconds between words; tests raise it to simulate a slow model

    async def health(self) -> bool:
        return True

    async def stream(self, messages, params: GenerationParams, usage: Usage):
        usage.prompt_tokens = sum(len(m["content"].split()) for m in messages)
        words = ['{"ok": ', "true}"] if params.format else self.words
        for word in words[: params.max_tokens]:
            await asyncio.sleep(self.delay)
            usage.completion_tokens += 1
            yield word
