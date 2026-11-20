"""All settings come from environment variables (or the .env file)."""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    backend: Literal["ollama", "fake"] = "ollama"
    model_id: str = "qwen2.5:1.5b"  # default model when a request doesn't name one
    # Extra models a request may choose, as JSON, e.g. ALLOWED_MODELS='["qwen2.5:0.5b"]'
    allowed_models: list[str] = []
    ollama_url: str = "http://localhost:11434"

    max_concurrency: int = 4        # how many answers are generated at the same time
    queue_size: int = 16            # how many requests may wait for a free slot
    request_timeout: float = 120.0  # seconds before we give up (HTTP 504)
    max_tokens_limit: int = 2048    # upper bound for max_tokens in a request
    cache_max_entries: int = 500    # answers remembered for repeated temperature-0 questions
    cache_ttl_seconds: float = 3600

    arena_models: list[str] = []    # models that compete in the arena; empty = all allowed models
    arena_db: str = "data/arena.db"  # SQLite file with battles and votes

    log_level: str = "INFO"

    @property
    def models(self) -> list[str]:
        """Every model a request may use; the default model is always first."""
        return [self.model_id] + [m for m in self.allowed_models if m != self.model_id]
