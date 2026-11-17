"""All settings come from environment variables (or the .env file)."""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    backend: Literal["ollama", "fake"] = "ollama"
    model_id: str = "qwen2.5:1.5b"
    ollama_url: str = "http://localhost:11434"

    max_concurrency: int = 4        # how many answers are generated at the same time
    queue_size: int = 16            # how many requests may wait for a free slot
    request_timeout: float = 120.0  # seconds before we give up (HTTP 504)
    max_tokens_limit: int = 2048    # upper bound for max_tokens in a request

    log_level: str = "INFO"
