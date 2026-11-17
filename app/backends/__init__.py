from app.config import Settings

from .base import Backend


def get_backend(s: Settings) -> Backend:
    """Pick the backend named in the BACKEND setting."""
    if s.backend == "ollama":
        from .ollama_backend import OllamaBackend
        return OllamaBackend(s.model_id, s.ollama_url)
    if s.backend == "fake":
        from .fake_backend import FakeBackend
        return FakeBackend()
    raise ValueError(f"unknown backend: {s.backend}")
