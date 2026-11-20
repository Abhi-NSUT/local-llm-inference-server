"""Shared test setup: run the real app in-process with the fake backend (no Ollama, no network)."""
from contextlib import asynccontextmanager

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.backends.fake_backend import FakeBackend
from app.main import app


@pytest.fixture
def make_client(monkeypatch):
    """Usage: `async with make_client(MAX_CONCURRENCY="2", delay=0.05) as client: ...`"""

    @asynccontextmanager
    async def _make(delay: float = 0.0, **env):
        monkeypatch.setenv("BACKEND", "fake")
        monkeypatch.setenv("MODEL_ID", "fake-model")
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        async with LifespanManager(app):
            app.state.backend = FakeBackend(delay=delay)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client

    return _make


def chat_body(text: str = "hi", **extra) -> dict:
    return {"messages": [{"role": "user", "content": text}], **extra}
