"""App entry point: `uv run uvicorn app.main:app`."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.backends import get_backend
from app.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    app.state.backend = get_backend(settings)
    await app.state.backend.startup()  # once at startup, not per request
    yield
    await app.state.backend.shutdown()


app = FastAPI(title="Local LLM Inference Server", version="0.1.0", lifespan=lifespan)
app.include_router(router)
