"""App entry point: `uv run uvicorn app.main:app`."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.backends import get_backend
from app.config import Settings
from app.core.concurrency import ConcurrencyLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app.state.settings = settings
    app.state.backend = get_backend(settings)
    app.state.limiter = ConcurrencyLimiter(settings.max_concurrency, settings.queue_size)
    await app.state.backend.startup()  # once at startup, not per request
    yield
    await app.state.backend.shutdown()


app = FastAPI(title="Local LLM Inference Server", version="0.1.0", lifespan=lifespan)
app.include_router(router)
