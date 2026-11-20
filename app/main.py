"""App entry point: `uv run uvicorn app.main:app`."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.arena.routes import router as arena_router
from app.arena.store import ArenaStore
from app.backends import get_backend
from app.config import Settings
from app.core.cache import ResponseCache
from app.core.concurrency import ConcurrencyLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app.state.settings = settings
    app.state.backend = get_backend(settings)
    app.state.limiter = ConcurrencyLimiter(settings.max_concurrency, settings.queue_size)
    app.state.cache = ResponseCache(settings.cache_max_entries, settings.cache_ttl_seconds)
    app.state.arena = ArenaStore(settings.arena_db)
    await app.state.backend.startup()  # once at startup, not per request
    yield
    await app.state.backend.shutdown()
    app.state.arena.db.close()


app = FastAPI(title="Local LLM Inference Server", version="0.1.0", lifespan=lifespan)
app.include_router(router)
app.include_router(arena_router)
# The demo page. Mounted last so the API routes above take priority.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
