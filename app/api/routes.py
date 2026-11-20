"""HTTP endpoints."""
import asyncio
import json
import time
import uuid
from collections.abc import Callable

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.backends.base import GenerationParams, GenerationResult, Usage
from app.core.concurrency import QueueFullError, with_deadline
from app.core.metrics import STATS, RequestTimer
from app.schemas import ChatRequest, GenerateRequest

router = APIRouter()


def error(status: int, kind: str, message: str) -> JSONResponse:
    headers = {"Retry-After": "1"} if status == 429 else None
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status, headers=headers)


@router.get("/health")
async def health(request: Request):
    backend, limiter = request.app.state.backend, request.app.state.limiter
    ok = await backend.health()
    body = {"status": "ok" if ok else "unavailable", "backend": backend.name, "model": backend.model_id,
            "in_flight": limiter.in_flight, "waiting": limiter.waiting,
            "capacity": limiter.capacity, "queue_size": limiter.queue_size}
    return JSONResponse(body, status_code=200 if ok else 503)


@router.get("/metrics")
async def metrics(request: Request):
    return {**STATS.summary(), "cache": request.app.state.cache.stats()}


@router.get("/metrics/timeseries")
async def metrics_timeseries():
    return STATS.timeseries()


@router.get("/metrics/details")
async def metrics_details(request: Request):
    """Recent requests (with their timeline), per-model comparison, and models in memory."""
    return {"recent": list(STATS.recent), "by_model": STATS.by_model(),
            "loaded_models": await request.app.state.backend.loaded_models()}


@router.get("/metrics/prometheus")
async def metrics_prometheus():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/models")
@router.get("/v1/models")
async def models(request: Request):
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "local"}
                                       for m in request.app.state.settings.models]}


@router.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    state = request.app.state
    backend, limiter, settings, cache = state.backend, state.limiter, state.settings, state.cache
    if req.model and req.model not in settings.models:
        return error(400, "invalid_request", f"Unknown model '{req.model}'. Available: {settings.models}")
    json_format = req.response_format.to_backend_format() if req.response_format else None
    params = GenerationParams(max_tokens=min(req.max_tokens, settings.max_tokens_limit),
                              temperature=req.temperature, top_p=req.top_p,
                              model=req.model or settings.model_id, format=json_format)
    messages = [m.model_dump() for m in req.messages]

    # Cache: only for repeatable (temperature 0) requests. A hit skips the queue and the model.
    cache_key = cache.key(messages, params) if req.cache and req.temperature == 0 else None
    if cache_key and (hit := cache.get(cache_key)):
        return cached_response(hit, params, req.stream)

    if limiter.is_full():  # quick "server busy" before doing any work
        return error(429, "overloaded", "Server is busy, please retry shortly.")

    if req.stream:
        save = (lambda text, usage, _metrics: cache.put(cache_key, GenerationResult(text, usage))) \
            if cache_key else None
        return sse_response(stream_chat(backend, limiter, settings, messages, params, on_complete=save))

    timer = RequestTimer(params.model)
    try:
        async with limiter.slot():
            timer.got_slot()
            async with asyncio.timeout(settings.request_timeout):
                result = await backend.generate(messages, params)
    except QueueFullError:
        timer.finish(0, "rejected")
        return error(429, "overloaded", "Server is busy, please retry shortly.")
    except TimeoutError:
        timer.finish(0, "timeout")
        return error(504, "timeout", "Generation took too long.")
    except Exception as e:  # e.g. Ollama not running
        timer.finish(0, "error")
        return error(503, "backend_unavailable", str(e) or type(e).__name__)
    if cache_key:
        cache.put(cache_key, result)
    metrics = timer.finish(result.usage.completion_tokens)
    if json_format:
        metrics["valid_json"] = is_json(result.text)
    return completion_body(result, params.model, metrics)


def completion_body(result: GenerationResult, model: str, metrics: dict) -> dict:
    """A full (non-streaming) OpenAI-style chat completion."""
    u = result.usage
    return {
        "id": new_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text},
                     "finish_reason": result.finish_reason}],
        "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                  "total_tokens": u.prompt_tokens + u.completion_tokens},
        "metrics": metrics,
    }


def cached_response(hit: GenerationResult, params: GenerationParams, stream: bool):
    metrics = {**RequestTimer(params.model).finish(hit.usage.completion_tokens, "cached"), "cached": True}
    if not stream:
        return completion_body(hit, params.model, metrics)

    async def replay():
        cid = new_id()
        yield sse(chunk(cid, params.model, {"role": "assistant"}))
        yield sse(chunk(cid, params.model, {"content": hit.text}))
        yield sse(chunk(cid, params.model, {}, finish="stop", metrics=metrics))
        yield sse("[DONE]")

    return sse_response(replay())


def new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def sse(data) -> str:
    """Format one Server-Sent Event: a `data:` line followed by a blank line."""
    return f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"


def sse_response(events) -> StreamingResponse:
    return StreamingResponse(events, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def chunk(cid: str, model: str, delta: dict, finish: str | None = None, **extra) -> dict:
    """One OpenAI-style `chat.completion.chunk` event."""
    return {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}


async def stream_chat(backend, limiter, settings, messages: list[dict], params: GenerationParams,
                      on_complete: Callable[[str, Usage, dict], None] | None = None):
    """Yield the answer as OpenAI-style `chat.completion.chunk` events, then `[DONE]`.

    The concurrency slot is held for the WHOLE stream. If the client disconnects, this generator
    is cancelled and the `finally` blocks free the slot and stop the model.
    `on_complete(text, usage, metrics)` is called only when the answer finished normally.
    """
    cid = new_id()
    timer, usage, pieces, status = RequestTimer(params.model), Usage(), [], "ok"
    try:
        async with limiter.slot():
            timer.got_slot()
            yield sse(chunk(cid, params.model, {"role": "assistant"}))
            async for text in with_deadline(backend.stream(messages, params, usage), settings.request_timeout):
                timer.first_token()
                pieces.append(text)
                yield sse(chunk(cid, params.model, {"content": text}))
            m = timer.finish(usage.completion_tokens or len(pieces))
            status = "done"
            if on_complete:
                on_complete("".join(pieces), usage, m)
            # `metrics` is an extra field; OpenAI clients simply ignore it, our demo page shows it.
            yield sse(chunk(cid, params.model, {}, finish="stop", metrics=m))
    except QueueFullError:
        status = "rejected"
        yield sse({"error": {"type": "overloaded", "message": "Server is busy, please retry shortly."}})
    except TimeoutError:
        status = "timeout"
        yield sse({"error": {"type": "timeout", "message": "Generation took too long."}})
    except Exception as e:
        status = "error"
        yield sse({"error": {"type": "backend_unavailable", "message": str(e) or type(e).__name__}})
    finally:
        if status != "done":
            timer.finish(usage.completion_tokens or len(pieces), "cancelled" if status == "ok" else status)
    yield sse("[DONE]")


@router.post("/v1/generate")
async def generate(req: GenerateRequest, request: Request):
    """Plain prompt -> text. Internally it is a one-message chat."""
    chat_req = ChatRequest(model=req.model, messages=[{"role": "user", "content": req.prompt}],
                           max_tokens=req.max_tokens, temperature=req.temperature, stream=req.stream)
    return await chat(chat_req, request)
