"""HTTP endpoints."""
import asyncio
import json
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.backends.base import GenerationParams, Usage
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
async def metrics():
    return STATS.summary()


@router.get("/models")
@router.get("/v1/models")
async def models(request: Request):
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "local"}
                                       for m in request.app.state.settings.models]}


@router.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    state = request.app.state
    backend, limiter, settings = state.backend, state.limiter, state.settings
    if req.model and req.model not in settings.models:
        return error(400, "invalid_request", f"Unknown model '{req.model}'. Available: {settings.models}")
    json_format = req.response_format.to_backend_format() if req.response_format else None
    params = GenerationParams(max_tokens=min(req.max_tokens, settings.max_tokens_limit),
                              temperature=req.temperature, top_p=req.top_p,
                              model=req.model or settings.model_id, format=json_format)
    messages = [m.model_dump() for m in req.messages]

    if limiter.is_full():  # quick "server busy" before doing any work
        return error(429, "overloaded", "Server is busy, please retry shortly.")

    if req.stream:
        return StreamingResponse(stream_chat(backend, limiter, settings, messages, params),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    timer = RequestTimer()
    try:
        async with limiter.slot():
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
    u = result.usage
    metrics = timer.finish(u.completion_tokens)
    if json_format:
        metrics["valid_json"] = is_json(result.text)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": params.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text},
                     "finish_reason": result.finish_reason}],
        "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                  "total_tokens": u.prompt_tokens + u.completion_tokens},
        "metrics": metrics,
    }


def is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def sse(data) -> str:
    """Format one Server-Sent Event: a `data:` line followed by a blank line."""
    return f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"


async def stream_chat(backend, limiter, settings, messages: list[dict], params: GenerationParams):
    """Yield the answer as OpenAI-style `chat.completion.chunk` events, then `[DONE]`.

    The concurrency slot is held for the WHOLE stream. If the client disconnects, this generator
    is cancelled and the `finally` blocks free the slot and stop the model.
    """
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    def chunk(delta: dict, finish: str | None = None, **extra) -> dict:
        return {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": params.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

    timer, usage, n_chunks, status = RequestTimer(), Usage(), 0, "ok"
    try:
        async with limiter.slot():
            yield sse(chunk({"role": "assistant"}))
            async for text in with_deadline(backend.stream(messages, params, usage), settings.request_timeout):
                timer.first_token()
                n_chunks += 1
                yield sse(chunk({"content": text}))
            m = timer.finish(usage.completion_tokens or n_chunks)
            status = "done"
            # `metrics` is an extra field; OpenAI clients simply ignore it, our demo page shows it.
            yield sse(chunk({}, finish="stop", metrics=m))
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
        if status not in ("done",):
            timer.finish(usage.completion_tokens or n_chunks, "cancelled" if status == "ok" else status)
    yield sse("[DONE]")


@router.post("/v1/generate")
async def generate(req: GenerateRequest, request: Request):
    """Plain prompt -> text. Internally it is a one-message chat."""
    chat_req = ChatRequest(model=req.model, messages=[{"role": "user", "content": req.prompt}],
                           max_tokens=req.max_tokens, temperature=req.temperature, stream=req.stream)
    return await chat(chat_req, request)
