"""HTTP endpoints."""
import json
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.backends.base import GenerationParams, Usage
from app.schemas import ChatRequest, GenerateRequest

router = APIRouter()


def error(status: int, kind: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status, headers=headers)


@router.get("/health")
async def health(request: Request):
    backend = request.app.state.backend
    ok = await backend.health()
    body = {"status": "ok" if ok else "unavailable", "backend": backend.name, "model": backend.model_id}
    return JSONResponse(body, status_code=200 if ok else 503)


@router.get("/models")
@router.get("/v1/models")
async def models(request: Request):
    model_id = request.app.state.backend.model_id
    return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]}


@router.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    backend = request.app.state.backend
    settings = request.app.state.settings
    params = GenerationParams(min(req.max_tokens, settings.max_tokens_limit), req.temperature, req.top_p)
    messages = [m.model_dump() for m in req.messages]

    if req.stream:
        return StreamingResponse(stream_chat(backend, messages, params), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    result = await backend.generate(messages, params)
    u = result.usage
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": backend.model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text},
                     "finish_reason": result.finish_reason}],
        "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                  "total_tokens": u.prompt_tokens + u.completion_tokens},
    }


def sse(data) -> str:
    """Format one Server-Sent Event: a `data:` line followed by a blank line."""
    return f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"


async def stream_chat(backend, messages: list[dict], params: GenerationParams):
    """Yield the answer as OpenAI-style `chat.completion.chunk` events, then `[DONE]`."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    def chunk(delta: dict, finish: str | None = None) -> dict:
        return {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": backend.model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    usage = Usage()
    yield sse(chunk({"role": "assistant"}))
    async for text in backend.stream(messages, params, usage):
        yield sse(chunk({"content": text}))
    yield sse(chunk({}, finish="stop"))
    yield sse("[DONE]")


@router.post("/v1/generate")
async def generate(req: GenerateRequest, request: Request):
    """Plain prompt -> text. Internally it is a one-message chat."""
    chat_req = ChatRequest(messages=[{"role": "user", "content": req.prompt}],
                           max_tokens=req.max_tokens, temperature=req.temperature, stream=req.stream)
    return await chat(chat_req, request)
