"""HTTP endpoints."""
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.backends.base import GenerationParams
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


@router.post("/v1/generate")
async def generate(req: GenerateRequest, request: Request):
    """Plain prompt -> text. Internally it is a one-message chat."""
    chat_req = ChatRequest(messages=[{"role": "user", "content": req.prompt}],
                           max_tokens=req.max_tokens, temperature=req.temperature, stream=req.stream)
    return await chat(chat_req, request)
