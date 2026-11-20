"""Talks to the Ollama daemon, which runs the model with llama.cpp (Metal GPU on Macs)."""
import json

import httpx

from .base import Backend, GenerationParams, Usage


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model_id: str, base_url: str):
        self.model_id = model_id
        self.base_url = base_url
        self.client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(300, connect=5))

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()

    async def health(self) -> bool:
        try:
            r = await self.client.get("/api/tags")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def stream(self, messages, params: GenerationParams, usage: Usage):
        payload = {
            "model": params.model or self.model_id,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
            },
        }
        if params.format:
            # Ollama restricts which tokens the model may pick so the output always parses as JSON
            # (and matches the schema, if one is given). This is called constrained decoding.
            payload["format"] = params.format
        # Ollama streams NDJSON: one JSON object per line; the last one has "done": true.
        async with self.client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    usage.prompt_tokens = data.get("prompt_eval_count", 0)
                    usage.completion_tokens = data.get("eval_count", 0)
                    break
                if chunk := data.get("message", {}).get("content"):
                    yield chunk
