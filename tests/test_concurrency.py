"""Many requests at once: limits, 429, timeouts, and freeing slots on disconnect."""
import asyncio

from app.api.routes import stream_chat
from app.backends.base import GenerationParams
from app.main import app
from tests.conftest import chat_body


async def test_extra_requests_get_429(make_client):
    # 2 answering + 2 waiting are accepted; the other 6 are turned away immediately.
    async with make_client(delay=0.02, MAX_CONCURRENCY=2, QUEUE_SIZE=2) as c:
        limiter = app.state.limiter
        peak = 0

        async def watch():
            nonlocal peak
            while True:
                peak = max(peak, limiter.in_flight)
                await asyncio.sleep(0.005)

        watcher = asyncio.create_task(watch())
        responses = await asyncio.gather(*(c.post("/v1/chat/completions", json=chat_body())
                                           for _ in range(10)))
        watcher.cancel()

    codes = sorted(r.status_code for r in responses)
    assert codes == [200] * 4 + [429] * 6
    assert peak == 2                       # never more than MAX_CONCURRENCY at once
    busy = next(r for r in responses if r.status_code == 429)
    assert busy.headers["retry-after"] == "1"
    assert limiter.in_flight == 0 and limiter.waiting == 0


async def test_non_stream_timeout_returns_504(make_client):
    async with make_client(delay=0.05, REQUEST_TIMEOUT=0.1) as c:
        r = await c.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 504
    assert app.state.limiter.in_flight == 0


async def test_stream_rejected_when_queue_full(make_client):
    async with make_client(QUEUE_SIZE=0) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(stream=True))
    assert r.status_code == 429


async def test_client_disconnect_frees_the_slot(make_client):
    async with make_client(delay=0.01) as c:
        state = app.state
        gen = stream_chat(state.backend, state.limiter, state.settings,
                          [{"role": "user", "content": "hi"}], GenerationParams())
        await anext(gen)                        # role chunk: slot is now taken
        await anext(gen)                        # first word
        assert state.limiter.in_flight == 1
        await gen.aclose()                      # what happens when the browser tab closes
        assert state.limiter.in_flight == 0

        r = await c.get("/metrics")             # the cut-off request is counted, not lost
    assert r.json()["requests_by_status"].get("cancelled", 0) >= 1
