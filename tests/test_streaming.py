"""Server-Sent Events: order and format of the streamed chunks."""
import json

from tests.conftest import chat_body


def parse_sse(raw: str) -> list:
    events = []
    for block in raw.strip().split("\n\n"):
        assert block.startswith("data: ")
        data = block.removeprefix("data: ")
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


async def test_stream_event_order(make_client):
    async with make_client() as c:
        r = await c.post("/v1/chat/completions", json=chat_body(stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(r.text)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}       # 1. role
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events[1:-2])
    assert content == "Hello from the fake model."                         # 2. text pieces
    final = events[-2]
    assert final["choices"][0]["finish_reason"] == "stop"                  # 3. stop + metrics
    assert final["metrics"]["completion_tokens"] == 6
    assert events[-1] == "[DONE]"                                          # 4. done
    assert len({e["id"] for e in events[:-1]}) == 1                        # same id throughout


async def test_stream_timeout_sends_error_event(make_client):
    async with make_client(delay=0.05, REQUEST_TIMEOUT=0.1) as c:  # 6 words x 0.05 s > 0.1 s
        r = await c.post("/v1/chat/completions", json=chat_body(stream=True))
    events = parse_sse(r.text)
    assert events[-2] == {"error": {"type": "timeout", "message": "Generation took too long."}}
    assert events[-1] == "[DONE]"
