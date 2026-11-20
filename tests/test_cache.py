"""Response cache: repeated temperature-0 questions skip the model."""
from app.backends.base import GenerationParams, GenerationResult
from app.core.cache import ResponseCache
from app.main import app
from tests.conftest import chat_body
from tests.test_streaming import parse_sse


async def test_second_identical_request_is_a_cache_hit(make_client):
    async with make_client(delay=0.02) as c:
        first = (await c.post("/v1/chat/completions", json=chat_body(temperature=0))).json()
        second = (await c.post("/v1/chat/completions", json=chat_body(temperature=0))).json()
        stats = (await c.get("/metrics")).json()["cache"]
    assert "cached" not in first["metrics"]
    assert second["metrics"]["cached"] is True
    assert second["choices"][0]["message"]["content"] == first["choices"][0]["message"]["content"]
    assert second["metrics"]["latency_ms"] < first["metrics"]["latency_ms"]
    assert stats["hits"] == 1 and stats["misses"] == 1


async def test_streaming_answer_is_cached_and_replayed(make_client):
    async with make_client() as c:
        await c.post("/v1/chat/completions", json=chat_body(temperature=0, stream=True))
        r = await c.post("/v1/chat/completions", json=chat_body(temperature=0, stream=True))
    events = parse_sse(r.text)
    assert events[1]["choices"][0]["delta"]["content"] == "Hello from the fake model."
    assert events[-2]["metrics"]["cached"] is True
    assert events[-1] == "[DONE]"


async def test_cache_hit_works_even_when_server_is_full(make_client):
    async with make_client(QUEUE_SIZE=0) as c:
        app.state.cache.put(ResponseCache.key([{"role": "user", "content": "hi"}],
                                              GenerationParams(max_tokens=256, model="fake-model")),
                            GenerationResult("cached answer"))
        r = await c.post("/v1/chat/completions", json=chat_body(temperature=0))
    assert r.status_code == 200  # served from memory, so the full queue doesn't matter
    assert r.json()["choices"][0]["message"]["content"] == "cached answer"


async def test_random_answers_are_never_cached(make_client):
    async with make_client() as c:
        for _ in range(2):
            r = await c.post("/v1/chat/completions", json=chat_body(temperature=0.7))
            assert "cached" not in r.json()["metrics"]
        no_cache = await c.post("/v1/chat/completions", json=chat_body(temperature=0, cache=False))
        stats = (await c.get("/metrics")).json()["cache"]
    assert "cached" not in no_cache.json()["metrics"]
    assert stats == {"hits": 0, "misses": 0, "size": 0, "hit_rate": 0.0}


def test_lru_evicts_least_recently_used():
    cache = ResponseCache(max_entries=2)
    cache.put("a", GenerationResult("A"))
    cache.put("b", GenerationResult("B"))
    cache.get("a")                      # "a" is now more recent than "b"
    cache.put("c", GenerationResult("C"))
    assert cache.get("b") is None       # "b" was dropped
    assert cache.get("a").text == "A" and cache.get("c").text == "C"


def test_entries_expire_after_ttl():
    now = [0.0]
    cache = ResponseCache(ttl_seconds=10, clock=lambda: now[0])
    cache.put("q", GenerationResult("answer"))
    now[0] = 5
    assert cache.get("q").text == "answer"
    now[0] = 11
    assert cache.get("q") is None
