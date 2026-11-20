"""Health, models, non-streaming chat, validation."""
from app.backends.ollama_backend import OllamaBackend
from app.main import app
from tests.conftest import chat_body

FAKE_TEXT = "Hello from the fake model."


async def test_health(make_client):
    async with make_client() as c:
        r = await c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["backend"] == "fake"
    assert body["in_flight"] == 0 and body["waiting"] == 0


async def test_models(make_client):
    async with make_client() as c:
        r = await c.get("/v1/models")
    assert r.json()["data"][0]["id"] == "fake-model"


async def test_chat_non_stream(make_client):
    async with make_client() as c:
        r = await c.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": FAKE_TEXT}
    assert body["usage"]["completion_tokens"] == 6
    assert body["metrics"]["latency_ms"] >= 0


async def test_generate_endpoint(make_client):
    async with make_client() as c:
        r = await c.post("/v1/generate", json={"prompt": "hi"})
    assert r.json()["choices"][0]["message"]["content"] == FAKE_TEXT


async def test_max_tokens_is_capped_by_server_limit(make_client):
    async with make_client(MAX_TOKENS_LIMIT=3) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(max_tokens=1000))
    assert r.json()["usage"]["completion_tokens"] == 3


async def test_validation_errors(make_client):
    async with make_client() as c:
        empty = await c.post("/v1/chat/completions", json={"messages": []})
        bad_role = await c.post("/v1/chat/completions",
                                json={"messages": [{"role": "robot", "content": "hi"}]})
        bad_temp = await c.post("/v1/chat/completions", json=chat_body(temperature=5))
    assert empty.status_code == bad_role.status_code == bad_temp.status_code == 422


async def test_backend_down_returns_503(make_client):
    async with make_client() as c:
        dead = OllamaBackend("any-model", "http://127.0.0.1:1")  # nothing listens on port 1
        await dead.startup()
        app.state.backend = dead
        health = await c.get("/health")
        chat = await c.post("/v1/chat/completions", json=chat_body())
        await dead.shutdown()
    assert health.status_code == 503
    assert chat.status_code == 503
    assert chat.json()["error"]["type"] == "backend_unavailable"


async def test_demo_page_is_served(make_client):
    async with make_client() as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Stress test" in r.text


async def test_non_stream_speed_is_realistic(make_client):
    # 6 words with 0.02 s between them: about 50 tokens/sec, never thousands.
    async with make_client(delay=0.02) as c:
        r = await c.post("/v1/chat/completions", json=chat_body())
    m = r.json()["metrics"]
    assert 20 < m["tokens_per_sec"] < 60
    assert m["ttft_ms"] == m["latency_ms"]  # without streaming, the first word comes with the last


async def test_benchmark_summary_is_served(make_client):
    async with make_client() as c:
        r = await c.get("/bench.json")
    data = r.json()
    assert {"machine", "models", "concurrency"} <= set(data)
    assert all({"label", "decode_tok_s", "gsm8k", "mmlu"} <= set(m) for m in data["models"])
