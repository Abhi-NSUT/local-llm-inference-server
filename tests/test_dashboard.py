"""Live dashboard data and the Prometheus endpoint."""
from tests.conftest import chat_body


async def test_timeseries_counts_requests_and_tokens(make_client):
    async with make_client() as c:
        await c.post("/v1/chat/completions", json=chat_body())
        data = (await c.get("/metrics/timeseries")).json()
    assert data["uptime_s"] >= 0
    last = data["points"][-1]
    assert set(last) == {"t", "requests", "tokens", "errors", "ttft_ms", "in_flight", "waiting"}
    assert last["requests"] >= 1 and last["tokens"] >= 6 and last["ttft_ms"]


async def test_prometheus_format(make_client):
    async with make_client() as c:
        await c.post("/v1/chat/completions", json=chat_body())
        r = await c.get("/metrics/prometheus")
    assert r.headers["content-type"].startswith("text/plain")
    for name in ("llm_requests_total", "llm_completion_tokens_total", "llm_ttft_seconds_bucket",
                 "llm_request_latency_seconds_bucket", "llm_in_flight", "llm_waiting"):
        assert name in r.text
    assert 'llm_requests_total{status="ok"}' in r.text


async def test_details_show_request_timeline_and_models(make_client):
    async with make_client(delay=0.01) as c:
        await c.post("/v1/chat/completions", json=chat_body(stream=True))
        data = (await c.get("/metrics/details")).json()
    last = data["recent"][0]
    assert last["model"] == "fake-model" and last["status"] == "ok"
    assert 0 <= last["queue_ms"] <= last["ttft_ms"] <= last["latency_ms"]  # the timeline is in order
    assert any(m["model"] == "fake-model" and m["requests"] >= 1 for m in data["by_model"])
    assert data["loaded_models"] == []  # the fake backend has nothing in memory
