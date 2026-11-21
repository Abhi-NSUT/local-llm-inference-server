"""Limits used on the public demo: per-visitor rate limit and maximum input size."""
from app.core.rate_limit import RateLimiter
from tests.conftest import chat_body


async def test_rate_limit_per_visitor(make_client):
    async with make_client(RATE_LIMIT_PER_MINUTE=2) as c:
        codes = [(await c.post("/v1/chat/completions", json=chat_body(),
                               headers={"X-Forwarded-For": "1.2.3.4"})).status_code for _ in range(3)]
        other_visitor = await c.post("/v1/chat/completions", json=chat_body(),
                                     headers={"X-Forwarded-For": "5.6.7.8"})
        arena = await c.post("/arena/battles", json={"prompt": "hi"}, headers={"X-Forwarded-For": "1.2.3.4"})
    assert codes == [200, 200, 429]
    assert other_visitor.status_code == 200   # each visitor has their own allowance
    assert arena.status_code == 429           # the arena shares the same allowance


async def test_too_long_input_is_rejected(make_client):
    async with make_client(MAX_INPUT_CHARS=10) as c:
        r = await c.post("/v1/chat/completions", json=chat_body("x" * 11))
    assert r.status_code == 400
    assert "too long" in r.json()["error"]["message"]


def test_sliding_window_forgets_old_requests():
    now = [0.0]
    limiter = RateLimiter(2, window_seconds=60, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    now[0] = 61
    assert limiter.allow("a")
    assert RateLimiter(0).allow("anyone")  # 0 = no limit
