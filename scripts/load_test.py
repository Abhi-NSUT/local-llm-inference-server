"""Simulate N users at once and report speed and errors.

    uv run python scripts/load_test.py --users 8 --requests 2 --max-tokens 64
"""
import argparse
import asyncio
import json
import time

import httpx

PROMPT = "Explain what an API is in two sentences."


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


async def one_request(client: httpx.AsyncClient, url: str, max_tokens: int) -> dict:
    # cache=False: we want to measure the model, not answers replayed from memory
    body = {"stream": True, "max_tokens": max_tokens, "temperature": 0, "cache": False,
            "messages": [{"role": "user", "content": PROMPT}]}
    t0 = time.perf_counter()
    ttft, tokens = None, 0
    async with client.stream("POST", url, json=body) as r:
        if r.status_code != 200:
            return {"status": str(r.status_code)}
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "error" in data:
                return {"status": data["error"]["type"]}
            if data["choices"][0]["delta"].get("content"):
                ttft = ttft or time.perf_counter() - t0
            if "metrics" in data:
                tokens = data["metrics"]["completion_tokens"]
    total = time.perf_counter() - t0
    return {"status": "ok", "ttft": ttft or total, "latency": total, "tokens": tokens}


async def user(client, url, n, max_tokens, results):
    for _ in range(n):
        results.append(await one_request(client, url, max_tokens))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--requests", type=int, default=2, help="requests per user, one after another")
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()

    results: list[dict] = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(base_url=a.url, timeout=300) as client:
        await asyncio.gather(*(user(client, "/v1/chat/completions", a.requests, a.max_tokens, results)
                               for _ in range(a.users)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["status"] == "ok"]
    statuses: dict[str, int] = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    total_tokens = sum(r["tokens"] for r in ok)
    per_req_tps = [r["tokens"] / max(r["latency"] - r["ttft"], 1e-6) for r in ok]
    print(f"users={a.users} requests={len(results)} wall={wall:.1f}s statuses={statuses}")
    if ok:
        print(f"TTFT     p50={pct([r['ttft'] for r in ok], 50) * 1000:.0f}ms  p95={pct([r['ttft'] for r in ok], 95) * 1000:.0f}ms")
        print(f"latency  p50={pct([r['latency'] for r in ok], 50):.2f}s  p95={pct([r['latency'] for r in ok], 95):.2f}s")
        print(f"tok/s per request (avg)={sum(per_req_tps) / len(per_req_tps):.1f}   aggregate tok/s={total_tokens / wall:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
