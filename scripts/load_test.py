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


async def run_load(url: str, users: int, requests: int, max_tokens: int) -> dict:
    """Run the simulated users and return a summary (also used by bench/run_concurrency.py)."""
    results: list[dict] = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(base_url=url, timeout=300) as client:
        await asyncio.gather(*(user(client, "/v1/chat/completions", requests, max_tokens, results)
                               for _ in range(users)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["status"] == "ok"]
    statuses: dict[str, int] = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    per_req_tps = [r["tokens"] / max(r["latency"] - r["ttft"], 1e-6) for r in ok]
    return {
        "users": users, "requests": len(results), "ok": len(ok), "statuses": statuses, "wall_s": round(wall, 2),
        "ttft_p50_ms": round(pct([r["ttft"] for r in ok], 50) * 1000),
        "ttft_p95_ms": round(pct([r["ttft"] for r in ok], 95) * 1000),
        "latency_p50_s": round(pct([r["latency"] for r in ok], 50), 2),
        "latency_p95_s": round(pct([r["latency"] for r in ok], 95), 2),
        "per_request_tok_s": round(sum(per_req_tps) / len(per_req_tps), 1) if ok else 0.0,
        "aggregate_tok_s": round(sum(r["tokens"] for r in ok) / wall, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--requests", type=int, default=2, help="requests per user, one after another")
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()

    s = asyncio.run(run_load(a.url, a.users, a.requests, a.max_tokens))
    print(f"users={s['users']} requests={s['requests']} wall={s['wall_s']}s statuses={s['statuses']}")
    if s["ok"]:
        print(f"TTFT     p50={s['ttft_p50_ms']}ms  p95={s['ttft_p95_ms']}ms")
        print(f"latency  p50={s['latency_p50_s']}s  p95={s['latency_p95_s']}s")
        print(f"tok/s per request (avg)={s['per_request_tok_s']}   aggregate tok/s={s['aggregate_tok_s']}")


if __name__ == "__main__":
    main()
