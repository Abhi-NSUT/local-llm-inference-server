"""Record real server responses for the static copy of the demo page (GitHub Pages).

    uv run python scripts/record_demo.py          (needs Ollama running; takes ~5 minutes)

Starts its own server with the public-demo limits (2 answering, 4 waiting), sends the same requests
the page sends, and saves every response with its timing to app/static/demo/recordings.json.
On a static host the page replays these instead of calling a server.
The arena uses a copy of data/arena.db, so real votes are shown but never changed.
"""
import asyncio
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "app" / "static" / "index.html"
OUT = ROOT / "app" / "static" / "demo" / "recordings.json"
PORT = 8765
MODELS = ["qwen2.5:1.5b", "qwen2.5:1.5b-instruct-q8_0", "qwen2.5:0.5b"]
CHAT_PROMPTS = [
    "What is an API? Explain in two sentences.",
    "Ted starts with $200. He buys 3 books for 16 dollars each and 3 pencils for 6 dollars each. "
    "How much did he spend in total?",
    "Write a haiku about a laptop running an AI model.",
]
ARENA_PROMPTS = [
    "What is the capital of Italy? One sentence.",
    "A train travels 60 km in 45 minutes. What is its speed in km/h?",
    "Give one tip for writing clean Python code.",
]
STRESS_COUNTS = [6, 12]
# Must match the JSON-mode example in index.html.
JSON_TEXT = "Hi, I'm Priya, I turned 27 last month and I live in Pune."
PERSON_SCHEMA = {"type": "object", "required": ["name", "age", "city"],
                 "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}}}

responses: dict[str, dict] = {}


def stress_questions() -> list[str]:
    """Read the stress-test questions from the page itself, so they can't drift apart."""
    block = re.search(r"const QUESTIONS = \[(.*?)\];", PAGE.read_text(), re.S).group(1)
    return json.loads("[" + block.strip().rstrip(",") + "]")


async def record(client: httpx.AsyncClient, key: str, method: str, path: str, body: dict | None = None) -> dict:
    """Send one request and store the response: JSON with its delay, or SSE events with timestamps."""
    t0 = time.perf_counter()
    ms = lambda: round((time.perf_counter() - t0) * 1000)
    async with client.stream(method, "/" + path, json=body) as r:
        if r.headers.get("content-type", "").startswith("text/event-stream"):
            events = [[ms(), line[6:]] async for line in r.aiter_lines() if line.startswith("data: ")]
            rec = {"status": r.status_code, "events": events}
        else:
            rec = {"status": r.status_code, "json": json.loads(await r.aread()), "t": ms()}
    responses[key] = rec
    return rec


def chat_body(model: str, prompt: str, stream: bool, **extra) -> dict:
    return {"messages": [{"role": "user", "content": prompt}], "stream": stream, "max_tokens": 256,
            "model": model, **extra}


async def main(arena_db: str):
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=600) as c:
        for path in ("health", "v1/models"):
            await record(c, path, "GET", path)

        for model in MODELS:
            for prompt in CHAT_PROMPTS:
                for stream in (True, False):
                    await record(c, f"chat|{model}|{str(stream).lower()}||{prompt}", "POST",
                                 "v1/chat/completions", chat_body(model, prompt, stream))
            json_prompt = f"Extract the person's name, age and city from: {JSON_TEXT}"
            fmt = {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON_SCHEMA}}
            await record(c, f"chat|{model}|false|json|{json_prompt}", "POST", "v1/chat/completions",
                         {**chat_body(model, json_prompt, False, response_format=fmt), "max_tokens": 100})
            print(f"recorded chat for {model}", flush=True)

        questions = stress_questions()
        for count in STRESS_COUNTS:  # all at once, like the page does
            await asyncio.gather(*(record(c, f"stress-{count}-{i}", "POST", "v1/chat/completions",
                                          {"messages": [{"role": "user", "content": questions[i % len(questions)]}],
                                           "stream": True, "max_tokens": 60})
                                   for i in range(count)))
            print(f"recorded stress test with {count} questions", flush=True)
        for path in ("metrics", "metrics/details", "metrics/timeseries"):  # dashboard snapshot
            await record(c, path, "GET", path)

        battles = {}
        for prompt in ARENA_PROMPTS:
            rec = await record(c, f"battle|{prompt}", "POST", "arena/battles", {"prompt": prompt})
            bid = rec["json"]["battle_id"]
            await asyncio.gather(*(record(c, f"arena/battles/{bid}/stream/{s}", "GET",
                                          f"arena/battles/{bid}/stream/{s}") for s in "ab"))
            with sqlite3.connect(arena_db) as db:  # which models fought; revealed after a vote
                model_a, model_b = db.execute("SELECT model_a, model_b FROM battles WHERE id = ?", (bid,)).fetchone()
            battles[bid] = {"model_a": model_a, "model_b": model_b}
        await record(c, "arena/leaderboard", "GET", "arena/leaderboard")
        print("recorded arena battles", flush=True)

    machine = json.loads((ROOT / "bench" / "results" / "machine.json").read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "recorded_at": date.today().isoformat(), "machine": machine,
        "chat_prompts": CHAT_PROMPTS, "arena_prompts": ARENA_PROMPTS, "stress_counts": STRESS_COUNTS,
        "battles": battles, "responses": responses,
    }) + "\n")
    print(f"wrote {len(responses)} recordings to {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    arena_db = os.path.join(tmp, "arena.db")
    if (ROOT / "data" / "arena.db").exists():
        shutil.copy(ROOT / "data" / "arena.db", arena_db)
    env = {**os.environ, "MAX_CONCURRENCY": "2", "QUEUE_SIZE": "4", "RATE_LIMIT_PER_MINUTE": "0",
           "ALLOWED_MODELS": json.dumps(MODELS[1:]), "MODEL_ID": MODELS[0], "ARENA_DB": arena_db}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        asyncio.run(main(arena_db))
    finally:
        server.terminate()
        shutil.rmtree(tmp, ignore_errors=True)
