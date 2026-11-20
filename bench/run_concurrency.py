"""Many users at once: how do waiting time and total throughput change with Ollama's parallel slots?

    uv run uvicorn app.main:app --port 8000      (in another terminal)
    uv run python bench/run_concurrency.py

macOS + Homebrew only: it restarts the Ollama service with OLLAMA_NUM_PARALLEL = 1, then 4,
and runs scripts/load_test.py's logic at 1, 2, 4 and 8 simultaneous users each time.
At the end Ollama is left at 4 parallel slots.
"""
import asyncio
import csv
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from load_test import run_load  # noqa: E402  (reuse the load-test code instead of copying it)

SERVER = "http://localhost:8000"
PARALLEL = [1, 4]
USERS = [1, 2, 4, 8]
REQUESTS_PER_USER = 2
MAX_TOKENS = 64
RESULTS = Path(__file__).parent / "results"


def restart_ollama(parallel: int) -> None:
    subprocess.run(["launchctl", "setenv", "OLLAMA_NUM_PARALLEL", str(parallel)], check=True)
    subprocess.run(["brew", "services", "restart", "ollama"], check=True, capture_output=True)
    for _ in range(60):  # wait until Ollama answers again
        try:
            if httpx.get("http://localhost:11434/api/tags", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("Ollama did not come back after restart")


def main():
    if httpx.get(f"{SERVER}/health", timeout=5).status_code != 200:
        sys.exit("Start the server first: uv run uvicorn app.main:app --port 8000")
    rows = []
    for parallel in PARALLEL:
        restart_ollama(parallel)
        asyncio.run(run_load(SERVER, 1, 1, 8))  # warm-up: load the model before measuring
        for users in USERS:
            s = asyncio.run(run_load(SERVER, users, REQUESTS_PER_USER, MAX_TOKENS))
            s.pop("statuses")
            rows.append({"ollama_parallel": parallel, **s})
            print(f"parallel={parallel} users={users}: first word p50 {s['ttft_p50_ms']} ms, "
                  f"per request {s['per_request_tok_s']} tok/s, total {s['aggregate_tok_s']} tok/s", flush=True)
    if PARALLEL[-1] != 4:
        restart_ollama(4)

    with open(RESULTS / "concurrency.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {RESULTS / 'concurrency.csv'}")


if __name__ == "__main__":
    main()
