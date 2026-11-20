"""Speed and memory per model version: FP16 vs Q8 vs Q4 (and a smaller 0.5B model).

    uv run python bench/run_speed.py          (needs Ollama running; takes a few minutes)

Talks to Ollama directly, so we measure the model, not our server. Ollama reports exact timings
(in nanoseconds) for reading the prompt ("prefill") and writing the answer ("decode").
Each model is unloaded first, so we also measure the cold-start load time.
"""
import csv
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import httpx

OLLAMA = "http://localhost:11434"
MODELS = ["qwen2.5:1.5b-instruct-fp16", "qwen2.5:1.5b-instruct-q8_0", "qwen2.5:1.5b", "qwen2.5:0.5b"]
RUNS = 5
MAX_TOKENS = 128
RESULTS = Path(__file__).parent / "results"


def prompts() -> dict[str, str]:
    """A short prompt, and a long one (~1,000 tokens) built from 16 GSM8K questions."""
    data = Path(__file__).parent / "data" / "gsm8k_100.jsonl"
    questions = [json.loads(line)["question"] for line in data.read_text().splitlines()[:16]]
    return {
        "short": "Explain what a CPU does in a few sentences.",
        "long": "Here are some math word problems:\n\n" + "\n\n".join(questions)
                + "\n\nIn a few sentences, what everyday topics do these problems cover?",
    }


def unload_all(client: httpx.Client) -> None:
    for m in client.get("/api/ps").json()["models"]:
        client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})


def run_once(client: httpx.Client, model: str, prompt: str) -> dict:
    """One streamed generation. Returns timings in ms and speeds in tokens/sec."""
    body = {"model": model, "prompt": prompt, "stream": True,
            "options": {"temperature": 0, "seed": 0, "num_predict": MAX_TOKENS}}
    t0 = time.perf_counter()
    ttft = None
    with client.stream("POST", "/api/generate", json=body) as r:
        for line in r.iter_lines():
            data = json.loads(line)
            if ttft is None and data.get("response"):
                ttft = (time.perf_counter() - t0) * 1000
            if data.get("done"):
                final = data
    ns = 1e9
    return {
        "ttft_ms": round(ttft, 1),
        "load_ms": round(final.get("load_duration", 0) / 1e6, 1),
        "prompt_tokens": final["prompt_eval_count"],
        "prefill_tok_s": round(final["prompt_eval_count"] / (final["prompt_eval_duration"] / ns), 1),
        "gen_tokens": final["eval_count"],
        "decode_tok_s": round(final["eval_count"] / (final["eval_duration"] / ns), 1),
    }


def machine_info(client: httpx.Client) -> dict:
    def sysctl(key):
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    return {"chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
            "ram_gb": round(int(sysctl("hw.memsize") or 0) / 2**30),
            "os": platform.platform(), "ollama": client.get("/api/version").json()["version"]}


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    with httpx.Client(base_url=OLLAMA, timeout=600) as client:
        (RESULTS / "machine.json").write_text(json.dumps(machine_info(client), indent=2) + "\n")
        for model in MODELS:
            details = client.post("/api/show", json={"model": model}).json()["details"]
            unload_all(client)
            cold = run_once(client, model, "Say hello.")  # first request also loads the model
            memory = next(m for m in client.get("/api/ps").json()["models"] if m["name"] == model)
            for kind, prompt in prompts().items():
                for i in range(RUNS):
                    # A different first line each run stops Ollama from reusing the previous
                    # run's prompt processing, so prefill is measured fresh every time.
                    r = run_once(client, model, f"(run {i}) {prompt}")
                    rows.append({"model": model, "params": details["parameter_size"],
                                 "quantization": details["quantization_level"], "prompt": kind, "run": i,
                                 **r, "load_ms": cold["load_ms"],
                                 "memory_gb": round(memory["size"] / 1e9, 2)})
            summary = {k: statistics.median(r["decode_tok_s"] for r in rows
                                            if r["model"] == model and r["prompt"] == k) for k in ("short", "long")}
            print(f"{model:30} {details['quantization_level']:7} load {cold['load_ms']:7.0f} ms  "
                  f"mem {memory['size'] / 1e9:4.2f} GB  decode median short/long {summary['short']}/{summary['long']} tok/s")
        unload_all(client)

    with open(RESULTS / "speed.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {RESULTS / 'speed.csv'}")


if __name__ == "__main__":
    main()
