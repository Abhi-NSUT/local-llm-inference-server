"""Turn the raw benchmark CSVs into one summary for the demo page and the README.

    uv run python bench/summarize.py

Writes app/static/bench.json (read by the Benchmarks tab) and prints a Markdown table.
"""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "bench" / "results"


def read(name: str) -> list[dict]:
    return list(csv.DictReader((RESULTS / name).open()))


def label(params: str, quant: str) -> str:
    """'494.03M', 'Q4_K_M' -> '0.5B Q4';  '1.5B', 'F16' -> '1.5B FP16'"""
    size = f"{float(params[:-1]) / 1000:.1f}B" if params.endswith("M") else params
    return f"{size} {quant.replace('F16', 'FP16').replace('_K_M', '').replace('_0', '')}"


def main():
    speed, quality, concurrency = read("speed.csv"), read("quality.csv"), read("concurrency.csv")

    models = []
    for model in dict.fromkeys(r["model"] for r in speed):  # keeps file order
        runs = [r for r in speed if r["model"] == model]
        median = lambda key, prompt: statistics.median(float(r[key]) for r in runs if r["prompt"] == prompt)
        acc = {r["dataset"]: float(r["accuracy"]) for r in quality if r["model"] == model}
        models.append({
            "model": model,
            "label": label(runs[0]["params"], runs[0]["quantization"]),
            "memory_gb": float(runs[0]["memory_gb"]),
            "load_s": round(float(runs[0]["load_ms"]) / 1000, 1),
            "decode_tok_s": round(median("decode_tok_s", "short"), 1),
            "prefill_tok_s": round(median("prefill_tok_s", "long")),
            "ttft_long_s": round(median("ttft_ms", "long") / 1000, 1),
            "gsm8k": acc.get("gsm8k"),
            "mmlu": acc.get("mmlu"),
        })

    summary = {
        "machine": json.loads((RESULTS / "machine.json").read_text()),
        "models": models,
        "concurrency": [{k: float(v) for k, v in r.items()} for r in concurrency],
    }
    (ROOT / "app" / "static" / "bench.json").write_text(json.dumps(summary, indent=1) + "\n")

    print("| Model | Memory | Cold start | Writing (tok/s) | Reading 1k-token prompt (tok/s) | Math (GSM8K) | Knowledge (MMLU) |")
    print("|---|---|---|---|---|---|---|")
    for m in models:
        print(f"| {m['label']} | {m['memory_gb']} GB | {m['load_s']} s | {m['decode_tok_s']} | {m['prefill_tok_s']} "
              f"| {m['gsm8k']:.0%} | {m['mmlu']:.0%} |")


if __name__ == "__main__":
    main()
