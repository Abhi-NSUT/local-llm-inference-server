"""Answer quality per model version on the same 100 GSM8K + 100 MMLU questions.

    uv run python bench/run_quality.py                       (all models; ~30-60 min on a laptop)
    uv run python bench/run_quality.py --models qwen2.5:0.5b --datasets mmlu

Temperature 0, so answers are repeatable and the comparison is fair. Every answer is saved to
results/answers/, so wrong answers can be inspected, not just counted.
"""
import argparse
import asyncio
import csv
import json
import re
import time
from pathlib import Path

import httpx

OLLAMA = "http://localhost:11434"
MODELS = ["qwen2.5:1.5b-instruct-fp16", "qwen2.5:1.5b-instruct-q8_0", "qwen2.5:1.5b", "qwen2.5:0.5b"]
PARALLEL = 4  # questions in flight at once (matches OLLAMA_NUM_PARALLEL)
BENCH = Path(__file__).parent
RESULTS = BENCH / "results"


def gsm8k_prompt(q: dict) -> str:
    return (f"{q['question']}\n\nSolve it step by step, then write the final answer on the last line "
            f"as 'Answer: <number>'.")


def gsm8k_extract(text: str) -> str | None:
    """The number after 'Answer:', or else the last number in the text."""
    marked = re.findall(r"Answer:\s*\**\$?\s*(-?[\d,]*\.?\d+)", text)
    numbers = marked or re.findall(r"-?[\d,]*\.?\d+", text)
    return numbers[-1].replace(",", "").rstrip(".") if numbers else None


def gsm8k_correct(extracted: str | None, expected: str) -> bool:
    try:
        return extracted is not None and float(extracted) == float(expected)
    except ValueError:
        return False


def mmlu_prompt(q: dict) -> str:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", q["choices"]))
    return f"{q['question']}\n\n{options}\n\nAnswer with only the letter (A, B, C or D)."


def mmlu_extract(text: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else None


DATASETS = {
    #         prompt builder, answer extractor, scorer,                      max tokens
    "gsm8k": (gsm8k_prompt, gsm8k_extract, gsm8k_correct, 512),
    "mmlu": (mmlu_prompt, mmlu_extract, lambda got, want: got == want, 8),
}


async def ask(client: httpx.AsyncClient, sem: asyncio.Semaphore, model: str, prompt: str, max_tokens: int) -> dict:
    async with sem:
        r = await client.post("/api/chat", json={
            "model": model, "stream": False, "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens}})
        r.raise_for_status()
        data = r.json()
        return {"output": data["message"]["content"], "gen_tokens": data.get("eval_count", 0)}


async def evaluate(client: httpx.AsyncClient, model: str, dataset: str) -> dict:
    build, extract, score, max_tokens = DATASETS[dataset]
    questions = [json.loads(line) for line in (BENCH / "data" / f"{dataset}_100.jsonl").read_text().splitlines()]
    sem = asyncio.Semaphore(PARALLEL)
    t0 = time.perf_counter()
    replies = await asyncio.gather(*(ask(client, sem, model, build(q), max_tokens) for q in questions))
    minutes = (time.perf_counter() - t0) / 60

    answers = []
    for q, reply in zip(questions, replies):
        got = extract(reply["output"])
        answers.append({"id": q["id"], "expected": q["answer"], "extracted": got,
                        "correct": score(got, q["answer"]), **reply})
    out = RESULTS / "answers" / f"{model.replace(':', '_')}__{dataset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(a) + "\n" for a in answers))

    correct = sum(a["correct"] for a in answers)
    return {"model": model, "dataset": dataset, "correct": correct, "total": len(answers),
            "accuracy": round(correct / len(answers), 3),
            "no_answer_found": sum(a["extracted"] is None for a in answers),
            "avg_gen_tokens": round(sum(a["gen_tokens"] for a in answers) / len(answers), 1),
            "minutes": round(minutes, 1)}


async def main(models: list[str], datasets: list[str]):
    csv_path = RESULTS / "quality.csv"
    # Keep earlier results for model/dataset pairs we are not re-running now.
    rows = [r for r in csv.DictReader(csv_path.open())] if csv_path.exists() else []
    rows = [r for r in rows if not (r["model"] in models and r["dataset"] in datasets)]
    async with httpx.AsyncClient(base_url=OLLAMA, timeout=900) as client:
        for model in models:
            for m in (await client.get("/api/ps")).json()["models"]:  # free memory for this model
                await client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})
            for dataset in datasets:
                row = await evaluate(client, model, dataset)
                rows.append(row)
                print(f"{model:30} {dataset:6} {row['correct']:3}/{row['total']}  "
                      f"({row['no_answer_found']} unparsed, {row['minutes']} min)", flush=True)
    rows.sort(key=lambda r: (MODELS.index(r["model"]) if r["model"] in MODELS else 99, r["dataset"]))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    a = ap.parse_args()
    asyncio.run(main(a.models, a.datasets))
