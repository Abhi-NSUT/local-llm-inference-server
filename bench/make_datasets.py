"""Download two public benchmarks and save a fixed random sample of 100 questions from each.

    uv run --with pyarrow python bench/make_datasets.py

- GSM8K: grade-school math word problems (OpenAI, MIT licence). Tests multi-step reasoning.
- MMLU: multiple-choice questions across 57 subjects (Hendrycks et al., MIT licence). Tests knowledge.

The seed is fixed, so everyone running this gets the same 100 questions, and every model is
tested on exactly the same set. The output files are committed so results are reproducible.
"""
import io
import json
import random
from pathlib import Path

import httpx

SEED, N = 42, 100
DATA = Path(__file__).parent / "data"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MMLU_URL = "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet"


def gsm8k(client: httpx.Client) -> list[dict]:
    rows = [json.loads(line) for line in client.get(GSM8K_URL).text.splitlines() if line.strip()]
    sample = random.Random(SEED).sample(rows, N)
    # The reference answer is the number after "####" at the end of the worked solution.
    return [{"id": f"gsm8k-{i}", "question": r["question"],
             "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
            for i, r in enumerate(sample)]


def mmlu(client: httpx.Client) -> list[dict]:
    import pyarrow.parquet as pq  # only needed here: run with `uv run --with pyarrow`

    table = pq.read_table(io.BytesIO(client.get(MMLU_URL).content)).to_pylist()
    sample = random.Random(SEED).sample(table, N)
    return [{"id": f"mmlu-{i}", "subject": r["subject"], "question": r["question"],
             "choices": list(r["choices"]), "answer": "ABCD"[r["answer"]]}
            for i, r in enumerate(sample)]


def save(name: str, rows: list[dict]) -> None:
    path = DATA / f"{name}_{N}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} questions to {path}")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        save("gsm8k", gsm8k(client))
        save("mmlu", mmlu(client))
