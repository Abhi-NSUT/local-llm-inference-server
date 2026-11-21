# Local LLM Inference Server + Benchmark Lab

[![tests](https://github.com/Abhi-NSUT/local-llm-inference-server/actions/workflows/test.yml/badge.svg)](https://github.com/Abhi-NSUT/local-llm-inference-server/actions/workflows/test.yml)

An OpenAI-compatible server that runs open-source language models (Qwen 2.5) on an 8 GB laptop, streams answers word
by word, handles many users at once, and comes with measured benchmarks of what model compression costs in speed,
memory and accuracy.

**In plain English:** it's a small, self-hosted version of the service behind ChatGPT, running a free AI model on my
own laptop instead of a paid API. It has a web page where you can chat with it, compare two hidden models side by
side, stress-test it with many simultaneous questions, and see live performance charts. I then measured how much
faster and smaller the model gets when compressed, and what that does to its answers.

## Try it

- **Demo page: https://abhi-nsut.github.io/local-llm-inference-server/**. A static copy of the app on GitHub
  Pages. There is no AI server behind it, so Chat, Arena and Stress test replay real responses recorded from the
  server, at their original speed. The Benchmarks tab shows the real measurements.
- **Live:** [run it locally](#run-it-locally) (about 5 minutes), or I can share a live link from my laptop during a
  call (`scripts/share_live.sh`).

---

## What I found

All numbers were measured on an **Apple M1 MacBook with 8 GB RAM** (Ollama 0.40), using the same 100 questions for
every model with randomness turned off. Raw data and scripts are in [`bench/`](bench/).

| Model | Memory | Cold start | Writing (tok/s) | Reading a 1k-token prompt (tok/s) | Math (GSM8K) | Knowledge (MMLU) |
|---|---|---|---|---|---|---|
| Qwen2.5 1.5B FP16 | 3.41 GB | 8.0 s | 16.6 | 432 | 78% | 50% |
| Qwen2.5 1.5B Q8 | 2.04 GB | 4.3 s | 24.7 | 342 | 75% | 52% |
| Qwen2.5 1.5B Q4 | 1.30 GB | 3.0 s | 30.7 | 301 | 70% | 51% |
| Qwen2.5 0.5B Q4 | 0.58 GB | 2.5 s | 63.3 | 1148 | 45% | 31% |

FP16 / Q8 / Q4 = each of the model's weights stored in 16, 8 or 4 bits (GGUF formats run by llama.cpp inside Ollama).
Memory is what Ollama reports for the loaded model, including its context cache for 4 parallel slots.

**1. Compression makes the model write faster but read slower.**
Q4 writes 1.85× faster than FP16, but reads a long prompt about 30% slower, so on a 1,000-token prompt FP16
actually shows its first word sooner (2.3 s vs 3.3 s). Writing (*decode*) produces one token at a time and must
stream every weight from memory for each one, so fewer bytes per weight means more tokens per second. Reading
(*prefill*) processes the whole prompt in parallel and is limited by computation instead, and quantized weights
must be unpacked before they can be used. "4-bit is faster" is only true for the writing part.

**2. Compression costs multi-step reasoning, not general knowledge.**
On math word problems, accuracy falls step by step (78 → 75 → 70%). On multiple-choice knowledge questions, the
three 1.5B versions are within noise of each other. A typical Q4 failure: asked how much Ted *spent* on books and
pencils, it correctly computed **$66**, then subtracted it from his $200 and answered **$134**, which is the answer
to a different question. Compared question by question, Q4 lost 12 math questions FP16 got right and gained 4, a
difference that is suggestive but not statistically conclusive at this sample size (McNemar p ≈ 0.08).

**3. Q8 is the sweet spot on this laptop.** It is 1.5× faster than FP16 at writing, uses 40% less memory, and scores
within noise of FP16 on both tests.

**4. Serving several requests at once more than doubles total output, at a cost to each user.**
With Ollama answering one request at a time, total output stayed flat at ~27 tokens/sec however many users arrived,
and with 8 users the median wait for the first word was 11.3 s. With 4 parallel slots:

| 8 users at once | 1 slot | 4 slots |
|---|---|---|
| Wait for first word (median) | 11.3 s | **3.1 s** |
| Each user's writing speed | 27.8 tok/s | 15.4 tok/s |
| Total output | 27.0 tok/s | **58.3 tok/s** |

Each decoding step reads the weights once and shares them across all active requests, so the hardware does more
useful work per step, while every individual stream gets a smaller share.

**5. The smallest model has a position bias.** The 0.5B model scores 31% on MMLU, below the 34% you would get by always
answering "A" on this sample, because it picks "D" 58% of the time.

### How sure are these numbers?
100 questions per test give roughly ±9 percentage points of uncertainty, so differences of a few points are noise.
Speeds are medians of 5 runs per setting, and each run starts with a different first line so Ollama cannot reuse the
previous run's prompt processing. Everything ran on one machine; other chips will differ.

---

## Things I discovered while building it

- **The server allowed 4 requests at a time, but Ollama only processed one.** With 4 users the first word took
  4.3 s. Setting `OLLAMA_NUM_PARALLEL=4` brought it to 0.37 s, and that became finding 4 above.
- **A cold start hides inside "reading the prompt".** The dashboard's per-request timeline showed requests that got a
  slot instantly but still waited ~4 s for the first word: Ollama had unloaded the idle model and was loading it
  again. Warm, the same requests started in ~250 ms.
- **My first speed metric was wrong.** Non-streaming requests reported 277,469 tokens/sec, because the "first token"
  timestamp was taken at the very end. It is now fixed and covered by a test.
- **The response cache made my load test meaningless.** Once identical temperature-0 questions were cached, the load
  test was measuring memory lookups, not the model. The load test now sends `cache: false`.
- **Tests passed or failed depending on my local `.env` file.** The test setup now pins every setting it relies on.

---

## Features

| Feature | What it does |
|---|---|
| **OpenAI-compatible API** | `/v1/chat/completions` and `/v1/models` follow OpenAI's format, so the official `openai` SDK and LangChain work by changing one URL |
| **Streaming** | Tokens are sent as they are generated, using Server-Sent Events |
| **Concurrency control** | An asyncio semaphore limits simultaneous generations, a bounded queue answers `429` when full instead of slowing everyone down, timeouts return `504`, and a client disconnect frees its slot and stops generation |
| **Model choice + JSON mode** | Each request can pick an allowed model. `response_format` with a JSON schema forces valid structured output (constrained decoding) |
| **Response cache** | LRU + TTL cache for repeatable (temperature 0) requests. A hit skips the queue and the model |
| **Model Arena** | Two hidden models answer the same prompt and a person votes. The arena keeps an Elo leaderboard and exports votes as `{prompt, chosen, rejected}` preference data (the format used for DPO fine-tuning) |
| **Metrics** | Per-request time to first token, total time and tokens/sec, plus a per-request timeline (waiting / reading prompt / writing), a live dashboard, and a Prometheus endpoint |
| **Benchmark Lab** | Reproducible scripts for speed, memory, accuracy and concurrency, with results committed |
| **Tests** | 38 tests run against a fake model backend, so they need no GPU or Ollama. They run in GitHub Actions on every push |

### Demo page tabs
**Chat** (streaming on/off, model picker, JSON-mode example) · **Arena** · **Stress test** (fires several questions at
once and shows which are answering, waiting or turned away) · **Dashboard** · **Benchmarks** · **How it works**

---

## Architecture

```
Browser / OpenAI SDK / curl
        │  HTTP (JSON, or Server-Sent Events when streaming)
        ▼
FastAPI app (uvicorn, one worker)
  ├─ validation (pydantic) ── model allowlist, JSON-mode schema
  ├─ response cache ─────────  temperature-0 hits return immediately
  ├─ concurrency limiter ─────  semaphore (N running) + bounded queue (429) + timeout (504)
  ├─ metrics ─────────────────  per-request timeline, per-second history, Prometheus
  ├─ arena ───────────────────  SQLite: battles, votes, Elo, preference export
  └─ backend interface
        ├─ OllamaBackend ─────  httpx stream → Ollama (llama.cpp, Metal GPU) → Qwen 2.5
        └─ FakeBackend ───────  deterministic, used by the tests
```

The whole app runs in one process, and concurrency comes from asyncio rather than multiple workers. Each worker would load
its own copy of the model, and an 8 GB machine has no room for that.

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | Chat; `stream`, `model`, `temperature`, `max_tokens`, `response_format`, `cache` |
| POST | `/v1/generate` | Single prompt → text |
| GET | `/v1/models` | Models a request may use |
| GET | `/health` | Backend status, requests answering / waiting |
| GET | `/metrics`, `/metrics/details`, `/metrics/timeseries`, `/metrics/prometheus` | Summaries, request timeline, per-second history, Prometheus format |
| POST / GET | `/arena/battles`, `/arena/battles/{id}/stream/{a\|b}`, `/arena/battles/{id}/vote` | Arena battle flow |
| GET | `/arena/leaderboard`, `/arena/export.jsonl` | Elo table, preference data |

Interactive docs are at `/docs` when the server is running.

```bash
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"stream": true, "messages": [{"role": "user", "content": "Write a haiku about servers"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
reply = client.chat.completions.create(model="qwen2.5:1.5b", messages=[{"role": "user", "content": "Hi"}])
print(reply.choices[0].message.content)
```

---

## Run it locally

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), about 6 GB of free disk
for all four model versions (1 GB for the default model only).

```bash
# 1. Ollama with 4 parallel slots, and the models
brew install ollama
launchctl setenv OLLAMA_NUM_PARALLEL 4 && launchctl setenv OLLAMA_MAX_LOADED_MODELS 3
brew services start ollama
ollama pull qwen2.5:1.5b                          # default (Q4)
ollama pull qwen2.5:1.5b-instruct-q8_0            # optional: model picker + arena
ollama pull qwen2.5:0.5b                          # optional: model picker + arena

# 2. The server
git clone https://github.com/Abhi-NSUT/local-llm-inference-server.git && cd local-llm-inference-server
cp .env.example .env
uv sync
uv run uvicorn app.main:app --port 8000           # open http://localhost:8000

# 3. Tests (no Ollama needed)
uv run pytest -q
```

### Share it live, or run it in Docker
```bash
brew install cloudflared
scripts/share_live.sh               # prints a temporary public https link; public-demo limits switched on

docker build -t llm-server . && docker run -p 7860:7860 llm-server   # any Docker host, e.g. a cloud VM
```

### Update the static demo
```bash
uv run python scripts/record_demo.py   # re-records app/static/demo/recordings.json; Pages republishes on push
```

### Reproduce the benchmarks
```bash
ollama pull qwen2.5:1.5b-instruct-fp16
uv run --with pyarrow python bench/make_datasets.py   # optional: the samples are already committed
uv run python bench/run_speed.py                      # ~5 min
uv run python bench/run_quality.py                    # ~30 min
uv run python bench/run_concurrency.py                # ~10 min, needs the server running; macOS/Homebrew
uv run python bench/summarize.py                      # updates the Benchmarks tab and prints the table
```

---

## Project layout

```
app/
  main.py               app setup, startup/shutdown, background metrics sampler
  config.py             settings from environment variables / .env
  schemas.py            request models (OpenAI-compatible)
  api/routes.py         chat, streaming, cache, health, metrics endpoints
  arena/                battles, votes, Elo leaderboard, preference export (SQLite)
  backends/             backend interface, Ollama backend, fake backend for tests
  core/                 concurrency limiter, response cache, metrics
  static/               demo page (one HTML file), benchmark summary, recorded responses for the static copy
bench/                  datasets, benchmark scripts, committed results
scripts/                load test, terminal client, demo recorder, live-sharing script
tests/                  38 tests, run in GitHub Actions
```

---

## Limitations and next steps

- **One laptop, small samples.** 100 questions per test and one machine. The next step is running the full GSM8K/MMLU
  sets and a second machine.
- **CPU/Apple-GPU quantization only.** INT8/INT4 here means GGUF Q8_0/Q4_K_M. On an NVIDIA GPU I would compare
  bitsandbytes, AWQ and GPTQ, and serve with vLLM, whose continuous batching should scale much further than
  Ollama's parallel slots.
- **In-memory state.** The cache, metrics and limiter live in one process, so running several servers would need
  Redis or a similar shared store.
- **No authentication.** Fine on localhost. A public deployment would need API keys and per-key rate limits.

## Tech stack
Python 3.12 · FastAPI · uvicorn · pydantic · httpx · Ollama (llama.cpp) · Qwen 2.5 · SQLite · prometheus-client ·
Chart.js · pytest · GitHub Actions · uv
