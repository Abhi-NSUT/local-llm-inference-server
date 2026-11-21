#!/usr/bin/env bash
# Container entry point: start Ollama, wait until it answers, then start the web server.
set -euo pipefail

# Set here rather than in the Dockerfile, because Docker's ENV would strip the JSON quotes.
export ALLOWED_MODELS="${ALLOWED_MODELS:-[\"qwen2.5:0.5b\"]}"

ollama serve &
until ollama list > /dev/null 2>&1; do sleep 1; done

exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
