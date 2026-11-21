#!/usr/bin/env bash
# Share the live app from this Mac at a temporary public https link (free Cloudflare quick tunnel):
#
#   scripts/share_live.sh        then send the printed https://....trycloudflare.com link
#
# Anyone with the link can use it, so the public-demo limits are switched on. Ctrl-C stops both.
set -euo pipefail
command -v cloudflared > /dev/null || { echo "Install it first: brew install cloudflared"; exit 1; }

MAX_CONCURRENCY=2 QUEUE_SIZE=4 MAX_TOKENS_LIMIT=256 MAX_INPUT_CHARS=4000 RATE_LIMIT_PER_MINUTE=10 \
  uv run uvicorn app.main:app --port 8000 &
SERVER=$!
trap 'kill $SERVER' EXIT

cloudflared tunnel --url http://localhost:8000
