#!/usr/bin/env bash
# Publish the current commit to a Hugging Face Space:
#
#   scripts/deploy_space.sh <hf-username>/<space-name>
#
# Spaces read their settings from a block at the top of README.md. To keep the GitHub README
# clean, the block is added only to a temporary copy of the repo, which is pushed to the Space.
# git asks for your Hugging Face username and an access token (with write permission) as password.
set -euo pipefail

SPACE="${1:?usage: scripts/deploy_space.sh <hf-username>/<space-name>}"
TMP="$(mktemp -d)"
trap 'git worktree remove --force "$TMP"' EXIT

git worktree add --detach "$TMP" HEAD > /dev/null
cd "$TMP"
{
  cat <<'EOF'
---
title: Local LLM Inference Server
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: OpenAI-compatible LLM server with arena and benchmarks
---

EOF
  cat README.md
} > README.space.md
mv README.space.md README.md
git add README.md
git commit -q -m "Hugging Face Space settings"

# Force: each deploy is a fresh copy of the current commit, so the Space history is replaced.
git push --force "https://huggingface.co/spaces/$SPACE" HEAD:main
echo "Deployed. Building at https://huggingface.co/spaces/$SPACE (takes ~10 minutes)."
