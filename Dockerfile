# One container runs both Ollama (the model) and the FastAPI app, for any Docker host (e.g. a cloud VM).
#   docker build -t llm-server . && docker run -p 7860:7860 llm-server
FROM ollama/ollama:0.40.16

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv

# Run as an ordinary user (uid 1000), not root; everything lives in its home directory.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    OLLAMA_MODELS=/home/user/.ollama/models \
    UV_PYTHON_INSTALL_DIR=/home/user/.python \
    UV_COMPILE_BYTECODE=1
WORKDIR /home/user/app

# Download the models while building, so starting the container needs no download.
RUN (ollama serve > /dev/null 2>&1 &) && sleep 5 \
    && ollama pull qwen2.5:1.5b && ollama pull qwen2.5:0.5b

# Install Python + dependencies before copying the code, so code changes don't redo this slow step.
COPY --chown=user pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .

# Public demo on the free tier (2 vCPUs): 2 answers at a time, short answers, per-visitor limit.
# OLLAMA_KEEP_ALIVE=-1 keeps models loaded, so visitors don't hit a cold start.
ENV BACKEND=ollama MODEL_ID=qwen2.5:1.5b \
    OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=-1 \
    MAX_CONCURRENCY=2 QUEUE_SIZE=4 MAX_TOKENS_LIMIT=256 MAX_INPUT_CHARS=4000 RATE_LIMIT_PER_MINUTE=10 \
    ARENA_DB=/home/user/app/data/arena.db

EXPOSE 7860
ENTRYPOINT []
CMD ["bash", "start.sh"]
