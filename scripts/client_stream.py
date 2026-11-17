"""Demo: print a streamed answer in the terminal.  uv run python scripts/client_stream.py "your question" """
import json
import sys

import httpx

question = sys.argv[1] if len(sys.argv) > 1 else "Tell me a short joke."
body = {"stream": True, "messages": [{"role": "user", "content": question}]}

with httpx.stream("POST", "http://localhost:8000/v1/chat/completions", json=body, timeout=None) as r:
    for line in r.iter_lines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            break
        print(json.loads(data)["choices"][0]["delta"].get("content", ""), end="", flush=True)
print()
