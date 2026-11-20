"""Remembers recent answers so an identical question is answered instantly, without the model.

Only used for temperature 0 requests: those are repeatable (same input -> same output), so a stored
answer is exactly what the model would say again. Random (temperature > 0) answers are never cached.

LRU = "least recently used": when the cache is full, the entry nobody asked for longest is dropped.
TTL = "time to live": entries older than ttl_seconds are treated as missing.
"""
import hashlib
import json
import time
from collections import OrderedDict

from app.backends.base import GenerationParams, GenerationResult


class ResponseCache:
    def __init__(self, max_entries: int = 500, ttl_seconds: float = 3600, clock=time.monotonic):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.clock = clock  # replaceable in tests
        self._data: OrderedDict[str, tuple[float, GenerationResult]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(messages: list[dict], params: GenerationParams) -> str:
        """Same model + same conversation + same settings -> same key."""
        raw = json.dumps({"messages": messages, "model": params.model, "max_tokens": params.max_tokens,
                          "top_p": params.top_p, "format": params.format}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> GenerationResult | None:
        entry = self._data.get(key)
        if entry is None or self.clock() - entry[0] > self.ttl_seconds:
            self._data.pop(key, None)
            self.misses += 1
            return None
        self._data.move_to_end(key)  # mark as recently used
        self.hits += 1
        return entry[1]

    def put(self, key: str, result: GenerationResult) -> None:
        self._data[key] = (self.clock(), result)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)  # drop the least recently used

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "size": len(self._data),
                "hit_rate": round(self.hits / total, 3) if total else 0.0}
