"""Per-visitor rate limit for the public demo: at most N requests per minute from one IP address.

Sliding window: we remember the times of each visitor's recent requests and drop the ones older
than the window. Different from the concurrency limiter, which protects the server as a whole;
this stops one visitor from using up the shared demo.
"""
import time
from collections import defaultdict, deque

from fastapi import Request


class RateLimiter:
    def __init__(self, limit_per_window: int, window_seconds: float = 60, clock=time.monotonic):
        self.limit = limit_per_window  # 0 = no limit
        self.window = window_seconds
        self.clock = clock
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now, recent = self.clock(), self.hits[key]
        while recent and now - recent[0] >= self.window:
            recent.popleft()
        if len(recent) >= self.limit:
            return False
        recent.append(now)
        return True


def client_ip(request: Request) -> str:
    """Behind a proxy (like Hugging Face Spaces) the visitor's address is in X-Forwarded-For.
    That header can be faked by a determined client, so this is a fair-use limit, not security."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
