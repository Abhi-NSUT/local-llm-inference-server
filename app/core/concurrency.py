"""Controls how many requests run at once, and how many may wait.

Analogy: a help desk with N agents (the semaphore) and a waiting room with limited chairs (queue_size).
If every chair is taken, new visitors are turned away immediately (HTTP 429) instead of waiting forever.
"""
import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class QueueFullError(Exception):
    pass


class ConcurrencyLimiter:
    def __init__(self, max_concurrency: int, queue_size: int):
        self._sem = asyncio.Semaphore(max_concurrency)
        self.capacity = max_concurrency
        self.queue_size = queue_size
        self.waiting = 0    # requests sitting in the waiting room
        self.in_flight = 0  # requests being answered right now

    def is_full(self) -> bool:
        return self.waiting >= self.queue_size

    @asynccontextmanager
    async def slot(self):
        if self.is_full():
            raise QueueFullError()
        self.waiting += 1
        try:
            await self._sem.acquire()  # wait here until an "agent" is free
        finally:
            self.waiting -= 1
        self.in_flight += 1
        try:
            yield
        finally:  # always runs: normal end, error, timeout, or client disconnect
            self.in_flight -= 1
            self._sem.release()


async def with_deadline(stream: AsyncIterator[str], seconds: float) -> AsyncIterator[str]:
    """Pass items through from `stream`, but raise TimeoutError if the whole thing takes too long."""
    deadline = time.monotonic() + seconds
    it = aiter(stream)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            try:
                item = await asyncio.wait_for(anext(it), remaining)
            except StopAsyncIteration:
                return
            yield item
    finally:
        await it.aclose()  # stop the model if we leave early
