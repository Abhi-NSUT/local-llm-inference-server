"""Measures each request: time to first token, total time, and tokens per second."""
import logging
import time

log = logging.getLogger("llm.metrics")


class RequestTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.t_first: float | None = None

    def first_token(self) -> None:
        if self.t_first is None:
            self.t_first = time.perf_counter()

    def finish(self, completion_tokens: int, status: str = "ok") -> dict:
        t_end = time.perf_counter()
        t_first = self.t_first or t_end
        decode_time = max(t_end - t_first, 1e-6)  # time spent writing tokens after the first one
        m = {
            "status": status,
            "ttft_ms": round((t_first - self.t0) * 1000, 1),
            "latency_ms": round((t_end - self.t0) * 1000, 1),
            "completion_tokens": completion_tokens,
            "tokens_per_sec": round(max(completion_tokens - 1, 0) / decode_time, 1),
        }
        log.info("request_done %s", m)
        STATS.record(m)
        return m


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


class Stats:
    """Running totals since the server started (kept in memory, last 1000 requests)."""

    def __init__(self, keep: int = 1000):
        self.keep = keep
        self.records: list[dict] = []
        self.status_counts: dict[str, int] = {}

    def record(self, m: dict) -> None:
        self.status_counts[m["status"]] = self.status_counts.get(m["status"], 0) + 1
        if m["status"] == "ok":
            self.records = (self.records + [m])[-self.keep:]

    def summary(self) -> dict:
        ttft = [r["ttft_ms"] for r in self.records]
        lat = [r["latency_ms"] for r in self.records]
        tps = [r["tokens_per_sec"] for r in self.records]
        return {
            "requests_by_status": self.status_counts,
            "ttft_ms": {"p50": percentile(ttft, 50), "p95": percentile(ttft, 95)},
            "latency_ms": {"p50": percentile(lat, 50), "p95": percentile(lat, 95)},
            "tokens_per_sec_avg": round(sum(tps) / len(tps), 1) if tps else 0.0,
        }


STATS = Stats()
