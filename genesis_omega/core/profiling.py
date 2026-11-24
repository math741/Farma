"""Lightweight built-in profiler (spec §535). Telemetry only — never science."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    def __init__(self, sync=None):
        self.total = defaultdict(float)
        self.count = defaultdict(int)
        self.sync = sync  # callable that synchronises the device (only when measuring)
        self.enabled = True

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.total[name] += time.perf_counter() - t0
            self.count[name] += 1

    def report(self) -> dict:
        return {k: {"total_s": round(v, 4), "calls": self.count[k],
                    "mean_ms": round(1000 * v / max(1, self.count[k]), 4)} for k, v in sorted(self.total.items())}

    def reset(self) -> None:
        self.total.clear()
        self.count.clear()
