"""Opt-in stage timing; CUDA synchronization occurs only when summarizing.

Wall times measure host elapsed intervals (including any synchronization already
performed by the code under inspection). CUDA event times measure stream
intervals, not pure kernel occupancy. Do not add device and wall times together.
"""
from __future__ import annotations

from contextlib import contextmanager
import time

import torch


class StageProfiler:
    def __init__(self, device="cpu", enabled=False):
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self._started = None
        self._wall = {}
        self._events = {}
        self._cached = None

    @contextmanager
    def stage(self, name, gpu=True):
        """Measure one interval; set gpu=False for CPU-only read/logging work."""
        if not self.enabled:
            yield
            return
        if self._started is None:
            self._started = time.perf_counter()
        self._cached = None
        start_event = end_event = None
        if gpu and self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self._wall.setdefault(name, []).append(elapsed)
            if end_event is not None:
                end_event.record(torch.cuda.current_stream(self.device))
                self._events.setdefault(name, []).append((start_event, end_event))

    def summary(self):
        if self._cached is not None:
            return self._cached
        synchronized = bool(self.enabled and self._events)
        if synchronized:
            torch.cuda.synchronize(self.device)
        stages = {}
        for name, wall in self._wall.items():
            device = [start.elapsed_time(end) for start, end in self._events.get(name, [])]
            stages[name] = {"calls": len(wall), "wall_ms_total": sum(wall),
                            "wall_ms_mean": sum(wall) / len(wall),
                            "device_ms_total": sum(device) if device else None,
                            "device_ms_mean": sum(device) / len(device) if device else None}
        self._cached = {"enabled": self.enabled, "device": str(self.device),
                        "elapsed_wall_ms": (time.perf_counter() - self._started) * 1000.0
                        if self._started is not None else 0.0,
                        "synchronized_cuda": synchronized, "stages": stages}
        return self._cached
