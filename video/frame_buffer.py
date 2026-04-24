"""Thread-safe latest-frame buffer for low-latency video ingestion."""

from __future__ import annotations

import threading
import time


class LatestFrameBuffer:
    """Keep only the newest frame to avoid inference backlog."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._sequence = 0

    def put(self, frame, *, timestamp: float | None = None) -> int:
        frame_ts = float(timestamp if timestamp is not None else time.time())
        with self._lock:
            self._sequence += 1
            self._frame = frame
            self._timestamp = frame_ts
            return self._sequence

    def get_latest(self):
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy() if hasattr(self._frame, "copy") else self._frame
            return frame, self._timestamp, self._sequence

    def get_if_newer(self, sequence: int | None):
        with self._lock:
            if self._frame is None:
                return None
            if sequence is not None and self._sequence <= sequence:
                return None
            frame = self._frame.copy() if hasattr(self._frame, "copy") else self._frame
            return frame, self._timestamp, self._sequence

    def latest_age(self, *, now_ts: float | None = None) -> float | None:
        with self._lock:
            if self._frame is None:
                return None
            current_ts = float(now_ts if now_ts is not None else time.time())
            return max(0.0, current_ts - self._timestamp)
