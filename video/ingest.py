"""Video ingest sessions with latest-frame semantics."""

from __future__ import annotations

import threading
import time

import cv2

from services.source_service import normalize_source_url
from video.frame_buffer import LatestFrameBuffer


class SourceIngestSession:
    """Background frame reader that always keeps the newest frame only."""

    def __init__(
        self,
        *,
        source: dict,
        capture_factory=None,
        idle_sleep_seconds: float = 0.01,
    ):
        self.source = source
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.idle_sleep_seconds = idle_sleep_seconds
        self.buffer = LatestFrameBuffer()
        self.capture = None
        self._thread = None
        self._stop_event = threading.Event()
        self.last_error = ""
        self.started_at = 0.0

    def start(self) -> bool:
        normalized_source = normalize_source_url(self.source["source_type"], self.source["source_url"])
        self.capture = self.capture_factory(normalized_source)
        if not self.capture or not self.capture.isOpened():
            self.last_error = "Не удалось открыть видеопоток."
            self.close()
            return False
        self.started_at = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, name=f"ingest-{self.source['id']}", daemon=True)
        self._thread.start()
        return True

    def _reader_loop(self):
        while not self._stop_event.is_set():
            if self.capture is None:
                self.last_error = "Источник видеоданных не инициализирован."
                return
            ret, frame = self.capture.read()
            if not ret:
                self.last_error = "Не удалось получить кадр."
                self.close(from_reader=True)
                return
            self.buffer.put(frame)
            if self.idle_sleep_seconds > 0:
                time.sleep(self.idle_sleep_seconds)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def get_latest_frame(self, *, last_sequence: int | None = None):
        if last_sequence is None:
            return self.buffer.get_latest()
        return self.buffer.get_if_newer(last_sequence)

    def latest_frame_age(self, *, now_ts: float | None = None) -> float | None:
        return self.buffer.latest_age(now_ts=now_ts)

    def close(self, *, from_reader: bool = False):
        self._stop_event.set()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if not from_reader and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._thread = None
