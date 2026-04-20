"""24/7 background worker for production video sources."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import cv2

from config.app_config import SYSTEM_SETTING_DEFAULTS
from db.repository import (
    db_insert_event,
    load_active_video_sources,
    load_system_settings,
    set_video_source_active,
    update_video_source_last_seen,
    upsert_worker_status,
)
from services.events import create_domain_entry_event
from services.source_service import normalize_source_url
from video.pipeline import load_worker_model, process_source_frame
from video.runtime import build_snapshot_path, create_runtime_session, ensure_runtime_dirs


class SourceWorker:
    """Long-running processor for server-side RTSP/USB/URL sources."""

    def __init__(self):
        self.source_models = {}
        self.source_sessions = {}
        self.session_state = SimpleNamespace(events=[], notifications=[], db_insert_event=db_insert_event)
        self.captures = {}
        self.connection_state = {}

    def run_forever(self):
        ensure_runtime_dirs()
        while True:
            processed_sources = self.run_once()
            if processed_sources == 0:
                time.sleep(2)
                continue
            time.sleep(0.05)

    def run_once(self) -> int:
        """Process one polling cycle so the worker can also be used in service wrappers and tests."""
        ensure_runtime_dirs()
        settings = self._read_settings()
        sources = load_active_video_sources()
        if not sources:
            return 0
        for source in sources:
            self._process_source(source, settings)
        return len(sources)

    def _read_settings(self):
        settings = SYSTEM_SETTING_DEFAULTS.copy()
        settings.update(load_system_settings())
        return {
            "confidence_threshold": float(settings["confidence_threshold"]),
            "frame_skip": int(settings["frame_skip"]),
            "inference_size": int(settings["inference_size"]),
            "event_cooldown": int(settings["event_cooldown"]),
            "reconnect_interval": int(settings["reconnect_interval"]),
            "source_timeout": int(settings["source_timeout"]),
            "model_name": settings["model_name"],
            "default_access_point_id": int(settings["active_access_point_id"]) if settings["active_access_point_id"] else None,
        }

    def _process_source(self, source: dict, settings: dict):
        source_id = source["id"]
        source_runtime = self.connection_state.setdefault(
            source_id,
            {
                "frame_index": 0,
                "reconnect_count": 0,
                "last_success_ts": 0.0,
                "offline_event_sent": False,
                "next_retry_ts": 0.0,
                "last_error_text": "",
            },
        )
        now_ts = time.time()
        if source_runtime["next_retry_ts"] and now_ts < source_runtime["next_retry_ts"]:
            self._write_status(
                source,
                "reconnecting",
                False,
                0.0,
                source_runtime.get("last_error_text", "Ожидание переподключения."),
                "",
                last_frame_at=source.get("last_seen"),
            )
            return
        cap = self.captures.get(source_id)
        if cap is None or not cap.isOpened():
            cap = self._open_capture(source, settings, source_runtime)
            if cap is None:
                return
            self.captures[source_id] = cap

        ret, frame = cap.read()
        if not ret:
            self._handle_stream_failure(source, settings, source_runtime, "Не удалось получить кадр.")
            return

        source_runtime["frame_index"] += 1
        if settings["frame_skip"] > 0 and source_runtime["frame_index"] % (settings["frame_skip"] + 1) != 0:
            self._write_status(source, "online", True, 0.0, "", "")
            return

        model = self.source_models.get(source_id)
        if model is None:
            model = load_worker_model(settings["model_name"])
            self.source_models[source_id] = model

        session = self.source_sessions.get(source_id)
        if session is None:
            session = create_runtime_session(source, settings["model_name"])
            self.source_sessions[source_id] = session

        roi_config = {
            "enable_roi": True,
            "roi_x": 20,
            "roi_y": 20,
            "roi_w": 60,
            "roi_h": 60,
        }
        event_settings = {
            "rule_count_enabled": False,
            "rule_class": "person",
            "rule_n": 3,
            "rule_t": 10,
            "rule_disappear_enabled": True,
            "rule_disappear_seconds": 5,
            "enable_notifications": False,
            "notify_conf_threshold": settings["confidence_threshold"],
            "notify_classes": ["person"],
            "enable_roi": True,
            "default_access_point_id": settings["default_access_point_id"],
            "prolonged_presence_seconds": 10,
        }

        frame_rgb, detections, processing_time_ms = process_source_frame(
            frame_bgr=frame,
            model=model,
            source=source,
            session_state=self.session_state,
            session=session,
            frame_index=source_runtime["frame_index"],
            conf_threshold=settings["confidence_threshold"],
            inference_size=settings["inference_size"],
            roi_config=roi_config,
            event_settings=event_settings,
        )
        snapshot_path = build_snapshot_path(source_id)
        cv2.imwrite(snapshot_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        now_ts = time.time()
        update_video_source_last_seen(source_id=source_id, last_seen=now_ts)
        fps = 1000.0 / processing_time_ms if processing_time_ms > 0 else 0.0
        source_runtime["last_success_ts"] = now_ts
        source_runtime["next_retry_ts"] = 0.0
        source_runtime["last_error_text"] = ""
        self._write_status(source, "online", True, fps, "", snapshot_path, last_frame_at=now_ts)

        if self.connection_state[source_id]["offline_event_sent"]:
            self.connection_state[source_id]["offline_event_sent"] = False
            create_domain_entry_event(
                self.session_state,
                db_insert_event,
                session=session,
                event_type="camera_reconnected",
                source_type=source["source_type"],
                frame_index=source_runtime["frame_index"],
                class_name="camera",
                confidence=1.0,
                message=f"Источник видеоданных '{source['name']}' восстановил соединение",
                access_point_id=settings["default_access_point_id"],
            )

    def _open_capture(self, source: dict, settings: dict, source_runtime: dict):
        normalized_source = normalize_source_url(source["source_type"], source["source_url"])
        cap = cv2.VideoCapture(normalized_source)
        if not cap.isOpened():
            self._handle_stream_failure(source, settings, source_runtime, "Не удалось открыть видеопоток.")
            return None
        return cap

    def _handle_stream_failure(self, source: dict, settings: dict, source_runtime: dict, error_text: str):
        source_id = source["id"]
        existing_cap = self.captures.get(source_id)
        if existing_cap is not None:
            existing_cap.release()
        self.captures[source_id] = None
        source_runtime["reconnect_count"] += 1
        source_runtime["last_error_text"] = error_text
        source_runtime["next_retry_ts"] = time.time() + max(int(settings["reconnect_interval"]), 1)
        self._write_status(
            source,
            "reconnecting" if source_runtime["offline_event_sent"] else "offline",
            False,
            0.0,
            error_text,
            "",
            last_frame_at=source.get("last_seen"),
        )
        if not source_runtime["offline_event_sent"]:
            source_runtime["offline_event_sent"] = True
            session = self.source_sessions.setdefault(source_id, create_runtime_session(source, settings["model_name"]))
            create_domain_entry_event(
                self.session_state,
                db_insert_event,
                session=session,
                event_type="stream_offline",
                source_type=source["source_type"],
                frame_index=source_runtime["frame_index"],
                class_name="camera",
                confidence=0.0,
                message=f"Источник видеоданных '{source['name']}' временно недоступен",
                access_point_id=settings["default_access_point_id"],
            )

    def _write_status(
        self,
        source: dict,
        status: str,
        is_connected: bool,
        fps: float,
        error_text: str,
        snapshot_path: str,
        last_frame_at=None,
    ):
        now_ts = time.time()
        source_runtime = self.connection_state[source["id"]]
        upsert_worker_status(
            source_id=source["id"],
            status=status,
            is_connected=is_connected,
            last_heartbeat=now_ts,
            last_frame_at=last_frame_at,
            fps=fps,
            reconnect_count=source_runtime["reconnect_count"],
            last_error=error_text,
            last_snapshot_path=snapshot_path,
        )

    def close(self):
        """Release all captures so the worker can be stopped cleanly by a supervisor."""
        for cap in self.captures.values():
            if cap is not None:
                cap.release()
        self.captures.clear()


def main(*, run_once: bool = False):
    worker = SourceWorker()
    try:
        if run_once:
            worker.run_once()
        else:
            worker.run_forever()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
