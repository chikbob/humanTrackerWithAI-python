"""24/7 background worker for production video sources."""

from __future__ import annotations

import time
from types import SimpleNamespace

import cv2

from config.app_config import SYSTEM_SETTING_DEFAULTS, normalize_source_processing_config, normalize_tracker_type
from db.repository import (
    db_insert_event,
    load_active_video_sources,
    load_system_settings,
    load_zone_rules,
    load_zones,
    set_video_source_active,
    update_video_source_last_seen,
    upsert_worker_status,
)
from services.events import create_domain_entry_event
from services.rules import build_effective_rule_profile
from video.ingest import SourceIngestSession
from video.pipeline import load_worker_model, process_source_frame
from video.runtime import create_runtime_session, ensure_runtime_dirs, write_snapshot_atomic


class SourceWorker:
    """Long-running processor for server-side RTSP/USB/URL sources."""

    def __init__(self):
        self.source_models = {}
        self.source_sessions = {}
        self.session_state = SimpleNamespace(events=[], notifications=[], db_insert_event=db_insert_event)
        self.ingests = {}
        self.connection_state = {}
        self.runtime_stats = {
            "worker_started_at": time.time(),
            "run_cycles_total": 0,
            "processed_sources_total": 0,
            "frames_processed_total": 0,
            "frames_skipped_total": 0,
            "stream_failures_total": 0,
        }

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
        self.runtime_stats["run_cycles_total"] += 1
        settings = self._read_settings()
        sources = load_active_video_sources()
        if not sources:
            return 0
        for source in sources:
            self._process_source(source, settings)
        self.runtime_stats["processed_sources_total"] += len(sources)
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
            "tracker_type": normalize_tracker_type(settings.get("tracker_type")),
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
                "last_frame_ts": 0.0,
                "last_processed_sequence": None,
                "offline_event_sent": False,
                "next_retry_ts": 0.0,
                "last_error_text": "",
                "frames_processed": 0,
                "frames_skipped": 0,
                "stream_failures": 0,
                "last_processing_time_ms": 0.0,
                "last_detection_count": 0,
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
        ingest = self._get_or_create_ingest(source, settings, source_runtime)
        if ingest is None:
            return
        frame_packet = ingest.get_latest_frame(last_sequence=source_runtime.get("last_processed_sequence"))
        if frame_packet is None:
            frame_age = ingest.latest_frame_age(now_ts=now_ts)
            if ingest.last_error:
                self._handle_stream_failure(source, settings, source_runtime, ingest.last_error)
                return
            if frame_age is None:
                if now_ts - ingest.started_at > max(1, int(settings["source_timeout"])):
                    self._handle_stream_failure(source, settings, source_runtime, "Источник не отдал кадры после запуска.")
                else:
                    self._write_status(source, "connecting", False, 0.0, "", "", last_frame_at=source_runtime.get("last_frame_ts"))
                return
            if frame_age > float(settings["source_timeout"]):
                self._handle_stream_failure(source, settings, source_runtime, "Источник перестал отдавать свежие кадры.")
                return
            self._write_status(source, "online", True, 0.0, "", "", last_frame_at=source_runtime.get("last_frame_ts"))
            return
        frame, frame_ts, frame_sequence = frame_packet

        source_runtime["frame_index"] += 1
        source_runtime["last_processed_sequence"] = frame_sequence
        source_runtime["last_frame_ts"] = frame_ts
        if settings["frame_skip"] > 0 and source_runtime["frame_index"] % (settings["frame_skip"] + 1) != 0:
            source_runtime["frames_skipped"] += 1
            self.runtime_stats["frames_skipped_total"] += 1
            self._write_status(source, "online", True, 0.0, "", "", last_frame_at=frame_ts)
            return

        model = self.source_models.get(source_id)
        if model is None:
            model = load_worker_model(settings["model_name"])
            self.source_models[source_id] = model

        session = self.source_sessions.get(source_id)
        if session is None:
            session = create_runtime_session(source, settings["model_name"])
            self.source_sessions[source_id] = session

        source_config = normalize_source_processing_config(source)
        active_zones = load_zones(source_id=source_id)
        active_rules = load_zone_rules(source_id=source_id)
        rule_profile = build_effective_rule_profile(source=source, zones=active_zones, zone_rules=active_rules)
        roi_config = rule_profile["roi_config"]
        event_settings = rule_profile["event_settings"]
        event_settings.update(
            {
                "notify_conf_threshold": settings["confidence_threshold"],
                "notify_classes": ["person"],
                "default_access_point_id": settings["default_access_point_id"],
                # Keep source-level config as fallback metadata for legacy behavior and future rules.
                "legacy_source_config": source_config,
            }
        )

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
            tracker_type=settings["tracker_type"],
        )
        snapshot_path = write_snapshot_atomic(source_id, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        now_ts = time.time()
        update_video_source_last_seen(source_id=source_id, last_seen=now_ts)
        fps = 1000.0 / processing_time_ms if processing_time_ms > 0 else 0.0
        source_runtime["last_success_ts"] = now_ts
        source_runtime["last_frame_ts"] = frame_ts
        source_runtime["next_retry_ts"] = 0.0
        source_runtime["last_error_text"] = ""
        source_runtime["frames_processed"] += 1
        source_runtime["last_processing_time_ms"] = processing_time_ms
        source_runtime["last_detection_count"] = len(detections)
        self.runtime_stats["frames_processed_total"] += 1
        self._write_status(source, "online", True, fps, "", snapshot_path, last_frame_at=frame_ts)

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

    def _get_or_create_ingest(self, source: dict, settings: dict, source_runtime: dict):
        source_id = source["id"]
        ingest = self.ingests.get(source_id)
        if ingest is not None and ingest.is_running():
            return ingest
        if ingest is not None:
            ingest.close()
        ingest = SourceIngestSession(source=source)
        if not ingest.start():
            self._handle_stream_failure(source, settings, source_runtime, ingest.last_error or "Не удалось открыть видеопоток.")
            self.ingests[source_id] = None
            return None
        source_runtime["last_error_text"] = ""
        self.ingests[source_id] = ingest
        return ingest

    def _handle_stream_failure(self, source: dict, settings: dict, source_runtime: dict, error_text: str):
        source_id = source["id"]
        existing_ingest = self.ingests.get(source_id)
        if existing_ingest is not None:
            existing_ingest.close()
        self.ingests[source_id] = None
        source_runtime["reconnect_count"] += 1
        source_runtime["last_error_text"] = error_text
        source_runtime["last_processed_sequence"] = None
        source_runtime["stream_failures"] += 1
        self.runtime_stats["stream_failures_total"] += 1
        source_runtime["next_retry_ts"] = time.time() + max(int(settings["reconnect_interval"]), 1)
        self._write_status(
            source,
            "reconnecting" if source_runtime["offline_event_sent"] else "offline",
            False,
            0.0,
            error_text,
            "",
            last_frame_at=source_runtime.get("last_frame_ts") or source.get("last_seen"),
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

    def get_runtime_snapshot(self) -> dict:
        now_ts = time.time()
        per_source = {}
        for source_id, runtime in self.connection_state.items():
            per_source[source_id] = {
                "frame_index": runtime.get("frame_index", 0),
                "frames_processed": runtime.get("frames_processed", 0),
                "frames_skipped": runtime.get("frames_skipped", 0),
                "stream_failures": runtime.get("stream_failures", 0),
                "reconnect_count": runtime.get("reconnect_count", 0),
                "last_processing_time_ms": runtime.get("last_processing_time_ms", 0.0),
                "last_detection_count": runtime.get("last_detection_count", 0),
                "last_success_age_sec": max(0.0, now_ts - float(runtime.get("last_success_ts") or now_ts)),
                "last_frame_age_sec": max(0.0, now_ts - float(runtime.get("last_frame_ts") or now_ts)),
            }
        return {
            **self.runtime_stats,
            "uptime_sec": max(0.0, now_ts - float(self.runtime_stats["worker_started_at"])),
            "active_ingests": sum(1 for ingest in self.ingests.values() if ingest is not None and ingest.is_running()),
            "tracked_sources": len(self.connection_state),
            "sources": per_source,
        }

    def close(self):
        """Release all captures so the worker can be stopped cleanly by a supervisor."""
        for ingest in self.ingests.values():
            if ingest is not None:
                ingest.close()
        self.ingests.clear()


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
