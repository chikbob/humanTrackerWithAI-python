"""24/7 background worker for production video sources."""

from __future__ import annotations

import time
from types import SimpleNamespace

import cv2

from config.app_config import SYSTEM_SETTING_DEFAULTS, build_ai_runtime_settings, normalize_source_processing_config, normalize_tracker_type
from db.repository import (
    attach_event_evidence,
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
from video.runtime import (
    EVIDENCE_CLIP_DIR,
    INCIDENT_SNAPSHOT_DIR,
    append_runtime_frame,
    collect_evidence_frames,
    create_runtime_session,
    ensure_runtime_dirs,
    purge_expired_runtime_files,
    trim_runtime_frame_buffer,
    write_evidence_clip_atomic,
    write_incident_snapshot_atomic,
    write_snapshot_atomic,
)


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
        self._cleanup_expired_evidence(settings)
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
            "ai_quality_profile": settings.get("ai_quality_profile", "balanced"),
            "incident_score_threshold": float(settings.get("incident_score_threshold", 0.55)),
            "tracking_iou_threshold": float(settings.get("tracking_iou_threshold", 0.5)),
            "incident_evidence_pre_seconds": max(1, int(settings.get("incident_evidence_pre_seconds", 4))),
            "incident_evidence_post_seconds": max(0, int(settings.get("incident_evidence_post_seconds", 4))),
            "incident_evidence_fps": max(1, int(settings.get("incident_evidence_fps", 8))),
            "incident_evidence_retention_days": max(1, int(settings.get("incident_evidence_retention_days", 14))),
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
            session = self.source_sessions.get(source_id)
            if session is not None:
                self._flush_ready_incident_evidence(source, session, settings, now_ts=now_ts)
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
        append_runtime_frame(session, frame_bgr=frame, frame_ts=frame_ts)
        trim_runtime_frame_buffer(
            session,
            keep_seconds=max(
                settings["incident_evidence_pre_seconds"] + settings["incident_evidence_post_seconds"] + 2,
                settings["source_timeout"],
            ),
            max_frames=max(
                settings["incident_evidence_fps"]
                * (settings["incident_evidence_pre_seconds"] + settings["incident_evidence_post_seconds"] + 4),
                120,
            ),
        )

        source_config = normalize_source_processing_config(source)
        ai_runtime = build_ai_runtime_settings(settings, source)
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
                "ai_runtime": ai_runtime,
            }
        )
        event_count_before = len(self.session_state.events)

        frame_rgb, detections, processing_time_ms = process_source_frame(
            frame_bgr=frame,
            model=model,
            source=source,
            session_state=self.session_state,
            session=session,
            frame_index=source_runtime["frame_index"],
            conf_threshold=ai_runtime["confidence_threshold"],
            inference_size=ai_runtime["inference_size"],
            roi_config=roi_config,
            event_settings=event_settings,
            tracker_type=ai_runtime["tracker_type"],
            tracking_iou_threshold=ai_runtime["tracking_iou_threshold"],
            incident_score_threshold=ai_runtime["incident_score_threshold"],
        )
        snapshot_path = write_snapshot_atomic(source_id, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        self._queue_incident_evidence_jobs(
            source,
            session,
            settings,
            frame_bgr=frame,
            new_events=self.session_state.events[event_count_before:],
        )
        self._flush_ready_incident_evidence(source, session, settings, now_ts=time.time())
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

    def _queue_incident_evidence_jobs(self, source: dict, session: dict, settings: dict, *, frame_bgr, new_events: list[dict]):
        if not new_events:
            return
        retention_until = time.time() + (settings["incident_evidence_retention_days"] * 86400)
        pending_jobs = session.setdefault("pending_evidence_jobs", [])
        existing_event_ids = {job["event_id"] for job in pending_jobs}
        for event in new_events:
            if event.get("event_scope") != "domain":
                continue
            if event.get("event_id") in existing_event_ids:
                continue
            event_source_id = source["id"] if event.get("session_id") == session["id"] else None
            if event_source_id != source["id"]:
                continue
            snapshot_path = write_incident_snapshot_atomic(source["id"], event["event_id"], frame_bgr)
            pending_jobs.append(
                {
                    "event_id": event["event_id"],
                    "event_ts": float(event.get("timestamp") or time.time()),
                    "snapshot_path": snapshot_path,
                    "retention_until": retention_until,
                    "target_ready_ts": float(event.get("timestamp") or time.time()) + settings["incident_evidence_post_seconds"],
                }
            )

    def _flush_ready_incident_evidence(self, source: dict, session: dict, settings: dict, *, now_ts: float | None = None):
        pending_jobs = session.setdefault("pending_evidence_jobs", [])
        if not pending_jobs:
            return
        current_ts = float(now_ts if now_ts is not None else time.time())
        remaining_jobs = []
        max_clip_frames = settings["incident_evidence_fps"] * (
            settings["incident_evidence_pre_seconds"] + settings["incident_evidence_post_seconds"] + 1
        )
        for job in pending_jobs:
            if current_ts < float(job["target_ready_ts"]):
                remaining_jobs.append(job)
                continue
            clip_start_ts = float(job["event_ts"]) - settings["incident_evidence_pre_seconds"]
            clip_end_ts = float(job["event_ts"]) + settings["incident_evidence_post_seconds"]
            frames_bgr = collect_evidence_frames(
                session,
                start_ts=clip_start_ts,
                end_ts=clip_end_ts,
                max_frames=max_clip_frames,
            )
            clip_path = ""
            if frames_bgr:
                clip_path = write_evidence_clip_atomic(
                    source["id"],
                    job["event_id"],
                    frames_bgr,
                    fps=settings["incident_evidence_fps"],
                )
            attach_event_evidence(
                event_id=job["event_id"],
                snapshot_path=job["snapshot_path"],
                evidence_clip_path=clip_path,
                evidence_retention_until=job["retention_until"],
            )
        session["pending_evidence_jobs"] = remaining_jobs

    def _cleanup_expired_evidence(self, settings: dict):
        now_ts = time.time()
        last_cleanup_ts = float(self.runtime_stats.get("last_evidence_cleanup_ts") or 0.0)
        if now_ts - last_cleanup_ts < 300:
            return
        self.runtime_stats["last_evidence_cleanup_ts"] = now_ts
        expire_before_ts = now_ts - (settings["incident_evidence_retention_days"] * 86400)
        removed_total = 0
        removed_total += purge_expired_runtime_files(INCIDENT_SNAPSHOT_DIR, expire_before_ts=expire_before_ts)
        removed_total += purge_expired_runtime_files(EVIDENCE_CLIP_DIR, expire_before_ts=expire_before_ts)
        self.runtime_stats["expired_evidence_removed_total"] = int(self.runtime_stats.get("expired_evidence_removed_total", 0)) + removed_total

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
