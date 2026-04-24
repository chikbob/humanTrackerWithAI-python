"""Runtime helpers for source sessions and filesystem layout."""

from __future__ import annotations

import os
import time
import uuid

import cv2


RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime_data")
SNAPSHOT_DIR = os.path.join(RUNTIME_DIR, "snapshots")


def ensure_runtime_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def create_runtime_session(source: dict, model_name: str) -> dict:
    now_ts = time.time()
    return {
        "id": f"worker-{source['id']}-{uuid.uuid4().hex[:8]}",
        "model": model_name,
        "source_type": source["source_type"],
        "source_path": source["source_url"],
        "animal_filter": "всё",
        "class_filter": ["person"],
        "rotation_angle": 0,
        "started_at": now_ts,
        "finished_at": None,
        "total_frames": 0,
        "processed_frames": 0,
        "events_count": 0,
        "seen_track_keys": set(),
        "notified_track_keys": set(),
        "track_inside_roi": {},
        "track_last_seen": {},
        "track_class_by_key": {},
        "disappeared_track_keys": set(),
        "class_event_times": {},
        "rule_last_alert_ts": {},
        "track_first_seen": {},
        "track_domain_flags": {},
        "frames": [],
    }


def build_snapshot_path(source_id: int) -> str:
    ensure_runtime_dirs()
    return os.path.join(SNAPSHOT_DIR, f"source_{source_id}_latest.jpg")


def write_snapshot_atomic(source_id: int, frame_bgr) -> str:
    ensure_runtime_dirs()
    snapshot_path = build_snapshot_path(source_id)
    temp_path = f"{snapshot_path}.tmp"
    if not cv2.imwrite(temp_path, frame_bgr):
        raise RuntimeError(f"Не удалось сохранить snapshot для source_id={source_id}")
    os.replace(temp_path, snapshot_path)
    return snapshot_path
