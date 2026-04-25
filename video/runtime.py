"""Runtime helpers for source sessions and filesystem layout."""

from __future__ import annotations

from collections import deque
import os
import time
import uuid

import cv2


RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime_data")
SNAPSHOT_DIR = os.path.join(RUNTIME_DIR, "snapshots")
INCIDENT_SNAPSHOT_DIR = os.path.join(RUNTIME_DIR, "incident_snapshots")
EVIDENCE_CLIP_DIR = os.path.join(RUNTIME_DIR, "evidence_clips")


def ensure_runtime_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(INCIDENT_SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(EVIDENCE_CLIP_DIR, exist_ok=True)


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
        "track_entry_timestamps": {},
        "evidence_buffer": deque(),
        "pending_evidence_jobs": [],
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


def build_incident_snapshot_path(source_id: int, event_id: str) -> str:
    ensure_runtime_dirs()
    return os.path.join(INCIDENT_SNAPSHOT_DIR, f"source_{source_id}_event_{event_id}.jpg")


def write_incident_snapshot_atomic(source_id: int, event_id: str, frame_bgr) -> str:
    ensure_runtime_dirs()
    snapshot_path = build_incident_snapshot_path(source_id, event_id)
    temp_path = f"{snapshot_path}.tmp"
    if not cv2.imwrite(temp_path, frame_bgr):
        raise RuntimeError(f"Не удалось сохранить incident snapshot для source_id={source_id}, event_id={event_id}")
    os.replace(temp_path, snapshot_path)
    return snapshot_path


def build_evidence_clip_path(source_id: int, event_id: str) -> str:
    ensure_runtime_dirs()
    return os.path.join(EVIDENCE_CLIP_DIR, f"source_{source_id}_event_{event_id}.mp4")


def write_evidence_clip_atomic(source_id: int, event_id: str, frames_bgr: list, *, fps: int) -> str:
    if not frames_bgr:
        raise ValueError("frames_bgr must not be empty")
    ensure_runtime_dirs()
    clip_path = build_evidence_clip_path(source_id, event_id)
    temp_path = f"{clip_path}.tmp.mp4"
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        temp_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть VideoWriter для source_id={source_id}, event_id={event_id}")
    try:
        for frame_bgr in frames_bgr:
            writer.write(frame_bgr)
    finally:
        writer.release()
    os.replace(temp_path, clip_path)
    return clip_path


def append_runtime_frame(session: dict, *, frame_bgr, frame_ts: float):
    buffer = session.setdefault("evidence_buffer", deque())
    buffer.append({"timestamp": float(frame_ts), "frame_bgr": frame_bgr.copy()})


def trim_runtime_frame_buffer(session: dict, *, keep_seconds: float, max_frames: int | None = None):
    buffer = session.setdefault("evidence_buffer", deque())
    if not buffer:
        return
    keep_seconds = max(1.0, float(keep_seconds))
    cutoff_ts = float(buffer[-1]["timestamp"]) - keep_seconds
    while buffer and float(buffer[0]["timestamp"]) < cutoff_ts:
        buffer.popleft()
    if max_frames is not None and max_frames > 0:
        while len(buffer) > int(max_frames):
            buffer.popleft()


def collect_evidence_frames(session: dict, *, start_ts: float, end_ts: float, max_frames: int | None = None) -> list:
    frames = [
        item["frame_bgr"].copy()
        for item in session.get("evidence_buffer", [])
        if float(start_ts) <= float(item["timestamp"]) <= float(end_ts)
    ]
    if max_frames is not None and max_frames > 0 and len(frames) > int(max_frames):
        step = max(1, len(frames) // int(max_frames))
        frames = frames[::step][: int(max_frames)]
    return frames


def purge_expired_runtime_files(directory: str, *, expire_before_ts: float) -> int:
    if not os.path.isdir(directory):
        return 0
    removed = 0
    for entry in os.scandir(directory):
        if not entry.is_file():
            continue
        try:
            if entry.stat().st_mtime < float(expire_before_ts):
                os.remove(entry.path)
                removed += 1
        except FileNotFoundError:
            continue
    return removed
