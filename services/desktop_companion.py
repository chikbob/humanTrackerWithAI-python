"""In-memory relay state for desktop camera companion sessions."""

from __future__ import annotations

import threading
import time
from typing import Any


_SESSION_TTL_SEC = 10.0
_sessions: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _session_connected(session: dict[str, Any], now_ts: float) -> bool:
    updated_at = float(session.get("updated_at") or 0.0)
    return bool(session.get("frame_bytes")) and (now_ts - updated_at) <= _SESSION_TTL_SEC


def store_desktop_companion_frame(
    *,
    session_id: str,
    frame_bytes: bytes,
    camera_index: int,
    width: int,
    height: int,
    source_label: str,
    backend_label: str = "",
    host_name: str = "",
) -> dict[str, Any]:
    now_ts = time.time()
    with _lock:
        _sessions[session_id] = {
            "session_id": session_id,
            "frame_bytes": frame_bytes,
            "camera_index": int(camera_index),
            "width": int(width),
            "height": int(height),
            "source_label": source_label.strip() or "Windows desktop companion",
            "backend_label": backend_label.strip(),
            "host_name": host_name.strip(),
            "updated_at": now_ts,
        }
        session = dict(_sessions[session_id])
    session.pop("frame_bytes", None)
    session["connected"] = True
    return session


def get_desktop_companion_status(session_id: str) -> dict[str, Any]:
    now_ts = time.time()
    with _lock:
        session = dict(_sessions.get(session_id) or {})
    if not session:
        return {
            "session_id": session_id,
            "connected": False,
            "has_frame": False,
            "updated_at": None,
        }
    has_frame = bool(session.get("frame_bytes"))
    connected = _session_connected(session, now_ts)
    session.pop("frame_bytes", None)
    session["connected"] = connected
    session["has_frame"] = has_frame
    session["age_ms"] = round(max(0.0, now_ts - float(session.get("updated_at") or 0.0)) * 1000)
    return session


def get_desktop_companion_frame(session_id: str) -> bytes | None:
    now_ts = time.time()
    with _lock:
        session = _sessions.get(session_id)
        if not session or not _session_connected(session, now_ts):
            return None
        return bytes(session["frame_bytes"])
