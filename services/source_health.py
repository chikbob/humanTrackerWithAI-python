"""Shared worker/source health classification helpers."""

from __future__ import annotations

import time


HEALTH_LABELS = {
    "healthy": "healthy",
    "degraded": "degraded",
    "offline": "offline",
    "idle": "idle",
}

CONNECTION_LABELS = {
    "online": "online",
    "connecting": "connecting",
    "reconnecting": "reconnecting",
    "offline": "offline",
    "idle": "idle",
}


def normalize_source_runtime_status(
    status: dict | None,
    *,
    source_timeout: int | float = 15,
    now_ts: float | None = None,
) -> dict:
    status = status or {}
    current_ts = float(now_ts if now_ts is not None else time.time())
    timeout_seconds = max(1.0, float(source_timeout))
    is_connected = bool(status.get("is_connected"))
    raw_status = (status.get("status") or "").strip().lower()
    reconnect_count = int(status.get("reconnect_count") or 0)
    fps = float(status.get("fps") or 0.0)
    last_frame_at = status.get("last_frame_at")
    last_error = (status.get("last_error") or "").strip()
    has_stale_frame = bool(last_frame_at and (current_ts - float(last_frame_at)) > timeout_seconds)

    if raw_status in {"connecting", "reconnecting"}:
        connection_status = raw_status
    elif is_connected:
        connection_status = "online"
    elif reconnect_count > 0:
        connection_status = "reconnecting"
    elif last_error:
        connection_status = "offline"
    else:
        connection_status = "idle"

    if connection_status in {"offline", "reconnecting"}:
        health_status = "offline"
    elif connection_status == "connecting":
        health_status = "degraded"
    elif connection_status == "idle":
        health_status = "idle"
    elif has_stale_frame or fps < 3.0 or reconnect_count > 0 or last_error:
        health_status = "degraded"
    else:
        health_status = "healthy"

    return {
        "connection_status": connection_status,
        "health_status": health_status,
        "is_connected": is_connected,
        "fps": round(fps, 2),
        "reconnect_count": reconnect_count,
        "last_frame_at": last_frame_at,
        "last_error": last_error,
        "is_stale": has_stale_frame,
    }
