"""Operational telemetry helpers for worker and API observability."""

from __future__ import annotations

from datetime import datetime

from services.source_health import normalize_source_runtime_status


def build_worker_runtime_metrics(*, video_sources: list[dict], worker_statuses: list[dict], events: list[dict], settings: dict) -> dict:
    statuses_by_id = {status["source_id"]: status for status in worker_statuses}
    normalized_statuses = [
        normalize_source_runtime_status(status, source_timeout=int(settings.get("source_timeout", 15) or 15))
        for status in worker_statuses
    ]
    active_sources = [source for source in video_sources if source.get("is_active")]
    online_sources = [status for status in normalized_statuses if status["connection_status"] == "online"]
    offline_sources = [status for status in normalized_statuses if status["health_status"] == "offline"]
    degraded_sources = [status for status in normalized_statuses if status["health_status"] == "degraded"]
    now = datetime.now().date()
    today_events = [event for event in events if event.get("timestamp") and datetime.fromtimestamp(event["timestamp"]).date() == now]

    avg_fps = 0.0
    if normalized_statuses:
        avg_fps = sum(float(status.get("fps") or 0.0) for status in normalized_statuses) / max(1, len(normalized_statuses))

    stale_sources = 0
    source_timeout = int(settings.get("source_timeout", 15) or 15)
    now_ts = datetime.now().timestamp()
    for source in active_sources:
        status = statuses_by_id.get(source["id"], {})
        last_frame_at = status.get("last_frame_at")
        if last_frame_at and (now_ts - float(last_frame_at)) > source_timeout:
            stale_sources += 1

    return {
        "source_count_total": len(video_sources),
        "source_count_active": len(active_sources),
        "source_count_online": len(online_sources),
        "source_count_offline": len(offline_sources),
        "source_count_degraded": len(degraded_sources),
        "source_count_stale": stale_sources,
        "worker_avg_fps": round(avg_fps, 3),
        "worker_max_reconnect_count": max((int(status.get("reconnect_count") or 0) for status in normalized_statuses), default=0),
        "events_today_total": len(today_events),
        "events_today_suspicious": sum(1 for event in today_events if event.get("is_suspicious")),
    }


def build_prometheus_metrics(metrics: dict) -> str:
    lines = []
    for key, value in metrics.items():
        metric_name = f"human_tracker_{key}"
        lines.append(f"# TYPE {metric_name} gauge")
        lines.append(f"{metric_name} {float(value)}")
    return "\n".join(lines) + "\n"
