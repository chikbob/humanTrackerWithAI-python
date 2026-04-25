"""Read-side helpers for the backend API layer."""

from __future__ import annotations

import time

from analytics.access import build_kpi_summary, enrich_event_rows
from db.repository import (
    load_access_points,
    load_audit_logs,
    load_employees,
    load_events,
    load_incidents,
    load_system_settings,
    load_video_sources,
    load_worker_statuses,
)
from services.telemetry import build_worker_runtime_metrics


def build_incident_summary(incidents: list[dict]) -> dict:
    active_statuses = {"new", "acknowledged", "in_progress", "on_hold", "escalated"}
    ack_durations = []
    resolution_durations = []
    overdue_active = 0
    now_ts = time.time()
    for incident in incidents:
        started_at = float(incident.get("started_at") or 0.0)
        acknowledged_at = incident.get("acknowledged_at")
        resolved_at = incident.get("resolved_at")
        if started_at and acknowledged_at:
            ack_durations.append(max(0.0, float(acknowledged_at) - started_at) / 60.0)
        if started_at and resolved_at:
            resolution_durations.append(max(0.0, float(resolved_at) - started_at) / 60.0)
        if incident.get("status") in active_statuses and started_at and (now_ts - started_at) > 15 * 60:
            overdue_active += 1
    return {
        "total": len(incidents),
        "active": sum(1 for incident in incidents if incident.get("status") in active_statuses),
        "critical": sum(1 for incident in incidents if incident.get("severity") == "critical"),
        "high": sum(1 for incident in incidents if incident.get("severity") == "high"),
        "false_positive": sum(1 for incident in incidents if incident.get("status") == "false_positive"),
        "resolved": sum(1 for incident in incidents if incident.get("status") == "resolved"),
        "assigned": sum(1 for incident in incidents if (incident.get("assigned_to") or "").strip()),
        "mean_ack_minutes": round(sum(ack_durations) / len(ack_durations), 2) if ack_durations else 0.0,
        "mean_resolution_minutes": round(sum(resolution_durations) / len(resolution_durations), 2) if resolution_durations else 0.0,
        "overdue_active": overdue_active,
    }


def load_dashboard_summary(*, event_limit: int = 200) -> dict:
    settings = load_system_settings()
    video_sources = load_video_sources()
    worker_statuses = load_worker_statuses()
    access_points = load_access_points()
    employees = load_employees()
    events = enrich_event_rows(load_events(limit=event_limit), video_sources, worker_statuses)
    incidents = load_incidents(limit=event_limit)

    return {
        "settings": settings,
        "summary": build_kpi_summary(events, worker_statuses),
        "incidents_summary": build_incident_summary(incidents),
        "telemetry": build_worker_runtime_metrics(
            video_sources=video_sources,
            worker_statuses=worker_statuses,
            events=events,
            settings=settings,
        ),
        "video_sources": video_sources,
        "worker_statuses": worker_statuses,
        "access_points": access_points,
        "employees": employees,
        "recent_events": events[: min(event_limit, len(events))],
        "recent_incidents": incidents[: min(event_limit, len(incidents))],
        "recent_audit_logs": load_audit_logs(limit=min(event_limit, 100)),
    }
