"""Read-side helpers for the backend API layer."""

from __future__ import annotations

from analytics.access import build_kpi_summary, enrich_event_rows
from db.repository import (
    load_access_points,
    load_employees,
    load_events,
    load_system_settings,
    load_video_sources,
    load_worker_statuses,
)
from services.telemetry import build_worker_runtime_metrics


def load_dashboard_summary(*, event_limit: int = 200) -> dict:
    settings = load_system_settings()
    video_sources = load_video_sources()
    worker_statuses = load_worker_statuses()
    access_points = load_access_points()
    employees = load_employees()
    events = enrich_event_rows(load_events(limit=event_limit), video_sources, worker_statuses)

    return {
        "settings": settings,
        "summary": build_kpi_summary(events, worker_statuses),
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
    }
