"""FastAPI application for backend access to monitoring data."""

from __future__ import annotations

from fastapi.responses import PlainTextResponse
from fastapi import FastAPI, HTTPException, Query

from analytics.access import enrich_event_rows
from db.repository import (
    init_db,
    link_event_to_employee,
    load_employees,
    load_events,
    load_system_settings,
    load_video_sources,
    load_worker_statuses,
    set_system_setting,
    set_video_source_active,
)
from services.system_api import load_dashboard_summary
from services.telemetry import build_prometheus_metrics, build_worker_runtime_metrics


def create_app() -> FastAPI:
    init_db()
    app = FastAPI(
        title="Human Tracker Backend API",
        version="0.1.0",
        description="Backend API layer for monitoring, analytics, and control endpoints.",
    )

    @app.get("/health")
    def health():
        settings = load_system_settings()
        video_sources = load_video_sources()
        worker_statuses = load_worker_statuses()
        events = enrich_event_rows(load_events(limit=200), video_sources, worker_statuses)
        telemetry = build_worker_runtime_metrics(
            video_sources=video_sources,
            worker_statuses=worker_statuses,
            events=events,
            settings=settings,
        )
        status = "ok" if telemetry["source_count_stale"] == 0 else "degraded"
        return {"status": status, "telemetry": telemetry}

    @app.get("/health/details")
    def health_details():
        return load_dashboard_summary(event_limit=200)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        settings = load_system_settings()
        video_sources = load_video_sources()
        worker_statuses = load_worker_statuses()
        events = enrich_event_rows(load_events(limit=500), video_sources, worker_statuses)
        telemetry = build_worker_runtime_metrics(
            video_sources=video_sources,
            worker_statuses=worker_statuses,
            events=events,
            settings=settings,
        )
        return build_prometheus_metrics(telemetry)

    @app.get("/api/v1/system/settings")
    def get_system_settings():
        return {"items": load_system_settings()}

    @app.put("/api/v1/system/settings/{key}")
    def put_system_setting(key: str, value: str = Query(..., min_length=1)):
        set_system_setting(key=key, value=value)
        return {"key": key, "value": value}

    @app.get("/api/v1/video-sources")
    def get_video_sources():
        return {"items": load_video_sources()}

    @app.put("/api/v1/video-sources/{source_id}/active")
    def put_video_source_active(source_id: int, is_active: bool = Query(...)):
        existing = {source["id"] for source in load_video_sources()}
        if source_id not in existing:
            raise HTTPException(status_code=404, detail="source_not_found")
        set_video_source_active(source_id=source_id, is_active=is_active)
        return {"source_id": source_id, "is_active": is_active}

    @app.get("/api/v1/worker-status")
    def get_worker_statuses():
        return {"items": load_worker_statuses()}

    @app.get("/api/v1/employees")
    def get_employees():
        return {"items": load_employees()}

    @app.get("/api/v1/events")
    def get_events(limit: int = Query(200, ge=1, le=5000)):
        video_sources = load_video_sources()
        worker_statuses = load_worker_statuses()
        events = enrich_event_rows(load_events(limit=limit), video_sources, worker_statuses)
        return {"items": events}

    @app.put("/api/v1/events/{event_id}/link")
    def put_event_link(event_id: str, employee_id: int = Query(..., ge=1), identification_status: str = Query(...), note: str = ""):
        try:
            link_event_to_employee(
                event_id=event_id,
                employee_id=employee_id,
                identification_status=identification_status,
                note=note,
            )
        except ValueError as exc:
            detail = str(exc)
            status_code = 404 if detail.startswith(("event_not_found", "employee_not_found")) else 400
            raise HTTPException(status_code=status_code, detail=detail) from exc
        return {
            "event_id": event_id,
            "employee_id": employee_id,
            "identification_status": identification_status,
            "note": note,
        }

    @app.get("/api/v1/dashboard/summary")
    def get_dashboard_summary(event_limit: int = Query(200, ge=1, le=5000)):
        return load_dashboard_summary(event_limit=event_limit)

    return app


app = create_app()
