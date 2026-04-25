"""FastAPI application for backend access to monitoring data."""

from __future__ import annotations

from fastapi.responses import PlainTextResponse
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Query

from analytics.access import enrich_event_rows
from db.repository import (
    append_audit_log,
    init_db,
    link_event_to_employee,
    load_audit_logs,
    load_employees,
    load_events,
    load_incidents,
    load_system_settings,
    load_video_sources,
    load_worker_statuses,
    set_system_setting,
    set_video_source_active,
    update_incident_status,
)
from services.system_api import build_incident_summary, load_dashboard_summary
from services.telemetry import build_health_payload, build_operational_summary, build_prometheus_metrics, build_worker_runtime_metrics


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
        incidents = load_incidents(limit=200)
        payload = build_health_payload(
            video_sources=video_sources,
            worker_statuses=worker_statuses,
            events=events,
            incidents=incidents,
            settings=settings,
        )
        return payload

    @app.get("/health/live")
    def health_live():
        return {"status": "ok", "service": "api"}

    @app.get("/health/ready")
    def health_ready():
        settings = load_system_settings()
        video_sources = load_video_sources()
        worker_statuses = load_worker_statuses()
        incidents = load_incidents(limit=500)
        operational = build_operational_summary(
            video_sources=video_sources,
            worker_statuses=worker_statuses,
            incidents=incidents,
            settings=settings,
        )
        ready = operational["readiness"] == "ready"
        status_code = 200 if ready else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if ready else operational["readiness"],
                "issues": operational["issues"],
                "coverage_ratio": operational["coverage_ratio"],
            },
        )

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

    @app.get("/api/v1/telemetry")
    def get_telemetry():
        settings = load_system_settings()
        video_sources = load_video_sources()
        worker_statuses = load_worker_statuses()
        events = enrich_event_rows(load_events(limit=500), video_sources, worker_statuses)
        incidents = load_incidents(limit=500)
        telemetry = build_worker_runtime_metrics(
            video_sources=video_sources,
            worker_statuses=worker_statuses,
            events=events,
            settings=settings,
        )
        operational = build_operational_summary(
            video_sources=video_sources,
            worker_statuses=worker_statuses,
            incidents=incidents,
            settings=settings,
        )
        return {"telemetry": telemetry, "operational": operational}

    @app.get("/api/v1/system/settings")
    def get_system_settings():
        return {"items": load_system_settings()}

    @app.put("/api/v1/system/settings/{key}")
    def put_system_setting(key: str, value: str = Query(..., min_length=1), actor_name: str = "api", actor_role: str = "admin"):
        set_system_setting(key=key, value=value)
        append_audit_log(
            actor_name=actor_name,
            actor_role=actor_role,
            action="system_setting.updated",
            resource_type="system_setting",
            resource_id=key,
            details={"value": value},
        )
        return {"key": key, "value": value}

    @app.get("/api/v1/video-sources")
    def get_video_sources():
        return {"items": load_video_sources()}

    @app.put("/api/v1/video-sources/{source_id}/active")
    def put_video_source_active(source_id: int, is_active: bool = Query(...), actor_name: str = "api", actor_role: str = "admin"):
        existing = {source["id"] for source in load_video_sources()}
        if source_id not in existing:
            raise HTTPException(status_code=404, detail="source_not_found")
        set_video_source_active(source_id=source_id, is_active=is_active)
        append_audit_log(
            actor_name=actor_name,
            actor_role=actor_role,
            action="video_source.activation_changed",
            resource_type="video_source",
            resource_id=str(source_id),
            details={"is_active": bool(is_active)},
        )
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

    @app.get("/api/v1/incidents")
    def get_incidents(limit: int = Query(200, ge=1, le=5000)):
        incidents = load_incidents(limit=limit)
        return {"items": incidents, "summary": build_incident_summary(incidents)}

    @app.put("/api/v1/incidents/{incident_id}/status")
    def put_incident_status(
        incident_id: int,
        status: str = Query(..., min_length=1),
        operator_comment: str = "",
        assigned_to: str = "",
        resolution_code: str = "",
        resolution_notes: str = "",
        actor_name: str = "api",
        actor_role: str = "admin",
    ):
        incident_ids = {incident["id"] for incident in load_incidents()}
        if incident_id not in incident_ids:
            raise HTTPException(status_code=404, detail="incident_not_found")
        update_incident_status(
            incident_id=incident_id,
            status=status,
            operator_comment=operator_comment,
            assigned_to=assigned_to,
            resolution_code=resolution_code,
            resolution_notes=resolution_notes,
        )
        append_audit_log(
            actor_name=actor_name,
            actor_role=actor_role,
            action="incident.status_updated",
            resource_type="incident",
            resource_id=str(incident_id),
            details={
                "status": status,
                "operator_comment": operator_comment,
                "assigned_to": assigned_to,
                "resolution_code": resolution_code,
                "resolution_notes": resolution_notes,
            },
        )
        return {
            "incident_id": incident_id,
            "status": status,
            "operator_comment": operator_comment,
            "assigned_to": assigned_to,
            "resolution_code": resolution_code,
            "resolution_notes": resolution_notes,
        }

    @app.put("/api/v1/events/{event_id}/link")
    def put_event_link(
        event_id: str,
        employee_id: int = Query(..., ge=1),
        identification_status: str = Query(...),
        note: str = "",
        actor_name: str = "api",
        actor_role: str = "admin",
    ):
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
        append_audit_log(
            actor_name=actor_name,
            actor_role=actor_role,
            action="incident.context_linked",
            resource_type="event",
            resource_id=event_id,
            details={"employee_id": employee_id, "identification_status": identification_status},
        )
        return {
            "event_id": event_id,
            "employee_id": employee_id,
            "identification_status": identification_status,
            "note": note,
        }

    @app.get("/api/v1/dashboard/summary")
    def get_dashboard_summary(event_limit: int = Query(200, ge=1, le=5000)):
        return load_dashboard_summary(event_limit=event_limit)

    @app.get("/api/v1/audit-logs")
    def get_audit_logs(limit: int = Query(200, ge=1, le=5000)):
        return {"items": load_audit_logs(limit=limit)}

    return app


app = create_app()
