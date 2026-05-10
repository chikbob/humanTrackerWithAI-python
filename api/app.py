"""FastAPI application for backend access to monitoring data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from analytics.access import enrich_event_rows
from db.repository import (
    append_audit_log,
    create_video_source,
    init_db,
    link_event_to_employee,
    load_access_points,
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
    update_video_source,
)
from services.system_api import build_incident_summary, load_dashboard_summary
from services.telemetry import build_health_payload, build_operational_summary, build_prometheus_metrics, build_worker_runtime_metrics


class ActorPayload(BaseModel):
    actor_name: str = "api"
    actor_role: str = "admin"


class VideoSourcePayload(ActorPayload):
    name: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    location: str = ""
    description: str = ""
    is_active: bool = False
    enable_roi: bool = True
    roi_x: float = 20
    roi_y: float = 20
    roi_w: float = 60
    roi_h: float = 60
    rule_count_enabled: bool = False
    rule_n: int = 3
    rule_t: int = 10
    rule_disappear_enabled: bool = True
    rule_disappear_seconds: int = 5
    prolonged_presence_seconds: int = 10
    ai_profile_override: str = ""
    conf_threshold_override: float | None = None
    inference_size_override: int | None = None
    tracker_type_override: str = ""
    incident_threshold_override: float | None = None


class IncidentStatusPayload(ActorPayload):
    status: str = Field(min_length=1)
    operator_comment: str = ""
    assigned_to: str = ""
    resolution_code: str = ""
    resolution_notes: str = ""


class EventLinkPayload(ActorPayload):
    employee_id: int = Field(ge=1)
    identification_status: str = Field(min_length=1)
    note: str = ""


class SystemSettingPayload(ActorPayload):
    value: str = Field(min_length=1)


class SystemSettingsBulkPayload(ActorPayload):
    items: dict[str, str]


FRONTEND_DIST_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"


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
    def put_system_setting(key: str, payload: SystemSettingPayload):
        set_system_setting(key=key, value=payload.value)
        append_audit_log(
            actor_name=payload.actor_name,
            actor_role=payload.actor_role,
            action="system_setting.updated",
            resource_type="system_setting",
            resource_id=key,
            details={"value": payload.value},
        )
        return {"key": key, "value": payload.value}

    @app.put("/api/v1/system/settings")
    def put_system_settings(payload: SystemSettingsBulkPayload):
        updated: dict[str, Any] = {}
        for key, value in payload.items.items():
            normalized_value = str(value)
            set_system_setting(key=key, value=normalized_value)
            updated[key] = normalized_value
        append_audit_log(
            actor_name=payload.actor_name,
            actor_role=payload.actor_role,
            action="system_settings.bulk_updated",
            resource_type="system_settings",
            resource_id="bulk",
            details={"keys": sorted(updated.keys())},
        )
        return {"items": updated}

    @app.get("/api/v1/video-sources")
    def get_video_sources():
        return {"items": load_video_sources()}

    @app.post("/api/v1/video-sources")
    def post_video_source(payload: VideoSourcePayload):
        create_video_source(**payload.model_dump(exclude={"actor_name", "actor_role"}))
        append_audit_log(
            actor_name=payload.actor_name,
            actor_role=payload.actor_role,
            action="video_source.created",
            resource_type="video_source",
            resource_id=payload.name,
            details={"source_type": payload.source_type},
        )
        return {"ok": True}

    @app.put("/api/v1/video-sources/{source_id}")
    def put_video_source(source_id: int, payload: VideoSourcePayload):
        existing = {source["id"] for source in load_video_sources()}
        if source_id not in existing:
            raise HTTPException(status_code=404, detail="source_not_found")
        update_video_source(source_id=source_id, **payload.model_dump(exclude={"actor_name", "actor_role"}))
        append_audit_log(
            actor_name=payload.actor_name,
            actor_role=payload.actor_role,
            action="video_source.updated",
            resource_type="video_source",
            resource_id=str(source_id),
            details={"source_type": payload.source_type, "name": payload.name},
        )
        return {"ok": True, "source_id": source_id}

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

    @app.get("/api/v1/access-points")
    def get_access_points():
        return {"items": load_access_points()}

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
    def put_incident_status(incident_id: int, payload: IncidentStatusPayload):
        incident_ids = {incident["id"] for incident in load_incidents()}
        if incident_id not in incident_ids:
            raise HTTPException(status_code=404, detail="incident_not_found")
        update_incident_status(
            incident_id=incident_id,
            status=payload.status,
            operator_comment=payload.operator_comment,
            assigned_to=payload.assigned_to,
            resolution_code=payload.resolution_code,
            resolution_notes=payload.resolution_notes,
        )
        append_audit_log(
            actor_name=payload.actor_name,
            actor_role=payload.actor_role,
            action="incident.status_updated",
            resource_type="incident",
            resource_id=str(incident_id),
            details={
                "status": payload.status,
                "operator_comment": payload.operator_comment,
                "assigned_to": payload.assigned_to,
                "resolution_code": payload.resolution_code,
                "resolution_notes": payload.resolution_notes,
            },
        )
        return {
            "incident_id": incident_id,
            "status": payload.status,
            "operator_comment": payload.operator_comment,
            "assigned_to": payload.assigned_to,
            "resolution_code": payload.resolution_code,
            "resolution_notes": payload.resolution_notes,
        }

    @app.put("/api/v1/events/{event_id}/link")
    def put_event_link(event_id: str, payload: EventLinkPayload):
        try:
            link_event_to_employee(
                event_id=event_id,
                employee_id=payload.employee_id,
                identification_status=payload.identification_status,
                note=payload.note,
            )
        except ValueError as exc:
            detail = str(exc)
            status_code = 404 if detail.startswith(("event_not_found", "employee_not_found")) else 400
            raise HTTPException(status_code=status_code, detail=detail) from exc
        append_audit_log(
            actor_name=payload.actor_name,
            actor_role=payload.actor_role,
            action="incident.context_linked",
            resource_type="event",
            resource_id=event_id,
            details={"employee_id": payload.employee_id, "identification_status": payload.identification_status},
        )
        return {
            "event_id": event_id,
            "employee_id": payload.employee_id,
            "identification_status": payload.identification_status,
            "note": payload.note,
        }

    @app.get("/api/v1/dashboard/summary")
    def get_dashboard_summary(event_limit: int = Query(200, ge=1, le=5000)):
        return load_dashboard_summary(event_limit=event_limit)

    @app.get("/api/v1/audit-logs")
    def get_audit_logs(limit: int = Query(200, ge=1, le=5000)):
        return {"items": load_audit_logs(limit=limit)}

    if FRONTEND_DIST_DIR.exists():
        assets_dir = FRONTEND_DIST_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

        @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
        def frontend_index():
            return FileResponse(FRONTEND_DIST_DIR / "index.html")

        @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
        def frontend_route(full_path: str):
            if full_path.startswith(("api/", "health", "metrics", "docs", "openapi.json", "redoc", "_stcore/")):
                raise HTTPException(status_code=404, detail="not_found")
            target = FRONTEND_DIST_DIR / full_path
            if target.exists() and target.is_file():
                return FileResponse(target)
            return FileResponse(FRONTEND_DIST_DIR / "index.html")

    return app


app = create_app()
