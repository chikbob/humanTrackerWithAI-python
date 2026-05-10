"""Streamlit entrypoint for the enterprise monitored-zone operations system."""

from __future__ import annotations

import logging

import streamlit as st

from analytics.access import enrich_event_rows
from core.detection import build_class_meta, load_model
from db.repository import (
    append_audit_log,
    create_employee,
    create_video_source,
    create_zone,
    create_zone_rule,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    ensure_demo_employees,
    init_db,
    load_access_points,
    load_audit_logs,
    load_events,
    load_history_from_db,
    load_employee_sync_state,
    load_employees,
    load_system_settings,
    load_video_sources,
    load_worker_statuses,
    load_zones,
    load_zone_rules,
    load_incidents,
    load_notification_deliveries,
    replace_employee_cache,
    reset_and_seed_demo_data,
    set_system_setting,
    set_video_source_active,
    set_zone_active,
    set_zone_rule_active,
    update_incident_status,
    upsert_notification_delivery,
    link_event_to_employee,
    upsert_incident,
    upsert_employee_sync_state,
    update_employee,
    update_employee_status,
    update_video_source,
    update_zone,
    update_zone_rule,
)
from services.auth import assert_permission, build_access_context, ensure_access_context
from services.employee_repository import build_employee_repository
from services.employee_sync import maybe_sync_employee_directory
from services.identity_service import build_identity_runtime_state
from services.incidents import sync_incidents_from_events
from services.notifications import process_incident_notifications
from services.source_service import test_video_source_connection
from services.state import init_session_state
from ui.analytics_views import render_access_analytics
from ui.dashboard import render_dashboard
from ui.employees import render_employees
from ui.journal import render_event_journal
from ui.monitoring import render_online_monitoring
from ui.page import configure_page
from ui.settings import render_system_settings
from ui.sidebar import ANIMAL_CLASSES, render_app_sidebar
from ui.security import render_security_audit
from ui.sources import render_video_sources
from ui.zones import render_zones

PRODUCTION_SOURCE_TYPES = {"rtsp", "stream_url", "usb_camera"}
OPERATOR_MONITOR_SOURCE_TYPES = PRODUCTION_SOURCE_TYPES | {"browser_camera"}


# WebRTC worker threads emit a harmless Streamlit context warning on every frame.
# Lower this logger to keep the demo console readable without affecting behavior.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

EVENT_LIMIT_MONITORING = 800
EVENT_LIMIT_DASHBOARD = 2000
EVENT_LIMIT_JOURNAL = 2000
EVENT_LIMIT_ANALYTICS = 5000

EVENT_SECTIONS = {"Ситуационный центр", "Оперативный мониторинг", "Журнал инцидентов", "Аналитика и отчеты"}
INCIDENT_SECTIONS = {"Ситуационный центр", "Журнал инцидентов"}
EMPLOYEE_SECTIONS = {"Ситуационный центр", "Оперативный мониторинг", "Журнал инцидентов", "Справочник персонала"}
ACCESS_POINT_SECTIONS = {"Ситуационный центр", "Оперативный мониторинг", "Настройки системы"}


@st.cache_data(ttl=3, show_spinner=False)
def load_cached_system_settings():
    return load_system_settings()


@st.cache_data(ttl=3, show_spinner=False)
def load_cached_video_sources():
    return load_video_sources()


@st.cache_data(ttl=3, show_spinner=False)
def load_cached_worker_statuses():
    return load_worker_statuses()


@st.cache_data(ttl=10, show_spinner=False)
def load_cached_access_points():
    return load_access_points()


@st.cache_data(ttl=10, show_spinner=False)
def load_cached_zones():
    return load_zones()


@st.cache_data(ttl=10, show_spinner=False)
def load_cached_zone_rules():
    return load_zone_rules()


@st.cache_data(ttl=10, show_spinner=False)
def load_cached_audit_logs(limit: int):
    return load_audit_logs(limit=limit)


@st.cache_data(ttl=5, show_spinner=False)
def load_cached_events(limit: int):
    return load_events(limit=limit)


@st.cache_data(ttl=10, show_spinner=False)
def load_cached_employees():
    return load_employees()


@st.cache_data(ttl=10, show_spinner=False)
def load_cached_employee_sync_state():
    return load_employee_sync_state()


def invalidate_ui_cache(session_state) -> None:
    st.cache_data.clear()
    session_state.pop("incident_sync_marker", None)
    session_state.pop("notification_sync_marker", None)


def maybe_run_incident_sync(session_state, events: list[dict]) -> None:
    if not events:
        return
    latest_event = events[0]
    marker = (
        latest_event.get("event_id"),
        latest_event.get("timestamp"),
        len(events),
    )
    if session_state.get("incident_sync_marker") == marker:
        return
    sync_incidents_from_events(events, upsert_incident_fn=upsert_incident)
    session_state.incident_sync_marker = marker


def maybe_process_notifications(session_state, incidents: list[dict], settings: dict) -> None:
    if str(settings.get("notifications_enabled", "0")) != "1" or not incidents:
        return
    latest_incident = incidents[0]
    marker = (
        latest_incident.get("id"),
        latest_incident.get("updated_at"),
        len(incidents),
        settings.get("incident_notify_min_severity"),
        settings.get("webhook_enabled"),
        settings.get("webhook_url"),
        settings.get("telegram_enabled"),
        settings.get("telegram_chat_id"),
    )
    if session_state.get("notification_sync_marker") == marker:
        return
    process_incident_notifications(
        incidents=incidents,
        settings=settings,
        load_notification_deliveries_fn=load_notification_deliveries,
        upsert_notification_delivery_fn=upsert_notification_delivery,
    )
    session_state.notification_sync_marker = marker


def section_event_limit(section: str) -> int:
    if section == "Оперативный мониторинг":
        return EVENT_LIMIT_MONITORING
    if section == "Журнал инцидентов":
        return EVENT_LIMIT_JOURNAL
    if section == "Аналитика и отчеты":
        return EVENT_LIMIT_ANALYTICS
    return EVENT_LIMIT_DASHBOARD


def main():
    init_db()
    init_session_state(st.session_state, load_history_from_db, load_history=False)
    ensure_access_context(st.session_state)
    query_params = st.query_params
    standalone_live_mode = query_params.get("view", "") == "live-window"
    standalone_overlay_enabled = query_params.get("overlay", "1") != "0"
    configure_page(st, standalone_mode=standalone_live_mode)
    preferred_live_source = query_params.get("source", "")
    preferred_live_source_id = query_params.get("source_id", "")
    preferred_live_source_kind = query_params.get("source_kind", "")
    if standalone_live_mode:
        st.markdown(
            """
            <style>
                header[data-testid="stHeader"],
                [data-testid="stToolbar"],
                [data-testid="stSidebar"],
                [data-testid="stDecoration"],
                #MainMenu,
                footer {
                    display: none !important;
                }
                .block-container {
                    max-width: 100vw !important;
                    padding: 0 !important;
                    margin: 0 !important;
                }
                .stApp,
                [data-testid="stAppViewContainer"],
                [data-testid="stMainBlockContainer"] {
                    background: #000 !important;
                    padding: 0 !important;
                    margin: 0 !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    system_settings = load_cached_system_settings()
    video_sources = load_cached_video_sources()
    production_video_sources = [source for source in video_sources if source.get("source_type") in PRODUCTION_SOURCE_TYPES]
    operator_monitor_sources = [source for source in video_sources if source.get("source_type") in OPERATOR_MONITOR_SOURCE_TYPES]
    access_context = build_access_context(st.session_state)
    if standalone_live_mode:
        sidebar_state = {
            "section": "Онлайн-мониторинг",
            "model_name": system_settings.get("model_name", "yolov8s.pt"),
        }
    else:
        monitored_source_count = len(st.session_state.get("monitoring_selected_labels") or []) or int(
            st.session_state.get("monitoring_selected_count") or 0
        )
        sidebar_state = render_app_sidebar(
            st,
            video_sources=production_video_sources,
            system_settings=system_settings,
            access_context=access_context,
            monitored_source_count=monitored_source_count,
        )
        access_context = build_access_context(st.session_state)
        if not st.session_state.get("demo_employees_checked"):
            ensure_demo_employees()
            st.session_state.demo_employees_checked = True

    section = "Оперативный мониторинг" if standalone_live_mode else sidebar_state["section"]

    access_points = load_cached_access_points() if section in ACCESS_POINT_SECTIONS else []
    worker_statuses = load_cached_worker_statuses()

    employee_repository = None
    employee_sync_state = {}
    employees = []
    if section in EMPLOYEE_SECTIONS:
        employee_repository = build_employee_repository(
            load_employees_fn=load_cached_employees,
            replace_cache_fn=replace_employee_cache,
            load_sync_state_fn=load_cached_employee_sync_state,
            upsert_sync_state_fn=upsert_employee_sync_state,
        )
        employee_sync_state = employee_repository.get_status()
        auto_sync_interval = int(system_settings.get("employee_sync_interval", 300))
        sync_triggered, employee_sync_state = maybe_sync_employee_directory(
            employee_repository,
            employee_sync_state,
            interval_seconds=auto_sync_interval,
        )
        employees = employee_repository.list_employees()
        if sync_triggered:
            invalidate_ui_cache(st.session_state)
            employee_repository = build_employee_repository(
                load_employees_fn=load_cached_employees,
                replace_cache_fn=replace_employee_cache,
                load_sync_state_fn=load_cached_employee_sync_state,
                upsert_sync_state_fn=upsert_employee_sync_state,
            )
            employee_sync_state = employee_repository.get_status()
            employees = employee_repository.list_employees()
        identity_backend = system_settings.get("identity_backend", "disabled")
        st.session_state.identity_gallery_state = build_identity_runtime_state(
            employees=employees,
            sync_state=employee_sync_state,
            identity_backend=identity_backend,
        )

    audit_logs = load_cached_audit_logs(limit=300) if section == "Доступ и аудит" else []
    zones = load_cached_zones() if section == "Камеры и зоны" else []
    zone_rules = load_cached_zone_rules() if section == "Камеры и зоны" else []
    events = []
    incidents = []
    if section in EVENT_SECTIONS:
        raw_events = load_cached_events(section_event_limit(section))
        events = enrich_event_rows(raw_events, video_sources, worker_statuses)
    if section in INCIDENT_SECTIONS:
        maybe_run_incident_sync(st.session_state, events)
        incidents = load_incidents(limit=EVENT_LIMIT_JOURNAL)
        maybe_process_notifications(st.session_state, incidents, system_settings)

    def audit(action: str, resource_type: str, resource_id: str = "", details: dict | None = None):
        append_audit_log(
            actor_name=access_context["actor_name"],
            actor_role=access_context["role"],
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
        )

    def guarded_set_system_setting(*, key: str, value: str):
        assert_permission(access_context, "manage_settings")
        set_system_setting(key=key, value=value)
        audit("system_setting.updated", "system_setting", key, {"value": value})
        invalidate_ui_cache(st.session_state)

    def guarded_update_incident_status(
        *,
        incident_id: int,
        status: str,
        operator_comment: str | None = None,
        assigned_to: str | None = None,
        resolution_code: str | None = None,
        resolution_notes: str | None = None,
    ):
        assert_permission(access_context, "update_incidents")
        update_incident_status(
            incident_id=incident_id,
            status=status,
            operator_comment=operator_comment,
            assigned_to=assigned_to,
            resolution_code=resolution_code,
            resolution_notes=resolution_notes,
        )
        audit(
            "incident.status_updated",
            "incident",
            str(incident_id),
            {
                "status": status,
                "operator_comment": operator_comment or "",
                "assigned_to": assigned_to or "",
                "resolution_code": resolution_code or "",
                "resolution_notes": resolution_notes or "",
            },
        )
        invalidate_ui_cache(st.session_state)

    def guarded_link_event_to_employee(*, event_id: str, employee_id: int, identification_status: str, note: str = ""):
        assert_permission(access_context, "link_incidents")
        link_event_to_employee(
            event_id=event_id,
            employee_id=employee_id,
            identification_status=identification_status,
            note=note,
        )
        audit(
            "incident.context_linked",
            "event",
            event_id,
            {"employee_id": employee_id, "identification_status": identification_status},
        )
        invalidate_ui_cache(st.session_state)

    def guarded_create_video_source(**kwargs):
        assert_permission(access_context, "manage_sources")
        create_video_source(**kwargs)
        audit("video_source.created", "video_source", kwargs.get("name", ""), {"source_type": kwargs.get("source_type")})
        invalidate_ui_cache(st.session_state)

    def guarded_update_video_source(*, source_id: int, **kwargs):
        assert_permission(access_context, "manage_sources")
        update_video_source(source_id=source_id, **kwargs)
        audit("video_source.updated", "video_source", str(source_id), {"source_type": kwargs.get("source_type"), "name": kwargs.get("name")})
        invalidate_ui_cache(st.session_state)

    def guarded_set_video_source_active(*, source_id: int, is_active: bool):
        assert_permission(access_context, "manage_sources")
        set_video_source_active(source_id=source_id, is_active=is_active)
        audit("video_source.activation_changed", "video_source", str(source_id), {"is_active": bool(is_active)})
        invalidate_ui_cache(st.session_state)

    def guarded_create_zone(**kwargs):
        assert_permission(access_context, "manage_zones")
        create_zone(**kwargs)
        audit("zone.created", "zone", kwargs.get("name", ""), {"source_id": kwargs.get("source_id"), "zone_type": kwargs.get("zone_type")})
        invalidate_ui_cache(st.session_state)

    def guarded_update_zone(*, zone_id: int, **kwargs):
        assert_permission(access_context, "manage_zones")
        update_zone(zone_id=zone_id, **kwargs)
        audit("zone.updated", "zone", str(zone_id), {"source_id": kwargs.get("source_id"), "zone_type": kwargs.get("zone_type")})
        invalidate_ui_cache(st.session_state)

    def guarded_set_zone_active(*, zone_id: int, is_active: bool):
        assert_permission(access_context, "manage_zones")
        set_zone_active(zone_id=zone_id, is_active=is_active)
        audit("zone.activation_changed", "zone", str(zone_id), {"is_active": bool(is_active)})
        invalidate_ui_cache(st.session_state)

    def guarded_create_zone_rule(**kwargs):
        assert_permission(access_context, "manage_zones")
        create_zone_rule(**kwargs)
        audit("zone_rule.created", "zone_rule", kwargs.get("rule_type", ""), {"zone_id": kwargs.get("zone_id"), "severity": kwargs.get("severity")})
        invalidate_ui_cache(st.session_state)

    def guarded_update_zone_rule(*, rule_id: int, **kwargs):
        assert_permission(access_context, "manage_zones")
        update_zone_rule(rule_id=rule_id, **kwargs)
        audit("zone_rule.updated", "zone_rule", str(rule_id), {"zone_id": kwargs.get("zone_id"), "severity": kwargs.get("severity")})
        invalidate_ui_cache(st.session_state)

    def guarded_set_zone_rule_active(*, rule_id: int, is_active: bool):
        assert_permission(access_context, "manage_zones")
        set_zone_rule_active(rule_id=rule_id, is_active=is_active)
        audit("zone_rule.activation_changed", "zone_rule", str(rule_id), {"is_active": bool(is_active)})
        invalidate_ui_cache(st.session_state)

    def guarded_create_employee(**kwargs):
        assert_permission(access_context, "manage_directory")
        create_employee(**kwargs)
        audit("employee.created", "employee", kwargs.get("employee_number", ""), {"full_name": kwargs.get("full_name")})
        invalidate_ui_cache(st.session_state)

    def guarded_update_employee(*, employee_id: int, **kwargs):
        assert_permission(access_context, "manage_directory")
        update_employee(employee_id=employee_id, **kwargs)
        audit("employee.updated", "employee", str(employee_id), {"full_name": kwargs.get("full_name"), "status": kwargs.get("status")})
        invalidate_ui_cache(st.session_state)

    def guarded_update_employee_status(*, employee_id: int, status: str):
        assert_permission(access_context, "manage_directory")
        update_employee_status(employee_id=employee_id, status=status)
        audit("employee.status_updated", "employee", str(employee_id), {"status": status})
        invalidate_ui_cache(st.session_state)

    def guarded_sync_employee_directory():
        assert_permission(access_context, "manage_directory")
        result = employee_repository.sync()
        audit("employee_directory.synced", "employee_directory", employee_repository.source_name, {"status": result.get("sync_status"), "last_error": result.get("last_error", "")})
        invalidate_ui_cache(st.session_state)
        return result

    def guarded_reset_and_seed_demo_data(*, employee_count: int = 120, visit_count: int = 900, seed: int = 42):
        assert_permission(access_context, "manage_settings")
        result = reset_and_seed_demo_data(employee_count=employee_count, visit_count=visit_count, seed=seed)
        audit(
            "database.demo_seeded",
            "database",
            "monitoring",
            {"employees": employee_count, "visits": visit_count, "seed": seed},
        )
        invalidate_ui_cache(st.session_state)
        return result

    if not standalone_live_mode:
        with st.container():
            status_col1, status_col2, status_col3, status_col4 = st.columns([1.0, 1.0, 1.2, 0.8])
            status_col1.metric("Раздел", sidebar_state["section"])
            status_col2.metric("Фоновый worker", "online" if any(status.get("is_connected") for status in worker_statuses) else "standby")
            status_col3.metric(
                "Источники в мониторинге",
                len(st.session_state.get("monitoring_selected_labels") or [])
                or int(st.session_state.get("monitoring_selected_count") or sum(1 for source in video_sources if source.get("is_active"))),
            )
            if status_col4.button("Обновить данные"):
                invalidate_ui_cache(st.session_state)
                st.rerun()

    current_access_point_name = access_points[0]["name"] if access_points else "не задана"
    current_confidence = float(system_settings.get("confidence_threshold", 0.45))
    current_inference_size = int(system_settings.get("inference_size", 512))
    current_frame_skip = int(system_settings.get("frame_skip", 1))

    if section == "Ситуационный центр":
        render_dashboard(
            st,
            events=events,
            incidents=incidents,
            worker_statuses=worker_statuses,
            video_sources=video_sources,
            access_points=access_points,
            employees=employees,
            settings=system_settings,
        )
    elif section == "Оперативный мониторинг":
        model = load_model(sidebar_state["model_name"])
        _, class_meta = build_class_meta(model.names, ANIMAL_CLASSES)
        render_online_monitoring(
            st,
            active_sources=[source for source in operator_monitor_sources if source.get("is_active")],
            worker_statuses=worker_statuses,
            events=events,
            model_name=sidebar_state["model_name"],
            model=model,
            class_meta=class_meta,
            inference_size=current_inference_size,
            conf_threshold=current_confidence,
            frame_skip=current_frame_skip,
            tracker_type=system_settings.get("tracker_type", "bytetrack"),
            access_point_name=current_access_point_name,
            session_state=st.session_state,
            db_insert_event=db_insert_event,
            db_insert_frame=db_insert_frame,
            db_upsert_session=db_upsert_session,
            preferred_source=preferred_live_source,
            preferred_source_id=preferred_live_source_id,
            preferred_source_kind=preferred_live_source_kind,
            standalone_mode=standalone_live_mode,
            standalone_overlay_enabled=standalone_overlay_enabled,
        )
    elif section == "Журнал инцидентов":
        render_event_journal(
            st,
            incidents=incidents,
            employees=employees,
            access_context=access_context,
            link_event_to_employee_fn=guarded_link_event_to_employee,
            update_incident_status_fn=guarded_update_incident_status,
        )
    elif section == "Аналитика и отчеты":
        render_access_analytics(st, events=events, worker_statuses=worker_statuses)
    elif section == "Камеры и зоны":
        render_zones(
            st,
            video_sources=video_sources,
            worker_statuses=worker_statuses,
            zones=zones,
            zone_rules=zone_rules,
            access_context=access_context,
            create_zone_fn=guarded_create_zone,
            update_zone_fn=guarded_update_zone,
            set_zone_active_fn=guarded_set_zone_active,
            create_zone_rule_fn=guarded_create_zone_rule,
            update_zone_rule_fn=guarded_update_zone_rule,
            set_zone_rule_active_fn=guarded_set_zone_rule_active,
        )
    elif section == "Подключение камер":
        render_video_sources(
            st,
            video_sources=video_sources,
            worker_statuses=worker_statuses,
            access_context=access_context,
            create_video_source_fn=guarded_create_video_source,
            update_video_source_fn=guarded_update_video_source,
            set_video_source_active_fn=guarded_set_video_source_active,
            test_connection_fn=test_video_source_connection,
        )
    elif section == "Справочник персонала":
        render_employees(
            st,
            employees=employees,
            sync_state=employee_sync_state,
            employee_data_source=employee_repository.source_name,
            employee_directory_read_only=employee_repository.is_read_only(),
            access_context=access_context,
            sync_employee_directory_fn=guarded_sync_employee_directory,
            create_employee_fn=guarded_create_employee,
            update_employee_fn=guarded_update_employee,
            update_employee_status_fn=guarded_update_employee_status,
        )
    elif section == "Настройки системы":
        render_system_settings(
            st,
            settings=system_settings,
            access_points=access_points,
            access_context=access_context,
            set_system_setting_fn=guarded_set_system_setting,
            reset_and_seed_demo_data_fn=guarded_reset_and_seed_demo_data,
        )
    elif section == "Доступ и аудит":
        render_security_audit(st, access_context=access_context, audit_logs=audit_logs)


if __name__ == "__main__":
    main()
