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


# WebRTC worker threads emit a harmless Streamlit context warning on every frame.
# Lower this logger to keep the demo console readable without affecting behavior.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)


def main():
    init_db()
    init_session_state(st.session_state, load_history_from_db)
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

    system_settings = load_system_settings()
    video_sources = load_video_sources()
    production_video_sources = [source for source in video_sources if source.get("source_type") in PRODUCTION_SOURCE_TYPES]
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
        ensure_demo_employees()

    access_points = load_access_points()
    employee_repository = build_employee_repository(
        load_employees_fn=load_employees,
        replace_cache_fn=replace_employee_cache,
        load_sync_state_fn=load_employee_sync_state,
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
        employee_sync_state = employee_repository.get_status()
        employees = employee_repository.list_employees()
    identity_backend = system_settings.get("identity_backend", "disabled")
    st.session_state.identity_gallery_state = build_identity_runtime_state(
        employees=employees,
        sync_state=employee_sync_state,
        identity_backend=identity_backend,
    )
    worker_statuses = load_worker_statuses()
    audit_logs = load_audit_logs(limit=300)
    zones = load_zones()
    zone_rules = load_zone_rules()
    raw_events = load_events(limit=5000)
    events = enrich_event_rows(raw_events, video_sources, worker_statuses)
    sync_incidents_from_events(events, upsert_incident_fn=upsert_incident)
    incidents = load_incidents(limit=5000)
    process_incident_notifications(
        incidents=incidents,
        settings=system_settings,
        load_notification_deliveries_fn=load_notification_deliveries,
        upsert_notification_delivery_fn=upsert_notification_delivery,
    )

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

    def guarded_create_video_source(**kwargs):
        assert_permission(access_context, "manage_sources")
        create_video_source(**kwargs)
        audit("video_source.created", "video_source", kwargs.get("name", ""), {"source_type": kwargs.get("source_type")})

    def guarded_update_video_source(*, source_id: int, **kwargs):
        assert_permission(access_context, "manage_sources")
        update_video_source(source_id=source_id, **kwargs)
        audit("video_source.updated", "video_source", str(source_id), {"source_type": kwargs.get("source_type"), "name": kwargs.get("name")})

    def guarded_set_video_source_active(*, source_id: int, is_active: bool):
        assert_permission(access_context, "manage_sources")
        set_video_source_active(source_id=source_id, is_active=is_active)
        audit("video_source.activation_changed", "video_source", str(source_id), {"is_active": bool(is_active)})

    def guarded_create_zone(**kwargs):
        assert_permission(access_context, "manage_zones")
        create_zone(**kwargs)
        audit("zone.created", "zone", kwargs.get("name", ""), {"source_id": kwargs.get("source_id"), "zone_type": kwargs.get("zone_type")})

    def guarded_update_zone(*, zone_id: int, **kwargs):
        assert_permission(access_context, "manage_zones")
        update_zone(zone_id=zone_id, **kwargs)
        audit("zone.updated", "zone", str(zone_id), {"source_id": kwargs.get("source_id"), "zone_type": kwargs.get("zone_type")})

    def guarded_set_zone_active(*, zone_id: int, is_active: bool):
        assert_permission(access_context, "manage_zones")
        set_zone_active(zone_id=zone_id, is_active=is_active)
        audit("zone.activation_changed", "zone", str(zone_id), {"is_active": bool(is_active)})

    def guarded_create_zone_rule(**kwargs):
        assert_permission(access_context, "manage_zones")
        create_zone_rule(**kwargs)
        audit("zone_rule.created", "zone_rule", kwargs.get("rule_type", ""), {"zone_id": kwargs.get("zone_id"), "severity": kwargs.get("severity")})

    def guarded_update_zone_rule(*, rule_id: int, **kwargs):
        assert_permission(access_context, "manage_zones")
        update_zone_rule(rule_id=rule_id, **kwargs)
        audit("zone_rule.updated", "zone_rule", str(rule_id), {"zone_id": kwargs.get("zone_id"), "severity": kwargs.get("severity")})

    def guarded_set_zone_rule_active(*, rule_id: int, is_active: bool):
        assert_permission(access_context, "manage_zones")
        set_zone_rule_active(rule_id=rule_id, is_active=is_active)
        audit("zone_rule.activation_changed", "zone_rule", str(rule_id), {"is_active": bool(is_active)})

    def guarded_create_employee(**kwargs):
        assert_permission(access_context, "manage_directory")
        create_employee(**kwargs)
        audit("employee.created", "employee", kwargs.get("employee_number", ""), {"full_name": kwargs.get("full_name")})

    def guarded_update_employee(*, employee_id: int, **kwargs):
        assert_permission(access_context, "manage_directory")
        update_employee(employee_id=employee_id, **kwargs)
        audit("employee.updated", "employee", str(employee_id), {"full_name": kwargs.get("full_name"), "status": kwargs.get("status")})

    def guarded_update_employee_status(*, employee_id: int, status: str):
        assert_permission(access_context, "manage_directory")
        update_employee_status(employee_id=employee_id, status=status)
        audit("employee.status_updated", "employee", str(employee_id), {"status": status})

    def guarded_sync_employee_directory():
        assert_permission(access_context, "manage_directory")
        result = employee_repository.sync()
        audit("employee_directory.synced", "employee_directory", employee_repository.source_name, {"status": result.get("sync_status"), "last_error": result.get("last_error", "")})
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
                st.rerun()

    current_access_point_name = access_points[0]["name"] if access_points else "не задана"
    current_confidence = float(system_settings.get("confidence_threshold", 0.45))
    current_inference_size = int(system_settings.get("inference_size", 512))
    current_frame_skip = int(system_settings.get("frame_skip", 1))

    if standalone_live_mode:
        section = "Оперативный мониторинг"
    else:
        section = sidebar_state["section"]
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
            active_sources=[source for source in production_video_sources if source.get("is_active")],
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
            reset_and_seed_demo_data_fn=reset_and_seed_demo_data,
        )
    elif section == "Доступ и аудит":
        render_security_audit(st, access_context=access_context, audit_logs=audit_logs)


if __name__ == "__main__":
    main()
