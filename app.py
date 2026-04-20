"""Streamlit entrypoint for the enterprise employee access monitoring system."""

from __future__ import annotations

import streamlit as st

from analytics.access import enrich_event_rows
from core.detection import build_class_meta, load_model
from db.repository import (
    create_employee,
    create_video_source,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    ensure_demo_employees,
    init_db,
    load_access_points,
    load_employees,
    load_events,
    load_history_from_db,
    load_system_settings,
    load_video_sources,
    load_worker_statuses,
    reset_and_seed_demo_data,
    set_system_setting,
    set_video_source_active,
    update_employee,
    update_employee_status,
    update_video_source,
)
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
from ui.sources import render_video_sources


def main():
    init_db()
    init_session_state(st.session_state, load_history_from_db)
    configure_page(st)
    query_params = st.query_params
    standalone_live_mode = query_params.get("view", "") == "live-window"
    preferred_live_source = query_params.get("source", "")

    system_settings = load_system_settings()
    video_sources = load_video_sources()
    if standalone_live_mode:
        sidebar_state = {
            "section": "Онлайн-мониторинг",
            "demo_mode": False,
            "model_name": system_settings.get("model_name", "yolov8s.pt"),
        }
        st.markdown("### Отдельное окно live monitoring")
        st.caption("Окно сфокусировано только на live-потоке и панели состояния.")
    else:
        sidebar_state = render_app_sidebar(st, video_sources=video_sources, system_settings=system_settings)
        if sidebar_state["demo_mode"]:
            ensure_demo_employees()

    access_points = load_access_points()
    employees = load_employees()
    worker_statuses = load_worker_statuses()
    raw_events = load_events(limit=5000)
    events = enrich_event_rows(raw_events, video_sources, worker_statuses)

    if not standalone_live_mode:
        with st.container():
            status_col1, status_col2, status_col3, status_col4 = st.columns([1.0, 1.0, 1.2, 0.8])
            status_col1.metric("Раздел", sidebar_state["section"])
            status_col2.metric("Фоновый worker", "online" if any(status.get("is_connected") for status in worker_statuses) else "standby")
            status_col3.metric("Production-источники", sum(1 for source in video_sources if source.get("is_active")))
            if status_col4.button("Обновить данные"):
                st.rerun()

    current_access_point_name = access_points[0]["name"] if access_points else "не задана"
    current_confidence = float(system_settings.get("confidence_threshold", 0.45))
    current_inference_size = int(system_settings.get("inference_size", 512))
    current_frame_skip = int(system_settings.get("frame_skip", 1))

    if standalone_live_mode:
        section = "Онлайн-мониторинг"
    else:
        section = sidebar_state["section"]
    if section == "Дашборд":
        render_dashboard(
            st,
            events=events,
            worker_statuses=worker_statuses,
            video_sources=video_sources,
            access_points=access_points,
            employees=employees,
        )
    elif section == "Онлайн-мониторинг":
        model = load_model(sidebar_state["model_name"])
        _, class_meta = build_class_meta(model.names, ANIMAL_CLASSES)
        render_online_monitoring(
            st,
            active_sources=[source for source in video_sources if source.get("is_active")],
            worker_statuses=worker_statuses,
            events=events,
            model_name=sidebar_state["model_name"],
            model=model,
            class_meta=class_meta,
            inference_size=current_inference_size,
            conf_threshold=current_confidence,
            frame_skip=current_frame_skip,
            access_point_name=current_access_point_name,
            session_state=st.session_state,
            db_insert_event=db_insert_event,
            db_insert_frame=db_insert_frame,
            db_upsert_session=db_upsert_session,
            demo_mode=sidebar_state["demo_mode"],
            preferred_source=preferred_live_source,
            standalone_mode=standalone_live_mode,
        )
    elif section == "Сотрудники":
        render_employees(
            st,
            employees=employees,
            create_employee_fn=create_employee,
            update_employee_fn=update_employee,
            update_employee_status_fn=update_employee_status,
        )
    elif section == "Журнал событий":
        render_event_journal(st, events=events)
    elif section == "Аналитика":
        render_access_analytics(st, events=events, worker_statuses=worker_statuses)
    elif section == "Источники видео":
        render_video_sources(
            st,
            video_sources=video_sources,
            worker_statuses=worker_statuses,
            create_video_source_fn=create_video_source,
            update_video_source_fn=update_video_source,
            set_video_source_active_fn=set_video_source_active,
            test_connection_fn=test_video_source_connection,
        )
    elif section == "Настройки системы":
        render_system_settings(
            st,
            settings=system_settings,
            access_points=access_points,
            set_system_setting_fn=set_system_setting,
            reset_and_seed_demo_data_fn=reset_and_seed_demo_data,
        )


if __name__ == "__main__":
    main()
