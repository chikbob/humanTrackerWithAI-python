"""Dashboard view for enterprise monitored-zone operations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from analytics.access import (
    build_camera_health_summary,
    build_incident_hourly_distribution,
    build_incident_queue_rows,
    build_incident_severity_distribution,
    build_incident_status_summary,
    build_kpi_summary,
    build_offline_source_summary,
    build_source_risk_rows,
    build_top_event_types,
    build_zone_risk_rows,
)


def render_dashboard(
    st,
    *,
    events: list[dict],
    incidents: list[dict],
    worker_statuses: list[dict],
    video_sources: list[dict],
    access_points: list[dict],
    employees: list[dict],
):
    summary = build_kpi_summary(events, worker_statuses)
    incident_summary = build_incident_status_summary(incidents)
    health_summary = build_camera_health_summary(video_sources, worker_statuses)
    zone_risk_rows = build_zone_risk_rows(incidents)
    source_risk_rows = build_source_risk_rows(video_sources, worker_statuses, incidents)
    incident_queue_rows = build_incident_queue_rows(incidents, limit=12)
    severity_distribution = build_incident_severity_distribution(incidents)
    active_point = access_points[0]["name"] if access_points else "не задана"
    operator_count = sum(1 for employee in employees if employee.get("status") == "active")

    top1, top2, top3, top4, top5, top6 = st.columns(6)
    top1.metric("Камер online", summary["online_cameras"])
    top2.metric("Активные инциденты", incident_summary["active"])
    top3.metric("Критические", incident_summary["critical"])
    top4.metric("Зон под тревогой", incident_summary["zones_under_alert"])
    top5.metric("Средняя реакция, мин", incident_summary["mean_response_minutes"])
    top6.metric("Активных операторов", operator_count)

    second1, second2, second3, second4 = st.columns(4)
    second1.metric("Камер healthy", health_summary["healthy"])
    second2.metric("Камер degraded", health_summary["degraded"])
    second3.metric("Камер offline", health_summary["offline"])
    second4.metric("Ложные срабатывания", incident_summary["false_positive"])

    left_col, right_col = st.columns([1.25, 1.0], gap="large")
    with left_col:
        with st.container(border=True):
            st.subheader("Оперативная картина по площадке")
            statuses_by_id = {status["source_id"]: status for status in worker_statuses}
            active_sources = [source for source in video_sources if source.get("is_active")]
            active_source = active_sources[0] if active_sources else None
            active_status = statuses_by_id.get(active_source["id"]) if active_source else None
            if active_status and active_status.get("last_snapshot_path") and Path(active_status["last_snapshot_path"]).exists():
                st.image(active_status["last_snapshot_path"], width="stretch", caption=f"Источник обзора: {active_source['name']}")
            else:
                st.info("Обзорный snapshot пока недоступен. Ниже остаются аналитические сводки по инцидентам и состоянию камер.")
            overview_col1, overview_col2, overview_col3 = st.columns(3)
            overview_col1.metric("Приоритетная зона", active_point)
            overview_col2.metric("Инцидентов за сегодня", summary["suspicious_today"])
            overview_col3.metric("Обнаружений за сегодня", summary["detections_today"])
            if active_source:
                st.caption(
                    f"Обзорный источник: {active_source['name']} · Тип: {active_source['source_type']} · "
                    f"Локация: {active_source.get('location') or 'не указана'}"
                )

        with st.container(border=True):
            st.subheader("Интенсивность инцидентов по часам")
            distribution = build_incident_hourly_distribution(
                [
                    incident
                    for incident in incidents
                    if incident.get("started_at")
                    and datetime.fromtimestamp(float(incident["started_at"])).date() == datetime.now().date()
                ]
            )
            if distribution.empty:
                st.dataframe(pd.DataFrame(columns=["Час", "Инцидентов"]), width="stretch", hide_index=True)
                st.caption("За текущий день распределение инцидентов еще не сформировано.")
            else:
                chart_df = distribution.rename(columns={"hour": "Час", "count": "Инцидентов"}).set_index("Час")
                st.bar_chart(chart_df)

        with st.container(border=True):
            st.subheader("Риск по зонам")
            if zone_risk_rows:
                st.dataframe(pd.DataFrame(zone_risk_rows), width="stretch", hide_index=True)
            else:
                st.dataframe(
                    pd.DataFrame(columns=["Зона", "Всего инцидентов", "Активных", "Критических", "High+", "Последний инцидент"]),
                    width="stretch",
                    hide_index=True,
                )

    with right_col:
        with st.container(border=True):
            st.subheader("Очередь активных инцидентов")
            if incident_queue_rows:
                st.dataframe(pd.DataFrame(incident_queue_rows), width="stretch", hide_index=True)
            else:
                st.dataframe(
                    pd.DataFrame(columns=["ID", "Серьезность", "Инцидент", "Источник", "Зона", "Статус", "Время"]),
                    width="stretch",
                    hide_index=True,
                )
        with st.container(border=True):
            st.subheader("Распределение по серьезности")
            if severity_distribution.empty:
                st.dataframe(pd.DataFrame(columns=["severity", "count"]), width="stretch", hide_index=True)
            else:
                st.bar_chart(
                    severity_distribution.rename(columns={"severity": "Серьезность", "count": "Инцидентов"}).set_index("Серьезность")
                )
        with st.container(border=True):
            st.subheader("Состояние камер и эксплуатационные риски")
            st.dataframe(pd.DataFrame(source_risk_rows), width="stretch", hide_index=True)
        with st.container(border=True):
            st.subheader("Топ инцидентов и сбои")
            top_events = build_top_event_types(events, limit=6)
            st.dataframe(top_events.rename(columns={"event_type": "Тип инцидента", "count": "Количество"}), width="stretch", hide_index=True)
            offline_df = build_offline_source_summary(events)
            if offline_df.empty:
                st.caption("За текущий период offline-инциденты по камерам не зафиксированы.")
            else:
                st.dataframe(
                    offline_df.rename(columns={"source_name": "Источник", "offline_events": "Offline-инцидентов"}),
                    width="stretch",
                    hide_index=True,
                )


def _fmt_ts(timestamp_value):
    if not timestamp_value:
        return "—"
    return datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
