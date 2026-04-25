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
    build_operator_workload_rows,
    build_source_risk_rows,
    build_top_event_types,
    build_zone_risk_rows,
)
from services.source_health import normalize_source_runtime_status
from services.telemetry import build_operational_summary


def _build_dashboard_guidance(*, video_sources: list[dict], worker_statuses: list[dict], incidents: list[dict]) -> list[str]:
    guidance = []
    active_sources = [source for source in video_sources if source.get("is_active")]
    if not video_sources:
        guidance.append("Камеры ещё не добавлены. Начните с раздела подключения камер и сохраните хотя бы один production-источник.")
    elif not active_sources:
        guidance.append("Источники есть, но ни один не активирован. Включите хотя бы одну production-камеру для запуска worker-first контура.")
    elif not worker_statuses:
        guidance.append("Worker ещё не записал статусы камер. Запустите worker и дождитесь первого heartbeat.")
    elif all(not status.get("is_connected") for status in worker_statuses):
        guidance.append("Ни одна камера сейчас не online. Проверьте подключение, статусы камер и логи worker.")
    if active_sources and not incidents:
        guidance.append("Активные камеры есть, но инцидентов пока нет. Это нормально для спокойного периода или до появления первых событий.")
    return guidance


def render_dashboard(
    st,
    *,
    events: list[dict],
    incidents: list[dict],
    worker_statuses: list[dict],
    video_sources: list[dict],
    access_points: list[dict],
    employees: list[dict],
    settings: dict | None = None,
):
    settings = settings or {}
    summary = build_kpi_summary(events, worker_statuses)
    incident_summary = build_incident_status_summary(incidents)
    health_summary = build_camera_health_summary(video_sources, worker_statuses)
    operational_summary = build_operational_summary(
        video_sources=video_sources,
        worker_statuses=worker_statuses,
        incidents=incidents,
        settings=settings,
    )
    zone_risk_rows = build_zone_risk_rows(incidents)
    source_risk_rows = build_source_risk_rows(video_sources, worker_statuses, incidents)
    incident_queue_rows = build_incident_queue_rows(incidents, limit=12)
    operator_workload_rows = build_operator_workload_rows(incidents, limit=8)
    severity_distribution = build_incident_severity_distribution(incidents)
    active_point = access_points[0]["name"] if access_points else "не задана"
    operator_count = sum(1 for employee in employees if employee.get("status") == "active")
    guidance_items = _build_dashboard_guidance(video_sources=video_sources, worker_statuses=worker_statuses, incidents=incidents)

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
    second4.metric("Active без owner", incident_summary["unassigned_active"])
    third1, third2, third3 = st.columns(3)
    third1.metric("Coverage active камер, %", operational_summary["coverage_ratio"])
    third2.metric("Active incidents без owner", operational_summary["active_incidents_unassigned"])
    third3.metric("Operational status", operational_summary["status"])

    for message in guidance_items:
        st.info(message)

    left_col, right_col = st.columns([1.25, 1.0], gap="large")
    with left_col:
        with st.container(border=True):
            st.subheader("Оперативная картина по площадке")
            statuses_by_id = {status["source_id"]: status for status in worker_statuses}
            active_sources = [source for source in video_sources if source.get("is_active")]
            active_source = active_sources[0] if active_sources else None
            active_status = statuses_by_id.get(active_source["id"]) if active_source else None
            active_health = normalize_source_runtime_status(active_status or {})
            if active_status and active_status.get("last_snapshot_path") and Path(active_status["last_snapshot_path"]).exists():
                st.image(active_status["last_snapshot_path"], width="stretch", caption=f"Источник обзора: {active_source['name']}")
            else:
                if active_source:
                    st.info(
                        "Обзорный snapshot пока недоступен. Проверьте, что worker запущен, источник активирован и камера уже отдала первый кадр."
                    )
                else:
                    st.info("Нет активного обзорного источника. Выберите и активируйте production-камеру в разделе подключения камер.")
            overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
            overview_col1.metric("Приоритетная зона", active_point)
            overview_col2.metric("Инцидентов за сегодня", summary["suspicious_today"])
            overview_col3.metric("Обнаружений за сегодня", summary["detections_today"])
            overview_col4.metric("Состояние источника", active_health["health_status"])
            if active_source:
                st.caption(
                    f"Обзорный источник: {active_source['name']} · Тип: {active_source['source_type']} · "
                    f"Локация: {active_source.get('location') or 'не указана'} · "
                    f"Соединение: {active_health['connection_status']}"
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
                st.caption("Риск по зонам появится после накопления инцидентов хотя бы по одной зоне.")

        with st.container(border=True):
            st.subheader("Operational readiness")
            if operational_summary["issues"]:
                for issue in operational_summary["issues"]:
                    st.warning(issue)
            else:
                st.success("Система готова к эксплуатационному сценарию: активные камеры online, критичных operational issues не найдено.")

    with right_col:
        with st.container(border=True):
            st.subheader("Очередь активных инцидентов")
            if incident_queue_rows:
                st.dataframe(pd.DataFrame(incident_queue_rows), width="stretch", hide_index=True)
            else:
                st.dataframe(
                    pd.DataFrame(columns=["ID", "Серьезность", "Инцидент", "Источник", "Зона", "Статус", "Owner", "Возраст, мин", "SLA", "Время"]),
                    width="stretch",
                    hide_index=True,
                )
                st.caption("Активных инцидентов сейчас нет. Очередь автоматически заполнится новыми или эскалированными кейсами.")
        with st.container(border=True):
            st.subheader("Нагрузка по ответственным")
            if operator_workload_rows:
                st.dataframe(pd.DataFrame(operator_workload_rows), width="stretch", hide_index=True)
            else:
                st.dataframe(
                    pd.DataFrame(columns=["Ответственный", "Активных кейсов", "Critical", "Overdue", "Последний кейс"]),
                    width="stretch",
                    hide_index=True,
                )
                st.caption("Блок появится, когда активные кейсы будут назначены операторам.")
        with st.container(border=True):
            st.subheader("Распределение по серьезности")
            if severity_distribution.empty:
                st.dataframe(pd.DataFrame(columns=["severity", "count"]), width="stretch", hide_index=True)
                st.caption("Распределение появится после регистрации первых инцидентов.")
            else:
                st.bar_chart(
                    severity_distribution.rename(columns={"severity": "Серьезность", "count": "Инцидентов"}).set_index("Серьезность")
                )
        with st.container(border=True):
            st.subheader("Состояние камер и эксплуатационные риски")
            if source_risk_rows:
                st.dataframe(pd.DataFrame(source_risk_rows), width="stretch", hide_index=True)
            else:
                st.dataframe(
                    pd.DataFrame(columns=["Источник", "Статус", "Соединение", "FPS", "Активных инцидентов", "Критических", "Reconnect", "Последняя ошибка"]),
                    width="stretch",
                    hide_index=True,
                )
                st.caption("После добавления камер и первого heartbeat worker здесь появятся эксплуатационные статусы.")
        with st.container(border=True):
            st.subheader("Топ инцидентов и сбои")
            top_events = build_top_event_types(events, limit=6)
            if top_events.empty:
                st.dataframe(pd.DataFrame(columns=["Тип инцидента", "Количество"]), width="stretch", hide_index=True)
                st.caption("Топ событий ещё не сформирован: журнал событий пока пуст.")
            else:
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
