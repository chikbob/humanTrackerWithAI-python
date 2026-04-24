"""Dashboard view for enterprise monitored-zone operations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from analytics.access import (
    build_kpi_summary,
    build_offline_source_summary,
    build_source_status_rows,
    build_time_distribution,
    build_top_event_types,
)


def render_dashboard(
    st,
    *,
    events: list[dict],
    worker_statuses: list[dict],
    video_sources: list[dict],
    access_points: list[dict],
    employees: list[dict],
):
    summary = build_kpi_summary(events, worker_statuses)
    statuses_by_id = {status["source_id"]: status for status in worker_statuses}
    active_sources = [source for source in video_sources if source.get("is_active")]
    active_source = active_sources[0] if active_sources else None
    active_status = statuses_by_id.get(active_source["id"]) if active_source else None
    active_point = access_points[0]["name"] if access_points else "не задана"

    system_status = "online" if summary["online_cameras"] > 0 else "offline"
    camera_status = "connected" if active_status and active_status.get("is_connected") else "disconnected"

    top1, top2, top3, top4, top5, top6 = st.columns(6)
    top1.metric("Статус системы", system_status)
    top2.metric("Статус приоритетной камеры", camera_status)
    top3.metric("Приоритетная зона контроля", active_point)
    top4.metric("Обнаружения за сегодня", summary["detections_today"])
    top5.metric("Инциденты входа в зону", summary["entries_today"])
    top6.metric("Тревожные инциденты", summary["suspicious_today"])

    second1, second2, second3, second4 = st.columns(4)
    second1.metric("Источники online", summary["online_cameras"])
    second2.metric("Всего событий за день", summary["total_events_today"])
    second3.metric("Сотрудников в системе", len(employees))
    second4.metric("Активных сотрудников", sum(1 for employee in employees if employee.get("status") == "active"))

    left_col, right_col = st.columns([1.55, 1.0], gap="large")
    with left_col:
        with st.container(border=True):
            st.subheader("Оперативная обстановка по контролируемой зоне")
            if active_status and active_status.get("last_snapshot_path") and Path(active_status["last_snapshot_path"]).exists():
                st.image(active_status["last_snapshot_path"], width="stretch", caption=f"Источник: {active_source['name']}")
            else:
                st.info("Для активной камеры пока нет актуального snapshot. Проверьте worker или источник видеопотока.")
            if active_source:
                st.caption(
                    f"Источник: {active_source['name']} · Тип: {active_source['source_type']} · "
                    f"Локация: {active_source.get('location') or 'не указана'}"
                )

        with st.container(border=True):
            st.subheader("Интенсивность инцидентов по часам")
            distribution = build_time_distribution(
                [event for event in events if datetime.fromtimestamp(event["timestamp"]).date() == datetime.now().date()]
            )
            if distribution.empty:
                st.dataframe(pd.DataFrame(columns=["Час", "Событий"]), width="stretch", hide_index=True)
                st.caption("За текущий день распределение инцидентов еще не сформировано.")
            else:
                chart_df = distribution.rename(columns={"hour": "Час", "count": "Событий"}).set_index("Час")
                st.bar_chart(chart_df)

        with st.container(border=True):
            st.subheader("Статус источников видео")
            st.dataframe(pd.DataFrame(build_source_status_rows(video_sources, worker_statuses)), width="stretch", hide_index=True)

    with right_col:
        with st.container(border=True):
            st.subheader("Последние зафиксированные инциденты")
            recent_rows = [
                {
                    "Время": datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S"),
                    "Инцидент": event.get("event_type"),
                    "Источник": event.get("source_name"),
                    "Зона": event.get("access_point_name") or "не задана",
                    "Контекст": event.get("employee_name") or "не установлен",
                    "Уверенность": round(event.get("confidence") or 0.0, 3),
                }
                for event in events[:10]
            ]
            st.dataframe(pd.DataFrame(recent_rows), width="stretch", hide_index=True)
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
