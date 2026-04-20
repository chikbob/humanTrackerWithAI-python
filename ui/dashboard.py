"""Dashboard view for enterprise entry-zone monitoring."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from analytics.access import build_kpi_summary, build_time_distribution


def render_dashboard(st, *, events: list[dict], worker_statuses: list[dict], video_sources: list[dict], access_points: list[dict]):
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
    top2.metric("Статус камеры", camera_status)
    top3.metric("Активная точка прохода", active_point)
    top4.metric("Обнаружения за сегодня", summary["detections_today"])
    top5.metric("Входы в зону прохода", summary["entries_today"])
    top6.metric("Подозрительные события", summary["suspicious_today"])

    left_col, right_col = st.columns([1.7, 1.0], gap="large")
    with left_col:
        with st.container(border=True):
            st.subheader("Онлайн-поток с камеры")
            if active_status and active_status.get("last_snapshot_path") and Path(active_status["last_snapshot_path"]).exists():
                st.image(active_status["last_snapshot_path"], use_container_width=True, caption=f"Источник: {active_source['name']}")
            else:
                st.info("Фоновый worker еще не сохранил актуальный кадр. Проверьте активный источник или дождитесь первого heartbeat.")
            if active_source:
                st.caption(
                    f"Источник: {active_source['name']} · Тип: {active_source['source_type']} · "
                    f"Локация: {active_source.get('location') or 'не указана'}"
                )

        with st.container(border=True):
            st.subheader("Краткая аналитика за день")
            distribution = build_time_distribution(
                [event for event in events if datetime.fromtimestamp(event["timestamp"]).date() == datetime.now().date()]
            )
            if distribution.empty:
                st.dataframe(pd.DataFrame(columns=["Час", "Событий"]), use_container_width=True, hide_index=True)
                st.caption("За текущий день распределение событий еще не сформировано.")
            else:
                chart_df = distribution.rename(columns={"hour": "Час", "count": "Событий"}).set_index("Час")
                st.bar_chart(chart_df)

    with right_col:
        with st.container(border=True):
            st.subheader("Последние события в реальном времени")
            recent_rows = [
                {
                    "Время": datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S"),
                    "Событие": event.get("event_type"),
                    "Источник": event.get("source_name"),
                    "Уверенность": round(event.get("confidence") or 0.0, 3),
                }
                for event in events[:10]
            ]
            st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, hide_index=True)
        with st.container(border=True):
            st.subheader("Статус источников")
            rows = []
            for source in video_sources:
                status = statuses_by_id.get(source["id"], {})
                rows.append(
                    {
                        "Источник": source["name"],
                        "Тип": source["source_type"],
                        "Статус": status.get("status", "idle"),
                        "Соединение": "online" if status.get("is_connected") else "offline",
                        "Heartbeat": _fmt_ts(status.get("last_heartbeat")),
                        "FPS": round(status.get("fps") or 0.0, 2),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _fmt_ts(timestamp_value):
    if not timestamp_value:
        return "—"
    return datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
