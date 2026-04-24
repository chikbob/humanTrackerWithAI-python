"""Analytics screens for enterprise monitored-zone operations."""

from __future__ import annotations

import pandas as pd

from analytics.access import (
    build_access_point_distribution,
    build_daily_entries,
    build_kpi_summary,
    build_offline_source_summary,
    build_time_distribution,
    build_top_event_types,
)


def render_access_analytics(st, *, events: list[dict], worker_statuses: list[dict]):
    summary = build_kpi_summary(events, worker_statuses)
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Обнаружения людей", summary["detections_today"])
    top2.metric("Инциденты входа в зону", summary["entries_today"])
    top3.metric("Длительные присутствия", sum(1 for event in events if event.get("event_type") == "prolonged_presence_near_entry"))
    top4.metric("Offline-инциденты камер", sum(1 for event in events if event.get("event_type") == "stream_offline"))

    row1, row2 = st.columns(2, gap="large")
    with row1:
        with st.container(border=True):
            st.subheader("Обнаружения по часам")
            hourly = build_time_distribution(events)
            if hourly.empty:
                st.dataframe(pd.DataFrame(columns=["hour", "count"]), width="stretch", hide_index=True)
            else:
                st.line_chart(hourly.rename(columns={"hour": "Час", "count": "Событий"}).set_index("Час"))
        with st.container(border=True):
            st.subheader("Инциденты по зонам контроля")
            by_point = build_access_point_distribution(events)
            if by_point.empty:
                st.dataframe(pd.DataFrame(columns=["access_point", "count"]), width="stretch", hide_index=True)
            else:
                st.bar_chart(by_point.rename(columns={"access_point": "Зона контроля", "count": "Инцидентов"}).set_index("Зона контроля"))
    with row2:
        with st.container(border=True):
            st.subheader("Входы в контролируемые зоны по дням")
            daily = build_daily_entries(events, days=14)
            if daily.empty:
                st.dataframe(pd.DataFrame(columns=["date", "count"]), width="stretch", hide_index=True)
            else:
                st.bar_chart(daily.rename(columns={"date": "Дата", "count": "Входов"}).set_index("Дата"))
        with st.container(border=True):
            st.subheader("Топ типов инцидентов")
            top_events = build_top_event_types(events)
            if top_events.empty:
                st.dataframe(pd.DataFrame(columns=["event_type", "count"]), width="stretch", hide_index=True)
            else:
                st.bar_chart(top_events.rename(columns={"event_type": "Тип инцидента", "count": "Количество"}).set_index("Тип инцидента"))

    with st.container(border=True):
        st.subheader("Offline-инциденты по камерам")
        offline_df = build_offline_source_summary(events)
        if offline_df.empty:
            st.dataframe(pd.DataFrame(columns=["source_name", "offline_events"]), width="stretch", hide_index=True)
            st.caption("Offline-инциденты по камерам пока не зафиксированы.")
        else:
            st.dataframe(
                offline_df.rename(columns={"source_name": "Источник", "offline_events": "Offline-инцидентов"}),
                width="stretch",
                hide_index=True,
            )
