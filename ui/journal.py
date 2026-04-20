"""Event journal UI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def render_event_journal(st, *, events: list[dict]):
    st.subheader("Журнал событий проходной")
    if not events:
        st.dataframe(
            pd.DataFrame(columns=["Время", "Тип события", "Источник", "Точка доступа", "Сотрудник", "Уверенность"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Журнал пуст. После запуска фонового worker или demo-режима события появятся автоматически.")
        return

    df = pd.DataFrame(
        [
            {
                "event_id": event["event_id"],
                "Время": datetime.fromtimestamp(event["timestamp"]),
                "Тип события": event.get("event_type"),
                "Уровень": event.get("event_scope"),
                "Источник": event.get("source_name"),
                "Точка доступа": event.get("access_point_name") or "не задана",
                "Сотрудник": event.get("employee_name") or "не определен",
                "Уверенность": round(event.get("confidence") or 0.0, 3),
                "Описание": event.get("message") or "",
            }
            for event in events
        ]
    )
    min_date = df["Время"].min().date()
    max_date = df["Время"].max().date()
    date_from, date_to = st.date_input("Период", value=(min_date, max_date))
    filters = st.columns(4)
    with filters[0]:
        event_types = st.multiselect("Тип события", options=sorted(df["Тип события"].dropna().unique().tolist()))
    with filters[1]:
        sources = st.multiselect("Источник", options=sorted(df["Источник"].dropna().unique().tolist()))
    with filters[2]:
        scopes = st.multiselect("Уровень", options=sorted(df["Уровень"].dropna().unique().tolist()))
    with filters[3]:
        employees = st.multiselect("Сотрудник", options=sorted(df["Сотрудник"].dropna().unique().tolist()))

    filtered = df[(df["Время"].dt.date >= date_from) & (df["Время"].dt.date <= date_to)]
    if event_types:
        filtered = filtered[filtered["Тип события"].isin(event_types)]
    if sources:
        filtered = filtered[filtered["Источник"].isin(sources)]
    if scopes:
        filtered = filtered[filtered["Уровень"].isin(scopes)]
    if employees:
        filtered = filtered[filtered["Сотрудник"].isin(employees)]

    st.dataframe(filtered.sort_values("Время", ascending=False), use_container_width=True, hide_index=True)
    csv_data = filtered.copy()
    csv_data["Время"] = csv_data["Время"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.download_button(
        "Экспорт CSV",
        data=csv_data.to_csv(index=False).encode("utf-8-sig"),
        file_name="access_event_journal.csv",
        mime="text/csv",
    )

    event_options = {f"{row['event_id']} · {row['Тип события']}": row["event_id"] for _, row in filtered.head(50).iterrows()}
    if not event_options:
        st.caption("Для выбранных фильтров карточка события недоступна.")
        return
    selected_label = st.selectbox("Карточка события", options=list(event_options.keys()))
    selected_id = event_options[selected_label]
    selected_event = next(event for event in events if event["event_id"] == selected_id)
    detail_col, image_col = st.columns([1.2, 1.0], gap="large")
    with detail_col:
        with st.container(border=True):
            st.markdown("**Детали события**")
            st.write(f"Тип события: `{selected_event.get('event_type')}`")
            st.write(f"Источник: `{selected_event.get('source_name')}`")
            st.write(f"Точка доступа: `{selected_event.get('access_point_name') or 'не задана'}`")
            st.write(f"Сотрудник: `{selected_event.get('employee_name') or 'не определен'}`")
            st.write(f"Confidence: `{round(selected_event.get('confidence') or 0.0, 3)}`")
            st.write(f"Статус идентификации: `{selected_event.get('identification_status') or 'not_configured'}`")
            st.caption(selected_event.get("message") or "Описание отсутствует.")
    with image_col:
        with st.container(border=True):
            st.markdown("**Snapshot / thumbnail**")
            snapshot_path = selected_event.get("snapshot_path")
            if snapshot_path and Path(snapshot_path).exists():
                st.image(snapshot_path, use_container_width=True, caption="Последний доступный кадр выбранного источника")
            else:
                st.info("Для выбранного события отдельный snapshot пока не сохранен.")
