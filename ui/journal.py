"""Event journal UI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


IDENTIFICATION_STATUS_LABELS = {
    "unknown": "Не установлен",
    "unlinked": "Не связан",
    "pending_operator_confirmation": "Ожидает подтверждения оператора",
    "linked_from_directory": "Связан со справочником",
    "linked_from_access_control": "Связан со СКУД/внешним контуром",
    "inactive_employee": "Неактивный сотрудник",
    "db_unavailable": "Внешний справочник недоступен",
    "no_reference_data": "Нет опорных данных",
    "not_enough_reference_data": "Недостаточно опорных данных",
    "low_confidence": "Низкая достоверность",
}


def _format_identification_status(status: str) -> str:
    return IDENTIFICATION_STATUS_LABELS.get(status or "", status or "Не указан")


def render_event_journal(
    st,
    *,
    events: list[dict],
    employees: list[dict],
    link_event_to_employee_fn,
):
    st.subheader("Журнал событий проходной")
    if not events:
        st.dataframe(
            pd.DataFrame(
                columns=[
                    "Время",
                    "Тип события",
                    "Источник",
                    "Точка доступа",
                    "Сотрудник",
                    "Статус связи",
                    "Уверенность",
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption("Журнал пуст. После запуска worker или активного источника записи появятся автоматически.")
        return

    df = pd.DataFrame(
        [
            {
                "event_id": event["event_id"],
                "employee_id": event.get("employee_id") or event.get("identified_employee_id"),
                "Время": datetime.fromtimestamp(event["timestamp"]),
                "Тип события": event.get("event_type") or "—",
                "Уровень": event.get("event_scope") or "raw",
                "Источник": event.get("source_name") or "—",
                "Точка доступа": event.get("access_point_name") or "не задана",
                "Сотрудник": event.get("employee_name") or "не установлен",
                "Табельный номер": event.get("employee_number") or "—",
                "Статус связи": _format_identification_status(event.get("identification_status") or "unlinked"),
                "Уверенность": round(event.get("confidence") or 0.0, 3),
                "Описание": event.get("message") or "",
            }
            for event in events
        ]
    )
    min_date = df["Время"].min().date()
    max_date = df["Время"].max().date()
    date_from, date_to = st.date_input("Период", value=(min_date, max_date))
    filters = st.columns(5)
    with filters[0]:
        event_types = st.multiselect("Тип события", options=sorted(df["Тип события"].dropna().unique().tolist()))
    with filters[1]:
        sources = st.multiselect("Источник", options=sorted(df["Источник"].dropna().unique().tolist()))
    with filters[2]:
        scopes = st.multiselect("Уровень", options=sorted(df["Уровень"].dropna().unique().tolist()))
    with filters[3]:
        employees_filter = st.multiselect("Сотрудник", options=sorted(df["Сотрудник"].dropna().unique().tolist()))
    with filters[4]:
        statuses = st.multiselect("Статус связи", options=sorted(df["Статус связи"].dropna().unique().tolist()))

    filtered = df[(df["Время"].dt.date >= date_from) & (df["Время"].dt.date <= date_to)]
    if event_types:
        filtered = filtered[filtered["Тип события"].isin(event_types)]
    if sources:
        filtered = filtered[filtered["Источник"].isin(sources)]
    if scopes:
        filtered = filtered[filtered["Уровень"].isin(scopes)]
    if employees_filter:
        filtered = filtered[filtered["Сотрудник"].isin(employees_filter)]
    if statuses:
        filtered = filtered[filtered["Статус связи"].isin(statuses)]

    st.dataframe(
        filtered.sort_values("Время", ascending=False),
        width="stretch",
        hide_index=True,
    )
    csv_data = filtered.copy()
    csv_data["Время"] = csv_data["Время"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.download_button(
        "Экспорт CSV",
        data=csv_data.to_csv(index=False).encode("utf-8-sig"),
        file_name="access_event_journal.csv",
        mime="text/csv",
    )

    event_options = {f"{row['event_id']} · {row['Тип события']} · {row['Источник']}": row["event_id"] for _, row in filtered.head(100).iterrows()}
    if not event_options:
        st.caption("Для выбранных фильтров карточка события недоступна.")
        return
    selected_label = st.selectbox("Карточка события", options=list(event_options.keys()))
    selected_id = event_options[selected_label]
    selected_event = next(event for event in events if event["event_id"] == selected_id)
    detail_col, image_col = st.columns([1.25, 0.95], gap="large")
    with detail_col:
        with st.container(border=True):
            st.markdown("**Детали события**")
            st.write(f"Тип события: `{selected_event.get('event_type')}`")
            st.write(f"Источник: `{selected_event.get('source_name') or '—'}`")
            st.write(f"Точка доступа: `{selected_event.get('access_point_name') or 'не задана'}`")
            st.write(
                "Сотрудник: "
                f"`{selected_event.get('employee_name') or 'не установлен'}`"
                + (
                    f" · таб. № `{selected_event.get('employee_number')}`"
                    if selected_event.get("employee_number")
                    else ""
                )
            )
            st.write(f"Confidence: `{round(selected_event.get('confidence') or 0.0, 3)}`")
            st.write(f"Статус связи: `{_format_identification_status(selected_event.get('identification_status') or 'unlinked')}`")
            st.write(f"Track/session: `{selected_event.get('track_id') or '—'}` / `{selected_event.get('session_id') or '—'}`")
            st.caption(selected_event.get("message") or "Описание отсутствует.")
        with st.container(border=True):
            st.markdown("**Ручная привязка к сотруднику**")
            employee_options = {
                f"{employee.get('display_name') or employee.get('full_name')} [{employee.get('employee_number') or employee['id']}]": employee["id"]
                for employee in employees
            }
            if not employee_options:
                st.info("Справочник сотрудников пуст. Для ручной привязки сначала добавьте карточки сотрудников.")
            else:
                selected_employee_label = st.selectbox(
                    "Сотрудник справочника",
                    options=list(employee_options.keys()),
                    key=f"event_link_employee_{selected_id}",
                )
                link_status = st.selectbox(
                    "Статус связи",
                    options=[
                        "pending_operator_confirmation",
                        "linked_from_directory",
                        "linked_from_access_control",
                    ],
                    format_func=_format_identification_status,
                    key=f"event_link_status_{selected_id}",
                )
                link_note = st.text_area(
                    "Комментарий оператора",
                    key=f"event_link_note_{selected_id}",
                    placeholder="Например: подтвержден по пропуску охраны или по журналу СКУД.",
                )
                if st.button("Связать событие с сотрудником", key=f"event_link_submit_{selected_id}", width="stretch"):
                    try:
                        link_event_to_employee_fn(
                            event_id=selected_id,
                            employee_id=employee_options[selected_employee_label],
                            identification_status=link_status,
                            note=link_note,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Событие связано с карточкой сотрудника.")
                        st.rerun()
    with image_col:
        with st.container(border=True):
            st.markdown("**Snapshot / thumbnail**")
            snapshot_path = selected_event.get("snapshot_path")
            if snapshot_path and Path(snapshot_path).exists():
                st.image(snapshot_path, width="stretch", caption="Последний доступный кадр выбранного источника")
            else:
                st.info("Для выбранного события отдельный snapshot пока не сохранен.")
