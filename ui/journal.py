"""Incident journal UI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from services.incidents import INCIDENT_STATUS_OPTIONS


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

SEVERITY_LABELS = {
    "low": "Низкая",
    "medium": "Средняя",
    "high": "Высокая",
    "critical": "Критическая",
}


def _format_identification_status(status: str) -> str:
    return IDENTIFICATION_STATUS_LABELS.get(status or "", status or "Не указан")


def render_event_journal(
    st,
    *,
    incidents: list[dict],
    employees: list[dict],
    link_event_to_employee_fn,
    update_incident_status_fn,
):
    st.subheader("Журнал инцидентов")
    if not incidents:
        st.dataframe(
            pd.DataFrame(
                columns=[
                    "Время",
                    "Тип инцидента",
                    "Источник",
                    "Зона",
                    "Серьезность",
                    "Статус",
                    "Контекст",
                    "Уверенность",
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption("Журнал пуст. После запуска worker и появления прикладных инцидентов записи появятся автоматически.")
        return

    df = pd.DataFrame(
        [
            {
                "incident_id": incident["id"],
                "event_id": incident["event_id"],
                "employee_id": incident.get("employee_id"),
                "Время": datetime.fromtimestamp(incident["started_at"]),
                "Тип инцидента": incident.get("incident_type") or "—",
                "Источник": incident.get("source_name") or incident.get("source_id") or "—",
                "Зона": incident.get("zone_name") or "не задана",
                "Серьезность": SEVERITY_LABELS.get(incident.get("severity"), incident.get("severity") or "—"),
                "Статус": INCIDENT_STATUS_OPTIONS.get(incident.get("status"), incident.get("status") or "—"),
                "Контекст": _format_identification_status(incident.get("identification_status") or "unlinked"),
                "Уверенность": round(incident.get("confidence") or 0.0, 3),
                "Комментарий": incident.get("operator_comment") or "",
            }
            for incident in incidents
        ]
    )
    min_date = df["Время"].min().date()
    max_date = df["Время"].max().date()
    date_from, date_to = st.date_input("Период", value=(min_date, max_date))
    filters = st.columns(5)
    with filters[0]:
        incident_types = st.multiselect("Тип инцидента", options=sorted(df["Тип инцидента"].dropna().unique().tolist()))
    with filters[1]:
        zones = st.multiselect("Зона", options=sorted(df["Зона"].dropna().unique().tolist()))
    with filters[2]:
        severities = st.multiselect("Серьезность", options=sorted(df["Серьезность"].dropna().unique().tolist()))
    with filters[3]:
        statuses = st.multiselect("Статус", options=sorted(df["Статус"].dropna().unique().tolist()))
    with filters[4]:
        contexts = st.multiselect("Контекст", options=sorted(df["Контекст"].dropna().unique().tolist()))

    filtered = df[(df["Время"].dt.date >= date_from) & (df["Время"].dt.date <= date_to)]
    if incident_types:
        filtered = filtered[filtered["Тип инцидента"].isin(incident_types)]
    if zones:
        filtered = filtered[filtered["Зона"].isin(zones)]
    if severities:
        filtered = filtered[filtered["Серьезность"].isin(severities)]
    if statuses:
        filtered = filtered[filtered["Статус"].isin(statuses)]
    if contexts:
        filtered = filtered[filtered["Контекст"].isin(contexts)]

    st.dataframe(filtered.sort_values("Время", ascending=False), width="stretch", hide_index=True)
    csv_data = filtered.copy()
    csv_data["Время"] = csv_data["Время"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.download_button(
        "Экспорт CSV",
        data=csv_data.to_csv(index=False).encode("utf-8-sig"),
        file_name="incident_journal.csv",
        mime="text/csv",
    )

    incident_options = {
        f"{row['incident_id']} · {row['Тип инцидента']} · {row['Зона']}": row["incident_id"]
        for _, row in filtered.head(100).iterrows()
    }
    if not incident_options:
        st.caption("Для выбранных фильтров карточка инцидента недоступна.")
        return
    selected_label = st.selectbox("Карточка инцидента", options=list(incident_options.keys()))
    selected_id = incident_options[selected_label]
    selected_incident = next(incident for incident in incidents if incident["id"] == selected_id)

    detail_col, image_col = st.columns([1.25, 0.95], gap="large")
    with detail_col:
        with st.container(border=True):
            st.markdown("**Детали инцидента**")
            st.write(f"Тип инцидента: `{selected_incident.get('incident_type')}`")
            st.write(f"Источник: `{selected_incident.get('source_name') or selected_incident.get('source_id') or '—'}`")
            st.write(f"Зона: `{selected_incident.get('zone_name') or 'не задана'}`")
            st.write(f"Серьезность: `{SEVERITY_LABELS.get(selected_incident.get('severity'), selected_incident.get('severity') or '—')}`")
            st.write(f"Статус: `{INCIDENT_STATUS_OPTIONS.get(selected_incident.get('status'), selected_incident.get('status') or '—')}`")
            st.write(f"Контекст: `{_format_identification_status(selected_incident.get('identification_status') or 'unlinked')}`")
            st.write(f"Confidence: `{round(selected_incident.get('confidence') or 0.0, 3)}`")
            st.write(f"Event ID: `{selected_incident.get('event_id') or '—'}`")
            st.caption(selected_incident.get("operator_comment") or "Комментарий оператора отсутствует.")

        with st.container(border=True):
            st.markdown("**Операторская обработка**")
            status = st.selectbox(
                "Статус инцидента",
                options=list(INCIDENT_STATUS_OPTIONS.keys()),
                index=list(INCIDENT_STATUS_OPTIONS.keys()).index(selected_incident.get("status", "new"))
                if selected_incident.get("status", "new") in INCIDENT_STATUS_OPTIONS
                else 0,
                format_func=lambda key: INCIDENT_STATUS_OPTIONS[key],
                key=f"incident_status_{selected_id}",
            )
            operator_comment = st.text_area(
                "Комментарий оператора",
                value=selected_incident.get("operator_comment") or "",
                key=f"incident_comment_{selected_id}",
                placeholder="Например: подтверждено по регламенту охраны, направлен запрос на проверку или признано ложным срабатыванием.",
            )
            if st.button("Сохранить статус инцидента", key=f"incident_status_submit_{selected_id}", width="stretch"):
                update_incident_status_fn(
                    incident_id=selected_id,
                    status=status,
                    operator_comment=operator_comment,
                )
                st.success("Статус инцидента обновлен.")
                st.rerun()

        with st.container(border=True):
            st.markdown("**Ручное уточнение контекста**")
            employee_options = {
                f"{employee.get('display_name') or employee.get('full_name')} [{employee.get('employee_number') or employee['id']}]": employee["id"]
                for employee in employees
            }
            if not employee_options:
                st.info("Справочник персонала пуст. Для уточнения контекста сначала добавьте карточки сотрудников.")
            else:
                selected_employee_label = st.selectbox(
                    "Сотрудник справочника",
                    options=list(employee_options.keys()),
                    key=f"incident_link_employee_{selected_id}",
                )
                link_status = st.selectbox(
                    "Статус идентификации",
                    options=[
                        "pending_operator_confirmation",
                        "linked_from_directory",
                        "linked_from_access_control",
                    ],
                    format_func=_format_identification_status,
                    key=f"incident_link_status_{selected_id}",
                )
                link_note = st.text_area(
                    "Комментарий к привязке",
                    key=f"incident_link_note_{selected_id}",
                    placeholder="Например: подтвержден оператором смены или сопоставлен с внешним контуром доступа.",
                )
                if st.button("Связать событие-основание с карточкой сотрудника", key=f"incident_link_submit_{selected_id}", width="stretch"):
                    try:
                        link_event_to_employee_fn(
                            event_id=selected_incident["event_id"],
                            employee_id=employee_options[selected_employee_label],
                            identification_status=link_status,
                            note=link_note,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Контекст инцидента обновлен через базовое событие.")
                        st.rerun()
    with image_col:
        with st.container(border=True):
            st.markdown("**Snapshot / evidence**")
            snapshot_path = selected_incident.get("snapshot_path")
            if snapshot_path and Path(snapshot_path).exists():
                st.image(snapshot_path, width="stretch", caption="Последний доступный кадр выбранного источника")
            else:
                st.info("Для выбранного инцидента отдельный snapshot пока не сохранен.")
