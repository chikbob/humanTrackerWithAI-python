"""Employee registry UI."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


STATUS_OPTIONS = ["active", "inactive", "on_leave", "blocked"]


def render_employees(
    st,
    *,
    employees: list[dict],
    create_employee_fn,
    update_employee_fn,
    update_employee_status_fn,
):
    col_table, col_side = st.columns([1.6, 1.0], gap="large")
    with col_table:
        with st.container(border=True):
            st.subheader("Список сотрудников")
            rows = [
                {
                    "ID": employee["id"],
                    "ФИО": employee["full_name"],
                    "Подразделение": employee.get("department") or "",
                    "Должность": employee.get("position") or "",
                    "Статус": employee.get("status") or "",
                    "Создан": datetime.fromtimestamp(employee["created_at"]).strftime("%Y-%m-%d %H:%M")
                    if employee.get("created_at")
                    else "",
                }
                for employee in employees
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if not employees:
                st.caption("Справочник сотрудников пуст. Для демонстрации можно добавить тестовую карточку через форму справа.")

    with col_side:
        with st.container(border=True):
            st.subheader("Карточка сотрудника")
            if employees:
                labels = {f"{employee['full_name']} [{employee['id']}]": employee for employee in employees}
                selected_label = st.selectbox("Выберите сотрудника", options=list(labels.keys()))
                selected = labels[selected_label]
                st.markdown(
                    f"""
                    <div style="padding:1rem;border:1px dashed rgba(148,163,184,.25);border-radius:16px;text-align:center;">
                        <div style="font-size:48px;">👤</div>
                        <div><strong>{selected['full_name']}</strong></div>
                        <div>{selected.get('position') or 'Должность не указана'}</div>
                        <div>{selected.get('department') or 'Подразделение не указано'}</div>
                        <div>Статус: {selected.get('status') or 'не задан'}</div>
                        <div style="margin-top:.5rem;font-size:.9rem;color:#8aa0b6;">Подготовлено к будущему модулю идентификации сотрудников.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("После создания записи здесь будет отображаться краткая карточка сотрудника.")

        with st.container(border=True):
            st.subheader("Добавление сотрудника")
            with st.form("employee_create_form", clear_on_submit=True):
                full_name = st.text_input("ФИО")
                department = st.text_input("Подразделение")
                position = st.text_input("Должность")
                status = st.selectbox("Статус", options=STATUS_OPTIONS, index=0)
                submitted = st.form_submit_button("Создать карточку")
            if submitted:
                if not full_name.strip():
                    st.error("Поле ФИО обязательно.")
                else:
                    create_employee_fn(
                        full_name=full_name,
                        department=department,
                        position=position,
                        status=status,
                    )
                    st.success("Сотрудник добавлен.")
                    st.rerun()

        with st.container(border=True):
            st.subheader("Редактирование и статус")
            if not employees:
                st.info("Нет данных для редактирования.")
            else:
                labels = {f"{employee['full_name']} [{employee['id']}]": employee for employee in employees}
                selected_label = st.selectbox("Сотрудник", options=list(labels.keys()), key="employee_edit_select")
                selected = labels[selected_label]
                default_status_index = STATUS_OPTIONS.index(selected["status"]) if selected["status"] in STATUS_OPTIONS else 0
                with st.form("employee_edit_form"):
                    edit_full_name = st.text_input("ФИО", value=selected["full_name"])
                    edit_department = st.text_input("Подразделение", value=selected.get("department") or "")
                    edit_position = st.text_input("Должность", value=selected.get("position") or "")
                    edit_status = st.selectbox("Статус", options=STATUS_OPTIONS, index=default_status_index)
                    save = st.form_submit_button("Сохранить изменения")
                if save:
                    update_employee_fn(
                        employee_id=selected["id"],
                        full_name=edit_full_name,
                        department=edit_department,
                        position=edit_position,
                        status=edit_status,
                    )
                    st.success("Карточка сотрудника обновлена.")
                    st.rerun()
                quick_status = st.selectbox("Быстрая деактивация/смена статуса", options=STATUS_OPTIONS, key="employee_quick_status")
                if st.button("Применить статус"):
                    update_employee_status_fn(employee_id=selected["id"], status=quick_status)
                    st.success("Статус сотрудника обновлен.")
                    st.rerun()
