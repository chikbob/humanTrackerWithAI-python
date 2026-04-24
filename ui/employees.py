"""Employee registry UI."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from services.employee_repository import format_employee_sync_status
from services.employee_sync import employee_directory_summary


STATUS_OPTIONS = ["active", "inactive", "on_leave", "blocked"]
STATUS_LABELS = {
    "active": "Активен",
    "inactive": "Неактивен",
    "on_leave": "В отпуске",
    "blocked": "Заблокирован",
}


def _format_employee_status(status: str) -> str:
    return STATUS_LABELS.get(status or "", status or "Не задан")


def _to_date(timestamp_value):
    if not timestamp_value:
        return None
    return datetime.fromtimestamp(float(timestamp_value)).date()


def render_employees(
    st,
    *,
    employees: list[dict],
    sync_state: dict | None,
    employee_data_source: str,
    employee_directory_read_only: bool,
    access_context: dict,
    sync_employee_directory_fn,
    create_employee_fn,
    update_employee_fn,
    update_employee_status_fn,
):
    can_manage = access_context.get("role") == "admin"
    directory_summary = employee_directory_summary(sync_state)
    sync_cols = st.columns([1.0, 1.15, 1.15, 1.0])
    sync_cols[0].metric("Источник данных", employee_data_source)
    sync_cols[1].metric("Синхронизация", format_employee_sync_status(sync_state))
    sync_cols[2].metric(
        "Последнее обновление",
        datetime.fromtimestamp(directory_summary["last_synced_at"]).strftime("%Y-%m-%d %H:%M")
        if directory_summary.get("last_synced_at")
        else "—",
    )
    sync_cols[3].metric("Режим доступа", "read-only" if employee_directory_read_only else "read-write")
    if directory_summary.get("last_error"):
        st.warning(f"Ошибка синхронизации: {directory_summary['last_error']}")

    action_cols = st.columns([0.95, 2.05])
    with action_cols[0]:
        if st.button("Синхронизировать справочник", width="stretch"):
            if not can_manage:
                st.error("Недостаточно прав для синхронизации справочника.")
            else:
                sync_result = sync_employee_directory_fn()
                if sync_result.get("last_error"):
                    st.error(f"Синхронизация завершилась с ошибкой: {sync_result['last_error']}")
                else:
                    st.success("Справочник сотрудников обновлен.")
            st.rerun()
    with action_cols[1]:
        st.caption(
            "Справочник сотрудников поддерживает локальный режим и внешние источники. "
            "При недоступности удаленной БД применяется локальный кэш без потери отображения карточек."
        )

    col_table, col_side = st.columns([1.65, 1.0], gap="large")
    with col_table:
        with st.container(border=True):
            st.subheader("Список сотрудников предприятия")
            rows = [
                {
                    "ID": employee["id"],
                    "Табельный номер": employee.get("employee_number") or "—",
                    "ФИО": employee.get("display_name") or employee.get("full_name") or "—",
                    "Подразделение": employee.get("department") or "—",
                    "Должность": employee.get("position") or "—",
                    "Статус": _format_employee_status(employee.get("status") or ""),
                    "Дата приема": datetime.fromtimestamp(employee["hire_date"]).strftime("%Y-%m-%d")
                    if employee.get("hire_date")
                    else "—",
                    "Источник": employee.get("source_system") or "local",
                }
                for employee in employees
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if not employees:
                st.info("Справочник сотрудников пуст. Добавьте карточку вручную или выполните синхронизацию/seed данных.")

    with col_side:
        with st.container(border=True):
            st.subheader("Карточка сотрудника")
            if employees:
                labels = {
                    f"{employee.get('display_name') or employee['full_name']} [{employee.get('employee_number') or employee['id']}]": employee
                    for employee in employees
                }
                selected_label = st.selectbox("Выберите сотрудника", options=list(labels.keys()))
                selected = labels[selected_label]
                hire_date = (
                    datetime.fromtimestamp(selected["hire_date"]).strftime("%d.%m.%Y")
                    if selected.get("hire_date")
                    else "Не указана"
                )
                st.markdown(
                    f"""
                    <div style="padding:1rem;border:1px solid rgba(148,163,184,.18);border-radius:18px;background:rgba(15,23,42,.55);">
                        <div style="display:flex;gap:1rem;align-items:center;">
                            <div style="width:72px;height:72px;border-radius:18px;background:linear-gradient(135deg,#0f172a,#1e293b);display:flex;align-items:center;justify-content:center;font-size:34px;">👤</div>
                            <div>
                                <div style="font-size:1.05rem;font-weight:700;">{selected.get('display_name') or selected.get('full_name')}</div>
                                <div style="color:#94a3b8;">{selected.get('position') or 'Должность не указана'}</div>
                                <div style="color:#94a3b8;">{selected.get('department') or 'Подразделение не указано'}</div>
                            </div>
                        </div>
                        <div style="margin-top:1rem;display:grid;grid-template-columns:1fr 1fr;gap:.4rem 1rem;font-size:.92rem;">
                            <div><strong>Табельный номер:</strong> {selected.get('employee_number') or '—'}</div>
                            <div><strong>Статус:</strong> {_format_employee_status(selected.get('status') or '')}</div>
                            <div><strong>Дата приема:</strong> {hire_date}</div>
                            <div><strong>Источник:</strong> {selected.get('source_system') or 'local'}</div>
                        </div>
                        <div style="margin-top:.8rem;color:#94a3b8;font-size:.88rem;">Фото профиля используется как справочная карточка сотрудника и не применяется для автоматической биометрической идентификации.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("После создания записи здесь появится карточка сотрудника.")

        with st.container(border=True):
            st.subheader("Добавление сотрудника")
            if employee_directory_read_only:
                st.info("Справочник доступен только для чтения. Добавление сотрудников выполняется во внешней системе.")
            elif not can_manage:
                st.info("Добавление сотрудников доступно только администратору.")
            else:
                with st.form("employee_create_form", clear_on_submit=True):
                    create_cols = st.columns(3)
                    last_name = create_cols[0].text_input("Фамилия")
                    first_name = create_cols[1].text_input("Имя")
                    middle_name = create_cols[2].text_input("Отчество")
                    meta_cols = st.columns(2)
                    employee_number = meta_cols[0].text_input("Табельный номер")
                    hire_date_value = meta_cols[1].date_input("Дата приема", value=date.today())
                    department = st.text_input("Подразделение")
                    position = st.text_input("Должность")
                    status = st.selectbox("Статус", options=STATUS_OPTIONS, format_func=_format_employee_status, index=0)
                    profile_photo_url = st.text_input("URL фото профиля", placeholder="https://...")
                    submitted = st.form_submit_button("Создать карточку")
                if submitted:
                    if not last_name.strip() or not first_name.strip():
                        st.error("Для карточки сотрудника обязательны фамилия и имя.")
                    else:
                        create_employee_fn(
                            full_name=f"{last_name} {first_name} {middle_name}".strip(),
                            last_name=last_name,
                            first_name=first_name,
                            middle_name=middle_name,
                            employee_number=employee_number,
                            department=department,
                            position=position,
                            status=status,
                            hire_date=datetime.combine(hire_date_value, datetime.min.time()).timestamp(),
                            profile_photo_url=profile_photo_url,
                        )
                        st.success("Карточка сотрудника добавлена.")
                        st.rerun()

        with st.container(border=True):
            st.subheader("Редактирование и статус")
            if not employees:
                st.info("Нет сотрудников для редактирования.")
            elif employee_directory_read_only:
                st.info("Справочник синхронизируется из внешнего источника. Локальное редактирование отключено.")
            elif not can_manage:
                st.info("Редактирование сотрудников доступно только администратору.")
            else:
                labels = {
                    f"{employee.get('display_name') or employee['full_name']} [{employee.get('employee_number') or employee['id']}]": employee
                    for employee in employees
                }
                selected_label = st.selectbox("Сотрудник", options=list(labels.keys()), key="employee_edit_select")
                selected = labels[selected_label]
                default_status_index = STATUS_OPTIONS.index(selected["status"]) if selected["status"] in STATUS_OPTIONS else 0
                with st.form("employee_edit_form"):
                    edit_cols = st.columns(3)
                    edit_last_name = edit_cols[0].text_input("Фамилия", value=selected.get("last_name") or "")
                    edit_first_name = edit_cols[1].text_input("Имя", value=selected.get("first_name") or "")
                    edit_middle_name = edit_cols[2].text_input("Отчество", value=selected.get("middle_name") or "")
                    edit_meta_cols = st.columns(2)
                    edit_employee_number = edit_meta_cols[0].text_input("Табельный номер", value=selected.get("employee_number") or "")
                    edit_hire_date = edit_meta_cols[1].date_input(
                        "Дата приема",
                        value=_to_date(selected.get("hire_date")) or date.today(),
                    )
                    edit_department = st.text_input("Подразделение", value=selected.get("department") or "")
                    edit_position = st.text_input("Должность", value=selected.get("position") or "")
                    edit_status = st.selectbox(
                        "Статус",
                        options=STATUS_OPTIONS,
                        format_func=_format_employee_status,
                        index=default_status_index,
                    )
                    edit_photo_url = st.text_input("URL фото профиля", value=selected.get("profile_photo_url") or "")
                    save = st.form_submit_button("Сохранить изменения")
                if save:
                    if not edit_last_name.strip() or not edit_first_name.strip():
                        st.error("Фамилия и имя обязательны.")
                    else:
                        update_employee_fn(
                            employee_id=selected["id"],
                            full_name=f"{edit_last_name} {edit_first_name} {edit_middle_name}".strip(),
                            last_name=edit_last_name,
                            first_name=edit_first_name,
                            middle_name=edit_middle_name,
                            employee_number=edit_employee_number,
                            department=edit_department,
                            position=edit_position,
                            status=edit_status,
                            hire_date=datetime.combine(edit_hire_date, datetime.min.time()).timestamp(),
                            profile_photo_url=edit_photo_url,
                        )
                        st.success("Карточка сотрудника обновлена.")
                        st.rerun()
                quick_status = st.selectbox(
                    "Быстрая смена статуса",
                    options=STATUS_OPTIONS,
                    format_func=_format_employee_status,
                    key="employee_quick_status",
                )
                if st.button("Применить статус"):
                    update_employee_status_fn(employee_id=selected["id"], status=quick_status)
                    st.success("Статус сотрудника обновлен.")
                    st.rerun()
