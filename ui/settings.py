"""System settings UI."""

from __future__ import annotations

from config.app_config import (
    DEFAULT_IDENTITY_BACKEND,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRACKER_TYPE,
    IDENTITY_BACKEND_OPTIONS,
    TRACKER_OPTIONS,
)
from ui.sidebar import MODEL_OPTIONS


def render_system_settings(
    st,
    *,
    settings: dict,
    access_points: list[dict],
    set_system_setting_fn,
    reset_and_seed_demo_data_fn,
):
    st.subheader("Настройки системы")
    point_options = {point["name"]: point["id"] for point in access_points}
    default_point_name = next(
        (point["name"] for point in access_points if str(point["id"]) == str(settings.get("active_access_point_id", ""))),
        access_points[0]["name"] if access_points else None,
    )
    with st.container(border=True):
        with st.form("system_settings_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                confidence_threshold = st.slider(
                    "Порог confidence",
                    min_value=0.10,
                    max_value=0.95,
                    value=float(settings.get("confidence_threshold", 0.45)),
                    step=0.05,
                )
                frame_skip = st.slider(
                    "Пропуск кадров",
                    min_value=0,
                    max_value=10,
                    value=int(settings.get("frame_skip", 1)),
                    step=1,
                )
                inference_size = st.selectbox(
                    "Размер кадра для инференса",
                    options=[320, 416, 512, 640, 960],
                    index=[320, 416, 512, 640, 960].index(int(settings.get("inference_size", 512))),
                )
            with col2:
                event_cooldown = st.slider(
                    "Интервал подавления дубликатов",
                    min_value=1,
                    max_value=60,
                    value=int(settings.get("event_cooldown", 5)),
                    step=1,
                )
                reconnect_interval = st.slider(
                    "Интервал переподключения",
                    min_value=1,
                    max_value=60,
                    value=int(settings.get("reconnect_interval", 5)),
                    step=1,
                )
                source_timeout = st.slider(
                    "Таймаут источника",
                    min_value=5,
                    max_value=120,
                    value=int(settings.get("source_timeout", 15)),
                    step=5,
                )
                employee_sync_interval = st.slider(
                    "Интервал синхронизации сотрудников, сек",
                    min_value=0,
                    max_value=3600,
                    value=int(settings.get("employee_sync_interval", 300)),
                    step=30,
                    help="0 отключает автоматическую синхронизацию справочника сотрудников.",
                )
            with col3:
                model_name = st.selectbox(
                    "Активная модель",
                    options=MODEL_OPTIONS,
                    index=MODEL_OPTIONS.index(settings.get("model_name", DEFAULT_MODEL_NAME))
                    if settings.get("model_name", DEFAULT_MODEL_NAME) in MODEL_OPTIONS
                    else 1,
                )
                tracker_type = st.selectbox(
                    "Трекинг-стратегия",
                    options=list(TRACKER_OPTIONS.keys()),
                    index=list(TRACKER_OPTIONS.keys()).index(settings.get("tracker_type", DEFAULT_TRACKER_TYPE))
                    if settings.get("tracker_type", DEFAULT_TRACKER_TYPE) in TRACKER_OPTIONS
                    else 0,
                    format_func=lambda key: TRACKER_OPTIONS[key]["label"],
                )
                identity_backend = st.selectbox(
                    "Identity backend",
                    options=list(IDENTITY_BACKEND_OPTIONS.keys()),
                    index=list(IDENTITY_BACKEND_OPTIONS.keys()).index(settings.get("identity_backend", DEFAULT_IDENTITY_BACKEND))
                    if settings.get("identity_backend", DEFAULT_IDENTITY_BACKEND) in IDENTITY_BACKEND_OPTIONS
                    else 0,
                    format_func=lambda key: IDENTITY_BACKEND_OPTIONS[key]["label"],
                )
                debug_mode = st.toggle("Режим отладки", value=str(settings.get("debug_mode", "0")) == "1")
                active_access_point = st.selectbox(
                    "Активная точка доступа",
                    options=list(point_options.keys()) if point_options else ["не задана"],
                    index=list(point_options.keys()).index(default_point_name) if default_point_name in point_options else 0,
                )
            submitted = st.form_submit_button("Сохранить настройки")
        if submitted:
            set_system_setting_fn(key="confidence_threshold", value=str(confidence_threshold))
            set_system_setting_fn(key="frame_skip", value=str(frame_skip))
            set_system_setting_fn(key="inference_size", value=str(inference_size))
            set_system_setting_fn(key="event_cooldown", value=str(event_cooldown))
            set_system_setting_fn(key="reconnect_interval", value=str(reconnect_interval))
            set_system_setting_fn(key="source_timeout", value=str(source_timeout))
            set_system_setting_fn(key="employee_sync_interval", value=str(employee_sync_interval))
            set_system_setting_fn(key="model_name", value=model_name)
            set_system_setting_fn(key="tracker_type", value=tracker_type)
            set_system_setting_fn(key="identity_backend", value=identity_backend)
            set_system_setting_fn(key="debug_mode", value="1" if debug_mode else "0")
            if point_options:
                set_system_setting_fn(key="active_access_point_id", value=str(point_options[active_access_point]))
            st.success("Настройки сохранены.")
            st.rerun()
    st.caption(
        "Параметры применяются к фоновому worker и production-режиму. "
        "При изменении критичных настроек источников рекомендуется перезапустить worker."
    )
    with st.container(border=True):
        st.subheader("Сервисные операции с БД")
        st.caption(
            "Полная очистка базы предназначена для демонстрационного или production-bootstrap сценария. "
            "Операция удаляет текущие записи и создает новый массив предметных данных."
        )
        seed_col1, seed_col2, seed_col3 = st.columns(3)
        with seed_col1:
            employee_count = st.number_input("Сотрудников", min_value=20, max_value=500, value=120, step=10)
        with seed_col2:
            visit_count = st.number_input("Цепочек проходов", min_value=100, max_value=5000, value=900, step=100)
        with seed_col3:
            random_seed = st.number_input("Seed", min_value=1, max_value=99999, value=42, step=1)
        confirm_reset = st.checkbox("Подтверждаю полную очистку и пересоздание данных")
        if st.button("Очистить БД и заполнить демонстрационными данными", type="primary"):
            if not confirm_reset:
                st.error("Подтвердите операцию очистки базы.")
            else:
                result = reset_and_seed_demo_data_fn(
                    employee_count=int(employee_count),
                    visit_count=int(visit_count),
                    seed=int(random_seed),
                )
                st.success(
                    "База данных пересоздана: "
                    f"{result['employees']} сотрудников, {result['video_sources']} источника, "
                    f"{result['visits']} цепочек проходов."
                )
                st.rerun()
