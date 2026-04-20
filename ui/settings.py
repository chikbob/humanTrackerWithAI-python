"""System settings UI."""

from __future__ import annotations

from config.app_config import DEFAULT_MODEL_NAME
from ui.sidebar import MODEL_OPTIONS


def render_system_settings(st, *, settings: dict, access_points: list[dict], set_system_setting_fn):
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
                    "confidence threshold",
                    min_value=0.10,
                    max_value=0.95,
                    value=float(settings.get("confidence_threshold", 0.45)),
                    step=0.05,
                )
                frame_skip = st.slider(
                    "frame skip",
                    min_value=0,
                    max_value=10,
                    value=int(settings.get("frame_skip", 1)),
                    step=1,
                )
                inference_size = st.selectbox(
                    "inference size",
                    options=[320, 416, 512, 640, 960],
                    index=[320, 416, 512, 640, 960].index(int(settings.get("inference_size", 512))),
                )
            with col2:
                event_cooldown = st.slider(
                    "event cooldown",
                    min_value=1,
                    max_value=60,
                    value=int(settings.get("event_cooldown", 5)),
                    step=1,
                )
                reconnect_interval = st.slider(
                    "reconnect interval",
                    min_value=1,
                    max_value=60,
                    value=int(settings.get("reconnect_interval", 5)),
                    step=1,
                )
                source_timeout = st.slider(
                    "source timeout",
                    min_value=5,
                    max_value=120,
                    value=int(settings.get("source_timeout", 15)),
                    step=5,
                )
            with col3:
                model_name = st.selectbox(
                    "active model",
                    options=MODEL_OPTIONS,
                    index=MODEL_OPTIONS.index(settings.get("model_name", DEFAULT_MODEL_NAME))
                    if settings.get("model_name", DEFAULT_MODEL_NAME) in MODEL_OPTIONS
                    else 1,
                )
                debug_mode = st.toggle("debug mode", value=str(settings.get("debug_mode", "0")) == "1")
                active_access_point = st.selectbox(
                    "active access point",
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
            set_system_setting_fn(key="model_name", value=model_name)
            set_system_setting_fn(key="debug_mode", value="1" if debug_mode else "0")
            if point_options:
                set_system_setting_fn(key="active_access_point_id", value=str(point_options[active_access_point]))
            st.success("Настройки сохранены.")
            st.rerun()
    st.caption(
        "Параметры применяются к фоновому worker и production-режиму. "
        "При изменении критичных настроек источников рекомендуется перезапустить worker."
    )
