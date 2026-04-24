"""Video source management UI."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


SOURCE_TYPES = {
    "rtsp": "RTSP/IP-камера",
    "stream_url": "HLS / HTTP-поток",
    "usb_camera": "USB / локальная камера на сервере",
    "browser_camera": "Браузерная камера",
}


def _render_source_processing_controls(st, *, prefix: str, source: dict | None = None):
    source = source or {}
    st.caption("Параметры обработки применяются worker-процессом для конкретного источника.")
    enable_roi = st.checkbox(
        "Ограничивать события зоной ROI",
        value=bool(source.get("enable_roi", True)),
        key=f"{prefix}_enable_roi",
    )
    roi_col1, roi_col2 = st.columns(2)
    with roi_col1:
        roi_x = st.slider("ROI X, %", min_value=0, max_value=95, value=int(source.get("roi_x", 20)), key=f"{prefix}_roi_x")
        roi_w_max = max(1, 100 - roi_x)
        roi_w = st.slider("ROI W, %", min_value=1, max_value=roi_w_max, value=min(int(source.get("roi_w", 60)), roi_w_max), key=f"{prefix}_roi_w")
    with roi_col2:
        roi_y = st.slider("ROI Y, %", min_value=0, max_value=95, value=int(source.get("roi_y", 20)), key=f"{prefix}_roi_y")
        roi_h_max = max(1, 100 - roi_y)
        roi_h = st.slider("ROI H, %", min_value=1, max_value=roi_h_max, value=min(int(source.get("roi_h", 60)), roi_h_max), key=f"{prefix}_roi_h")

    st.markdown("**Правила событий**")
    rule_count_enabled = st.checkbox(
        "Включить правило N/T для подсчета объектов",
        value=bool(source.get("rule_count_enabled", False)),
        key=f"{prefix}_rule_count_enabled",
    )
    rule_col1, rule_col2, rule_col3 = st.columns(3)
    with rule_col1:
        rule_n = st.number_input("N объектов", min_value=1, max_value=20, value=int(source.get("rule_n", 3)), step=1, key=f"{prefix}_rule_n")
    with rule_col2:
        rule_t = st.number_input("T секунд", min_value=1, max_value=300, value=int(source.get("rule_t", 10)), step=1, key=f"{prefix}_rule_t")
    with rule_col3:
        prolonged_presence_seconds = st.number_input(
            "Длительное присутствие, сек",
            min_value=1,
            max_value=3600,
            value=int(source.get("prolonged_presence_seconds", 10)),
            step=1,
            key=f"{prefix}_prolonged_presence_seconds",
        )
    rule_disappear_enabled = st.checkbox(
        "Фиксировать исчезновение трека",
        value=bool(source.get("rule_disappear_enabled", True)),
        key=f"{prefix}_rule_disappear_enabled",
    )
    rule_disappear_seconds = st.number_input(
        "Порог исчезновения, сек",
        min_value=1,
        max_value=120,
        value=int(source.get("rule_disappear_seconds", 5)),
        step=1,
        key=f"{prefix}_rule_disappear_seconds",
    )

    return {
        "enable_roi": enable_roi,
        "roi_x": roi_x,
        "roi_y": roi_y,
        "roi_w": roi_w,
        "roi_h": roi_h,
        "rule_count_enabled": rule_count_enabled,
        "rule_n": int(rule_n),
        "rule_t": int(rule_t),
        "rule_disappear_enabled": rule_disappear_enabled,
        "rule_disappear_seconds": int(rule_disappear_seconds),
        "prolonged_presence_seconds": int(prolonged_presence_seconds),
    }


def render_video_sources(
    st,
    *,
    video_sources: list[dict],
    worker_statuses: list[dict],
    create_video_source_fn,
    update_video_source_fn,
    set_video_source_active_fn,
    test_connection_fn,
):
    statuses_by_source = {status["source_id"]: status for status in worker_statuses}
    left_col, right_col = st.columns([1.4, 1.0], gap="large")
    with left_col:
        with st.container(border=True):
            st.subheader("Список подключенных источников")
            rows = []
            for source in video_sources:
                status = statuses_by_source.get(source["id"], {})
                rows.append(
                    {
                        "ID": source["id"],
                        "Наименование": source["name"],
                        "Тип": SOURCE_TYPES.get(source["source_type"], source["source_type"]),
                        "Локация": source.get("location") or "",
                        "ROI": "вкл" if source.get("enable_roi") else "выкл",
                        "Активен": "да" if source.get("is_active") else "нет",
                        "Статус": status.get("status", "idle"),
                        "Последний heartbeat": _fmt_ts(status.get("last_heartbeat")),
                        "Последний кадр": _fmt_ts(status.get("last_frame_at")),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with right_col:
        with st.container(border=True):
            st.subheader("Добавление источника")
            with st.form("create_video_source_form", clear_on_submit=True):
                name = st.text_input("Наименование")
                source_type = st.selectbox("Тип источника", options=list(SOURCE_TYPES.keys()), format_func=lambda key: SOURCE_TYPES[key])
                source_url_label = "URL / индекс устройства" if source_type != "browser_camera" else "Идентификатор источника"
                source_url_placeholder = "" if source_type != "browser_camera" else "browser_camera"
                source_url = st.text_input(source_url_label, value=source_url_placeholder)
                location = st.text_input("Локация")
                description = st.text_area("Описание")
                is_active = st.checkbox("Сделать активным сразу", value=True)
                processing_config = _render_source_processing_controls(st, prefix="create_source")
                submitted = st.form_submit_button("Сохранить источник")
            if submitted:
                normalized_source_url = source_url.strip() or ("browser_camera" if source_type == "browser_camera" else "")
                if not name.strip() or not normalized_source_url:
                    st.error("Необходимо указать название и значение источника.")
                else:
                    create_video_source_fn(
                        name=name,
                        source_type=source_type,
                        source_url=normalized_source_url,
                        location=location,
                        description=description,
                        is_active=is_active,
                        **processing_config,
                    )
                    st.success("Источник видеоданных добавлен.")
                    st.rerun()

        with st.container(border=True):
            st.subheader("Проверка подключения")
            if not video_sources:
                st.info("Сначала добавьте источник.")
            else:
                labels = {f"{source['name']} [{source['id']}]": source for source in video_sources}
                selected_label = st.selectbox("Источник для проверки", options=list(labels.keys()), key="source_check_select")
                selected = labels[selected_label]
                if st.button("Тест подключения"):
                    success, message = test_connection_fn(selected["source_type"], selected["source_url"])
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        with st.container(border=True):
            st.subheader("Активация и редактирование")
            if not video_sources:
                st.info("Нет источников для изменения.")
            else:
                labels = {f"{source['name']} [{source['id']}]": source for source in video_sources}
                selected_label = st.selectbox("Источник", options=list(labels.keys()), key="source_edit_select")
                selected = labels[selected_label]
                with st.form("edit_video_source_form"):
                    name = st.text_input("Наименование", value=selected["name"])
                    source_type = st.selectbox(
                        "Тип источника",
                        options=list(SOURCE_TYPES.keys()),
                        index=list(SOURCE_TYPES.keys()).index(selected["source_type"]),
                        format_func=lambda key: SOURCE_TYPES[key],
                    )
                    source_url_label = "URL / индекс устройства" if source_type != "browser_camera" else "Идентификатор источника"
                    source_url = st.text_input(source_url_label, value=selected["source_url"])
                    location = st.text_input("Локация", value=selected.get("location") or "")
                    description = st.text_area("Описание", value=selected.get("description") or "")
                    processing_config = _render_source_processing_controls(st, prefix=f"edit_source_{selected['id']}", source=selected)
                    save = st.form_submit_button("Сохранить изменения")
                if save:
                    update_video_source_fn(
                        source_id=selected["id"],
                        name=name,
                        source_type=source_type,
                        source_url=source_url,
                        location=location,
                        description=description,
                        **processing_config,
                    )
                    st.success("Источник обновлен.")
                    st.rerun()
                toggle_col1, toggle_col2 = st.columns(2)
                with toggle_col1:
                    if st.button("Активировать", key=f"activate_source_{selected['id']}"):
                        set_video_source_active_fn(source_id=selected["id"], is_active=True)
                        st.success("Источник активирован.")
                        st.rerun()
                with toggle_col2:
                    if st.button("Деактивировать", key=f"deactivate_source_{selected['id']}"):
                        set_video_source_active_fn(source_id=selected["id"], is_active=False)
                        st.warning("Источник деактивирован.")
                        st.rerun()


def _fmt_ts(timestamp_value):
    if not timestamp_value:
        return "—"
    return datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
