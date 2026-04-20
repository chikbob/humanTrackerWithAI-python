"""Video source management UI."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


SOURCE_TYPES = {
    "rtsp": "RTSP/IP camera",
    "stream_url": "HLS / HTTP stream",
    "usb_camera": "USB / локальная камера на сервере",
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
                        "Тип": source["source_type"],
                        "Локация": source.get("location") or "",
                        "Активен": "да" if source.get("is_active") else "нет",
                        "Статус": status.get("status", "idle"),
                        "Последний heartbeat": _fmt_ts(status.get("last_heartbeat")),
                        "Последний кадр": _fmt_ts(status.get("last_frame_at")),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with right_col:
        with st.container(border=True):
            st.subheader("Добавление источника")
            with st.form("create_video_source_form", clear_on_submit=True):
                name = st.text_input("Наименование")
                source_type = st.selectbox("Тип источника", options=list(SOURCE_TYPES.keys()), format_func=lambda key: SOURCE_TYPES[key])
                source_url = st.text_input("URL / индекс устройства")
                location = st.text_input("Локация")
                description = st.text_area("Описание")
                is_active = st.checkbox("Сделать активным сразу", value=True)
                submitted = st.form_submit_button("Сохранить источник")
            if submitted:
                if not name.strip() or not source_url.strip():
                    st.error("Необходимо указать название и URL/индекс устройства.")
                else:
                    create_video_source_fn(
                        name=name,
                        source_type=source_type,
                        source_url=source_url,
                        location=location,
                        description=description,
                        is_active=is_active,
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
                    source_url = st.text_input("URL / индекс устройства", value=selected["source_url"])
                    location = st.text_input("Локация", value=selected.get("location") or "")
                    description = st.text_area("Описание", value=selected.get("description") or "")
                    save = st.form_submit_button("Сохранить изменения")
                if save:
                    update_video_source_fn(
                        source_id=selected["id"],
                        name=name,
                        source_type=source_type,
                        source_url=source_url,
                        location=location,
                        description=description,
                    )
                    st.success("Источник обновлен.")
                    st.rerun()
                toggle_col1, toggle_col2 = st.columns(2)
                with toggle_col1:
                    if st.button("Activate", key=f"activate_source_{selected['id']}"):
                        set_video_source_active_fn(source_id=selected["id"], is_active=True)
                        st.success("Источник активирован.")
                        st.rerun()
                with toggle_col2:
                    if st.button("Deactivate", key=f"deactivate_source_{selected['id']}"):
                        set_video_source_active_fn(source_id=selected["id"], is_active=False)
                        st.warning("Источник деактивирован.")
                        st.rerun()


def _fmt_ts(timestamp_value):
    if not timestamp_value:
        return "—"
    return datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
