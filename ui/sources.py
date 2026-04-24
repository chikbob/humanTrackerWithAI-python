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


def _render_source_processing_defaults():
    return {
        "enable_roi": True,
        "roi_x": 20,
        "roi_y": 20,
        "roi_w": 60,
        "roi_h": 60,
        "rule_count_enabled": False,
        "rule_n": 3,
        "rule_t": 10,
        "rule_disappear_enabled": True,
        "rule_disappear_seconds": 5,
        "prolonged_presence_seconds": 10,
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
    source_labels = {f"{source['name']} [{source['id']}]": source for source in video_sources}
    active_ids = [source["id"] for source in video_sources if source.get("is_active")]

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Всего источников", len(video_sources))
    summary_col2.metric("Активных источников", len(active_ids))
    summary_col3.metric("Online / live", sum(1 for status in worker_statuses if status.get("is_connected")))

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

    tab_add, tab_manage, tab_check, tab_iphone = st.tabs(
        ["Добавить источник", "Управление и активация", "Проверка подключения", "iPhone / мобильная камера"]
    )

    with tab_add:
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

    with tab_manage:
        if not video_sources:
            st.info("Нет источников для изменения.")
        else:
            selected_active_ids = st.multiselect(
                "Активные источники",
                options=[source["id"] for source in video_sources],
                default=active_ids,
                format_func=lambda source_id: next(
                    f"{source['name']} [{SOURCE_TYPES.get(source['source_type'], source['source_type'])}]"
                    for source in video_sources
                    if source["id"] == source_id
                ),
                help="Здесь можно оставить активными сразу несколько камер. Сохранение применяет состояние ко всему списку.",
            )
            if st.button("Сохранить набор активных источников", type="primary"):
                selected_set = set(selected_active_ids)
                for source in video_sources:
                    set_video_source_active_fn(source_id=source["id"], is_active=source["id"] in selected_set)
                st.success("Набор активных источников обновлен.")
                st.rerun()

            selected_label = st.selectbox("Источник для редактирования", options=list(source_labels.keys()), key="source_edit_select")
            selected = source_labels[selected_label]
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

    with tab_check:
        if not video_sources:
            st.info("Сначала добавьте источник.")
        else:
            selected_label = st.selectbox("Источник для проверки", options=list(source_labels.keys()), key="source_check_select")
            selected = source_labels[selected_label]
            if st.button("Тест подключения"):
                success, message = test_connection_fn(selected["source_type"], selected["source_url"])
                if success:
                    st.success(message)
                else:
                    st.error(message)

    with tab_iphone:
        st.info(
            "Для iPhone не используйте `localhost`. Открывайте UI по IP-адресу компьютера или сервера в той же сети, например `http://<ip-компьютера>:8501`, либо через внешний HTTPS/VPN."
        )
        st.markdown(
            """
            **Варианты подключения iPhone**

            1. `Safari / браузерная камера`
            Открываете приложение с iPhone и используете камеру браузера. Это удобно для проверки, но поток живет внутри мобильной сессии и не становится постоянным production-источником для всех операторов.

            2. `IP camera / RTSP / HLS`
            Самый правильный вариант для постоянного источника: iPhone публикует поток через приложение-IP-камеру, а система читает его как обычный `RTSP/IP` или `HLS / HTTP` источник.

            3. `Continuity Camera на macOS`
            Если приложение запущено локально на Mac, iPhone можно использовать как системную камеру macOS и подключить как `USB / локальная камера на сервере`.
            """
        )
        quick_col1, quick_col2 = st.columns(2, gap="large")
        with quick_col1:
            st.caption("Быстрый preset для браузерной камеры iPhone")
            if st.button("Добавить источник «iPhone Safari Camera»"):
                create_video_source_fn(
                    name="iPhone Safari Camera",
                    source_type="browser_camera",
                    source_url="browser_camera",
                    location="mobile",
                    description="Мобильная браузерная камера iPhone",
                    is_active=True,
                    **_render_source_processing_defaults(),
                )
                st.success("Источник для Safari-камеры iPhone добавлен.")
                st.rerun()
        with quick_col2:
            with st.form("iphone_network_camera_form"):
                st.caption("Добавить iPhone как сетевую камеру")
                iphone_name = st.text_input("Наименование", value="iPhone IP Camera")
                iphone_source_type = st.selectbox(
                    "Тип потока",
                    options=["rtsp", "stream_url"],
                    format_func=lambda key: SOURCE_TYPES[key],
                )
                iphone_url = st.text_input("URL потока", placeholder="rtsp://... или https://...")
                iphone_location = st.text_input("Локация", value="mobile")
                iphone_active = st.checkbox("Активировать сразу", value=True)
                iphone_submit = st.form_submit_button("Добавить сетевой источник iPhone")
            if iphone_submit:
                if not iphone_name.strip() or not iphone_url.strip():
                    st.error("Для сетевой камеры iPhone нужно указать название и URL потока.")
                else:
                    create_video_source_fn(
                        name=iphone_name,
                        source_type=iphone_source_type,
                        source_url=iphone_url.strip(),
                        location=iphone_location.strip(),
                        description="iPhone network camera",
                        is_active=iphone_active,
                        **_render_source_processing_defaults(),
                    )
                    st.success("Сетевой источник iPhone добавлен.")
                    st.rerun()


def _fmt_ts(timestamp_value):
    if not timestamp_value:
        return "—"
    return datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
