"""Video source management UI."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from config.app_config import AI_QUALITY_PROFILES, TRACKER_OPTIONS
from services.source_health import normalize_source_runtime_status
from services.source_service import build_source_setup_hint, validate_source_definition


SOURCE_TYPES = {
    "rtsp": "RTSP/IP-камера",
    "stream_url": "HLS / HTTP-поток",
    "usb_camera": "USB / локальная камера на сервере",
    "browser_camera": "Браузерная камера",
}
PRODUCTION_SOURCE_TYPES = ("rtsp", "stream_url", "usb_camera")
LAB_SOURCE_TYPES = ("browser_camera",)


def split_video_sources(video_sources: list[dict]) -> tuple[list[dict], list[dict]]:
    production_sources = [source for source in video_sources if source.get("source_type") in PRODUCTION_SOURCE_TYPES]
    lab_sources = [source for source in video_sources if source.get("source_type") in LAB_SOURCE_TYPES]
    return production_sources, lab_sources


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


def _build_default_source_name(source_type: str, source_url: str) -> str:
    source_url = (source_url or "").strip()
    if source_type == "usb_camera":
        return f"USB Camera {source_url or '0'}"
    if source_type == "stream_url":
        return "HTTP Stream Camera"
    if source_type == "browser_camera":
        return "Browser Live Camera"
    return "RTSP Camera"


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
    st.markdown("**AI / исследовательские overrides**")
    ai_col1, ai_col2 = st.columns(2)
    with ai_col1:
        ai_profile_options = [""] + list(AI_QUALITY_PROFILES.keys())
        ai_profile_override = st.selectbox(
            "Профиль качества",
            options=ai_profile_options,
            index=ai_profile_options.index(source.get("ai_profile_override", ""))
            if source.get("ai_profile_override", "") in ai_profile_options
            else 0,
            format_func=lambda key: "Использовать системный профиль" if not key else AI_QUALITY_PROFILES[key]["label"],
            key=f"{prefix}_ai_profile_override",
        )
        conf_threshold_override_enabled = st.checkbox(
            "Переопределить confidence",
            value=source.get("conf_threshold_override") is not None,
            key=f"{prefix}_conf_threshold_override_enabled",
        )
        conf_threshold_override = (
            st.slider(
                "Camera confidence threshold",
                min_value=0.05,
                max_value=0.95,
                value=float(source.get("conf_threshold_override") or 0.45),
                step=0.05,
                key=f"{prefix}_conf_threshold_override",
            )
            if conf_threshold_override_enabled
            else None
        )
        incident_threshold_override_enabled = st.checkbox(
            "Переопределить incident score",
            value=source.get("incident_threshold_override") is not None,
            key=f"{prefix}_incident_threshold_override_enabled",
        )
        incident_threshold_override = (
            st.slider(
                "Camera incident score threshold",
                min_value=0.05,
                max_value=0.95,
                value=float(source.get("incident_threshold_override") or 0.55),
                step=0.05,
                key=f"{prefix}_incident_threshold_override",
            )
            if incident_threshold_override_enabled
            else None
        )
    with ai_col2:
        inference_size_override_enabled = st.checkbox(
            "Переопределить размер инференса",
            value=source.get("inference_size_override") is not None,
            key=f"{prefix}_inference_size_override_enabled",
        )
        inference_size_override = (
            st.selectbox(
                "Camera inference size",
                options=[320, 416, 512, 640, 960, 1280],
                index=[320, 416, 512, 640, 960, 1280].index(int(source.get("inference_size_override") or 512)),
                key=f"{prefix}_inference_size_override",
            )
            if inference_size_override_enabled
            else None
        )
        tracker_type_override = st.selectbox(
            "Camera tracking strategy",
            options=[""] + list(TRACKER_OPTIONS.keys()),
            index=([""] + list(TRACKER_OPTIONS.keys())).index(source.get("tracker_type_override", ""))
            if source.get("tracker_type_override", "") in ([""] + list(TRACKER_OPTIONS.keys()))
            else 0,
            format_func=lambda key: "Использовать системный tracker" if not key else TRACKER_OPTIONS[key]["label"],
            key=f"{prefix}_tracker_type_override",
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
        "ai_profile_override": ai_profile_override,
        "conf_threshold_override": conf_threshold_override,
        "inference_size_override": int(inference_size_override) if inference_size_override is not None else None,
        "tracker_type_override": tracker_type_override,
        "incident_threshold_override": incident_threshold_override,
    }


def render_video_sources(
    st,
    *,
    video_sources: list[dict],
    worker_statuses: list[dict],
    access_context: dict,
    create_video_source_fn,
    update_video_source_fn,
    set_video_source_active_fn,
    test_connection_fn,
):
    can_manage = access_context.get("role") == "admin"
    if not can_manage:
        st.info("Управление подключением камер доступно только администратору. Экран открыт в режиме просмотра.")
    production_sources, lab_sources = split_video_sources(video_sources)
    statuses_by_source = {status["source_id"]: status for status in worker_statuses}
    source_labels = {f"{source['name']} [{source['id']}]": source for source in video_sources}
    active_production_ids = [source["id"] for source in production_sources if source.get("is_active")]

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric("Production-камер", len(production_sources))
    summary_col2.metric("Активных production", len(active_production_ids))
    summary_col3.metric("Online / live", sum(1 for status in worker_statuses if status.get("is_connected")))
    summary_col4.metric("Лабораторных источников", len(lab_sources))

    with st.container(border=True):
        st.subheader("Реестр подключенных источников")
        st.caption(
            "Production-контур должен опираться на RTSP/HLS/USB-камеры. "
            "Browser-источники и мобильные сценарии относятся к лабораторному контуру диагностики."
        )
        rows = []
        for source in video_sources:
            raw_status = statuses_by_source.get(source["id"], {})
            status = normalize_source_runtime_status(raw_status)
            rows.append(
                {
                    "ID": source["id"],
                    "Наименование": source["name"],
                    "Тип": SOURCE_TYPES.get(source["source_type"], source["source_type"]),
                    "Локация": source.get("location") or "",
                    "ROI": "вкл" if source.get("enable_roi") else "выкл",
                    "Активен": "да" if source.get("is_active") else "нет",
                    "Статус": status["health_status"],
                    "Соединение": status["connection_status"],
                    "Последний heartbeat": _fmt_ts(raw_status.get("last_heartbeat")),
                    "Последний кадр": _fmt_ts(status.get("last_frame_at")),
                    "Ошибка": status.get("last_error") or "—",
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    tab_add, tab_manage, tab_check, tab_lab = st.tabs(
        ["Production-камеры", "Управление и активация", "Проверка подключения", "Лаборатория и мобильные камеры"]
    )

    with tab_add:
        st.subheader("Быстрое добавление камеры")
        st.caption(
            "Основной путь: вставьте RTSP/HTTP-адрес или индекс USB-камеры, проверьте подключение и сохраните. "
            "Сложные ROI/AI-настройки скрыты ниже и нужны только при нестандартном сценарии."
        )
        draft_input = st.text_input(
            "Адрес потока или индекс устройства",
            key="quick_source_input",
            placeholder="rtsp://... / https://... / 0",
        )
        source_hint = build_source_setup_hint(draft_input)
        hint_col1, hint_col2 = st.columns([0.8, 1.2], gap="large")
        with hint_col1:
            st.info(f"Автоопределение: {source_hint['label']}")
        with hint_col2:
            st.caption(source_hint["help"])
            st.code(source_hint["placeholder"])

        with st.form("create_video_source_form", clear_on_submit=True):
            source_type = st.selectbox(
                "Тип production-источника",
                options=list(PRODUCTION_SOURCE_TYPES),
                index=list(PRODUCTION_SOURCE_TYPES).index(source_hint["source_type"])
                if source_hint["source_type"] in PRODUCTION_SOURCE_TYPES
                else 0,
                format_func=lambda key: SOURCE_TYPES[key],
                help="Можно оставить автоопределение или скорректировать вручную.",
            )
            suggested_name = _build_default_source_name(source_type, draft_input)
            name = st.text_input("Наименование", value=suggested_name)
            source_url = st.text_input(
                "URL / индекс устройства",
                value=draft_input,
                placeholder=build_source_setup_hint(draft_input or source_type)["placeholder"]
                if draft_input
                else build_source_setup_hint(source_type)["placeholder"],
            )
            location = st.text_input("Локация", placeholder="Например: Главная проходная")
            description = st.text_area("Описание", placeholder="Необязательно. Например: Северный вход, обзор турникета.")
            is_active = st.checkbox("Сделать активным сразу", value=True)
            use_advanced = st.checkbox("Открыть расширенные настройки ROI/AI", value=False)
            processing_config = (
                _render_source_processing_controls(st, prefix="create_source")
                if use_advanced
                else _render_source_processing_defaults()
            )
            submitted = st.form_submit_button("Проверить и сохранить источник")
        if submitted:
            validation_errors, normalized_source_url = validate_source_definition(
                name=name,
                source_type=source_type,
                source_url=source_url,
            )
            if not can_manage:
                st.error("Недостаточно прав для добавления камер.")
            elif validation_errors:
                for error_text in validation_errors:
                    st.error(error_text)
            else:
                success, message = test_connection_fn(source_type, normalized_source_url)
                if not success:
                    st.error(message)
                    st.stop()
                create_video_source_fn(
                    name=name,
                    source_type=source_type,
                    source_url=normalized_source_url,
                    location=location,
                    description=description,
                    is_active=is_active,
                    **processing_config,
                )
                st.success("Источник видеоданных добавлен и успешно прошёл предварительную проверку.")
                st.rerun()

    with tab_manage:
        if not video_sources:
            st.info("Нет источников для изменения.")
        else:
            selected_active_ids = st.multiselect(
                "Активные production-источники",
                options=[source["id"] for source in production_sources],
                default=active_production_ids,
                format_func=lambda source_id: next(
                    f"{source['name']} [{SOURCE_TYPES.get(source['source_type'], source['source_type'])}]"
                    for source in production_sources
                    if source["id"] == source_id
                ),
                help="В production-мониторинг попадают только эти камеры. Лабораторные browser-источники активируются отдельно.",
            )
            if st.button("Сохранить набор активных источников", type="primary"):
                if not can_manage:
                    st.error("Недостаточно прав для изменения активных камер.")
                else:
                    selected_set = set(selected_active_ids)
                    for source in production_sources:
                        set_video_source_active_fn(source_id=source["id"], is_active=source["id"] in selected_set)
                    st.success("Набор active production-камер обновлен.")
                st.rerun()

            if lab_sources:
                st.caption(
                    f"Лабораторные источники: {', '.join(source['name'] for source in lab_sources)}. "
                    "Они не попадают в основную video wall и используются только для диагностики."
                )

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
                if not can_manage:
                    st.error("Недостаточно прав для редактирования камер.")
                else:
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
            selected_hint = build_source_setup_hint(selected["source_url"])
            st.caption(
                f"Тип: {SOURCE_TYPES.get(selected['source_type'], selected['source_type'])}. "
                f"Ожидаемый формат: `{selected_hint['placeholder']}`"
            )
            if st.button("Тест подключения"):
                success, message = test_connection_fn(selected["source_type"], selected["source_url"])
                if success:
                    st.success(message)
                else:
                    st.error(message)

    with tab_lab:
        with st.container(border=True):
            st.subheader("Лабораторный контур")
            st.caption(
                "Здесь настраиваются browser-live и мобильные камеры для демонстрации, быстрой диагностики и полевых проверок. "
                "Основной операторский контур на них не опирается."
            )
            lab_rows = [
                {
                    "ID": source["id"],
                    "Источник": source["name"],
                    "Тип": SOURCE_TYPES.get(source["source_type"], source["source_type"]),
                    "Активен": "да" if source.get("is_active") else "нет",
                }
                for source in lab_sources
            ]
            if lab_rows:
                st.dataframe(pd.DataFrame(lab_rows), width="stretch", hide_index=True)
            else:
                st.info("Лабораторные источники ещё не добавлены.")

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
        browser_col, iphone_cols = st.columns([0.9, 1.1], gap="large")
        with browser_col:
            with st.form("create_browser_lab_source", clear_on_submit=True):
                st.caption("Добавить browser-live источник для диагностики")
                browser_name = st.text_input("Наименование", value="Browser Live Camera")
                browser_location = st.text_input("Локация", value="lab")
                browser_active = st.checkbox("Включить источник", value=False)
                browser_submit = st.form_submit_button("Добавить browser-live источник")
            if browser_submit:
                if not can_manage:
                    st.error("Недостаточно прав для добавления лабораторного источника.")
                elif not browser_name.strip():
                    st.error("Укажите наименование browser-live источника.")
                else:
                    create_video_source_fn(
                        name=browser_name.strip(),
                        source_type="browser_camera",
                        source_url="browser_camera",
                        location=browser_location.strip(),
                        description="Лабораторный browser-live источник",
                        is_active=browser_active,
                        **_render_source_processing_defaults(),
                    )
                    st.success("Browser-live источник добавлен в лабораторный контур.")
                    st.rerun()
        with iphone_cols:
            quick_col1, quick_col2 = st.columns(2, gap="large")
        with quick_col1:
            st.caption("Быстрый preset для браузерной камеры iPhone")
            if st.button("Добавить источник «iPhone Safari Camera»"):
                if not can_manage:
                    st.error("Недостаточно прав для добавления мобильной камеры.")
                else:
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
                if not can_manage:
                    st.error("Недостаточно прав для добавления сетевой камеры.")
                elif not iphone_name.strip() or not iphone_url.strip():
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
