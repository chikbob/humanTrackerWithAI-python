"""Online monitoring UI with production-first and demo fallback modes."""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from services.events import add_notification, process_disappeared_tracks, register_detection_and_entry_events
from services.state import finish_session, get_current_session, log_frame, start_session
from ui.sidebar import ANIMAL_CLASSES
from utils.performance import DEFAULT_SESSION_PERSIST_INTERVAL, DEFAULT_UI_REFRESH_INTERVAL_SEC
from utils.vision import draw_fancy_box, rotate_frame


def render_online_monitoring(
    st,
    *,
    active_sources: list[dict],
    worker_statuses: list[dict],
    events: list[dict],
    model_name: str,
    model,
    class_meta: dict,
    inference_size: int,
    conf_threshold: float,
    frame_skip: int,
    access_point_name: str,
    session_state,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    demo_mode: bool,
):
    statuses_by_id = {status["source_id"]: status for status in worker_statuses}
    left_col, right_col = st.columns([1.9, 1.0], gap="large")

    with left_col:
        with st.container(border=True):
            st.subheader("Live monitoring")
            if active_sources:
                source_labels = {f"{source['name']} [{source['source_type']}]": source for source in active_sources}
                selected_label = st.selectbox("Источник production-потока", options=list(source_labels.keys()))
                selected_source = source_labels[selected_label]
                selected_status = statuses_by_id.get(selected_source["id"], {})
                snapshot_path = selected_status.get("last_snapshot_path")
                if snapshot_path and Path(snapshot_path).exists():
                    st.image(snapshot_path, use_container_width=True, caption=f"Входная зона: {selected_source['name']}")
                else:
                    st.info("Фоновый worker подключен, но изображение еще не получено.")
                st.caption(
                    "ROI входной зоны и детекции сотрудников отрисовываются серверным worker "
                    "и поступают в интерфейс через БД и runtime snapshots."
                )
            else:
                st.warning("Нет активных production-источников. Добавьте RTSP/IP/USB источник в разделе «Источники видео».")

        with st.container(border=True):
            st.subheader("Последние события входной зоны")
            latest_rows = [
                {
                    "Время": datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S"),
                    "Тип события": event.get("event_type"),
                    "Источник": event.get("source_name"),
                    "Уверенность": round(event.get("confidence") or 0.0, 3),
                }
                for event in events[:12]
            ]
            st.dataframe(pd.DataFrame(latest_rows), use_container_width=True, hide_index=True)

    with right_col:
        with st.container(border=True):
            st.subheader("Панель состояния")
            selected_source = active_sources[0] if active_sources else None
            selected_status = statuses_by_id.get(selected_source["id"], {}) if selected_source else {}
            st.metric("FPS", round(selected_status.get("fps") or 0.0, 2))
            st.metric("confidence threshold", round(conf_threshold, 2))
            st.metric("Источник потока", selected_source["name"] if selected_source else "не выбран")
            st.metric("Активная модель", model_name)
            st.metric("Точка прохода", access_point_name)
            st.metric("Последний кадр", _fmt_ts(selected_status.get("last_frame_at")))
            if selected_status.get("last_error"):
                st.error(selected_status["last_error"])

    with st.expander("Демо и fallback режимы", expanded=demo_mode):
        st.caption(
            "Этот блок сохраняет демонстрационные сценарии: загрузку видеофайла, снимка, браузерную камеру "
            "и локальную камеру. Основной production-путь должен использовать серверный источник и фоновый worker."
        )
        _render_demo_workspace(
            st,
            model_name=model_name,
            model=model,
            class_meta=class_meta,
            inference_size=inference_size,
            conf_threshold=conf_threshold,
            frame_skip=frame_skip,
            session_state=session_state,
            db_insert_event=db_insert_event,
            db_insert_frame=db_insert_frame,
            db_upsert_session=db_upsert_session,
        )


def _render_demo_workspace(
    st,
    *,
    model_name: str,
    model,
    class_meta: dict,
    inference_size: int,
    conf_threshold: float,
    frame_skip: int,
    session_state,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
):
    mode = st.radio(
        "Режим fallback",
        options=["Загрузить фото", "Загрузить видео", "Браузерная камера", "Локальная камера"],
        horizontal=True,
    )
    rotation_angle = 0
    roi_config = {"enable_roi": True, "roi_x": 20, "roi_y": 20, "roi_w": 60, "roi_h": 60}
    event_settings = {
        "rule_count_enabled": False,
        "rule_class": "person",
        "rule_n": 3,
        "rule_t": 10,
        "rule_disappear_enabled": True,
        "rule_disappear_seconds": 5,
        "enable_notifications": False,
        "notify_conf_threshold": conf_threshold,
        "notify_classes": ["person"],
        "enable_roi": True,
        "default_access_point_id": None,
        "prolonged_presence_seconds": 10,
        "event_cooldown": 5,
    }

    def notify(text: str):
        add_notification(session_state, text, enabled=False, toast_callback=None)

    def register_event_pipeline(*, frame_index: int, detection: dict, source_type: str, session: dict):
        register_detection_and_entry_events(
            session_state,
            db_insert_event,
            session=session,
            frame_index=frame_index,
            detection=detection,
            source_type=source_type,
            settings=event_settings,
            notify_callback=notify,
        )

    def process_disappeared(*, frame_index: int, source_type: str, session: dict, frame_width: int, frame_height: int):
        process_disappeared_tracks(
            session_state,
            db_insert_event,
            session=session,
            frame_index=frame_index,
            source_type=source_type,
            frame_width=frame_width,
            frame_height=frame_height,
            rule_disappear_enabled=event_settings["rule_disappear_enabled"],
            rule_disappear_seconds=event_settings["rule_disappear_seconds"],
            enable_notifications=False,
            notify_callback=notify,
            default_access_point_id=None,
        )

    frame_display = st.empty()
    if mode == "Загрузить фото":
        uploaded_image = st.file_uploader("Файл изображения", type=["jpg", "jpeg", "png"], key="fallback_image")
        if uploaded_image:
            start_session(
                session_state,
                db_upsert_session,
                model_name=model_name,
                source_type="image",
                source_path=uploaded_image.name,
                animal_filter="всё",
                track_classes=["person"],
                rotation_angle=rotation_angle,
            )
            image = Image.open(uploaded_image).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            _process_single_frame(
                st=st,
                frame_bgr=frame_bgr,
                frame_index=0,
                source_type="image",
                use_tracking=False,
                frame_display=frame_display,
                model=model,
                class_meta=class_meta,
                inference_size=inference_size,
                conf_threshold=conf_threshold,
                session_state=session_state,
                db_insert_frame=db_insert_frame,
                db_upsert_session=db_upsert_session,
                rotation_angle=rotation_angle,
                register_event_pipeline=register_event_pipeline,
                process_disappeared=process_disappeared,
            )
            finish_session(session_state, db_upsert_session)
    elif mode == "Загрузить видео":
        uploaded_video = st.file_uploader("Файл видео", type=["mp4", "avi", "mov"], key="fallback_video")
        if uploaded_video:
            start_session(
                session_state,
                db_upsert_session,
                model_name=model_name,
                source_type="video",
                source_path=uploaded_video.name,
                animal_filter="всё",
                track_classes=["person"],
                rotation_angle=rotation_angle,
            )
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(uploaded_video.read())
            temp_video.flush()
            temp_path = temp_video.name
            temp_video.close()
            cap = cv2.VideoCapture(temp_path)
            frame_index = 0
            last_ui_draw_ts = 0.0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
                    frame_index += 1
                    continue
                frame_rgb = _process_single_frame(
                    st=st,
                    frame_bgr=frame,
                    frame_index=frame_index,
                    source_type="video",
                    use_tracking=True,
                    frame_display=frame_display,
                    model=model,
                    class_meta=class_meta,
                    inference_size=inference_size,
                    conf_threshold=conf_threshold,
                    session_state=session_state,
                    db_insert_frame=db_insert_frame,
                    db_upsert_session=db_upsert_session,
                    rotation_angle=rotation_angle,
                    register_event_pipeline=register_event_pipeline,
                    process_disappeared=process_disappeared,
                    draw_now=time.time() - last_ui_draw_ts >= DEFAULT_UI_REFRESH_INTERVAL_SEC,
                )
                if frame_rgb is not None and time.time() - last_ui_draw_ts >= DEFAULT_UI_REFRESH_INTERVAL_SEC:
                    frame_display.image(frame_rgb, channels="RGB")
                    last_ui_draw_ts = time.time()
                frame_index += 1
            cap.release()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            finish_session(session_state, db_upsert_session)
            st.success("Видео обработано в demo/fallback режиме.")
    elif mode == "Браузерная камера":
        shot = st.camera_input("Снимок из браузерной камеры", key="fallback_browser_camera")
        if shot is not None:
            start_session(
                session_state,
                db_upsert_session,
                model_name=model_name,
                source_type="webcam_browser",
                source_path="browser_camera",
                animal_filter="всё",
                track_classes=["person"],
                rotation_angle=rotation_angle,
            )
            image = Image.open(shot).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            _process_single_frame(
                st=st,
                frame_bgr=frame_bgr,
                frame_index=0,
                source_type="webcam_browser",
                use_tracking=False,
                frame_display=frame_display,
                model=model,
                class_meta=class_meta,
                inference_size=inference_size,
                conf_threshold=conf_threshold,
                session_state=session_state,
                db_insert_frame=db_insert_frame,
                db_upsert_session=db_upsert_session,
                rotation_angle=rotation_angle,
                register_event_pipeline=register_event_pipeline,
                process_disappeared=process_disappeared,
            )
            finish_session(session_state, db_upsert_session)
    else:
        camera_index = st.number_input("Индекс локальной камеры", min_value=0, step=1, value=0)
        run_col1, run_col2 = st.columns(2)
        with run_col1:
            start_button = st.button("Запустить локальную камеру")
        with run_col2:
            stop_button = st.button("Остановить локальную камеру")
        if "fallback_camera_running" not in session_state:
            session_state.fallback_camera_running = False
        if start_button:
            session_state.fallback_camera_running = True
        if stop_button:
            session_state.fallback_camera_running = False
        if session_state.fallback_camera_running:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error("Не удалось открыть локальную камеру.")
                session_state.fallback_camera_running = False
                return
            start_session(
                session_state,
                db_upsert_session,
                model_name=model_name,
                source_type="webcam",
                source_path=f"camera:{camera_index}",
                animal_filter="всё",
                track_classes=["person"],
                rotation_angle=rotation_angle,
            )
            frame_index = 0
            last_ui_draw_ts = 0.0
            while session_state.fallback_camera_running:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
                    frame_index += 1
                    continue
                frame_rgb = _process_single_frame(
                    st=st,
                    frame_bgr=frame,
                    frame_index=frame_index,
                    source_type="webcam",
                    use_tracking=True,
                    frame_display=frame_display,
                    model=model,
                    class_meta=class_meta,
                    inference_size=inference_size,
                    conf_threshold=conf_threshold,
                    session_state=session_state,
                    db_insert_frame=db_insert_frame,
                    db_upsert_session=db_upsert_session,
                    rotation_angle=rotation_angle,
                    register_event_pipeline=register_event_pipeline,
                    process_disappeared=process_disappeared,
                    draw_now=time.time() - last_ui_draw_ts >= DEFAULT_UI_REFRESH_INTERVAL_SEC,
                )
                if frame_rgb is not None and time.time() - last_ui_draw_ts >= DEFAULT_UI_REFRESH_INTERVAL_SEC:
                    frame_display.image(frame_rgb, channels="RGB")
                    last_ui_draw_ts = time.time()
                frame_index += 1
            cap.release()
            finish_session(session_state, db_upsert_session)
            session_state.fallback_camera_running = False


def _process_single_frame(
    *,
    st,
    frame_bgr,
    frame_index: int,
    source_type: str,
    use_tracking: bool,
    frame_display,
    model,
    class_meta: dict,
    inference_size: int,
    conf_threshold: float,
    session_state,
    db_insert_frame,
    db_upsert_session,
    rotation_angle: int,
    register_event_pipeline,
    process_disappeared,
    draw_now: bool = True,
):
    from core.detection import detect_and_annotate

    frame_bgr = rotate_frame(frame_bgr, rotation_angle)
    frame_rgb, detections_meta, processing_time_ms = detect_and_annotate(
        frame_bgr,
        frame_index=frame_index,
        source_type=source_type,
        use_tracking=use_tracking,
        model=model,
        conf_threshold=conf_threshold,
        inference_size=inference_size,
        session=get_current_session(session_state),
        class_meta=class_meta,
        animal_filter="всё",
        animal_classes=ANIMAL_CLASSES,
        track_classes=["person"],
        roi_config={"enable_roi": True, "roi_x": 20, "roi_y": 20, "roi_w": 60, "roi_h": 60},
        event_settings={
            "rule_count_enabled": False,
            "rule_class": "person",
            "rule_n": 3,
            "rule_t": 10,
            "rule_disappear_enabled": True,
            "rule_disappear_seconds": 5,
            "enable_notifications": False,
            "notify_conf_threshold": conf_threshold,
            "notify_classes": ["person"],
            "enable_roi": True,
            "default_access_point_id": None,
            "prolonged_presence_seconds": 10,
            "event_cooldown": 5,
        },
        register_event_fn=register_event_pipeline,
        process_disappeared_fn=process_disappeared,
        draw_box_fn=draw_fancy_box,
        warning_callback=st.warning,
    )
    log_frame(
        session_state,
        db_insert_frame,
        db_upsert_session,
        frame_index=frame_index,
        frame_shape=frame_rgb.shape,
        processing_time_ms=processing_time_ms,
        detections_meta=detections_meta,
        rotation_angle=rotation_angle,
        persist_interval=DEFAULT_SESSION_PERSIST_INTERVAL,
        force_session_sync=not use_tracking,
    )
    if draw_now:
        frame_display.image(frame_rgb, channels="RGB")
    return frame_rgb


def _fmt_ts(timestamp_value):
    if not timestamp_value:
        return "—"
    return datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
