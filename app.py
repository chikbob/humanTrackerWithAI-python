import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.detection import build_class_meta, detect_and_annotate, detect_and_draw_live, load_model
from db.repository import (
    create_employee,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    ensure_demo_employees,
    init_db,
    load_access_logs,
    load_employees,
    load_history_from_db,
    update_employee,
    update_employee_status,
)
from services.events import add_notification, process_disappeared_tracks, register_detection_and_entry_events
from services.state import finish_session, get_current_session, init_session_state, log_frame, start_session
from ui.analytics import render_analytics, render_status_panel
from ui.page import configure_page
from ui.sidebar import ANIMAL_CLASSES, MODEL_MAP, render_detection_sidebar, render_primary_sidebar
from utils.performance import DEFAULT_SESSION_PERSIST_INTERVAL, DEFAULT_UI_REFRESH_INTERVAL_SEC
from utils.vision import draw_fancy_box, rotate_frame

try:
    import av
    from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False
    av = None
    RTCConfiguration = None
    WebRtcMode = None
    webrtc_streamer = None


RTC_CONFIG = (
    RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    if WEBRTC_AVAILABLE
    else None
)


def main():
    init_db()
    init_session_state(st.session_state, load_history_from_db)
    configure_page(st)

    primary_config = render_primary_sidebar(st)
    demo_mode = primary_config["demo_mode"]
    if demo_mode:
        ensure_demo_employees()

    employees = load_employees()
    access_logs = load_access_logs()
    model_name = MODEL_MAP[primary_config["model_choice"]]
    model = load_model(model_name)

    all_class_names, class_meta = build_class_meta(model.names, ANIMAL_CLASSES)
    secondary_config = render_detection_sidebar(
        st,
        all_class_names=all_class_names,
        show_advanced=primary_config["show_advanced"],
    )

    source_mode = primary_config["source_mode"]
    rotation_angle = primary_config["rotation_angle"]
    conf_threshold = 0.45 if demo_mode else primary_config["conf_threshold"]
    notify_conf_threshold = 0.45 if demo_mode else primary_config["notify_conf_threshold"]
    inference_size = 512 if demo_mode else primary_config["inference_size"]
    frame_skip = 1 if demo_mode else primary_config["frame_skip"]
    enable_notifications = True if demo_mode else primary_config["enable_notifications"]
    animal_filter = secondary_config["animal_filter"]
    track_classes = secondary_config["track_classes"]
    roi_config = secondary_config["roi_config"]

    event_settings = {
        **secondary_config["event_settings"],
        "enable_notifications": enable_notifications,
        "notify_conf_threshold": notify_conf_threshold,
        "notify_classes": secondary_config["notify_classes"],
        "enable_roi": roi_config["enable_roi"],
        # ROI в текущей прикладной постановке интерпретируется как контролируемая входная зона предприятия.
        "default_access_point_id": None,
        "prolonged_presence_seconds": 10,
    }
    if demo_mode:
        roi_config["enable_roi"] = True
        event_settings["rule_disappear_enabled"] = True
        event_settings["rule_disappear_seconds"] = 5

    def notify(text: str):
        add_notification(
            st.session_state,
            text,
            enabled=enable_notifications,
            toast_callback=st.toast,
        )

    def register_event_pipeline(*, frame_index: int, detection: dict, source_type: str, session: dict):
        # Разделение низкоуровневой телеметрии детекции и предметно-ориентированных событий мониторинга.
        register_detection_and_entry_events(
            st.session_state,
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
            st.session_state,
            db_insert_event,
            session=session,
            frame_index=frame_index,
            source_type=source_type,
            frame_width=frame_width,
            frame_height=frame_height,
            rule_disappear_enabled=event_settings["rule_disappear_enabled"],
            rule_disappear_seconds=event_settings["rule_disappear_seconds"],
            enable_notifications=enable_notifications,
            notify_callback=notify,
            default_access_point_id=event_settings["default_access_point_id"],
        )

    st.markdown("---")
    work_col, info_col = st.columns([2.2, 1.0], gap="large")

    with work_col:
        with st.container(border=True):
            st.subheader("🎥 Мониторинг входной зоны в реальном времени")
            if demo_mode:
                st.caption(
                    "Демо-режим активен: включены устойчивые параметры анализа и подготовлен тестовый набор сотрудников."
                )
            frame_display = st.empty()

            if source_mode == "📁 Загрузить фото":
                uploaded_image = st.file_uploader(
                    "Загрузите изображение входной зоны",
                    type=["jpg", "jpeg", "png"],
                    key="img_uploader",
                )
                if uploaded_image:
                    start_session(
                        st.session_state,
                        db_upsert_session,
                        model_name=model_name,
                        source_type="image",
                        source_path=uploaded_image.name,
                        animal_filter=animal_filter,
                        track_classes=track_classes,
                        rotation_angle=rotation_angle,
                    )

                    image = Image.open(uploaded_image).convert("RGB")
                    img_array = np.array(image)
                    img_array = rotate_frame(img_array, rotation_angle)
                    frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    frame_rgb, detections_meta, processing_time_ms = detect_and_annotate(
                        frame_bgr,
                        frame_index=0,
                        source_type="image",
                        use_tracking=False,
                        model=model,
                        conf_threshold=conf_threshold,
                        inference_size=inference_size,
                        session=get_current_session(st.session_state),
                        class_meta=class_meta,
                        animal_filter=animal_filter,
                        animal_classes=ANIMAL_CLASSES,
                        track_classes=track_classes,
                        roi_config=roi_config,
                        event_settings=event_settings,
                        register_event_fn=register_event_pipeline,
                        process_disappeared_fn=process_disappeared,
                        draw_box_fn=draw_fancy_box,
                        warning_callback=st.warning,
                    )

                    log_frame(
                        st.session_state,
                        db_insert_frame,
                        db_upsert_session,
                        frame_index=0,
                        frame_shape=frame_rgb.shape,
                        processing_time_ms=processing_time_ms,
                        detections_meta=detections_meta,
                        rotation_angle=rotation_angle,
                        persist_interval=DEFAULT_SESSION_PERSIST_INTERVAL,
                        force_session_sync=True,
                    )
                    finish_session(st.session_state, db_upsert_session)
                    frame_display.image(frame_rgb, channels="RGB")
                    st.success("Изображение входной зоны обработано.")
                else:
                    st.info("Загрузите файл, чтобы выполнить проверку входной зоны.")

            elif source_mode == "🎞️ Загрузить видео":
                uploaded_video = st.file_uploader(
                    "Загрузите видео входной зоны предприятия",
                    type=["mp4", "avi", "mov"],
                    key="video_uploader",
                )
                if uploaded_video:
                    start_session(
                        st.session_state,
                        db_upsert_session,
                        model_name=model_name,
                        source_type="video",
                        source_path=uploaded_video.name,
                        animal_filter=animal_filter,
                        track_classes=track_classes,
                        rotation_angle=rotation_angle,
                    )
                    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    temp_video.write(uploaded_video.read())
                    temp_video.flush()
                    temp_path = temp_video.name
                    temp_video.close()

                    cap = cv2.VideoCapture(temp_path)
                    st.info("▶️ Выполняется интеллектуальный анализ видеопотока входной зоны...")
                    frame_index = 0
                    last_ui_draw_ts = 0.0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
                            frame_index += 1
                            continue
                        frame = rotate_frame(frame, rotation_angle)
                        frame_rgb, detections_meta, processing_time_ms = detect_and_annotate(
                            frame,
                            frame_index=frame_index,
                            source_type="video",
                            use_tracking=True,
                            model=model,
                            conf_threshold=conf_threshold,
                            inference_size=inference_size,
                            session=get_current_session(st.session_state),
                            class_meta=class_meta,
                            animal_filter=animal_filter,
                            animal_classes=ANIMAL_CLASSES,
                            track_classes=track_classes,
                            roi_config=roi_config,
                            event_settings=event_settings,
                            register_event_fn=register_event_pipeline,
                            process_disappeared_fn=process_disappeared,
                            draw_box_fn=draw_fancy_box,
                            warning_callback=st.warning,
                        )
                        log_frame(
                            st.session_state,
                            db_insert_frame,
                            db_upsert_session,
                            frame_index=frame_index,
                            frame_shape=frame_rgb.shape,
                            processing_time_ms=processing_time_ms,
                            detections_meta=detections_meta,
                            rotation_angle=rotation_angle,
                            persist_interval=DEFAULT_SESSION_PERSIST_INTERVAL,
                        )
                        frame_index += 1
                        if time.time() - last_ui_draw_ts >= DEFAULT_UI_REFRESH_INTERVAL_SEC:
                            frame_display.image(frame_rgb, channels="RGB")
                            last_ui_draw_ts = time.time()

                    cap.release()
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except PermissionError:
                        pass

                    finish_session(st.session_state, db_upsert_session)
                    st.success("✅ Видео входной зоны обработано.")
                else:
                    st.info("Загрузите видеофайл, чтобы начать мониторинг прохода.")

            elif source_mode == "📷 Веб-камера":
                camera_mode = st.radio(
                    "Режим камеры входной зоны",
                    options=[
                        "Браузерная камера RT",
                        "Браузерная камера (снимок)",
                        "Локальная OpenCV камера",
                    ],
                    index=0,
                    horizontal=False,
                    key="camera_mode",
                )

                if camera_mode == "Браузерная камера RT":
                    if not WEBRTC_AVAILABLE:
                        st.error("Для режима онлайн-мониторинга установите зависимость streamlit-webrtc.")
                    else:
                        if "browser_rt_on" not in st.session_state:
                            st.session_state.browser_rt_on = False

                        ctl1, ctl2 = st.columns(2)
                        with ctl1:
                            start_rt = st.button("▶️ Запустить камеру", key="browser_rt_start")
                        with ctl2:
                            stop_rt = st.button("⏹ Остановить камеру", key="browser_rt_stop")

                        if start_rt and not st.session_state.browser_rt_on:
                            st.session_state.browser_rt_on = True
                        if stop_rt and st.session_state.browser_rt_on:
                            st.session_state.browser_rt_on = False
                            st.success("🛑 Мониторинг камеры остановлен.")

                        st.caption("Нажмите Start в виджете камеры и разрешите доступ к видеоустройству входной зоны в браузере.")

                        def _video_frame_callback(frame):
                            frame_bgr = frame.to_ndarray(format="bgr24")
                            rotated = rotate_frame(frame_bgr, rotation_angle)
                            frame_rgb = detect_and_draw_live(
                                rotated,
                                model=model,
                                conf_threshold=conf_threshold,
                                inference_size=inference_size,
                                class_meta=class_meta,
                                animal_filter=animal_filter,
                                animal_classes=ANIMAL_CLASSES,
                                track_classes=track_classes,
                                roi_config=roi_config,
                                draw_box_fn=draw_fancy_box,
                            )
                            return av.VideoFrame.from_ndarray(
                                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                                format="bgr24",
                            )

                        if st.session_state.browser_rt_on:
                            webrtc_streamer(
                                key="browser_webrtc_stream",
                                mode=WebRtcMode.SENDRECV,
                                rtc_configuration=RTC_CONFIG,
                                media_stream_constraints={"video": True, "audio": False},
                                video_frame_callback=_video_frame_callback,
                                async_processing=True,
                            )
                        else:
                            st.info("Нажмите «Запустить камеру», чтобы начать онлайн-мониторинг входной зоны.")

                elif camera_mode == "Браузерная камера (снимок)":
                    st.info("Режим снимка: выполните захват изображения входной зоны предприятия.")
                    shot = st.camera_input("Снимок с камеры входной зоны", key="browser_cam_input")
                    if shot is not None:
                        start_session(
                            st.session_state,
                            db_upsert_session,
                            model_name=model_name,
                            source_type="webcam_browser",
                            source_path="browser_camera",
                            animal_filter=animal_filter,
                            track_classes=track_classes,
                            rotation_angle=rotation_angle,
                        )
                        image = Image.open(shot).convert("RGB")
                        img_array = np.array(image)
                        img_array = rotate_frame(img_array, rotation_angle)
                        frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        frame_rgb, detections_meta, processing_time_ms = detect_and_annotate(
                            frame_bgr,
                            frame_index=0,
                            source_type="webcam_browser",
                            use_tracking=False,
                            model=model,
                            conf_threshold=conf_threshold,
                            inference_size=inference_size,
                            session=get_current_session(st.session_state),
                            class_meta=class_meta,
                            animal_filter=animal_filter,
                            animal_classes=ANIMAL_CLASSES,
                            track_classes=track_classes,
                            roi_config=roi_config,
                            event_settings=event_settings,
                            register_event_fn=register_event_pipeline,
                            process_disappeared_fn=process_disappeared,
                            draw_box_fn=draw_fancy_box,
                            warning_callback=st.warning,
                        )
                        log_frame(
                            st.session_state,
                            db_insert_frame,
                            db_upsert_session,
                            frame_index=0,
                            frame_shape=frame_rgb.shape,
                            processing_time_ms=processing_time_ms,
                            detections_meta=detections_meta,
                            rotation_angle=rotation_angle,
                            persist_interval=DEFAULT_SESSION_PERSIST_INTERVAL,
                            force_session_sync=True,
                        )
                        finish_session(st.session_state, db_upsert_session)
                        frame_display.image(frame_rgb, channels="RGB")
                        st.success("Снимок входной зоны обработан.")
                else:
                    camera_index = st.number_input("Номер локальной камеры", min_value=0, step=1, value=0, key="cam_index")
                    run_col1, run_col2 = st.columns(2)
                    with run_col1:
                        start_button = st.button("▶️ Запустить мониторинг", key="webcam_start")
                    with run_col2:
                        stop_button = st.button("⏹ Остановить мониторинг", key="webcam_stop")

                    if start_button:
                        st.session_state.running = True
                    if stop_button:
                        st.session_state.running = False

                    if st.session_state.running:
                        cap = cv2.VideoCapture(camera_index)
                        if not cap.isOpened():
                            st.error("❌ Не удалось открыть локальную камеру. Для веб-версии используйте браузерную камеру.")
                            st.session_state.running = False
                        else:
                            start_session(
                                st.session_state,
                                db_upsert_session,
                                model_name=model_name,
                                source_type="webcam",
                                source_path=f"camera:{camera_index}",
                                animal_filter=animal_filter,
                                track_classes=track_classes,
                                rotation_angle=rotation_angle,
                            )
                            st.info("✅ Мониторинг входной зоны запущен. Выполняется сопровождение объектов в зоне прохода.")
                            prev_time = time.time()
                            frame_index = 0

                            while st.session_state.running:
                                ret, frame = cap.read()
                                if not ret:
                                    st.warning("⚠️ Кадр с видеоустройства входной зоны не получен.")
                                    break
                                if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
                                    frame_index += 1
                                    continue
                                frame = rotate_frame(frame, rotation_angle)
                                frame_rgb, detections_meta, processing_time_ms = detect_and_annotate(
                                    frame,
                                    frame_index=frame_index,
                                    source_type="webcam",
                                    use_tracking=True,
                                    model=model,
                                    conf_threshold=conf_threshold,
                                    inference_size=inference_size,
                                    session=get_current_session(st.session_state),
                                    class_meta=class_meta,
                                    animal_filter=animal_filter,
                                    animal_classes=ANIMAL_CLASSES,
                                    track_classes=track_classes,
                                    roi_config=roi_config,
                                    event_settings=event_settings,
                                    register_event_fn=register_event_pipeline,
                                    process_disappeared_fn=process_disappeared,
                                    draw_box_fn=draw_fancy_box,
                                    warning_callback=st.warning,
                                )
                                log_frame(
                                    st.session_state,
                                    db_insert_frame,
                                    db_upsert_session,
                                    frame_index=frame_index,
                                    frame_shape=frame_rgb.shape,
                                    processing_time_ms=processing_time_ms,
                                    detections_meta=detections_meta,
                                    rotation_angle=rotation_angle,
                                    persist_interval=DEFAULT_SESSION_PERSIST_INTERVAL,
                                )
                                frame_index += 1

                                if time.time() - prev_time > DEFAULT_UI_REFRESH_INTERVAL_SEC:
                                    frame_display.image(frame_rgb, channels="RGB")
                                    prev_time = time.time()

                            cap.release()
                            finish_session(st.session_state, db_upsert_session)
                            st.session_state.running = False
                            st.success("🛑 Мониторинг входной зоны остановлен.")
                    else:
                        st.info("Нажмите «Запустить мониторинг», чтобы начать обработку видеопотока входной зоны.")

    with info_col:
        render_status_panel(
            st,
            source_mode=source_mode,
            model_name=model_name,
            conf_threshold=conf_threshold,
            notify_conf_threshold=notify_conf_threshold,
            rotation_angle=rotation_angle,
            animal_filter=animal_filter,
            track_classes=track_classes,
            notifications=st.session_state.notifications,
        )

    render_analytics(
        st,
        sessions=st.session_state.sessions,
        events=st.session_state.events,
        notifications=st.session_state.notifications,
        show_advanced=primary_config["show_advanced"],
        model=model,
        employees=employees,
        access_logs=access_logs,
        create_employee_fn=create_employee,
        update_employee_fn=update_employee,
        update_employee_status_fn=update_employee_status,
    )


if __name__ == "__main__":
    main()
