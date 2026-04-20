import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.detection import build_class_meta, detect_and_annotate, detect_and_draw_live, load_model
from db.repository import db_insert_event, db_insert_frame, db_upsert_session, init_db, load_history_from_db
from services.events import add_notification, process_disappeared_tracks, register_detection_and_entry_events
from services.state import finish_session, get_current_session, init_session_state, log_frame, start_session
from ui.analytics import render_analytics, render_status_panel
from ui.page import configure_page
from ui.sidebar import ANIMAL_CLASSES, MODEL_MAP, render_detection_sidebar, render_primary_sidebar
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
    conf_threshold = primary_config["conf_threshold"]
    notify_conf_threshold = primary_config["notify_conf_threshold"]
    enable_notifications = primary_config["enable_notifications"]
    animal_filter = secondary_config["animal_filter"]
    track_classes = secondary_config["track_classes"]
    roi_config = secondary_config["roi_config"]

    event_settings = {
        **secondary_config["event_settings"],
        "enable_notifications": enable_notifications,
        "notify_conf_threshold": notify_conf_threshold,
        "notify_classes": secondary_config["notify_classes"],
        "enable_roi": roi_config["enable_roi"],
        # ROI in this project is interpreted as the enterprise entry zone.
        "default_access_point_id": None,
        "prolonged_presence_seconds": 10,
    }

    def notify(text: str):
        add_notification(
            st.session_state,
            text,
            enabled=enable_notifications,
            toast_callback=st.toast,
        )

    def register_event_pipeline(*, frame_index: int, detection: dict, source_type: str, session: dict):
        # Split CV telemetry from domain events of the entry-zone monitoring model.
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
            st.subheader("🎯 Обработка потока")
            frame_display = st.empty()

            if source_mode == "📁 Загрузить фото":
                uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"], key="img_uploader")
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
                    )
                    finish_session(st.session_state, db_upsert_session)
                    frame_display.image(frame_rgb, channels="RGB")
                    st.success("Изображение обработано.")
                else:
                    st.info("Загрузите файл, чтобы запустить анализ.")

            elif source_mode == "🎞️ Загрузить видео":
                uploaded_video = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"], key="video_uploader")
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
                    st.info("▶️ Обработка видео с трекингом...")
                    frame_index = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = rotate_frame(frame, rotation_angle)
                        frame_rgb, detections_meta, processing_time_ms = detect_and_annotate(
                            frame,
                            frame_index=frame_index,
                            source_type="video",
                            use_tracking=True,
                            model=model,
                            conf_threshold=conf_threshold,
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
                        )
                        frame_index += 1
                        frame_display.image(frame_rgb, channels="RGB")

                    cap.release()
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except PermissionError:
                        pass

                    finish_session(st.session_state, db_upsert_session)
                    st.success("✅ Видео обработано.")
                else:
                    st.info("Загрузите видеофайл, чтобы начать анализ.")

            elif source_mode == "📷 Веб-камера":
                camera_mode = st.radio(
                    "Режим камеры",
                    options=[
                        "Браузерная камера RT (для Streamlit Cloud)",
                        "Браузерная камера (снимок)",
                        "Локальная OpenCV камера (только на вашем ПК)",
                    ],
                    index=0,
                    horizontal=False,
                    key="camera_mode",
                )

                if camera_mode == "Браузерная камера RT (для Streamlit Cloud)":
                    if not WEBRTC_AVAILABLE:
                        st.error("Для realtime-режима установите зависимость streamlit-webrtc.")
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
                            st.success("🛑 Камера остановлена.")

                        st.caption("Нажмите Start в виджете камеры и разрешите доступ в браузере.")

                        def _video_frame_callback(frame):
                            frame_bgr = frame.to_ndarray(format="bgr24")
                            rotated = rotate_frame(frame_bgr, rotation_angle)
                            frame_rgb = detect_and_draw_live(
                                rotated,
                                model=model,
                                conf_threshold=conf_threshold,
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
                            st.info("Нажмите «Запустить камеру», чтобы начать realtime.")

                elif camera_mode == "Браузерная камера (снимок)":
                    st.info("Режим снимка: нажмите кнопку камеры ниже и сделайте фото.")
                    shot = st.camera_input("Снимок с камеры", key="browser_cam_input")
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
                        )
                        finish_session(st.session_state, db_upsert_session)
                        frame_display.image(frame_rgb, channels="RGB")
                        st.success("Снимок обработан.")
                else:
                    camera_index = st.number_input("Номер камеры", min_value=0, step=1, value=0, key="cam_index")
                    run_col1, run_col2 = st.columns(2)
                    with run_col1:
                        start_button = st.button("▶️ Запустить", key="webcam_start")
                    with run_col2:
                        stop_button = st.button("⏹ Остановить", key="webcam_stop")

                    if start_button:
                        st.session_state.running = True
                    if stop_button:
                        st.session_state.running = False

                    if st.session_state.running:
                        cap = cv2.VideoCapture(camera_index)
                        if not cap.isOpened():
                            st.error("❌ Не удалось открыть камеру через OpenCV. На Streamlit Cloud используйте режим «Браузерная камера».")
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
                            st.info("✅ Камера запущена. Идёт трекинг объектов.")
                            prev_time = time.time()
                            frame_index = 0

                            while st.session_state.running:
                                ret, frame = cap.read()
                                if not ret:
                                    st.warning("⚠️ Кадр не получен.")
                                    break
                                frame = rotate_frame(frame, rotation_angle)
                                frame_rgb, detections_meta, processing_time_ms = detect_and_annotate(
                                    frame,
                                    frame_index=frame_index,
                                    source_type="webcam",
                                    use_tracking=True,
                                    model=model,
                                    conf_threshold=conf_threshold,
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
                                )
                                frame_index += 1

                                if time.time() - prev_time > 0.1:
                                    frame_display.image(frame_rgb, channels="RGB")
                                    prev_time = time.time()

                            cap.release()
                            finish_session(st.session_state, db_upsert_session)
                            st.session_state.running = False
                            st.success("🛑 Распознавание остановлено.")
                    else:
                        st.info("Нажмите «Запустить», чтобы начать обработку камеры.")

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
    )


if __name__ == "__main__":
    main()
