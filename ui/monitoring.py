"""Online monitoring UI with production-first and demo fallback modes."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from analytics.access import build_monitoring_source_cards
from config.rtc_config import build_rtc_configuration
from core.detection import track_and_draw_live
from services.events import add_notification, process_disappeared_tracks, register_detection_and_entry_events
from services.state import finish_session, get_current_session, log_frame, start_session
from ui.sidebar import ANIMAL_CLASSES
from utils.performance import DEFAULT_SESSION_PERSIST_INTERVAL, DEFAULT_UI_REFRESH_INTERVAL_SEC
from utils.vision import draw_fancy_box, rotate_frame

try:
    import av
    from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    av = None
    RTCConfiguration = None
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False


RTC_CONFIG = RTCConfiguration(build_rtc_configuration()) if WEBRTC_AVAILABLE and build_rtc_configuration() else None

CAMERA_BACKEND_OPTIONS = [
    ("auto", None, "Автовыбор"),
    ("avfoundation", getattr(cv2, "CAP_AVFOUNDATION", None), "AVFoundation"),
    ("any", getattr(cv2, "CAP_ANY", None), "CAP_ANY"),
]


def _safe_stream_key(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_") or "browser_camera"


def _available_camera_backends():
    return [(key, api, label) for key, api, label in CAMERA_BACKEND_OPTIONS if api is not None or key == "auto"]


def _open_camera_capture(camera_index: int, backend_key: str = "auto"):
    candidates = _available_camera_backends()
    if backend_key != "auto":
        candidates = [row for row in candidates if row[0] == backend_key] or candidates
    tried = []
    for key, api, label in candidates:
        try:
            cap = cv2.VideoCapture(int(camera_index)) if api is None else cv2.VideoCapture(int(camera_index), api)
        except Exception as exc:
            tried.append(f"{label}: exception {type(exc).__name__}")
            continue
        if not cap.isOpened():
            cap.release()
            tried.append(f"{label}: open_failed")
            continue
        ret, _ = cap.read()
        if ret:
            return cap, {"backend_key": key, "backend_label": label, "attempts": tried}
        cap.release()
        tried.append(f"{label}: no_frames")
    return None, {"backend_key": backend_key, "backend_label": "—", "attempts": tried}


def _probe_camera_backends(camera_index: int) -> list[dict]:
    rows = []
    for key, api, label in _available_camera_backends():
        try:
            cap = cv2.VideoCapture(int(camera_index)) if api is None else cv2.VideoCapture(int(camera_index), api)
            opened = cap.isOpened()
            got_frame = False
            if opened:
                got_frame, _ = cap.read()
            cap.release()
            rows.append(
                {
                    "backend": label,
                    "opened": "да" if opened else "нет",
                    "frame": "да" if got_frame else "нет",
                    "status": "ok" if opened and got_frame else "failed",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "backend": label,
                    "opened": "ошибка",
                    "frame": "ошибка",
                    "status": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


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
    preferred_source: str = "",
    preferred_source_id: str = "",
    preferred_source_kind: str = "",
    standalone_mode: bool = False,
):
    statuses_by_id = {status["source_id"]: status for status in worker_statuses}
    source_bindings = _build_source_bindings(active_sources, statuses_by_id)
    source_cards = {
        card["source_id"]: card
        for card in build_monitoring_source_cards(active_sources, worker_statuses, events)
    }
    selectable_labels = [binding["label"] for binding in source_bindings]
    if not selectable_labels:
        st.warning("Нет доступных источников для мониторинга.")
        return

    standalone_binding = _resolve_standalone_binding(
        source_bindings,
        preferred_source=preferred_source,
        preferred_source_id=preferred_source_id,
        preferred_source_kind=preferred_source_kind,
    )
    if standalone_mode:
        selected_bindings = [standalone_binding] if standalone_binding else [source_bindings[0]]
    else:
        default_selection = _resolve_default_selection(source_bindings, preferred_source, preferred_source_id)
        selected_labels = st.multiselect(
            "Источники онлайн-мониторинга",
            options=selectable_labels,
            default=default_selection,
            help="Production-источники отображаются через snapshots worker. Browser/local режимы активируются как foreground live-source.",
        )
        if not selected_labels:
            selected_labels = default_selection
        selected_bindings = [binding for binding in source_bindings if binding["label"] in selected_labels]

    layout_mode = "single"
    primary_binding = selected_bindings[0]
    if not standalone_mode:
        control_col1, control_col2, control_col3 = st.columns([1.6, 1.0, 1.2])
        with control_col1:
            layout_mode = st.selectbox(
                "Режим отображения",
                options=["single", "2x2 grid", "list", "auto layout"],
                format_func=lambda value: {
                    "single": "Фокус",
                    "2x2 grid": "Сетка 2x2",
                    "list": "Список",
                    "auto layout": "Авто-компоновка",
                }[value],
            )
        with control_col2:
            primary_label = st.selectbox(
                "Главный источник",
                options=[binding["label"] for binding in selected_bindings],
                index=0,
            )
            primary_binding = next(binding for binding in selected_bindings if binding["label"] == primary_label)
        with control_col3:
            st.metric("Одновременных карточек", min(len(selected_bindings), _max_rendered_sources(layout_mode)))
        selected_bindings = _prioritize_primary_binding(selected_bindings, primary_binding)

    selected_binding = primary_binding
    selected_source = selected_binding["source"]
    selected_status = selected_binding["status"]
    selected_last_frame_at = _resolve_binding_last_frame_at(selected_binding, session_state)

    if standalone_mode:
        _render_standalone_live_window(
            st,
            selected_binding=selected_binding,
            selected_source=selected_source,
            selected_status=selected_status,
            selected_last_frame_at=selected_last_frame_at,
            model_name=model_name,
            model=model,
            class_meta=class_meta,
            inference_size=inference_size,
            conf_threshold=conf_threshold,
            access_point_name=access_point_name,
            session_state=session_state,
            db_insert_event=db_insert_event,
            db_insert_frame=db_insert_frame,
            db_upsert_session=db_upsert_session,
        )
        return

    render_limit = _max_rendered_sources(layout_mode)
    displayed_bindings = selected_bindings[:render_limit]
    if len(selected_bindings) > render_limit:
        st.info(
            f"Для стабильной работы отображаются первые {render_limit} источника. "
            f"Главный источник всегда имеет приоритет."
        )

    left_col, right_col = st.columns([2.2, 0.9], gap="large")

    with left_col:
        with st.container(border=True):
            st.subheader("Онлайн-мониторинг входной зоны")
            if not standalone_mode:
                live_window_url = _build_live_window_url(selected_binding)
                st.markdown(
                    f'<a href="{live_window_url}" target="_blank" rel="noopener noreferrer">Открыть live monitoring в отдельном окне</a>',
                    unsafe_allow_html=True,
                )
            st.caption(
                "Production-источники поступают из worker runtime. Browser/local источники доступны как foreground live mode "
                "и не дублируют server-side pipeline."
            )
            _render_source_layout(
                st,
                bindings=displayed_bindings,
                primary_binding=primary_binding,
                source_cards=source_cards,
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
                layout_mode=layout_mode,
            )

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
            if selected_binding["kind"] == "production":
                status_fps = round(selected_status.get("fps") or 0.0, 2)
            elif selected_binding["kind"] == "local_camera":
                status_fps = round(session_state.get("local_camera_fps") or 0.0, 2)
            else:
                status_fps = "—"
            st.metric("FPS", status_fps)
            if selected_binding["kind"] == "production":
                stream_mode_label = "Server pipeline"
            elif selected_binding["kind"] == "local_camera":
                stream_mode_label = "Local device"
            else:
                stream_mode_label = "Browser live"
            st.metric("Режим потока", stream_mode_label)
            st.metric("confidence threshold", round(conf_threshold, 2))
            if selected_binding["kind"] == "production" and selected_source is not None:
                source_name = selected_source["name"]
            elif selected_binding["kind"] == "local_camera":
                source_name = "Встроенная камера MacBook"
            else:
                source_name = "Браузерная камера"
            st.metric("Источник потока", source_name)
            st.metric("Активная модель", model_name)
            st.metric("Точка прохода", access_point_name)
            st.metric("Последний кадр", _fmt_ts(selected_last_frame_at))
            if selected_status.get("last_error"):
                st.error(selected_status["last_error"])
        with st.container(border=True):
            st.subheader("Статусы источников")
            for binding in displayed_bindings:
                card = source_cards.get(binding.get("source_id"), {})
                st.markdown(
                    _render_source_status_badge(
                        title=binding["name"],
                        source_type=binding["kind_label"],
                        status=card.get("status") or binding["status"].get("status", "offline"),
                        fps=card.get("fps"),
                        last_frame_at=_fmt_ts(_resolve_binding_last_frame_at(binding, session_state)),
                        recent_event_count=card.get("recent_event_count", 0),
                        error_text=card.get("last_error") or "",
                        live_window_url=_build_live_window_url(binding),
                    ),
                    unsafe_allow_html=True,
                )

    if not standalone_mode:
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


def _build_source_bindings(active_sources: list[dict], statuses_by_id: dict) -> list[dict]:
    bindings = []
    for source in active_sources:
        binding_kind = "browser_camera" if source["source_type"] == "browser_camera" else "production"
        bindings.append(
            {
                "source_id": source["id"],
                "kind": binding_kind,
                "kind_label": source["source_type"],
                "source": source,
                "status": statuses_by_id.get(source["id"], {}),
                "name": source["name"],
                "label": f"{source['name']} [{source['source_type']}]",
            }
        )
    bindings.append(
        {
            "source_id": "browser-live",
            "kind": "browser_camera",
            "kind_label": "browser_camera",
            "source": None,
            "status": {},
            "name": "Браузерная камера",
            "label": "Браузерная камера",
        }
    )
    bindings.append(
        {
            "source_id": "local-macbook",
            "kind": "local_camera",
            "kind_label": "local_camera",
            "source": None,
            "status": {},
            "name": "Локальная камера MacBook",
            "label": "Локальная камера MacBook",
        }
    )
    return bindings


def _resolve_default_selection(source_bindings: list[dict], preferred_source: str, preferred_source_id: str) -> list[str]:
    if preferred_source_id:
        for binding in source_bindings:
            if str(binding["source_id"]) == str(preferred_source_id):
                return [binding["label"]]
    if preferred_source == "browser_camera":
        for binding in source_bindings:
            if binding["kind"] == "browser_camera":
                return [binding["label"]]
    return [source_bindings[0]["label"]]


def _resolve_standalone_binding(source_bindings: list[dict], *, preferred_source: str, preferred_source_id: str, preferred_source_kind: str):
    if preferred_source_id:
        for binding in source_bindings:
            if str(binding["source_id"]) == str(preferred_source_id):
                return binding
    if preferred_source_kind:
        for binding in source_bindings:
            if binding["kind"] == preferred_source_kind:
                return binding
    if preferred_source:
        for binding in source_bindings:
            if binding["kind"] == preferred_source or binding["label"] == preferred_source:
                return binding
    return source_bindings[0] if source_bindings else None


def _prioritize_primary_binding(bindings: list[dict], primary_binding: dict) -> list[dict]:
    return [primary_binding] + [binding for binding in bindings if binding["label"] != primary_binding["label"]]


def _resolve_binding_last_frame_at(binding: dict, session_state):
    if binding["kind"] == "local_camera":
        return session_state.get("local_camera_last_frame_at")
    if binding["kind"] == "browser_camera":
        return session_state.get("browser_camera_last_frame_at")
    return binding.get("status", {}).get("last_frame_at")


def _max_rendered_sources(layout_mode: str) -> int:
    if layout_mode == "single":
        return 1
    if layout_mode == "2x2 grid":
        return 4
    if layout_mode == "list":
        return 6
    return 4


def _build_live_window_url(binding: dict) -> str:
    return (
        f"?view=live-window&source={binding['kind']}"
        f"&source_id={binding['source_id']}&source_kind={binding['kind']}"
    )


def _render_source_layout(
    st,
    *,
    bindings: list[dict],
    primary_binding: dict,
    source_cards: dict,
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
    layout_mode: str,
):
    if layout_mode == "list":
        for binding in bindings:
            _render_source_tile(
                st,
                binding=binding,
                source_card=source_cards.get(binding.get("source_id"), {}),
                is_primary=binding["label"] == primary_binding["label"],
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
        return

    if layout_mode == "2x2 grid":
        columns = st.columns(2, gap="medium")
        for index, binding in enumerate(bindings[:4]):
            with columns[index % 2]:
                _render_source_tile(
                    st,
                    binding=binding,
                    source_card=source_cards.get(binding.get("source_id"), {}),
                    is_primary=binding["label"] == primary_binding["label"],
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
        return

    for binding in bindings[:1 if layout_mode == "single" else len(bindings)]:
        _render_source_tile(
            st,
            binding=binding,
            source_card=source_cards.get(binding.get("source_id"), {}),
            is_primary=binding["label"] == primary_binding["label"],
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


def _render_source_tile(
    st,
    *,
    binding: dict,
    source_card: dict,
    is_primary: bool,
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
    with st.container(border=True):
        badge = "Главный источник" if is_primary else "Дополнительный источник"
        st.markdown(f"**{binding['name']}**  \n`{binding['kind_label']}` • {badge}")
        if binding["kind"] == "production" and binding["source"] is not None:
            snapshot_path = binding["status"].get("last_snapshot_path")
            if snapshot_path and Path(snapshot_path).exists():
                st.image(snapshot_path, use_container_width=True)
            else:
                st.info("Worker еще не сохранил snapshot для этого источника.")
            if source_card.get("last_error"):
                st.caption(f"Ошибка: {source_card['last_error']}")
            return

        if not is_primary:
            st.info(
                "Интерактивные browser/local источники рендерятся только для главного окна. "
                "Для этого источника используйте «Открыть отдельно»."
            )
            return

        if binding["kind"] == "browser_camera":
            _render_browser_camera_monitor(
                st,
                source_label=binding["name"],
                model_name=model_name,
                model=model,
                class_meta=class_meta,
                inference_size=inference_size,
                conf_threshold=conf_threshold,
                session_state=session_state,
                db_insert_event=db_insert_event,
                db_insert_frame=db_insert_frame,
                db_upsert_session=db_upsert_session,
            )
            return

        if binding["kind"] == "local_camera":
            _render_local_camera_monitor(
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
                standalone_mode=False,
            )
            return


def _render_source_status_badge(*, title: str, source_type: str, status: str, fps, last_frame_at: str, recent_event_count: int, error_text: str, live_window_url: str) -> str:
    status_colors = {
        "online": "#10b981",
        "reconnecting": "#f59e0b",
        "offline": "#ef4444",
    }
    status_color = status_colors.get(status, "#94a3b8")
    error_html = f"<div style='margin-top:6px;color:#fca5a5'>{error_text}</div>" if error_text else ""
    return f"""
    <div style="border:1px solid rgba(148,163,184,.18);border-radius:16px;padding:12px 14px;margin-bottom:10px;background:rgba(15,23,42,.35);">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
            <div>
                <div style="font-weight:600;color:#e2e8f0;">{title}</div>
                <div style="font-size:12px;color:#94a3b8;">{source_type}</div>
            </div>
            <span style="padding:4px 10px;border-radius:999px;background:{status_color};color:#fff;font-size:12px;">{status}</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px;font-size:12px;color:#cbd5e1;">
            <div>FPS: {fps if fps not in (None, '') else '—'}</div>
            <div>События: {recent_event_count}</div>
            <div>Последний кадр: {last_frame_at}</div>
            <div><a href="{live_window_url}" target="_blank" rel="noopener noreferrer">Открыть отдельно</a></div>
        </div>
        {error_html}
    </div>
    """


def _render_standalone_live_window(
    st,
    *,
    selected_binding: dict,
    selected_source,
    selected_status: dict,
    selected_last_frame_at,
    model_name: str,
    model,
    class_meta: dict,
    inference_size: int,
    conf_threshold: float,
    access_point_name: str,
    session_state,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
):
    st.markdown(
        """
        <style>
            .standalone-live-shell {
                width: 100vw;
                height: 100vh;
                overflow: hidden;
                background: #000;
            }
            .standalone-live-shell video {
                width: 100vw !important;
                height: 100vh !important;
                object-fit: cover !important;
                background: #000 !important;
            }
            .standalone-overlay {
                position: fixed;
                top: 18px;
                left: 18px;
                z-index: 50;
                color: #fff;
                background: rgba(0, 0, 0, 0.45);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 14px;
                padding: 10px 14px;
                backdrop-filter: blur(8px);
                font-size: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    source_name = selected_source["name"] if selected_source is not None else "Браузерная камера"
    st.markdown(
        f"""
        <div class="standalone-overlay">
            <div><strong>{source_name}</strong></div>
            <div>Точка доступа: {access_point_name}</div>
            <div>Последний кадр: {_fmt_ts(selected_last_frame_at)}</div>
        </div>
        <div class="standalone-live-shell">
        """,
        unsafe_allow_html=True,
    )
    if selected_binding["kind"] == "production" and selected_source is not None:
        snapshot_path = selected_status.get("last_snapshot_path")
        if snapshot_path and Path(snapshot_path).exists():
            st.image(snapshot_path, use_container_width=True)
        else:
            st.warning("Для production-источника пока нет актуального snapshot от worker.")
    elif selected_binding["kind"] == "local_camera":
        _render_local_camera_monitor(
            st,
            model_name=model_name,
            model=model,
            class_meta=class_meta,
            inference_size=inference_size,
            conf_threshold=conf_threshold,
            frame_skip=0,
            session_state=session_state,
            db_insert_event=db_insert_event,
            db_insert_frame=db_insert_frame,
            db_upsert_session=db_upsert_session,
            standalone_mode=True,
        )
    else:
        _render_browser_camera_monitor(
            st,
            source_label=selected_source["name"] if selected_source is not None else "Браузерная камера",
            model_name=model_name,
            model=model,
            class_meta=class_meta,
            inference_size=inference_size,
            conf_threshold=conf_threshold,
            session_state=session_state,
            db_insert_event=db_insert_event,
            db_insert_frame=db_insert_frame,
            db_upsert_session=db_upsert_session,
            standalone_mode=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_local_camera_monitor(
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
    standalone_mode: bool = False,
):
    """Continuous local-device monitoring for MacBook internal camera and other local webcams."""
    camera_index = 0
    backend_key = "auto"
    if not standalone_mode:
        setup_col1, setup_col2 = st.columns(2)
        with setup_col1:
            camera_index = st.number_input(
                "Индекс локальной камеры",
                min_value=0,
                step=1,
                value=0,
                key="live_local_camera_index",
            )
        with setup_col2:
            backend_labels = {key: label for key, _api, label in _available_camera_backends()}
            backend_key = st.selectbox(
                "Backend захвата",
                options=list(backend_labels.keys()),
                format_func=lambda key: backend_labels[key],
                key="live_local_camera_backend",
            )
    if "local_camera_running" not in session_state:
        session_state.local_camera_running = False

    if standalone_mode:
        session_state.local_camera_running = True
    else:
        control_col1, control_col2 = st.columns(2)
        with control_col1:
            if st.button("Запустить локальную камеру", key="live_local_camera_start"):
                session_state.local_camera_running = True
        with control_col2:
            if st.button("Остановить локальную камеру", key="live_local_camera_stop"):
                session_state.local_camera_running = False
        with st.expander("Диагностика локальной камеры", expanded=False):
            st.dataframe(pd.DataFrame(_probe_camera_backends(int(camera_index))), use_container_width=True, hide_index=True)

    if not session_state.local_camera_running:
        st.info("Запустите локальную камеру для непрерывного мониторинга.")
        return session_state.get("local_camera_last_frame_at")

    cap, backend_meta = _open_camera_capture(int(camera_index), backend_key=backend_key)
    if cap is None:
        attempt_text = ", ".join(backend_meta.get("attempts") or []) or "нет подробностей"
        st.error(
            "Не удалось открыть встроенную камеру устройства через OpenCV. "
            f"Попытки: {attempt_text}"
        )
        session_state.local_camera_running = False
        return session_state.get("local_camera_last_frame_at")
    session_state.local_camera_backend = backend_meta.get("backend_label")
    if not standalone_mode:
        st.caption(f"Активный backend захвата: {backend_meta.get('backend_label')}")

    frame_display = st.empty()
    start_session(
        session_state,
        db_upsert_session,
        model_name=model_name,
        source_type="webcam",
        source_path=f"camera:{camera_index}",
        animal_filter="всё",
        track_classes=["person"],
        rotation_angle=0,
    )
    frame_index = 0
    last_ui_draw_ts = 0.0
    frame_counter = 0
    fps_window_start = time.time()

    def register_event_pipeline(*, frame_index: int, detection: dict, source_type: str, session: dict):
        register_detection_and_entry_events(
            session_state,
            db_insert_event,
            session=session,
            frame_index=frame_index,
            detection=detection,
            source_type=source_type,
            settings={
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
            notify_callback=lambda _text: add_notification(session_state, _text, enabled=False, toast_callback=None),
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
            rule_disappear_enabled=True,
            rule_disappear_seconds=5,
            enable_notifications=False,
            notify_callback=lambda _text: None,
            default_access_point_id=None,
        )

    while session_state.local_camera_running:
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
            rotation_angle=0,
            register_event_pipeline=register_event_pipeline,
            process_disappeared=process_disappeared,
            draw_now=time.time() - last_ui_draw_ts >= DEFAULT_UI_REFRESH_INTERVAL_SEC,
        )
        frame_counter += 1
        elapsed = time.time() - fps_window_start
        if elapsed > 0:
            session_state.local_camera_fps = frame_counter / elapsed
        session_state.local_camera_last_frame_at = time.time()
        if frame_rgb is not None and time.time() - last_ui_draw_ts >= DEFAULT_UI_REFRESH_INTERVAL_SEC:
            frame_display.image(frame_rgb, channels="RGB", use_container_width=True)
            last_ui_draw_ts = time.time()
        frame_index += 1

    cap.release()
    finish_session(session_state, db_upsert_session)
    if not standalone_mode:
        session_state.local_camera_running = False
    return session_state.get("local_camera_last_frame_at")


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


def _render_browser_camera_monitor(
    st,
    *,
    source_label: str,
    model_name: str,
    model,
    class_meta: dict,
    inference_size: int,
    conf_threshold: float,
    session_state,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    standalone_mode: bool = False,
):
    """Render browser camera via all realistic methods available in the current environment."""
    methods = ["WebRTC live", "Browser snapshot", "Диагностика"]
    if not standalone_mode:
        method = st.radio(
            "Метод браузерной камеры",
            options=methods,
            horizontal=True,
            key=f"browser_camera_method_{_safe_stream_key(source_label)}",
        )
    else:
        method = "WebRTC live"

    if method == "Browser snapshot":
        shot = st.camera_input("Кадр из браузерной камеры", key=f"browser_camera_snapshot_{_safe_stream_key(source_label)}")
        if shot is None:
            st.info("Разреши доступ к камере в браузере и сделай кадр для проверки этого метода.")
            return session_state.get("browser_camera_last_frame_at")
        start_session(
            session_state,
            db_upsert_session,
            model_name=model_name,
            source_type="webcam_browser",
            source_path="browser_camera_snapshot",
            animal_filter="всё",
            track_classes=["person"],
            rotation_angle=0,
        )
        image = Image.open(shot).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame_display = st.empty()
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
            rotation_angle=0,
            register_event_pipeline=lambda **kwargs: register_detection_and_entry_events(
                session_state,
                db_insert_event,
                session=kwargs["session"],
                frame_index=kwargs["frame_index"],
                detection=kwargs["detection"],
                source_type=kwargs["source_type"],
                settings={
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
                notify_callback=lambda _text: add_notification(session_state, _text, enabled=False, toast_callback=None),
            ),
            process_disappeared=lambda **kwargs: process_disappeared_tracks(
                session_state,
                db_insert_event,
                session=kwargs["session"],
                frame_index=kwargs["frame_index"],
                source_type=kwargs["source_type"],
                frame_width=kwargs["frame_width"],
                frame_height=kwargs["frame_height"],
                rule_disappear_enabled=True,
                rule_disappear_seconds=5,
                enable_notifications=False,
                notify_callback=lambda _text: None,
                default_access_point_id=None,
            ),
            draw_now=True,
        )
        session_state.browser_camera_last_frame_at = time.time()
        finish_session(session_state, db_upsert_session)
        return session_state.get("browser_camera_last_frame_at")

    if method == "Диагностика":
        st.warning(
            "Ниже показана реальная диагностика окружения. Если WebRTC недоступен, браузерный live-поток "
            "в этом окружении не заработает без установки зависимостей и корректного HTTPS/TURN."
        )
        diag_rows = [
            {"Проверка": "Python executable", "Статус": sys.executable},
            {"Проверка": "OpenCV", "Статус": cv2.__version__},
            {"Проверка": "PyAV", "Статус": "ok" if av is not None else "missing"},
            {"Проверка": "streamlit-webrtc", "Статус": "ok" if webrtc_streamer is not None else "missing"},
            {"Проверка": "RTC config", "Статус": "configured" if RTC_CONFIG is not None else "empty"},
            {"Проверка": "Browser snapshot", "Статус": "available"},
        ]
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Для локального all-time мониторинга на MacBook используй «Локальная камера MacBook». "
            "Для браузерного live здесь нужен рабочий `streamlit-webrtc` + `av`."
        )
        return session_state.get("browser_camera_last_frame_at")

    if not WEBRTC_AVAILABLE:
        st.error(
            "WebRTC live недоступен: в текущем окружении отсутствуют `streamlit-webrtc` и/или `av`. "
            "Ниже доступен browser snapshot, а для непрерывного потока используй локальную OpenCV-камеру."
        )
        st.caption(
            "Техническая причина подтверждена диагностикой окружения: browser live не может стартовать без этих модулей."
        )
        shot = st.camera_input("Fallback: снимок из браузерной камеры", key=f"browser_camera_fallback_{_safe_stream_key(source_label)}")
        if shot is not None:
            image = Image.open(shot).convert("RGB")
            st.image(image, caption="Получен кадр из браузерной камеры", use_container_width=True)
            session_state.browser_camera_last_frame_at = time.time()
        return session_state.get("browser_camera_last_frame_at")

    if not standalone_mode:
        st.caption(
            "Браузерная камера работает как непрерывный live mode через WebRTC. "
            "Если соединение не устанавливается, проверь HTTPS, TURN и сетевое окружение."
        )

    def _video_frame_callback(frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_rgb = track_and_draw_live(
            frame_bgr,
            model=model,
            conf_threshold=conf_threshold,
            inference_size=inference_size,
            class_meta=class_meta,
            animal_filter="всё",
            animal_classes=ANIMAL_CLASSES,
            track_classes=["person"],
            roi_config={"enable_roi": True, "roi_x": 20, "roi_y": 20, "roi_w": 60, "roi_h": 60},
            draw_box_fn=draw_fancy_box,
        )
        session_state.browser_camera_last_frame_at = time.time()
        return av.VideoFrame.from_ndarray(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), format="bgr24")

    webrtc_streamer(
        key=f"browser_camera_stream_{_safe_stream_key(source_label)}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=_video_frame_callback,
        async_processing=True,
    )
    if not standalone_mode:
        st.info(
            "Если видите долгую установку соединения, это не snapshot-проблема, а WebRTC/TURN-сценарий. "
            "Для удаленного запуска используйте подготовленный coturn-контур."
        )
    return session_state.get("browser_camera_last_frame_at")


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
