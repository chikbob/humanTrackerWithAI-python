"""Online monitoring UI with production-first and demo fallback modes."""

from __future__ import annotations

import base64
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
import streamlit.components.v1 as components

from analytics.access import build_monitoring_source_cards
from config.rtc_config import build_rtc_configuration, describe_rtc_environment
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
LOCAL_CAMERA_RESOLUTIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}
SOURCE_KIND_LABELS = {
    "production": "Production source",
    "browser_camera": "Browser live",
    "local_camera": "Local camera",
    "rtsp": "RTSP/IP",
    "stream_url": "HLS/HTTP",
    "usb_camera": "USB camera",
}
STATUS_LABELS = {
    "online": "online",
    "offline": "offline",
    "reconnecting": "reconnecting",
    "live": "live",
    "ready": "ready",
    "standby": "standby",
}


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


def _apply_camera_preferences(cap, *, resolution_label: str):
    width, height = LOCAL_CAMERA_RESOLUTIONS.get(resolution_label, LOCAL_CAMERA_RESOLUTIONS["720p"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def _read_camera_frame(cap, *, max_retries: int = 3):
    for attempt in range(max_retries):
        ret, frame = cap.read()
        if ret:
            return True, frame, attempt
    return False, None, max_retries


def _detect_alt_webrtc_runtime() -> str:
    alt_runtime = Path(".venv311/bin/python")
    return str(alt_runtime) if alt_runtime.exists() else ""


def _record_interactive_frame(session_state, stream_key: str):
    now_ts = time.time()
    frame_counter_key = f"{stream_key}_frame_counter"
    window_started_key = f"{stream_key}_window_started_at"
    last_frame_key = f"{stream_key}_last_frame_at"
    fps_key = f"{stream_key}_fps"

    if not session_state.get(window_started_key):
        session_state[window_started_key] = now_ts
        session_state[frame_counter_key] = 0

    session_state[frame_counter_key] = int(session_state.get(frame_counter_key, 0)) + 1
    elapsed = max(now_ts - float(session_state.get(window_started_key, now_ts)), 1e-6)
    session_state[fps_key] = session_state[frame_counter_key] / elapsed
    session_state[last_frame_key] = now_ts
    return now_ts


def _resolve_stream_mode_label(binding: dict) -> str:
    if binding["kind"] == "production":
        return "Server pipeline"
    if binding["kind"] == "local_camera":
        return "Local device"
    return "Browser live"


def _resolve_source_name(binding: dict, selected_source) -> str:
    if binding["kind"] == "production" and selected_source is not None:
        return selected_source["name"]
    if binding["kind"] == "local_camera":
        return "Встроенная камера MacBook"
    return "Браузерная камера"


def _format_source_kind_label(kind_label: str) -> str:
    return SOURCE_KIND_LABELS.get(kind_label, SOURCE_KIND_LABELS.get(str(kind_label), str(kind_label)))


def _format_status_label(status: str) -> str:
    return STATUS_LABELS.get(status or "", status or "unknown")


def _status_chip_color(status: str) -> str:
    palette = {
        "online": "rgba(16,185,129,.18)",
        "live": "rgba(16,185,129,.18)",
        "ready": "rgba(59,130,246,.18)",
        "reconnecting": "rgba(245,158,11,.18)",
        "offline": "rgba(239,68,68,.18)",
        "standby": "rgba(148,163,184,.18)",
    }
    return palette.get(status, "rgba(148,163,184,.18)")


def _render_fullscreen_frame(frame_placeholder, frame_rgb):
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")
        return
    encoded_image = base64.b64encode(encoded.tobytes()).decode("ascii")
    frame_placeholder.markdown(
        f"""
        <img
            src="data:image/jpeg;base64,{encoded_image}"
            style="position:fixed;inset:0;width:100vw;height:100vh;object-fit:cover;background:#000;z-index:1;"
        />
        """,
        unsafe_allow_html=True,
    )


def _get_request_host(st) -> str:
    context = getattr(st, "context", None)
    headers = getattr(context, "headers", None)
    if not headers:
        return ""
    host = headers.get("host") or headers.get("Host") or ""
    return str(host).strip().lower()


def _is_https_request(st) -> bool:
    context = getattr(st, "context", None)
    headers = getattr(context, "headers", None)
    if not headers:
        return False
    proto = (headers.get("x-forwarded-proto") or headers.get("X-Forwarded-Proto") or "").lower()
    if proto:
        return proto == "https"
    origin = (headers.get("origin") or headers.get("Origin") or "").lower()
    return origin.startswith("https://")


def _is_local_browser_session(st) -> bool:
    host = _get_request_host(st)
    return host.startswith("localhost") or host.startswith("127.0.0.1") or host.startswith("0.0.0.0")


def _is_remote_host_session(st) -> bool:
    host = _get_request_host(st)
    return bool(host) and not _is_local_browser_session(st)


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
    tracker_type: str,
    access_point_name: str,
    session_state,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    preferred_source: str = "",
    preferred_source_id: str = "",
    preferred_source_kind: str = "",
    standalone_mode: bool = False,
    standalone_overlay_enabled: bool = True,
):
    st.markdown(
        """
        <style>
            .video-wall-shell {
                margin-top: .15rem;
            }
            .video-wall-toolbar {
                padding: .75rem .95rem;
                border-radius: 16px;
                background: linear-gradient(180deg, rgba(16,24,35,.96), rgba(11,18,28,.96));
                border: 1px solid rgba(122, 144, 168, 0.14);
                margin-bottom: .8rem;
            }
            .video-wall-note {
                margin-top: .45rem;
                font-size: .84rem;
                color: #8fa6bc;
            }
            .video-link-row {
                display: flex;
                gap: .75rem;
                flex-wrap: wrap;
                margin-bottom: .55rem;
            }
            .video-link-row a {
                display: inline-flex;
                align-items: center;
                padding: .42rem .72rem;
                border-radius: 12px;
                border: 1px solid rgba(88,166,255,.22);
                background: rgba(17, 28, 40, .9);
                color: #d7e8f8;
                text-decoration: none;
                font-size: .86rem;
            }
            .video-link-row a:hover {
                border-color: rgba(88,166,255,.42);
                color: #fff;
            }
            .ops-panel {
                padding: .9rem;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(16,24,35,.98), rgba(11,18,28,.98));
                border: 1px solid rgba(122, 144, 168, 0.16);
            }
            .ops-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: .6rem;
                margin-top: .7rem;
            }
            .ops-card {
                padding: .72rem .75rem;
                border-radius: 14px;
                background: rgba(20, 31, 44, .82);
                border: 1px solid rgba(122, 144, 168, 0.12);
            }
            .ops-label {
                font-size: .72rem;
                text-transform: uppercase;
                letter-spacing: .08em;
                color: #7f9bb5;
            }
            .ops-value {
                margin-top: .3rem;
                font-size: 1rem;
                font-weight: 650;
                color: #edf6ff;
                line-height: 1.25;
            }
            .ops-alert {
                margin-top: .7rem;
                padding: .7rem .8rem;
                border-radius: 14px;
                background: rgba(127, 29, 29, .18);
                border: 1px solid rgba(248, 113, 113, .24);
                color: #fecaca;
                font-size: .84rem;
            }
            .source-status-card {
                border: 1px solid rgba(148,163,184,.14);
                border-radius: 14px;
                padding: .75rem .85rem;
                margin-bottom: .55rem;
                background: rgba(15,23,42,.3);
            }
            .source-status-meta {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: .45rem .8rem;
                margin-top: .55rem;
                font-size: .77rem;
                color: #c6d4e2;
            }
            .source-status-chip {
                padding: .26rem .55rem;
                border-radius: 999px;
                font-size: .74rem;
                border: 1px solid rgba(255,255,255,.08);
                color: #f8fbff;
            }
            .source-tile-shell {
                padding: .65rem .7rem .8rem .7rem;
                border-radius: 16px;
                background: rgba(10, 18, 28, .68);
            }
            .source-tile-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: .65rem;
                margin-bottom: .45rem;
            }
            .source-tile-title {
                font-size: 1rem;
                font-weight: 700;
                color: #eef6ff;
            }
            .source-tile-subtitle {
                margin-top: .12rem;
                font-size: .72rem;
                text-transform: uppercase;
                letter-spacing: .08em;
                color: #8ea4ba;
            }
            .source-meta-row {
                display: flex;
                gap: .45rem;
                flex-wrap: wrap;
                margin-bottom: .55rem;
            }
            .source-meta-pill {
                padding: .22rem .5rem;
                border-radius: 999px;
                font-size: .73rem;
                color: #d8e6f5;
                background: rgba(17, 28, 40, .82);
                border: 1px solid rgba(122, 144, 168, 0.14);
            }
            .compact-caption {
                margin-top: .4rem;
                font-size: .78rem;
                color: #89a0b6;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
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
        stored_selection = session_state.get("monitoring_selected_labels")
        if stored_selection:
            valid_stored = [label for label in stored_selection if label in selectable_labels]
            if valid_stored:
                default_selection = valid_stored
        selected_labels = st.multiselect(
            "Источники онлайн-мониторинга",
            options=selectable_labels,
            default=default_selection,
            key="monitoring_selected_labels",
            help="Production-источники отображаются через snapshots worker. Browser/local режимы активируются как foreground live-source.",
        )
        if not selected_labels:
            selected_labels = default_selection
        selected_bindings = [binding for binding in source_bindings if binding["label"] in selected_labels]
        session_state.monitoring_selected_labels = selected_labels

    layout_mode = "single"
    primary_binding = selected_bindings[0]
    if not standalone_mode:
        control_col1, control_col2, control_col3, control_col4 = st.columns([1.45, 1.0, 0.9, 1.15])
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
            primary_options = [binding["label"] for binding in selected_bindings]
            if session_state.get("monitoring_primary_label") not in primary_options:
                session_state.monitoring_primary_label = primary_options[0]
            primary_label = st.selectbox(
                "Главный источник",
                options=primary_options,
                index=primary_options.index(session_state.monitoring_primary_label),
                key="monitoring_primary_label",
            )
            primary_binding = next(binding for binding in selected_bindings if binding["label"] == primary_label)
        with control_col3:
            st.metric("Одновременных карточек", min(len(selected_bindings), _max_rendered_sources(layout_mode)))
        with control_col4:
            session_state.monitoring_embed_secondary_live = st.toggle(
                "Встроить доп. live",
                value=bool(session_state.get("monitoring_embed_secondary_live", True)),
                help="Дополнительные browser/local источники будут открываться как встроенные standalone-view. Это тяжелее по ресурсам, но позволяет видеть несколько интерактивных камер сразу.",
            )
    selected_bindings = _prioritize_primary_binding(selected_bindings, primary_binding)

    selected_binding = primary_binding
    selected_source = selected_binding["source"]
    selected_status = selected_binding["status"]
    selected_last_frame_at = _resolve_binding_last_frame_at(selected_binding, session_state)
    session_state.monitoring_selected_count = len(selected_bindings)
    session_state.monitoring_primary_source_name = selected_binding["name"]

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
            overlay_enabled=standalone_overlay_enabled,
        )
        return

    render_limit = _max_rendered_sources(layout_mode)
    displayed_bindings = selected_bindings[:render_limit]
    if len(selected_bindings) > render_limit:
        st.info(
            f"Для стабильной работы отображаются первые {render_limit} источника. "
            f"Главный источник всегда имеет приоритет."
        )

    left_col, right_col = st.columns([2.9, 0.72], gap="medium")
    with right_col:
        state_panel_placeholder = st.empty()
        statuses_panel_placeholder = st.empty()

    def _render_status_sidebars():
        with state_panel_placeholder.container(border=True):
            st.markdown("### Панель состояния")
            if selected_binding["kind"] == "production":
                status_fps = round(selected_status.get("fps") or 0.0, 2)
            elif selected_binding["kind"] == "local_camera":
                status_fps = round(session_state.get("local_camera_fps") or 0.0, 2)
            elif selected_binding["kind"] == "browser_camera":
                browser_fps = session_state.get("browser_camera_fps")
                status_fps = round(browser_fps or 0.0, 2) if browser_fps else "—"
            else:
                status_fps = "—"
            st.markdown(
                f"""
                <div class="ops-panel">
                    <div class="source-tile-header">
                        <div>
                            <div class="source-tile-title">{_resolve_source_name(selected_binding, selected_source)}</div>
                            <div class="source-tile-subtitle">{_format_source_kind_label(selected_binding.get('kind_label') or selected_binding.get('kind'))}</div>
                        </div>
                        <span class="source-status-chip" style="background:{_status_chip_color(_resolve_binding_status(selected_binding, session_state, source_card=source_cards.get(selected_binding.get('source_id'), {})))};">
                            {_format_status_label(_resolve_binding_status(selected_binding, session_state, source_card=source_cards.get(selected_binding.get('source_id'), {})))}
                        </span>
                    </div>
                    <div class="ops-grid">
                        <div class="ops-card"><div class="ops-label">FPS</div><div class="ops-value">{status_fps}</div></div>
                        <div class="ops-card"><div class="ops-label">Последний кадр</div><div class="ops-value">{_fmt_ts(_resolve_binding_last_frame_at(selected_binding, session_state))}</div></div>
                        <div class="ops-card"><div class="ops-label">Режим потока</div><div class="ops-value">{_resolve_stream_mode_label(selected_binding)}</div></div>
                        <div class="ops-card"><div class="ops-label">Порог confidence</div><div class="ops-value">{round(conf_threshold, 2)}</div></div>
                        <div class="ops-card"><div class="ops-label">Активная модель</div><div class="ops-value">{model_name}</div></div>
                        <div class="ops-card"><div class="ops-label">Точка прохода</div><div class="ops-value">{access_point_name}</div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if selected_status.get("last_error"):
                st.markdown(f'<div class="ops-alert">{selected_status["last_error"]}</div>', unsafe_allow_html=True)

        with statuses_panel_placeholder.container(border=True):
            st.markdown("### Источники")
            for binding in displayed_bindings:
                card = source_cards.get(binding.get("source_id"), {})
                st.markdown(
                    _render_source_status_badge(
                        title=binding["name"],
                        source_type=binding["kind_label"],
                        status=_resolve_binding_status(binding, session_state, source_card=card),
                        fps=_resolve_binding_fps(binding, session_state, source_card=card),
                        last_frame_at=_fmt_ts(_resolve_binding_last_frame_at(binding, session_state)),
                        recent_event_count=card.get("recent_event_count", 0),
                        error_text=card.get("last_error") or "",
                        live_window_url=_build_live_window_url(binding, overlay_enabled=True),
                    ),
                    unsafe_allow_html=True,
                )

    _render_monitoring_wall_summary(
        st,
        displayed_bindings=displayed_bindings,
        primary_binding=primary_binding,
        worker_statuses=worker_statuses,
        source_cards=source_cards,
    )
    if hasattr(st, "fragment"):
        refresh_interval = "2s" if selected_binding["kind"] in {"browser_camera", "local_camera"} else "5s"

        @st.fragment(run_every=refresh_interval)
        def _render_status_sidebars_fragment():
            _render_status_sidebars()

        _render_status_sidebars_fragment()
    else:
        _render_status_sidebars()

    with left_col:
        with st.container(border=True):
            st.markdown("### Онлайн-мониторинг входной зоны")
            if not standalone_mode:
                live_window_url = _build_live_window_url(selected_binding, overlay_enabled=True)
                clean_window_url = _build_live_window_url(selected_binding, overlay_enabled=False)
                st.markdown(
                    f"""
                    <div class="video-link-row">
                        <a href="{live_window_url}" target="_blank" rel="noopener noreferrer">Открыть отдельное окно</a>
                        <a href="{clean_window_url}" target="_blank" rel="noopener noreferrer">Чистое окно без overlay</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="video-wall-note">Production-источники читаются из worker runtime. Browser/local live работает как foreground-режим и не вмешивается в server-side pipeline.</div>',
                unsafe_allow_html=True,
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
                tracker_type=tracker_type,
                session_state=session_state,
                db_insert_event=db_insert_event,
                db_insert_frame=db_insert_frame,
                db_upsert_session=db_upsert_session,
                layout_mode=layout_mode,
                embed_secondary_live=bool(session_state.get("monitoring_embed_secondary_live", False)),
                status_panel_callback=_render_status_sidebars,
            )

        with st.container(border=True):
            st.markdown("### Последние события")
            latest_rows = [
                {
                    "Время": datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S"),
                    "Тип события": event.get("event_type"),
                    "Источник": event.get("source_name"),
                    "Уверенность": round(event.get("confidence") or 0.0, 3),
                }
                for event in events[:8]
            ]
            st.dataframe(pd.DataFrame(latest_rows), width="stretch", hide_index=True)

    if not hasattr(st, "fragment"):
        _render_status_sidebars()

    if not standalone_mode:
        with st.expander("Демонстрационные сценарии и fallback", expanded=False):
            st.caption(
                "Этот блок содержит загрузку файлов и fallback-режимы для локальной демонстрации. "
                "Production-путь должен использовать worker и server-side источники."
            )
            _render_demo_workspace(
                st,
                model_name=model_name,
                model=model,
                class_meta=class_meta,
                inference_size=inference_size,
                conf_threshold=conf_threshold,
                frame_skip=frame_skip,
                tracker_type=tracker_type,
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


def _resolve_binding_status(binding: dict, session_state, *, source_card: dict | None = None) -> str:
    source_card = source_card or {}
    if binding["kind"] == "local_camera":
        return "live" if session_state.get("local_camera_last_frame_at") else "ready"
    if binding["kind"] == "browser_camera":
        return "live" if session_state.get("browser_camera_last_frame_at") else "ready"
    return source_card.get("status") or binding.get("status", {}).get("status", "offline")


def _resolve_binding_fps(binding: dict, session_state, *, source_card: dict | None = None):
    source_card = source_card or {}
    if binding["kind"] == "local_camera":
        return round(session_state.get("local_camera_fps") or 0.0, 2)
    if binding["kind"] == "browser_camera":
        browser_fps = session_state.get("browser_camera_fps")
        return round(browser_fps or 0.0, 2) if browser_fps else "—"
    return source_card.get("fps")


def _max_rendered_sources(layout_mode: str) -> int:
    if layout_mode == "single":
        return 1
    if layout_mode == "2x2 grid":
        return 4
    if layout_mode == "list":
        return 6
    return 4


def _build_live_window_url(binding: dict, *, overlay_enabled: bool = True) -> str:
    return (
        f"?view=live-window&source={binding['kind']}"
        f"&source_id={binding['source_id']}&source_kind={binding['kind']}"
        f"&overlay={'1' if overlay_enabled else '0'}"
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
    tracker_type: str,
    session_state,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    layout_mode: str,
    embed_secondary_live: bool,
    status_panel_callback=None,
):
    if layout_mode == "auto layout":
        primary = bindings[:1]
        secondary = bindings[1:]
        for binding in primary:
            _render_source_tile(
                st,
                binding=binding,
                source_card=source_cards.get(binding.get("source_id"), {}),
                is_primary=True,
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
                embed_secondary_live=embed_secondary_live,
                status_panel_callback=status_panel_callback,
            )
        if secondary:
            columns = st.columns(2, gap="medium")
            for index, binding in enumerate(secondary[:4]):
                with columns[index % 2]:
                    _render_source_tile(
                        st,
                        binding=binding,
                        source_card=source_cards.get(binding.get("source_id"), {}),
                        is_primary=False,
                        model_name=model_name,
                        model=model,
                        class_meta=class_meta,
                        inference_size=inference_size,
                        conf_threshold=conf_threshold,
                        frame_skip=frame_skip,
                        tracker_type=tracker_type,
                        session_state=session_state,
                        db_insert_event=db_insert_event,
                        db_insert_frame=db_insert_frame,
                        db_upsert_session=db_upsert_session,
                        embed_secondary_live=embed_secondary_live,
                        status_panel_callback=status_panel_callback,
                    )
        return

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
                embed_secondary_live=embed_secondary_live,
                status_panel_callback=status_panel_callback,
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
                    tracker_type=tracker_type,
                    session_state=session_state,
                    db_insert_event=db_insert_event,
                    db_insert_frame=db_insert_frame,
                    db_upsert_session=db_upsert_session,
                    embed_secondary_live=embed_secondary_live,
                    status_panel_callback=status_panel_callback,
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
                    tracker_type=tracker_type,
                    session_state=session_state,
            db_insert_event=db_insert_event,
            db_insert_frame=db_insert_frame,
            db_upsert_session=db_upsert_session,
            embed_secondary_live=embed_secondary_live,
            status_panel_callback=status_panel_callback,
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
    tracker_type: str,
    session_state,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    embed_secondary_live: bool,
    status_panel_callback=None,
):
    with st.container(border=True):
        badge = "Главный источник" if is_primary else "Дополнительный источник"
        source_status = _resolve_binding_status(binding, session_state, source_card=source_card)
        st.markdown(
            f"""
            <div class="source-tile-shell">
                <div class="source-tile-header">
                    <div>
                        <div class="source-tile-title">{binding['name']}</div>
                        <div class="source-tile-subtitle">{_format_source_kind_label(binding['kind_label'])} • {badge}</div>
                    </div>
                    <span class="source-status-chip" style="background:{_status_chip_color(source_status)};">{_format_status_label(source_status)}</span>
                </div>
                <div class="source-meta-row">
                    <span class="source-meta-pill">FPS: {_resolve_binding_fps(binding, session_state, source_card=source_card)}</span>
                    <span class="source-meta-pill">Последний кадр: {_fmt_ts(_resolve_binding_last_frame_at(binding, session_state))}</span>
                    <span class="source-meta-pill">События: {source_card.get("recent_event_count", 0)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if binding["kind"] == "production" and binding["source"] is not None:
            snapshot_path = binding["status"].get("last_snapshot_path")
            if snapshot_path and Path(snapshot_path).exists():
                st.image(snapshot_path, width="stretch")
            else:
                st.info("Worker еще не сохранил snapshot для этого источника.")
            if source_card.get("last_error"):
                st.markdown(f'<div class="compact-caption">Ошибка: {source_card["last_error"]}</div>', unsafe_allow_html=True)
            return

        if not is_primary:
            if embed_secondary_live:
                _render_embedded_live_source(st, binding)
                st.markdown(
                    f'<a href="{_build_live_window_url(binding, overlay_enabled=False)}" target="_blank" rel="noopener noreferrer">Открыть этот источник в чистом окне</a>',
                    unsafe_allow_html=True,
                )
            else:
                _render_embedded_live_source(st, binding)
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
                tracker_type=tracker_type,
                session_state=session_state,
                db_insert_event=db_insert_event,
                db_insert_frame=db_insert_frame,
                db_upsert_session=db_upsert_session,
                status_panel_callback=status_panel_callback,
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
                tracker_type=tracker_type,
                session_state=session_state,
                db_insert_event=db_insert_event,
                db_insert_frame=db_insert_frame,
                db_upsert_session=db_upsert_session,
                standalone_mode=False,
                status_panel_callback=status_panel_callback,
            )
            return


def _render_embedded_live_source(st, binding: dict):
    iframe_src = _build_live_window_url(binding)
    components.html(
        f"""
        <iframe
            src="{iframe_src}"
            style="width:100%;height:420px;border:1px solid rgba(148,163,184,.16);border-radius:16px;background:#000;"
            allow="camera; microphone; autoplay; fullscreen"
        ></iframe>
        """,
        height=430,
    )


def _render_source_status_badge(*, title: str, source_type: str, status: str, fps, last_frame_at: str, recent_event_count: int, error_text: str, live_window_url: str) -> str:
    status_color = _status_chip_color(status)
    error_html = f"<div style='margin-top:6px;color:#fca5a5'>{error_text}</div>" if error_text else ""
    return f"""
    <div class="source-status-card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
            <div>
                <div style="font-weight:650;color:#e2e8f0;">{title}</div>
                <div style="font-size:11px;color:#94a3b8;">{_format_source_kind_label(source_type)}</div>
            </div>
            <span class="source-status-chip" style="background:{status_color};">{_format_status_label(status)}</span>
        </div>
        <div class="source-status-meta">
            <div>FPS: {fps if fps not in (None, '') else '—'}</div>
            <div>События: {recent_event_count}</div>
            <div>Последний кадр: {last_frame_at}</div>
            <div><a href="{live_window_url}" target="_blank" rel="noopener noreferrer">Открыть отдельно</a></div>
        </div>
        {error_html}
    </div>
    """


def _render_monitoring_wall_summary(st, *, displayed_bindings: list[dict], primary_binding: dict, worker_statuses: list[dict], source_cards: dict):
    online_count = sum(1 for status in worker_statuses if status.get("is_connected"))
    interactive_count = sum(1 for binding in displayed_bindings if binding["kind"] in {"browser_camera", "local_camera"})
    total_recent_events = sum((source_cards.get(binding.get("source_id")) or {}).get("recent_event_count", 0) for binding in displayed_bindings)
    st.markdown(
        f"""
        <div class="video-wall-shell" style="display:grid;grid-template-columns:2.2fr .9fr .9fr .95fr;gap:10px;margin-bottom:10px;">
            <div class="video-wall-toolbar" style="padding:12px 14px;border-radius:18px;background:linear-gradient(135deg, rgba(9,18,28,.92), rgba(16,30,44,.92));border:1px solid rgba(88,166,255,.18);margin-bottom:0;">
                <div style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#7f9bb5;">Security Video Wall</div>
                <div style="margin-top:4px;font-size:20px;font-weight:700;color:#eef6ff;">{primary_binding['name']}</div>
                <div style="margin-top:3px;font-size:.84rem;color:#9fb3c8;">Главный источник видеостены • мультиэкранный мониторинг входной зоны</div>
            </div>
            <div class="video-wall-toolbar" style="padding:12px 14px;margin-bottom:0;">
                <div style="font-size:12px;color:#7f9bb5;text-transform:uppercase;">Отображается</div>
                <div style="margin-top:6px;font-size:22px;font-weight:700;color:#eef6ff;">{len(displayed_bindings)}</div>
            </div>
            <div class="video-wall-toolbar" style="padding:12px 14px;margin-bottom:0;">
                <div style="font-size:12px;color:#7f9bb5;text-transform:uppercase;">Online источники</div>
                <div style="margin-top:6px;font-size:22px;font-weight:700;color:#7ee787;">{online_count}</div>
            </div>
            <div class="video-wall-toolbar" style="padding:12px 14px;margin-bottom:0;">
                <div style="font-size:12px;color:#7f9bb5;text-transform:uppercase;">Live / события</div>
                <div style="margin-top:6px;font-size:22px;font-weight:700;color:#eef6ff;">{interactive_count} / {total_recent_events}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    tracker_type: str,
    access_point_name: str,
    session_state,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    overlay_enabled: bool,
):
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"],
            [data-testid="stMainBlockContainer"] {
                background: #000 !important;
            }
            [data-testid="stImage"] img,
            video {
                width: 100vw !important;
                height: 100vh !important;
                object-fit: cover !important;
                background: #000 !important;
                margin: 0 !important;
                display: block !important;
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
    source_name = selected_binding.get("name") or (selected_source["name"] if selected_source is not None else "Источник не задан")
    access_point_label = selected_source.get("location") if selected_source is not None and selected_source.get("location") else access_point_name
    if overlay_enabled:
        st.markdown(
            f"""
            <div class="standalone-overlay">
                <div><strong>{source_name}</strong></div>
                <div>Режим: {selected_binding.get('kind_label', selected_binding.get('kind', 'unknown'))}</div>
                <div>Точка доступа: {access_point_label}</div>
                <div>Последний кадр: {_fmt_ts(selected_last_frame_at)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if selected_binding["kind"] == "production" and selected_source is not None:
        snapshot_path = selected_status.get("last_snapshot_path")
        if snapshot_path and Path(snapshot_path).exists():
            encoded_image = base64.b64encode(Path(snapshot_path).read_bytes()).decode("ascii")
            st.markdown(
                f"""
                <img
                    src="data:image/jpeg;base64,{encoded_image}"
                    style="position:fixed;inset:0;width:100vw;height:100vh;object-fit:cover;background:#000;z-index:1;"
                />
                """,
                unsafe_allow_html=True,
            )
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
            tracker_type=tracker_type,
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
            tracker_type=tracker_type,
            session_state=session_state,
            db_insert_event=db_insert_event,
            db_insert_frame=db_insert_frame,
            db_upsert_session=db_upsert_session,
            standalone_mode=True,
        )


def _render_local_camera_monitor(
    st,
    *,
    model_name: str,
    model,
    class_meta: dict,
    inference_size: int,
    conf_threshold: float,
    frame_skip: int,
    tracker_type: str,
    session_state,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    standalone_mode: bool = False,
    status_panel_callback=None,
):
    """Continuous local-device monitoring for MacBook internal camera and other local webcams."""
    if _is_remote_host_session(st):
        st.error(
            "Режим «Локальная камера MacBook» доступен только при локальном запуске UI на той же машине, "
            "где физически подключена камера."
        )
        st.info(
            "Для удаленного хоста используйте Browser live/WebRTC или production-источник RTSP/HLS/USB на стороне сервера."
        )
        return session_state.get("local_camera_last_frame_at")

    camera_index = 0
    backend_key = "auto"
    resolution_label = "720p"
    mirror_preview = True
    if not standalone_mode:
        setup_col1, setup_col2, setup_col3, setup_col4 = st.columns(4)
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
        with setup_col3:
            resolution_label = st.selectbox(
                "Разрешение",
                options=list(LOCAL_CAMERA_RESOLUTIONS.keys()),
                index=1,
                key="live_local_camera_resolution",
            )
        with setup_col4:
            mirror_preview = st.toggle("Mirror preview", value=True, key="live_local_camera_mirror")
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
            st.dataframe(pd.DataFrame(_probe_camera_backends(int(camera_index))), width="stretch", hide_index=True)

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
    _apply_camera_preferences(cap, resolution_label=resolution_label)
    session_state.local_camera_backend = backend_meta.get("backend_label")
    session_state.local_camera_resolution = resolution_label
    if not standalone_mode:
        st.caption(
            f"Активный backend захвата: {backend_meta.get('backend_label')} • "
            f"Разрешение: {resolution_label}"
        )

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
    consecutive_failures = 0

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
        ret, frame, retry_attempts = _read_camera_frame(cap, max_retries=3)
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= 2:
                cap.release()
                cap, backend_meta = _open_camera_capture(int(camera_index), backend_key=backend_key)
                if cap is None:
                    st.error("Локальная камера потеряла поток и не смогла переподключиться.")
                    break
                _apply_camera_preferences(cap, resolution_label=resolution_label)
                consecutive_failures = 0
            continue
        consecutive_failures = 0
        if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
            frame_index += 1
            continue
        if mirror_preview:
            frame = cv2.flip(frame, 1)
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
            tracker_type=tracker_type,
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
        session_state.local_camera_retry_attempts = retry_attempts
        _record_interactive_frame(session_state, "local_camera")
        if status_panel_callback is not None:
            status_panel_callback()
        if frame_rgb is not None and time.time() - last_ui_draw_ts >= DEFAULT_UI_REFRESH_INTERVAL_SEC:
            if standalone_mode:
                _render_fullscreen_frame(frame_display, frame_rgb)
            else:
                frame_display.image(frame_rgb, channels="RGB", width="stretch")
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
    tracker_type: str,
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
                tracker_type=tracker_type,
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
                    tracker_type=tracker_type,
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
                tracker_type=tracker_type,
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
                    tracker_type=tracker_type,
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
    tracker_type: str,
    session_state,
    db_insert_event,
    db_insert_frame,
    db_upsert_session,
    standalone_mode: bool = False,
    status_panel_callback=None,
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
            tracker_type=tracker_type,
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
        _record_interactive_frame(session_state, "browser_camera")
        if status_panel_callback is not None:
            status_panel_callback()
        finish_session(session_state, db_upsert_session)
        return session_state.get("browser_camera_last_frame_at")

    if method == "Диагностика":
        alt_runtime = _detect_alt_webrtc_runtime()
        rtc_diag = describe_rtc_environment()
        request_mode = "remote" if _is_remote_host_session(st) else "local"
        st.warning(
            "Ниже показана реальная диагностика окружения. Если WebRTC недоступен, браузерный live-поток "
            "в этом окружении не заработает без установки зависимостей и корректного HTTPS/TURN."
        )
        diag_rows = [
            {"Проверка": "Python executable", "Статус": sys.executable},
            {"Проверка": "Browser session", "Статус": request_mode},
            {"Проверка": "HTTPS", "Статус": "yes" if _is_https_request(st) else "no"},
            {"Проверка": "OpenCV", "Статус": cv2.__version__},
            {"Проверка": "PyAV", "Статус": "ok" if av is not None else "missing"},
            {"Проверка": "streamlit-webrtc", "Статус": "ok" if webrtc_streamer is not None else "missing"},
            {"Проверка": "RTC config", "Статус": "configured" if RTC_CONFIG is not None else "empty"},
            {"Проверка": "ICE servers", "Статус": rtc_diag["ice_server_count"]},
            {"Проверка": "STUN", "Статус": "configured" if rtc_diag["has_stun"] else "missing"},
            {"Проверка": "TURN", "Статус": "configured" if rtc_diag["has_turn"] else "missing"},
            {"Проверка": "Browser snapshot", "Статус": "available"},
            {"Проверка": "Alt WebRTC runtime", "Статус": alt_runtime or "not_found"},
        ]
        st.dataframe(pd.DataFrame(diag_rows), width="stretch", hide_index=True)
        st.caption(
            "Для browser live нужны рабочие `streamlit-webrtc` + `av`, HTTPS на удаленном хосте и ICE servers "
            "с TURN для проблемных сетей/NAT."
        )
        return session_state.get("browser_camera_last_frame_at")

    if not WEBRTC_AVAILABLE:
        alt_runtime = _detect_alt_webrtc_runtime()
        st.error(
            "WebRTC live недоступен: в текущем окружении отсутствуют `streamlit-webrtc` и/или `av`. "
            "Ниже доступен browser snapshot, а для непрерывного потока используй локальную OpenCV-камеру."
        )
        st.caption(
            "Техническая причина подтверждена диагностикой окружения: browser live не может стартовать без этих модулей. "
            f"Для полноценного browser live используй `.venv311` и `scripts/run_ui_py311.sh`."
        )
        if alt_runtime:
            st.info(f"Обнаружен готовый runtime с WebRTC: `{alt_runtime}`")
        shot = st.camera_input("Fallback: снимок из браузерной камеры", key=f"browser_camera_fallback_{_safe_stream_key(source_label)}")
        if shot is not None:
            image = Image.open(shot).convert("RGB")
            st.image(image, caption="Получен кадр из браузерной камеры", width="stretch")
            _record_interactive_frame(session_state, "browser_camera")
            if status_panel_callback is not None:
                status_panel_callback()
        return session_state.get("browser_camera_last_frame_at")

    if not standalone_mode:
        rtc_diag = describe_rtc_environment()
        st.caption(
            "Браузерная камера работает как непрерывный live mode через WebRTC. "
            "Если соединение не устанавливается, проверь HTTPS, TURN и сетевое окружение. "
            "Для локального запуска используй UI из `.venv311`."
        )
        if _is_remote_host_session(st) and not _is_https_request(st):
            st.warning("Удаленная browser-сессия работает без HTTPS. Для стабильного WebRTC это критичный риск.")
        if rtc_diag["ice_server_count"] == 0:
            st.error("RTC configuration не содержит ICE servers. Настрой STUN_URLS и TURN_URLS.")
        elif not rtc_diag["has_turn"] and _is_remote_host_session(st):
            st.warning("TURN не настроен. На удаленном хосте это частая причина подвисания ICE-соединения.")

    def _video_frame_callback(frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_rgb = track_and_draw_live(
            frame_bgr,
            model=model,
            conf_threshold=conf_threshold,
            inference_size=inference_size,
            tracker_type=tracker_type,
            class_meta=class_meta,
            animal_filter="всё",
            animal_classes=ANIMAL_CLASSES,
            track_classes=["person"],
            roi_config={"enable_roi": True, "roi_x": 20, "roi_y": 20, "roi_w": 60, "roi_h": 60},
            draw_box_fn=draw_fancy_box,
        )
        _record_interactive_frame(session_state, "browser_camera")
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
    tracker_type: str,
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
        tracker_type=tracker_type,
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
