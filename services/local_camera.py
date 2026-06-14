"""Local camera access helpers for API-side preview on desktop hosts."""

from __future__ import annotations

import platform
import time

import cv2


def _build_camera_backend_options():
    options = [("auto", None, "Автовыбор")]
    system_name = platform.system().lower()
    if system_name == "windows":
        options.extend(
            [
                ("dshow", getattr(cv2, "CAP_DSHOW", None), "DirectShow"),
                ("msmf", getattr(cv2, "CAP_MSMF", None), "Media Foundation"),
                ("winrt", getattr(cv2, "CAP_WINRT", None), "Windows Runtime"),
            ]
        )
    elif system_name == "darwin":
        options.append(("avfoundation", getattr(cv2, "CAP_AVFOUNDATION", None), "AVFoundation"))
    elif system_name == "linux":
        options.extend(
            [
                ("v4l2", getattr(cv2, "CAP_V4L2", None), "Video4Linux2"),
                ("gstreamer", getattr(cv2, "CAP_GSTREAMER", None), "GStreamer"),
            ]
        )
    options.extend(
        [
            ("any", getattr(cv2, "CAP_ANY", None), "CAP_ANY"),
            ("ffmpeg", getattr(cv2, "CAP_FFMPEG", None), "FFmpeg"),
        ]
    )
    return [(key, api, label) for key, api, label in options if api is not None or key == "auto"]


CAMERA_BACKEND_OPTIONS = _build_camera_backend_options()


def _apply_camera_preferences(cap, width: int, height: int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if platform.system().lower() == "windows":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def _warmup_capture(cap, attempts: int = 12, delay_sec: float = 0.08):
    last_frame = None
    for attempt in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None:
            return True, frame, attempt
        last_frame = frame
        if delay_sec > 0:
            time.sleep(delay_sec)
    return False, last_frame, attempts


def open_local_camera(camera_index: int, *, width: int = 640, height: int = 480):
    attempts: list[str] = []
    for backend_key, api, backend_label in CAMERA_BACKEND_OPTIONS:
        try:
            cap = cv2.VideoCapture(int(camera_index)) if api is None else cv2.VideoCapture(int(camera_index), api)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            attempts.append(f"{backend_label}: exception {type(exc).__name__}")
            continue
        if not cap.isOpened():
            cap.release()
            attempts.append(f"{backend_label}: open_failed")
            continue
        _apply_camera_preferences(cap, width=width, height=height)
        ok, _frame, warmup_attempt = _warmup_capture(cap)
        if ok:
            return cap, {
                "backend_key": backend_key,
                "backend_label": backend_label,
                "warmup_attempt": warmup_attempt,
                "attempts": attempts,
            }
        cap.release()
        attempts.append(f"{backend_label}: no_frames")
    return None, {"backend_label": "—", "attempts": attempts}


def read_local_camera_frame(camera_index: int, *, width: int = 640, height: int = 480):
    cap, meta = open_local_camera(camera_index, width=width, height=height)
    if cap is None:
        return None, meta
    try:
        ok, frame, _warmup_attempt = _warmup_capture(cap, attempts=5, delay_sec=0.05)
        if not ok or frame is None:
            return None, meta | {"attempts": [*meta.get("attempts", []), f"{meta.get('backend_label')}: no_frames_after_open"]}
        return frame, meta
    finally:
        cap.release()


def encode_frame_as_jpeg(frame, *, quality: int = 85) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return encoded.tobytes()
