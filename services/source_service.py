"""Video source management and connectivity helpers."""

from __future__ import annotations

import cv2


def normalize_source_url(source_type: str, source_url: str):
    """Convert source input into a cv2.VideoCapture-compatible value."""
    if source_type == "usb_camera":
        try:
            return int(source_url)
        except ValueError:
            return source_url
    if source_type == "browser_camera":
        return "browser_camera"
    return source_url


def test_video_source_connection(source_type: str, source_url: str, timeout_frames: int = 30) -> tuple[bool, str]:
    """Perform a safe, short connection check for a video source."""
    if source_type == "browser_camera":
        return True, "Источник сохранен. Браузерная камера доступна из раздела онлайн-мониторинга."
    normalized_source = normalize_source_url(source_type, source_url)
    cap = cv2.VideoCapture(normalized_source)
    if not cap.isOpened():
        return False, "Источник не открылся."

    success = False
    for _ in range(timeout_frames):
        ret, _ = cap.read()
        if ret:
            success = True
            break
    cap.release()
    if success:
        return True, "Подключение успешно."
    return False, "Источник открылся, но кадры не поступают."
