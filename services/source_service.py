"""Video source management and connectivity helpers."""

from __future__ import annotations

import cv2


SOURCE_TYPE_GUIDES = {
    "rtsp": {
        "label": "RTSP/IP-камера",
        "placeholder": "rtsp://user:password@192.168.1.50:554/stream1",
        "help": "Основной production-вариант для IP-камер и мобильных IP camera apps.",
    },
    "stream_url": {
        "label": "HLS / HTTP-поток",
        "placeholder": "https://example.com/live.m3u8",
        "help": "Подходит для HLS, MJPEG и других HTTP/HTTPS-видеопотоков.",
    },
    "usb_camera": {
        "label": "USB / локальная камера на сервере",
        "placeholder": "0",
        "help": "Обычно это индекс устройства: 0, 1, 2. Камера должна быть подключена к серверу/хосту.",
    },
    "browser_camera": {
        "label": "Камера устройства через браузер",
        "placeholder": "browser_camera",
        "help": "Подходит для камеры ноутбука или телефона через браузер. На проде работает через HTTPS/WebRTC в сессии оператора.",
    },
}


def normalize_source_url(source_type: str, source_url: str):
    """Convert source input into a cv2.VideoCapture-compatible value."""
    source_url = str(source_url or "").strip()
    if source_type == "usb_camera":
        try:
            return int(source_url)
        except ValueError:
            return source_url
    if source_type == "browser_camera":
        return "browser_camera"
    return source_url


def infer_source_type(source_url: str) -> str:
    source_url = str(source_url or "").strip()
    lowered = source_url.lower()
    if lowered == "browser_camera":
        return "browser_camera"
    if lowered.startswith("rtsp://"):
        return "rtsp"
    if lowered.startswith(("http://", "https://")):
        return "stream_url"
    if source_url.isdigit():
        return "usb_camera"
    return "rtsp"


def build_source_setup_hint(source_url: str) -> dict:
    inferred_type = infer_source_type(source_url)
    guide = SOURCE_TYPE_GUIDES[inferred_type]
    return {
        "source_type": inferred_type,
        "label": guide["label"],
        "placeholder": guide["placeholder"],
        "help": guide["help"],
    }


def validate_source_definition(*, name: str, source_type: str, source_url: str) -> tuple[list[str], str]:
    normalized_name = (name or "").strip()
    normalized_url = (source_url or "").strip()
    errors = []
    if not normalized_name:
        errors.append("Укажите понятное название камеры.")
    if source_type == "browser_camera":
        return errors, "browser_camera"
    if not normalized_url:
        errors.append("Укажите адрес потока или индекс устройства.")
        return errors, normalized_url
    if source_type == "rtsp" and not normalized_url.lower().startswith("rtsp://"):
        errors.append("Для RTSP-камеры строка должна начинаться с `rtsp://`.")
    if source_type == "stream_url" and not normalized_url.lower().startswith(("http://", "https://")):
        errors.append("Для HLS/HTTP-источника нужен URL с `http://` или `https://`.")
    if source_type == "usb_camera" and not (normalized_url.isdigit() or normalized_url):
        errors.append("Для локальной камеры укажите индекс устройства, например `0`.")
    return errors, normalized_url


def test_video_source_connection(source_type: str, source_url: str, timeout_frames: int = 30) -> tuple[bool, str]:
    """Perform a safe, short connection check for a video source."""
    if source_type == "browser_camera":
        return True, "Источник сохранен. Браузерная камера доступна из раздела онлайн-мониторинга."
    validation_errors, normalized_source_url = validate_source_definition(
        name="connection_check",
        source_type=source_type,
        source_url=source_url,
    )
    if validation_errors:
        return False, " ".join(validation_errors)
    normalized_source = normalize_source_url(source_type, source_url)
    cap = cv2.VideoCapture(normalized_source)
    if not cap.isOpened():
        if source_type == "usb_camera":
            return False, "Локальная камера не открылась. Проверьте индекс устройства, права доступа и подключение к серверу."
        return False, "Источник не открылся. Проверьте URL, учетные данные, транспорт потока и доступность камеры."

    success = False
    for _ in range(timeout_frames):
        ret, _ = cap.read()
        if ret:
            success = True
            break
    cap.release()
    if success:
        return True, "Подключение успешно: источник открылся и отдал кадры."
    return False, "Источник открылся, но кадры не поступают. Возможен таймаут, неверный транспорт, пустой поток или нестабильная сеть."
