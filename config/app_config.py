"""Application defaults for enterprise access monitoring."""

DEFAULT_MODEL_NAME = "yolov8s.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.45
DEFAULT_INFERENCE_SIZE = 512
DEFAULT_FRAME_SKIP = 1
DEFAULT_EVENT_COOLDOWN = 5
DEFAULT_RECONNECT_INTERVAL = 5
DEFAULT_SOURCE_TIMEOUT = 15
DEFAULT_DEBUG_MODE = False
DEFAULT_UI_REFRESH_SECONDS = 2
DEFAULT_EMPLOYEE_SYNC_INTERVAL = 300

SYSTEM_SETTING_DEFAULTS = {
    "confidence_threshold": str(DEFAULT_CONFIDENCE_THRESHOLD),
    "frame_skip": str(DEFAULT_FRAME_SKIP),
    "inference_size": str(DEFAULT_INFERENCE_SIZE),
    "event_cooldown": str(DEFAULT_EVENT_COOLDOWN),
    "reconnect_interval": str(DEFAULT_RECONNECT_INTERVAL),
    "source_timeout": str(DEFAULT_SOURCE_TIMEOUT),
    "employee_sync_interval": str(DEFAULT_EMPLOYEE_SYNC_INTERVAL),
    "debug_mode": "0",
    "model_name": DEFAULT_MODEL_NAME,
    "active_access_point_id": "",
}
