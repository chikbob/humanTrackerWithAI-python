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
DEFAULT_TRACKER_TYPE = "bytetrack"
DEFAULT_IDENTITY_BACKEND = "disabled"
DEFAULT_NOTIFY_MIN_SEVERITY = "high"
DEFAULT_UI_ROLE = "admin"
DEFAULT_UI_ACTOR = "Главный оператор"

SOURCE_PROCESSING_DEFAULTS = {
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
    "tracker_type": DEFAULT_TRACKER_TYPE,
    "identity_backend": DEFAULT_IDENTITY_BACKEND,
    "active_access_point_id": "",
    "notifications_enabled": "0",
    "incident_notify_min_severity": DEFAULT_NOTIFY_MIN_SEVERITY,
    "webhook_enabled": "0",
    "webhook_url": "",
    "telegram_enabled": "0",
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "security_rbac_enabled": "1",
}

TRACKER_OPTIONS = {
    "bytetrack": {"label": "ByteTrack", "tracker_config": "bytetrack.yaml", "use_tracking": True},
    "botsort": {"label": "BoT-SORT", "tracker_config": "botsort.yaml", "use_tracking": True},
    "detect_only": {"label": "Только детекция", "tracker_config": None, "use_tracking": False},
}

IDENTITY_BACKEND_OPTIONS = {
    "disabled": {"label": "Отключено"},
    "face_placeholder": {"label": "Face pipeline placeholder"},
    "reid_placeholder": {"label": "ReID pipeline placeholder"},
}


def build_default_source_processing_config() -> dict:
    return SOURCE_PROCESSING_DEFAULTS.copy()


def normalize_tracker_type(value: str | None) -> str:
    tracker_type = (value or DEFAULT_TRACKER_TYPE).strip().lower()
    return tracker_type if tracker_type in TRACKER_OPTIONS else DEFAULT_TRACKER_TYPE


def build_tracker_runtime_config(tracker_type: str | None) -> dict:
    tracker_key = normalize_tracker_type(tracker_type)
    config = TRACKER_OPTIONS[tracker_key]
    return {
        "tracker_type": tracker_key,
        "tracker_label": config["label"],
        "tracker_config": config["tracker_config"],
        "use_tracking": config["use_tracking"],
    }


def normalize_identity_backend(value: str | None) -> str:
    backend = (value or DEFAULT_IDENTITY_BACKEND).strip().lower()
    return backend if backend in IDENTITY_BACKEND_OPTIONS else DEFAULT_IDENTITY_BACKEND


def build_identity_backend_config(value: str | None) -> dict:
    backend_key = normalize_identity_backend(value)
    config = IDENTITY_BACKEND_OPTIONS[backend_key]
    return {
        "backend": backend_key,
        "label": config["label"],
        "enabled": backend_key != "disabled",
    }


def normalize_source_processing_config(source: dict | None = None) -> dict:
    source = source or {}
    defaults = build_default_source_processing_config()
    normalized = {
        "enable_roi": bool(int(source.get("enable_roi", 1 if defaults["enable_roi"] else 0)))
        if str(source.get("enable_roi", "")).strip() not in {"", "True", "False"}
        else bool(source.get("enable_roi", defaults["enable_roi"])),
        "roi_x": float(source.get("roi_x", defaults["roi_x"])),
        "roi_y": float(source.get("roi_y", defaults["roi_y"])),
        "roi_w": float(source.get("roi_w", defaults["roi_w"])),
        "roi_h": float(source.get("roi_h", defaults["roi_h"])),
        "rule_count_enabled": bool(int(source.get("rule_count_enabled", 1 if defaults["rule_count_enabled"] else 0)))
        if str(source.get("rule_count_enabled", "")).strip() not in {"", "True", "False"}
        else bool(source.get("rule_count_enabled", defaults["rule_count_enabled"])),
        "rule_n": int(source.get("rule_n", defaults["rule_n"])),
        "rule_t": int(source.get("rule_t", defaults["rule_t"])),
        "rule_disappear_enabled": bool(
            int(source.get("rule_disappear_enabled", 1 if defaults["rule_disappear_enabled"] else 0))
        )
        if str(source.get("rule_disappear_enabled", "")).strip() not in {"", "True", "False"}
        else bool(source.get("rule_disappear_enabled", defaults["rule_disappear_enabled"])),
        "rule_disappear_seconds": int(source.get("rule_disappear_seconds", defaults["rule_disappear_seconds"])),
        "prolonged_presence_seconds": int(source.get("prolonged_presence_seconds", defaults["prolonged_presence_seconds"])),
    }
    normalized["roi_x"] = max(0.0, min(100.0, normalized["roi_x"]))
    normalized["roi_y"] = max(0.0, min(100.0, normalized["roi_y"]))
    normalized["roi_w"] = max(1.0, min(100.0 - normalized["roi_x"], normalized["roi_w"]))
    normalized["roi_h"] = max(1.0, min(100.0 - normalized["roi_y"], normalized["roi_h"]))
    normalized["rule_n"] = max(1, normalized["rule_n"])
    normalized["rule_t"] = max(1, normalized["rule_t"])
    normalized["rule_disappear_seconds"] = max(1, normalized["rule_disappear_seconds"])
    normalized["prolonged_presence_seconds"] = max(1, normalized["prolonged_presence_seconds"])
    return normalized
