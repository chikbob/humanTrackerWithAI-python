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
    "active_access_point_id": "",
}


def build_default_source_processing_config() -> dict:
    return SOURCE_PROCESSING_DEFAULTS.copy()


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
