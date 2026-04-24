"""Zone rule catalog and effective rule-profile helpers."""

from __future__ import annotations

from config.app_config import normalize_source_processing_config


ZONE_RULE_TYPES = {
    "person_in_zone": {
        "label": "Человек в зоне",
        "description": "Фиксирует вход и присутствие человека внутри активной зоны контроля.",
    },
    "loitering": {
        "label": "Длительное присутствие",
        "description": "Срабатывает, если объект остается в зоне дольше заданного времени.",
    },
    "crowding": {
        "label": "Скопление людей",
        "description": "Срабатывает при превышении порога количества объектов в зоне за интервал времени.",
    },
    "track_loss": {
        "label": "Потеря трека",
        "description": "Фиксирует исчезновение объекта из зоны после таймаута.",
    },
    "camera_offline": {
        "label": "Камера недоступна",
        "description": "Используется для эксплуатационного контроля состояния источника.",
    },
    "repeated_presence": {
        "label": "Повторное появление",
        "description": "Сценарий повторного входа или возврата объекта в зону.",
    },
}

RULE_SEVERITY_OPTIONS = {
    "low": "Низкий",
    "medium": "Средний",
    "high": "Высокий",
    "critical": "Критический",
}


def build_effective_rule_profile(*, source: dict, zones: list[dict], zone_rules: list[dict]) -> dict:
    source_config = normalize_source_processing_config(source)
    active_zones = [zone for zone in zones if zone.get("is_active")]
    active_rules = [rule for rule in zone_rules if rule.get("is_active")]

    primary_zone = _resolve_primary_zone(active_zones)
    roi_config = {
        "enable_roi": source_config["enable_roi"],
        "roi_x": source_config["roi_x"],
        "roi_y": source_config["roi_y"],
        "roi_w": source_config["roi_w"],
        "roi_h": source_config["roi_h"],
    }
    if primary_zone is not None:
        roi_config = {
            "enable_roi": True,
            "roi_x": primary_zone["x"],
            "roi_y": primary_zone["y"],
            "roi_w": primary_zone["w"],
            "roi_h": primary_zone["h"],
        }

    event_settings = {
        "rule_count_enabled": source_config["rule_count_enabled"],
        "rule_class": "person",
        "rule_n": source_config["rule_n"],
        "rule_t": source_config["rule_t"],
        "rule_disappear_enabled": source_config["rule_disappear_enabled"],
        "rule_disappear_seconds": source_config["rule_disappear_seconds"],
        "enable_notifications": False,
        "enable_roi": roi_config["enable_roi"],
        "prolonged_presence_seconds": source_config["prolonged_presence_seconds"],
        "repeated_presence_window_seconds": 60,
        "active_rule_types": [rule["rule_type"] for rule in active_rules],
        "camera_offline_enabled": any(rule["rule_type"] == "camera_offline" for rule in active_rules),
    }

    for rule in active_rules:
        if rule["rule_type"] == "crowding":
            event_settings["rule_count_enabled"] = True
            event_settings["rule_n"] = int(rule["threshold_count"])
            event_settings["rule_t"] = int(rule["threshold_seconds"])
        elif rule["rule_type"] == "loitering":
            event_settings["prolonged_presence_seconds"] = int(rule["threshold_seconds"])
        elif rule["rule_type"] == "track_loss":
            event_settings["rule_disappear_enabled"] = True
            event_settings["rule_disappear_seconds"] = int(rule["threshold_seconds"])
        elif rule["rule_type"] == "repeated_presence":
            event_settings["repeated_presence_window_seconds"] = int(rule["threshold_seconds"])

    return {
        "primary_zone": primary_zone,
        "active_zones": active_zones,
        "active_rules": active_rules,
        "roi_config": roi_config,
        "event_settings": event_settings,
    }


def _resolve_primary_zone(zones: list[dict]):
    if not zones:
        return None
    zone_priority = {
        "entry": 0,
        "restricted": 1,
        "service": 2,
        "observation": 3,
        "line_cross": 4,
    }
    return sorted(zones, key=lambda zone: (zone_priority.get(zone.get("zone_type"), 99), zone.get("id", 0)))[0]
