"""Incident synchronization and severity helpers."""

from __future__ import annotations


INCIDENT_SEVERITY_BY_EVENT = {
    "stream_offline": "critical",
    "camera_reconnected": "low",
    "prolonged_presence_near_entry": "high",
    "repeated_entry_attempt": "high",
    "unknown_person_detected": "high",
    "person_entered_entry_zone": "medium",
    "person_left_entry_zone": "low",
    "person_detected_near_entry": "low",
    "rule_count": "medium",
}

INCIDENT_STATUS_OPTIONS = {
    "new": "Новый",
    "acknowledged": "Подтвержден оператором",
    "false_positive": "Ложное срабатывание",
    "resolved": "Обработан",
    "escalated": "Эскалирован",
}


def sync_incidents_from_events(events: list[dict], *, upsert_incident_fn):
    for event in events:
        if not _should_create_incident(event):
            continue
        upsert_incident_fn(
            event_id=event["event_id"],
            source_id=event.get("source_id"),
            zone_name=event.get("access_point_name") or "не задана",
            incident_type=event.get("event_type") or "unknown",
            severity=infer_incident_severity(event),
            status="new",
            confidence=float(event.get("confidence") or 0.0),
            snapshot_path=event.get("snapshot_path") or "",
            operator_comment="",
            employee_id=event.get("employee_id") or event.get("identified_employee_id"),
            identification_status=event.get("identification_status") or "unlinked",
            started_at=float(event.get("timestamp") or 0.0),
        )


def infer_incident_severity(event: dict) -> str:
    return INCIDENT_SEVERITY_BY_EVENT.get(event.get("event_type"), "medium")


def _should_create_incident(event: dict) -> bool:
    event_scope = event.get("event_scope")
    event_type = event.get("event_type") or ""
    if event_scope == "domain":
        return True
    return event_type in {"stream_offline", "camera_reconnected"}
