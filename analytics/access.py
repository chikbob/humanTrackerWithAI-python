"""Pure helpers for dashboard and journal analytics."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Optional

import pandas as pd

from services.source_health import normalize_source_runtime_status


SUSPICIOUS_EVENT_TYPES = {
    "prolonged_presence_near_entry",
    "unknown_person_detected",
    "repeated_entry_attempt",
    "stream_offline",
}

ACTIVE_INCIDENT_STATUSES = {"new", "acknowledged", "in_progress", "on_hold", "escalated"}
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
INCIDENT_STATUS_ORDER = ("new", "acknowledged", "in_progress", "on_hold", "escalated", "resolved", "false_positive")


def infer_worker_source_id(session_id: Optional[str]) -> Optional[int]:
    if not session_id or not session_id.startswith("worker-"):
        return None
    parts = session_id.split("-")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def enrich_event_rows(events: list[dict], sources: list[dict], statuses: list[dict]) -> list[dict]:
    sources_by_id = {source["id"]: source for source in sources}
    statuses_by_source = {status["source_id"]: status for status in statuses}
    rows = []
    for event in events:
        source_id = infer_worker_source_id(event.get("session_id"))
        source_row = sources_by_id.get(source_id)
        source_status = statuses_by_source.get(source_id, {})
        if source_row is not None:
            source_name = source_row["name"]
            source_location = source_row.get("location") or ""
        else:
            source_name = event.get("source_path") or event.get("source_type") or "не указан"
            source_location = ""
        rows.append(
            {
                **event,
                "source_id": source_id,
                "source_name": source_name,
                "source_location": source_location,
                "snapshot_path": event.get("snapshot_path") or source_status.get("last_snapshot_path"),
                "evidence_clip_path": event.get("evidence_clip_path") or "",
                "evidence_retention_until": event.get("evidence_retention_until"),
                "is_suspicious": event.get("event_type") in SUSPICIOUS_EVENT_TYPES,
            }
        )
    return rows


def build_kpi_summary(events: list[dict], statuses: list[dict]) -> dict:
    now = datetime.now()
    today_rows = [
        event for event in events if event.get("timestamp") and datetime.fromtimestamp(event["timestamp"]).date() == now.date()
    ]
    detections_today = sum(1 for event in today_rows if event.get("event_type") == "person_detected_near_entry")
    entries_today = sum(1 for event in today_rows if event.get("event_type") == "person_entered_entry_zone")
    suspicious_today = sum(1 for event in today_rows if event.get("event_type") in SUSPICIOUS_EVENT_TYPES)
    online_cameras = sum(1 for status in statuses if status.get("is_connected"))
    return {
        "detections_today": detections_today,
        "entries_today": entries_today,
        "suspicious_today": suspicious_today,
        "online_cameras": online_cameras,
        "total_events_today": len(today_rows),
    }


def build_time_distribution(events: list[dict]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=["hour", "count"])
    rows = [{"hour": datetime.fromtimestamp(event["timestamp"]).strftime("%H:00"), "count": 1} for event in events]
    df = pd.DataFrame(rows)
    return df.groupby("hour", as_index=False)["count"].sum()


def build_daily_entries(events: list[dict], days: int = 7) -> pd.DataFrame:
    now = datetime.now()
    filtered = []
    for event in events:
        if event.get("event_type") != "person_entered_entry_zone":
            continue
        event_dt = datetime.fromtimestamp(event["timestamp"])
        if (now.date() - event_dt.date()).days <= days:
            filtered.append({"date": event_dt.strftime("%Y-%m-%d"), "count": 1})
    if not filtered:
        return pd.DataFrame(columns=["date", "count"])
    df = pd.DataFrame(filtered)
    return df.groupby("date", as_index=False)["count"].sum()


def build_top_event_types(events: list[dict], limit: int = 8) -> pd.DataFrame:
    counter = Counter(event.get("event_type") or "unknown" for event in events)
    rows = [{"event_type": event_type, "count": count} for event_type, count in counter.most_common(limit)]
    return pd.DataFrame(rows)


def build_access_point_distribution(events: list[dict]) -> pd.DataFrame:
    counter = Counter(event.get("access_point_name") or "не задана" for event in events if event.get("event_scope") == "domain")
    rows = [{"access_point": access_point, "count": count} for access_point, count in counter.items()]
    return pd.DataFrame(rows)


def build_offline_source_summary(events: list[dict]) -> pd.DataFrame:
    counter = Counter(event.get("source_name") or "не указан" for event in events if event.get("event_type") == "stream_offline")
    rows = [{"source_name": source_name, "offline_events": count} for source_name, count in counter.items()]
    return pd.DataFrame(rows)


def build_source_status_rows(sources: list[dict], statuses: list[dict]) -> list[dict]:
    statuses_by_id = {status["source_id"]: status for status in statuses}
    rows = []
    for source in sources:
        status = normalize_source_runtime_status(statuses_by_id.get(source["id"], {}))
        rows.append(
            {
                "source_id": source["id"],
                "Источник": source["name"],
                "Тип": source["source_type"],
                "Активен": "да" if source.get("is_active") else "нет",
                "Статус": status["health_status"],
                "Соединение": status["connection_status"],
                "FPS": round(status.get("fps") or 0.0, 2),
                "Heartbeat": datetime.fromtimestamp(statuses_by_id.get(source["id"], {}).get("last_heartbeat")).strftime("%H:%M:%S")
                if statuses_by_id.get(source["id"], {}).get("last_heartbeat")
                else "—",
            }
        )
    return rows


def build_incident_status_summary(incidents: list[dict]) -> dict:
    active_incidents = [incident for incident in incidents if incident.get("status") in ACTIVE_INCIDENT_STATUSES]
    critical_incidents = [incident for incident in incidents if incident.get("severity") == "critical"]
    resolved_incidents = [incident for incident in incidents if incident.get("status") == "resolved"]
    false_positive_incidents = [incident for incident in incidents if incident.get("status") == "false_positive"]
    reaction_seconds = []
    for incident in incidents:
        if incident.get("status") not in {"acknowledged", "resolved", "escalated"}:
            continue
        started_at = incident.get("started_at")
        updated_at = incident.get("updated_at")
        if started_at and updated_at and float(updated_at) >= float(started_at):
            reaction_seconds.append(float(updated_at) - float(started_at))
    return {
        "total": len(incidents),
        "active": len(active_incidents),
        "critical": len(critical_incidents),
        "resolved": len(resolved_incidents),
        "false_positive": len(false_positive_incidents),
        "zones_under_alert": len({incident.get("zone_name") or "не задана" for incident in active_incidents}),
        "mean_response_minutes": round(sum(reaction_seconds) / len(reaction_seconds) / 60, 1) if reaction_seconds else 0.0,
        "unassigned_active": sum(1 for incident in active_incidents if not (incident.get("assigned_to") or "").strip()),
    }


def build_incident_severity_distribution(incidents: list[dict]) -> pd.DataFrame:
    counter = Counter(incident.get("severity") or "unknown" for incident in incidents)
    rows = [
        {"severity": severity, "count": count}
        for severity, count in sorted(counter.items(), key=lambda item: SEVERITY_ORDER.get(item[0], 99))
    ]
    return pd.DataFrame(rows)


def build_incident_queue_breakdown(incidents: list[dict]) -> list[dict]:
    active_incidents = [incident for incident in incidents if incident.get("status") in ACTIVE_INCIDENT_STATUSES]
    counter = Counter(incident.get("status") or "unknown" for incident in active_incidents)
    rows = [
        {"Статус": status, "Кейсов": counter.get(status, 0)}
        for status in ("new", "acknowledged", "in_progress", "on_hold", "escalated")
    ]
    return rows


def build_incident_sla_summary(incidents: list[dict]) -> dict:
    now_ts = datetime.now().timestamp()
    active_incidents = [incident for incident in incidents if incident.get("status") in ACTIVE_INCIDENT_STATUSES]
    fresh = 0
    due_soon = 0
    overdue = 0
    oldest_age_minutes = 0.0
    for incident in active_incidents:
        age_minutes = _incident_age_minutes(incident, now_ts=now_ts)
        if age_minutes > 15:
            overdue += 1
        elif age_minutes >= 10:
            due_soon += 1
        else:
            fresh += 1
        oldest_age_minutes = max(oldest_age_minutes, age_minutes)
    return {
        "active": len(active_incidents),
        "fresh": fresh,
        "due_soon": due_soon,
        "overdue": overdue,
        "oldest_age_minutes": round(oldest_age_minutes, 1),
    }


def build_incident_age_buckets(incidents: list[dict]) -> list[dict]:
    now_ts = datetime.now().timestamp()
    buckets = {
        "0-5 мин": 0,
        "5-15 мин": 0,
        "15-30 мин": 0,
        "30+ мин": 0,
    }
    for incident in incidents:
        if incident.get("status") not in ACTIVE_INCIDENT_STATUSES:
            continue
        age_minutes = _incident_age_minutes(incident, now_ts=now_ts)
        if age_minutes < 5:
            buckets["0-5 мин"] += 1
        elif age_minutes < 15:
            buckets["5-15 мин"] += 1
        elif age_minutes < 30:
            buckets["15-30 мин"] += 1
        else:
            buckets["30+ мин"] += 1
    return [{"Возраст": bucket, "Кейсов": count} for bucket, count in buckets.items()]


def build_incident_queue_rows(incidents: list[dict], limit: int = 10) -> list[dict]:
    now_ts = datetime.now().timestamp()
    active_incidents = sorted(
        [incident for incident in incidents if incident.get("status") in ACTIVE_INCIDENT_STATUSES],
        key=lambda incident: (
            SEVERITY_ORDER.get(incident.get("severity"), 99),
            0 if not (incident.get("assigned_to") or "").strip() else 1,
            -(float(incident.get("started_at") or 0.0)),
        ),
    )
    rows = []
    for incident in active_incidents[:limit]:
        age_minutes = _incident_age_minutes(incident, now_ts=now_ts)
        rows.append(
            {
                "ID": incident.get("id"),
                "Серьезность": incident.get("severity") or "—",
                "Инцидент": incident.get("incident_type") or "—",
                "Источник": incident.get("source_name") or incident.get("source_id") or "—",
                "Зона": incident.get("zone_name") or "не задана",
                "Статус": incident.get("status") or "—",
                "Owner": (incident.get("assigned_to") or "").strip() or "не назначен",
                "Возраст, мин": age_minutes,
                "SLA": "overdue" if age_minutes > 15 else "ok",
                "Время": _fmt_ts_short(incident.get("started_at")),
            }
        )
    return rows


def build_operator_workload_rows(incidents: list[dict], *, limit: int = 6) -> list[dict]:
    now_ts = datetime.now().timestamp()
    buckets = {}
    for incident in incidents:
        if incident.get("status") not in ACTIVE_INCIDENT_STATUSES:
            continue
        owner = (incident.get("assigned_to") or "").strip() or "не назначен"
        bucket = buckets.setdefault(
            owner,
            {
                "Ответственный": owner,
                "Активных кейсов": 0,
                "Critical": 0,
                "Overdue": 0,
                "Последний кейс": 0.0,
            },
        )
        bucket["Активных кейсов"] += 1
        if incident.get("severity") == "critical":
            bucket["Critical"] += 1
        started_at = float(incident.get("started_at") or 0.0)
        if started_at:
            if _incident_age_minutes(incident, now_ts=now_ts) > 15:
                bucket["Overdue"] += 1
            bucket["Последний кейс"] = max(bucket["Последний кейс"], started_at)
    rows = sorted(
        buckets.values(),
        key=lambda row: (
            0 if row["Ответственный"] == "не назначен" else 1,
            -row["Overdue"],
            -row["Critical"],
            -row["Активных кейсов"],
            -row["Последний кейс"],
        ),
    )[:limit]
    for row in rows:
        row["Последний кейс"] = _fmt_ts_short(row["Последний кейс"])
    return rows


def build_zone_risk_rows(incidents: list[dict], limit: int = 6) -> list[dict]:
    grouped = {}
    for incident in incidents:
        zone_name = incident.get("zone_name") or "не задана"
        bucket = grouped.setdefault(
            zone_name,
            {
                "Зона": zone_name,
                "Всего инцидентов": 0,
                "Активных": 0,
                "Критических": 0,
                "High+": 0,
                "Последний инцидент": None,
            },
        )
        bucket["Всего инцидентов"] += 1
        if incident.get("status") in ACTIVE_INCIDENT_STATUSES:
            bucket["Активных"] += 1
        if incident.get("severity") == "critical":
            bucket["Критических"] += 1
        if incident.get("severity") in {"critical", "high"}:
            bucket["High+"] += 1
        started_at = incident.get("started_at")
        if started_at and (bucket["Последний инцидент"] is None or float(started_at) > float(bucket["Последний инцидент"])):
            bucket["Последний инцидент"] = float(started_at)

    rows = sorted(
        grouped.values(),
        key=lambda row: (-row["Активных"], -row["Критических"], -row["High+"], -float(row["Последний инцидент"] or 0.0)),
    )[:limit]
    for row in rows:
        row["Последний инцидент"] = _fmt_ts_short(row["Последний инцидент"])
    return rows


def build_source_risk_rows(sources: list[dict], statuses: list[dict], incidents: list[dict]) -> list[dict]:
    statuses_by_id = {status["source_id"]: status for status in statuses}
    incidents_by_source = {}
    for incident in incidents:
        source_id = incident.get("source_id")
        if source_id is None:
            continue
        incidents_by_source.setdefault(source_id, []).append(incident)

    rows = []
    for source in sources:
        source_incidents = incidents_by_source.get(source["id"], [])
        active_incidents = [incident for incident in source_incidents if incident.get("status") in ACTIVE_INCIDENT_STATUSES]
        critical_incidents = [incident for incident in source_incidents if incident.get("severity") == "critical"]
        status = normalize_source_runtime_status(statuses_by_id.get(source["id"], {}))
        rows.append(
            {
                "Источник": source["name"],
                "Статус": status["health_status"],
                "Соединение": status["connection_status"],
                "FPS": round(status.get("fps") or 0.0, 2),
                "Активных инцидентов": len(active_incidents),
                "Критических": len(critical_incidents),
                "Reconnect": int(status.get("reconnect_count") or 0),
                "Последняя ошибка": status.get("last_error") or "—",
            }
        )
    priority = {"offline": 0, "degraded": 1, "healthy": 2, "idle": 3}
    rows.sort(key=lambda row: (priority.get(row["Статус"], 99), -row["Активных инцидентов"], -row["Критических"], row["FPS"]))
    return rows


def build_incident_hourly_distribution(incidents: list[dict]) -> pd.DataFrame:
    if not incidents:
        return pd.DataFrame(columns=["hour", "count"])
    rows = []
    for incident in incidents:
        started_at = incident.get("started_at")
        if not started_at:
            continue
        rows.append({"hour": datetime.fromtimestamp(float(started_at)).strftime("%H:00"), "count": 1})
    if not rows:
        return pd.DataFrame(columns=["hour", "count"])
    df = pd.DataFrame(rows)
    return df.groupby("hour", as_index=False)["count"].sum()


def build_camera_health_summary(sources: list[dict], statuses: list[dict]) -> dict:
    statuses_by_id = {status["source_id"]: status for status in statuses}
    offline = 0
    degraded = 0
    healthy = 0
    for source in sources:
        status = normalize_source_runtime_status(statuses_by_id.get(source["id"], {}))
        if status["health_status"] == "offline":
            offline += 1
            continue
        if status["health_status"] == "degraded":
            degraded += 1
        else:
            healthy += 1
    return {"healthy": healthy, "degraded": degraded, "offline": offline}


def _fmt_ts_short(timestamp_value):
    if not timestamp_value:
        return "—"
    return datetime.fromtimestamp(float(timestamp_value)).strftime("%H:%M:%S")


def _incident_age_minutes(incident: dict, *, now_ts: float | None = None) -> float:
    now_ts = now_ts if now_ts is not None else datetime.now().timestamp()
    started_at = float(incident.get("started_at") or 0.0)
    if not started_at:
        return 0.0
    return round(max(0.0, now_ts - started_at) / 60.0, 1)


def build_monitoring_source_cards(
    sources: list[dict],
    statuses: list[dict],
    events: list[dict],
    *,
    interval_seconds: int = 300,
) -> list[dict]:
    now_ts = datetime.now().timestamp()
    statuses_by_id = {status["source_id"]: status for status in statuses}
    recent_event_counter = Counter()
    for event in events:
        source_id = event.get("source_id")
        if source_id is None or not event.get("timestamp"):
            continue
        if now_ts - float(event["timestamp"]) <= interval_seconds:
            recent_event_counter[source_id] += 1

    cards = []
    for source in sources:
        raw_status = statuses_by_id.get(source["id"], {})
        status = normalize_source_runtime_status(raw_status)
        cards.append(
            {
                "source_id": source["id"],
                "name": source["name"],
                "source_type": source["source_type"],
                "status": status["health_status"],
                "connection_status": status["connection_status"],
                "health_status": status["health_status"],
                "is_connected": bool(status.get("is_connected")),
                "fps": round(status.get("fps") or 0.0, 2),
                "last_frame_at": status.get("last_frame_at"),
                "last_snapshot_path": raw_status.get("last_snapshot_path"),
                "last_error": status.get("last_error") or "",
                "recent_event_count": recent_event_counter.get(source["id"], 0),
                "location": source.get("location") or "",
                "is_active": bool(source.get("is_active")),
            }
        )
    return cards
