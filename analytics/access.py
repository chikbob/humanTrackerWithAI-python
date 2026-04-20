"""Pure helpers for dashboard and journal analytics."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Optional

import pandas as pd


SUSPICIOUS_EVENT_TYPES = {
    "prolonged_presence_near_entry",
    "unknown_person_detected",
    "repeated_entry_attempt",
    "stream_offline",
}


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
                "snapshot_path": source_status.get("last_snapshot_path"),
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
