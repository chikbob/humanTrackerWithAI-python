import json
import os
import random
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

from config.app_config import SOURCE_PROCESSING_DEFAULTS, SYSTEM_SETTING_DEFAULTS, normalize_source_processing_config


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(APP_DIR, "monitoring.db")
DB_PATH = os.getenv("MONITORING_DB_PATH", DEFAULT_DB_PATH)

if not os.access(os.path.dirname(DB_PATH) or ".", os.W_OK):
    DB_PATH = "/tmp/monitoring.db"


def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_snapshot_dir() -> str:
    snapshot_dir = os.path.join(os.path.dirname(DB_PATH), "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    return snapshot_dir


def _resolve_day_bounds(day: str | None = None) -> tuple[float, float]:
    if day:
        target_day = datetime.strptime(day, "%Y-%m-%d")
    else:
        target_day = datetime.now()
    start = datetime(target_day.year, target_day.month, target_day.day)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


def build_employee_full_name(last_name: str, first_name: str, middle_name: str = "") -> str:
    parts = [last_name.strip(), first_name.strip(), middle_name.strip()]
    return " ".join(part for part in parts if part)


def split_employee_full_name(full_name: str) -> tuple[str, str, str]:
    parts = [part for part in (full_name or "").strip().split() if part]
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return parts[0], parts[1], " ".join(parts[2:])


def build_employee_display_name(employee: dict | sqlite3.Row | None) -> str:
    if not employee:
        return ""
    last_name = (employee.get("last_name") if isinstance(employee, dict) else employee["last_name"]) if "last_name" in employee.keys() else ""
    first_name = (employee.get("first_name") if isinstance(employee, dict) else employee["first_name"]) if "first_name" in employee.keys() else ""
    middle_name = (employee.get("middle_name") if isinstance(employee, dict) else employee["middle_name"]) if "middle_name" in employee.keys() else ""
    if last_name or first_name or middle_name:
        return build_employee_full_name(last_name or "", first_name or "", middle_name or "")
    if isinstance(employee, dict):
        return employee.get("full_name") or ""
    return employee["full_name"] if "full_name" in employee.keys() else ""


def normalize_identification_status(status: Optional[str]) -> str:
    mapping = {
        None: "unlinked",
        "": "unlinked",
        "not_configured": "unlinked",
        "matched": "linked_from_directory",
        "verified": "linked_from_directory",
        "ambiguous_match": "pending_operator_confirmation",
    }
    return mapping.get(status, status)


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _get_table_columns(conn, table_name: str) -> set[str]:
    if not _table_exists(conn, table_name):
        return set()
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _ensure_columns(conn, table_name: str, columns: list[tuple[str, str]]):
    existing_columns = _get_table_columns(conn, table_name)
    for column_name, column_sql in columns:
        if column_name in existing_columns:
            continue
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")


def _video_source_columns_sql() -> list[tuple[str, str]]:
    return [
        ("enable_roi", f"enable_roi INTEGER DEFAULT {1 if SOURCE_PROCESSING_DEFAULTS['enable_roi'] else 0}"),
        ("roi_x", f"roi_x REAL DEFAULT {SOURCE_PROCESSING_DEFAULTS['roi_x']}"),
        ("roi_y", f"roi_y REAL DEFAULT {SOURCE_PROCESSING_DEFAULTS['roi_y']}"),
        ("roi_w", f"roi_w REAL DEFAULT {SOURCE_PROCESSING_DEFAULTS['roi_w']}"),
        ("roi_h", f"roi_h REAL DEFAULT {SOURCE_PROCESSING_DEFAULTS['roi_h']}"),
        (
            "rule_count_enabled",
            f"rule_count_enabled INTEGER DEFAULT {1 if SOURCE_PROCESSING_DEFAULTS['rule_count_enabled'] else 0}",
        ),
        ("rule_n", f"rule_n INTEGER DEFAULT {SOURCE_PROCESSING_DEFAULTS['rule_n']}"),
        ("rule_t", f"rule_t INTEGER DEFAULT {SOURCE_PROCESSING_DEFAULTS['rule_t']}"),
        (
            "rule_disappear_enabled",
            f"rule_disappear_enabled INTEGER DEFAULT {1 if SOURCE_PROCESSING_DEFAULTS['rule_disappear_enabled'] else 0}",
        ),
        ("rule_disappear_seconds", f"rule_disappear_seconds INTEGER DEFAULT {SOURCE_PROCESSING_DEFAULTS['rule_disappear_seconds']}"),
        ("prolonged_presence_seconds", f"prolonged_presence_seconds INTEGER DEFAULT {SOURCE_PROCESSING_DEFAULTS['prolonged_presence_seconds']}"),
        ("ai_profile_override", "ai_profile_override TEXT"),
        ("conf_threshold_override", "conf_threshold_override REAL"),
        ("inference_size_override", "inference_size_override INTEGER"),
        ("tracker_type_override", "tracker_type_override TEXT"),
        ("incident_threshold_override", "incident_threshold_override REAL"),
    ]


def _video_source_select_columns() -> str:
    return """
        id, name, source_type, source_url, location, is_active, last_seen, description, created_at,
        enable_roi, roi_x, roi_y, roi_w, roi_h,
        rule_count_enabled, rule_n, rule_t, rule_disappear_enabled, rule_disappear_seconds,
        prolonged_presence_seconds, ai_profile_override, conf_threshold_override, inference_size_override,
        tracker_type_override, incident_threshold_override
    """


def _zone_select_columns() -> str:
    return """
        id, source_id, name, zone_type, x, y, w, h, is_active, description, created_at
    """


def _zone_rule_select_columns() -> str:
    return """
        zr.id, zr.zone_id, zr.rule_type, zr.threshold_seconds, zr.threshold_count, zr.cooldown_seconds,
        zr.is_active, zr.severity, zr.description, zr.created_at, z.source_id
    """


def _incident_select_columns() -> str:
    return """
        id, event_id, source_id, zone_name, incident_type, severity, status, confidence,
        snapshot_path, evidence_clip_path, evidence_retention_until, operator_comment,
        assigned_to, acknowledged_at, resolved_at, resolution_code, resolution_notes,
        employee_id, identification_status, started_at, updated_at,
        source_name
    """


def normalize_zone_config(zone: dict | None = None) -> dict:
    zone = zone or {}
    normalized = {
        "name": (zone.get("name") or "").strip() or "Новая зона",
        "zone_type": (zone.get("zone_type") or "observation").strip() or "observation",
        "x": float(zone.get("x", 20)),
        "y": float(zone.get("y", 20)),
        "w": float(zone.get("w", 60)),
        "h": float(zone.get("h", 60)),
        "is_active": bool(zone.get("is_active", True)),
        "description": (zone.get("description") or "").strip(),
    }
    normalized["x"] = max(0.0, min(100.0, normalized["x"]))
    normalized["y"] = max(0.0, min(100.0, normalized["y"]))
    normalized["w"] = max(1.0, min(100.0 - normalized["x"], normalized["w"]))
    normalized["h"] = max(1.0, min(100.0 - normalized["y"], normalized["h"]))
    return normalized


def _normalize_zone_row(row: dict | sqlite3.Row) -> dict:
    zone = dict(row)
    zone.update(normalize_zone_config(zone))
    zone["source_id"] = int(zone["source_id"])
    zone["id"] = int(zone["id"])
    return zone


def normalize_zone_rule_config(rule: dict | None = None) -> dict:
    rule = rule or {}
    normalized = {
        "rule_type": (rule.get("rule_type") or "person_in_zone").strip() or "person_in_zone",
        "threshold_seconds": max(1, int(rule.get("threshold_seconds", 10))),
        "threshold_count": max(1, int(rule.get("threshold_count", 3))),
        "cooldown_seconds": max(0, int(rule.get("cooldown_seconds", 5))),
        "is_active": bool(rule.get("is_active", True)),
        "severity": (rule.get("severity") or "medium").strip() or "medium",
        "description": (rule.get("description") or "").strip(),
    }
    return normalized


def _normalize_zone_rule_row(row: dict | sqlite3.Row) -> dict:
    rule = dict(row)
    rule.update(normalize_zone_rule_config(rule))
    rule["zone_id"] = int(rule["zone_id"])
    rule["id"] = int(rule["id"])
    if rule.get("source_id") is not None:
        rule["source_id"] = int(rule["source_id"])
    return rule


def normalize_incident_config(incident: dict | None = None) -> dict:
    incident = incident or {}
    normalized = {
        "severity": (incident.get("severity") or "medium").strip() or "medium",
        "status": (incident.get("status") or "new").strip() or "new",
        "confidence": float(incident.get("confidence") or 0.0),
        "snapshot_path": (incident.get("snapshot_path") or "").strip(),
        "evidence_clip_path": (incident.get("evidence_clip_path") or "").strip(),
        "evidence_retention_until": float(incident.get("evidence_retention_until"))
        if incident.get("evidence_retention_until") not in {None, ""}
        else None,
        "operator_comment": (incident.get("operator_comment") or "").strip(),
        "assigned_to": (incident.get("assigned_to") or "").strip(),
        "acknowledged_at": float(incident.get("acknowledged_at")) if incident.get("acknowledged_at") not in {None, ""} else None,
        "resolved_at": float(incident.get("resolved_at")) if incident.get("resolved_at") not in {None, ""} else None,
        "resolution_code": (incident.get("resolution_code") or "").strip(),
        "resolution_notes": (incident.get("resolution_notes") or "").strip(),
        "zone_name": (incident.get("zone_name") or "не задана").strip() or "не задана",
        "incident_type": (incident.get("incident_type") or "unknown").strip() or "unknown",
        "identification_status": normalize_identification_status(incident.get("identification_status")),
    }
    return normalized


def _normalize_incident_row(row: dict | sqlite3.Row) -> dict:
    incident = dict(row)
    incident.update(normalize_incident_config(incident))
    incident["id"] = int(incident["id"])
    if incident.get("source_id") is not None:
        incident["source_id"] = int(incident["source_id"])
    if incident.get("employee_id") is not None:
        incident["employee_id"] = int(incident["employee_id"])
    return incident


def _normalize_video_source_row(row: dict | sqlite3.Row) -> dict:
    source = dict(row)
    source.update(normalize_source_processing_config(source))
    source["is_active"] = bool(source.get("is_active"))
    return source


def init_db():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            model TEXT,
            source_type TEXT,
            source_path TEXT,
            animal_filter TEXT,
            class_filter TEXT,
            rotation_angle INTEGER,
            started_at REAL,
            finished_at REAL,
            total_frames INTEGER,
            processed_frames INTEGER,
            events_count INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            frame_index INTEGER,
            timestamp REAL,
            width INTEGER,
            height INTEGER,
            rotation_angle INTEGER,
            processing_time_ms REAL,
            detections_count INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_name TEXT NOT NULL DEFAULT '',
            actor_role TEXT NOT NULL DEFAULT 'admin',
            action TEXT NOT NULL DEFAULT '',
            resource_type TEXT NOT NULL DEFAULT '',
            resource_id TEXT,
            details_json TEXT,
            created_at REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT,
            event_type TEXT,
            source_type TEXT,
            frame_index INTEGER,
            timestamp REAL,
            class_name TEXT,
            confidence REAL,
            track_id TEXT,
            animal_group TEXT,
            is_animal INTEGER,
            roi_inside INTEGER,
            center_x REAL,
            center_y REAL,
            frame_width INTEGER,
            frame_height INTEGER,
            message TEXT,
            event_scope TEXT DEFAULT 'raw',
            snapshot_path TEXT,
            evidence_clip_path TEXT,
            evidence_retention_until REAL,
            incident_score REAL,
            access_log_id INTEGER NULL,
            employee_id INTEGER NULL,
            access_point_id INTEGER NULL,
            identified_employee_id INTEGER NULL,
            identification_confidence REAL,
            identification_status TEXT DEFAULT 'not_configured'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            last_name TEXT,
            first_name TEXT,
            middle_name TEXT,
            employee_number TEXT,
            department TEXT,
            position TEXT,
            status TEXT,
            created_at REAL,
            hire_date REAL,
            external_id TEXT,
            source_system TEXT,
            profile_photo_url TEXT,
            reference_image_url TEXT,
            reference_count INTEGER DEFAULT 0,
            last_synced_at REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS access_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT,
            description TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NULL,
            timestamp REAL NOT NULL,
            access_point_id INTEGER,
            event_type TEXT,
            confidence REAL,
            note TEXT,
            FOREIGN KEY (employee_id) REFERENCES employees(id),
            FOREIGN KEY (access_point_id) REFERENCES access_points(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            access_point_id INTEGER,
            check_in_at REAL NOT NULL,
            check_out_at REAL,
            status TEXT NOT NULL DEFAULT 'on_site',
            source_type TEXT,
            model_name TEXT,
            detection_confidence REAL,
            snapshot_path TEXT,
            note TEXT,
            created_at REAL,
            updated_at REAL,
            FOREIGN KEY (employee_id) REFERENCES employees(id),
            FOREIGN KEY (access_point_id) REFERENCES access_points(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_events (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            access_log_id INTEGER NULL,
            employee_id INTEGER NULL,
            access_point_id INTEGER NULL,
            event_type TEXT,
            source_type TEXT,
            frame_index INTEGER,
            timestamp REAL,
            class_name TEXT,
            confidence REAL,
            track_id TEXT,
            roi_inside INTEGER,
            center_x REAL,
            center_y REAL,
            frame_width INTEGER,
            frame_height INTEGER,
            message TEXT,
            identified_employee_id INTEGER NULL,
            identification_confidence REAL,
            identification_status TEXT DEFAULT 'not_configured',
            FOREIGN KEY (access_log_id) REFERENCES access_logs(id),
            FOREIGN KEY (employee_id) REFERENCES employees(id),
            FOREIGN KEY (access_point_id) REFERENCES access_points(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS video_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_url TEXT NOT NULL,
            location TEXT,
            is_active INTEGER DEFAULT 0,
            last_seen REAL,
            description TEXT,
            created_at REAL,
            enable_roi INTEGER DEFAULT 1,
            roi_x REAL DEFAULT 20,
            roi_y REAL DEFAULT 20,
            roi_w REAL DEFAULT 60,
            roi_h REAL DEFAULT 60,
            rule_count_enabled INTEGER DEFAULT 0,
            rule_n INTEGER DEFAULT 3,
            rule_t INTEGER DEFAULT 10,
            rule_disappear_enabled INTEGER DEFAULT 1,
            rule_disappear_seconds INTEGER DEFAULT 5,
            prolonged_presence_seconds INTEGER DEFAULT 10
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS system_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS employee_sync_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            data_source TEXT,
            sync_status TEXT,
            last_synced_at REAL,
            last_error TEXT,
            cache_mode TEXT,
            updated_at REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS worker_status (
            source_id INTEGER PRIMARY KEY,
            status TEXT,
            is_connected INTEGER DEFAULT 0,
            last_heartbeat REAL,
            last_frame_at REAL,
            fps REAL,
            reconnect_count INTEGER DEFAULT 0,
            last_error TEXT,
            last_snapshot_path TEXT,
            updated_at REAL,
            FOREIGN KEY (source_id) REFERENCES video_sources(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            zone_type TEXT NOT NULL DEFAULT 'observation',
            x REAL DEFAULT 20,
            y REAL DEFAULT 20,
            w REAL DEFAULT 60,
            h REAL DEFAULT 60,
            is_active INTEGER DEFAULT 1,
            description TEXT,
            created_at REAL,
            FOREIGN KEY (source_id) REFERENCES video_sources(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT UNIQUE,
            scenario_name TEXT NOT NULL,
            source_path TEXT,
            notes TEXT,
            created_at REAL,
            completed_at REAL,
            status TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS zone_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id INTEGER NOT NULL,
            rule_type TEXT NOT NULL DEFAULT 'person_in_zone',
            threshold_seconds INTEGER DEFAULT 10,
            threshold_count INTEGER DEFAULT 3,
            cooldown_seconds INTEGER DEFAULT 5,
            is_active INTEGER DEFAULT 1,
            severity TEXT DEFAULT 'medium',
            description TEXT,
            created_at REAL,
            FOREIGN KEY (zone_id) REFERENCES zones(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            source_id INTEGER NULL,
            zone_name TEXT,
            incident_type TEXT NOT NULL,
            severity TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'new',
            confidence REAL DEFAULT 0.0,
            snapshot_path TEXT,
            evidence_clip_path TEXT,
            evidence_retention_until REAL,
            operator_comment TEXT,
            assigned_to TEXT,
            acknowledged_at REAL,
            resolved_at REAL,
            resolution_code TEXT,
            resolution_notes TEXT,
            employee_id INTEGER NULL,
            identification_status TEXT DEFAULT 'unlinked',
            started_at REAL,
            updated_at REAL,
            FOREIGN KEY (source_id) REFERENCES video_sources(id),
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id INTEGER NOT NULL,
            channel TEXT NOT NULL,
            destination TEXT NOT NULL,
            delivery_status TEXT NOT NULL DEFAULT 'pending',
            last_error TEXT,
            sent_at REAL,
            created_at REAL,
            updated_at REAL,
            UNIQUE (incident_id, channel, destination),
            FOREIGN KEY (incident_id) REFERENCES incidents(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            tracker_type TEXT NOT NULL,
            frame_limit INTEGER,
            warmup_frames INTEGER,
            frames_processed INTEGER,
            avg_latency_ms REAL,
            p95_latency_ms REAL,
            avg_fps REAL,
            avg_detections_per_frame REAL,
            tracked_frame_ratio REAL,
            detection_count_total INTEGER,
            metadata_json TEXT,
            created_at REAL,
            FOREIGN KEY (run_id) REFERENCES experiment_runs(id)
        )
        """
    )

    # Migration-safe column checks for databases created by older app versions.
    _ensure_columns(
        conn,
        "video_sources",
        [
            ("name", "name TEXT NOT NULL DEFAULT ''"),
            ("source_type", "source_type TEXT NOT NULL DEFAULT 'rtsp'"),
            ("source_url", "source_url TEXT NOT NULL DEFAULT ''"),
            ("location", "location TEXT"),
            ("is_active", "is_active INTEGER DEFAULT 0"),
            ("last_seen", "last_seen REAL"),
            ("description", "description TEXT"),
            ("created_at", "created_at REAL"),
        ]
        + _video_source_columns_sql(),
    )
    _ensure_columns(
        conn,
        "system_settings",
        [
            ("value", "value TEXT"),
            ("updated_at", "updated_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "employee_sync_state",
        [
            ("data_source", "data_source TEXT"),
            ("sync_status", "sync_status TEXT"),
            ("last_synced_at", "last_synced_at REAL"),
            ("last_error", "last_error TEXT"),
            ("cache_mode", "cache_mode TEXT"),
            ("updated_at", "updated_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "worker_status",
        [
            ("status", "status TEXT"),
            ("is_connected", "is_connected INTEGER DEFAULT 0"),
            ("last_heartbeat", "last_heartbeat REAL"),
            ("last_frame_at", "last_frame_at REAL"),
            ("fps", "fps REAL"),
            ("reconnect_count", "reconnect_count INTEGER DEFAULT 0"),
            ("last_error", "last_error TEXT"),
            ("last_snapshot_path", "last_snapshot_path TEXT"),
            ("updated_at", "updated_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "zones",
        [
            ("source_id", "source_id INTEGER"),
            ("name", "name TEXT NOT NULL DEFAULT 'Новая зона'"),
            ("zone_type", "zone_type TEXT NOT NULL DEFAULT 'observation'"),
            ("x", "x REAL DEFAULT 20"),
            ("y", "y REAL DEFAULT 20"),
            ("w", "w REAL DEFAULT 60"),
            ("h", "h REAL DEFAULT 60"),
            ("is_active", "is_active INTEGER DEFAULT 1"),
            ("description", "description TEXT"),
            ("created_at", "created_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "experiment_runs",
        [
            ("run_key", "run_key TEXT"),
            ("scenario_name", "scenario_name TEXT NOT NULL DEFAULT ''"),
            ("source_path", "source_path TEXT"),
            ("notes", "notes TEXT"),
            ("created_at", "created_at REAL"),
            ("completed_at", "completed_at REAL"),
            ("status", "status TEXT"),
        ],
    )
    _ensure_columns(
        conn,
        "zone_rules",
        [
            ("zone_id", "zone_id INTEGER"),
            ("rule_type", "rule_type TEXT NOT NULL DEFAULT 'person_in_zone'"),
            ("threshold_seconds", "threshold_seconds INTEGER DEFAULT 10"),
            ("threshold_count", "threshold_count INTEGER DEFAULT 3"),
            ("cooldown_seconds", "cooldown_seconds INTEGER DEFAULT 5"),
            ("is_active", "is_active INTEGER DEFAULT 1"),
            ("severity", "severity TEXT DEFAULT 'medium'"),
            ("description", "description TEXT"),
            ("created_at", "created_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "incidents",
        [
            ("event_id", "event_id TEXT"),
            ("source_id", "source_id INTEGER"),
            ("zone_name", "zone_name TEXT"),
            ("incident_type", "incident_type TEXT NOT NULL DEFAULT 'unknown'"),
            ("severity", "severity TEXT DEFAULT 'medium'"),
            ("status", "status TEXT DEFAULT 'new'"),
            ("confidence", "confidence REAL DEFAULT 0.0"),
            ("snapshot_path", "snapshot_path TEXT"),
            ("evidence_clip_path", "evidence_clip_path TEXT"),
            ("evidence_retention_until", "evidence_retention_until REAL"),
            ("operator_comment", "operator_comment TEXT"),
            ("assigned_to", "assigned_to TEXT"),
            ("acknowledged_at", "acknowledged_at REAL"),
            ("resolved_at", "resolved_at REAL"),
            ("resolution_code", "resolution_code TEXT"),
            ("resolution_notes", "resolution_notes TEXT"),
            ("employee_id", "employee_id INTEGER"),
            ("identification_status", "identification_status TEXT DEFAULT 'unlinked'"),
            ("started_at", "started_at REAL"),
            ("updated_at", "updated_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "notification_deliveries",
        [
            ("incident_id", "incident_id INTEGER"),
            ("channel", "channel TEXT NOT NULL DEFAULT 'webhook'"),
            ("destination", "destination TEXT NOT NULL DEFAULT ''"),
            ("delivery_status", "delivery_status TEXT NOT NULL DEFAULT 'pending'"),
            ("last_error", "last_error TEXT"),
            ("sent_at", "sent_at REAL"),
            ("created_at", "created_at REAL"),
            ("updated_at", "updated_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "audit_logs",
        [
            ("actor_name", "actor_name TEXT NOT NULL DEFAULT ''"),
            ("actor_role", "actor_role TEXT NOT NULL DEFAULT 'admin'"),
            ("action", "action TEXT NOT NULL DEFAULT ''"),
            ("resource_type", "resource_type TEXT NOT NULL DEFAULT ''"),
            ("resource_id", "resource_id TEXT"),
            ("details_json", "details_json TEXT"),
            ("created_at", "created_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "benchmark_results",
        [
            ("run_id", "run_id INTEGER"),
            ("model_name", "model_name TEXT NOT NULL DEFAULT ''"),
            ("tracker_type", "tracker_type TEXT NOT NULL DEFAULT ''"),
            ("frame_limit", "frame_limit INTEGER"),
            ("warmup_frames", "warmup_frames INTEGER"),
            ("frames_processed", "frames_processed INTEGER"),
            ("avg_latency_ms", "avg_latency_ms REAL"),
            ("p95_latency_ms", "p95_latency_ms REAL"),
            ("avg_fps", "avg_fps REAL"),
            ("avg_detections_per_frame", "avg_detections_per_frame REAL"),
            ("tracked_frame_ratio", "tracked_frame_ratio REAL"),
            ("detection_count_total", "detection_count_total INTEGER"),
            ("metadata_json", "metadata_json TEXT"),
            ("created_at", "created_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "events",
        [
            ("event_scope", "event_scope TEXT DEFAULT 'raw'"),
            ("snapshot_path", "snapshot_path TEXT"),
            ("evidence_clip_path", "evidence_clip_path TEXT"),
            ("evidence_retention_until", "evidence_retention_until REAL"),
            ("incident_score", "incident_score REAL"),
            ("access_log_id", "access_log_id INTEGER NULL"),
            ("employee_id", "employee_id INTEGER NULL"),
            ("access_point_id", "access_point_id INTEGER NULL"),
            ("identified_employee_id", "identified_employee_id INTEGER NULL"),
            ("identification_confidence", "identification_confidence REAL"),
            ("identification_status", "identification_status TEXT DEFAULT 'not_configured'"),
        ],
    )
    _ensure_columns(
        conn,
        "employees",
        [
            ("full_name", "full_name TEXT NOT NULL DEFAULT ''"),
            ("last_name", "last_name TEXT"),
            ("first_name", "first_name TEXT"),
            ("middle_name", "middle_name TEXT"),
            ("employee_number", "employee_number TEXT"),
            ("department", "department TEXT"),
            ("position", "position TEXT"),
            ("status", "status TEXT"),
            ("created_at", "created_at REAL"),
            ("hire_date", "hire_date REAL"),
            ("external_id", "external_id TEXT"),
            ("source_system", "source_system TEXT"),
            ("profile_photo_url", "profile_photo_url TEXT"),
            ("reference_image_url", "reference_image_url TEXT"),
            ("reference_count", "reference_count INTEGER DEFAULT 0"),
            ("last_synced_at", "last_synced_at REAL"),
            ("presence_status", "presence_status TEXT DEFAULT 'off_duty'"),
            ("last_check_in_at", "last_check_in_at REAL"),
            ("last_check_out_at", "last_check_out_at REAL"),
            ("last_presence_change_at", "last_presence_change_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "access_points",
        [
            ("name", "name TEXT NOT NULL DEFAULT ''"),
            ("location", "location TEXT"),
            ("description", "description TEXT"),
        ],
    )
    _ensure_columns(
        conn,
        "access_logs",
        [
            ("employee_id", "employee_id INTEGER NULL"),
            ("timestamp", "timestamp REAL"),
            ("access_point_id", "access_point_id INTEGER"),
            ("event_type", "event_type TEXT"),
            ("confidence", "confidence REAL"),
            ("note", "note TEXT"),
        ],
    )
    _ensure_columns(
        conn,
        "attendance_sessions",
        [
            ("employee_id", "employee_id INTEGER"),
            ("access_point_id", "access_point_id INTEGER"),
            ("check_in_at", "check_in_at REAL"),
            ("check_out_at", "check_out_at REAL"),
            ("status", "status TEXT DEFAULT 'on_site'"),
            ("source_type", "source_type TEXT"),
            ("model_name", "model_name TEXT"),
            ("detection_confidence", "detection_confidence REAL"),
            ("snapshot_path", "snapshot_path TEXT"),
            ("note", "note TEXT"),
            ("created_at", "created_at REAL"),
            ("updated_at", "updated_at REAL"),
        ],
    )
    _ensure_columns(
        conn,
        "detection_events",
        [
            ("session_id", "session_id TEXT"),
            ("access_log_id", "access_log_id INTEGER NULL"),
            ("employee_id", "employee_id INTEGER NULL"),
            ("access_point_id", "access_point_id INTEGER NULL"),
            ("event_type", "event_type TEXT"),
            ("source_type", "source_type TEXT"),
            ("frame_index", "frame_index INTEGER"),
            ("timestamp", "timestamp REAL"),
            ("class_name", "class_name TEXT"),
            ("confidence", "confidence REAL"),
            ("track_id", "track_id TEXT"),
            ("roi_inside", "roi_inside INTEGER"),
            ("center_x", "center_x REAL"),
            ("center_y", "center_y REAL"),
            ("frame_width", "frame_width INTEGER"),
            ("frame_height", "frame_height INTEGER"),
            ("message", "message TEXT"),
            ("identified_employee_id", "identified_employee_id INTEGER NULL"),
            ("identification_confidence", "identification_confidence REAL"),
            ("identification_status", "identification_status TEXT DEFAULT 'not_configured'"),
        ],
    )
    for key, value in SYSTEM_SETTING_DEFAULTS.items():
        conn.execute(
            """
            INSERT INTO system_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO NOTHING
            """,
            (key, value, time.time()),
        )
    access_point_count = conn.execute("SELECT COUNT(*) AS cnt FROM access_points").fetchone()["cnt"]
    if access_point_count == 0:
        cursor = conn.execute(
            """
            INSERT INTO access_points (name, location, description)
            VALUES (?, ?, ?)
            """,
            (
                "Главная проходная",
                "Входная зона предприятия",
                "Базовая точка доступа для мониторинга прохода сотрудников.",
            ),
        )
        conn.execute(
            """
            INSERT INTO system_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            ("active_access_point_id", str(cursor.lastrowid), time.time()),
        )
    conn.commit()
    conn.close()


def db_upsert_session(session: dict):
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO sessions (
            id, model, source_type, source_path, animal_filter, class_filter,
            rotation_angle, started_at, finished_at, total_frames, processed_frames, events_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            model=excluded.model,
            source_type=excluded.source_type,
            source_path=excluded.source_path,
            animal_filter=excluded.animal_filter,
            class_filter=excluded.class_filter,
            rotation_angle=excluded.rotation_angle,
            started_at=excluded.started_at,
            finished_at=excluded.finished_at,
            total_frames=excluded.total_frames,
            processed_frames=excluded.processed_frames,
            events_count=excluded.events_count
        """,
        (
            session["id"],
            session["model"],
            session["source_type"],
            session["source_path"],
            session["animal_filter"],
            json.dumps(session.get("class_filter", []), ensure_ascii=False),
            session["rotation_angle"],
            session["started_at"],
            session["finished_at"],
            session["total_frames"],
            session["processed_frames"],
            session["events_count"],
        ),
    )
    conn.commit()
    conn.close()


def db_insert_frame(session_id: str, frame_record: dict):
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO frames (
            session_id, frame_index, timestamp, width, height,
            rotation_angle, processing_time_ms, detections_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            frame_record["frame_index"],
            frame_record["timestamp"],
            frame_record["width"],
            frame_record["height"],
            frame_record["rotation_angle"],
            frame_record["processing_time_ms"],
            frame_record["detections_count"],
        ),
    )
    conn.commit()
    conn.close()


def db_insert_event(event: dict):
    conn = get_db_conn()
    event_scope = event.get("event_scope", "raw")
    access_log_id = event.get("access_log_id")

    # Reuse existing domain log linkage when the same event_id is rewritten.
    if access_log_id is None:
        existing_row = conn.execute(
            "SELECT access_log_id FROM events WHERE event_id = ?",
            (event["event_id"],),
        ).fetchone()
        if existing_row is not None:
            access_log_id = existing_row["access_log_id"]

    if event_scope == "domain" and access_log_id is None:
        cursor = conn.execute(
            """
            INSERT INTO access_logs (
                employee_id, timestamp, access_point_id, event_type, confidence, note
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("employee_id"),
                event["timestamp"],
                event.get("access_point_id"),
                event.get("event_type"),
                event.get("confidence"),
                event.get("message") or event.get("note"),
            ),
        )
        access_log_id = cursor.lastrowid

    conn.execute(
        """
        INSERT OR REPLACE INTO events (
            event_id, session_id, event_type, source_type, frame_index, timestamp,
            class_name, confidence, track_id, animal_group, is_animal, roi_inside,
            center_x, center_y, frame_width, frame_height, message, event_scope,
            snapshot_path, evidence_clip_path, evidence_retention_until, incident_score,
            access_log_id, employee_id, access_point_id, identified_employee_id,
            identification_confidence, identification_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event["event_id"],
            event["session_id"],
            event.get("event_type", "object_detected"),
            event["source_type"],
            event["frame_index"],
            event["timestamp"],
            event.get("class_name"),
            event.get("confidence"),
            str(event.get("track_id")) if event.get("track_id") is not None else None,
            event.get("animal_group"),
            1 if event.get("is_animal") else 0,
            1 if event.get("roi_inside") else 0,
            event.get("center_x"),
            event.get("center_y"),
            event.get("frame_width"),
            event.get("frame_height"),
            event.get("message"),
            event_scope,
            (event.get("snapshot_path") or "").strip(),
            (event.get("evidence_clip_path") or "").strip(),
            float(event["evidence_retention_until"]) if event.get("evidence_retention_until") not in {None, ""} else None,
            float(event["incident_score"]) if event.get("incident_score") not in {None, ""} else None,
            access_log_id,
            event.get("employee_id"),
            event.get("access_point_id"),
            event.get("identified_employee_id"),
            event.get("identification_confidence"),
            normalize_identification_status(event.get("identification_status")),
        ),
    )
    # Raw telemetry stays in detection_events, while domain events are linked via access_logs.
    if event_scope == "raw":
        conn.execute(
            """
            INSERT OR REPLACE INTO detection_events (
                id, session_id, access_log_id, employee_id, access_point_id, event_type,
                source_type, frame_index, timestamp, class_name, confidence, track_id,
                roi_inside, center_x, center_y, frame_width, frame_height, message,
                identified_employee_id, identification_confidence, identification_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                event["session_id"],
                access_log_id,
                event.get("employee_id"),
                event.get("access_point_id"),
                event.get("event_type", "object_detected"),
                event["source_type"],
                event["frame_index"],
                event["timestamp"],
                event.get("class_name"),
                event.get("confidence"),
                str(event.get("track_id")) if event.get("track_id") is not None else None,
                1 if event.get("roi_inside") else 0,
                event.get("center_x"),
                event.get("center_y"),
                event.get("frame_width"),
                event.get("frame_height"),
                event.get("message"),
                event.get("identified_employee_id"),
                event.get("identification_confidence"),
                normalize_identification_status(event.get("identification_status")),
            ),
        )
    conn.commit()
    conn.close()


def load_history_from_db():
    conn = get_db_conn()
    session_rows = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC").fetchall()
    frame_rows = conn.execute("SELECT * FROM frames ORDER BY timestamp ASC").fetchall()
    event_rows = conn.execute("SELECT * FROM events ORDER BY timestamp ASC").fetchall()
    conn.close()

    sessions_map = {}
    sessions = []
    for row in session_rows:
        class_filter = []
        raw_class_filter = row["class_filter"]
        if raw_class_filter:
            try:
                class_filter = json.loads(raw_class_filter)
            except json.JSONDecodeError:
                class_filter = []
        session = {
            "id": row["id"],
            "model": row["model"],
            "source_type": row["source_type"],
            "source_path": row["source_path"],
            "animal_filter": row["animal_filter"] or "всё",
            "class_filter": class_filter,
            "rotation_angle": row["rotation_angle"] or 0,
            "started_at": row["started_at"] or time.time(),
            "finished_at": row["finished_at"],
            "total_frames": row["total_frames"] or 0,
            "processed_frames": row["processed_frames"] or 0,
            "events_count": row["events_count"] or 0,
            "seen_track_keys": set(),
            "notified_track_keys": set(),
            "track_inside_roi": {},
            "track_last_seen": {},
            "track_class_by_key": {},
            "disappeared_track_keys": set(),
            "class_event_times": {},
            "rule_last_alert_ts": {},
            "frames": [],
        }
        sessions_map[session["id"]] = session
        sessions.append(session)

    for row in frame_rows:
        sid = row["session_id"]
        if sid not in sessions_map:
            continue
        sessions_map[sid]["frames"].append(
            {
                "frame_index": row["frame_index"],
                "timestamp": row["timestamp"],
                "width": row["width"],
                "height": row["height"],
                "rotation_angle": row["rotation_angle"],
                "processing_time_ms": row["processing_time_ms"],
                "detections_count": row["detections_count"],
                "detections": [],
            }
        )

    events = []
    for row in event_rows:
        events.append(
            {
                "event_id": row["event_id"],
                "session_id": row["session_id"],
                "event_type": row["event_type"] or "object_detected",
                "source_type": row["source_type"],
                "frame_index": row["frame_index"],
                "timestamp": row["timestamp"],
                "class_name": row["class_name"] or "",
                "confidence": row["confidence"] if row["confidence"] is not None else 0.0,
                "track_id": row["track_id"],
                "animal_group": row["animal_group"],
                "is_animal": bool(row["is_animal"]),
                "roi_inside": bool(row["roi_inside"]),
                "center_x": row["center_x"],
                "center_y": row["center_y"],
                "frame_width": row["frame_width"],
                "frame_height": row["frame_height"],
                "message": row["message"] or "",
                "event_scope": row["event_scope"] if "event_scope" in row.keys() else "raw",
                "snapshot_path": row["snapshot_path"] if "snapshot_path" in row.keys() else "",
                "evidence_clip_path": row["evidence_clip_path"] if "evidence_clip_path" in row.keys() else "",
                "evidence_retention_until": row["evidence_retention_until"] if "evidence_retention_until" in row.keys() else None,
                "incident_score": row["incident_score"] if "incident_score" in row.keys() else None,
                "access_log_id": row["access_log_id"] if "access_log_id" in row.keys() else None,
                "employee_id": row["employee_id"] if "employee_id" in row.keys() else None,
                "access_point_id": row["access_point_id"] if "access_point_id" in row.keys() else None,
                "identified_employee_id": row["identified_employee_id"] if "identified_employee_id" in row.keys() else None,
                "identification_confidence": row["identification_confidence"] if "identification_confidence" in row.keys() else None,
                "identification_status": normalize_identification_status(row["identification_status"]) if "identification_status" in row.keys() else "unlinked",
            }
        )

    return sessions, events


def load_employees():
    conn = get_db_conn()
    rows = conn.execute(
        """
        SELECT
            id,
            full_name,
            last_name,
            first_name,
            middle_name,
            employee_number,
            department,
            position,
            status,
            created_at,
            hire_date,
            external_id,
            source_system,
            profile_photo_url,
            reference_image_url,
            reference_count,
            last_synced_at,
            presence_status,
            last_check_in_at,
            last_check_out_at,
            last_presence_change_at
        FROM employees
        ORDER BY employee_number ASC, last_name ASC, first_name ASC, middle_name ASC
        """
    ).fetchall()
    conn.close()
    employees = []
    for row in rows:
        employee = dict(row)
        employee["display_name"] = build_employee_display_name(employee)
        employees.append(employee)
    return employees


def load_access_points():
    conn = get_db_conn()
    rows = conn.execute(
        """
        SELECT id, name, location, description
        FROM access_points
        ORDER BY name ASC
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def create_employee(
    *,
    full_name: str,
    department: str,
    position: str,
    status: str,
    last_name: str = "",
    first_name: str = "",
    middle_name: str = "",
    employee_number: str = "",
    hire_date: Optional[float] = None,
    profile_photo_url: str = "",
):
    if not (last_name.strip() or first_name.strip() or middle_name.strip()):
        last_name, first_name, middle_name = split_employee_full_name(full_name)
    normalized_full_name = build_employee_full_name(last_name, first_name, middle_name) or full_name.strip()
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO employees (
            full_name, last_name, first_name, middle_name, employee_number,
            department, position, status, created_at, hire_date,
            source_system, profile_photo_url, reference_count,
            presence_status, last_check_in_at, last_check_out_at, last_presence_change_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            normalized_full_name,
            last_name.strip(),
            first_name.strip(),
            middle_name.strip(),
            employee_number.strip(),
            department.strip(),
            position.strip(),
            status.strip(),
            time.time(),
            hire_date,
            "local",
            profile_photo_url.strip(),
            0,
            "off_duty",
            None,
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()


def update_employee(
    *,
    employee_id: int,
    full_name: str,
    department: str,
    position: str,
    status: str,
    last_name: str = "",
    first_name: str = "",
    middle_name: str = "",
    employee_number: str = "",
    hire_date: Optional[float] = None,
    profile_photo_url: str = "",
):
    if not (last_name.strip() or first_name.strip() or middle_name.strip()):
        last_name, first_name, middle_name = split_employee_full_name(full_name)
    normalized_full_name = build_employee_full_name(last_name, first_name, middle_name) or full_name.strip()
    conn = get_db_conn()
    conn.execute(
        """
        UPDATE employees
        SET full_name = ?, last_name = ?, first_name = ?, middle_name = ?,
            employee_number = ?, department = ?, position = ?, status = ?,
            hire_date = ?, profile_photo_url = ?
        WHERE id = ?
        """,
        (
            normalized_full_name,
            last_name.strip(),
            first_name.strip(),
            middle_name.strip(),
            employee_number.strip(),
            department.strip(),
            position.strip(),
            status.strip(),
            hire_date,
            profile_photo_url.strip(),
            employee_id,
        ),
    )
    conn.commit()
    conn.close()


def update_employee_status(*, employee_id: int, status: str):
    conn = get_db_conn()
    conn.execute(
        """
        UPDATE employees
        SET status = ?
        WHERE id = ?
        """,
        (
            status.strip(),
            employee_id,
        ),
    )
    conn.commit()
    conn.close()


def ensure_demo_employees():
    """
    Seed a minimal employee list for a stable thesis demo.

    The function is idempotent and does not overwrite existing records.
    """
    conn = get_db_conn()
    existing_count = conn.execute("SELECT COUNT(*) AS cnt FROM employees").fetchone()["cnt"]
    if existing_count > 0:
        conn.close()
        return False

    now_ts = time.time()
    demo_rows = [
        (
            "Иванов Иван Иванович",
            "Иванов",
            "Иван",
            "Иванович",
            "A-1001",
            "Служба эксплуатации",
            "Инженер по эксплуатации",
            "active",
            now_ts,
            now_ts - 420 * 86400,
            "local",
            "",
            0,
        ),
        (
            "Петров Петр Сергеевич",
            "Петров",
            "Петр",
            "Сергеевич",
            "B-2045",
            "Служба безопасности",
            "Оператор центра мониторинга",
            "active",
            now_ts,
            now_ts - 780 * 86400,
            "local",
            "",
            0,
        ),
        (
            "Сидорова Анна Викторовна",
            "Сидорова",
            "Анна",
            "Викторовна",
            "HR-0312",
            "Отдел кадров",
            "Специалист по персоналу",
            "inactive",
            now_ts,
            now_ts - 1100 * 86400,
            "local",
            "",
            0,
        ),
    ]
    conn.executemany(
        """
        INSERT INTO employees (
            full_name, last_name, first_name, middle_name, employee_number,
            department, position, status, created_at, hire_date,
            source_system, profile_photo_url, reference_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        demo_rows,
    )
    conn.commit()
    conn.close()
    return True


def load_access_logs():
    conn = get_db_conn()
    rows = conn.execute(
        """
        SELECT
            access_logs.id,
            access_logs.employee_id,
            employees.full_name AS employee_name,
            employees.employee_number AS employee_number,
            access_logs.timestamp,
            access_logs.access_point_id,
            access_points.name AS access_point_name,
            access_logs.event_type,
            access_logs.confidence,
            access_logs.note
        FROM access_logs
        LEFT JOIN employees ON employees.id = access_logs.employee_id
        LEFT JOIN access_points ON access_points.id = access_logs.access_point_id
        ORDER BY access_logs.timestamp DESC
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def load_attendance_sessions(*, day: str | None = None, limit: int | None = 500) -> list[dict]:
    conn = get_db_conn()
    params: list[object] = []
    query = """
        SELECT
            attendance_sessions.id,
            attendance_sessions.employee_id,
            attendance_sessions.access_point_id,
            attendance_sessions.check_in_at,
            attendance_sessions.check_out_at,
            attendance_sessions.status,
            attendance_sessions.source_type,
            attendance_sessions.model_name,
            attendance_sessions.detection_confidence,
            attendance_sessions.snapshot_path,
            attendance_sessions.note,
            attendance_sessions.created_at,
            attendance_sessions.updated_at,
            employees.full_name AS employee_name,
            employees.employee_number AS employee_number,
            employees.department AS department,
            employees.position AS position,
            access_points.name AS access_point_name
        FROM attendance_sessions
        LEFT JOIN employees ON employees.id = attendance_sessions.employee_id
        LEFT JOIN access_points ON access_points.id = attendance_sessions.access_point_id
    """
    if day:
        day_start, day_end = _resolve_day_bounds(day)
        query += " WHERE attendance_sessions.check_in_at >= ? AND attendance_sessions.check_in_at < ?"
        params.extend([day_start, day_end])
    query += " ORDER BY attendance_sessions.check_in_at DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    items = []
    for row in rows:
        item = dict(row)
        check_in_at = float(item.get("check_in_at") or 0.0)
        check_out_at = item.get("check_out_at")
        if check_out_at is not None:
            duration_seconds = max(0.0, float(check_out_at) - check_in_at)
        else:
            duration_seconds = max(0.0, time.time() - check_in_at)
        item["duration_seconds"] = duration_seconds
        items.append(item)
    return items


def load_attendance_today() -> dict:
    today_key = datetime.now().strftime("%Y-%m-%d")
    sessions = load_attendance_sessions(day=today_key, limit=1000)
    currently_on_site = sum(1 for session in sessions if session.get("status") == "on_site" and session.get("check_out_at") is None)
    check_ins = len(sessions)
    check_outs = sum(1 for session in sessions if session.get("check_out_at") is not None)
    average_duration = round(
        sum(session["duration_seconds"] for session in sessions) / len(sessions) / 60.0,
        1,
    ) if sessions else 0.0
    return {
        "day": today_key,
        "summary": {
            "check_ins": check_ins,
            "check_outs": check_outs,
            "currently_on_site": currently_on_site,
            "average_duration_minutes": average_duration,
        },
        "items": sessions,
    }


def register_employee_attendance(
    *,
    employee_id: int,
    access_point_id: int | None,
    model_name: str,
    source_type: str,
    detection_confidence: float | None,
    snapshot_path: str = "",
    note: str = "",
) -> dict:
    conn = get_db_conn()
    employee = conn.execute(
        """
        SELECT id, full_name, employee_number, department, position, status
        FROM employees
        WHERE id = ?
        """,
        (employee_id,),
    ).fetchone()
    if employee is None:
        conn.close()
        raise ValueError(f"employee_not_found:{employee_id}")
    if (employee["status"] or "").strip() != "active":
        conn.close()
        raise ValueError(f"employee_inactive:{employee_id}")

    access_point = None
    if access_point_id is not None:
        access_point = conn.execute(
            "SELECT id, name FROM access_points WHERE id = ?",
            (access_point_id,),
        ).fetchone()
        if access_point is None:
            conn.close()
            raise ValueError(f"access_point_not_found:{access_point_id}")

    now_ts = time.time()
    open_session = conn.execute(
        """
        SELECT id, check_in_at
        FROM attendance_sessions
        WHERE employee_id = ? AND check_out_at IS NULL
        ORDER BY check_in_at DESC
        LIMIT 1
        """,
        (employee_id,),
    ).fetchone()

    if open_session is None:
        attendance_status = "check_in"
        cursor = conn.execute(
            """
            INSERT INTO attendance_sessions (
                employee_id, access_point_id, check_in_at, check_out_at, status,
                source_type, model_name, detection_confidence, snapshot_path, note, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                employee_id,
                access_point_id,
                now_ts,
                None,
                "on_site",
                source_type.strip(),
                model_name.strip(),
                detection_confidence,
                snapshot_path.strip(),
                note.strip(),
                now_ts,
                now_ts,
            ),
        )
        attendance_session_id = cursor.lastrowid
        conn.execute(
            """
            UPDATE employees
            SET presence_status = ?, last_check_in_at = ?, last_presence_change_at = ?
            WHERE id = ?
            """,
            ("on_site", now_ts, now_ts, employee_id),
        )
        event_type = "employee_checked_in"
        message = (
            f"{employee['full_name']} отмечен на рабочем месте"
            + (f" через {access_point['name']}" if access_point is not None else "")
        )
    else:
        attendance_status = "check_out"
        attendance_session_id = int(open_session["id"])
        conn.execute(
            """
            UPDATE attendance_sessions
            SET check_out_at = ?, status = ?, access_point_id = COALESCE(?, access_point_id),
                source_type = ?, model_name = ?, detection_confidence = ?, snapshot_path = ?, note = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                now_ts,
                "completed",
                access_point_id,
                source_type.strip(),
                model_name.strip(),
                detection_confidence,
                snapshot_path.strip(),
                note.strip(),
                now_ts,
                attendance_session_id,
            ),
        )
        conn.execute(
            """
            UPDATE employees
            SET presence_status = ?, last_check_out_at = ?, last_presence_change_at = ?
            WHERE id = ?
            """,
            ("off_duty", now_ts, now_ts, employee_id),
        )
        event_type = "employee_checked_out"
        message = (
            f"{employee['full_name']} завершил рабочую смену"
            + (f" через {access_point['name']}" if access_point is not None else "")
        )

    cursor = conn.execute(
        """
        INSERT INTO access_logs (employee_id, timestamp, access_point_id, event_type, confidence, note)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (employee_id, now_ts, access_point_id, event_type, detection_confidence, note.strip() or message),
    )
    access_log_id = cursor.lastrowid
    event_id = f"attendance-{uuid.uuid4().hex[:10]}"
    session_id = f"portal-{access_point_id or 0}"
    event = {
        "event_id": event_id,
        "session_id": session_id,
        "event_scope": "domain",
        "event_type": event_type,
        "source_type": source_type.strip(),
        "frame_index": 0,
        "timestamp": now_ts,
        "class_name": "person",
        "confidence": detection_confidence or 0.0,
        "track_id": None,
        "roi_inside": True,
        "message": message,
        "access_log_id": access_log_id,
        "employee_id": employee_id,
        "access_point_id": access_point_id,
        "identified_employee_id": employee_id,
        "identification_confidence": detection_confidence,
        "identification_status": "linked_from_directory",
        "snapshot_path": snapshot_path.strip(),
    }
    conn.commit()
    conn.close()
    db_insert_event(event)
    return {
        "attendance_session_id": attendance_session_id,
        "attendance_status": attendance_status,
        "event_id": event_id,
        "event_type": event_type,
        "timestamp": now_ts,
        "employee": {
            "id": employee["id"],
            "full_name": employee["full_name"],
            "employee_number": employee["employee_number"] or "",
            "department": employee["department"] or "",
            "position": employee["position"] or "",
        },
        "access_point_name": access_point["name"] if access_point is not None else "",
        "message": message,
    }


def load_events(limit=None):
    conn = get_db_conn()
    query = """
        SELECT
            events.event_id,
            events.session_id,
            events.event_type,
            events.source_type,
            events.frame_index,
            events.timestamp,
            events.class_name,
            events.confidence,
            events.track_id,
            events.animal_group,
            events.is_animal,
            events.roi_inside,
            events.center_x,
            events.center_y,
            events.frame_width,
            events.frame_height,
            events.message,
            events.event_scope,
            events.snapshot_path,
            events.evidence_clip_path,
            events.evidence_retention_until,
            events.incident_score,
            events.access_log_id,
            events.employee_id,
            events.access_point_id,
            events.identified_employee_id,
            events.identification_confidence,
            events.identification_status,
            sessions.source_path,
            sessions.model,
            employees.full_name AS employee_name,
            employees.employee_number AS employee_number,
            employees.department AS employee_department,
            employees.position AS employee_position,
            employees.status AS employee_status,
            employees.profile_photo_url AS employee_photo_url,
            access_points.name AS access_point_name
        FROM events
        LEFT JOIN sessions ON sessions.id = events.session_id
        LEFT JOIN access_logs ON access_logs.id = events.access_log_id
        LEFT JOIN employees ON employees.id = COALESCE(events.employee_id, events.identified_employee_id, access_logs.employee_id)
        LEFT JOIN access_points ON access_points.id = COALESCE(events.access_point_id, access_logs.access_point_id)
        ORDER BY events.timestamp DESC
    """
    params = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    normalized_rows = []
    for row in rows:
        event = dict(row)
        event["identification_status"] = normalize_identification_status(event.get("identification_status"))
        normalized_rows.append(event)
    return normalized_rows


def link_event_to_employee(
    *,
    event_id: str,
    employee_id: int,
    identification_status: str,
    note: str = "",
):
    conn = get_db_conn()
    event_row = conn.execute(
        """
        SELECT event_id, access_log_id, employee_id, identified_employee_id, message
        FROM events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()
    if event_row is None:
        conn.close()
        raise ValueError(f"event_not_found:{event_id}")

    employee_row = conn.execute(
        """
        SELECT id, full_name, employee_number, status
        FROM employees
        WHERE id = ?
        """,
        (employee_id,),
    ).fetchone()
    if employee_row is None:
        conn.close()
        raise ValueError(f"employee_not_found:{employee_id}")

    employee_status = employee_row["status"] or ""
    if employee_status != "active":
        identification_status = "inactive_employee"

    note_parts = [part for part in [event_row["message"], note.strip()] if part]
    if employee_row["full_name"]:
        note_parts.append(
            f"Оператор связал событие с сотрудником {employee_row['full_name']}"
            + (f" ({employee_row['employee_number']})" if employee_row["employee_number"] else "")
        )
    merged_note = ". ".join(dict.fromkeys(note_parts))

    conn.execute(
        """
        UPDATE events
        SET employee_id = ?, identified_employee_id = ?, identification_status = ?, message = ?
        WHERE event_id = ?
        """,
        (employee_id, employee_id, identification_status, merged_note, event_id),
    )
    if event_row["access_log_id"] is not None:
        conn.execute(
            """
            UPDATE access_logs
            SET employee_id = ?, note = ?
            WHERE id = ?
            """,
            (employee_id, merged_note, event_row["access_log_id"]),
        )
    conn.commit()
    conn.close()


def load_video_sources():
    conn = get_db_conn()
    rows = conn.execute(
        f"""
        SELECT {_video_source_select_columns()}
        FROM video_sources
        ORDER BY is_active DESC, name ASC
        """
    ).fetchall()
    conn.close()
    return [_normalize_video_source_row(row) for row in rows]


def load_zones(*, source_id: int | None = None):
    conn = get_db_conn()
    if source_id is None:
        rows = conn.execute(
            f"""
            SELECT {_zone_select_columns()}
            FROM zones
            ORDER BY is_active DESC, source_id ASC, name ASC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT {_zone_select_columns()}
            FROM zones
            WHERE source_id = ?
            ORDER BY is_active DESC, name ASC
            """,
            (source_id,),
        ).fetchall()
    conn.close()
    return [_normalize_zone_row(row) for row in rows]


def load_zone_rules(*, source_id: int | None = None, zone_id: int | None = None):
    conn = get_db_conn()
    where_clauses = []
    params = []
    if source_id is not None:
        where_clauses.append("z.source_id = ?")
        params.append(int(source_id))
    if zone_id is not None:
        where_clauses.append("zr.zone_id = ?")
        params.append(int(zone_id))
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    rows = conn.execute(
        f"""
        SELECT {_zone_rule_select_columns()}
        FROM zone_rules zr
        JOIN zones z ON z.id = zr.zone_id
        {where_sql}
        ORDER BY zr.is_active DESC, zr.zone_id ASC, zr.rule_type ASC
        """,
        tuple(params),
    ).fetchall()
    conn.close()
    return [_normalize_zone_rule_row(row) for row in rows]


def load_incidents(*, limit: int | None = None):
    conn = get_db_conn()
    sql = f"""
        SELECT
            incidents.id,
            incidents.event_id,
            incidents.source_id,
            incidents.zone_name,
            incidents.incident_type,
            incidents.severity,
            incidents.status,
            incidents.confidence,
            incidents.snapshot_path,
            incidents.evidence_clip_path,
            incidents.evidence_retention_until,
            incidents.operator_comment,
            incidents.assigned_to,
            incidents.acknowledged_at,
            incidents.resolved_at,
            incidents.resolution_code,
            incidents.resolution_notes,
            incidents.employee_id,
            incidents.identification_status,
            incidents.started_at,
            incidents.updated_at,
            video_sources.name AS source_name
        FROM incidents
        LEFT JOIN video_sources ON video_sources.id = incidents.source_id
        ORDER BY incidents.started_at DESC, incidents.id DESC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    conn.close()
    return [_normalize_incident_row(row) for row in rows]


def load_notification_deliveries(*, incident_id: int | None = None):
    conn = get_db_conn()
    if incident_id is None:
        rows = conn.execute(
            """
            SELECT id, incident_id, channel, destination, delivery_status, last_error, sent_at, created_at, updated_at
            FROM notification_deliveries
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, incident_id, channel, destination, delivery_status, last_error, sent_at, created_at, updated_at
            FROM notification_deliveries
            WHERE incident_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (int(incident_id),),
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def append_audit_log(
    *,
    actor_name: str,
    actor_role: str,
    action: str,
    resource_type: str,
    resource_id: str = "",
    details: dict | None = None,
):
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO audit_logs (
            actor_name, actor_role, action, resource_type, resource_id, details_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            actor_name.strip(),
            actor_role.strip(),
            action.strip(),
            resource_type.strip(),
            str(resource_id).strip(),
            json.dumps(details or {}, ensure_ascii=False),
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def load_audit_logs(*, limit: int = 200):
    conn = get_db_conn()
    rows = conn.execute(
        """
        SELECT id, actor_name, actor_role, action, resource_type, resource_id, details_json, created_at
        FROM audit_logs
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    conn.close()
    normalized_rows = []
    for row in rows:
        item = dict(row)
        try:
            item["details"] = json.loads(item.get("details_json") or "{}")
        except json.JSONDecodeError:
            item["details"] = {}
        normalized_rows.append(item)
    return normalized_rows


def replace_employee_cache(employees: list[dict], *, source_system: str, synced_at=None):
    conn = get_db_conn()
    synced_at = synced_at or time.time()
    conn.execute("DELETE FROM employees")
    conn.execute("DELETE FROM sqlite_sequence WHERE name = 'employees'")
    rows = []
    for employee in employees:
        last_name = (employee.get("last_name") or "").strip()
        first_name = (employee.get("first_name") or "").strip()
        middle_name = (employee.get("middle_name") or "").strip()
        if not (last_name or first_name or middle_name):
            last_name, first_name, middle_name = split_employee_full_name(employee.get("full_name", ""))
        full_name = build_employee_full_name(last_name, first_name, middle_name) or employee.get("full_name", "").strip()
        rows.append(
            (
                full_name,
                last_name,
                first_name,
                middle_name,
                (employee.get("employee_number") or "").strip(),
                employee.get("department", "").strip(),
                employee.get("position", "").strip(),
                employee.get("status", "").strip() or "active",
                employee.get("created_at") or synced_at,
                employee.get("hire_date"),
                employee.get("external_id"),
                source_system,
                employee.get("profile_photo_url"),
                employee.get("reference_image_url"),
                int(employee.get("reference_count") or 0),
                synced_at,
            )
        )
    conn.executemany(
        """
        INSERT INTO employees (
            full_name, last_name, first_name, middle_name, employee_number,
            department, position, status, created_at, hire_date,
            external_id, source_system, profile_photo_url, reference_image_url,
            reference_count, last_synced_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def load_employee_sync_state():
    conn = get_db_conn()
    row = conn.execute(
        """
        SELECT data_source, sync_status, last_synced_at, last_error, cache_mode, updated_at
        FROM employee_sync_state
        WHERE id = 1
        """
    ).fetchone()
    conn.close()
    return dict(row) if row is not None else None


def upsert_employee_sync_state(
    *,
    data_source: str,
    sync_status: str,
    last_synced_at=None,
    last_error: str = "",
    cache_mode: str = "read_write",
):
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO employee_sync_state (
            id, data_source, sync_status, last_synced_at, last_error, cache_mode, updated_at
        ) VALUES (1, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            data_source = excluded.data_source,
            sync_status = excluded.sync_status,
            last_synced_at = excluded.last_synced_at,
            last_error = excluded.last_error,
            cache_mode = excluded.cache_mode,
            updated_at = excluded.updated_at
        """,
        (data_source, sync_status, last_synced_at, last_error, cache_mode, time.time()),
    )
    conn.commit()
    conn.close()


def load_active_video_sources():
    conn = get_db_conn()
    rows = conn.execute(
        f"""
        SELECT {_video_source_select_columns()}
        FROM video_sources
        WHERE is_active = 1
        ORDER BY id ASC
        """
    ).fetchall()
    conn.close()
    return [_normalize_video_source_row(row) for row in rows]


def create_video_source(
    *,
    name: str,
    source_type: str,
    source_url: str,
    location: str,
    description: str,
    is_active: bool,
    enable_roi: bool = True,
    roi_x: float = 20,
    roi_y: float = 20,
    roi_w: float = 60,
    roi_h: float = 60,
    rule_count_enabled: bool = False,
    rule_n: int = 3,
    rule_t: int = 10,
    rule_disappear_enabled: bool = True,
    rule_disappear_seconds: int = 5,
    prolonged_presence_seconds: int = 10,
    ai_profile_override: str = "",
    conf_threshold_override: float | None = None,
    inference_size_override: int | None = None,
    tracker_type_override: str = "",
    incident_threshold_override: float | None = None,
):
    config = normalize_source_processing_config(
        {
            "enable_roi": enable_roi,
            "roi_x": roi_x,
            "roi_y": roi_y,
            "roi_w": roi_w,
            "roi_h": roi_h,
            "rule_count_enabled": rule_count_enabled,
            "rule_n": rule_n,
            "rule_t": rule_t,
            "rule_disappear_enabled": rule_disappear_enabled,
            "rule_disappear_seconds": rule_disappear_seconds,
            "prolonged_presence_seconds": prolonged_presence_seconds,
            "ai_profile_override": ai_profile_override,
            "conf_threshold_override": conf_threshold_override,
            "inference_size_override": inference_size_override,
            "tracker_type_override": tracker_type_override,
            "incident_threshold_override": incident_threshold_override,
        }
    )
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO video_sources (
            name, source_type, source_url, location, is_active, last_seen, description, created_at,
            enable_roi, roi_x, roi_y, roi_w, roi_h,
            rule_count_enabled, rule_n, rule_t, rule_disappear_enabled, rule_disappear_seconds,
            prolonged_presence_seconds, ai_profile_override, conf_threshold_override, inference_size_override,
            tracker_type_override, incident_threshold_override
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name.strip(),
            source_type.strip(),
            source_url.strip(),
            location.strip(),
            1 if is_active else 0,
            None,
            description.strip(),
            time.time(),
            1 if config["enable_roi"] else 0,
            config["roi_x"],
            config["roi_y"],
            config["roi_w"],
            config["roi_h"],
            1 if config["rule_count_enabled"] else 0,
            config["rule_n"],
            config["rule_t"],
            1 if config["rule_disappear_enabled"] else 0,
            config["rule_disappear_seconds"],
            config["prolonged_presence_seconds"],
            config["ai_profile_override"] or None,
            config["conf_threshold_override"],
            config["inference_size_override"],
            config["tracker_type_override"] or None,
            config["incident_threshold_override"],
        ),
    )
    conn.commit()
    conn.close()


def create_zone(
    *,
    source_id: int,
    name: str,
    zone_type: str,
    x: float,
    y: float,
    w: float,
    h: float,
    is_active: bool = True,
    description: str = "",
):
    config = normalize_zone_config(
        {
            "name": name,
            "zone_type": zone_type,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "is_active": is_active,
            "description": description,
        }
    )
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO zones (source_id, name, zone_type, x, y, w, h, is_active, description, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(source_id),
            config["name"],
            config["zone_type"],
            config["x"],
            config["y"],
            config["w"],
            config["h"],
            1 if config["is_active"] else 0,
            config["description"],
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def create_zone_rule(
    *,
    zone_id: int,
    rule_type: str,
    threshold_seconds: int = 10,
    threshold_count: int = 3,
    cooldown_seconds: int = 5,
    is_active: bool = True,
    severity: str = "medium",
    description: str = "",
):
    config = normalize_zone_rule_config(
        {
            "rule_type": rule_type,
            "threshold_seconds": threshold_seconds,
            "threshold_count": threshold_count,
            "cooldown_seconds": cooldown_seconds,
            "is_active": is_active,
            "severity": severity,
            "description": description,
        }
    )
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO zone_rules (
            zone_id, rule_type, threshold_seconds, threshold_count, cooldown_seconds,
            is_active, severity, description, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(zone_id),
            config["rule_type"],
            config["threshold_seconds"],
            config["threshold_count"],
            config["cooldown_seconds"],
            1 if config["is_active"] else 0,
            config["severity"],
            config["description"],
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def upsert_incident(
    *,
    event_id: str,
    source_id: int | None,
    zone_name: str,
    incident_type: str,
    severity: str,
    status: str = "new",
    confidence: float = 0.0,
    snapshot_path: str = "",
    evidence_clip_path: str = "",
    evidence_retention_until: float | None = None,
    operator_comment: str = "",
    assigned_to: str = "",
    acknowledged_at: float | None = None,
    resolved_at: float | None = None,
    resolution_code: str = "",
    resolution_notes: str = "",
    employee_id: int | None = None,
    identification_status: str = "unlinked",
    started_at: float | None = None,
):
    config = normalize_incident_config(
        {
            "zone_name": zone_name,
            "incident_type": incident_type,
            "severity": severity,
            "status": status,
            "confidence": confidence,
            "snapshot_path": snapshot_path,
            "evidence_clip_path": evidence_clip_path,
            "evidence_retention_until": evidence_retention_until,
            "operator_comment": operator_comment,
            "assigned_to": assigned_to,
            "acknowledged_at": acknowledged_at,
            "resolved_at": resolved_at,
            "resolution_code": resolution_code,
            "resolution_notes": resolution_notes,
            "identification_status": identification_status,
        }
    )
    now_ts = time.time()
    started_at = float(started_at if started_at is not None else now_ts)
    conn = get_db_conn()
    existing = conn.execute(
        """
        SELECT id, status, operator_comment, assigned_to, acknowledged_at, resolved_at, resolution_code, resolution_notes
        FROM incidents
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()
    if existing is None:
        conn.execute(
            """
            INSERT INTO incidents (
                event_id, source_id, zone_name, incident_type, severity, status, confidence,
                snapshot_path, evidence_clip_path, evidence_retention_until, operator_comment,
                assigned_to, acknowledged_at, resolved_at, resolution_code, resolution_notes,
                employee_id, identification_status, started_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                source_id,
                config["zone_name"],
                config["incident_type"],
                config["severity"],
                config["status"],
                config["confidence"],
                config["snapshot_path"],
                config["evidence_clip_path"],
                config["evidence_retention_until"],
                config["operator_comment"],
                config["assigned_to"],
                config["acknowledged_at"],
                config["resolved_at"],
                config["resolution_code"],
                config["resolution_notes"],
                employee_id,
                config["identification_status"],
                started_at,
                now_ts,
            ),
        )
    else:
        preserved_status = existing["status"] or config["status"]
        preserved_comment = existing["operator_comment"] or config["operator_comment"]
        preserved_assigned_to = existing["assigned_to"] or config["assigned_to"]
        preserved_acknowledged_at = existing["acknowledged_at"] if existing["acknowledged_at"] is not None else config["acknowledged_at"]
        preserved_resolved_at = existing["resolved_at"] if existing["resolved_at"] is not None else config["resolved_at"]
        preserved_resolution_code = existing["resolution_code"] or config["resolution_code"]
        preserved_resolution_notes = existing["resolution_notes"] or config["resolution_notes"]
        conn.execute(
            """
            UPDATE incidents
            SET source_id = ?, zone_name = ?, incident_type = ?, severity = ?, status = ?, confidence = ?,
                snapshot_path = ?, evidence_clip_path = ?, evidence_retention_until = ?, operator_comment = ?,
                assigned_to = ?, acknowledged_at = ?, resolved_at = ?, resolution_code = ?, resolution_notes = ?,
                employee_id = ?, identification_status = ?, updated_at = ?
            WHERE event_id = ?
            """,
            (
                source_id,
                config["zone_name"],
                config["incident_type"],
                config["severity"],
                preserved_status,
                config["confidence"],
                config["snapshot_path"],
                config["evidence_clip_path"],
                config["evidence_retention_until"],
                preserved_comment,
                preserved_assigned_to,
                preserved_acknowledged_at,
                preserved_resolved_at,
                preserved_resolution_code,
                preserved_resolution_notes,
                employee_id,
                config["identification_status"],
                now_ts,
                event_id,
            ),
        )
    conn.commit()
    conn.close()


def attach_event_evidence(
    *,
    event_id: str,
    snapshot_path: str = "",
    evidence_clip_path: str = "",
    evidence_retention_until: float | None = None,
):
    conn = get_db_conn()
    now_ts = time.time()
    snapshot_path = snapshot_path.strip()
    evidence_clip_path = evidence_clip_path.strip()
    retention_value = float(evidence_retention_until) if evidence_retention_until is not None else None
    conn.execute(
        """
        UPDATE events
        SET snapshot_path = ?, evidence_clip_path = ?, evidence_retention_until = ?
        WHERE event_id = ?
        """,
        (snapshot_path, evidence_clip_path, retention_value, event_id),
    )
    conn.execute(
        """
        UPDATE incidents
        SET snapshot_path = CASE WHEN ? <> '' THEN ? ELSE snapshot_path END,
            evidence_clip_path = CASE WHEN ? <> '' THEN ? ELSE evidence_clip_path END,
            evidence_retention_until = COALESCE(?, evidence_retention_until),
            updated_at = ?
        WHERE event_id = ?
        """,
        (
            snapshot_path,
            snapshot_path,
            evidence_clip_path,
            evidence_clip_path,
            retention_value,
            now_ts,
            event_id,
        ),
    )
    conn.commit()
    conn.close()


def update_video_source(
    *,
    source_id: int,
    name: str,
    source_type: str,
    source_url: str,
    location: str,
    description: str,
    enable_roi: bool = True,
    roi_x: float = 20,
    roi_y: float = 20,
    roi_w: float = 60,
    roi_h: float = 60,
    rule_count_enabled: bool = False,
    rule_n: int = 3,
    rule_t: int = 10,
    rule_disappear_enabled: bool = True,
    rule_disappear_seconds: int = 5,
    prolonged_presence_seconds: int = 10,
    ai_profile_override: str = "",
    conf_threshold_override: float | None = None,
    inference_size_override: int | None = None,
    tracker_type_override: str = "",
    incident_threshold_override: float | None = None,
):
    config = normalize_source_processing_config(
        {
            "enable_roi": enable_roi,
            "roi_x": roi_x,
            "roi_y": roi_y,
            "roi_w": roi_w,
            "roi_h": roi_h,
            "rule_count_enabled": rule_count_enabled,
            "rule_n": rule_n,
            "rule_t": rule_t,
            "rule_disappear_enabled": rule_disappear_enabled,
            "rule_disappear_seconds": rule_disappear_seconds,
            "prolonged_presence_seconds": prolonged_presence_seconds,
            "ai_profile_override": ai_profile_override,
            "conf_threshold_override": conf_threshold_override,
            "inference_size_override": inference_size_override,
            "tracker_type_override": tracker_type_override,
            "incident_threshold_override": incident_threshold_override,
        }
    )
    conn = get_db_conn()
    conn.execute(
        """
        UPDATE video_sources
        SET name = ?, source_type = ?, source_url = ?, location = ?, description = ?,
            enable_roi = ?, roi_x = ?, roi_y = ?, roi_w = ?, roi_h = ?,
            rule_count_enabled = ?, rule_n = ?, rule_t = ?, rule_disappear_enabled = ?,
            rule_disappear_seconds = ?, prolonged_presence_seconds = ?, ai_profile_override = ?,
            conf_threshold_override = ?, inference_size_override = ?, tracker_type_override = ?,
            incident_threshold_override = ?
        WHERE id = ?
        """,
        (
            name.strip(),
            source_type.strip(),
            source_url.strip(),
            location.strip(),
            description.strip(),
            1 if config["enable_roi"] else 0,
            config["roi_x"],
            config["roi_y"],
            config["roi_w"],
            config["roi_h"],
            1 if config["rule_count_enabled"] else 0,
            config["rule_n"],
            config["rule_t"],
            1 if config["rule_disappear_enabled"] else 0,
            config["rule_disappear_seconds"],
            config["prolonged_presence_seconds"],
            config["ai_profile_override"] or None,
            config["conf_threshold_override"],
            config["inference_size_override"],
            config["tracker_type_override"] or None,
            config["incident_threshold_override"],
            source_id,
        ),
    )
    conn.commit()
    conn.close()


def update_zone(
    *,
    zone_id: int,
    source_id: int,
    name: str,
    zone_type: str,
    x: float,
    y: float,
    w: float,
    h: float,
    description: str = "",
):
    config = normalize_zone_config(
        {
            "name": name,
            "zone_type": zone_type,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "description": description,
        }
    )
    conn = get_db_conn()
    conn.execute(
        """
        UPDATE zones
        SET source_id = ?, name = ?, zone_type = ?, x = ?, y = ?, w = ?, h = ?, description = ?
        WHERE id = ?
        """,
        (
            int(source_id),
            config["name"],
            config["zone_type"],
            config["x"],
            config["y"],
            config["w"],
            config["h"],
            config["description"],
            int(zone_id),
        ),
    )
    conn.commit()
    conn.close()


def update_zone_rule(
    *,
    rule_id: int,
    zone_id: int,
    rule_type: str,
    threshold_seconds: int,
    threshold_count: int,
    cooldown_seconds: int,
    severity: str,
    description: str = "",
):
    config = normalize_zone_rule_config(
        {
            "rule_type": rule_type,
            "threshold_seconds": threshold_seconds,
            "threshold_count": threshold_count,
            "cooldown_seconds": cooldown_seconds,
            "severity": severity,
            "description": description,
        }
    )
    conn = get_db_conn()
    conn.execute(
        """
        UPDATE zone_rules
        SET zone_id = ?, rule_type = ?, threshold_seconds = ?, threshold_count = ?, cooldown_seconds = ?,
            severity = ?, description = ?
        WHERE id = ?
        """,
        (
            int(zone_id),
            config["rule_type"],
            config["threshold_seconds"],
            config["threshold_count"],
            config["cooldown_seconds"],
            config["severity"],
            config["description"],
            int(rule_id),
        ),
    )
    conn.commit()
    conn.close()


def update_incident_status(
    *,
    incident_id: int,
    status: str,
    operator_comment: str | None = None,
    assigned_to: str | None = None,
    resolution_code: str | None = None,
    resolution_notes: str | None = None,
):
    normalized_status = (status or "new").strip() or "new"
    now_ts = time.time()
    conn = get_db_conn()
    existing = conn.execute(
        """
        SELECT acknowledged_at, resolved_at, assigned_to, operator_comment, resolution_code, resolution_notes
        FROM incidents
        WHERE id = ?
        """,
        (int(incident_id),),
    ).fetchone()
    if existing is None:
        conn.close()
        return
    resolved_statuses = {"resolved", "false_positive", "rejected"}
    acknowledged_statuses = {"acknowledged", "in_progress", "on_hold", "escalated"} | resolved_statuses
    acknowledged_at = existing["acknowledged_at"]
    if normalized_status in acknowledged_statuses and acknowledged_at is None:
        acknowledged_at = now_ts
    resolved_at = existing["resolved_at"]
    if normalized_status in resolved_statuses:
        resolved_at = now_ts
    elif normalized_status not in resolved_statuses:
        resolved_at = None
    conn.execute(
        """
        UPDATE incidents
        SET status = ?, operator_comment = ?, assigned_to = ?, acknowledged_at = ?, resolved_at = ?,
            resolution_code = ?, resolution_notes = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            normalized_status,
            existing["operator_comment"] if operator_comment is None else operator_comment.strip(),
            existing["assigned_to"] if assigned_to is None else assigned_to.strip(),
            acknowledged_at,
            resolved_at,
            existing["resolution_code"] if resolution_code is None else resolution_code.strip(),
            existing["resolution_notes"] if resolution_notes is None else resolution_notes.strip(),
            now_ts,
            int(incident_id),
        ),
    )
    conn.commit()
    conn.close()


def upsert_notification_delivery(
    *,
    incident_id: int,
    channel: str,
    destination: str,
    delivery_status: str,
    last_error: str = "",
    sent_at: float | None = None,
):
    now_ts = time.time()
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO notification_deliveries (
            incident_id, channel, destination, delivery_status, last_error, sent_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(incident_id, channel, destination) DO UPDATE SET
            delivery_status = excluded.delivery_status,
            last_error = excluded.last_error,
            sent_at = excluded.sent_at,
            updated_at = excluded.updated_at
        """,
        (
            int(incident_id),
            channel.strip(),
            destination.strip(),
            delivery_status.strip(),
            last_error.strip(),
            sent_at,
            now_ts,
            now_ts,
        ),
    )
    conn.commit()
    conn.close()


def set_video_source_active(*, source_id: int, is_active: bool):
    conn = get_db_conn()
    conn.execute("UPDATE video_sources SET is_active = ? WHERE id = ?", (1 if is_active else 0, source_id))
    conn.commit()
    conn.close()


def set_zone_active(*, zone_id: int, is_active: bool):
    conn = get_db_conn()
    conn.execute("UPDATE zones SET is_active = ? WHERE id = ?", (1 if is_active else 0, int(zone_id)))
    conn.commit()
    conn.close()


def set_zone_rule_active(*, rule_id: int, is_active: bool):
    conn = get_db_conn()
    conn.execute("UPDATE zone_rules SET is_active = ? WHERE id = ?", (1 if is_active else 0, int(rule_id)))
    conn.commit()
    conn.close()


def update_video_source_last_seen(*, source_id: int, last_seen: float):
    conn = get_db_conn()
    conn.execute("UPDATE video_sources SET last_seen = ? WHERE id = ?", (last_seen, source_id))
    conn.commit()
    conn.close()


def load_system_settings():
    conn = get_db_conn()
    rows = conn.execute("SELECT key, value, updated_at FROM system_settings ORDER BY key ASC").fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}


def set_system_setting(*, key: str, value: str):
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO system_settings (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, value, time.time()),
    )
    conn.commit()
    conn.close()


def load_worker_statuses():
    conn = get_db_conn()
    rows = conn.execute(
        """
        SELECT source_id, status, is_connected, last_heartbeat, last_frame_at, fps, reconnect_count, last_error, last_snapshot_path, updated_at
        FROM worker_status
        ORDER BY source_id ASC
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def upsert_worker_status(
    *,
    source_id: int,
    status: str,
    is_connected: bool,
    last_heartbeat: float,
    last_frame_at,
    fps: float,
    reconnect_count: int,
    last_error: str,
    last_snapshot_path: str,
):
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO worker_status (
            source_id, status, is_connected, last_heartbeat, last_frame_at, fps,
            reconnect_count, last_error, last_snapshot_path, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            status = excluded.status,
            is_connected = excluded.is_connected,
            last_heartbeat = excluded.last_heartbeat,
            last_frame_at = excluded.last_frame_at,
            fps = excluded.fps,
            reconnect_count = excluded.reconnect_count,
            last_error = excluded.last_error,
            last_snapshot_path = excluded.last_snapshot_path,
            updated_at = excluded.updated_at
        """,
        (
            source_id,
            status,
            1 if is_connected else 0,
            last_heartbeat,
            last_frame_at,
            fps,
            reconnect_count,
            last_error,
            last_snapshot_path,
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def create_experiment_run(*, run_key: str, scenario_name: str, source_path: str, notes: str = "", status: str = "running"):
    conn = get_db_conn()
    cursor = conn.execute(
        """
        INSERT INTO experiment_runs (run_key, scenario_name, source_path, notes, created_at, completed_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_key, scenario_name, source_path, notes, time.time(), None, status),
    )
    conn.commit()
    run_id = cursor.lastrowid
    conn.close()
    return run_id


def complete_experiment_run(*, run_id: int, status: str = "completed"):
    conn = get_db_conn()
    conn.execute(
        """
        UPDATE experiment_runs
        SET completed_at = ?, status = ?
        WHERE id = ?
        """,
        (time.time(), status, run_id),
    )
    conn.commit()
    conn.close()


def insert_benchmark_result(
    *,
    run_id: int,
    model_name: str,
    tracker_type: str,
    frame_limit: int,
    warmup_frames: int,
    frames_processed: int,
    avg_latency_ms: float,
    p95_latency_ms: float,
    avg_fps: float,
    avg_detections_per_frame: float,
    tracked_frame_ratio: float,
    detection_count_total: int,
    metadata: dict | None = None,
):
    conn = get_db_conn()
    cursor = conn.execute(
        """
        INSERT INTO benchmark_results (
            run_id, model_name, tracker_type, frame_limit, warmup_frames, frames_processed,
            avg_latency_ms, p95_latency_ms, avg_fps, avg_detections_per_frame,
            tracked_frame_ratio, detection_count_total, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            model_name,
            tracker_type,
            frame_limit,
            warmup_frames,
            frames_processed,
            avg_latency_ms,
            p95_latency_ms,
            avg_fps,
            avg_detections_per_frame,
            tracked_frame_ratio,
            detection_count_total,
            json.dumps(metadata or {}, ensure_ascii=False),
            time.time(),
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def load_experiment_runs(limit: int | None = None):
    conn = get_db_conn()
    query = """
        SELECT id, run_key, scenario_name, source_path, notes, created_at, completed_at, status
        FROM experiment_runs
        ORDER BY created_at DESC
    """
    params = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def load_benchmark_results(*, run_id: int | None = None):
    conn = get_db_conn()
    query = """
        SELECT
            benchmark_results.id,
            benchmark_results.run_id,
            experiment_runs.run_key,
            experiment_runs.scenario_name,
            experiment_runs.source_path,
            benchmark_results.model_name,
            benchmark_results.tracker_type,
            benchmark_results.frame_limit,
            benchmark_results.warmup_frames,
            benchmark_results.frames_processed,
            benchmark_results.avg_latency_ms,
            benchmark_results.p95_latency_ms,
            benchmark_results.avg_fps,
            benchmark_results.avg_detections_per_frame,
            benchmark_results.tracked_frame_ratio,
            benchmark_results.detection_count_total,
            benchmark_results.metadata_json,
            benchmark_results.created_at
        FROM benchmark_results
        JOIN experiment_runs ON experiment_runs.id = benchmark_results.run_id
    """
    params = ()
    if run_id is not None:
        query += " WHERE benchmark_results.run_id = ?"
        params = (run_id,)
    query += " ORDER BY benchmark_results.created_at DESC, benchmark_results.id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    results = []
    for row in rows:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            item["metadata"] = {}
        results.append(item)
    return results


def reset_and_seed_demo_data(*, employee_count: int = 120, visit_count: int = 900, seed: int = 42):
    """Recreate the database contents with a large deterministic enterprise demo dataset."""
    conn = get_db_conn()
    rng = random.Random(seed)
    now_ts = time.time()

    for table_name in [
        "worker_status",
        "detection_events",
        "access_logs",
        "events",
        "frames",
        "sessions",
        "video_sources",
        "employees",
        "access_points",
        "system_settings",
    ]:
        conn.execute(f"DELETE FROM {table_name}")
    conn.execute(
        "DELETE FROM sqlite_sequence WHERE name IN ('employees', 'access_points', 'access_logs', 'frames', 'video_sources')"
    )

    for key, value in SYSTEM_SETTING_DEFAULTS.items():
        conn.execute(
            """
            INSERT INTO system_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, value, now_ts),
        )

    access_points = [
        ("Главная проходная", "Центральный вход", "Основная точка прохода сотрудников"),
        ("Административный вход", "Корпус A", "Поток административного персонала"),
        ("Служебный вход", "Производственный блок", "Служебный доступ дежурных смен"),
        ("Складской вход", "Логистическая зона", "Контроль прохода в складской сектор"),
    ]
    access_point_ids = []
    for row in access_points:
        cursor = conn.execute(
            "INSERT INTO access_points (name, location, description) VALUES (?, ?, ?)",
            row,
        )
        access_point_ids.append(cursor.lastrowid)

    conn.execute(
        """
        UPDATE system_settings
        SET value = ?, updated_at = ?
        WHERE key = 'active_access_point_id'
        """,
        (str(access_point_ids[0]), now_ts),
    )

    male_last_names = [
        "Иванов", "Петров", "Сидоров", "Кузнецов", "Смирнов", "Волков", "Морозов", "Соколов",
        "Новиков", "Егоров", "Орлов", "Федоров", "Макаров", "Павлов", "Захаров", "Белов",
        "Тарасов", "Громов", "Андреев", "Никитин", "Комаров", "Лебедев", "Матвеев", "Бобров",
    ]
    female_last_names = [
        "Иванова", "Петрова", "Сидорова", "Кузнецова", "Смирнова", "Волкова", "Морозова", "Соколова",
        "Новикова", "Егорова", "Орлова", "Федорова", "Макарова", "Павлова", "Захарова", "Белова",
        "Тарасова", "Громова", "Андреева", "Никитина", "Комарова", "Лебедева", "Матвеева", "Боброва",
    ]
    male_first_names = [
        "Иван", "Петр", "Алексей", "Дмитрий", "Андрей", "Сергей", "Никита", "Максим",
        "Павел", "Владимир", "Михаил", "Константин", "Евгений", "Антон", "Игорь", "Роман",
    ]
    female_first_names = [
        "Анна", "Елена", "Мария", "Ольга", "Наталья", "Татьяна", "Светлана", "Виктория",
        "Ирина", "Дарья", "Екатерина", "Полина", "Людмила", "Ксения", "Алина", "Юлия",
    ]
    male_patronymics = [
        "Иванович", "Петрович", "Сергеевич", "Алексеевич", "Андреевич", "Дмитриевич", "Михайлович",
        "Владимирович", "Николаевич", "Павлович", "Евгеньевич", "Константинович", "Олегович", "Игоревич",
    ]
    female_patronymics = [
        "Ивановна", "Петровна", "Сергеевна", "Алексеевна", "Андреевна", "Дмитриевна", "Михайловна",
        "Владимировна", "Николаевна", "Павловна", "Евгеньевна", "Константиновна", "Олеговна", "Игоревна",
    ]
    departments = [
        "Служба безопасности",
        "ИТ-служба",
        "Администрация",
        "Производственный департамент",
        "Логистический центр",
        "Служба эксплуатации",
        "Отдел кадров",
        "Финансовый блок",
    ]
    positions = [
        "Инженер по эксплуатации",
        "Оператор центра мониторинга",
        "Старший смены",
        "Специалист по безопасности",
        "Менеджер административного блока",
        "Системный администратор",
        "Контролер доступа",
        "Логист-координатор",
    ]
    employee_statuses = ["active", "active", "active", "inactive", "on_leave", "blocked"]
    employee_ids = []
    for index in range(employee_count):
        is_female = index % 2 == 1
        if is_female:
            last_name = female_last_names[index % len(female_last_names)]
            first_name = female_first_names[(index * 3) % len(female_first_names)]
            middle_name = female_patronymics[(index * 5) % len(female_patronymics)]
        else:
            last_name = male_last_names[index % len(male_last_names)]
            first_name = male_first_names[(index * 3) % len(male_first_names)]
            middle_name = male_patronymics[(index * 5) % len(male_patronymics)]
        full_name = build_employee_full_name(last_name, first_name, middle_name)
        hire_date = now_ts - rng.randint(120, 2400) * 86400
        cursor = conn.execute(
            """
            INSERT INTO employees (
                full_name, last_name, first_name, middle_name, employee_number,
                department, position, status, created_at, hire_date,
                source_system, profile_photo_url, reference_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                full_name,
                last_name,
                first_name,
                middle_name,
                f"EMP-{10000 + index}",
                rng.choice(departments),
                rng.choice(positions),
                rng.choice(employee_statuses),
                now_ts - rng.randint(10, 180) * 86400,
                hire_date,
                "local",
                "",
                0,
            ),
        )
        employee_ids.append(cursor.lastrowid)

    video_sources = [
        {
            "name": "Камера центрального входа",
            "source_type": "rtsp",
            "source_url": "rtsp://demo-main",
            "location": "Центральный вход",
            "is_active": 1,
            "last_seen": now_ts - 25,
            "description": "Основной производственный поток",
            "config": {"enable_roi": True, "roi_x": 18, "roi_y": 18, "roi_w": 56, "roi_h": 62},
        },
        {
            "name": "Камера административного входа",
            "source_type": "rtsp",
            "source_url": "rtsp://demo-admin",
            "location": "Корпус A",
            "is_active": 1,
            "last_seen": now_ts - 70,
            "description": "Вторичный поток доступа",
            "config": {"enable_roi": True, "roi_x": 24, "roi_y": 22, "roi_w": 48, "roi_h": 54},
        },
        {
            "name": "USB пост охраны",
            "source_type": "usb_camera",
            "source_url": "0",
            "location": "Пост охраны",
            "is_active": 1,
            "last_seen": now_ts - 42,
            "description": "Локальная камера сервера",
            "config": {"enable_roi": False},
        },
        {
            "name": "Браузер оператора",
            "source_type": "browser_camera",
            "source_url": "browser_camera",
            "location": "АРМ оператора",
            "is_active": 1,
            "last_seen": None,
            "description": "Клиентский поток оператора",
            "config": {"enable_roi": False},
        },
        {
            "name": "iPhone Safari Camera",
            "source_type": "browser_camera",
            "source_url": "browser_camera",
            "location": "mobile",
            "is_active": 1,
            "last_seen": None,
            "description": "Мобильная браузерная камера iPhone",
            "config": {"enable_roi": False},
        },
    ]
    source_ids = []
    for source_row in video_sources:
        config = normalize_source_processing_config(source_row.get("config"))
        cursor = conn.execute(
            """
            INSERT INTO video_sources (
                name, source_type, source_url, location, is_active, last_seen, description, created_at,
                enable_roi, roi_x, roi_y, roi_w, roi_h,
                rule_count_enabled, rule_n, rule_t, rule_disappear_enabled, rule_disappear_seconds,
                prolonged_presence_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_row["name"],
                source_row["source_type"],
                source_row["source_url"],
                source_row["location"],
                source_row["is_active"],
                source_row["last_seen"],
                source_row["description"],
                now_ts - rng.randint(1, 15) * 86400,
                1 if config["enable_roi"] else 0,
                config["roi_x"],
                config["roi_y"],
                config["roi_w"],
                config["roi_h"],
                1 if config["rule_count_enabled"] else 0,
                config["rule_n"],
                config["rule_t"],
                1 if config["rule_disappear_enabled"] else 0,
                config["rule_disappear_seconds"],
                config["prolonged_presence_seconds"],
            ),
        )
        source_ids.append(cursor.lastrowid)

    worker_rows = [
        (source_ids[0], "online", 1, now_ts - 5, now_ts - 8, 12.8, 1, "", "", now_ts),
        (source_ids[1], "online", 1, now_ts - 18, now_ts - 24, 9.6, 3, "", "", now_ts),
        (source_ids[2], "online", 1, now_ts - 11, now_ts - 15, 7.4, 2, "", "", now_ts),
        (source_ids[3], "standby", 0, None, None, 0.0, 0, "", "", now_ts),
        (source_ids[4], "standby", 0, None, None, 0.0, 0, "", "", now_ts),
    ]
    conn.executemany(
        """
        INSERT INTO worker_status (
            source_id, status, is_connected, last_heartbeat, last_frame_at, fps,
            reconnect_count, last_error, last_snapshot_path, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        worker_rows,
    )

    session_ids = []
    for source_id in source_ids[:2]:
        for day_offset in range(14):
            started_at = now_ts - day_offset * 86400 - rng.randint(600, 36000)
            session_id = f"worker-{source_id}-{uuid.uuid4().hex[:8]}"
            session_ids.append((session_id, source_id, started_at))
            conn.execute(
                """
                INSERT INTO sessions (
                    id, model, source_type, source_path, animal_filter, class_filter,
                    rotation_angle, started_at, finished_at, total_frames, processed_frames, events_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    "yolov8s.pt",
                    "rtsp",
                    f"rtsp://source-{source_id}",
                    "всё",
                    json.dumps(["person"], ensure_ascii=False),
                    0,
                    started_at,
                    started_at + rng.randint(1800, 14400),
                    rng.randint(800, 3000),
                    rng.randint(700, 2800),
                    rng.randint(40, 220),
                ),
            )

    suspicious_types = ["prolonged_presence_near_entry", "unknown_person_detected", "repeated_entry_attempt"]
    raw_types = ["object_detected", "roi_enter", "roi_exit", "object_disappeared"]
    domain_types = [
        "person_detected_near_entry",
        "person_entered_entry_zone",
        "person_left_entry_zone",
    ]

    for visit_index in range(visit_count):
        session_id, source_id, started_at = session_ids[visit_index % len(session_ids)]
        event_ts = started_at + rng.randint(30, 7200)
        employee_id = employee_ids[visit_index % len(employee_ids)] if rng.random() > 0.18 else None
        access_point_id = access_point_ids[source_id % len(access_point_ids)]
        confidence = round(rng.uniform(0.72, 0.98), 3)
        base_track = visit_index % 300 + 1

        access_cursor = conn.execute(
            """
            INSERT INTO access_logs (employee_id, timestamp, access_point_id, event_type, confidence, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                employee_id,
                event_ts,
                access_point_id,
                "person_entered_entry_zone",
                confidence,
                "Синтетическая демонстрационная запись прохода",
            ),
        )
        access_log_id = access_cursor.lastrowid

        domain_events = [
            ("person_detected_near_entry", event_ts - 3, "Человек обнаружен рядом со входной зоной"),
            ("person_entered_entry_zone", event_ts, "Зафиксирован вход в зону прохода"),
            ("person_left_entry_zone", event_ts + rng.randint(8, 35), "Человек покинул входную зону"),
        ]
        if rng.random() < 0.12:
            domain_events.append(
                (
                    suspicious_types[visit_index % len(suspicious_types)],
                    event_ts + rng.randint(4, 45),
                    "Зафиксировано подозрительное поведение в зоне прохода",
                )
            )

        for event_type, timestamp_value, message in domain_events:
            event_id = uuid.uuid4().hex[:8]
            conn.execute(
                """
                INSERT INTO events (
                    event_id, session_id, event_type, source_type, frame_index, timestamp,
                    class_name, confidence, track_id, animal_group, is_animal, roi_inside,
                    center_x, center_y, frame_width, frame_height, message, event_scope,
                    access_log_id, employee_id, access_point_id, identified_employee_id,
                    identification_confidence, identification_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    event_type,
                    "rtsp",
                    visit_index % 500,
                    timestamp_value,
                    "person",
                    confidence,
                    str(base_track),
                    None,
                    0,
                    1,
                    rng.uniform(100, 580),
                    rng.uniform(80, 380),
                    640,
                    480,
                    message,
                    "domain",
                    access_log_id,
                    employee_id,
                    access_point_id,
                    employee_id if employee_id and rng.random() > 0.25 else None,
                    round(rng.uniform(0.66, 0.95), 3) if employee_id else None,
                    "linked_from_directory" if employee_id and rng.random() > 0.25 else "unknown",
                ),
            )

        for raw_index in range(2):
            event_id = uuid.uuid4().hex[:8]
            raw_type = raw_types[(visit_index + raw_index) % len(raw_types)]
            raw_ts = event_ts - 5 + raw_index * 4
            raw_row = (
                event_id,
                session_id,
                None,
                employee_id,
                access_point_id,
                raw_type,
                "rtsp",
                visit_index % 500,
                raw_ts,
                "person",
                confidence,
                str(base_track),
                1,
                rng.uniform(100, 580),
                rng.uniform(80, 380),
                640,
                480,
                "Синтетическая raw-телеметрия детекции",
                employee_id if employee_id and rng.random() > 0.45 else None,
                round(rng.uniform(0.55, 0.93), 3) if employee_id else None,
                "linked_from_directory" if employee_id and rng.random() > 0.45 else "unknown",
            )
            conn.execute(
                """
                INSERT INTO detection_events (
                    id, session_id, access_log_id, employee_id, access_point_id, event_type,
                    source_type, frame_index, timestamp, class_name, confidence, track_id,
                    roi_inside, center_x, center_y, frame_width, frame_height, message,
                    identified_employee_id, identification_confidence, identification_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                raw_row,
            )
            conn.execute(
                """
                INSERT INTO events (
                    event_id, session_id, event_type, source_type, frame_index, timestamp,
                    class_name, confidence, track_id, animal_group, is_animal, roi_inside,
                    center_x, center_y, frame_width, frame_height, message, event_scope,
                    access_log_id, employee_id, access_point_id, identified_employee_id,
                    identification_confidence, identification_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    raw_type,
                    "rtsp",
                    visit_index % 500,
                    raw_ts,
                    "person",
                    confidence,
                    str(base_track),
                    None,
                    0,
                    1,
                    raw_row[13],
                    raw_row[14],
                    640,
                    480,
                    raw_row[17],
                    "raw",
                    None,
                    employee_id,
                    access_point_id,
                    raw_row[18],
                    raw_row[19],
                    raw_row[20],
                ),
            )

    for offline_index in range(24):
        event_id = uuid.uuid4().hex[:8]
        timestamp_value = now_ts - offline_index * 21600
        session_id, _, _ = session_ids[offline_index % len(session_ids)]
        conn.execute(
            """
            INSERT INTO events (
                event_id, session_id, event_type, source_type, frame_index, timestamp,
                class_name, confidence, track_id, animal_group, is_animal, roi_inside,
                center_x, center_y, frame_width, frame_height, message, event_scope,
                access_log_id, employee_id, access_point_id, identified_employee_id,
                identification_confidence, identification_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                session_id,
                "stream_offline" if offline_index % 2 == 0 else "camera_reconnected",
                "rtsp",
                0,
                timestamp_value,
                "camera",
                0.0 if offline_index % 2 == 0 else 1.0,
                None,
                None,
                0,
                0,
                None,
                None,
                640,
                480,
                "Синтетическое служебное событие состояния потока",
                "domain",
                None,
                None,
                access_point_ids[offline_index % len(access_point_ids)],
                None,
                None,
                "unknown",
            ),
        )

    conn.commit()
    conn.close()
    return {
        "employees": employee_count,
        "access_points": len(access_points),
        "video_sources": len(video_sources),
        "visits": visit_count,
        "sessions": len(session_ids),
    }
