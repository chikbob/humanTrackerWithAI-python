import time
import uuid
from typing import Optional


def ensure_runtime_state(session: dict):
    if "class_filter" not in session:
        session["class_filter"] = []
    if "events_count" not in session:
        session["events_count"] = 0
    if "seen_track_keys" not in session or not isinstance(session["seen_track_keys"], set):
        session["seen_track_keys"] = set()
    if "notified_track_keys" not in session or not isinstance(session["notified_track_keys"], set):
        session["notified_track_keys"] = set()
    if "track_inside_roi" not in session or not isinstance(session["track_inside_roi"], dict):
        session["track_inside_roi"] = {}
    if "track_last_seen" not in session or not isinstance(session["track_last_seen"], dict):
        session["track_last_seen"] = {}
    if "track_class_by_key" not in session or not isinstance(session["track_class_by_key"], dict):
        session["track_class_by_key"] = {}
    if "disappeared_track_keys" not in session or not isinstance(session["disappeared_track_keys"], set):
        session["disappeared_track_keys"] = set()
    if "class_event_times" not in session or not isinstance(session["class_event_times"], dict):
        session["class_event_times"] = {}
    if "rule_last_alert_ts" not in session or not isinstance(session["rule_last_alert_ts"], dict):
        session["rule_last_alert_ts"] = {}
    if "track_first_seen" not in session or not isinstance(session["track_first_seen"], dict):
        session["track_first_seen"] = {}
    if "track_domain_flags" not in session or not isinstance(session["track_domain_flags"], dict):
        session["track_domain_flags"] = {}
    if "track_entry_timestamps" not in session or not isinstance(session["track_entry_timestamps"], dict):
        session["track_entry_timestamps"] = {}


def init_session_state(session_state, load_history_from_db):
    if "history_loaded" not in session_state:
        loaded_sessions, loaded_events = load_history_from_db()
        session_state.sessions = loaded_sessions
        session_state.events = loaded_events
        session_state.history_loaded = True
    if "sessions" not in session_state:
        session_state.sessions = []
    if "events" not in session_state:
        session_state.events = []
    if "current_session_id" not in session_state:
        session_state.current_session_id = None
    if "notifications" not in session_state:
        session_state.notifications = []
    if "running" not in session_state:
        session_state.running = False
    if "current_user_role" not in session_state:
        session_state.current_user_role = "admin"
    if "current_user_name" not in session_state:
        session_state.current_user_name = "Главный оператор"
    for session in session_state.sessions:
        ensure_runtime_state(session)


def get_current_session(session_state):
    sid = session_state.get("current_session_id")
    if not sid:
        return None
    for session in session_state.sessions:
        if session["id"] == sid:
            return session
    return None


def start_session(
    session_state,
    db_upsert_session,
    *,
    model_name: str,
    source_type: str,
    source_path: Optional[str],
    animal_filter: str,
    track_classes: list[str],
    rotation_angle: int,
):
    """Create a new monitoring session with the current UI settings."""
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "model": model_name,
        "source_type": source_type,
        "source_path": source_path,
        "animal_filter": animal_filter,
        "class_filter": track_classes,
        "rotation_angle": rotation_angle,
        "started_at": time.time(),
        "finished_at": None,
        "total_frames": 0,
        "processed_frames": 0,
        "events_count": 0,
        "seen_track_keys": set(),
        "notified_track_keys": set(),
        "track_inside_roi": {},
        "track_last_seen": {},
        "track_class_by_key": {},
        "disappeared_track_keys": set(),
        "class_event_times": {},
        "rule_last_alert_ts": {},
        "track_first_seen": {},
        "track_domain_flags": {},
        "track_entry_timestamps": {},
        "frames": [],
    }
    session_state.current_session_id = session_id
    session_state.sessions.append(session)
    db_upsert_session(session)
    return session


def finish_session(session_state, db_upsert_session):
    session = get_current_session(session_state)
    if session and session["finished_at"] is None:
        session["finished_at"] = time.time()
        session["total_frames"] = len(session["frames"])
        session["processed_frames"] = len(session["frames"])
        db_upsert_session(session)


def log_frame(
    session_state,
    db_insert_frame,
    db_upsert_session,
    *,
    frame_index: int,
    frame_shape,
    processing_time_ms: float,
    detections_meta: list[dict],
    rotation_angle: int,
    persist_interval: int = 10,
    force_session_sync: bool = False,
):
    session = get_current_session(session_state)
    if not session:
        return
    h, w, _ = frame_shape
    frame_record = {
        "frame_index": frame_index,
        "timestamp": time.time(),
        "width": w,
        "height": h,
        "rotation_angle": rotation_angle,
        "processing_time_ms": processing_time_ms,
        "detections_count": len(detections_meta),
        "detections": detections_meta,
    }
    session["frames"].append(frame_record)
    db_insert_frame(session["id"], frame_record)
    if force_session_sync or (persist_interval > 0 and len(session["frames"]) % persist_interval == 0):
        db_upsert_session(session)
