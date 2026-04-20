import json
import os
import sqlite3
import time


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(APP_DIR, "monitoring.db")
DB_PATH = os.getenv("MONITORING_DB_PATH", DEFAULT_DB_PATH)

if not os.access(os.path.dirname(DB_PATH) or ".", os.W_OK):
    DB_PATH = "/tmp/monitoring.db"


def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


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
            department TEXT,
            position TEXT,
            status TEXT,
            created_at REAL
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

    # Migration-safe column checks for databases created by older app versions.
    _ensure_columns(
        conn,
        "events",
        [
            ("event_scope", "event_scope TEXT DEFAULT 'raw'"),
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
            ("department", "department TEXT"),
            ("position", "position TEXT"),
            ("status", "status TEXT"),
            ("created_at", "created_at REAL"),
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
            access_log_id, employee_id, access_point_id, identified_employee_id,
            identification_confidence, identification_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            access_log_id,
            event.get("employee_id"),
            event.get("access_point_id"),
            event.get("identified_employee_id"),
            event.get("identification_confidence"),
            event.get("identification_status", "not_configured"),
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                event.get("identification_status", "not_configured"),
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
                "access_log_id": row["access_log_id"] if "access_log_id" in row.keys() else None,
                "employee_id": row["employee_id"] if "employee_id" in row.keys() else None,
                "access_point_id": row["access_point_id"] if "access_point_id" in row.keys() else None,
                "identified_employee_id": row["identified_employee_id"] if "identified_employee_id" in row.keys() else None,
                "identification_confidence": row["identification_confidence"] if "identification_confidence" in row.keys() else None,
                "identification_status": row["identification_status"] if "identification_status" in row.keys() else "not_configured",
            }
        )

    return sessions, events


def load_employees():
    conn = get_db_conn()
    rows = conn.execute(
        """
        SELECT id, full_name, department, position, status, created_at
        FROM employees
        ORDER BY full_name ASC
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def create_employee(*, full_name: str, department: str, position: str, status: str):
    conn = get_db_conn()
    conn.execute(
        """
        INSERT INTO employees (full_name, department, position, status, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            full_name.strip(),
            department.strip(),
            position.strip(),
            status.strip(),
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def update_employee(*, employee_id: int, full_name: str, department: str, position: str, status: str):
    conn = get_db_conn()
    conn.execute(
        """
        UPDATE employees
        SET full_name = ?, department = ?, position = ?, status = ?
        WHERE id = ?
        """,
        (
            full_name.strip(),
            department.strip(),
            position.strip(),
            status.strip(),
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

    demo_rows = [
        ("Иванов Иван Иванович", "Служба эксплуатации", "Инженер", "active", time.time()),
        ("Петров Петр Сергеевич", "Отдел безопасности", "Оператор", "active", time.time()),
        ("Сидорова Анна Викторовна", "Администрация", "Менеджер", "inactive", time.time()),
    ]
    conn.executemany(
        """
        INSERT INTO employees (full_name, department, position, status, created_at)
        VALUES (?, ?, ?, ?, ?)
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
