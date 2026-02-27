import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time
import uuid  # для идентификаторов сеансов
import json
import sqlite3
from datetime import datetime
import pandas as pd
from typing import Optional
from collections import Counter

DB_PATH = os.path.join(os.path.dirname(__file__), "monitoring.db")


def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


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
            message TEXT
        )
        """
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
    conn.execute(
        """
        INSERT OR REPLACE INTO events (
            event_id, session_id, event_type, source_type, frame_index, timestamp,
            class_name, confidence, track_id, animal_group, is_animal, roi_inside,
            center_x, center_y, frame_width, frame_height, message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            "frames": []
        }
        sessions_map[session["id"]] = session
        sessions.append(session)

    for row in frame_rows:
        sid = row["session_id"]
        if sid not in sessions_map:
            continue
        sessions_map[sid]["frames"].append({
            "frame_index": row["frame_index"],
            "timestamp": row["timestamp"],
            "width": row["width"],
            "height": row["height"],
            "rotation_angle": row["rotation_angle"],
            "processing_time_ms": row["processing_time_ms"],
            "detections_count": row["detections_count"],
            "detections": []
        })

    events = []
    for row in event_rows:
        events.append({
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
            "message": row["message"] or ""
        })

    return sessions, events


# === ИНИЦИАЛИЗАЦИЯ "ПСЕВДО-БД" В session_state ===
init_db()
if "history_loaded" not in st.session_state:
    loaded_sessions, loaded_events = load_history_from_db()
    st.session_state.sessions = loaded_sessions
    st.session_state.events = loaded_events
    st.session_state.history_loaded = True
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "events" not in st.session_state:
    st.session_state.events = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "running" not in st.session_state:
    st.session_state.running = False
for s in st.session_state.sessions:
    if "class_filter" not in s:
        s["class_filter"] = []
    if "events_count" not in s:
        s["events_count"] = 0
    if "seen_track_keys" not in s or not isinstance(s["seen_track_keys"], set):
        s["seen_track_keys"] = set()
    if "notified_track_keys" not in s or not isinstance(s["notified_track_keys"], set):
        s["notified_track_keys"] = set()
    if "track_inside_roi" not in s or not isinstance(s["track_inside_roi"], dict):
        s["track_inside_roi"] = {}
    if "track_last_seen" not in s or not isinstance(s["track_last_seen"], dict):
        s["track_last_seen"] = {}
    if "track_class_by_key" not in s or not isinstance(s["track_class_by_key"], dict):
        s["track_class_by_key"] = {}
    if "disappeared_track_keys" not in s or not isinstance(s["disappeared_track_keys"], set):
        s["disappeared_track_keys"] = set()
    if "class_event_times" not in s or not isinstance(s["class_event_times"], dict):
        s["class_event_times"] = {}
    if "rule_last_alert_ts" not in s or not isinstance(s["rule_last_alert_ts"], dict):
        s["rule_last_alert_ts"] = {}

# === Настройки страницы ===
st.set_page_config(page_title="Мониторинг и анализ объектов", layout="wide")
st.markdown(
    """
    <style>
        .block-container {padding-top: 0.8rem; padding-bottom: 1rem; max-width: 1400px;}
        h1 {margin-bottom: 0.4rem;}
        .stSelectbox div[data-baseweb="select"] input {pointer-events: none;}
        .stSelectbox div[data-baseweb="select"] {cursor: pointer;}
        .stButton>button {width: 100%; border-radius: 10px; font-size: 16px;}
        .stRadio>div {justify-content: center;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📡 Система мониторинга и интеллектуального анализа объектов")
st.caption("Детекция и трекинг в реальном времени, журнал событий, уведомления, динамика и экспорт отчётов.")

# === Боковая панель конфигурации ===
st.sidebar.header("⚙️ Параметры системы")

model_choice = st.sidebar.selectbox(
    "Модель YOLO",
    options=[
        "yolov8n.pt (самая быстрая, базовая)",
        "yolov8s.pt (сбалансированная)",
        "yolov8m.pt (точная, но медленнее)"
    ],
    index=1
)
model_map = {
    "yolov8n.pt (самая быстрая, базовая)": "yolov8n.pt",
    "yolov8s.pt (сбалансированная)": "yolov8s.pt",
    "yolov8m.pt (точная, но медленнее)": "yolov8m.pt"
}


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return YOLO(model_path)


model = load_model(model_map[model_choice])

source_mode = st.sidebar.radio(
    "Источник данных",
    options=["📷 Веб-камера", "📁 Загрузить фото", "🎞️ Загрузить видео"],
    index=0
)
show_advanced = st.sidebar.checkbox("Расширенные настройки", value=False)

animal_classes = {
    "коты": ["cat"],
    "собаки": ["dog"],
    "птицы": ["bird"],
    "прочие": ["horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"]
}

# === Формирование метаданных классов текущей модели (сущность Classes) ===
all_class_names = list(model.names.values())
class_meta = {name: {"is_animal": False, "animal_group": None} for name in all_class_names}

for group_name, names in animal_classes.items():
    for name in names:
        if name in class_meta:
            class_meta[name]["is_animal"] = True
            class_meta[name]["animal_group"] = group_name

rotation_options = ["0°", "90° вправо", "180°", "90° влево"]
rotation_choice = st.sidebar.selectbox("Поворот изображения", rotation_options, index=0)
rotation_map = {"0°": 0, "90° вправо": 90, "180°": 180, "90° влево": 270}
st.session_state.rotation_angle = rotation_map[rotation_choice]

conf_threshold = st.sidebar.slider("Порог confidence", 0.1, 0.95, 0.5, 0.05)
notify_conf_threshold = st.sidebar.slider("Порог уведомлений", 0.1, 0.95, 0.5, 0.05)
enable_notifications = st.sidebar.checkbox("Включить уведомления", value=True)

notify_classes = st.sidebar.multiselect(
    "Классы для уведомлений",
    options=all_class_names,
    default=[cls for cls in ["person", "keyboard", "mouse", "bottle"] if cls in all_class_names]
)
st.sidebar.markdown("---")
st.sidebar.caption("Шаги: выберите источник -> нажмите старт/загрузите файл -> смотрите события.")

animal_filter = "всё"
track_classes = []
enable_roi = False
roi_x, roi_y, roi_w, roi_h = 20, 20, 60, 60
rule_count_enabled = False
rule_class = "person" if "person" in all_class_names else (all_class_names[0] if all_class_names else "")
rule_n, rule_t = 3, 10
rule_disappear_enabled = False
rule_disappear_seconds = 5

if show_advanced:
    st.sidebar.subheader("Расширенные фильтры")
    animal_filter = st.sidebar.selectbox(
        "Показывать только:",
        options=["всё", "коты", "собаки", "птицы", "прочие"],
        index=0
    )
    track_classes = st.sidebar.multiselect(
        "Фильтр по классам",
        options=all_class_names,
        default=[]
    )

    st.sidebar.subheader("ROI и правила алертов")
    enable_roi = st.sidebar.checkbox("Включить ROI", value=True)
    roi_x = st.sidebar.slider("ROI X (%)", 0, 95, 20, 1)
    roi_y = st.sidebar.slider("ROI Y (%)", 0, 95, 20, 1)
    roi_w = st.sidebar.slider("ROI ширина (%)", 5, 100, 60, 1)
    roi_h = st.sidebar.slider("ROI высота (%)", 5, 100, 60, 1)

    rule_count_enabled = st.sidebar.checkbox("Правило: N объектов класса X за T сек", value=True)
    rule_class = st.sidebar.selectbox(
        "Класс X для правила N/T",
        options=all_class_names,
        index=all_class_names.index("person") if "person" in all_class_names else 0
    )
    rule_n = st.sidebar.number_input("N объектов", min_value=1, max_value=100, value=3, step=1)
    rule_t = st.sidebar.number_input("T секунд", min_value=1, max_value=600, value=10, step=1)

    rule_disappear_enabled = st.sidebar.checkbox("Правило: объект исчез > T сек", value=True)
    rule_disappear_seconds = st.sidebar.number_input(
        "T секунд для исчезновения",
        min_value=1,
        max_value=120,
        value=5,
        step=1
    )


def get_class_meta(cls_name: str):
    meta = class_meta.get(cls_name, {})
    return meta.get("is_animal", False), meta.get("animal_group")


def class_allowed(cls_name: str) -> bool:
    if animal_filter != "всё":
        allowed_animals = animal_classes.get(animal_filter, [])
        if cls_name not in allowed_animals:
            return False
    if track_classes and cls_name not in track_classes:
        return False
    return True


def get_roi_rect(frame_w: int, frame_h: int):
    x1 = int(frame_w * (roi_x / 100.0))
    y1 = int(frame_h * (roi_y / 100.0))
    x2 = int(frame_w * min(1.0, (roi_x + roi_w) / 100.0))
    y2 = int(frame_h * min(1.0, (roi_y + roi_h) / 100.0))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def is_inside_roi(cx: float, cy: float, frame_w: int, frame_h: int) -> bool:
    if not enable_roi:
        return True
    x1, y1, x2, y2 = get_roi_rect(frame_w, frame_h)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def draw_roi_overlay(frame_rgb):
    if not enable_roi:
        return frame_rgb
    h, w, _ = frame_rgb.shape
    x1, y1, x2, y2 = get_roi_rect(w, h)
    overlay = frame_rgb.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 120, 255), -1)
    cv2.addWeighted(overlay, 0.15, frame_rgb, 0.85, 0, frame_rgb)
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (60, 120, 255), 2)
    cv2.putText(frame_rgb, "ROI", (x1 + 6, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 120, 255), 2)
    return frame_rgb


def add_notification(text: str):
    st.session_state.notifications.append({
        "timestamp": time.time(),
        "text": text
    })
    if len(st.session_state.notifications) > 200:
        st.session_state.notifications = st.session_state.notifications[-200:]
    if enable_notifications:
        st.toast(text)


def create_event(
    session: dict,
    event_type: str,
    source_type: str,
    frame_index: int,
    class_name: str = "",
    confidence: Optional[float] = None,
    track_id: Optional[int] = None,
    animal_group: Optional[str] = None,
    is_animal: bool = False,
    roi_inside: bool = False,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    message: str = "",
):
    event = {
        "event_id": str(uuid.uuid4())[:8],
        "session_id": session["id"],
        "event_type": event_type,
        "source_type": source_type,
        "frame_index": frame_index,
        "timestamp": time.time(),
        "class_name": class_name,
        "confidence": confidence if confidence is not None else 0.0,
        "track_id": track_id,
        "animal_group": animal_group,
        "is_animal": is_animal,
        "roi_inside": roi_inside,
        "center_x": center_x,
        "center_y": center_y,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "message": message,
    }
    st.session_state.events.append(event)
    session["events_count"] += 1
    db_insert_event(event)
    return event


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def register_event(frame_index: int, detection: dict, source_type: str):
    session = get_current_session()
    if not session:
        return

    track_id = detection.get("track_id")
    track_key = None
    is_new_track = False
    if track_id is not None:
        track_key = f"{session['id']}:{track_id}:{detection['class_name']}"
        if track_key not in session["seen_track_keys"]:
            session["seen_track_keys"].add(track_key)
            is_new_track = True

    should_store_event = (track_id is None) or is_new_track
    if should_store_event:
        create_event(
            session=session,
            event_type="object_detected",
            source_type=source_type,
            frame_index=frame_index,
            class_name=detection["class_name"],
            confidence=detection["confidence"],
            track_id=track_id,
            animal_group=detection["animal_group"],
            is_animal=detection["is_animal"],
            roi_inside=detection.get("roi_inside", False),
            center_x=detection.get("center_x"),
            center_y=detection.get("center_y"),
            frame_width=detection.get("frame_width"),
            frame_height=detection.get("frame_height"),
            message=f"Обнаружен объект {detection['class_name']}",
        )

    if detection.get("roi_enter"):
        create_event(
            session=session,
            event_type="roi_enter",
            source_type=source_type,
            frame_index=frame_index,
            class_name=detection["class_name"],
            confidence=detection["confidence"],
            track_id=track_id,
            animal_group=detection["animal_group"],
            is_animal=detection["is_animal"],
            roi_inside=True,
            center_x=detection.get("center_x"),
            center_y=detection.get("center_y"),
            frame_width=detection.get("frame_width"),
            frame_height=detection.get("frame_height"),
            message=f"Вход в ROI: {detection['class_name']}",
        )

    if rule_count_enabled and should_store_event and detection["class_name"] == rule_class:
        ts_now = time.time()
        bucket = session["class_event_times"].get(rule_class, [])
        bucket = [ts for ts in bucket if ts_now - ts <= float(rule_t)]
        bucket.append(ts_now)
        session["class_event_times"][rule_class] = bucket
        last_alert = session["rule_last_alert_ts"].get(rule_class, 0)
        if len(bucket) >= int(rule_n) and (ts_now - last_alert) > float(rule_t):
            session["rule_last_alert_ts"][rule_class] = ts_now
            msg = f"Правило N/T: {len(bucket)} объектов класса {rule_class} за {int(rule_t)} сек"
            create_event(
                session=session,
                event_type="rule_count",
                source_type=source_type,
                frame_index=frame_index,
                class_name=rule_class,
                confidence=detection["confidence"],
                track_id=track_id,
                animal_group=detection["animal_group"],
                is_animal=detection["is_animal"],
                roi_inside=detection.get("roi_inside", False),
                center_x=detection.get("center_x"),
                center_y=detection.get("center_y"),
                frame_width=detection.get("frame_width"),
                frame_height=detection.get("frame_height"),
                message=msg,
            )
            if enable_notifications:
                add_notification(msg)

    should_notify_detection = (
        enable_notifications
        and detection["confidence"] >= notify_conf_threshold
        and detection["class_name"] in notify_classes
        and (not enable_roi or detection.get("roi_enter", False))
    )
    if should_notify_detection:
        if track_key is not None:
            if track_key in session["notified_track_keys"]:
                return
            session["notified_track_keys"].add(track_key)
        add_notification(
            f"Событие: {detection['class_name']} (conf={detection['confidence']:.2f}), кадр {frame_index}"
        )


def detect_and_annotate(frame_bgr, frame_index: int, source_type: str, use_tracking: bool):
    session = get_current_session()
    t0 = time.time()
    if use_tracking:
        try:
            results = model.track(
                frame_bgr,
                imgsz=640,
                conf=conf_threshold,
                iou=0.5,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )
        except ModuleNotFoundError:
            st.warning("Трекинг-недоступен: отсутствуют зависимости. Выполняется только детекция.")
            results = model.predict(frame_bgr, imgsz=640, conf=conf_threshold, verbose=False)
    else:
        results = model.predict(frame_bgr, imgsz=640, conf=conf_threshold, verbose=False)
    processing_time_ms = (time.time() - t0) * 1000

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame_rgb.shape
    detections_meta = []
    current_seen_track_keys = set()

    for r in results:
        boxes = r.boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None
        xyxy = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()

        for i, box in enumerate(xyxy):
            cls_id = int(cls_arr[i])
            cls_name = model.names[cls_id]
            conf = float(conf_arr[i])
            track_id = int(ids[i]) if ids is not None else None
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            roi_inside = is_inside_roi(cx, cy, frame_w, frame_h)
            is_animal, animal_group = get_class_meta(cls_name)
            track_key = f"{session['id']}:{track_id}:{cls_name}" if (session and track_id is not None) else None
            prev_inside = session["track_inside_roi"].get(track_key, False) if track_key else False
            roi_enter = bool(roi_inside and (not prev_inside))
            if track_key and session is not None:
                session["track_inside_roi"][track_key] = roi_inside
                session["track_last_seen"][track_key] = time.time()
                session["track_class_by_key"][track_key] = cls_name
                current_seen_track_keys.add(track_key)
                if track_key in session["disappeared_track_keys"]:
                    session["disappeared_track_keys"].remove(track_key)

            detection = {
                "class_id": cls_id,
                "class_name": cls_name,
                "is_animal": is_animal,
                "animal_group": animal_group,
                "confidence": conf,
                "box": [x1, y1, x2, y2],
                "track_id": track_id,
                "center_x": cx,
                "center_y": cy,
                "frame_width": frame_w,
                "frame_height": frame_h,
                "roi_inside": roi_inside,
                "roi_enter": roi_enter if track_id is not None else roi_inside
            }
            detections_meta.append(detection)
            register_event(frame_index=frame_index, detection=detection, source_type=source_type)

            if not class_allowed(cls_name):
                continue
            if enable_roi and not roi_inside:
                continue

            label = f"{cls_name} id:{track_id}" if track_id is not None else cls_name
            frame_rgb = draw_fancy_box(frame_rgb, box, label, conf)

    if session is not None and use_tracking and rule_disappear_enabled:
        now_ts = time.time()
        for track_key, last_seen in list(session["track_last_seen"].items()):
            if now_ts - last_seen > float(rule_disappear_seconds):
                if track_key in session["disappeared_track_keys"]:
                    continue
                session["disappeared_track_keys"].add(track_key)
                disappeared_class = session["track_class_by_key"].get(track_key, "")
                msg = f"Объект исчез > {int(rule_disappear_seconds)} сек: {disappeared_class}"
                create_event(
                    session=session,
                    event_type="object_disappeared",
                    source_type=source_type,
                    frame_index=frame_index,
                    class_name=disappeared_class,
                    confidence=0.0,
                    track_id=None,
                    animal_group=None,
                    is_animal=False,
                    roi_inside=False,
                    center_x=None,
                    center_y=None,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    message=msg,
                )
                if enable_notifications:
                    add_notification(msg)
        # Очистка старых ключей, чтобы не накапливать бесконечно
        for track_key, last_seen in list(session["track_last_seen"].items()):
            if time.time() - last_seen > float(rule_disappear_seconds) * 20:
                session["track_last_seen"].pop(track_key, None)
                session["track_inside_roi"].pop(track_key, None)
                session["track_class_by_key"].pop(track_key, None)
                session["disappeared_track_keys"].discard(track_key)

    frame_rgb = draw_roi_overlay(frame_rgb)
    return frame_rgb, detections_meta, processing_time_ms


# === ФУНКЦИИ ДЛЯ ОБРАБОТКИ КАДРОВ И ЛОГИРОВАНИЯ (Sessions, Frames, Detections) ===
def rotate_frame(frame):
    angle = st.session_state.rotation_angle
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def draw_fancy_box(img, box, label, conf):
    """Рисует стильную рамку и крупный текст"""
    x1, y1, x2, y2 = map(int, box)
    h, w, _ = img.shape
    y1 = max(0, y1)
    x1 = max(0, x1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    color = (0, 255, 127)
    thickness = 3

    # рамка с «тенью»
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 50, 0), thickness + 3)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # крупный текст
    label_text = f"{label} {conf:.2f}"
    font_scale = max(1.2, min(3, w / 500))
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
    text_x = x1
    text_y = max(text_h + 15, y1 - 10)

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (text_x - 5, text_y - text_h - 10),
        (text_x + text_w + 10, text_y + 5),
        color,
        -1
    )
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(
        img,
        label_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        3
    )

    return img


def start_session(source_type: str, source_path: Optional[str] = None):
    """Создаёт новый сеанс (аналог записи в таблице Sessions)"""
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "model": model_map[model_choice],
        "source_type": source_type,           # image / video / webcam
        "source_path": source_path,          # путь к файлу или camera:N
        "animal_filter": animal_filter,
        "class_filter": track_classes,
        "rotation_angle": st.session_state.rotation_angle,
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
        "frames": []                         # список кадров (Frames)
    }
    st.session_state.current_session_id = session_id
    st.session_state.sessions.append(session)
    db_upsert_session(session)
    return session


def get_current_session():
    sid = st.session_state.get("current_session_id")
    if not sid:
        return None
    for s in st.session_state.sessions:
        if s["id"] == sid:
            return s
    return None


def finish_session():
    """Отмечает завершение текущего сеанса, подсчитывает кадры"""
    session = get_current_session()
    if session and session["finished_at"] is None:
        session["finished_at"] = time.time()
        session["total_frames"] = len(session["frames"])
        session["processed_frames"] = len(session["frames"])
        db_upsert_session(session)


def log_frame(frame_index: int, frame_shape, processing_time_ms: float, detections_meta: list[dict]):
    """
    Добавляет информацию о кадре и детекциях (аналог таблиц Frames и Detections).
    detections_meta – список словарей:
      {
        "class_id", "class_name", "is_animal", "animal_group",
        "confidence", "box": [x1, y1, x2, y2]
      }
    """
    session = get_current_session()
    if not session:
        return
    h, w, _ = frame_shape
    frame_record = {
        "frame_index": frame_index,
        "timestamp": time.time(),
        "width": w,
        "height": h,
        "rotation_angle": st.session_state.rotation_angle,
        "processing_time_ms": processing_time_ms,
        "detections_count": len(detections_meta),
        "detections": detections_meta
    }
    session["frames"].append(frame_record)
    db_insert_frame(session["id"], frame_record)
    db_upsert_session(session)


# === Основное окно ===
st.markdown("---")
work_col, info_col = st.columns([2.2, 1.0], gap="large")

with work_col:
    with st.container(border=True):
        st.subheader("🎯 Обработка потока")
        frame_display = st.empty()

        if source_mode == "📁 Загрузить фото":
            uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"], key="img_uploader")
            if uploaded_image:
                start_session(source_type="image", source_path=uploaded_image.name)

                image = Image.open(uploaded_image).convert("RGB")
                img_array = np.array(image)
                img_array = rotate_frame(img_array)
                frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                frame_rgb, detections_meta, processing_time_ms = detect_and_annotate(
                    frame_bgr,
                    frame_index=0,
                    source_type="image",
                    use_tracking=False
                )

                log_frame(0, frame_rgb.shape, processing_time_ms, detections_meta)
                finish_session()
                frame_display.image(frame_rgb, channels="RGB")
                st.success("Изображение обработано.")
            else:
                st.info("Загрузите файл, чтобы запустить анализ.")

        elif source_mode == "🎞️ Загрузить видео":
            uploaded_video = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"], key="video_uploader")
            if uploaded_video:
                start_session(source_type="video", source_path=uploaded_video.name)
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_video.write(uploaded_video.read())
                temp_video.flush()
                temp_path = temp_video.name
                temp_video.close()

                cap = cv2.VideoCapture(temp_path)
                st.info("▶️ Обработка видео с трекингом...")
                frame_index = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = rotate_frame(frame)
                    frame_rgb, detections_meta, processing_time_ms = detect_and_annotate(
                        frame,
                        frame_index=frame_index,
                        source_type="video",
                        use_tracking=True
                    )
                    log_frame(frame_index, frame_rgb.shape, processing_time_ms, detections_meta)
                    frame_index += 1
                    frame_display.image(frame_rgb, channels="RGB")

                cap.release()
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except PermissionError:
                    pass

                finish_session()
                st.success("✅ Видео обработано.")
            else:
                st.info("Загрузите видеофайл, чтобы начать анализ.")

        elif source_mode == "📷 Веб-камера":
            camera_index = st.number_input("Номер камеры", min_value=0, step=1, value=0, key="cam_index")
            run_col1, run_col2 = st.columns(2)
            with run_col1:
                start_button = st.button("▶️ Запустить", key="webcam_start")
            with run_col2:
                stop_button = st.button("⏹ Остановить", key="webcam_stop")

            if start_button:
                st.session_state.running = True
            if stop_button:
                st.session_state.running = False

            if st.session_state.running:
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    st.error("❌ Не удалось открыть камеру.")
                    st.session_state.running = False
                else:
                    start_session(source_type="webcam", source_path=f"camera:{camera_index}")
                    st.info("✅ Камера запущена. Идёт трекинг объектов.")
                    prev_time = time.time()
                    frame_index = 0

                    while st.session_state.running:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("⚠️ Кадр не получен.")
                            break
                        frame = rotate_frame(frame)
                        frame_rgb, detections_meta, processing_time_ms = detect_and_annotate(
                            frame,
                            frame_index=frame_index,
                            source_type="webcam",
                            use_tracking=True
                        )
                        log_frame(frame_index, frame_rgb.shape, processing_time_ms, detections_meta)
                        frame_index += 1

                        if time.time() - prev_time > 0.1:
                            frame_display.image(frame_rgb, channels="RGB")
                            prev_time = time.time()

                    cap.release()
                    finish_session()
                    st.session_state.running = False
                    st.success("🛑 Распознавание остановлено.")
            else:
                st.info("Нажмите «Запустить», чтобы начать обработку камеры.")

with info_col:
    with st.container(border=True):
        st.subheader("🧭 Быстрый статус")
        st.write(f"Источник: **{source_mode}**")
        st.write(f"Модель: **{model_map[model_choice]}**")
        st.write(f"Порог детекции: **{conf_threshold:.2f}**")
        st.write(f"Порог уведомлений: **{notify_conf_threshold:.2f}**")
        st.write(f"Угол поворота: **{st.session_state.rotation_angle}°**")
        st.write(f"Фильтр животных: **{animal_filter}**")
        st.write(f"Фильтр классов: **{', '.join(track_classes) if track_classes else 'все'}**")

    with st.container(border=True):
        st.subheader("🔔 Алерты")
        if st.session_state.notifications:
            recent = st.session_state.notifications[-8:]
            for n in reversed(recent):
                ts = datetime.fromtimestamp(n["timestamp"]).strftime("%H:%M:%S")
                st.markdown(f"- `{ts}` {n['text']}")
        else:
            st.caption("Уведомлений пока нет.")


# === МОНИТОРИНГ, АНАЛИТИКА И ОТЧЁТЫ ===
st.markdown("---")
st.subheader("📈 Панель мониторинга и аналитика")

sessions = st.session_state.sessions
events = st.session_state.events

total_frames = sum(len(s["frames"]) for s in sessions)
total_events = len(events)
class_counter = Counter(e["class_name"] for e in events)
top_class = class_counter.most_common(1)[0][0] if class_counter else "—"

met1, met2, met3, met4 = st.columns(4)
met1.metric("Сеансов", len(sessions))
met2.metric("Кадров", total_frames)
met3.metric("Событий object detected", total_events)
met4.metric("Топ-класс", top_class)

if st.session_state.notifications:
    with st.expander("🔔 Последние уведомления", expanded=False):
        notif_df = pd.DataFrame([
            {
                "Время": datetime.fromtimestamp(n["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                "Сообщение": n["text"]
            }
            for n in st.session_state.notifications[-20:]
        ])
        st.dataframe(notif_df.iloc[::-1], use_container_width=True, hide_index=True)

if show_advanced:
    tab_sessions, tab_events, tab_export, tab_kpi = st.tabs(
        ["Сеансы", "События и динамика", "Экспорт отчётов", "KPI модели"]
    )
else:
    tab_sessions, tab_events = st.tabs(["Сеансы", "События"])
    tab_export = None
    tab_kpi = None

with tab_sessions:
    if not sessions:
        st.info("Пока нет ни одного сеанса распознавания.")
    else:
        sessions_summary = []
        for idx, s in enumerate(sessions, start=1):
            started = datetime.fromtimestamp(s["started_at"]).strftime("%Y-%m-%d %H:%M:%S")
            finished = (
                datetime.fromtimestamp(s["finished_at"]).strftime("%Y-%m-%d %H:%M:%S")
                if s["finished_at"] is not None else ""
            )
            duration = (
                s["finished_at"] - s["started_at"]
                if s["finished_at"] is not None else None
            )
            sessions_summary.append({
                "№": idx,
                "ID (сокр.)": s["id"][:8],
                "Модель": s["model"],
                "Источник": s["source_type"],
                "Путь / камера": s["source_path"],
                "Фильтр животных": s["animal_filter"],
                "Фильтр классов": ", ".join(s["class_filter"]) if s["class_filter"] else "все",
                "Кадров в сеансе": len(s["frames"]),
                "Событий": s["events_count"],
                "Начало": started,
                "Конец": finished,
                "Длительность, с": round(duration, 2) if duration is not None else ""
            })

        df_sessions = pd.DataFrame(sessions_summary)
        st.dataframe(df_sessions, use_container_width=True, hide_index=True)

        session_index = st.number_input(
            "Выберите номер сеанса для детализации",
            min_value=1,
            max_value=len(sessions),
            value=len(sessions),
            step=1
        )
        sel_session = sessions[session_index - 1]
        frames = sel_session["frames"]

        if frames:
            df_frames = pd.DataFrame([
                {
                    "Кадр": f["frame_index"],
                    "Время кадра": datetime.fromtimestamp(f["timestamp"]).strftime("%H:%M:%S"),
                    "Размер (W×H)": f"{f['width']}×{f['height']}",
                    "Угол": f["rotation_angle"],
                    "Время обработки, мс": round(f["processing_time_ms"], 2),
                    "Кол-во детекций": f["detections_count"]
                }
                for f in frames
            ])
            st.dataframe(df_frames, use_container_width=True, hide_index=True)
        else:
            st.info("В выбранном сеансе нет кадров.")

with tab_events:
    if not events:
        st.info("Журнал событий пока пуст.")
    else:
        df_events = pd.DataFrame([
            {
                "event_id": e["event_id"],
                "session_id": e["session_id"][:8],
                "event_type": e.get("event_type", "object_detected"),
                "source_type": e["source_type"],
                "frame_index": e["frame_index"],
                "timestamp": datetime.fromtimestamp(e["timestamp"]),
                "class_name": e["class_name"],
                "confidence": round(e["confidence"], 3),
                "track_id": e["track_id"] if e["track_id"] is not None else "",
                "animal_group": e["animal_group"] or "",
                "roi_inside": "да" if e.get("roi_inside") else "нет",
                "message": e.get("message", ""),
                "center_x": e.get("center_x"),
                "center_y": e.get("center_y"),
                "frame_width": e.get("frame_width"),
                "frame_height": e.get("frame_height"),
            }
            for e in events
        ])

        if not show_advanced:
            simple_events = df_events[["timestamp", "class_name", "event_type", "message"]].copy()
            simple_events = simple_events.rename(
                columns={
                    "timestamp": "Время",
                    "class_name": "Класс",
                    "event_type": "Тип события",
                    "message": "Описание"
                }
            )
            st.dataframe(simple_events.sort_values("Время", ascending=False), use_container_width=True, hide_index=True)
            st.caption("Включите «Расширенные настройки», чтобы увидеть динамику, тепловую карту и фильтры.")
        else:
            col_evt1, col_evt2, col_evt3 = st.columns(3)
            with col_evt1:
                selected_source = st.selectbox(
                    "Источник событий",
                    options=["все"] + sorted(df_events["source_type"].unique().tolist()),
                    index=0
                )
            with col_evt2:
                selected_classes = st.multiselect(
                    "Классы для динамики",
                    options=sorted(df_events["class_name"].unique().tolist()),
                    default=[]
                )
            with col_evt3:
                selected_event_types = st.multiselect(
                    "Типы событий",
                    options=sorted(df_events["event_type"].unique().tolist()),
                    default=[]
                )

            filtered_events = df_events.copy()
            if selected_source != "все":
                filtered_events = filtered_events[filtered_events["source_type"] == selected_source]
            if selected_classes:
                filtered_events = filtered_events[filtered_events["class_name"].isin(selected_classes)]
            if selected_event_types:
                filtered_events = filtered_events[filtered_events["event_type"].isin(selected_event_types)]

            st.dataframe(
                filtered_events.sort_values("timestamp", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            timeline = filtered_events.copy()
            timeline["minute"] = timeline["timestamp"].dt.floor("min")
            timeline_series = timeline.groupby("minute").size().rename("events")
            st.caption("Динамика количества событий по минутам")
            st.line_chart(timeline_series)

            class_bar = filtered_events["class_name"].value_counts().rename_axis("class_name").to_frame("count")
            st.caption("Распределение событий по классам")
            st.bar_chart(class_bar)

            heat_df = filtered_events.dropna(subset=["center_x", "center_y", "frame_width", "frame_height"])
            if not heat_df.empty:
                heat_size = 96
                heat = np.zeros((heat_size, heat_size), dtype=np.float32)
                for _, row in heat_df.iterrows():
                    fw = max(float(row["frame_width"]), 1.0)
                    fh = max(float(row["frame_height"]), 1.0)
                    nx = min(max(float(row["center_x"]) / fw, 0.0), 1.0)
                    ny = min(max(float(row["center_y"]) / fh, 0.0), 1.0)
                    xi = min(int(nx * (heat_size - 1)), heat_size - 1)
                    yi = min(int(ny * (heat_size - 1)), heat_size - 1)
                    heat[yi, xi] += 1.0

                heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=3, sigmaY=3)
                if float(heat.max()) > 0:
                    heat_norm = (heat / heat.max() * 255.0).astype(np.uint8)
                    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
                    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
                    st.caption("Тепловая карта движения/появлений объектов")
                    st.image(heat_color, use_container_width=False)

if show_advanced and tab_export is not None:
    with tab_export:
        if not sessions and not events:
            st.info("Нет данных для экспорта.")
        else:
            sessions_export = []
            for s in sessions:
                sessions_export.append({
                    "session_id": s["id"],
                    "model": s["model"],
                    "source_type": s["source_type"],
                    "source_path": s["source_path"],
                    "animal_filter": s["animal_filter"],
                    "class_filter": s["class_filter"],
                    "rotation_angle": s["rotation_angle"],
                    "started_at": datetime.fromtimestamp(s["started_at"]).isoformat(),
                    "finished_at": datetime.fromtimestamp(s["finished_at"]).isoformat() if s["finished_at"] else None,
                    "frames_count": len(s["frames"]),
                    "events_count": s["events_count"]
                })
    
            frames_export = []
            for s in sessions:
                for f in s["frames"]:
                    frames_export.append({
                        "session_id": s["id"],
                        "frame_index": f["frame_index"],
                        "timestamp": datetime.fromtimestamp(f["timestamp"]).isoformat(),
                        "width": f["width"],
                        "height": f["height"],
                        "rotation_angle": f["rotation_angle"],
                        "processing_time_ms": round(f["processing_time_ms"], 2),
                        "detections_count": f["detections_count"]
                    })
    
            events_export = []
            for e in events:
                events_export.append({
                    "event_id": e["event_id"],
                    "session_id": e["session_id"],
                    "event_type": e.get("event_type", "object_detected"),
                    "source_type": e["source_type"],
                    "frame_index": e["frame_index"],
                    "timestamp": datetime.fromtimestamp(e["timestamp"]).isoformat(),
                    "class_name": e["class_name"],
                    "confidence": round(e["confidence"], 3),
                    "track_id": e["track_id"],
                    "animal_group": e["animal_group"],
                    "is_animal": e["is_animal"],
                    "roi_inside": e.get("roi_inside"),
                    "center_x": e.get("center_x"),
                    "center_y": e.get("center_y"),
                    "frame_width": e.get("frame_width"),
                    "frame_height": e.get("frame_height"),
                    "message": e.get("message", "")
                })
    
            df_sessions_export = pd.DataFrame(sessions_export)
            df_frames_export = pd.DataFrame(frames_export)
            df_events_export = pd.DataFrame(events_export)
    
            st.download_button(
                "⬇️ Экспорт сессий (CSV)",
                data=df_sessions_export.to_csv(index=False).encode("utf-8"),
                file_name="sessions_report.csv",
                mime="text/csv"
            )
            st.download_button(
                "⬇️ Экспорт кадров (CSV)",
                data=df_frames_export.to_csv(index=False).encode("utf-8"),
                file_name="frames_report.csv",
                mime="text/csv"
            )
            st.download_button(
                "⬇️ Экспорт событий (CSV)",
                data=df_events_export.to_csv(index=False).encode("utf-8"),
                file_name="events_report.csv",
                mime="text/csv"
            )
    
            full_report = {
                "generated_at": datetime.now().isoformat(),
                "sessions": sessions_export,
                "frames": frames_export,
                "events": events_export
            }
            st.download_button(
                "⬇️ Экспорт полного отчёта (JSON)",
                data=json.dumps(full_report, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="full_report.json",
                mime="application/json"
            )

if show_advanced and tab_kpi is not None:
    with tab_kpi:
        st.markdown("### KPI модели: Precision / Recall")
        st.caption("Загрузите изображения и CSV-разметку: image_name,class_name,x1,y1,x2,y2")
        kpi_conf = st.slider("Порог confidence для KPI", 0.1, 0.95, 0.25, 0.05, key="kpi_conf")
        kpi_iou = st.slider("Порог IoU для матчинга", 0.1, 0.95, 0.5, 0.05, key="kpi_iou")
        kpi_images = st.file_uploader(
            "Изображения для валидации",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="kpi_images"
        )
        kpi_labels = st.file_uploader("CSV разметки (ground truth)", type=["csv"], key="kpi_labels")

        if st.button("Рассчитать Precision/Recall", key="run_kpi"):
            if not kpi_images or not kpi_labels:
                st.warning("Нужно загрузить изображения и CSV-разметку.")
            else:
                gt_df = pd.read_csv(kpi_labels)
                required_cols = {"image_name", "class_name", "x1", "y1", "x2", "y2"}
                if not required_cols.issubset(set(gt_df.columns)):
                    st.error("CSV должен содержать колонки: image_name,class_name,x1,y1,x2,y2")
                else:
                    per_class = {}
                    total_tp, total_fp, total_fn = 0, 0, 0

                    for uploaded in kpi_images:
                        image_name = uploaded.name
                        image = Image.open(uploaded).convert("RGB")
                        img_np = np.array(image)
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                        gt_rows = gt_df[gt_df["image_name"] == image_name]
                        gt_by_class = {}
                        for _, row in gt_rows.iterrows():
                            cls = str(row["class_name"])
                            gt_by_class.setdefault(cls, []).append([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])])

                        pred_results = model.predict(img_bgr, imgsz=640, conf=kpi_conf, verbose=False)
                        pred_by_class = {}
                        for r in pred_results:
                            for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                                cls_id = int(r.boxes.cls[i])
                                cls_name = model.names[cls_id]
                                pred_by_class.setdefault(cls_name, []).append([float(v) for v in box.tolist()])

                        all_classes_eval = set(gt_by_class.keys()) | set(pred_by_class.keys())
                        for cls_name in all_classes_eval:
                            gt_boxes = gt_by_class.get(cls_name, [])
                            pr_boxes = pred_by_class.get(cls_name, [])
                            matched_gt = set()
                            tp = 0
                            fp = 0

                            for pb in pr_boxes:
                                best_iou = 0.0
                                best_gt_idx = None
                                for gi, gb in enumerate(gt_boxes):
                                    if gi in matched_gt:
                                        continue
                                    iou = compute_iou(pb, gb)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_gt_idx = gi
                                if best_gt_idx is not None and best_iou >= kpi_iou:
                                    matched_gt.add(best_gt_idx)
                                    tp += 1
                                else:
                                    fp += 1

                            fn = len(gt_boxes) - len(matched_gt)
                            total_tp += tp
                            total_fp += fp
                            total_fn += fn
                            entry = per_class.setdefault(cls_name, {"TP": 0, "FP": 0, "FN": 0})
                            entry["TP"] += tp
                            entry["FP"] += fp
                            entry["FN"] += fn

                    rows = []
                    for cls_name, vals in sorted(per_class.items()):
                        tp = vals["TP"]
                        fp = vals["FP"]
                        fn = vals["FN"]
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        rows.append({
                            "class_name": cls_name,
                            "TP": tp,
                            "FP": fp,
                            "FN": fn,
                            "precision": round(precision, 4),
                            "recall": round(recall, 4),
                        })

                    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

                    met_a, met_b, met_c = st.columns(3)
                    met_a.metric("Overall Precision", f"{overall_precision:.3f}")
                    met_b.metric("Overall Recall", f"{overall_recall:.3f}")
                    met_c.metric("TP / FP / FN", f"{total_tp} / {total_fp} / {total_fn}")

                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
