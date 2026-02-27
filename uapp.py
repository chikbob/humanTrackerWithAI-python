import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time
import uuid  # для ідентифікаторів сеансів
from datetime import datetime
import pandas as pd

# === ІНІЦІАЛІЗАЦІЯ "ПСЕВДО-БД" У session_state ===
if "sessions" not in st.session_state:
    # список усіх сеансів розпізнавання за поточний запуск застосунку
    st.session_state.sessions = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# === Налаштування сторінки ===
st.set_page_config(page_title="Розпізнавання об'єктів", layout="centered")
st.markdown(
    """
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 900px; margin: auto;}
        h1 {text-align: center; margin-bottom: 1rem;}
        .stSelectbox div[data-baseweb="select"] input {pointer-events: none;}
        .stSelectbox div[data-baseweb="select"] {cursor: pointer;}
        .stButton>button {width: 100%; border-radius: 10px; font-size: 16px;}
        .stRadio>div {justify-content: center;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 Розпізнавання об'єктів")

# === Вибір моделі (аналог сутності Models) ===
model_choice = st.selectbox(
    "Оберіть модель YOLO для розпізнавання",
    options=[
        "yolov8n.pt (найшвидша, базова)",
        "yolov8s.pt (збалансована)",
        "yolov8m.pt (точна, але повільніша)"
    ],
    index=1
)
model_map = {
    "yolov8n.pt (найшвидша, базова)": "yolov8n.pt",
    "yolov8s.pt (збалансована)": "yolov8s.pt",
    "yolov8m.pt (точна, але повільніша)": "yolov8m.pt"
}
model = YOLO(model_map[model_choice])

# === Джерело ===
source_mode = st.radio(
    "Оберіть джерело:",
    options=["📷 Вебкамера", "📁 Завантажити фото", "🎞️ Завантажити відео"],
    horizontal=True
)

# === Фільтр за тваринами (AnimalFilters/Classes) ===
animal_filter = st.selectbox(
    "Показувати лише:",
    options=["усе", "коти", "собаки", "птахи", "інші"],
    index=0
)
animal_classes = {
    "коти": ["cat"],
    "собаки": ["dog"],
    "птахи": ["bird"],
    "інші": ["horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"]
}

# === Побудова метаданих класів поточної моделі (сутність Classes) ===
all_class_names = list(model.names.values())
class_meta = {name: {"is_animal": False, "animal_group": None} for name in all_class_names}

for group_name, names in animal_classes.items():
    for name in names:
        if name in class_meta:
            class_meta[name]["is_animal"] = True
            class_meta[name]["animal_group"] = group_name


def get_class_meta(cls_name: str):
    meta = class_meta.get(cls_name, {})
    return meta.get("is_animal", False), meta.get("animal_group")


# === Кнопки обертання зображення ===
st.markdown("### 🔄 Обертання зображення")
col1, col2, col3, col4 = st.columns(4)
if "rotation_angle" not in st.session_state:
    st.session_state.rotation_angle = 0

with col1:
    if st.button("↪️ 90° вліво"):
        st.session_state.rotation_angle = (st.session_state.rotation_angle - 90) % 360
with col2:
    if st.button("↕️ 180°"):
        st.session_state.rotation_angle = (st.session_state.rotation_angle + 180) % 360
with col3:
    if st.button("↩️ 90° вправо"):
        st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360
with col4:
    if st.button("🔄 Скинути"):
        st.session_state.rotation_angle = 0


# === ФУНКЦІЇ ДЛЯ ОБРОБКИ КАДРІВ ТА ЛОГУВАННЯ (Sessions, Frames, Detections) ===
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
    """Малює стильну рамку та великий текст"""
    x1, y1, x2, y2 = map(int, box)
    h, w, _ = img.shape
    y1 = max(0, y1)
    x1 = max(0, x1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    color = (0, 255, 127)
    thickness = 3

    # рамка з тінями
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 50, 0), thickness + 3)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # текст великий
    label_text = f"{label} {conf:.2f}"
    font_scale = max(1.2, min(3, w / 500))
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
    text_x = x1
    text_y = max(text_h + 15, y1 - 10)

    overlay = img.copy()
    cv2.rectangle(overlay, (text_x - 5, text_y - text_h - 10),
                  (text_x + text_w + 10, text_y + 5), color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, label_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)

    return img


def start_session(source_type: str, source_path: str | None = None):
    """Створює новий сеанс (аналог запису в таблиці Sessions)"""
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "model": model_map[model_choice],
        "source_type": source_type,           # image / video / webcam
        "source_path": source_path,          # шлях до файлу або camera:N
        "animal_filter": animal_filter,
        "rotation_angle": st.session_state.rotation_angle,
        "started_at": time.time(),
        "finished_at": None,
        "total_frames": 0,
        "processed_frames": 0,
        "frames": []                         # список кадрів (Frames)
    }
    st.session_state.current_session_id = session_id
    st.session_state.sessions.append(session)
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
    """Позначає завершення поточного сеансу, підраховує кадри"""
    session = get_current_session()
    if session and session["finished_at"] is None:
        session["finished_at"] = time.time()
        session["total_frames"] = len(session["frames"])
        session["processed_frames"] = len(session["frames"])


def log_frame(frame_index: int, frame_shape, processing_time_ms: float, detections_meta: list[dict]):
    """
    Додає інформацію про кадр і детекції (аналог таблиць Frames та Detections).
    detections_meta – список словників:
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


# === Основне вікно ===
st.markdown("---")
frame_display = st.empty()

# === Фото ===
if source_mode == "📁 Завантажити фото":
    uploaded_image = st.file_uploader("Завантажте зображення", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        # Новий сеанс для одного зображення
        start_session(source_type="image", source_path=uploaded_image.name)

        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)
        img_array = rotate_frame(img_array)

        frame_index = 0
        t0 = time.time()
        results = model.predict(img_array, imgsz=640, conf=0.5, verbose=False)
        t1 = time.time()
        processing_time_ms = (t1 - t0) * 1000

        frame_rgb = img_array.copy()
        detections_meta = []

        for r in results:
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                cls_id = int(r.boxes.cls[i])
                cls_name = model.names[cls_id]
                conf = float(r.boxes.conf[i])

                x1, y1, x2, y2 = map(int, box)
                is_animal, animal_group = get_class_meta(cls_name)

                # Запис в "БД"
                detections_meta.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "is_animal": is_animal,
                    "animal_group": animal_group,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })

                # Фільтрація лише для відображення
                if animal_filter != "усе":
                    allowed = animal_classes.get(animal_filter, [])
                    if cls_name not in allowed:
                        continue

                frame_rgb = draw_fancy_box(frame_rgb, box, cls_name, conf)

        log_frame(frame_index, frame_rgb.shape, processing_time_ms, detections_meta)
        finish_session()

        st.image(frame_rgb, channels="RGB")

# === Відео ===
# === Відео ===
elif source_mode == "🎞️ Завантажити відео":
    uploaded_video = st.file_uploader("Завантажте відео", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # 1) Створюємо сеанс для відео (ВАЖЛИВО)
        start_session(source_type="video", source_path=uploaded_video.name)

        # 2) Створюємо тимчасовий файл
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        temp_video.flush()
        temp_path = temp_video.name
        temp_video.close()  # закриваємо файл, щоб Windows зняв блокування

        cap = cv2.VideoCapture(temp_path)
        st.info("▶️ Обробка відео...")

        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = rotate_frame(frame)
            t0 = time.time()
            results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
            t1 = time.time()
            processing_time_ms = (t1 - t0) * 1000

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections_meta = []

            for r in results:
                for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                    cls_id = int(r.boxes.cls[i])
                    cls_name = model.names[cls_id]
                    conf = float(r.boxes.conf[i])

                    x1, y1, x2, y2 = map(int, box)
                    is_animal, animal_group = get_class_meta(cls_name)

                    detections_meta.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "is_animal": is_animal,
                        "animal_group": animal_group,
                        "confidence": conf,
                        "box": [x1, y1, x2, y2]
                    })

                    if animal_filter != "усе":
                        allowed = animal_classes.get(animal_filter, [])
                        if cls_name not in allowed:
                            continue

                    frame_rgb = draw_fancy_box(frame_rgb, box, cls_name, conf)

            # лог кадру в "БД сеансу"
            log_frame(frame_index, frame_rgb.shape, processing_time_ms, detections_meta)
            frame_index += 1

            frame_display.image(frame_rgb, channels="RGB")

        cap.release()

        # безпечне видалення файлу
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except PermissionError:
            pass

        # завершуємо сеанс
        finish_session()
        st.success("✅ Відео оброблено.")

# === Камера ===
elif source_mode == "📷 Вебкамера":
    camera_index = st.number_input("Номер камери (0 за замовчуванням)", min_value=0, step=1, value=0)
    start_button = st.button("▶️ Запустити розпізнавання")
    stop_button = st.button("⏹ Зупинити")

    if "running" not in st.session_state:
        st.session_state.running = False
    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False

    if st.session_state.running:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            st.error("❌ Не вдалося відкрити камеру.")
            st.session_state.running = False
        else:
            # Новий сеанс для вебкамери
            start_session(source_type="webcam", source_path=f"camera:{camera_index}")

            st.info("✅ Камеру запущено. Натисніть ⏹, щоб зупинити.")
            prev_time = time.time()
            frame_index = 0

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Кадр не отримано.")
                    break

                frame = rotate_frame(frame)
                t0 = time.time()
                results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
                t1 = time.time()
                processing_time_ms = (t1 - t0) * 1000

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections_meta = []

                for r in results:
                    for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                        cls_id = int(r.boxes.cls[i])
                        cls_name = model.names[cls_id]
                        conf = float(r.boxes.conf[i])

                        x1, y1, x2, y2 = map(int, box)
                        is_animal, animal_group = get_class_meta(cls_name)

                        detections_meta.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "is_animal": is_animal,
                            "animal_group": animal_group,
                            "confidence": conf,
                            "box": [x1, y1, x2, y2]
                        })

                        if animal_filter != "усе":
                            allowed = animal_classes.get(animal_filter, [])
                            if cls_name not in allowed:
                                continue

                        frame_rgb = draw_fancy_box(frame_rgb, box, cls_name, conf)

                log_frame(frame_index, frame_rgb.shape, processing_time_ms, detections_meta)
                frame_index += 1

                if time.time() - prev_time > 0.1:
                    frame_display.image(frame_rgb, channels="RGB")
                    prev_time = time.time()

            cap.release()
            finish_session()
            st.session_state.running = False
            st.success("🛑 Розпізнавання зупинено.")
    else:
        st.warning("Натисніть ▶️, щоб почати розпізнавання.")


# === ТАБЛИЧНИЙ ВИВІД ІНФОРМАЦІЇ ПРО СЕАНСИ, КАДРИ ТА ДЕТЕКЦІЇ ===
with st.expander("📊 Статистика сеансів розпізнавання (поточний запуск)"):
    sessions = st.session_state.sessions
    if not sessions:
        st.info("Поки що немає жодного сеансу розпізнавання.")
    else:
        # ---- Таблиця сеансів ----
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
                "ID (скорочено)": s["id"][:8],
                "Модель": s["model"],
                "Джерело": s["source_type"],
                "Шлях / камера": s["source_path"],
                "Фільтр тварин": s["animal_filter"],
                "Кут обертання": s["rotation_angle"],
                "Кадрів у сеансі": len(s["frames"]),
                "Початок": started,
                "Кінець": finished,
                "Тривалість, с": round(duration, 2) if duration is not None else ""
            })

        st.subheader("Сеанси")
        df_sessions = pd.DataFrame(sessions_summary)
        st.dataframe(df_sessions, use_container_width=True)

        # ---- Вибір сеансу для детальнішого перегляду ----
        session_index = st.number_input(
            "Оберіть номер сеансу для деталізації",
            min_value=1,
            max_value=len(sessions),
            value=len(sessions),
            step=1
        )
        sel_session = sessions[session_index - 1]

        # ---- Таблиця кадрів обраного сеансу ----
        frames = sel_session["frames"]
        if frames:
            frames_summary = []
            for f in frames:
                ts = datetime.fromtimestamp(f["timestamp"]).strftime("%H:%M:%S")
                frames_summary.append({
                    "Кадр": f["frame_index"],
                    "Час кадру": ts,
                    "Розмір (W×H)": f"{f['width']}×{f['height']}",
                    "Кут": f["rotation_angle"],
                    "Час обробки, мс": round(f["processing_time_ms"], 2),
                    "К-сть детекцій": f["detections_count"]
                })

            st.subheader("Кадри обраного сеансу")
            df_frames = pd.DataFrame(frames_summary)
            st.dataframe(df_frames, use_container_width=True)

            # ---- Зведення по детекціях (кількість по класах) ----
            detections_all = []
            for f in frames:
                for d in f["detections"]:
                    detections_all.append(d)

            if detections_all:
                det_summary = {}
                for d in detections_all:
                    cls_name = d["class_name"]
                    if cls_name not in det_summary:
                        det_summary[cls_name] = {
                            "Клас": cls_name,
                            "Тварина": "так" if d["is_animal"] else "ні",
                            "Група": d["animal_group"] or "",
                            "Кількість": 0
                        }
                    det_summary[cls_name]["Кількість"] += 1

                st.subheader("Зведення по детекціях (обраний сеанс)")
                df_det = pd.DataFrame(list(det_summary.values()))
                df_det = df_det.sort_values("Кількість", ascending=False)
                st.dataframe(df_det, use_container_width=True)
            else:
                st.info("У цьому сеансі не зафіксовано жодної детекції.")
        else:
            st.info("У вибраному сеансі немає кадрів.")