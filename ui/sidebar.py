MODEL_OPTIONS = [
    "yolov8n.pt (самая быстрая, базовая)",
    "yolov8s.pt (сбалансированная)",
    "yolov8m.pt (точная, но медленнее)",
]

MODEL_MAP = {
    "yolov8n.pt (самая быстрая, базовая)": "yolov8n.pt",
    "yolov8s.pt (сбалансированная)": "yolov8s.pt",
    "yolov8m.pt (точная, но медленнее)": "yolov8m.pt",
}

ANIMAL_CLASSES = {
    "коты": ["cat"],
    "собаки": ["dog"],
    "птицы": ["bird"],
    "прочие": ["horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"],
}

ROTATION_OPTIONS = ["0°", "90° вправо", "180°", "90° влево"]
ROTATION_MAP = {"0°": 0, "90° вправо": 90, "180°": 180, "90° влево": 270}


def render_primary_sidebar(st):
    st.sidebar.header("⚙️ Параметры проходной")
    model_choice = st.sidebar.selectbox("Модель анализа видеопотока", options=MODEL_OPTIONS, index=1)
    source_mode = st.sidebar.radio(
        "Источник данных входной зоны",
        options=["📷 Веб-камера", "📁 Загрузить фото", "🎞️ Загрузить видео"],
        index=0,
    )
    show_advanced = st.sidebar.checkbox("Показать расширенные настройки", value=False)
    rotation_choice = st.sidebar.selectbox("Поворот изображения камеры", ROTATION_OPTIONS, index=0)
    conf_threshold = st.sidebar.slider("Порог уверенности детекции", 0.1, 0.95, 0.5, 0.05)
    notify_conf_threshold = st.sidebar.slider("Порог уведомлений проходной", 0.1, 0.95, 0.5, 0.05)
    inference_size = st.sidebar.selectbox(
        "Размер кадра для инференса",
        options=INFERENCE_SIZE_OPTIONS,
        index=INFERENCE_SIZE_OPTIONS.index(DEFAULT_INFERENCE_SIZE),
    )
    frame_skip = st.sidebar.slider("Пропуск кадров", 0, 5, DEFAULT_FRAME_SKIP, 1)
    enable_notifications = st.sidebar.checkbox("Включить уведомления дежурному", value=True)
    st.session_state.rotation_angle = ROTATION_MAP[rotation_choice]
    return {
        "model_choice": model_choice,
        "source_mode": source_mode,
        "show_advanced": show_advanced,
        "rotation_angle": ROTATION_MAP[rotation_choice],
        "conf_threshold": conf_threshold,
        "notify_conf_threshold": notify_conf_threshold,
        "inference_size": inference_size,
        "frame_skip": frame_skip,
        "enable_notifications": enable_notifications,
    }


def render_detection_sidebar(st, all_class_names: list[str], show_advanced: bool):
    notify_classes = st.sidebar.multiselect(
        "Классы для уведомлений",
        options=all_class_names,
        default=[cls for cls in ["person"] if cls in all_class_names],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Порядок работы: выберите источник входной зоны, запустите анализ и просматривайте журнал проходов.")

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
        st.sidebar.subheader("Фильтры видеопотока")
        animal_filter = st.sidebar.selectbox(
            "Показывать объекты:",
            options=["всё", "коты", "собаки", "птицы", "прочие"],
            index=0,
        )
        track_classes = st.sidebar.multiselect("Фильтр по классам", options=all_class_names, default=[])

        st.sidebar.subheader("Входная зона и правила прохода")
        enable_roi = st.sidebar.checkbox("Включить контроль входной зоны", value=True)
        roi_x = st.sidebar.slider("Входная зона X (%)", 0, 95, 20, 1)
        roi_y = st.sidebar.slider("Входная зона Y (%)", 0, 95, 20, 1)
        roi_w = st.sidebar.slider("Ширина входной зоны (%)", 5, 100, 60, 1)
        roi_h = st.sidebar.slider("Высота входной зоны (%)", 5, 100, 60, 1)

        rule_count_enabled = st.sidebar.checkbox("Правило: N объектов класса X за T сек", value=True)
        rule_class = st.sidebar.selectbox(
            "Класс для правила N/T",
            options=all_class_names,
            index=all_class_names.index("person") if "person" in all_class_names else 0,
        )
        rule_n = st.sidebar.number_input("Количество объектов", min_value=1, max_value=100, value=3, step=1)
        rule_t = st.sidebar.number_input("Интервал, сек", min_value=1, max_value=600, value=10, step=1)

        rule_disappear_enabled = st.sidebar.checkbox("Правило: объект пропал > T сек", value=True)
        rule_disappear_seconds = st.sidebar.number_input(
            "Порог пропадания, сек",
            min_value=1,
            max_value=120,
            value=5,
            step=1,
        )

    return {
        "notify_classes": notify_classes,
        "animal_filter": animal_filter,
        "track_classes": track_classes,
        "roi_config": {
            "enable_roi": enable_roi,
            "roi_x": roi_x,
            "roi_y": roi_y,
            "roi_w": roi_w,
            "roi_h": roi_h,
        },
        "event_settings": {
            "rule_count_enabled": rule_count_enabled,
            "rule_class": rule_class,
            "rule_n": rule_n,
            "rule_t": rule_t,
            "rule_disappear_enabled": rule_disappear_enabled,
            "rule_disappear_seconds": rule_disappear_seconds,
        },
    }
from utils.performance import DEFAULT_FRAME_SKIP, DEFAULT_INFERENCE_SIZE, INFERENCE_SIZE_OPTIONS
