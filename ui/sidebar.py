"""Sidebar and navigation helpers for the enterprise monitoring UI."""

from __future__ import annotations

from config.app_config import DEFAULT_MODEL_NAME


SECTIONS = [
    "Дашборд",
    "Онлайн-мониторинг",
    "Сотрудники",
    "Журнал событий",
    "Аналитика",
    "Источники видео",
    "Настройки системы",
]

MODEL_OPTIONS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
]

MODEL_CAPTIONS = {
    "yolov8n.pt": "минимальная задержка",
    "yolov8s.pt": "сбалансированный режим",
    "yolov8m.pt": "повышенная точность",
}

ANIMAL_CLASSES = {
    "коты": ["cat"],
    "собаки": ["dog"],
    "птицы": ["bird"],
    "прочие": ["horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"],
}

ROTATION_OPTIONS = ["0°", "90° вправо", "180°", "90° влево"]
ROTATION_MAP = {"0°": 0, "90° вправо": 90, "180°": 180, "90° влево": 270}


def render_app_sidebar(st, *, video_sources: list[dict], system_settings: dict):
    st.sidebar.markdown("### Контур управления")
    section = st.sidebar.radio("Раздел системы", options=SECTIONS, index=0)
    st.sidebar.markdown("---")
    model_name = st.sidebar.selectbox(
        "Нейросетевая модель",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(system_settings.get("model_name", DEFAULT_MODEL_NAME))
        if system_settings.get("model_name", DEFAULT_MODEL_NAME) in MODEL_OPTIONS
        else 1,
    )
    st.sidebar.caption(MODEL_CAPTIONS.get(model_name, "рабочая модель"))
    active_sources = [source for source in video_sources if source.get("is_active")]
    st.sidebar.metric("Активных источников", len(active_sources))
    st.sidebar.metric("Режим production", "включен" if active_sources else "ожидает источник")
    st.sidebar.caption(
        "Production-источники обслуживаются worker-процессом. "
        "Browser live работает как client-side WebRTC режим, а загрузка файлов остается демонстрационным сценарием."
    )
    return {
        "section": section,
        "model_name": model_name,
    }
