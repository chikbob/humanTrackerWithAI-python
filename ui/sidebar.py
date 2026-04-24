"""Sidebar and navigation helpers for the enterprise monitoring UI."""

from __future__ import annotations

from config.app_config import DEFAULT_MODEL_NAME
from services.auth import ROLE_OPTIONS, get_visible_sections


SECTIONS = [
    "Ситуационный центр",
    "Оперативный мониторинг",
    "Журнал инцидентов",
    "Аналитика и отчеты",
    "Камеры и зоны",
    "Подключение камер",
    "Справочник персонала",
    "Настройки системы",
    "Доступ и аудит",
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


def render_app_sidebar(st, *, video_sources: list[dict], system_settings: dict, access_context: dict, monitored_source_count: int | None = None):
    st.sidebar.markdown("### Контур управления")
    visible_sections = get_visible_sections(SECTIONS, access_context)
    section = st.sidebar.radio("Раздел системы", options=visible_sections, index=0)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Доступ")
    actor_name = st.sidebar.text_input("Оператор", value=access_context.get("actor_name") or "", key="sidebar_current_user_name")
    role = st.sidebar.selectbox(
        "Роль",
        options=list(ROLE_OPTIONS.keys()),
        index=list(ROLE_OPTIONS.keys()).index(access_context.get("role", "admin")),
        format_func=lambda key: ROLE_OPTIONS[key],
        key="sidebar_current_user_role",
    )
    st.session_state.current_user_name = actor_name.strip() or access_context.get("actor_name") or "Оператор"
    st.session_state.current_user_role = role
    st.sidebar.caption(f"Текущий профиль: {ROLE_OPTIONS[role]}")
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
    effective_source_count = monitored_source_count if monitored_source_count is not None else len(active_sources)
    st.sidebar.metric("Активных источников", effective_source_count)
    st.sidebar.metric("Режим production", "включен" if effective_source_count else "ожидает источник")
    st.sidebar.caption(
        "Основной контур строится вокруг серверных RTSP/HLS/USB-камер, которые обслуживаются worker-процессом. "
        "Мобильные и browser-live сценарии вынесены в лабораторный контур диагностики."
    )
    return {
        "section": section,
        "model_name": model_name,
    }
