"""Role-based access helpers for the operator console."""

from __future__ import annotations

from config.app_config import DEFAULT_UI_ACTOR, DEFAULT_UI_ROLE


ROLE_OPTIONS = {
    "admin": "Администратор",
    "operator": "Оператор",
    "auditor": "Аудитор",
}

ROLE_PERMISSIONS = {
    "admin": {
        "view_dashboard",
        "view_monitoring",
        "view_incidents",
        "update_incidents",
        "link_incidents",
        "view_analytics",
        "view_configuration",
        "manage_sources",
        "manage_zones",
        "manage_directory",
        "manage_settings",
        "view_audit",
    },
    "operator": {
        "view_dashboard",
        "view_monitoring",
        "view_incidents",
        "update_incidents",
        "link_incidents",
        "view_analytics",
    },
    "auditor": {
        "view_dashboard",
        "view_incidents",
        "view_analytics",
        "view_audit",
    },
}

SECTION_PERMISSIONS = {
    "Ситуационный центр": "view_dashboard",
    "Оперативный мониторинг": "view_monitoring",
    "Журнал инцидентов": "view_incidents",
    "Аналитика и отчеты": "view_analytics",
    "Камеры и зоны": "view_configuration",
    "Подключение камер": "view_configuration",
    "Справочник персонала": "manage_directory",
    "Настройки системы": "manage_settings",
    "Доступ и аудит": "view_audit",
}


def ensure_access_context(session_state) -> dict:
    if "current_user_role" not in session_state:
        session_state.current_user_role = DEFAULT_UI_ROLE
    if "current_user_name" not in session_state:
        session_state.current_user_name = DEFAULT_UI_ACTOR
    return build_access_context(session_state)


def build_access_context(session_state) -> dict:
    role = normalize_role(session_state.get("current_user_role"))
    actor_name = (session_state.get("current_user_name") or DEFAULT_UI_ACTOR).strip() or DEFAULT_UI_ACTOR
    return {
        "role": role,
        "role_label": ROLE_OPTIONS[role],
        "actor_name": actor_name,
        "permissions": sorted(ROLE_PERMISSIONS[role]),
    }


def normalize_role(role: str | None) -> str:
    normalized = (role or DEFAULT_UI_ROLE).strip().lower()
    return normalized if normalized in ROLE_OPTIONS else DEFAULT_UI_ROLE


def has_permission(access_context: dict | None, permission: str) -> bool:
    if not access_context:
        return False
    return permission in ROLE_PERMISSIONS.get(normalize_role(access_context.get("role")), set())


def get_visible_sections(sections: list[str], access_context: dict) -> list[str]:
    visible_sections = [section for section in sections if has_permission(access_context, SECTION_PERMISSIONS.get(section, ""))]
    return visible_sections or [sections[0]]


def assert_permission(access_context: dict, permission: str):
    if has_permission(access_context, permission):
        return
    raise PermissionError(f"Недостаточно прав для действия: {permission}")
