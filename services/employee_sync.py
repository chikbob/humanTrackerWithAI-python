"""Employee directory sync helpers and identification gallery state."""

from __future__ import annotations

import time


def build_identity_gallery_state(employees: list[dict], sync_state: dict | None) -> dict:
    employees = employees or []
    sync_state = sync_state or {}
    active_employees = [employee for employee in employees if employee.get("status") == "active"]
    referenced_employees = [employee for employee in active_employees if int(employee.get("reference_count") or 0) > 0]
    return {
        "employee_count": len(employees),
        "active_employee_count": len(active_employees),
        "reference_employee_count": len(referenced_employees),
        "directory_source": sync_state.get("data_source", "sqlite"),
        "sync_status": sync_state.get("sync_status", "unknown"),
        "sync_error": sync_state.get("last_error", ""),
    }


def employee_directory_summary(sync_state: dict | None) -> dict:
    sync_state = sync_state or {}
    status = sync_state.get("sync_status", "unknown")
    if status == "ok":
        badge = "Синхронизация актуальна"
    elif status == "fallback_cache":
        badge = "Кэш read-only"
    elif status == "local_only":
        badge = "Локальный справочник"
    else:
        badge = "Нет синхронизации"
    return {
        "data_source": sync_state.get("data_source", "sqlite"),
        "sync_status": status,
        "last_synced_at": sync_state.get("last_synced_at"),
        "last_error": sync_state.get("last_error", ""),
        "cache_mode": sync_state.get("cache_mode", "read_only"),
        "updated_at": sync_state.get("updated_at"),
        "badge": badge,
    }


def should_auto_sync_directory(sync_state: dict | None, *, interval_seconds: int) -> bool:
    if interval_seconds <= 0:
        return False
    sync_state = sync_state or {}
    last_checkpoint = sync_state.get("updated_at") or sync_state.get("last_synced_at") or 0
    return (time.time() - float(last_checkpoint or 0)) >= interval_seconds


def maybe_sync_employee_directory(repository, sync_state: dict | None, *, interval_seconds: int) -> tuple[bool, dict]:
    """Run a throttled automatic directory sync for remote repositories."""
    if not getattr(repository, "is_remote", lambda: False)():
        return False, sync_state or {}
    if not should_auto_sync_directory(sync_state, interval_seconds=interval_seconds):
        return False, sync_state or {}
    return True, repository.sync()
