"""Employee directory repositories with local and remote providers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request


REMOTE_MODES = {"api", "supabase", "postgres", "mysql"}


def get_employee_db_mode() -> str:
    return os.getenv("EMPLOYEE_DB_MODE", "sqlite").strip().lower() or "sqlite"


def build_employee_directory_config() -> dict:
    return {
        "mode": get_employee_db_mode(),
        "db_url": os.getenv("EMPLOYEE_DB_URL", "").strip(),
        "supabase_url": os.getenv("SUPABASE_URL", "").strip(),
        "supabase_key": os.getenv("SUPABASE_KEY", "").strip(),
        "employee_api_url": os.getenv("EMPLOYEE_API_URL", "").strip(),
        "employee_api_token": os.getenv("EMPLOYEE_API_TOKEN", "").strip(),
    }


class EmployeeRepository:
    """Read-side abstraction for employee directory access."""

    source_name = "unknown"

    def list_employees(self) -> list[dict]:
        raise NotImplementedError

    def get_status(self) -> dict:
        raise NotImplementedError

    def sync(self) -> dict:
        raise NotImplementedError

    def is_read_only(self) -> bool:
        return True

    def is_remote(self) -> bool:
        return False


class LocalEmployeeRepository(EmployeeRepository):
    source_name = "sqlite"

    def __init__(self, *, load_employees_fn, load_sync_state_fn, upsert_sync_state_fn):
        self._load_employees = load_employees_fn
        self._load_sync_state = load_sync_state_fn
        self._upsert_sync_state = upsert_sync_state_fn

    def list_employees(self) -> list[dict]:
        employees = self._load_employees()
        if self._load_sync_state() is None:
            self._upsert_sync_state(
                data_source=self.source_name,
                sync_status="local_only",
                last_synced_at=None,
                last_error="",
                cache_mode="read_write",
            )
        return employees

    def get_status(self) -> dict:
        state = self._load_sync_state()
        if state is None:
            return {
                "data_source": self.source_name,
                "sync_status": "local_only",
                "last_synced_at": None,
                "last_error": "",
                "cache_mode": "read_write",
            }
        return state

    def sync(self) -> dict:
        self._upsert_sync_state(
            data_source=self.source_name,
            sync_status="local_only",
            last_synced_at=None,
            last_error="",
            cache_mode="read_write",
        )
        return self.get_status()

    def is_read_only(self) -> bool:
        return False


class RemoteEmployeeRepository(EmployeeRepository):
    """Remote directory provider with local cache fallback."""

    def __init__(
        self,
        *,
        config: dict,
        load_employees_fn,
        replace_cache_fn,
        load_sync_state_fn,
        upsert_sync_state_fn,
    ):
        self.config = config
        self._load_employees = load_employees_fn
        self._replace_cache = replace_cache_fn
        self._load_sync_state = load_sync_state_fn
        self._upsert_sync_state = upsert_sync_state_fn
        self.source_name = config["mode"]

    def list_employees(self) -> list[dict]:
        return self._load_employees()

    def get_status(self) -> dict:
        state = self._load_sync_state()
        if state is None:
            return {
                "data_source": self.source_name,
                "sync_status": "not_synced",
                "last_synced_at": None,
                "last_error": "",
                "cache_mode": "read_only",
            }
        return state

    def sync(self) -> dict:
        try:
            employees = self._fetch_remote_directory(raise_on_error=True)
        except RuntimeError as exc:
            self._upsert_sync_state(
                data_source=self.source_name,
                sync_status="fallback_cache",
                last_synced_at=self.get_status().get("last_synced_at"),
                last_error=str(exc),
                cache_mode="read_only",
            )
            return self.get_status()
        self._replace_cache(employees, source_system=self.source_name)
        self._upsert_sync_state(
            data_source=self.source_name,
            sync_status="ok",
            last_synced_at=max(employee.get("last_synced_at") or 0 for employee in employees) if employees else None,
            last_error="",
            cache_mode="read_only",
        )
        return self.get_status()

    def _fetch_remote_directory(self, *, raise_on_error: bool = False):
        try:
            if self.config["mode"] == "api":
                return _fetch_api_directory(self.config)
            if self.config["mode"] == "supabase":
                return _fetch_supabase_directory(self.config)
            raise RuntimeError(f"remote_mode_not_supported:{self.config['mode']}")
        except RuntimeError:
            if raise_on_error:
                raise
            return None

    def is_remote(self) -> bool:
        return True


def build_employee_repository(
    *,
    load_employees_fn,
    replace_cache_fn,
    load_sync_state_fn,
    upsert_sync_state_fn,
):
    config = build_employee_directory_config()
    mode = config["mode"]
    if mode == "sqlite":
        return LocalEmployeeRepository(
            load_employees_fn=load_employees_fn,
            load_sync_state_fn=load_sync_state_fn,
            upsert_sync_state_fn=upsert_sync_state_fn,
        )
    return RemoteEmployeeRepository(
        config=config,
        load_employees_fn=load_employees_fn,
        replace_cache_fn=replace_cache_fn,
        load_sync_state_fn=load_sync_state_fn,
        upsert_sync_state_fn=upsert_sync_state_fn,
    )


def format_employee_sync_status(sync_state: dict | None) -> str:
    sync_state = sync_state or {}
    status = sync_state.get("sync_status", "unknown")
    mapping = {
        "ok": "Синхронизация выполнена",
        "fallback_cache": "Используется локальный кэш",
        "local_only": "Локальный справочник",
        "not_synced": "Синхронизация не выполнялась",
        "unknown": "Статус не определен",
    }
    return mapping.get(status, status)


def _http_get_json(url: str, headers: dict):
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"remote_directory_unavailable:{exc}") from exc


def _normalize_remote_employees(payload) -> list[dict]:
    rows = payload if isinstance(payload, list) else payload.get("employees", [])
    normalized = []
    for row in rows:
        normalized.append(
            {
                "full_name": row.get("full_name") or row.get("name") or "",
                "department": row.get("department") or "",
                "position": row.get("position") or "",
                "status": row.get("status") or "active",
                "created_at": row.get("created_at"),
                "external_id": row.get("external_id") or row.get("id"),
                "reference_image_url": row.get("reference_image_url") or row.get("photo_url"),
                "reference_count": int(row.get("reference_count") or 0),
                "last_synced_at": row.get("last_synced_at"),
            }
        )
    return [row for row in normalized if row["full_name"]]


def _fetch_api_directory(config: dict) -> list[dict]:
    api_url = config["employee_api_url"]
    if not api_url:
        raise RuntimeError("employee_api_url_missing")
    headers = {"Accept": "application/json"}
    if config["employee_api_token"]:
        headers["Authorization"] = f"Bearer {config['employee_api_token']}"
    return _normalize_remote_employees(_http_get_json(api_url, headers))


def _fetch_supabase_directory(config: dict) -> list[dict]:
    supabase_url = config["supabase_url"]
    supabase_key = config["supabase_key"]
    if not supabase_url or not supabase_key:
        raise RuntimeError("supabase_credentials_missing")
    query = urllib.parse.urlencode(
        {
            "select": "id,full_name,department,position,status,created_at,reference_image_url,reference_count,last_synced_at",
            "order": "full_name.asc",
        }
    )
    url = f"{supabase_url.rstrip('/')}/rest/v1/employees?{query}"
    headers = {
        "Accept": "application/json",
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }
    return _normalize_remote_employees(_http_get_json(url, headers))
