import os
import unittest
from unittest.mock import patch

from services.employee_repository import (
    LocalEmployeeRepository,
    RemoteEmployeeRepository,
    build_employee_repository,
)


class EmployeeRepositoryTests(unittest.TestCase):
    def test_build_employee_repository_uses_local_provider(self):
        with patch.dict(os.environ, {"EMPLOYEE_DB_MODE": "sqlite"}, clear=False):
            repo = build_employee_repository(
                load_employees_fn=lambda: [],
                replace_cache_fn=lambda *args, **kwargs: None,
                load_sync_state_fn=lambda: None,
                upsert_sync_state_fn=lambda **kwargs: None,
            )
        self.assertIsInstance(repo, LocalEmployeeRepository)
        self.assertFalse(repo.is_read_only())

    def test_build_employee_repository_uses_remote_provider(self):
        with patch.dict(os.environ, {"EMPLOYEE_DB_MODE": "api", "EMPLOYEE_API_URL": "https://example/api"}, clear=False):
            repo = build_employee_repository(
                load_employees_fn=lambda: [],
                replace_cache_fn=lambda *args, **kwargs: None,
                load_sync_state_fn=lambda: None,
                upsert_sync_state_fn=lambda **kwargs: None,
            )
        self.assertIsInstance(repo, RemoteEmployeeRepository)
        self.assertTrue(repo.is_read_only())

    def test_remote_repository_reads_cached_employees_without_implicit_sync(self):
        sync_calls = []
        cache_rows = [{"id": 1, "full_name": "Иванов И.И.", "status": "active"}]
        repo = RemoteEmployeeRepository(
            config={"mode": "api", "employee_api_url": "https://example/api"},
            load_employees_fn=lambda: cache_rows,
            replace_cache_fn=lambda *args, **kwargs: None,
            load_sync_state_fn=lambda: {"last_synced_at": 123.0, "last_error": "timeout"},
            upsert_sync_state_fn=lambda **kwargs: sync_calls.append(kwargs),
        )
        rows = repo.list_employees()
        self.assertEqual(rows, cache_rows)
        self.assertEqual(sync_calls, [])


if __name__ == "__main__":
    unittest.main()
