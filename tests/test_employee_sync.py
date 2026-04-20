import unittest

from services.employee_sync import maybe_sync_employee_directory, should_auto_sync_directory


class FakeRemoteRepository:
    def __init__(self):
        self.sync_calls = 0

    def is_remote(self):
        return True

    def sync(self):
        self.sync_calls += 1
        return {"sync_status": "ok", "updated_at": 999.0}


class FakeLocalRepository:
    def is_remote(self):
        return False


class EmployeeSyncTests(unittest.TestCase):
    def test_should_auto_sync_when_no_previous_state(self):
        self.assertTrue(should_auto_sync_directory(None, interval_seconds=300))

    def test_should_not_auto_sync_before_interval(self):
        self.assertFalse(should_auto_sync_directory({"updated_at": 10**12}, interval_seconds=300))

    def test_maybe_sync_skips_local_repository(self):
        changed, state = maybe_sync_employee_directory(FakeLocalRepository(), {}, interval_seconds=300)
        self.assertFalse(changed)
        self.assertEqual(state, {})

    def test_maybe_sync_runs_remote_repository(self):
        repository = FakeRemoteRepository()
        changed, state = maybe_sync_employee_directory(repository, None, interval_seconds=300)
        self.assertTrue(changed)
        self.assertEqual(repository.sync_calls, 1)
        self.assertEqual(state["sync_status"], "ok")


if __name__ == "__main__":
    unittest.main()
