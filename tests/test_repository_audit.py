import os
import tempfile
import unittest

from db import repository


class RepositoryAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = repository.DB_PATH
        repository.DB_PATH = os.path.join(self.temp_dir.name, "audit_test.db")
        repository.init_db()

    def tearDown(self):
        repository.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_append_and_load_audit_logs(self):
        repository.append_audit_log(
            actor_name="Operator 1",
            actor_role="operator",
            action="incident.status_updated",
            resource_type="incident",
            resource_id="15",
            details={"status": "acknowledged"},
        )

        logs = repository.load_audit_logs(limit=10)

        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["actor_name"], "Operator 1")
        self.assertEqual(logs[0]["actor_role"], "operator")
        self.assertEqual(logs[0]["action"], "incident.status_updated")
        self.assertEqual(logs[0]["details"]["status"], "acknowledged")


if __name__ == "__main__":
    unittest.main()
