import os
import tempfile
import time
import unittest

from db import repository


class RepositoryEventLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = repository.DB_PATH
        repository.DB_PATH = os.path.join(self.temp_dir.name, "test_monitoring.db")
        repository.init_db()

    def tearDown(self):
        repository.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_link_event_to_employee_updates_journal(self):
        repository.create_employee(
            full_name="Иванов Иван Иванович",
            last_name="Иванов",
            first_name="Иван",
            middle_name="Иванович",
            employee_number="EMP-1001",
            department="Служба безопасности",
            position="Оператор",
            status="active",
            hire_date=time.time(),
        )
        employee = repository.load_employees()[0]
        repository.db_insert_event(
            {
                "event_id": "evt-1",
                "session_id": "session-1",
                "event_scope": "domain",
                "event_type": "person_detected_near_entry",
                "source_type": "rtsp",
                "frame_index": 1,
                "timestamp": time.time(),
                "class_name": "person",
                "confidence": 0.92,
                "track_id": "10",
                "roi_inside": True,
                "message": "Человек обнаружен рядом со входной зоной",
                "identification_status": "unlinked",
            }
        )

        repository.link_event_to_employee(
            event_id="evt-1",
            employee_id=employee["id"],
            identification_status="pending_operator_confirmation",
            note="Подтверждено оператором смены",
        )

        linked_event = repository.load_events(limit=1)[0]
        self.assertEqual(linked_event["employee_name"], "Иванов Иван Иванович")
        self.assertEqual(linked_event["employee_number"], "EMP-1001")
        self.assertEqual(linked_event["identification_status"], "pending_operator_confirmation")


if __name__ == "__main__":
    unittest.main()
