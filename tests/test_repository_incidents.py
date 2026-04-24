import os
import tempfile
import unittest

from db import repository


class RepositoryIncidentsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = repository.DB_PATH
        repository.DB_PATH = os.path.join(self.temp_dir.name, "test_monitoring.db")
        repository.init_db()
        repository.create_video_source(
            name="Gate camera",
            source_type="rtsp",
            source_url="rtsp://gate",
            location="Gate A",
            description="Primary gate camera",
            is_active=True,
        )
        self.source_id = repository.load_video_sources()[0]["id"]

    def tearDown(self):
        repository.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_incident_is_upserted_and_preserves_operator_state(self):
        repository.upsert_incident(
            event_id="evt-1",
            source_id=self.source_id,
            zone_name="Restricted area",
            incident_type="prolonged_presence_near_entry",
            severity="high",
            status="new",
            confidence=0.87,
            snapshot_path="/tmp/snapshot.jpg",
            identification_status="unlinked",
            started_at=100.0,
        )
        incident = repository.load_incidents()[0]
        self.assertEqual(incident["incident_type"], "prolonged_presence_near_entry")
        self.assertEqual(incident["severity"], "high")
        self.assertEqual(incident["status"], "new")
        self.assertEqual(incident["source_name"], "Gate camera")

        repository.update_incident_status(incident_id=incident["id"], status="acknowledged", operator_comment="Checked")
        repository.upsert_incident(
            event_id="evt-1",
            source_id=self.source_id,
            zone_name="Restricted area",
            incident_type="prolonged_presence_near_entry",
            severity="critical",
            status="new",
            confidence=0.91,
            snapshot_path="/tmp/other.jpg",
            identification_status="linked_from_directory",
            started_at=100.0,
        )
        incident = repository.load_incidents()[0]
        self.assertEqual(incident["status"], "acknowledged")
        self.assertEqual(incident["operator_comment"], "Checked")
        self.assertEqual(incident["severity"], "critical")


if __name__ == "__main__":
    unittest.main()
