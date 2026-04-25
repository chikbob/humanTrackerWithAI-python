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
            evidence_clip_path="/tmp/evidence.mp4",
            evidence_retention_until=555.0,
            identification_status="unlinked",
            started_at=100.0,
        )
        incident = repository.load_incidents()[0]
        self.assertEqual(incident["incident_type"], "prolonged_presence_near_entry")
        self.assertEqual(incident["severity"], "high")
        self.assertEqual(incident["status"], "new")
        self.assertEqual(incident["source_name"], "Gate camera")
        self.assertEqual(incident["evidence_clip_path"], "/tmp/evidence.mp4")
        self.assertEqual(incident["evidence_retention_until"], 555.0)

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
            evidence_clip_path="/tmp/evidence-2.mp4",
            evidence_retention_until=777.0,
            identification_status="linked_from_directory",
            started_at=100.0,
        )
        incident = repository.load_incidents()[0]
        self.assertEqual(incident["status"], "acknowledged")
        self.assertEqual(incident["operator_comment"], "Checked")
        self.assertEqual(incident["severity"], "critical")
        self.assertEqual(incident["evidence_clip_path"], "/tmp/evidence-2.mp4")
        self.assertEqual(incident["evidence_retention_until"], 777.0)

    def test_case_workflow_fields_are_updated_and_preserved(self):
        repository.upsert_incident(
            event_id="evt-case-1",
            source_id=self.source_id,
            zone_name="Gate A",
            incident_type="unknown_person_detected",
            severity="high",
            started_at=100.0,
        )
        incident = repository.load_incidents()[0]
        repository.update_incident_status(
            incident_id=incident["id"],
            status="in_progress",
            operator_comment="Investigating",
            assigned_to="Shift A",
            resolution_code="model_threshold_tuning",
            resolution_notes="Need to tune night profile",
        )
        updated = repository.load_incidents()[0]
        self.assertEqual(updated["assigned_to"], "Shift A")
        self.assertEqual(updated["operator_comment"], "Investigating")
        self.assertEqual(updated["resolution_code"], "model_threshold_tuning")
        self.assertEqual(updated["resolution_notes"], "Need to tune night profile")
        self.assertIsNotNone(updated["acknowledged_at"])
        self.assertIsNone(updated["resolved_at"])

        repository.upsert_incident(
            event_id="evt-case-1",
            source_id=self.source_id,
            zone_name="Gate A",
            incident_type="unknown_person_detected",
            severity="critical",
            started_at=100.0,
        )
        preserved = repository.load_incidents()[0]
        self.assertEqual(preserved["assigned_to"], "Shift A")
        self.assertEqual(preserved["resolution_code"], "model_threshold_tuning")
        self.assertEqual(preserved["resolution_notes"], "Need to tune night profile")

    def test_resolved_status_sets_resolved_timestamp(self):
        repository.upsert_incident(
            event_id="evt-case-2",
            source_id=self.source_id,
            zone_name="Gate A",
            incident_type="prolonged_presence_near_entry",
            severity="high",
            started_at=100.0,
        )
        incident = repository.load_incidents()[0]
        repository.update_incident_status(
            incident_id=incident["id"],
            status="resolved",
            operator_comment="Closed",
            assigned_to="Operator 1",
            resolution_code="confirmed_security_event",
            resolution_notes="Guard dispatched",
        )
        updated = repository.load_incidents()[0]
        self.assertIsNotNone(updated["acknowledged_at"])
        self.assertIsNotNone(updated["resolved_at"])
        self.assertEqual(updated["assigned_to"], "Operator 1")

    def test_attach_event_evidence_updates_incident_and_event_rows(self):
        repository.db_insert_event(
            {
                "event_id": "evt-evidence",
                "session_id": "worker-1-demo",
                "event_scope": "domain",
                "event_type": "unknown_person_detected",
                "source_type": "rtsp",
                "frame_index": 1,
                "timestamp": 123.0,
                "class_name": "person",
                "confidence": 0.8,
                "identification_status": "unlinked",
            }
        )
        repository.upsert_incident(
            event_id="evt-evidence",
            source_id=self.source_id,
            zone_name="Restricted area",
            incident_type="unknown_person_detected",
            severity="high",
            started_at=123.0,
        )
        repository.attach_event_evidence(
            event_id="evt-evidence",
            snapshot_path="/tmp/incident.jpg",
            evidence_clip_path="/tmp/incident.mp4",
            evidence_retention_until=999.0,
        )
        event = repository.load_events(limit=1)[0]
        incident = repository.load_incidents(limit=1)[0]
        self.assertEqual(event["snapshot_path"], "/tmp/incident.jpg")
        self.assertEqual(event["evidence_clip_path"], "/tmp/incident.mp4")
        self.assertEqual(incident["snapshot_path"], "/tmp/incident.jpg")
        self.assertEqual(incident["evidence_clip_path"], "/tmp/incident.mp4")
        self.assertEqual(incident["evidence_retention_until"], 999.0)


if __name__ == "__main__":
    unittest.main()
