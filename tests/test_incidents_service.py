import unittest

from services.incidents import infer_incident_severity, sync_incidents_from_events


class IncidentsServiceTests(unittest.TestCase):
    def test_sync_creates_incidents_for_domain_events(self):
        captured = []
        events = [
            {
                "event_id": "evt-1",
                "event_scope": "domain",
                "event_type": "prolonged_presence_near_entry",
                "source_id": 1,
                "access_point_name": "Restricted area",
                "confidence": 0.84,
                "snapshot_path": "/tmp/frame.jpg",
                "evidence_clip_path": "/tmp/evidence.mp4",
                "evidence_retention_until": 456.0,
                "timestamp": 123.0,
                "identification_status": "unlinked",
            },
            {
                "event_id": "evt-2",
                "event_scope": "raw",
                "event_type": "object_detected",
                "source_id": 1,
                "timestamp": 124.0,
            },
        ]
        sync_incidents_from_events(events, upsert_incident_fn=lambda **kwargs: captured.append(kwargs))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["event_id"], "evt-1")
        self.assertEqual(captured[0]["severity"], "high")
        self.assertEqual(captured[0]["evidence_clip_path"], "/tmp/evidence.mp4")
        self.assertEqual(captured[0]["evidence_retention_until"], 456.0)

    def test_severity_mapping(self):
        self.assertEqual(infer_incident_severity({"event_type": "stream_offline"}), "critical")
        self.assertEqual(infer_incident_severity({"event_type": "person_entered_entry_zone"}), "medium")


if __name__ == "__main__":
    unittest.main()
