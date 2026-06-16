import os
import tempfile
import time
import unittest
import importlib.util
from unittest.mock import patch

from db import repository


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None

if FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient

    from api.app import create_app


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class ApiAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = repository.DB_PATH
        repository.DB_PATH = os.path.join(self.temp_dir.name, "test_monitoring.db")
        repository.init_db()
        repository.create_video_source(
            name="Gate A",
            source_type="rtsp",
            source_url="rtsp://gate-a",
            location="Building A",
            description="Primary source",
            is_active=True,
            enable_roi=True,
            roi_x=10,
            roi_y=15,
            roi_w=40,
            roi_h=50,
        )
        repository.create_employee(
            full_name="Иванов Иван Иванович",
            last_name="Иванов",
            first_name="Иван",
            middle_name="Иванович",
            employee_number="EMP-1",
            department="Security",
            position="Operator",
            status="active",
            hire_date=time.time(),
        )
        repository.db_insert_event(
            {
                "event_id": "evt-api-1",
                "session_id": "worker-1-demo",
                "event_scope": "domain",
                "event_type": "person_detected_near_entry",
                "source_type": "rtsp",
                "frame_index": 1,
                "timestamp": time.time(),
                "class_name": "person",
                "confidence": 0.91,
                "track_id": "11",
                "roi_inside": True,
                "message": "API seed event",
                "identification_status": "unlinked",
            }
        )
        repository.upsert_incident(
            event_id="evt-api-1",
            source_id=1,
            zone_name="Gate A",
            incident_type="person_detected_near_entry",
            severity="medium",
            status="new",
            confidence=0.91,
            snapshot_path="",
            identification_status="unlinked",
            started_at=time.time(),
        )
        self.client = TestClient(create_app())

    def tearDown(self):
        repository.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["status"], {"ok", "degraded"})
        self.assertIn("telemetry", response.json())
        self.assertIn("operational", response.json())

    def test_liveness_and_readiness_endpoints(self):
        live_response = self.client.get("/health/live")
        self.assertEqual(live_response.status_code, 200)
        self.assertEqual(live_response.json()["status"], "ok")

        ready_response = self.client.get("/health/ready")
        self.assertIn(ready_response.status_code, {200, 503})
        self.assertIn("status", ready_response.json())
        self.assertIn("issues", ready_response.json())

    def test_video_sources_endpoint_returns_processing_config(self):
        response = self.client.get("/api/v1/video-sources")
        self.assertEqual(response.status_code, 200)
        payload = response.json()["items"]
        self.assertEqual(len(payload), 1)
        self.assertTrue(payload[0]["enable_roi"])
        self.assertEqual(payload[0]["roi_x"], 10.0)

    def test_dashboard_summary_contains_recent_events(self):
        response = self.client.get("/api/v1/dashboard/summary", params={"event_limit": 10})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("summary", payload)
        self.assertIn("incidents_summary", payload)
        self.assertIn("incident_queues", payload)
        self.assertIn("incident_queue", payload)
        self.assertIn("incident_sla", payload)
        self.assertIn("incident_age_buckets", payload)
        self.assertIn("operator_workload", payload)
        self.assertEqual(len(payload["recent_events"]), 1)
        self.assertEqual(payload["recent_events"][0]["event_id"], "evt-api-1")
        self.assertEqual(len(payload["recent_incidents"]), 1)
        self.assertEqual(payload["incident_queue"][0]["Owner"], "не назначен")

    def test_video_source_activation_endpoint(self):
        source_id = repository.load_video_sources()[0]["id"]
        response = self.client.put(f"/api/v1/video-sources/{source_id}/active", params={"is_active": "false"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(repository.load_video_sources()[0]["is_active"])

    def test_incidents_endpoint_and_status_update(self):
        response = self.client.get("/api/v1/incidents")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["items"]), 1)
        incident_id = payload["items"][0]["id"]
        self.assertIn("summary", payload)
        self.assertEqual(payload["summary"]["unassigned_active"], 1)
        self.assertIn("status_breakdown", payload["summary"])
        self.assertIn("sla_summary", payload["summary"])

        update_response = self.client.put(
            f"/api/v1/incidents/{incident_id}/status",
            json={
                "status": "in_progress",
                "operator_comment": "Checked by operator",
                "assigned_to": "Shift lead",
                "resolution_code": "external_follow_up",
                "resolution_notes": "Escalated to security desk",
            },
        )
        self.assertEqual(update_response.status_code, 200)
        updated = repository.load_incidents()[0]
        self.assertEqual(updated["status"], "in_progress")
        self.assertEqual(updated["operator_comment"], "Checked by operator")
        self.assertEqual(updated["assigned_to"], "Shift lead")
        self.assertEqual(updated["resolution_code"], "external_follow_up")
        self.assertEqual(updated["resolution_notes"], "Escalated to security desk")
        self.assertIsNotNone(updated["acknowledged_at"])

    def test_metrics_endpoint_returns_prometheus_payload(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("human_tracker_source_count_total", response.text)
        self.assertIn("human_tracker_source_count_degraded", response.text)

    def test_telemetry_endpoint_returns_operational_summary(self):
        response = self.client.get("/api/v1/telemetry")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("telemetry", payload)
        self.assertIn("operational", payload)
        self.assertIn("coverage_ratio", payload["operational"])

    def test_health_details_returns_dashboard_snapshot(self):
        response = self.client.get("/health/details")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("summary", payload)
        self.assertIn("telemetry", payload)
        self.assertIn("operational", payload)

    def test_models_endpoint_returns_local_model_registry(self):
        response = self.client.get("/api/v1/models")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(len(payload["items"]), 3)

    @patch("api.app.save_snapshot", return_value="/tmp/attendance.jpg")
    @patch(
        "api.app.analyze_frame",
        return_value={
            "model_name": "yolov8n.pt",
            "processing_time_ms": 41.2,
            "person_count": 1,
            "detections": [{"class_name": "person", "confidence": 0.88, "box": [1, 2, 3, 4]}],
            "annotated_image_base64": "data:image/jpeg;base64,AAAA",
            "frame_bgr": object(),
        },
    )
    def test_attendance_checkpoint_creates_check_in_and_check_out(self, _analyze_frame, _save_snapshot):
        employee_id = repository.load_employees()[0]["id"]
        access_point_id = repository.load_access_points()[0]["id"]

        check_in_response = self.client.post(
            "/api/v1/attendance/checkpoint",
            json={
                "employee_id": employee_id,
                "access_point_id": access_point_id,
                "image_base64": "data:image/jpeg;base64,AAAA",
                "model_name": "yolov8n.pt",
            },
        )
        self.assertEqual(check_in_response.status_code, 200)
        check_in_payload = check_in_response.json()
        self.assertEqual(check_in_payload["attendance_status"], "check_in")

        attendance_today = self.client.get("/api/v1/attendance/today")
        self.assertEqual(attendance_today.status_code, 200)
        self.assertEqual(attendance_today.json()["summary"]["check_ins"], 1)
        self.assertEqual(attendance_today.json()["summary"]["currently_on_site"], 1)

        check_out_response = self.client.post(
            "/api/v1/attendance/checkpoint",
            json={
                "employee_id": employee_id,
                "access_point_id": access_point_id,
                "image_base64": "data:image/jpeg;base64,AAAA",
                "model_name": "yolov8n.pt",
            },
        )
        self.assertEqual(check_out_response.status_code, 200)
        self.assertEqual(check_out_response.json()["attendance_status"], "check_out")

    def test_audit_logs_endpoint_returns_items(self):
        repository.append_audit_log(
            actor_name="API Operator",
            actor_role="admin",
            action="incident.status_updated",
            resource_type="incident",
            resource_id="1",
            details={"status": "new"},
        )
        response = self.client.get("/api/v1/audit-logs")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["actor_name"], "API Operator")

    def test_delete_employee_endpoint_removes_employee_and_related_records(self):
        employee_id = repository.load_employees()[0]["id"]
        access_point_id = repository.load_access_points()[0]["id"]
        repository.register_employee_attendance(
            employee_id=employee_id,
            access_point_id=access_point_id,
            model_name="yolov8n.pt",
            source_type="browser_camera",
            detection_confidence=0.9,
            note="seed attendance",
        )
        repository.upsert_incident(
            event_id="evt-api-employee-delete",
            source_id=1,
            zone_name="Gate A",
            incident_type="employee_related",
            severity="low",
            status="new",
            confidence=0.5,
            snapshot_path="",
            employee_id=employee_id,
            identification_status="linked_from_directory",
            started_at=time.time(),
        )

        response = self.client.delete(f"/api/v1/employees/{employee_id}")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["employee_id"], employee_id)
        self.assertGreaterEqual(payload["deleted"]["employees"], 1)
        self.assertEqual(repository.load_employees(), [])
        self.assertEqual(self.client.get("/api/v1/attendance/today").json()["items"], [])
        self.assertTrue(all(item.get("employee_id") != employee_id for item in repository.load_events()))


if __name__ == "__main__":
    unittest.main()
