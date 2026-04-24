import unittest

from services.identity_service import (
    build_identity_result,
    build_identity_runtime_state,
    identify_person,
    resolve_identification_status,
)


class IdentityServiceTests(unittest.TestCase):
    def test_resolve_identification_status_without_state(self):
        self.assertEqual(resolve_identification_status(None), "unlinked")

    def test_resolve_identification_status_reports_db_unavailable(self):
        state = {"sync_status": "fallback_cache", "sync_error": "timeout", "employee_count": 10, "active_employee_count": 10}
        self.assertEqual(resolve_identification_status(state), "db_unavailable")

    def test_resolve_identification_status_reports_no_reference_data(self):
        state = {"sync_status": "ok", "employee_count": 10, "active_employee_count": 10, "reference_employee_count": 0}
        self.assertEqual(resolve_identification_status(state), "no_reference_data")

    def test_resolve_identification_status_reports_not_enough_reference_data(self):
        state = {"sync_status": "ok", "employee_count": 10, "active_employee_count": 10, "reference_employee_count": 2}
        self.assertEqual(resolve_identification_status(state), "not_enough_reference_data")

    def test_resolve_identification_status_reports_unknown_when_directory_ready(self):
        state = {"sync_status": "ok", "employee_count": 10, "active_employee_count": 10, "reference_employee_count": 10}
        self.assertEqual(resolve_identification_status(state), "unknown")

    def test_build_identity_result_marks_inactive_employee(self):
        result = build_identity_result(identified_employee={"id": 5, "status": "inactive"}, confidence=0.9)
        self.assertEqual(result["identification_status"], "inactive_employee")

    def test_build_identity_result_marks_directory_linked_employee(self):
        result = build_identity_result(identified_employee={"id": 5, "status": "active"}, confidence=0.91)
        self.assertEqual(result["identification_status"], "linked_from_directory")

    def test_build_identity_runtime_state_exposes_backend(self):
        state = build_identity_runtime_state(
            employees=[
                {"id": 1, "status": "active", "reference_count": 1},
                {"id": 2, "status": "inactive", "reference_count": 0},
            ],
            sync_state={"sync_status": "ok", "data_source": "sqlite"},
            identity_backend="face_placeholder",
        )
        self.assertEqual(state["identity_backend"], "face_placeholder")
        self.assertTrue(state["identity_backend_enabled"])
        self.assertEqual(state["reference_employee_count"], 1)

    def test_identify_person_returns_placeholder_when_backend_disabled(self):
        result = identify_person(
            frame_bgr=object(),
            detection={"class_name": "person"},
            employee_gallery=[],
            identity_state={"identity_backend": "disabled"},
        )
        self.assertEqual(result["identification_status"], "unlinked")


if __name__ == "__main__":
    unittest.main()
