import unittest

from services.identity_service import build_identity_result, resolve_identification_status


class IdentityServiceTests(unittest.TestCase):
    def test_resolve_identification_status_without_state(self):
        self.assertEqual(resolve_identification_status(None), "not_configured")

    def test_resolve_identification_status_reports_db_unavailable(self):
        state = {"sync_status": "fallback_cache", "sync_error": "timeout", "employee_count": 10, "active_employee_count": 10}
        self.assertEqual(resolve_identification_status(state), "db_unavailable")

    def test_resolve_identification_status_reports_no_reference_data(self):
        state = {"sync_status": "ok", "employee_count": 10, "active_employee_count": 10, "reference_employee_count": 0}
        self.assertEqual(resolve_identification_status(state), "no_reference_data")

    def test_resolve_identification_status_reports_not_enough_reference_data(self):
        state = {"sync_status": "ok", "employee_count": 10, "active_employee_count": 10, "reference_employee_count": 2}
        self.assertEqual(resolve_identification_status(state), "not_enough_reference_data")

    def test_build_identity_result_marks_inactive_employee(self):
        result = build_identity_result(identified_employee={"id": 5, "status": "inactive"}, confidence=0.9)
        self.assertEqual(result["identification_status"], "inactive_employee")

    def test_build_identity_result_marks_verified_employee(self):
        result = build_identity_result(identified_employee={"id": 5, "status": "active"}, confidence=0.91)
        self.assertEqual(result["identification_status"], "verified")


if __name__ == "__main__":
    unittest.main()
