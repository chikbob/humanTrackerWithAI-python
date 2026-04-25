import unittest

try:
    from analytics.access import (
        build_camera_health_summary,
        build_incident_queue_rows,
        build_incident_status_summary,
        build_operator_workload_rows,
        build_source_risk_rows,
        build_zone_risk_rows,
    )
except ModuleNotFoundError:  # pragma: no cover - optional analytics dependency in minimal env
    build_camera_health_summary = None
    build_incident_queue_rows = None
    build_incident_status_summary = None
    build_operator_workload_rows = None
    build_source_risk_rows = None
    build_zone_risk_rows = None


@unittest.skipIf(build_incident_status_summary is None, "analytics dependencies are not installed")
class DashboardAnalyticsTests(unittest.TestCase):
    def test_incident_status_summary_counts_active_and_response_time(self):
        incidents = [
            {"id": 1, "severity": "critical", "status": "new", "zone_name": "Периметр", "started_at": 100.0, "updated_at": 100.0},
            {"id": 2, "severity": "high", "status": "acknowledged", "zone_name": "Вход", "started_at": 100.0, "updated_at": 220.0},
            {"id": 3, "severity": "medium", "status": "false_positive", "zone_name": "Вход", "started_at": 110.0, "updated_at": 130.0},
        ]

        summary = build_incident_status_summary(incidents)

        self.assertEqual(summary["active"], 2)
        self.assertEqual(summary["critical"], 1)
        self.assertEqual(summary["false_positive"], 1)
        self.assertEqual(summary["zones_under_alert"], 2)
        self.assertEqual(summary["mean_response_minutes"], 2.0)
        self.assertEqual(summary["unassigned_active"], 2)

    def test_zone_risk_rows_prioritize_active_and_critical(self):
        incidents = [
            {"severity": "critical", "status": "new", "zone_name": "Склад", "started_at": 300.0},
            {"severity": "high", "status": "escalated", "zone_name": "Склад", "started_at": 200.0},
            {"severity": "medium", "status": "resolved", "zone_name": "Вход", "started_at": 250.0},
        ]

        rows = build_zone_risk_rows(incidents)

        self.assertEqual(rows[0]["Зона"], "Склад")
        self.assertEqual(rows[0]["Активных"], 2)
        self.assertEqual(rows[0]["Критических"], 1)

    def test_source_risk_rows_put_offline_sources_first(self):
        sources = [
            {"id": 1, "name": "Камера 1"},
            {"id": 2, "name": "Камера 2"},
        ]
        statuses = [
            {"source_id": 1, "is_connected": True, "status": "online", "fps": 12.0, "reconnect_count": 0, "last_error": ""},
            {"source_id": 2, "is_connected": False, "status": "offline", "fps": 0.0, "reconnect_count": 3, "last_error": "timeout"},
        ]
        incidents = [
            {"source_id": 1, "severity": "high", "status": "new"},
            {"source_id": 2, "severity": "critical", "status": "escalated"},
        ]

        rows = build_source_risk_rows(sources, statuses, incidents)

        self.assertEqual(rows[0]["Источник"], "Камера 2")
        self.assertEqual(rows[0]["Соединение"], "offline")
        self.assertEqual(rows[1]["Источник"], "Камера 1")

    def test_camera_health_summary_detects_degraded_and_offline_sources(self):
        sources = [{"id": 1}, {"id": 2}, {"id": 3}]
        statuses = [
            {"source_id": 1, "is_connected": True, "fps": 10.0, "reconnect_count": 0, "last_error": ""},
            {"source_id": 2, "is_connected": True, "fps": 1.5, "reconnect_count": 1, "last_error": ""},
            {"source_id": 3, "is_connected": False, "fps": 0.0, "reconnect_count": 0, "last_error": "offline"},
        ]

        summary = build_camera_health_summary(sources, statuses)

        self.assertEqual(summary["healthy"], 1)
        self.assertEqual(summary["degraded"], 1)
        self.assertEqual(summary["offline"], 1)

    def test_incident_queue_rows_prioritize_critical_then_recent(self):
        incidents = [
            {
                "id": 1,
                "severity": "high",
                "status": "new",
                "incident_type": "loitering",
                "source_name": "Камера A",
                "zone_name": "Вход",
                "assigned_to": "Operator 1",
                "started_at": 100.0,
            },
            {
                "id": 3,
                "severity": "critical",
                "status": "new",
                "incident_type": "tailgating",
                "source_name": "Камера C",
                "zone_name": "Шлагбаум",
                "assigned_to": "Operator 2",
                "started_at": 110.0,
            },
            {
                "id": 2,
                "severity": "critical",
                "status": "new",
                "incident_type": "intrusion",
                "source_name": "Камера B",
                "zone_name": "Периметр",
                "assigned_to": "",
                "started_at": 90.0,
            },
        ]

        rows = build_incident_queue_rows(incidents, limit=3)

        self.assertEqual(rows[0]["ID"], 2)
        self.assertEqual(rows[1]["ID"], 3)
        self.assertEqual(rows[2]["ID"], 1)
        self.assertEqual(rows[0]["Owner"], "не назначен")
        self.assertIn("SLA", rows[0])

    def test_operator_workload_rows_group_by_owner(self):
        incidents = [
            {"status": "new", "severity": "critical", "assigned_to": "Shift A", "started_at": 100.0},
            {"status": "in_progress", "severity": "high", "assigned_to": "Shift A", "started_at": 200.0},
            {"status": "escalated", "severity": "medium", "assigned_to": "", "started_at": 300.0},
            {"status": "resolved", "severity": "critical", "assigned_to": "Shift B", "started_at": 400.0},
        ]

        rows = build_operator_workload_rows(incidents, limit=5)

        self.assertEqual(rows[0]["Ответственный"], "не назначен")
        self.assertEqual(rows[0]["Активных кейсов"], 1)
        self.assertGreaterEqual(rows[0]["Overdue"], 1)
        shift_a = next(row for row in rows if row["Ответственный"] == "Shift A")
        self.assertEqual(shift_a["Активных кейсов"], 2)
        self.assertEqual(shift_a["Critical"], 1)


if __name__ == "__main__":
    unittest.main()
