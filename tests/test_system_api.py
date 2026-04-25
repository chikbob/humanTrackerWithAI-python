import unittest
import importlib.util

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    from services.system_api import build_incident_summary


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is not installed")
class SystemApiTests(unittest.TestCase):
    def test_incident_summary_tracks_case_metrics(self):
        incidents = [
            {
                "status": "in_progress",
                "severity": "high",
                "assigned_to": "Shift A",
                "started_at": 100.0,
                "acknowledged_at": 160.0,
                "resolved_at": None,
            },
            {
                "status": "resolved",
                "severity": "critical",
                "assigned_to": "",
                "started_at": 100.0,
                "acknowledged_at": 130.0,
                "resolved_at": 400.0,
            },
        ]

        summary = build_incident_summary(incidents)

        self.assertEqual(summary["active"], 1)
        self.assertEqual(summary["critical"], 1)
        self.assertEqual(summary["assigned"], 1)
        self.assertEqual(summary["mean_ack_minutes"], 0.75)
        self.assertEqual(summary["mean_resolution_minutes"], 5.0)


if __name__ == "__main__":
    unittest.main()
