import importlib.util
import unittest


PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    from ui.journal import _build_incident_timeline_rows, _build_journal_summary


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is not installed")
class JournalHelperTests(unittest.TestCase):
    def test_build_journal_summary_counts_active_assigned_and_overdue(self):
        incidents = [
            {"status": "new", "severity": "critical", "assigned_to": "", "started_at": 1.0},
            {"status": "resolved", "severity": "high", "assigned_to": "Shift A", "started_at": 2.0},
        ]

        summary = _build_journal_summary(incidents)

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["active"], 1)
        self.assertEqual(summary["critical"], 1)
        self.assertEqual(summary["assigned"], 1)

    def test_build_incident_timeline_rows_returns_three_steps(self):
        rows = _build_incident_timeline_rows(
            {"started_at": 100.0, "acknowledged_at": 150.0, "resolved_at": 200.0}
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["Этап"], "Создан")
        self.assertEqual(rows[1]["Этап"], "Подтвержден")
        self.assertEqual(rows[2]["Этап"], "Закрыт")


if __name__ == "__main__":
    unittest.main()
