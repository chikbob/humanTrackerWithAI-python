import importlib.util
import unittest


PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    from ui.dashboard import _build_dashboard_guidance


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is not installed")
class DashboardGuidanceTests(unittest.TestCase):
    def test_guidance_mentions_missing_cameras(self):
        messages = _build_dashboard_guidance(video_sources=[], worker_statuses=[], incidents=[])
        self.assertTrue(any("Камеры ещё не добавлены" in message for message in messages))

    def test_guidance_mentions_inactive_sources(self):
        messages = _build_dashboard_guidance(
            video_sources=[{"id": 1, "is_active": False}],
            worker_statuses=[],
            incidents=[],
        )
        self.assertTrue(any("ни один не активирован" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
