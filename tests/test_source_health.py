import unittest

from services.source_health import normalize_source_runtime_status


class SourceHealthTests(unittest.TestCase):
    def test_online_source_with_good_fps_is_healthy(self):
        normalized = normalize_source_runtime_status(
            {"is_connected": True, "status": "online", "fps": 12.0, "reconnect_count": 0, "last_error": ""},
            source_timeout=15,
            now_ts=100.0,
        )
        self.assertEqual(normalized["connection_status"], "online")
        self.assertEqual(normalized["health_status"], "healthy")

    def test_low_fps_source_is_degraded(self):
        normalized = normalize_source_runtime_status(
            {"is_connected": True, "status": "online", "fps": 1.2, "reconnect_count": 0, "last_error": ""},
            source_timeout=15,
            now_ts=100.0,
        )
        self.assertEqual(normalized["health_status"], "degraded")

    def test_reconnecting_source_is_offline_for_health(self):
        normalized = normalize_source_runtime_status(
            {"is_connected": False, "status": "reconnecting", "fps": 0.0, "reconnect_count": 3, "last_error": "timeout"},
            source_timeout=15,
            now_ts=100.0,
        )
        self.assertEqual(normalized["connection_status"], "reconnecting")
        self.assertEqual(normalized["health_status"], "offline")


if __name__ == "__main__":
    unittest.main()
