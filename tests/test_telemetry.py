import unittest
import time

from services.telemetry import build_prometheus_metrics, build_worker_runtime_metrics


class TelemetryTests(unittest.TestCase):
    def test_build_worker_runtime_metrics_counts_sources(self):
        now_ts = time.time()
        metrics = build_worker_runtime_metrics(
            video_sources=[
                {"id": 1, "is_active": True},
                {"id": 2, "is_active": False},
            ],
            worker_statuses=[
                {"source_id": 1, "is_connected": True, "fps": 10.0, "reconnect_count": 2, "last_frame_at": now_ts},
                {"source_id": 2, "is_connected": False, "fps": 0.0, "reconnect_count": 4, "last_frame_at": None},
            ],
            events=[
                {"timestamp": now_ts, "is_suspicious": True},
                {"timestamp": now_ts, "is_suspicious": False},
            ],
            settings={"source_timeout": "15"},
        )
        self.assertEqual(metrics["source_count_total"], 2)
        self.assertEqual(metrics["source_count_active"], 1)
        self.assertEqual(metrics["source_count_online"], 1)
        self.assertEqual(metrics["source_count_offline"], 1)
        self.assertEqual(metrics["source_count_degraded"], 1)
        self.assertEqual(metrics["worker_max_reconnect_count"], 4)

    def test_build_prometheus_metrics_renders_gauges(self):
        output = build_prometheus_metrics({"worker_avg_fps": 12.5, "source_count_online": 3})
        self.assertIn("human_tracker_worker_avg_fps 12.5", output)
        self.assertIn("human_tracker_source_count_online 3.0", output)

    def test_build_worker_runtime_metrics_counts_degraded_sources(self):
        now_ts = time.time()
        metrics = build_worker_runtime_metrics(
            video_sources=[{"id": 1, "is_active": True}],
            worker_statuses=[
                {"source_id": 1, "is_connected": True, "status": "online", "fps": 1.5, "reconnect_count": 0, "last_frame_at": now_ts}
            ],
            events=[],
            settings={"source_timeout": "15"},
        )
        self.assertEqual(metrics["source_count_online"], 1)
        self.assertEqual(metrics["source_count_degraded"], 1)


if __name__ == "__main__":
    unittest.main()
