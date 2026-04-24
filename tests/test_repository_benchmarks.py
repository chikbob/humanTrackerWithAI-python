import os
import tempfile
import unittest

from db import repository


class RepositoryBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = repository.DB_PATH
        repository.DB_PATH = os.path.join(self.temp_dir.name, "test_monitoring.db")
        repository.init_db()

    def tearDown(self):
        repository.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_experiment_run_and_benchmark_results_are_persisted(self):
        run_id = repository.create_experiment_run(
            run_key="bench-001",
            scenario_name="detector_comparison",
            source_path="/tmp/video.mp4",
            notes="thesis benchmark",
        )
        repository.insert_benchmark_result(
            run_id=run_id,
            model_name="yolov8s.pt",
            tracker_type="bytetrack",
            frame_limit=120,
            warmup_frames=10,
            frames_processed=110,
            avg_latency_ms=18.5,
            p95_latency_ms=24.0,
            avg_fps=54.1,
            avg_detections_per_frame=1.4,
            tracked_frame_ratio=0.96,
            detection_count_total=154,
            metadata={"scenario_version": 1},
        )
        repository.complete_experiment_run(run_id=run_id)

        runs = repository.load_experiment_runs(limit=1)
        results = repository.load_benchmark_results(run_id=run_id)
        self.assertEqual(runs[0]["run_key"], "bench-001")
        self.assertEqual(runs[0]["status"], "completed")
        self.assertEqual(results[0]["model_name"], "yolov8s.pt")
        self.assertEqual(results[0]["metadata"]["scenario_version"], 1)


if __name__ == "__main__":
    unittest.main()
