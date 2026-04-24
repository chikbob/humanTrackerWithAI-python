import unittest

from research.export import build_markdown_table
from research.scenarios import build_named_scenario


class ResearchScenarioTests(unittest.TestCase):
    def test_detector_comparison_scenario_contains_all_models(self):
        scenario = build_named_scenario("detector_comparison", frame_limit=60, warmup_frames=5)
        self.assertEqual(scenario["name"], "detector_comparison")
        self.assertGreaterEqual(len(scenario["cases"]), 3)
        self.assertTrue(all(case["tracker_type"] == "bytetrack" for case in scenario["cases"]))

    def test_tracker_comparison_scenario_contains_detect_only(self):
        scenario = build_named_scenario("tracker_comparison", model_name="yolov8s.pt", frame_limit=80, warmup_frames=8)
        trackers = {case["tracker_type"] for case in scenario["cases"]}
        self.assertIn("detect_only", trackers)
        self.assertIn("bytetrack", trackers)

    def test_markdown_export_builds_table(self):
        output = build_markdown_table(
            [
                {
                    "scenario_name": "detector_comparison",
                    "model_name": "yolov8s.pt",
                    "tracker_type": "bytetrack",
                    "frame_limit": 120,
                    "warmup_frames": 10,
                    "frames_processed": 110,
                    "avg_latency_ms": 20.5,
                    "p95_latency_ms": 25.1,
                    "avg_fps": 48.7,
                    "avg_detections_per_frame": 1.3,
                    "tracked_frame_ratio": 0.95,
                    "detection_count_total": 143,
                }
            ]
        )
        self.assertIn("| scenario_name |", output)
        self.assertIn("yolov8s.pt", output)


if __name__ == "__main__":
    unittest.main()
