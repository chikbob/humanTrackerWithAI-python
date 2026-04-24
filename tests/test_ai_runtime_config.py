import unittest

from config.app_config import build_ai_runtime_settings, normalize_ai_quality_profile


class AiRuntimeConfigTests(unittest.TestCase):
    def test_normalize_ai_quality_profile_falls_back_to_balanced(self):
        self.assertEqual(normalize_ai_quality_profile("unknown"), "balanced")

    def test_build_ai_runtime_settings_uses_profile_defaults(self):
        runtime = build_ai_runtime_settings({"ai_quality_profile": "accuracy"})
        self.assertEqual(runtime["profile_key"], "accuracy")
        self.assertEqual(runtime["inference_size"], 640)
        self.assertEqual(runtime["tracker_type"], "botsort")

    def test_build_ai_runtime_settings_applies_source_overrides(self):
        runtime = build_ai_runtime_settings(
            {"ai_quality_profile": "balanced", "confidence_threshold": "0.45", "inference_size": "512", "tracker_type": "bytetrack"},
            {
                "conf_threshold_override": 0.7,
                "inference_size_override": 960,
                "tracker_type_override": "detect_only",
                "incident_threshold_override": 0.8,
                "ai_profile_override": "latency",
            },
        )
        self.assertEqual(runtime["profile_key"], "latency")
        self.assertEqual(runtime["confidence_threshold"], 0.7)
        self.assertEqual(runtime["inference_size"], 960)
        self.assertEqual(runtime["tracker_type"], "detect_only")
        self.assertEqual(runtime["incident_score_threshold"], 0.8)


if __name__ == "__main__":
    unittest.main()
