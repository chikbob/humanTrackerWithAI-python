import unittest

from config.app_config import build_tracker_runtime_config, normalize_tracker_type
from core.tracking import run_detection_with_optional_tracking


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(("predict", frame, kwargs))
        return ["predict-result"]

    def track(self, frame, **kwargs):
        self.calls.append(("track", frame, kwargs))
        return ["track-result"]


class TrackingRuntimeTests(unittest.TestCase):
    def test_normalize_tracker_type_falls_back_to_default(self):
        self.assertEqual(normalize_tracker_type("unknown"), "bytetrack")

    def test_build_tracker_runtime_config_for_detect_only(self):
        config = build_tracker_runtime_config("detect_only")
        self.assertFalse(config["use_tracking"])
        self.assertIsNone(config["tracker_config"])

    def test_run_detection_uses_predict_for_detect_only(self):
        model = FakeModel()
        result = run_detection_with_optional_tracking(
            model,
            frame_bgr="frame",
            tracker_type="detect_only",
            inference_size=512,
            conf_threshold=0.5,
        )
        self.assertEqual(result, ["predict-result"])
        self.assertEqual(model.calls[0][0], "predict")

    def test_run_detection_uses_track_for_tracking_mode(self):
        model = FakeModel()
        result = run_detection_with_optional_tracking(
            model,
            frame_bgr="frame",
            tracker_type="botsort",
            inference_size=640,
            conf_threshold=0.4,
        )
        self.assertEqual(result, ["track-result"])
        self.assertEqual(model.calls[0][0], "track")
        self.assertEqual(model.calls[0][2]["tracker"], "botsort.yaml")


if __name__ == "__main__":
    unittest.main()
