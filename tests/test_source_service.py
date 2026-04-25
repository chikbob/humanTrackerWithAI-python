import unittest
import importlib.util
from unittest.mock import patch

CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None

if CV2_AVAILABLE:
    from services import source_service


class FakeCapture:
    def __init__(self, opened=True, reads=None):
        self._opened = opened
        self._reads = list(reads or [])
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._reads:
            return self._reads.pop(0)
        return False, None

    def release(self):
        self.released = True


@unittest.skipUnless(CV2_AVAILABLE, "opencv-python is not installed")
class SourceServiceTests(unittest.TestCase):
    def test_infer_source_type_uses_common_input_patterns(self):
        self.assertEqual(source_service.infer_source_type("rtsp://cam"), "rtsp")
        self.assertEqual(source_service.infer_source_type("https://cam/live.m3u8"), "stream_url")
        self.assertEqual(source_service.infer_source_type("0"), "usb_camera")
        self.assertEqual(source_service.infer_source_type("browser_camera"), "browser_camera")

    def test_validate_source_definition_rejects_mismatched_rtsp(self):
        errors, normalized_url = source_service.validate_source_definition(
            name="Gate camera",
            source_type="rtsp",
            source_url="https://example.com/live.m3u8",
        )
        self.assertTrue(errors)
        self.assertEqual(normalized_url, "https://example.com/live.m3u8")

    def test_validate_source_definition_normalizes_browser_camera(self):
        errors, normalized_url = source_service.validate_source_definition(
            name="Browser camera",
            source_type="browser_camera",
            source_url="",
        )
        self.assertEqual(errors, [])
        self.assertEqual(normalized_url, "browser_camera")

    def test_normalize_source_url_for_usb_camera(self):
        self.assertEqual(source_service.normalize_source_url("usb_camera", "0"), 0)
        self.assertEqual(source_service.normalize_source_url("usb_camera", "cam0"), "cam0")

    def test_normalize_source_url_for_browser_camera(self):
        self.assertEqual(source_service.normalize_source_url("browser_camera", ""), "browser_camera")

    def test_browser_camera_connection_check_is_virtual_success(self):
        ok, message = source_service.test_video_source_connection("browser_camera", "")
        self.assertTrue(ok)
        self.assertIn("Браузерная камера", message)

    def test_connection_check_handles_unopened_capture(self):
        with patch.object(source_service.cv2, "VideoCapture", return_value=FakeCapture(opened=False)):
            ok, message = source_service.test_video_source_connection("rtsp", "rtsp://example")
        self.assertFalse(ok)
        self.assertIn("не открылся", message)

    def test_connection_check_returns_validation_error_before_capture(self):
        ok, message = source_service.test_video_source_connection("stream_url", "rtsp://example")
        self.assertFalse(ok)
        self.assertIn("http://", message)

    def test_connection_check_reads_frames_successfully(self):
        with patch.object(
            source_service.cv2,
            "VideoCapture",
            return_value=FakeCapture(opened=True, reads=[(False, None), (True, object())]),
        ):
            ok, message = source_service.test_video_source_connection("rtsp", "rtsp://example")
        self.assertTrue(ok)
        self.assertIn("успешно", message)


if __name__ == "__main__":
    unittest.main()
