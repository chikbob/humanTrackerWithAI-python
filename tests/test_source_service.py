import unittest
from unittest.mock import patch

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


class SourceServiceTests(unittest.TestCase):
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
