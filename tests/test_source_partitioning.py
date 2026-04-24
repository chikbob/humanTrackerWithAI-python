import unittest

try:
    from ui.sources import split_video_sources
except ModuleNotFoundError:  # pragma: no cover - optional UI dependency in minimal env
    split_video_sources = None


@unittest.skipIf(split_video_sources is None, "ui dependencies are not installed")
class SourcePartitioningTests(unittest.TestCase):
    def test_split_video_sources_separates_production_and_lab(self):
        production_sources, lab_sources = split_video_sources(
            [
                {"id": 1, "source_type": "rtsp", "name": "Cam 1"},
                {"id": 2, "source_type": "stream_url", "name": "Cam 2"},
                {"id": 3, "source_type": "usb_camera", "name": "Cam 3"},
                {"id": 4, "source_type": "browser_camera", "name": "Browser"},
            ]
        )

        self.assertEqual([source["id"] for source in production_sources], [1, 2, 3])
        self.assertEqual([source["id"] for source in lab_sources], [4])


if __name__ == "__main__":
    unittest.main()
