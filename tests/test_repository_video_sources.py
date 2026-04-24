import os
import tempfile
import unittest

from db import repository


class RepositoryVideoSourceConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = repository.DB_PATH
        repository.DB_PATH = os.path.join(self.temp_dir.name, "test_monitoring.db")
        repository.init_db()

    def tearDown(self):
        repository.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_video_source_processing_config_is_persisted(self):
        repository.create_video_source(
            name="Gate camera",
            source_type="rtsp",
            source_url="rtsp://gate",
            location="Gate A",
            description="Primary gate camera",
            is_active=True,
            enable_roi=True,
            roi_x=12,
            roi_y=18,
            roi_w=44,
            roi_h=52,
            rule_count_enabled=True,
            rule_n=4,
            rule_t=20,
            rule_disappear_enabled=False,
            rule_disappear_seconds=9,
            prolonged_presence_seconds=33,
        )

        source = repository.load_video_sources()[0]
        self.assertTrue(source["enable_roi"])
        self.assertEqual(source["roi_x"], 12.0)
        self.assertEqual(source["roi_y"], 18.0)
        self.assertEqual(source["roi_w"], 44.0)
        self.assertEqual(source["roi_h"], 52.0)
        self.assertTrue(source["rule_count_enabled"])
        self.assertEqual(source["rule_n"], 4)
        self.assertEqual(source["rule_t"], 20)
        self.assertFalse(source["rule_disappear_enabled"])
        self.assertEqual(source["rule_disappear_seconds"], 9)
        self.assertEqual(source["prolonged_presence_seconds"], 33)

    def test_video_source_processing_config_is_normalized_on_update(self):
        repository.create_video_source(
            name="Gate camera",
            source_type="rtsp",
            source_url="rtsp://gate",
            location="Gate A",
            description="Primary gate camera",
            is_active=False,
        )
        source_id = repository.load_video_sources()[0]["id"]

        repository.update_video_source(
            source_id=source_id,
            name="Gate camera",
            source_type="rtsp",
            source_url="rtsp://gate",
            location="Gate A",
            description="Updated",
            enable_roi=True,
            roi_x=98,
            roi_y=97,
            roi_w=50,
            roi_h=50,
            rule_count_enabled=True,
            rule_n=0,
            rule_t=0,
            rule_disappear_enabled=True,
            rule_disappear_seconds=0,
            prolonged_presence_seconds=0,
        )

        source = repository.load_video_sources()[0]
        self.assertEqual(source["roi_x"], 98.0)
        self.assertEqual(source["roi_y"], 97.0)
        self.assertEqual(source["roi_w"], 2.0)
        self.assertEqual(source["roi_h"], 3.0)
        self.assertEqual(source["rule_n"], 1)
        self.assertEqual(source["rule_t"], 1)
        self.assertEqual(source["rule_disappear_seconds"], 1)
        self.assertEqual(source["prolonged_presence_seconds"], 1)


if __name__ == "__main__":
    unittest.main()
