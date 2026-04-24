import os
import tempfile
import unittest

from db import repository


class RepositoryZonesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = repository.DB_PATH
        repository.DB_PATH = os.path.join(self.temp_dir.name, "test_monitoring.db")
        repository.init_db()
        repository.create_video_source(
            name="Gate camera",
            source_type="rtsp",
            source_url="rtsp://gate",
            location="Gate A",
            description="Primary gate camera",
            is_active=True,
        )
        self.source_id = repository.load_video_sources()[0]["id"]

    def tearDown(self):
        repository.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_zone_config_is_persisted_and_normalized(self):
        repository.create_zone(
            source_id=self.source_id,
            name="Restricted area",
            zone_type="restricted",
            x=95,
            y=97,
            w=50,
            h=50,
            is_active=True,
            description="Server room entry",
        )

        zone = repository.load_zones(source_id=self.source_id)[0]
        self.assertEqual(zone["name"], "Restricted area")
        self.assertEqual(zone["zone_type"], "restricted")
        self.assertEqual(zone["x"], 95.0)
        self.assertEqual(zone["y"], 97.0)
        self.assertEqual(zone["w"], 5.0)
        self.assertEqual(zone["h"], 3.0)
        self.assertTrue(zone["is_active"])

    def test_zone_can_be_updated_and_deactivated(self):
        repository.create_zone(
            source_id=self.source_id,
            name="Observation area",
            zone_type="observation",
            x=20,
            y=20,
            w=60,
            h=60,
            is_active=True,
            description="Initial zone",
        )
        zone_id = repository.load_zones(source_id=self.source_id)[0]["id"]

        repository.update_zone(
            zone_id=zone_id,
            source_id=self.source_id,
            name="Entry area",
            zone_type="entry",
            x=5,
            y=10,
            w=40,
            h=35,
            description="Updated zone",
        )
        repository.set_zone_active(zone_id=zone_id, is_active=False)

        zone = repository.load_zones(source_id=self.source_id)[0]
        self.assertEqual(zone["name"], "Entry area")
        self.assertEqual(zone["zone_type"], "entry")
        self.assertEqual(zone["x"], 5.0)
        self.assertEqual(zone["y"], 10.0)
        self.assertEqual(zone["w"], 40.0)
        self.assertEqual(zone["h"], 35.0)
        self.assertFalse(zone["is_active"])


if __name__ == "__main__":
    unittest.main()
