import os
import tempfile
import unittest

from db import repository


class RepositoryZoneRulesTests(unittest.TestCase):
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
        repository.create_zone(
            source_id=self.source_id,
            name="Restricted area",
            zone_type="restricted",
            x=15,
            y=10,
            w=55,
            h=60,
            is_active=True,
            description="Server room entry",
        )
        self.zone_id = repository.load_zones(source_id=self.source_id)[0]["id"]

    def tearDown(self):
        repository.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_zone_rule_is_persisted_and_normalized(self):
        repository.create_zone_rule(
            zone_id=self.zone_id,
            rule_type="loitering",
            threshold_seconds=0,
            threshold_count=0,
            cooldown_seconds=-1,
            is_active=True,
            severity="high",
            description="Presence over threshold",
        )
        rule = repository.load_zone_rules(source_id=self.source_id)[0]
        self.assertEqual(rule["rule_type"], "loitering")
        self.assertEqual(rule["threshold_seconds"], 1)
        self.assertEqual(rule["threshold_count"], 1)
        self.assertEqual(rule["cooldown_seconds"], 0)
        self.assertEqual(rule["severity"], "high")
        self.assertTrue(rule["is_active"])

    def test_zone_rule_can_be_updated_and_deactivated(self):
        repository.create_zone_rule(
            zone_id=self.zone_id,
            rule_type="crowding",
            threshold_seconds=15,
            threshold_count=4,
            cooldown_seconds=6,
            is_active=True,
            severity="critical",
            description="Crowding threshold",
        )
        rule_id = repository.load_zone_rules(source_id=self.source_id)[0]["id"]
        repository.update_zone_rule(
            rule_id=rule_id,
            zone_id=self.zone_id,
            rule_type="track_loss",
            threshold_seconds=9,
            threshold_count=2,
            cooldown_seconds=3,
            severity="medium",
            description="Track disappeared",
        )
        repository.set_zone_rule_active(rule_id=rule_id, is_active=False)
        rule = repository.load_zone_rules(zone_id=self.zone_id)[0]
        self.assertEqual(rule["rule_type"], "track_loss")
        self.assertEqual(rule["threshold_seconds"], 9)
        self.assertEqual(rule["cooldown_seconds"], 3)
        self.assertEqual(rule["severity"], "medium")
        self.assertFalse(rule["is_active"])


if __name__ == "__main__":
    unittest.main()
