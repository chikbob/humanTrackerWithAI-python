import unittest

from services.rules import build_effective_rule_profile


class RulesServiceTests(unittest.TestCase):
    def test_active_zone_and_rules_override_source_defaults(self):
        source = {
            "enable_roi": True,
            "roi_x": 20,
            "roi_y": 20,
            "roi_w": 60,
            "roi_h": 60,
            "rule_count_enabled": False,
            "rule_n": 3,
            "rule_t": 10,
            "rule_disappear_enabled": True,
            "rule_disappear_seconds": 5,
            "prolonged_presence_seconds": 10,
        }
        zones = [
            {"id": 1, "zone_type": "entry", "x": 10, "y": 15, "w": 30, "h": 35, "is_active": True},
            {"id": 2, "zone_type": "observation", "x": 40, "y": 40, "w": 30, "h": 30, "is_active": True},
        ]
        rules = [
            {"id": 1, "zone_id": 1, "rule_type": "crowding", "threshold_seconds": 12, "threshold_count": 5, "cooldown_seconds": 7, "is_active": True},
            {"id": 2, "zone_id": 1, "rule_type": "loitering", "threshold_seconds": 25, "threshold_count": 1, "cooldown_seconds": 5, "is_active": True},
            {"id": 3, "zone_id": 1, "rule_type": "track_loss", "threshold_seconds": 8, "threshold_count": 1, "cooldown_seconds": 5, "is_active": True},
        ]

        profile = build_effective_rule_profile(source=source, zones=zones, zone_rules=rules)

        self.assertEqual(profile["primary_zone"]["id"], 1)
        self.assertEqual(profile["roi_config"]["roi_x"], 10)
        self.assertEqual(profile["roi_config"]["roi_y"], 15)
        self.assertEqual(profile["event_settings"]["rule_n"], 5)
        self.assertEqual(profile["event_settings"]["rule_t"], 12)
        self.assertEqual(profile["event_settings"]["prolonged_presence_seconds"], 25)
        self.assertEqual(profile["event_settings"]["rule_disappear_seconds"], 8)


if __name__ == "__main__":
    unittest.main()
