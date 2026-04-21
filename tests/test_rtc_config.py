import os
import unittest
from unittest.mock import patch

from config.rtc_config import build_rtc_configuration, describe_rtc_environment


class RTCConfigTests(unittest.TestCase):
    def test_build_rtc_configuration_reads_multiple_urls(self):
        env = {
            "STUN_URLS": "stun:stun1.example.org:3478, stun:stun2.example.org:3478",
            "TURN_URLS": "turn:turn.example.org:3478?transport=udp",
            "TURN_USERNAME": "demo-user",
            "TURN_PASSWORD": "demo-pass",
        }
        with patch.dict(os.environ, env, clear=False):
            rtc = build_rtc_configuration()
            self.assertEqual(len(rtc["iceServers"]), 2)
            self.assertEqual(rtc["iceServers"][0]["urls"][0], "stun:stun1.example.org:3478")
            self.assertEqual(rtc["iceServers"][1]["username"], "demo-user")

    def test_describe_rtc_environment_reports_turn_presence(self):
        env = {
            "STUN_URLS": "stun:stun.example.org:3478",
            "TURN_URLS": "turn:turn.example.org:3478?transport=tcp",
            "TURN_USERNAME": "demo-user",
            "TURN_PASSWORD": "demo-pass",
        }
        with patch.dict(os.environ, env, clear=False):
            description = describe_rtc_environment()
            self.assertTrue(description["has_stun"])
            self.assertTrue(description["has_turn"])
            self.assertEqual(description["ice_server_count"], 2)


if __name__ == "__main__":
    unittest.main()
