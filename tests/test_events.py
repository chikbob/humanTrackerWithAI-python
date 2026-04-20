import unittest

from services.events import _notify_once_for_track


class EventLogicTests(unittest.TestCase):
    def test_duplicate_event_suppression_per_track_flag(self):
        session = {"track_domain_flags": {}}
        self.assertTrue(_notify_once_for_track(session, "track:1", "entered"))
        self.assertFalse(_notify_once_for_track(session, "track:1", "entered"))
        self.assertTrue(_notify_once_for_track(session, "track:1", "left"))


if __name__ == "__main__":
    unittest.main()
