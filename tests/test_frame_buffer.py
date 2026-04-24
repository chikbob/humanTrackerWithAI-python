import unittest

from video.frame_buffer import LatestFrameBuffer


class FakeFrame:
    def __init__(self, value):
        self.value = value

    def copy(self):
        return FakeFrame(self.value)


class LatestFrameBufferTests(unittest.TestCase):
    def test_get_latest_returns_most_recent_frame(self):
        buffer = LatestFrameBuffer()
        buffer.put(FakeFrame("frame-1"), timestamp=100.0)
        buffer.put(FakeFrame("frame-2"), timestamp=101.0)

        frame, timestamp, sequence = buffer.get_latest()
        self.assertEqual(frame.value, "frame-2")
        self.assertEqual(timestamp, 101.0)
        self.assertEqual(sequence, 2)

    def test_get_if_newer_skips_already_processed_sequence(self):
        buffer = LatestFrameBuffer()
        seq1 = buffer.put(FakeFrame("frame-1"), timestamp=100.0)
        self.assertIsNone(buffer.get_if_newer(seq1))

        seq2 = buffer.put(FakeFrame("frame-2"), timestamp=103.0)
        frame, timestamp, sequence = buffer.get_if_newer(seq1)
        self.assertEqual(sequence, seq2)
        self.assertEqual(frame.value, "frame-2")
        self.assertEqual(timestamp, 103.0)

    def test_latest_age_uses_last_frame_timestamp(self):
        buffer = LatestFrameBuffer()
        self.assertIsNone(buffer.latest_age(now_ts=100.0))

        buffer.put(FakeFrame("frame-1"), timestamp=98.5)
        self.assertAlmostEqual(buffer.latest_age(now_ts=100.0), 1.5)


if __name__ == "__main__":
    unittest.main()
