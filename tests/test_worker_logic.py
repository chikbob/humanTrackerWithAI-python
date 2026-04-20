import time
import unittest
from unittest.mock import patch

from video.worker import SourceWorker


class WorkerLogicTests(unittest.TestCase):
    def test_handle_stream_failure_suppresses_duplicate_offline_event(self):
        worker = SourceWorker()
        source = {"id": 1, "name": "Gate 1", "source_type": "rtsp", "source_url": "rtsp://gate-1", "last_seen": None}
        settings = {"reconnect_interval": 5, "model_name": "yolov8s.pt", "default_access_point_id": None}
        runtime = {"frame_index": 0, "reconnect_count": 0, "offline_event_sent": False, "next_retry_ts": 0.0, "last_error_text": ""}
        worker.connection_state[source["id"]] = runtime
        emitted = []

        with patch("video.worker.create_domain_entry_event", side_effect=lambda *args, **kwargs: emitted.append(kwargs)):
            worker._handle_stream_failure(source, settings, runtime, "timeout")
            worker._handle_stream_failure(source, settings, runtime, "timeout")

        self.assertEqual(len(emitted), 1)
        self.assertTrue(runtime["offline_event_sent"])
        self.assertGreater(runtime["next_retry_ts"], time.time() - 1)

    def test_process_source_in_retry_window_marks_reconnecting(self):
        worker = SourceWorker()
        source = {"id": 4, "name": "Gate 4", "source_type": "rtsp", "source_url": "rtsp://example", "last_seen": None}
        settings = {
            "confidence_threshold": 0.45,
            "frame_skip": 1,
            "inference_size": 512,
            "event_cooldown": 5,
            "reconnect_interval": 5,
            "source_timeout": 15,
            "model_name": "yolov8s.pt",
            "default_access_point_id": None,
        }
        worker.connection_state[source["id"]] = {
            "frame_index": 0,
            "reconnect_count": 1,
            "last_success_ts": 0.0,
            "offline_event_sent": True,
            "next_retry_ts": time.time() + 30,
            "last_error_text": "timeout",
        }
        written = []
        worker._write_status = lambda *args, **kwargs: written.append(args[1])

        worker._process_source(source, settings)
        self.assertIn("reconnecting", written)


if __name__ == "__main__":
    unittest.main()
