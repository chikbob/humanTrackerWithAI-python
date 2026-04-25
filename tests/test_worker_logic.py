import time
import unittest
import importlib.util
from unittest.mock import patch
from collections import deque

CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None

if CV2_AVAILABLE:
    from video.worker import SourceWorker


@unittest.skipUnless(CV2_AVAILABLE, "opencv-python is not installed")
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

    def test_flush_ready_incident_evidence_attaches_clip(self):
        worker = SourceWorker()
        source = {"id": 7, "name": "Gate 7", "source_type": "rtsp", "source_url": "rtsp://gate-7"}
        session = {
            "id": "worker-7-demo",
            "evidence_buffer": deque(
                [
                    {"timestamp": 100.0, "frame_bgr": "frame-1"},
                    {"timestamp": 101.0, "frame_bgr": "frame-2"},
                    {"timestamp": 102.0, "frame_bgr": "frame-3"},
                ]
            ),
            "pending_evidence_jobs": [
                {
                    "event_id": "evt-77",
                    "event_ts": 101.0,
                    "snapshot_path": "/tmp/incident.jpg",
                    "retention_until": 500.0,
                    "target_ready_ts": 102.0,
                }
            ],
        }
        settings = {
            "incident_evidence_pre_seconds": 1,
            "incident_evidence_post_seconds": 1,
            "incident_evidence_fps": 4,
        }
        attached = []

        with patch("video.worker.collect_evidence_frames", return_value=["frame-1", "frame-2"]) as collect_mock, patch(
            "video.worker.write_evidence_clip_atomic",
            return_value="/tmp/incident.mp4",
        ) as write_mock, patch(
            "video.worker.attach_event_evidence",
            side_effect=lambda **kwargs: attached.append(kwargs),
        ):
            worker._flush_ready_incident_evidence(source, session, settings, now_ts=103.0)

        collect_mock.assert_called_once()
        write_mock.assert_called_once()
        self.assertEqual(len(attached), 1)
        self.assertEqual(attached[0]["event_id"], "evt-77")
        self.assertEqual(attached[0]["evidence_clip_path"], "/tmp/incident.mp4")
        self.assertEqual(session["pending_evidence_jobs"], [])


if __name__ == "__main__":
    unittest.main()
