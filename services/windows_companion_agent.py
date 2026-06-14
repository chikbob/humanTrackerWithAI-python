"""Remote-host desktop companion that relays Windows camera frames to the backend API."""

from __future__ import annotations

import argparse
import base64
import json
import socket
import time
from urllib import error, request

from services.local_camera import encode_frame_as_jpeg, open_local_camera


def _build_request(server_url: str, payload: dict) -> request.Request:
    body = json.dumps(payload).encode("utf-8")
    return request.Request(
        url=f"{server_url.rstrip('/')}/api/v1/desktop-companion/frame",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _post_frame(server_url: str, payload: dict) -> None:
    req = _build_request(server_url, payload)
    with request.urlopen(req, timeout=10) as response:
        if response.status >= 400:
            raise RuntimeError(f"upload_failed:{response.status}")
        response.read()


def run_agent(server_url: str, session_id: str, camera_index: int, interval_ms: int) -> None:
    host_name = socket.gethostname()
    print(f"[desktop-companion] server={server_url} session_id={session_id} camera_index={camera_index}")
    while True:
        cap, meta = open_local_camera(camera_index, width=640, height=480)
        if cap is None:
            print(f"[desktop-companion] camera_open_failed attempts={meta.get('attempts', [])}")
            time.sleep(2.0)
            continue
        print(f"[desktop-companion] connected backend={meta.get('backend_label')} warmup_attempt={meta.get('warmup_attempt')}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print(f"[desktop-companion] frame_read_failed backend={meta.get('backend_label')}")
                    break
                jpeg_bytes = encode_frame_as_jpeg(frame, quality=82)
                if not jpeg_bytes:
                    continue
                payload = {
                    "session_id": session_id,
                    "camera_index": int(camera_index),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "source_label": "Windows desktop companion",
                    "backend_label": meta.get("backend_label", ""),
                    "host_name": host_name,
                    "frame_base64": base64.b64encode(jpeg_bytes).decode("ascii"),
                }
                try:
                    _post_frame(server_url, payload)
                except error.URLError as exc:
                    print(f"[desktop-companion] upload_error reason={exc.reason}")
                except Exception as exc:  # pragma: no cover - runtime guard
                    print(f"[desktop-companion] upload_error reason={exc}")
                time.sleep(max(0.05, interval_ms / 1000.0))
        finally:
            cap.release()
        time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Windows camera frames to a remote Human Tracker host.")
    parser.add_argument("--server-url", required=True, help="Base URL of the remote API host, for example https://example.up.railway.app")
    parser.add_argument("--session-id", required=True, help="Companion session ID shown by the web app")
    parser.add_argument("--camera-index", type=int, default=0, help="Local Windows camera index")
    parser.add_argument("--interval-ms", type=int, default=300, help="Upload interval in milliseconds")
    args = parser.parse_args()
    run_agent(
        server_url=args.server_url,
        session_id=args.session_id,
        camera_index=args.camera_index,
        interval_ms=args.interval_ms,
    )


if __name__ == "__main__":
    main()
