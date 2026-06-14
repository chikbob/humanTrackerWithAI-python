"""Standalone localhost camera bridge for Windows clients."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from services.local_camera import encode_frame_as_jpeg, open_local_camera, read_local_camera_frame


app = FastAPI(title="HumanTracker Windows Camera Bridge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/camera/probe")
def camera_probe(camera_index: int = Query(0, ge=0, le=10), width: int = Query(640, ge=160, le=1920), height: int = Query(480, ge=120, le=1080)):
    frame, meta = read_local_camera_frame(camera_index, width=width, height=height)
    return {
        "camera_index": camera_index,
        "width": width,
        "height": height,
        "ok": frame is not None,
        "backend": meta.get("backend_label"),
        "attempts": meta.get("attempts", []),
    }


@app.get("/camera/frame")
def camera_frame(camera_index: int = Query(0, ge=0, le=10), width: int = Query(640, ge=160, le=1920), height: int = Query(480, ge=120, le=1080)):
    frame, meta = read_local_camera_frame(camera_index, width=width, height=height)
    if frame is None:
        raise HTTPException(status_code=503, detail="local_bridge_open_failed:" + ", ".join(meta.get("attempts", [])))
    jpeg_bytes = encode_frame_as_jpeg(frame)
    if jpeg_bytes is None:
        raise HTTPException(status_code=500, detail="local_bridge_encode_failed")
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/camera/stream")
def camera_stream(camera_index: int = Query(0, ge=0, le=10), width: int = Query(640, ge=160, le=1920), height: int = Query(480, ge=120, le=1080)):
    def generate():
        cap, meta = open_local_camera(camera_index, width=width, height=height)
        if cap is None:
            message = ("local_bridge_open_failed:" + ", ".join(meta.get("attempts", []))).encode("utf-8")
            yield b"--frame\r\nContent-Type: text/plain\r\n\r\n" + message + b"\r\n"
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    message = f"local_bridge_frame_failed:{meta.get('backend_label')}".encode("utf-8")
                    yield b"--frame\r\nContent-Type: text/plain\r\n\r\n" + message + b"\r\n"
                    break
                jpeg_bytes = encode_frame_as_jpeg(frame)
                if jpeg_bytes is None:
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
        finally:
            cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
