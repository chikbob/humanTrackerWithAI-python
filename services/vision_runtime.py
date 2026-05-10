"""Runtime helpers for frame analysis and kiosk-style checkpoint recognition."""

from __future__ import annotations

import base64
import os
import time
import uuid
from functools import lru_cache

import cv2
import numpy as np

from config.app_config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_INFERENCE_SIZE, DEFAULT_MODEL_NAME
from db.repository import ensure_snapshot_dir


AVAILABLE_MODELS = ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt")
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def list_available_models() -> list[dict]:
    models = []
    for name in AVAILABLE_MODELS:
        path = os.path.join(APP_DIR, name)
        models.append(
            {
                "name": name,
                "label": _build_model_label(name),
                "available": os.path.exists(path),
                "path": path,
            }
        )
    return models


def _build_model_label(name: str) -> str:
    labels = {
        "yolov8n.pt": "YOLOv8 Nano",
        "yolov8s.pt": "YOLOv8 Small",
        "yolov8m.pt": "YOLOv8 Medium",
    }
    return labels.get(name, name)


def _normalize_model_name(model_name: str | None) -> str:
    candidate = (model_name or DEFAULT_MODEL_NAME).strip()
    available_names = {item["name"] for item in list_available_models() if item["available"]}
    if candidate in available_names:
        return candidate
    if DEFAULT_MODEL_NAME in available_names:
        return DEFAULT_MODEL_NAME
    if available_names:
        return sorted(available_names)[0]
    raise RuntimeError("no_local_models_available")


@lru_cache(maxsize=4)
def load_model(model_name: str):
    from ultralytics import YOLO

    normalized_name = _normalize_model_name(model_name)
    return YOLO(os.path.join(APP_DIR, normalized_name))


def decode_image_payload(image_base64: str):
    if not image_base64:
        raise ValueError("image_required")
    payload = image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64
    binary = base64.b64decode(payload)
    frame = cv2.imdecode(np.frombuffer(binary, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("invalid_image_payload")
    return frame


def encode_image_payload(frame_bgr) -> str:
    success, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    if not success:
        raise RuntimeError("image_encode_failed")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def save_snapshot(frame_bgr, *, prefix: str = "checkpoint") -> str:
    snapshot_dir = ensure_snapshot_dir()
    filename = f"{prefix}-{uuid.uuid4().hex[:12]}.jpg"
    target = os.path.join(snapshot_dir, filename)
    if not cv2.imwrite(target, frame_bgr):
        raise RuntimeError("snapshot_save_failed")
    return target


def analyze_frame(
    *,
    image_base64: str,
    model_name: str | None = None,
    confidence_threshold: float | None = None,
    inference_size: int | None = None,
    track_people_only: bool = True,
) -> dict:
    frame_bgr = decode_image_payload(image_base64)
    normalized_model = _normalize_model_name(model_name)
    model = load_model(normalized_model)
    conf = float(confidence_threshold or DEFAULT_CONFIDENCE_THRESHOLD)
    imgsz = int(inference_size or DEFAULT_INFERENCE_SIZE)

    started_at = time.time()
    results = model.predict(frame_bgr, imgsz=imgsz, conf=conf, verbose=False)
    processing_time_ms = round((time.time() - started_at) * 1000.0, 2)

    annotated = frame_bgr.copy()
    detections = []
    for result in results:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        for index, box in enumerate(xyxy):
            cls_id = int(cls_arr[index])
            class_name = model.names[cls_id]
            if track_people_only and class_name != "person":
                continue
            x1, y1, x2, y2 = map(int, box)
            score = float(conf_arr[index])
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": round(score, 4),
                    "box": [x1, y1, x2, y2],
                }
            )
            color = (65, 209, 131) if class_name == "person" else (83, 161, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{class_name} {score:.2f}",
                (x1, max(22, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

    return {
        "model_name": normalized_model,
        "processing_time_ms": processing_time_ms,
        "person_count": sum(1 for item in detections if item["class_name"] == "person"),
        "detections": detections,
        "image_width": int(frame_bgr.shape[1]),
        "image_height": int(frame_bgr.shape[0]),
        "annotated_image_base64": encode_image_payload(annotated),
        "frame_bgr": frame_bgr,
    }
