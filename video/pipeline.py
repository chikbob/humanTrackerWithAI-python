"""Background frame processing for production video sources."""

from __future__ import annotations

import time

import cv2
from ultralytics import YOLO

from config.app_config import DEFAULT_TRACKER_TYPE
from core.tracking import run_detection_with_optional_tracking
from services.events import process_disappeared_tracks, register_detection_and_entry_events


def load_worker_model(model_name: str):
    return YOLO(model_name)


def process_source_frame(
    *,
    frame_bgr,
    model,
    source: dict,
    session_state,
    session: dict,
    frame_index: int,
    conf_threshold: float,
    inference_size: int,
    roi_config: dict,
    event_settings: dict,
    tracker_type: str = DEFAULT_TRACKER_TYPE,
    tracking_iou_threshold: float = 0.5,
    incident_score_threshold: float = 0.55,
):
    """Process one frame from a production source and return an annotated image."""
    start_ts = time.time()
    results = run_detection_with_optional_tracking(
        model,
        frame_bgr,
        tracker_type=tracker_type,
        inference_size=inference_size,
        conf_threshold=conf_threshold,
        iou_threshold=tracking_iou_threshold,
    )
    processing_time_ms = (time.time() - start_ts) * 1000.0

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame_rgb.shape
    detections = []

    for result in results:
        boxes = result.boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None
        xyxy = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        for i, box in enumerate(xyxy):
            cls_id = int(cls_arr[i])
            cls_name = model.names[cls_id]
            if cls_name != "person":
                continue
            conf = float(conf_arr[i])
            track_id = int(ids[i]) if ids is not None else None
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            roi_inside = _is_inside_roi(cx, cy, frame_w, frame_h, roi_config)
            track_key = f"{session['id']}:{track_id}:{cls_name}" if track_id is not None else None
            prev_inside = session["track_inside_roi"].get(track_key, False) if track_key else False
            roi_enter = bool(roi_inside and not prev_inside)
            roi_exit = bool((not roi_inside) and prev_inside)
            if track_key is not None:
                session["track_inside_roi"][track_key] = roi_inside
                session["track_last_seen"][track_key] = time.time()
                session["track_class_by_key"][track_key] = cls_name
                session["disappeared_track_keys"].discard(track_key)

            detection = {
                "class_id": cls_id,
                "class_name": cls_name,
                "is_animal": False,
                "animal_group": None,
                "confidence": conf,
                "box": [x1, y1, x2, y2],
                "track_id": track_id,
                "center_x": cx,
                "center_y": cy,
                "frame_width": frame_w,
                "frame_height": frame_h,
                "roi_inside": roi_inside,
                "roi_enter": roi_enter if track_id is not None else roi_inside,
                "roi_exit": roi_exit if track_id is not None else False,
                "incident_score": _compute_incident_score(conf=conf, roi_inside=roi_inside, track_id=track_id),
            }
            detections.append(detection)
            if detection["incident_score"] >= incident_score_threshold:
                register_detection_and_entry_events(
                    session_state,
                    session_state.db_insert_event,
                    session=session,
                    frame_index=frame_index,
                    detection=detection,
                    source_type=source["source_type"],
                    settings=event_settings,
                    notify_callback=lambda _msg: None,
                )
            _draw_worker_box(frame_rgb, detection)

    process_disappeared_tracks(
        session_state,
        session_state.db_insert_event,
        session=session,
        frame_index=frame_index,
        source_type=source["source_type"],
        frame_width=frame_w,
        frame_height=frame_h,
        rule_disappear_enabled=event_settings["rule_disappear_enabled"],
        rule_disappear_seconds=event_settings["rule_disappear_seconds"],
        enable_notifications=False,
        notify_callback=lambda _msg: None,
        default_access_point_id=event_settings.get("default_access_point_id"),
    )
    frame_rgb = _draw_roi_overlay(frame_rgb, roi_config)
    return frame_rgb, detections, processing_time_ms


def _get_roi_rect(frame_w: int, frame_h: int, roi_config: dict):
    x1 = int(frame_w * (roi_config["roi_x"] / 100.0))
    y1 = int(frame_h * (roi_config["roi_y"] / 100.0))
    x2 = int(frame_w * min(1.0, (roi_config["roi_x"] + roi_config["roi_w"]) / 100.0))
    y2 = int(frame_h * min(1.0, (roi_config["roi_y"] + roi_config["roi_h"]) / 100.0))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _is_inside_roi(cx: float, cy: float, frame_w: int, frame_h: int, roi_config: dict) -> bool:
    if not roi_config["enable_roi"]:
        return True
    x1, y1, x2, y2 = _get_roi_rect(frame_w, frame_h, roi_config)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _draw_roi_overlay(frame_rgb, roi_config: dict):
    if not roi_config["enable_roi"]:
        return frame_rgb
    h, w, _ = frame_rgb.shape
    x1, y1, x2, y2 = _get_roi_rect(w, h, roi_config)
    overlay = frame_rgb.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), -1)
    cv2.addWeighted(overlay, 0.15, frame_rgb, 0.85, 0, frame_rgb)
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 180, 255), 2)
    cv2.putText(frame_rgb, "ENTRY ZONE", (x1 + 8, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
    return frame_rgb


def _draw_worker_box(frame_rgb, detection: dict):
    x1, y1, x2, y2 = map(int, detection["box"])
    score = round(float(detection.get("incident_score") or 0.0), 2)
    label = (
        f"person id:{detection['track_id']} s:{score}"
        if detection.get("track_id") is not None
        else f"person s:{score}"
    )
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (90, 220, 120), 2)
    cv2.putText(
        frame_rgb,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (90, 220, 120),
        2,
    )


def _compute_incident_score(*, conf: float, roi_inside: bool, track_id) -> float:
    score = float(conf)
    if roi_inside:
        score += 0.15
    if track_id is not None:
        score += 0.05
    return max(0.0, min(1.0, score))
