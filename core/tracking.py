"""Tracking strategy helpers for configurable inference runtime."""

from __future__ import annotations

from config.app_config import build_tracker_runtime_config


def run_detection_with_optional_tracking(
    model,
    frame_bgr,
    *,
    tracker_type: str,
    inference_size: int,
    conf_threshold: float,
    iou_threshold: float = 0.5,
    persist: bool = True,
    verbose: bool = False,
):
    tracker_config = build_tracker_runtime_config(tracker_type)
    if not tracker_config["use_tracking"]:
        return model.predict(frame_bgr, imgsz=inference_size, conf=conf_threshold, verbose=verbose)
    return model.track(
        frame_bgr,
        imgsz=inference_size,
        conf=conf_threshold,
        iou=iou_threshold,
        persist=persist,
        tracker=tracker_config["tracker_config"],
        verbose=verbose,
    )
