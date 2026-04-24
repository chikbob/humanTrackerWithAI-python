"""Benchmark runner for detector/tracker comparison experiments."""

from __future__ import annotations

import statistics
import time
import uuid
from pathlib import Path

from config.app_config import build_ai_runtime_settings, build_tracker_runtime_config
from core.tracking import run_detection_with_optional_tracking
from db.repository import complete_experiment_run, create_experiment_run, insert_benchmark_result
from research.scenarios import build_named_scenario


def _read_video_frames(source_path: str, *, frame_limit: int):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv_not_installed") from exc

    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        raise RuntimeError(f"video_open_failed:{source_path}")
    frames = []
    try:
        while len(frames) < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"no_frames_read:{source_path}")
    return frames


def _load_benchmark_model(model_name: str):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("ultralytics_not_installed") from exc
    return YOLO(model_name)


def _summarize_case(*, model_name: str, tracker_type: str, frame_limit: int, warmup_frames: int, frames, model, quality_profile: str = "balanced"):
    latencies_ms = []
    detection_counts = []
    tracked_frames = 0
    total_processed = 0
    tracker_runtime = build_tracker_runtime_config(tracker_type)
    ai_runtime = build_ai_runtime_settings({"ai_quality_profile": quality_profile}, {"tracker_type_override": tracker_type})

    for index, frame in enumerate(frames):
        started_at = time.perf_counter()
        results = run_detection_with_optional_tracking(
            model,
            frame,
            tracker_type=tracker_type,
            inference_size=ai_runtime["inference_size"],
            conf_threshold=ai_runtime["confidence_threshold"],
            iou_threshold=ai_runtime["tracking_iou_threshold"],
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        boxes_count = 0
        has_track_ids = False
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", None)
            boxes_count += len(xyxy) if xyxy is not None else 0
            has_track_ids = has_track_ids or getattr(boxes, "id", None) is not None
        if index < warmup_frames:
            continue
        latencies_ms.append(elapsed_ms)
        detection_counts.append(boxes_count)
        total_processed += 1
        if has_track_ids:
            tracked_frames += 1

    if not latencies_ms:
        raise RuntimeError("not_enough_frames_after_warmup")

    avg_latency_ms = statistics.fmean(latencies_ms)
    p95_latency_ms = sorted(latencies_ms)[max(0, min(len(latencies_ms) - 1, int(len(latencies_ms) * 0.95) - 1))]
    avg_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    avg_detections = statistics.fmean(detection_counts) if detection_counts else 0.0
    tracked_frame_ratio = tracked_frames / max(1, total_processed) if tracker_runtime["use_tracking"] else 0.0

    return {
        "model_name": model_name,
        "tracker_type": tracker_type,
        "frame_limit": frame_limit,
        "warmup_frames": warmup_frames,
        "frames_processed": total_processed,
        "avg_latency_ms": round(avg_latency_ms, 3),
        "p95_latency_ms": round(p95_latency_ms, 3),
        "avg_fps": round(avg_fps, 3),
        "avg_detections_per_frame": round(avg_detections, 3),
        "tracked_frame_ratio": round(tracked_frame_ratio, 4),
        "detection_count_total": int(sum(detection_counts)),
        "metadata": {
            "tracker_label": tracker_runtime["tracker_label"],
            "frames_sampled": len(frames),
            "quality_profile": quality_profile,
            "confidence_threshold": ai_runtime["confidence_threshold"],
            "inference_size": ai_runtime["inference_size"],
            "tracking_iou_threshold": ai_runtime["tracking_iou_threshold"],
            "incident_score_threshold": ai_runtime["incident_score_threshold"],
        },
    }


def run_benchmark_scenario(
    *,
    scenario_name: str,
    source_path: str,
    model_name: str | None = None,
    frame_limit: int = 120,
    warmup_frames: int = 10,
    notes: str = "",
    quality_profile: str = "balanced",
):
    scenario = build_named_scenario(
        scenario_name,
        model_name=model_name,
        frame_limit=frame_limit,
        warmup_frames=warmup_frames,
    )
    run_key = f"{scenario['name']}-{uuid.uuid4().hex[:8]}"
    run_id = create_experiment_run(
        run_key=run_key,
        scenario_name=scenario["name"],
        source_path=str(Path(source_path)),
        notes=notes or scenario["description"],
    )

    frames = _read_video_frames(source_path, frame_limit=max(case["frame_limit"] for case in scenario["cases"]))
    results = []
    try:
        for case in scenario["cases"]:
            model = _load_benchmark_model(case["model_name"])
            summary = _summarize_case(
                model_name=case["model_name"],
                tracker_type=case["tracker_type"],
                frame_limit=case["frame_limit"],
                warmup_frames=case["warmup_frames"],
                frames=frames[: case["frame_limit"]],
                model=model,
                quality_profile=quality_profile,
            )
            insert_benchmark_result(
                run_id=run_id,
                model_name=summary["model_name"],
                tracker_type=summary["tracker_type"],
                frame_limit=summary["frame_limit"],
                warmup_frames=summary["warmup_frames"],
                frames_processed=summary["frames_processed"],
                avg_latency_ms=summary["avg_latency_ms"],
                p95_latency_ms=summary["p95_latency_ms"],
                avg_fps=summary["avg_fps"],
                avg_detections_per_frame=summary["avg_detections_per_frame"],
                tracked_frame_ratio=summary["tracked_frame_ratio"],
                detection_count_total=summary["detection_count_total"],
                metadata=summary["metadata"],
            )
            results.append({"scenario_name": scenario["name"], "quality_profile": quality_profile, **summary})
    except Exception:
        complete_experiment_run(run_id=run_id, status="failed")
        raise

    complete_experiment_run(run_id=run_id, status="completed")
    return {
        "run_id": run_id,
        "run_key": run_key,
        "scenario": scenario,
        "results": results,
    }
