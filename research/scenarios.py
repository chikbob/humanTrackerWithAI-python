"""Predefined benchmark scenarios for detector/tracker comparisons."""

from __future__ import annotations

from config.app_config import TRACKER_OPTIONS
from ui.sidebar import MODEL_OPTIONS


def build_detector_comparison_scenario(*, frame_limit: int, warmup_frames: int) -> dict:
    return {
        "name": "detector_comparison",
        "description": "Сравнение моделей детекции при фиксированном baseline-трекере.",
        "cases": [
            {
                "model_name": model_name,
                "tracker_type": "bytetrack",
                "frame_limit": frame_limit,
                "warmup_frames": warmup_frames,
            }
            for model_name in MODEL_OPTIONS
        ],
    }


def build_tracker_comparison_scenario(*, model_name: str, frame_limit: int, warmup_frames: int) -> dict:
    tracker_keys = [key for key in TRACKER_OPTIONS if key in {"bytetrack", "botsort", "detect_only"}]
    return {
        "name": "tracker_comparison",
        "description": "Сравнение трекеров и режима detect-only для одной модели.",
        "cases": [
            {
                "model_name": model_name,
                "tracker_type": tracker_type,
                "frame_limit": frame_limit,
                "warmup_frames": warmup_frames,
            }
            for tracker_type in tracker_keys
        ],
    }


def build_latency_profile_scenario(*, model_name: str, frame_limit: int, warmup_frames: int) -> dict:
    return {
        "name": "latency_profile",
        "description": "Профилирование latency/FPS для одной конфигурации модели и трекера.",
        "cases": [
            {
                "model_name": model_name,
                "tracker_type": "bytetrack",
                "frame_limit": frame_limit,
                "warmup_frames": warmup_frames,
            },
            {
                "model_name": model_name,
                "tracker_type": "detect_only",
                "frame_limit": frame_limit,
                "warmup_frames": warmup_frames,
            },
        ],
    }


def build_named_scenario(name: str, *, model_name: str | None = None, frame_limit: int = 120, warmup_frames: int = 10) -> dict:
    scenario_name = (name or "detector_comparison").strip().lower()
    if scenario_name == "detector_comparison":
        return build_detector_comparison_scenario(frame_limit=frame_limit, warmup_frames=warmup_frames)
    if scenario_name == "tracker_comparison":
        return build_tracker_comparison_scenario(model_name=model_name or "yolov8s.pt", frame_limit=frame_limit, warmup_frames=warmup_frames)
    if scenario_name == "latency_profile":
        return build_latency_profile_scenario(model_name=model_name or "yolov8s.pt", frame_limit=frame_limit, warmup_frames=warmup_frames)
    raise ValueError(f"unknown_scenario:{name}")
