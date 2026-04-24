from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoSource:
    id: Optional[int]
    name: str
    source_type: str
    source_url: str
    location: str
    is_active: bool
    last_seen: Optional[float]
    description: str


@dataclass
class WorkerStatus:
    source_id: int
    status: str
    is_connected: bool
    last_heartbeat: Optional[float]
    last_frame_at: Optional[float]
    fps: float
    reconnect_count: int
    last_error: str
    last_snapshot_path: str


@dataclass
class SystemSetting:
    key: str
    value: str
    updated_at: Optional[float]


@dataclass
class ExperimentRun:
    id: Optional[int]
    run_key: str
    scenario_name: str
    source_path: str
    notes: str
    created_at: float
    completed_at: Optional[float]
    status: str


@dataclass
class BenchmarkResult:
    id: Optional[int]
    run_id: int
    model_name: str
    tracker_type: str
    frame_limit: int
    warmup_frames: int
    frames_processed: int
    avg_latency_ms: float
    p95_latency_ms: float
    avg_fps: float
    avg_detections_per_frame: float
    tracked_frame_ratio: float
    detection_count_total: int
    metadata_json: str
    created_at: float
