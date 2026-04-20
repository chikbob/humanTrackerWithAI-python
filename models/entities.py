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
