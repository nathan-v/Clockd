from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SpeedUnit(str, Enum):
    mph = "mph"
    kmh = "kmh"


class VehicleResult(BaseModel):
    track_id: int
    speed_avg: float
    speed_min: float
    speed_max: float
    unit: str
    mean_detection_confidence: float
    first_seen_frame: int
    last_seen_frame: int
    first_seen_timestamp_s: float
    last_seen_timestamp_s: float
    num_detections: int


class ProcessingResult(BaseModel):
    camera_id: str
    video_filename: str
    fps: float
    total_frames: int
    duration_s: float
    unit: str
    vehicles: list[VehicleResult]
    vehicles_filtered: int = 0  # tracks excluded by min_detections or speed_range
    processing_time_s: float
    warnings: list[str] = []


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    result: Optional[ProcessingResult] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
