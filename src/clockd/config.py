from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import re

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class CalibrationConfig(BaseModel):
    source_points: list[list[float]]
    target_width_m: float
    target_height_m: float

    @field_validator("source_points")
    @classmethod
    def validate_source_points(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 4:
            raise ValueError("source_points must contain exactly 4 points")
        for pt in v:
            if len(pt) != 2:
                raise ValueError("Each source point must be [x, y]")
        return v

    @field_validator("target_width_m", "target_height_m")
    @classmethod
    def validate_target_dims(cls, v: float) -> float:
        if v <= 0 or v > 1000:
            raise ValueError("Target dimension must be between 0 and 1000 meters")
        return v


class SpeedRange(BaseModel):
    min_mph: float = 3.0
    max_mph: float = 150.0

    @field_validator("min_mph", "max_mph")
    @classmethod
    def validate_speed(cls, v: float) -> float:
        if v < 0 or v > 500:
            raise ValueError("Speed must be between 0 and 500 mph")
        return v


ALLOWED_MODELS = {
    # YOLOv8
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    # YOLOv9
    "yolov9t.pt",
    "yolov9s.pt",
    "yolov9m.pt",
    "yolov9c.pt",
    "yolov9e.pt",
    # YOLOv10
    "yolov10n.pt",
    "yolov10s.pt",
    "yolov10m.pt",
    "yolov10l.pt",
    "yolov10x.pt",
    # YOLO11
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    # YOLOv12
    "yolov12n.pt",
    "yolov12s.pt",
    "yolov12m.pt",
    "yolov12l.pt",
    "yolov12x.pt",
    # YOLO26
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
}


class CameraConfig(BaseModel):
    camera_id: str
    description: str = ""
    calibration: CalibrationConfig
    resolution: Optional[list[int]] = None  # [width, height] expected video resolution
    roi_polygon: Optional[list[list[float]]] = None
    model_override: Optional[str] = None
    confidence_override: Optional[float] = None
    min_detections: int = 10
    smoothing_window: int = 5
    speed_range: SpeedRange = SpeedRange()
    speed_calibration_factor: float = 1.0  # multiply all speeds by this value

    @field_validator("confidence_override")
    @classmethod
    def validate_confidence(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v <= 0 or v > 1):
            raise ValueError("confidence_override must be between 0 and 1")
        return v

    @field_validator("min_detections")
    @classmethod
    def validate_min_detections(cls, v: int) -> int:
        if v < 2 or v > 10000:
            raise ValueError("min_detections must be between 2 and 10000")
        return v

    @field_validator("smoothing_window")
    @classmethod
    def validate_smoothing_window(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("smoothing_window must be between 1 and 100")
        return v

    @field_validator("camera_id")
    @classmethod
    def validate_camera_id(cls, v: str) -> str:
        if not SAFE_ID_RE.match(v):
            raise ValueError(
                "camera_id must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    @field_validator("model_override")
    @classmethod
    def validate_model_override(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ALLOWED_MODELS:
            raise ValueError(f"model_override must be one of: {', '.join(sorted(ALLOWED_MODELS))}")
        return v


class CodeProjectAIConfig(BaseModel):
    url: str = "http://localhost:32168"
    timeout: int = 30


class RoboflowInferenceConfig(BaseModel):
    url: str = "http://localhost:9001"
    model_id: str = "yolo11n-640"
    timeout: int = 30
    # Downscale frames so their longest side is at most this many pixels before
    # sending to the inference server. The server resizes to the model input size
    # anyway, so this trades local resize CPU for much smaller encode/transfer
    # payloads. None sends full-resolution frames (pushes all work to the server).
    resize_max_px: Optional[int] = Field(default=None, ge=64, le=8192)


class LocalAIConfig(BaseModel):
    url: str = "http://localhost:8080"
    model: str = "rfdetr-base"
    timeout: int = 30


class CoralAPIConfig(BaseModel):
    """coralapi Edge TPU inference server (https://github.com/nathan-v/coralapi)."""

    url: str = "http://localhost:8000"
    model: str = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu"
    timeout: int = 30
    # Same semantics as roboflow.resize_max_px: downscale before upload.
    # Edge TPU SSD models run at 300x300, so full-res frames are pure
    # transfer overhead; boxes come back normalized, so no accuracy cost.
    resize_max_px: Optional[int] = Field(default=None, ge=64, le=8192)


class UnifiProtectConfig(BaseModel):
    host: str = ""
    username: str = ""
    password: str = ""
    verify_ssl: bool = False
    poll_interval_s: int = 30
    lookback_s: int = 120
    event_end_timeout_s: int = 300
    smart_detect_types: list[str] = ["vehicle"]


class EventSourceConfig(BaseModel):
    enabled: bool = False
    camera_map: dict[str, str] = {}  # NVR camera ID -> clockd camera_id
    unit: str = "mph"


class UnifiEventSourceConfig(EventSourceConfig):
    unifi: UnifiProtectConfig = UnifiProtectConfig()


class PrometheusConfig(BaseModel):
    enabled: bool = False
    path: str = "/metrics"


class InfluxDBv1Config(BaseModel):
    enabled: bool = False
    url: str = "http://localhost:8086"
    database: str = "clockd"
    username: str = ""
    password: str = ""
    measurement: str = "vehicle_speed"


class InfluxDBv2Config(BaseModel):
    enabled: bool = False
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = ""
    bucket: str = "clockd"
    measurement: str = "vehicle_speed"


class MetricsConfig(BaseModel):
    prometheus: PrometheusConfig = PrometheusConfig()
    influxdb_v1: InfluxDBv1Config = InfluxDBv1Config()
    influxdb_v2: InfluxDBv2Config = InfluxDBv2Config()


class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLOCKD_", env_nested_delimiter="__")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Env vars beat YAML (passed as init kwargs) so secrets can be injected
        # at deploy time (e.g. from Kubernetes Secrets) without living in the
        # config file. Nested fields use "__", e.g.
        # CLOCKD_EVENT_SOURCES__HOME_NVR__UNIFI__PASSWORD.
        return (
            _ignore_k8s_service_links(env_settings),
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )

    host: str = "0.0.0.0"
    port: int = 8000
    verbose: bool = False  # enable detailed logging of processing results
    detection_backend: str = "local"  # "local", "roboflow", "localai", "codeproject_ai", "coralapi"
    # "local": if a remote detection_backend is unreachable, fall back to
    # local CPU inference (using `model`) for the rest of the job. "none":
    # unreachable backends yield empty detections (logged) as before.
    detection_fallback: str = "none"

    @field_validator("detection_fallback")
    @classmethod
    def _validate_detection_fallback(cls, v: str) -> str:
        if v not in ("none", "local"):
            raise ValueError("detection_fallback must be 'none' or 'local'")
        return v

    model: str = "yolo26n.pt"  # validated at startup, must be in ALLOWED_MODELS

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in ALLOWED_MODELS:
            raise ValueError(f"model must be one of: {', '.join(sorted(ALLOWED_MODELS))}")
        return v

    confidence: float = 0.3
    default_unit: str = "mph"
    max_upload_mb: int = 200
    max_workers: int = 2
    job_ttl_seconds: int = 3600
    cameras_dir: str = "configs/cameras"
    max_cameras: int = 50  # max camera configs the API will create
    upload_dir: str = "/tmp/clockd_uploads"
    codeproject_ai: CodeProjectAIConfig = CodeProjectAIConfig()
    roboflow: RoboflowInferenceConfig = RoboflowInferenceConfig()
    localai: LocalAIConfig = LocalAIConfig()
    coralapi: CoralAPIConfig = CoralAPIConfig()
    metrics: MetricsConfig = MetricsConfig()
    event_sources: dict[str, UnifiEventSourceConfig] = {}


def _ignore_k8s_service_links(env_source):
    """Wrap the env settings source to drop Kubernetes service-link values.

    A Service named "clockd" makes Kubernetes inject legacy Docker-link vars
    like CLOCKD_PORT=tcp://<cluster-ip>:8000 into every pod in the namespace,
    which collides with the CLOCKD_ env prefix and would fail port validation
    at startup. Prefer enableServiceLinks: false in the pod spec; this guard
    keeps startup working either way.
    """

    def _filtered() -> dict:
        values = env_source()
        port = values.get("port")
        if isinstance(port, str) and port.startswith("tcp://"):
            values.pop("port")
        return values

    return _filtered


def load_server_config(path: str = "configs/server.yaml") -> ServerConfig:
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return ServerConfig(**data)
    return ServerConfig()


def load_cameras(cameras_dir: str) -> dict[str, CameraConfig]:
    cameras: dict[str, CameraConfig] = {}
    cam_path = Path(cameras_dir)
    if not cam_path.exists():
        return cameras
    for f in sorted(cam_path.glob("*.yaml")):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        if data and "camera_id" in data:
            cam = CameraConfig(**data)
            cameras[cam.camera_id] = cam
    return cameras


def _safe_path(cameras_dir: str, filename: str) -> Path:
    cam_path = Path(cameras_dir).resolve()
    dest = (cam_path / filename).resolve()
    if not str(dest).startswith(str(cam_path) + os.sep) and dest != cam_path:
        raise ValueError("Invalid path: traversal detected")
    return dest


def save_camera(cameras_dir: str, camera: CameraConfig) -> None:
    cam_path = Path(cameras_dir)
    cam_path.mkdir(parents=True, exist_ok=True)
    dest = _safe_path(cameras_dir, f"{camera.camera_id}.yaml")
    with open(dest, "w") as f:
        yaml.dump(camera.model_dump(), f, default_flow_style=False, sort_keys=False)


def delete_camera_file(cameras_dir: str, camera_id: str) -> bool:
    dest = _safe_path(cameras_dir, f"{camera_id}.yaml")
    if dest.exists():
        dest.unlink()
        return True
    return False
