from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from typing import Optional

from clockd.config import MetricsConfig
from clockd.models import ProcessingResult

logger = logging.getLogger(__name__)


def _escape_tag(value: str) -> str:
    """Escape special characters for InfluxDB line protocol tag values."""
    return value.replace("\\", "\\\\").replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


class MetricsService:
    def __init__(self, config: MetricsConfig) -> None:
        self._config = config
        self._prom: Optional[_PrometheusMetrics] = None

        if config.prometheus.enabled:
            self._prom = _PrometheusMetrics()

    @property
    def prometheus(self) -> Optional[_PrometheusMetrics]:
        return self._prom

    def record(self, result: ProcessingResult) -> None:
        if self._prom:
            self._prom.record(result)
        if self._config.influxdb_v1.enabled:
            self._write_influxdb_v1(result)
        if self._config.influxdb_v2.enabled:
            self._write_influxdb_v2(result)

    def record_request(self, method: str, path: str, status: int, duration: float) -> None:
        if self._prom:
            self._prom.http_requests_total.labels(method=method, path=path, status=status).inc()
            self._prom.http_request_duration.labels(method=method, path=path).observe(duration)
        self._write_influxdb_service_point(
            "http_request",
            {"method": method, "path": path, "status": str(status)},
            {"duration_s": duration, "count": 1},
        )

    def record_job_submitted(self) -> None:
        if self._prom:
            self._prom.jobs_submitted_total.inc()
            self._prom.jobs_active.inc()

    def record_job_finished(self) -> None:
        if self._prom:
            self._prom.jobs_active.dec()

    def record_upload_size(self, size_bytes: int) -> None:
        if self._prom:
            self._prom.upload_bytes.observe(size_bytes)

    def _write_influxdb_service_point(
        self, measurement: str, tags: dict[str, str], fields: dict[str, object]
    ) -> None:
        if not self._config.influxdb_v1.enabled and not self._config.influxdb_v2.enabled:
            return
        ts_ns = int(time.time() * 1e9)
        tag_str = ",".join(f"{k}={_escape_tag(v)}" for k, v in tags.items())
        field_parts = []
        for k, v in fields.items():
            if isinstance(v, int):
                field_parts.append(f"{k}={v}i")
            else:
                field_parts.append(f"{k}={v}")
        field_str = ",".join(field_parts)
        line = (
            f"{measurement},{tag_str} {field_str} {ts_ns}"
            if tag_str
            else f"{measurement} {field_str} {ts_ns}"
        )

        if self._config.influxdb_v1.enabled:
            self._write_influxdb_line(line, version=1)
        if self._config.influxdb_v2.enabled:
            self._write_influxdb_line(line, version=2)

    def _build_line_protocol(self, result: ProcessingResult, measurement: str) -> str:
        lines = []
        ts_ns = int(time.time() * 1e9)
        # what actually ran, so dashboards can split by model and CPU-vs-GPU
        # backend (and alert on fallback frequency)
        detect_tags = (
            f"backend={_escape_tag(result.detection_backend or 'unknown')}"
            f",model={_escape_tag(result.detection_model or 'unknown')}"
            f",fallback={str(result.detection_fallback).lower()}"
        )
        for v in result.vehicles:
            cam_id = _escape_tag(result.camera_id)
            tags = (
                f"{measurement},camera_id={cam_id},track_id={v.track_id},"
                f"unit={v.unit},{detect_tags}"
            )
            fields = (
                f"speed_avg={v.speed_avg},"
                f"speed_min={v.speed_min},"
                f"speed_max={v.speed_max},"
                f"detection_confidence={v.mean_detection_confidence},"
                f"num_detections={v.num_detections}i"
            )
            lines.append(f"{tags} {fields} {ts_ns}")
        # Processing summary point
        tags = f"processing_summary,camera_id={_escape_tag(result.camera_id)},{detect_tags}"
        fields = (
            f"vehicle_count={len(result.vehicles)}i,"
            f"processing_time_s={result.processing_time_s},"
            f"total_frames={result.total_frames}i,"
            f"duration_s={result.duration_s}"
        )
        lines.append(f"{tags} {fields} {ts_ns}")
        return "\n".join(lines)

    def _write_influxdb_v1(self, result: ProcessingResult) -> None:
        cfg = self._config.influxdb_v1
        body = self._build_line_protocol(result, cfg.measurement)
        self._write_influxdb_line(body, version=1)

    def _write_influxdb_v2(self, result: ProcessingResult) -> None:
        cfg = self._config.influxdb_v2
        body = self._build_line_protocol(result, cfg.measurement)
        self._write_influxdb_line(body, version=2)

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"InfluxDB URL must use http or https, got: {parsed.scheme}")

    def _write_influxdb_line(self, body: str, version: int) -> None:
        if version == 1:
            cfg = self._config.influxdb_v1
            self._validate_url(cfg.url)
            params = urllib.parse.urlencode({"db": cfg.database})
            url = f"{cfg.url.rstrip('/')}/write?{params}"
            req = urllib.request.Request(url, data=body.encode(), method="POST")
            req.add_header("Content-Type", "text/plain")
            if cfg.username:
                creds = b64encode(f"{cfg.username}:{cfg.password}".encode()).decode()
                req.add_header("Authorization", f"Basic {creds}")
            label = "v1"
        else:
            cfg = self._config.influxdb_v2
            self._validate_url(cfg.url)
            params = urllib.parse.urlencode(
                {"org": cfg.org, "bucket": cfg.bucket, "precision": "ns"}
            )
            url = f"{cfg.url.rstrip('/')}/api/v2/write?{params}"
            req = urllib.request.Request(url, data=body.encode(), method="POST")
            req.add_header("Content-Type", "text/plain")
            req.add_header("Authorization", f"Token {cfg.token}")
            label = "v2"

        try:
            with urllib.request.urlopen(req, timeout=5):
                pass
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("InfluxDB %s write failed: %s", label, exc)


class _PrometheusMetrics:
    def __init__(self) -> None:
        from prometheus_client import Counter, Gauge, Histogram

        # -- Detection metrics --
        self.speed_histogram = Histogram(
            "clockd_vehicle_speed",
            "Vehicle speed observations",
            labelnames=["camera_id", "unit"],
            buckets=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120],
        )
        self.confidence_histogram = Histogram(
            "clockd_vehicle_detection_confidence",
            "Mean detection confidence per tracked vehicle",
            labelnames=["camera_id"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        )
        self.vehicles_total = Counter(
            "clockd_vehicles_detected_total",
            "Total vehicles detected",
            labelnames=["camera_id"],
        )
        self.processing_seconds = Histogram(
            "clockd_processing_seconds",
            "Video processing duration",
            labelnames=["camera_id"],
        )

        # -- Service metrics --
        self.http_requests_total = Counter(
            "clockd_http_requests_total",
            "Total HTTP requests",
            labelnames=["method", "path", "status"],
        )
        self.http_request_duration = Histogram(
            "clockd_http_request_duration_seconds",
            "HTTP request latency",
            labelnames=["method", "path"],
        )
        self.http_requests_in_progress = Gauge(
            "clockd_http_requests_in_progress",
            "HTTP requests currently being processed",
            labelnames=["method"],
        )
        self.jobs_submitted_total = Counter(
            "clockd_jobs_submitted_total",
            "Total async jobs submitted",
        )
        self.jobs_active = Gauge(
            "clockd_jobs_active",
            "Currently running async jobs",
        )
        self.upload_bytes = Histogram(
            "clockd_upload_bytes",
            "Uploaded file sizes in bytes",
            buckets=[1e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8],
        )

    def record(self, result: ProcessingResult) -> None:
        cam = result.camera_id
        self.processing_seconds.labels(camera_id=cam).observe(result.processing_time_s)
        self.vehicles_total.labels(camera_id=cam).inc(len(result.vehicles))
        for v in result.vehicles:
            self.speed_histogram.labels(camera_id=cam, unit=v.unit).observe(v.speed_avg)
            self.confidence_histogram.labels(camera_id=cam).observe(v.mean_detection_confidence)
