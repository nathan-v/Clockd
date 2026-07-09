"""Tests for Prometheus metrics recording and service-level metrics."""

from __future__ import annotations

from clockd.config import MetricsConfig, PrometheusConfig
from clockd.models import ProcessingResult, VehicleResult
from clockd.services.metrics import MetricsService, _escape_tag


_singleton_metrics = None


def _fresh_metrics():
    """Return a singleton MetricsService with Prometheus enabled (avoids registry duplication)."""
    global _singleton_metrics
    if _singleton_metrics is None:
        cfg = MetricsConfig(prometheus=PrometheusConfig(enabled=True))
        _singleton_metrics = MetricsService(cfg)
    return _singleton_metrics


def _make_result():
    return ProcessingResult(
        camera_id="test_cam",
        video_filename="test.mp4",
        fps=30.0,
        total_frames=300,
        duration_s=10.0,
        unit="mph",
        vehicles=[
            VehicleResult(
                track_id=1,
                speed_avg=35.0,
                speed_min=30.0,
                speed_max=40.0,
                unit="mph",
                mean_detection_confidence=0.85,
                first_seen_frame=10,
                last_seen_frame=200,
                first_seen_timestamp_s=0.33,
                last_seen_timestamp_s=6.67,
                num_detections=150,
            ),
        ],
        processing_time_s=5.0,
    )


def test_prometheus_record():
    svc = _fresh_metrics()
    svc.record(_make_result())
    # Verify counters incremented
    prom = svc.prometheus
    assert prom.vehicles_total._metrics


def test_record_request():
    svc = _fresh_metrics()
    svc.record_request("GET", "/health", 200, 0.05)
    assert svc.prometheus.http_requests_total._metrics


def test_record_job_lifecycle():
    svc = _fresh_metrics()
    svc.record_job_submitted()
    svc.record_job_finished()


def test_record_upload_size():
    svc = _fresh_metrics()
    svc.record_upload_size(1024 * 1024)


def test_escape_tag():
    assert _escape_tag("simple") == "simple"
    assert _escape_tag("has space") == r"has\ space"
    assert _escape_tag("has,comma") == r"has\,comma"
    assert _escape_tag("has=equals") == r"has\=equals"
    assert _escape_tag("a\\b") == "a\\\\b"


def test_validate_url_rejects_file():
    svc = _fresh_metrics()
    import pytest

    with pytest.raises(ValueError, match="http"):
        svc._validate_url("file:///etc/passwd")


def test_validate_url_accepts_http():
    svc = _fresh_metrics()
    svc._validate_url("http://localhost:8086")
    svc._validate_url("https://influx.example.com")
