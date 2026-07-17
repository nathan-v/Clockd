from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from clockd.config import InfluxDBv1Config, InfluxDBv2Config, MetricsConfig
from clockd.models import ProcessingResult, VehicleResult
from clockd.services.metrics import MetricsService


def _make_result() -> ProcessingResult:
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


def test_metrics_service_disabled():
    svc = MetricsService(MetricsConfig())
    # Should not raise even with everything disabled
    svc.record(_make_result())


def test_line_protocol_format():
    svc = MetricsService(MetricsConfig())
    result = _make_result()
    body = svc._build_line_protocol(result, "vehicle_speed")
    lines = body.strip().split("\n")
    assert len(lines) == 2  # 1 vehicle + 1 summary

    # Vehicle line
    assert "vehicle_speed,camera_id=test_cam,track_id=1,unit=mph" in lines[0]
    assert "speed_avg=35.0" in lines[0]
    assert "speed_min=30.0" in lines[0]
    assert "speed_max=40.0" in lines[0]
    assert "detection_confidence=0.85" in lines[0]
    assert "num_detections=150i" in lines[0]

    # Summary line
    assert "processing_summary,camera_id=test_cam" in lines[1]
    assert "vehicle_count=1i" in lines[1]


def test_influxdb_v1_write():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received.append(
                {
                    "path": self.path,
                    "body": self.rfile.read(length).decode(),
                    "auth": self.headers.get("Authorization"),
                }
            )
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()

    cfg = MetricsConfig(
        influxdb_v1=InfluxDBv1Config(
            enabled=True,
            url=f"http://127.0.0.1:{port}",
            database="testdb",
        )
    )
    svc = MetricsService(cfg)
    svc.record(_make_result())
    t.join(timeout=5)
    server.server_close()

    assert len(received) == 1
    assert "/write?db=testdb" in received[0]["path"]
    assert "speed_avg=35.0" in received[0]["body"]


def test_influxdb_v2_write():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received.append(
                {
                    "path": self.path,
                    "body": self.rfile.read(length).decode(),
                    "auth": self.headers.get("Authorization"),
                }
            )
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()

    cfg = MetricsConfig(
        influxdb_v2=InfluxDBv2Config(
            enabled=True,
            url=f"http://127.0.0.1:{port}",
            token="test-token",
            org="myorg",
            bucket="mybucket",
        )
    )
    svc = MetricsService(cfg)
    svc.record(_make_result())
    t.join(timeout=5)
    server.server_close()

    assert len(received) == 1
    assert "org=myorg" in received[0]["path"]
    assert "bucket=mybucket" in received[0]["path"]
    assert received[0]["auth"] == "Token test-token"
    assert "speed_avg=35.0" in received[0]["body"]


def test_influxdb_unreachable_does_not_raise():
    cfg = MetricsConfig(
        influxdb_v1=InfluxDBv1Config(
            enabled=True,
            url="http://127.0.0.1:1",  # nothing listening
        )
    )
    svc = MetricsService(cfg)
    # Should log warning but not raise
    svc.record(_make_result())


def test_line_protocol_detection_tags():
    svc = MetricsService(MetricsConfig())
    result = _make_result()
    result.detection_backend = "roboflow"
    result.detection_model = "yolo26l-640"
    result.detection_fallback = False
    lines = svc._build_line_protocol(result, "vehicle_speed").split("\n")

    for line in lines:
        assert "backend=roboflow" in line
        assert "model=yolo26l-640" in line
        assert "fallback=false" in line

    # fallback run is tagged with what actually performed detection
    result.detection_backend = "local"
    result.detection_model = "yolo26n.pt"
    result.detection_fallback = True
    line = svc._build_line_protocol(result, "vehicle_speed").split("\n")[0]
    assert "backend=local" in line
    assert "model=yolo26n.pt" in line
    assert "fallback=true" in line
