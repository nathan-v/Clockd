from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import numpy as np

from clockd.services.detector import (
    CodeProjectAIDetector,
    LocalAIDetector,
    LocalDetector,
    RoboflowInferenceDetector,
    create_detector,
)


def test_create_detector_local():
    det = create_detector(backend="local", model_name="yolo11n.pt", confidence=0.3)
    assert isinstance(det, LocalDetector)


def test_create_detector_codeproject():
    det = create_detector(
        backend="codeproject_ai",
        model_name="yolo11n.pt",
        confidence=0.4,
        codeproject_url="http://localhost:32168",
    )
    assert isinstance(det, CodeProjectAIDetector)


def test_create_detector_roboflow():
    det = create_detector(
        backend="roboflow",
        model_name="yolo11n.pt",
        confidence=0.3,
        roboflow_url="http://localhost:9001",
        roboflow_model_id="yolo11n-640",
    )
    assert isinstance(det, RoboflowInferenceDetector)


def test_create_detector_localai():
    det = create_detector(
        backend="localai",
        model_name="yolo11n.pt",
        confidence=0.3,
        localai_url="http://localhost:8080",
        localai_model="rfdetr-base",
    )
    assert isinstance(det, LocalAIDetector)


def test_codeproject_detector_success():
    response_data = {
        "success": True,
        "predictions": [
            {
                "label": "car",
                "confidence": 0.92,
                "x_min": 100,
                "y_min": 200,
                "x_max": 300,
                "y_max": 400,
            },
            {
                "label": "truck",
                "confidence": 0.85,
                "x_min": 500,
                "y_min": 200,
                "x_max": 700,
                "y_max": 400,
            },
            {
                "label": "person",
                "confidence": 0.95,
                "x_min": 50,
                "y_min": 50,
                "x_max": 100,
                "y_max": 200,
            },
        ],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            resp = json.dumps(response_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()

    det = CodeProjectAIDetector(url=f"http://127.0.0.1:{port}", confidence=0.3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = det.detect(frame)

    t.join(timeout=5)
    server.server_close()

    # Should have car (class 2) + truck (class 7) + person (class 0)
    assert len(detections) == 3
    assert 2 in detections.class_id  # car
    assert 7 in detections.class_id  # truck
    assert 0 in detections.class_id  # person


def test_codeproject_detector_no_predictions():
    response_data = {"success": True, "predictions": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            resp = json.dumps(response_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()

    det = CodeProjectAIDetector(url=f"http://127.0.0.1:{port}", confidence=0.3)
    detections = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    t.join(timeout=5)
    server.server_close()

    assert len(detections) == 0


def test_codeproject_detector_unreachable():
    det = CodeProjectAIDetector(url="http://127.0.0.1:1", confidence=0.3, timeout=1)
    detections = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(detections) == 0  # should not raise


def test_roboflow_detector_success():
    # Roboflow returns center x/y + width/height format
    response_data = {
        "predictions": [
            {"class": "car", "confidence": 0.91, "x": 200, "y": 300, "width": 100, "height": 80},
            {"class": "bus", "confidence": 0.78, "x": 500, "y": 350, "width": 200, "height": 150},
            {"class": "dog", "confidence": 0.95, "x": 50, "y": 50, "width": 30, "height": 40},
        ]
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            resp = json.dumps(response_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()

    det = RoboflowInferenceDetector(url=f"http://127.0.0.1:{port}", confidence=0.3)
    detections = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    t.join(timeout=5)
    server.server_close()

    # car + bus (dog is not in LABEL_TO_COCO)
    assert len(detections) == 2
    assert 2 in detections.class_id  # car
    assert 5 in detections.class_id  # bus
    # Verify center->corner conversion: car x=200,y=300,w=100,h=80 -> [150,260,250,340]
    car_idx = list(detections.class_id).index(2)
    np.testing.assert_allclose(detections.xyxy[car_idx], [150, 260, 250, 340], atol=1)


def test_roboflow_detector_unreachable():
    det = RoboflowInferenceDetector(url="http://127.0.0.1:1", confidence=0.3, timeout=1)
    detections = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(detections) == 0


def test_localai_detector_success():
    # LocalAI returns x/y (top-left) + width/height format
    response_data = [
        {"class_name": "car", "confidence": 0.88, "x": 100, "y": 200, "width": 150, "height": 100},
        {
            "class_name": "truck",
            "confidence": 0.72,
            "x": 400,
            "y": 250,
            "width": 180,
            "height": 120,
        },
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            resp = json.dumps(response_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = Thread(target=server.handle_request, daemon=True)
    t.start()

    det = LocalAIDetector(url=f"http://127.0.0.1:{port}", confidence=0.3)
    detections = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    t.join(timeout=5)
    server.server_close()

    assert len(detections) == 2
    assert 2 in detections.class_id  # car
    assert 7 in detections.class_id  # truck
    # Verify x,y,w,h -> xyxy: car x=100,y=200,w=150,h=100 -> [100,200,250,300]
    car_idx = list(detections.class_id).index(2)
    np.testing.assert_allclose(detections.xyxy[car_idx], [100, 200, 250, 300], atol=1)


def test_localai_detector_unreachable():
    det = LocalAIDetector(url="http://127.0.0.1:1", confidence=0.3, timeout=1)
    detections = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(detections) == 0
