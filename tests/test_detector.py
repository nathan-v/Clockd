from __future__ import annotations

import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import cv2
import numpy as np
import pytest

from clockd.services.detector import (
    CodeProjectAIDetector,
    CoralAPIDetector,
    DetectorUnavailableError,
    FallbackDetector,
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
    assert isinstance(det, FallbackDetector)
    assert isinstance(det._primary, CodeProjectAIDetector)


def test_create_detector_roboflow():
    det = create_detector(
        backend="roboflow",
        model_name="yolo11n.pt",
        confidence=0.3,
        roboflow_url="http://localhost:9001",
        roboflow_model_id="yolo11n-640",
    )
    assert isinstance(det, FallbackDetector)
    assert isinstance(det._primary, RoboflowInferenceDetector)


def test_create_detector_localai():
    det = create_detector(
        backend="localai",
        model_name="yolo11n.pt",
        confidence=0.3,
        localai_url="http://localhost:8080",
        localai_model="rfdetr-base",
    )
    assert isinstance(det, FallbackDetector)
    assert isinstance(det._primary, LocalAIDetector)


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
    with pytest.raises(DetectorUnavailableError):
        det.detect(np.zeros((480, 640, 3), dtype=np.uint8))


def test_roboflow_detector_success():
    # Roboflow returns center x/y + width/height format
    response_data = {
        "predictions": [
            {"class": "car", "confidence": 0.91, "x": 200, "y": 300, "width": 100, "height": 80},
            {"class": "bus", "confidence": 0.78, "x": 500, "y": 350, "width": 200, "height": 150},
            {"class": "dog", "confidence": 0.95, "x": 50, "y": 50, "width": 30, "height": 40},
        ]
    }

    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received["path"] = self.path
            received["body"] = json.loads(self.rfile.read(length))
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

    det = RoboflowInferenceDetector(
        url=f"http://127.0.0.1:{port}", model_id="yolo26n-640", confidence=0.3
    )
    detections = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    t.join(timeout=5)
    server.server_close()

    # Inference server >=1.x requires model_id and image in the JSON body
    assert received["path"] == "/infer/object_detection"
    assert received["body"]["model_id"] == "yolo26n-640"
    assert received["body"]["image"]["type"] == "base64"
    assert received["body"]["image"]["value"]

    # car + bus (dog is not in LABEL_TO_COCO)
    assert len(detections) == 2
    assert 2 in detections.class_id  # car
    assert 5 in detections.class_id  # bus
    # Verify center->corner conversion: car x=200,y=300,w=100,h=80 -> [150,260,250,340]
    car_idx = list(detections.class_id).index(2)
    np.testing.assert_allclose(detections.xyxy[car_idx], [150, 260, 250, 340], atol=1)


def test_roboflow_detector_resize_max_px():
    # Prediction in the downscaled image's pixel space (1280x720 -> 640x360)
    response_data = {
        "predictions": [
            {"class": "car", "confidence": 0.9, "x": 100, "y": 150, "width": 50, "height": 40},
        ]
    }
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received["body"] = json.loads(self.rfile.read(length))
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

    det = RoboflowInferenceDetector(
        url=f"http://127.0.0.1:{port}", confidence=0.3, resize_max_px=640
    )
    detections = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    t.join(timeout=5)
    server.server_close()

    # The uploaded frame must have been downscaled to 640x360
    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(received["body"]["image"]["value"]), np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert img.shape[:2] == (360, 640)

    # Boxes come back rescaled to the original 1280x720 space (scale=0.5)
    assert len(detections) == 1
    np.testing.assert_allclose(detections.xyxy[0], [150, 260, 250, 340], atol=1)


def test_roboflow_detector_resize_noop_when_frame_smaller():
    # Frame smaller than resize_max_px is sent as-is (resize path must not
    # crash on small frames; unreachable server then raises)
    det = RoboflowInferenceDetector(
        url="http://127.0.0.1:1", confidence=0.3, resize_max_px=640, timeout=1
    )
    with pytest.raises(DetectorUnavailableError):
        det.detect(np.zeros((480, 640, 3), dtype=np.uint8))


def test_roboflow_detector_unreachable():
    det = RoboflowInferenceDetector(url="http://127.0.0.1:1", confidence=0.3, timeout=1)
    with pytest.raises(DetectorUnavailableError):
        det.detect(np.zeros((480, 640, 3), dtype=np.uint8))


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
    with pytest.raises(DetectorUnavailableError):
        det.detect(np.zeros((480, 640, 3), dtype=np.uint8))


def test_create_detector_coralapi():
    det = create_detector(
        backend="coralapi",
        model_name="yolo11n.pt",
        confidence=0.3,
        coralapi_url="http://localhost:8000",
        coralapi_model="ssd_mobilenet_v2_coco_quant_postprocess_edgetpu",
    )
    assert isinstance(det, FallbackDetector)
    assert isinstance(det._primary, CoralAPIDetector)


def test_coralapi_detector_success():
    # coralapi returns normalized [ymin, xmin, ymax, xmax] boxes
    response_data = {
        "model": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu",
        "results": [
            {"box": [0.25, 0.125, 0.75, 0.375], "index": 2, "label": "car", "score": 0.9},
            {"box": [0.1, 0.5, 0.4, 0.9], "index": 5, "label": "bus", "score": 0.8},
            {"box": [0.0, 0.0, 0.2, 0.2], "index": 17, "label": "dog", "score": 0.95},
            {"box": [0.3, 0.3, 0.6, 0.6], "index": 99, "label": None, "score": 0.7},
        ],
    }
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received["path"] = self.path
            received["body"] = self.rfile.read(length)
            received["content_type"] = self.headers.get("Content-Type", "")
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

    det = CoralAPIDetector(
        url=f"http://127.0.0.1:{port}",
        model="ssd_mobilenet_v2_coco_quant_postprocess_edgetpu",
        confidence=0.4,
    )
    detections = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    t.join(timeout=5)
    server.server_close()

    # model + threshold go in the query string; image is a multipart "file" field
    assert received["path"].startswith("/v1/vision/detect?")
    assert "model=ssd_mobilenet_v2_coco_quant_postprocess_edgetpu" in received["path"]
    assert "threshold=0.4" in received["path"]
    assert "multipart/form-data" in received["content_type"]
    assert b'name="file"' in received["body"]

    # car + bus; dog is not a vehicle, null label skipped
    assert len(detections) == 2
    assert 2 in detections.class_id
    assert 5 in detections.class_id
    # normalized box -> pixel xyxy against the 640x480 frame:
    # car [ymin .25, xmin .125, ymax .75, xmax .375] -> [80, 120, 240, 360]
    car_idx = list(detections.class_id).index(2)
    np.testing.assert_allclose(detections.xyxy[car_idx], [80, 120, 240, 360], atol=1)


def test_coralapi_detector_resize_boxes_stay_in_original_space():
    response_data = {
        "model": "m",
        "results": [{"box": [0.5, 0.5, 1.0, 1.0], "index": 2, "label": "car", "score": 0.9}],
    }
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received["body"] = self.rfile.read(length)
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

    det = CoralAPIDetector(url=f"http://127.0.0.1:{port}", confidence=0.3, resize_max_px=640)
    detections = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    t.join(timeout=5)
    server.server_close()

    # uploaded image was downscaled (multipart jpeg smaller than raw frame)
    start = received["body"].index(b"\r\n\r\n", received["body"].index(b'name="file"')) + 4
    jpeg = received["body"][start : received["body"].rindex(b"\r\n----")]
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (360, 640)

    # normalized boxes scale against the ORIGINAL 1280x720 frame
    np.testing.assert_allclose(detections.xyxy[0], [640, 360, 1280, 720], atol=1)


def test_coralapi_detector_unreachable():
    det = CoralAPIDetector(url="http://127.0.0.1:1", confidence=0.3, timeout=1)
    with pytest.raises(DetectorUnavailableError):
        det.detect(np.zeros((480, 640, 3), dtype=np.uint8))


def test_coralapi_detector_index_fallback_when_no_labels():
    # Zoo models without a labels file return label=null; vehicles must be
    # recovered via the COCO-90 index. Known-but-non-vehicle labels stay skipped.
    response_data = {
        "model": "m",
        "results": [
            {"box": [0.1, 0.1, 0.5, 0.5], "index": 2, "label": None, "score": 0.6},  # car by index
            {
                "box": [0.2, 0.2, 0.6, 0.6],
                "index": 7,
                "label": None,
                "score": 0.5,
            },  # truck by index
            {"box": [0.0, 0.0, 0.9, 0.9], "index": 6, "label": None, "score": 0.5},  # train -> skip
            {
                "box": [0.3, 0.3, 0.7, 0.7],
                "index": 2,
                "label": "dog",
                "score": 0.9,
            },  # labeled non-vehicle -> skip
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

    det = CoralAPIDetector(url=f"http://127.0.0.1:{port}", confidence=0.3)
    detections = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    t.join(timeout=5)
    server.server_close()

    assert len(detections) == 2
    assert 2 in detections.class_id  # car via index fallback
    assert 7 in detections.class_id  # truck via index fallback
