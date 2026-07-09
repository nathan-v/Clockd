from __future__ import annotations

import base64
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from json import dumps as json_dumps
from json import loads as json_loads
from typing import Optional

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run object detection on a frame and return vehicle detections."""

    def close(self) -> None:
        pass


class LocalDetector(Detector):
    """Detection using a local YOLO model via ultralytics."""

    def __init__(self, model_name: str, confidence: float) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_name)
        self._confidence = confidence

    def detect(self, frame: np.ndarray) -> sv.Detections:
        results = self._model(frame, conf=self._confidence, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)


class CodeProjectAIDetector(Detector):
    """Detection using a remote CodeProject.AI server."""

    def __init__(
        self,
        url: str = "http://localhost:32168",
        confidence: float = 0.3,
        timeout: int = 30,
    ) -> None:
        self._url = url.rstrip("/")
        self._confidence = confidence
        self._timeout = timeout

    def detect(self, frame: np.ndarray) -> sv.Detections:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return sv.Detections.empty()

        # Build multipart form data
        boundary = "----ClockdBoundary"
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(b'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n')
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(buf.tobytes())
        body.extend(f"\r\n--{boundary}\r\n".encode())
        body.extend(b'Content-Disposition: form-data; name="min_confidence"\r\n\r\n')
        body.extend(f"{self._confidence}\r\n".encode())
        body.extend(f"--{boundary}--\r\n".encode())

        url = f"{self._url}/v1/vision/detection"
        req = urllib.request.Request(url, data=bytes(body), method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json_loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("CodeProject.AI detection failed: %s", exc)
            return sv.Detections.empty()

        if not data.get("success") or not data.get("predictions"):
            return sv.Detections.empty()

        boxes = []
        class_ids = []
        confidences = []

        for pred in data["predictions"]:
            label = pred.get("label", "").lower()
            coco_id = LABEL_TO_COCO.get(label)
            if coco_id is None:
                continue
            conf = pred.get("confidence", 0.0)
            x_min = pred.get("x_min", 0)
            y_min = pred.get("y_min", 0)
            x_max = pred.get("x_max", 0)
            y_max = pred.get("y_max", 0)
            boxes.append([x_min, y_min, x_max, y_max])
            class_ids.append(coco_id)
            confidences.append(conf)

        if not boxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
            confidence=np.array(confidences, dtype=np.float32),
        )


# Shared label-to-COCO mapping for remote backends
LABEL_TO_COCO = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "bicycle": 1,
    "person": 0,
}


class RoboflowInferenceDetector(Detector):
    """Detection using a self-hosted Roboflow Inference server."""

    def __init__(
        self,
        url: str = "http://localhost:9001",
        model_id: str = "yolo11n-640",
        confidence: float = 0.3,
        timeout: int = 30,
    ) -> None:
        self._url = url.rstrip("/")
        self._model_id = model_id
        self._confidence = confidence
        self._timeout = timeout

    def detect(self, frame: np.ndarray) -> sv.Detections:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return sv.Detections.empty()

        img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        payload = json_dumps(
            {
                "type": "base64",
                "value": img_b64,
                "confidence": self._confidence,
            }
        ).encode()

        url = f"{self._url}/infer/object_detection"
        # Model ID goes in query param
        url += f"?model_id={self._model_id}"
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json_loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Roboflow Inference detection failed: %s", exc)
            return sv.Detections.empty()

        predictions = data.get("predictions", [])
        if not predictions:
            return sv.Detections.empty()

        boxes = []
        class_ids = []
        confidences = []

        for pred in predictions:
            label = pred.get("class", "").lower()
            coco_id = LABEL_TO_COCO.get(label)
            if coco_id is None:
                continue
            conf = pred.get("confidence", 0.0)
            cx = pred.get("x", 0)
            cy = pred.get("y", 0)
            w = pred.get("width", 0)
            h = pred.get("height", 0)
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            class_ids.append(coco_id)
            confidences.append(conf)

        if not boxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
            confidence=np.array(confidences, dtype=np.float32),
        )


class LocalAIDetector(Detector):
    """Detection using a LocalAI server (RF-DETR model)."""

    def __init__(
        self,
        url: str = "http://localhost:8080",
        model: str = "rfdetr-base",
        confidence: float = 0.3,
        timeout: int = 30,
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._confidence = confidence
        self._timeout = timeout

    def detect(self, frame: np.ndarray) -> sv.Detections:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return sv.Detections.empty()

        img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        payload = json_dumps(
            {
                "model": self._model,
                "image": f"data:image/jpeg;base64,{img_b64}",
                "threshold": self._confidence,
            }
        ).encode()

        url = f"{self._url}/v1/detection"
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json_loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("LocalAI detection failed: %s", exc)
            return sv.Detections.empty()

        detections = data if isinstance(data, list) else data.get("detections", [])
        if not detections:
            return sv.Detections.empty()

        boxes = []
        class_ids = []
        confidences = []

        for det in detections:
            label = det.get("class_name", det.get("label", "")).lower()
            coco_id = LABEL_TO_COCO.get(label)
            if coco_id is None:
                continue
            conf = det.get("confidence", 0.0)
            x = det.get("x", 0)
            y = det.get("y", 0)
            w = det.get("width", 0)
            h = det.get("height", 0)
            boxes.append([x, y, x + w, y + h])
            class_ids.append(coco_id)
            confidences.append(conf)

        if not boxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
            confidence=np.array(confidences, dtype=np.float32),
        )


def create_detector(
    backend: str,
    model_name: str,
    confidence: float,
    codeproject_url: Optional[str] = None,
    codeproject_timeout: int = 30,
    roboflow_url: Optional[str] = None,
    roboflow_model_id: Optional[str] = None,
    roboflow_timeout: int = 30,
    localai_url: Optional[str] = None,
    localai_model: Optional[str] = None,
    localai_timeout: int = 30,
) -> Detector:
    if backend == "codeproject_ai":
        return CodeProjectAIDetector(
            url=codeproject_url or "http://localhost:32168",
            confidence=confidence,
            timeout=codeproject_timeout,
        )
    if backend == "roboflow":
        return RoboflowInferenceDetector(
            url=roboflow_url or "http://localhost:9001",
            model_id=roboflow_model_id or "yolo11n-640",
            confidence=confidence,
            timeout=roboflow_timeout,
        )
    if backend == "localai":
        return LocalAIDetector(
            url=localai_url or "http://localhost:8080",
            model=localai_model or "rfdetr-base",
            confidence=confidence,
            timeout=localai_timeout,
        )
    return LocalDetector(model_name=model_name, confidence=confidence)
