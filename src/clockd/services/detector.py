from __future__ import annotations

import base64
import logging
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from json import dumps as json_dumps
from json import loads as json_loads
from typing import Callable, Optional

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


class DetectorUnavailableError(RuntimeError):
    """The remote detection backend could not be reached."""


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
            raise DetectorUnavailableError(f"CodeProject.AI detection failed: {exc}") from exc

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
        resize_max_px: Optional[int] = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._model_id = model_id
        self._confidence = confidence
        self._timeout = timeout
        self._resize_max_px = resize_max_px

    def detect(self, frame: np.ndarray) -> sv.Detections:
        # Optionally downscale before encoding; predictions come back in the
        # sent image's pixel space, so scale them back up afterwards.
        scale = 1.0
        if self._resize_max_px:
            long_side = max(frame.shape[0], frame.shape[1])
            if long_side > self._resize_max_px:
                scale = self._resize_max_px / long_side
                frame = cv2.resize(
                    frame,
                    (round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return sv.Detections.empty()

        img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        payload = json_dumps(
            {
                "model_id": self._model_id,
                "image": {"type": "base64", "value": img_b64},
                "confidence": self._confidence,
            }
        ).encode()

        url = f"{self._url}/infer/object_detection"
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json_loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            raise DetectorUnavailableError(f"Roboflow Inference detection failed: {exc}") from exc

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
            cx = pred.get("x", 0) / scale
            cy = pred.get("y", 0) / scale
            w = pred.get("width", 0) / scale
            h = pred.get("height", 0) / scale
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
            raise DetectorUnavailableError(f"LocalAI detection failed: {exc}") from exc

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


# Vehicle classes in the 0-indexed COCO-90 scheme used by Coral zoo detection
# models (car, motorcycle, bus, truck) — values equal our COCO-80 ids.
CORAL_INDEX_TO_COCO = {2: 2, 3: 3, 5: 5, 7: 7}


class CoralAPIDetector(Detector):
    """Detection using a coralapi Edge TPU inference server.

    https://github.com/nathan-v/coralapi — POST /v1/vision/detect with a
    multipart image; detections come back with normalized [ymin, xmin,
    ymax, xmax] boxes (TFLite SSD convention) scaled here against the
    original frame, so optional client-side downscaling needs no box
    rescaling bookkeeping.
    """

    def __init__(
        self,
        url: str = "http://localhost:8000",
        model: str = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu",
        confidence: float = 0.3,
        timeout: int = 30,
        resize_max_px: Optional[int] = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._confidence = confidence
        self._timeout = timeout
        self._resize_max_px = resize_max_px

    def detect(self, frame: np.ndarray) -> sv.Detections:
        orig_h, orig_w = frame.shape[0], frame.shape[1]
        if self._resize_max_px and max(orig_h, orig_w) > self._resize_max_px:
            scale = self._resize_max_px / max(orig_h, orig_w)
            frame = cv2.resize(
                frame,
                (round(orig_w * scale), round(orig_h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return sv.Detections.empty()

        boundary = "----ClockdBoundary"
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(b'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n')
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(buf.tobytes())
        body.extend(f"\r\n--{boundary}--\r\n".encode())

        params = urllib.parse.urlencode({"model": self._model, "threshold": self._confidence})
        url = f"{self._url}/v1/vision/detect?{params}"
        req = urllib.request.Request(url, data=bytes(body), method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json_loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            raise DetectorUnavailableError(f"coralapi detection failed: {exc}") from exc

        boxes = []
        class_ids = []
        confidences = []

        for det in data.get("results", []):
            label = (det.get("label") or "").lower()
            coco_id = LABEL_TO_COCO.get(label)
            if coco_id is None and not label:
                # No labels file on the server: fall back to the class index.
                # Coral zoo detection models use 0-indexed COCO-90, whose
                # vehicle ids happen to equal our COCO-80 ids. A present but
                # unrecognized label (e.g. "dog") is still skipped above.
                coco_id = CORAL_INDEX_TO_COCO.get(det.get("index"))
            if coco_id is None:
                continue
            box = det.get("box") or [0, 0, 0, 0]
            ymin, xmin, ymax, xmax = box
            boxes.append([xmin * orig_w, ymin * orig_h, xmax * orig_w, ymax * orig_h])
            class_ids.append(coco_id)
            confidences.append(det.get("score", 0.0))

        if not boxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
            confidence=np.array(confidences, dtype=np.float32),
        )


class FallbackDetector(Detector):
    """Wraps a remote detector, optionally falling back to local CPU inference.

    Policy: if the remote backend has never succeeded this job, the first
    failure switches immediately (a down host shouldn't cost N timeouts);
    after it has succeeded, MAX_CONSECUTIVE_FAILURES consecutive failures
    trigger the switch. Once switched, the fallback is used for the rest of
    the job. Without a fallback factory, failures log a warning and yield
    empty detections (the pre-fallback behavior).

    The fallback detector is built lazily — LocalDetector loads YOLO weights
    in __init__, which healthy remote jobs must not pay for.
    """

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        primary: Detector,
        fallback_factory: Optional[Callable[[], Detector]] = None,
    ) -> None:
        self._primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: Optional[Detector] = None
        self._consecutive_failures = 0
        self._ever_succeeded = False
        self.using_fallback = False
        self.fallback_reason: Optional[str] = None

    def detect(self, frame: np.ndarray) -> sv.Detections:
        if self.using_fallback:
            assert self._fallback is not None
            return self._fallback.detect(frame)
        try:
            detections = self._primary.detect(frame)
        except DetectorUnavailableError as exc:
            self._consecutive_failures += 1
            should_switch = self._fallback_factory is not None and (
                not self._ever_succeeded
                or self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES
            )
            if should_switch:
                logger.warning(
                    "Remote detection backend unavailable (%s); "
                    "falling back to local CPU inference for the rest of this job",
                    exc,
                )
                assert self._fallback_factory is not None
                self._fallback = self._fallback_factory()
                self.using_fallback = True
                self.fallback_reason = str(exc)
                return self._fallback.detect(frame)
            logger.warning("Detection failed (%s); returning empty detections", exc)
            return sv.Detections.empty()
        self._consecutive_failures = 0
        self._ever_succeeded = True
        return detections


def create_detector(
    backend: str,
    model_name: str,
    confidence: float,
    codeproject_url: Optional[str] = None,
    codeproject_timeout: int = 30,
    roboflow_url: Optional[str] = None,
    roboflow_model_id: Optional[str] = None,
    roboflow_timeout: int = 30,
    roboflow_resize_max_px: Optional[int] = None,
    localai_url: Optional[str] = None,
    localai_model: Optional[str] = None,
    localai_timeout: int = 30,
    coralapi_url: Optional[str] = None,
    coralapi_model: Optional[str] = None,
    coralapi_timeout: int = 30,
    coralapi_resize_max_px: Optional[int] = None,
    fallback: str = "none",
) -> Detector:
    remote: Optional[Detector] = None
    if backend == "codeproject_ai":
        remote = CodeProjectAIDetector(
            url=codeproject_url or "http://localhost:32168",
            confidence=confidence,
            timeout=codeproject_timeout,
        )
    elif backend == "roboflow":
        remote = RoboflowInferenceDetector(
            url=roboflow_url or "http://localhost:9001",
            model_id=roboflow_model_id or "yolo11n-640",
            confidence=confidence,
            timeout=roboflow_timeout,
            resize_max_px=roboflow_resize_max_px,
        )
    elif backend == "localai":
        remote = LocalAIDetector(
            url=localai_url or "http://localhost:8080",
            model=localai_model or "rfdetr-base",
            confidence=confidence,
            timeout=localai_timeout,
        )
    elif backend == "coralapi":
        remote = CoralAPIDetector(
            url=coralapi_url or "http://localhost:8000",
            model=coralapi_model or "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu",
            confidence=confidence,
            timeout=coralapi_timeout,
            resize_max_px=coralapi_resize_max_px,
        )
    if remote is None:
        return LocalDetector(model_name=model_name, confidence=confidence)
    factory = (
        (lambda: LocalDetector(model_name=model_name, confidence=confidence))
        if fallback == "local"
        else None
    )
    return FallbackDetector(remote, fallback_factory=factory)
