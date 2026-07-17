from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Callable, Optional

import cv2
import numpy as np
import supervision as sv

from clockd.config import CameraConfig, ServerConfig
from clockd.models import ProcessingResult, VehicleResult
from clockd.services.detector import FallbackDetector, create_detector
from clockd.services.view_transformer import ViewTransformer
from clockd.utils.units import convert_speed, mph_to_ms
from clockd.utils.video import MAX_FRAMES, validate_video

logger = logging.getLogger(__name__)

VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def build_view_transformer(cam: CameraConfig) -> ViewTransformer:
    source = np.array(cam.calibration.source_points, dtype=np.float32)
    tw = cam.calibration.target_width_m
    th = cam.calibration.target_height_m
    target = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    return ViewTransformer(source, target)


def _smooth_positions(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(points) <= window:
        return points
    kernel = np.ones(window) / window
    smoothed_x = np.convolve(points[:, 0], kernel, mode="valid")
    smoothed_y = np.convolve(points[:, 1], kernel, mode="valid")
    return np.column_stack([smoothed_x, smoothed_y])


def _check_resolution(cap: cv2.VideoCapture, camera: CameraConfig) -> Optional[str]:
    if camera.resolution is None:
        return None
    expected_w, expected_h = camera.resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != expected_w or actual_h != expected_h:
        return (
            f"Video resolution {actual_w}x{actual_h} does not match "
            f"expected {expected_w}x{expected_h} — calibration may be inaccurate"
        )
    return None


def process_video(
    video_path: str,
    camera: CameraConfig,
    server_cfg: ServerConfig,
    unit: str = "mph",
    progress_cb: Optional[Callable[[float], None]] = None,
) -> ProcessingResult:
    t0 = time.time()
    warnings: list[str] = []

    verbose = server_cfg.verbose

    fps, total_frames = validate_video(video_path)
    duration_s = total_frames / fps

    if verbose:
        logger.info(
            "Processing video: camera=%s file=%s fps=%.1f frames=%d duration=%.1fs",
            camera.camera_id,
            video_path.rsplit("/", 1)[-1],
            fps,
            total_frames,
            duration_s,
        )

    model_name = camera.model_override or server_cfg.model
    confidence = camera.confidence_override or server_cfg.confidence
    detector = create_detector(
        backend=server_cfg.detection_backend,
        model_name=model_name,
        confidence=confidence,
        codeproject_url=server_cfg.codeproject_ai.url,
        codeproject_timeout=server_cfg.codeproject_ai.timeout,
        roboflow_url=server_cfg.roboflow.url,
        roboflow_model_id=server_cfg.roboflow.model_id,
        roboflow_timeout=server_cfg.roboflow.timeout,
        roboflow_resize_max_px=server_cfg.roboflow.resize_max_px,
        localai_url=server_cfg.localai.url,
        localai_model=server_cfg.localai.model,
        localai_timeout=server_cfg.localai.timeout,
        coralapi_url=server_cfg.coralapi.url,
        coralapi_model=server_cfg.coralapi.model,
        coralapi_timeout=server_cfg.coralapi.timeout,
        coralapi_resize_max_px=server_cfg.coralapi.resize_max_px,
        fallback=server_cfg.detection_fallback,
    )

    detection_backend = server_cfg.detection_backend
    if detection_backend == "roboflow":
        detection_model = server_cfg.roboflow.model_id
    elif detection_backend == "localai":
        detection_model = server_cfg.localai.model
    elif detection_backend == "coralapi":
        detection_model = server_cfg.coralapi.model
    elif detection_backend == "codeproject_ai":
        detection_model = "server-default"
    else:
        detection_model = model_name

    if verbose:
        logger.info(
            "Detector initialized: backend=%s model=%s confidence=%.2f",
            server_cfg.detection_backend,
            model_name,
            confidence,
        )

    tracker = sv.ByteTrack()

    roi_pts = camera.roi_polygon or camera.calibration.source_points
    roi_polygon = np.array(roi_pts, dtype=np.int32)
    zone = sv.PolygonZone(polygon=roi_polygon)

    vt = build_view_transformer(camera)

    # Speed range in m/s for filtering
    speed_min_ms = mph_to_ms(camera.speed_range.min_mph)
    speed_max_ms = mph_to_ms(camera.speed_range.max_mph)

    # track_id -> list of (frame_idx, real_world_point, confidence)
    track_data: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(list)

    cap = cv2.VideoCapture(video_path)

    # Check resolution
    res_warning = _check_resolution(cap, camera)
    if res_warning:
        warnings.append(res_warning)
        logger.warning(res_warning)

    frame_idx = 0

    while cap.isOpened():
        # Backstop: container metadata (already validated) can under-report the
        # real frame count, so cap the decode loop itself too.
        if frame_idx >= MAX_FRAMES:
            msg = f"Stopped at {MAX_FRAMES} frames; video contains more frames than its metadata reported"
            warnings.append(msg)
            logger.warning("camera=%s: %s", camera.camera_id, msg)
            break
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        # Filter to vehicle classes
        mask = np.isin(detections.class_id, VEHICLE_CLASS_IDS)
        detections = detections[mask]

        # Filter to ROI
        roi_mask = zone.trigger(detections)
        detections = detections[roi_mask]

        # Track
        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is not None and len(detections) > 0:
            # Bottom-center anchor points
            anchors = np.array([[(x1 + x2) / 2, y2] for x1, y1, x2, y2 in detections.xyxy])
            real_points = vt.transform_points(anchors)
            confs = detections.confidence

            for tid, pt, conf in zip(detections.tracker_id, real_points, confs):
                track_data[int(tid)].append((frame_idx, pt, float(conf)))

        frame_idx += 1
        if progress_cb and total_frames > 0:
            progress_cb(frame_idx / total_frames)

    cap.release()

    # Calculate speeds
    vehicles: list[VehicleResult] = []
    filtered_count = 0

    for tid, entries in track_data.items():
        # Minimum detections filter
        if len(entries) < camera.min_detections:
            if len(entries) >= 2:
                filtered_count += 1
            continue

        frames = [e[0] for e in entries]
        points = np.array([e[1] for e in entries])
        confidences = [e[2] for e in entries]

        first_frame = frames[0]
        last_frame = frames[-1]
        total_time = (last_frame - first_frame) / fps

        if total_time <= 0:
            continue

        # Light smoothing to stabilize anchor positions before computing
        # segment displacements. Heavier smoothing is unnecessary because
        # the segment-based approach already averages over ~1s windows.
        smoothed = _smooth_positions(points, camera.smoothing_window)

        if len(smoothed) < 2:
            continue

        # Average speed: displacement from first to last smoothed position
        # divided by total time. Robust to per-frame bbox jitter and
        # non-uniform homography distortion — only the endpoints matter,
        # each already stabilized by the smoothing window.
        displacement = np.linalg.norm(smoothed[-1] - smoothed[0])
        avg_speed_ms = displacement / total_time

        # Apply calibration factor
        cal = camera.speed_calibration_factor
        avg_speed_ms *= cal

        # Filter by plausible speed range (after calibration)
        if avg_speed_ms < speed_min_ms or avg_speed_ms > speed_max_ms:
            filtered_count += 1
            continue

        # Segment-based min/max: split the track into ~1-second overlapping
        # segments and compute displacement within each. This is far more
        # stable than frame-to-frame instantaneous speeds because each
        # measurement spans enough distance for a good signal-to-noise ratio.
        offset = len(points) - len(smoothed)
        smoothed_frames = frames[offset:]
        segment_len = max(2, int(fps))  # ~1 second of frames
        step = max(1, segment_len // 2)  # 50% overlap
        segment_speeds_ms: list[float] = []
        for s in range(0, len(smoothed) - segment_len, step):
            e = s + segment_len
            seg_disp = np.linalg.norm(smoothed[e] - smoothed[s])
            seg_dt = (smoothed_frames[e] - smoothed_frames[s]) / fps
            if seg_dt > 0:
                segment_speeds_ms.append(seg_disp / seg_dt * cal)

        if not segment_speeds_ms:
            # Track too short for segments — fall back to avg
            segment_speeds_ms = [avg_speed_ms]

        seg_arr = np.array(segment_speeds_ms)
        avg_speed = convert_speed(avg_speed_ms, unit)
        min_speed = convert_speed(float(np.min(seg_arr)), unit)
        max_speed = convert_speed(float(np.max(seg_arr)), unit)
        mean_conf = round(float(np.mean(confidences)), 3)

        vehicles.append(
            VehicleResult(
                track_id=tid,
                speed_avg=avg_speed,
                speed_min=min_speed,
                speed_max=max_speed,
                unit=unit,
                mean_detection_confidence=mean_conf,
                first_seen_frame=first_frame,
                last_seen_frame=last_frame,
                first_seen_timestamp_s=round(first_frame / fps, 2),
                last_seen_timestamp_s=round(last_frame / fps, 2),
                num_detections=len(entries),
            )
        )

    vehicles.sort(key=lambda v: v.first_seen_frame)

    if filtered_count > 0:
        warnings.append(
            f"{filtered_count} vehicle(s) excluded by min_detections or speed_range filters"
        )

    processing_time = round(time.time() - t0, 2)

    if verbose:
        logger.info(
            "Processing complete: camera=%s vehicles=%d filtered=%d time=%.2fs",
            camera.camera_id,
            len(vehicles),
            filtered_count,
            processing_time,
        )
        for v in vehicles:
            logger.info(
                "  Vehicle track=%d speed_avg=%.1f speed_min=%.1f speed_max=%.1f %s "
                "confidence=%.3f detections=%d frames=%d-%d",
                v.track_id,
                v.speed_avg,
                v.speed_min,
                v.speed_max,
                v.unit,
                v.mean_detection_confidence,
                v.num_detections,
                v.first_seen_frame,
                v.last_seen_frame,
            )

    fallback_used = isinstance(detector, FallbackDetector) and detector.using_fallback
    if fallback_used:
        detection_backend = "local"
        detection_model = model_name
        warnings.append("Remote detection backend unavailable; fell back to local CPU inference")

    return ProcessingResult(
        camera_id=camera.camera_id,
        video_filename=video_path.rsplit("/", 1)[-1],
        fps=fps,
        total_frames=frame_idx,
        duration_s=round(duration_s, 2),
        unit=unit,
        vehicles=vehicles,
        vehicles_filtered=filtered_count,
        processing_time_s=processing_time,
        warnings=warnings,
        detection_backend=detection_backend,
        detection_model=detection_model,
        detection_fallback=fallback_used,
    )
