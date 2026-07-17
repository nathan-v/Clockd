"""Pipeline tests with mocked detector to cover the full detect->track->speed flow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import supervision as sv

from clockd.config import CalibrationConfig, CameraConfig, ServerConfig, SpeedRange
from clockd.services.pipeline import _check_resolution, process_video


def _make_camera(**overrides) -> CameraConfig:
    defaults = dict(
        camera_id="test",
        calibration=CalibrationConfig(
            source_points=[[0, 0], [640, 0], [640, 480], [0, 480]],
            target_width_m=8.0,
            target_height_m=40.0,
        ),
        min_detections=2,
        smoothing_window=1,
        speed_range=SpeedRange(min_mph=1.0, max_mph=200.0),
    )
    defaults.update(overrides)
    return CameraConfig(**defaults)


def _make_server_cfg(tmp_path) -> ServerConfig:
    return ServerConfig(upload_dir=str(tmp_path), cameras_dir=str(tmp_path))


def _make_test_video(tmp_path, fps=30, frames=60, w=640, h=480):
    path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for _ in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _make_detections(boxes, cls_ids, confs):
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(cls_ids, dtype=int),
        confidence=np.array(confs, dtype=np.float32),
    )


def _mock_detector_returning(detections_or_fn):
    """Create a mock detector that returns fixed detections or calls a function."""
    mock_det = MagicMock()
    if callable(detections_or_fn) and not isinstance(detections_or_fn, sv.Detections):
        mock_det.detect = MagicMock(side_effect=detections_or_fn)
    else:
        mock_det.detect = MagicMock(return_value=detections_or_fn)
    return mock_det


def test_process_video_no_detections(tmp_path):
    video_path = _make_test_video(tmp_path)
    camera = _make_camera()
    cfg = _make_server_cfg(tmp_path)

    mock_det = _mock_detector_returning(_make_detections([], [], []))
    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        result = process_video(video_path, camera, cfg, "mph")

    assert result.camera_id == "test"
    assert len(result.vehicles) == 0
    assert result.fps == 30.0
    assert result.total_frames == 60


def test_process_video_with_moving_vehicle(tmp_path):
    video_path = _make_test_video(tmp_path, fps=30, frames=30)
    camera = _make_camera(min_detections=2, smoothing_window=1)
    cfg = _make_server_cfg(tmp_path)

    frame_count = [0]

    def detect_fn(frame):
        idx = frame_count[0]
        frame_count[0] += 1
        x1 = 100 + idx * 10
        return _make_detections([[x1, 200, x1 + 80, 280]], [2], [0.9])

    mock_det = _mock_detector_returning(detect_fn)
    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        result = process_video(video_path, camera, cfg, "mph")

    assert len(result.vehicles) > 0
    v = result.vehicles[0]
    assert v.speed_avg > 0
    assert v.speed_min > 0
    assert v.speed_max > 0
    assert v.mean_detection_confidence == 0.9
    assert v.num_detections >= 2
    assert v.unit == "mph"


def test_process_video_filters_non_vehicle_classes(tmp_path):
    video_path = _make_test_video(tmp_path, frames=10)
    camera = _make_camera(min_detections=2)
    cfg = _make_server_cfg(tmp_path)

    mock_det = _mock_detector_returning(_make_detections([[100, 100, 200, 200]], [0], [0.95]))
    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        result = process_video(video_path, camera, cfg, "mph")

    assert len(result.vehicles) == 0


def test_process_video_min_detections_filter(tmp_path):
    video_path = _make_test_video(tmp_path, frames=5)
    camera = _make_camera(min_detections=100)
    cfg = _make_server_cfg(tmp_path)

    frame_count = [0]

    def detect_fn(frame):
        idx = frame_count[0]
        frame_count[0] += 1
        return _make_detections([[100 + idx * 20, 200, 200 + idx * 20, 300]], [2], [0.8])

    mock_det = _mock_detector_returning(detect_fn)
    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        result = process_video(video_path, camera, cfg, "mph")

    assert len(result.vehicles) == 0
    assert result.vehicles_filtered > 0


def test_process_video_speed_range_filter(tmp_path):
    video_path = _make_test_video(tmp_path, fps=30, frames=30)
    camera = _make_camera(
        min_detections=2,
        smoothing_window=1,
        speed_range=SpeedRange(min_mph=100.0, max_mph=200.0),
    )
    cfg = _make_server_cfg(tmp_path)

    frame_count = [0]

    def detect_fn(frame):
        idx = frame_count[0]
        frame_count[0] += 1
        return _make_detections([[100 + idx, 200, 180 + idx, 280]], [2], [0.85])

    mock_det = _mock_detector_returning(detect_fn)
    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        result = process_video(video_path, camera, cfg, "mph")

    assert len(result.vehicles) == 0
    assert result.vehicles_filtered > 0


def test_process_video_resolution_warning(tmp_path):
    video_path = _make_test_video(tmp_path, w=640, h=480)
    camera = _make_camera(resolution=[1920, 1080])
    cfg = _make_server_cfg(tmp_path)

    mock_det = _mock_detector_returning(_make_detections([], [], []))
    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        result = process_video(video_path, camera, cfg, "mph")

    assert len(result.warnings) > 0
    assert "resolution" in result.warnings[0].lower()


def test_process_video_kmh_unit(tmp_path):
    video_path = _make_test_video(tmp_path, frames=20)
    camera = _make_camera(min_detections=2, smoothing_window=1)
    cfg = _make_server_cfg(tmp_path)

    frame_count = [0]

    def detect_fn(frame):
        idx = frame_count[0]
        frame_count[0] += 1
        return _make_detections([[100 + idx * 15, 200, 200 + idx * 15, 300]], [2], [0.9])

    mock_det = _mock_detector_returning(detect_fn)
    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        result = process_video(video_path, camera, cfg, "kmh")

    if result.vehicles:
        assert result.vehicles[0].unit == "kmh"


def test_process_video_progress_callback(tmp_path):
    video_path = _make_test_video(tmp_path, frames=10)
    camera = _make_camera()
    cfg = _make_server_cfg(tmp_path)

    mock_det = _mock_detector_returning(_make_detections([], [], []))
    progress_values = []

    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        process_video(video_path, camera, cfg, "mph", progress_cb=progress_values.append)

    assert len(progress_values) == 10
    assert progress_values[-1] == pytest.approx(1.0)


def test_check_resolution_match():
    camera = _make_camera(resolution=[640, 480])
    cap = MagicMock()
    cap.get = MagicMock(
        side_effect=lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)
    )
    assert _check_resolution(cap, camera) is None


def test_check_resolution_mismatch():
    camera = _make_camera(resolution=[1920, 1080])
    cap = MagicMock()
    cap.get = MagicMock(
        side_effect=lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)
    )
    warning = _check_resolution(cap, camera)
    assert warning is not None
    assert "640x480" in warning


def test_check_resolution_none():
    camera = _make_camera(resolution=None)
    assert _check_resolution(MagicMock(), camera) is None


def test_process_video_frame_cap(tmp_path, monkeypatch):
    import clockd.services.pipeline as pipeline_mod

    video_path = _make_test_video(tmp_path, frames=60)
    camera = _make_camera()
    cfg = _make_server_cfg(tmp_path)
    monkeypatch.setattr(pipeline_mod, "MAX_FRAMES", 30)

    mock_det = _mock_detector_returning(_make_detections([], [], []))
    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        result = process_video(video_path, camera, cfg, "mph")

    assert result.total_frames == 30
    assert any("Stopped at 30 frames" in w for w in result.warnings)


def test_process_video_min_displacement_filters_stationary_jitter(tmp_path):
    # A parked vehicle with jittery boxes: oscillates +/-3px around a fixed
    # spot, so net real-world displacement is near zero. With a displacement
    # floor it must be excluded; without one it leaks through as a low-speed
    # phantom whenever the speed floor is disabled.
    video_path = _make_test_video(tmp_path, fps=30, frames=30)
    cfg = _make_server_cfg(tmp_path)

    def jitter_detector():
        frame_count = [0]

        def detect_fn(frame):
            idx = frame_count[0]
            frame_count[0] += 1
            wobble = 3 if idx % 2 else -3
            return _make_detections([[100 + wobble, 200, 180 + wobble, 280]], [2], [0.85])

        return _mock_detector_returning(detect_fn)

    camera = _make_camera(
        min_detections=2,
        smoothing_window=1,
        speed_range=SpeedRange(min_mph=0.0, max_mph=200.0),
        min_displacement_m=5.0,
    )
    with patch("clockd.services.pipeline.create_detector", return_value=jitter_detector()):
        result = process_video(video_path, camera, cfg, "mph")
    assert len(result.vehicles) == 0
    assert result.vehicles_filtered > 0

    # Control: same scene with the filter disabled leaks the phantom track.
    camera = _make_camera(
        min_detections=2,
        smoothing_window=1,
        speed_range=SpeedRange(min_mph=0.0, max_mph=200.0),
        min_displacement_m=0.0,
    )
    with patch("clockd.services.pipeline.create_detector", return_value=jitter_detector()):
        result = process_video(video_path, camera, cfg, "mph")
    assert len(result.vehicles) == 1


def test_process_video_min_displacement_keeps_moving_vehicle(tmp_path):
    # A genuinely moving vehicle must survive the displacement floor.
    video_path = _make_test_video(tmp_path, fps=30, frames=30)
    cfg = _make_server_cfg(tmp_path)
    camera = _make_camera(
        min_detections=2,
        smoothing_window=1,
        speed_range=SpeedRange(min_mph=0.0, max_mph=200.0),
        min_displacement_m=5.0,
    )

    frame_count = [0]

    def detect_fn(frame):
        idx = frame_count[0]
        frame_count[0] += 1
        return _make_detections([[100 + idx * 15, 200, 180 + idx * 15, 280]], [2], [0.85])

    mock_det = _mock_detector_returning(detect_fn)
    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        result = process_video(video_path, camera, cfg, "mph")
    assert len(result.vehicles) == 1
