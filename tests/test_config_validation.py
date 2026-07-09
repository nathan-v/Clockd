from __future__ import annotations

import pytest
from pydantic import ValidationError

from clockd.config import CalibrationConfig, CameraConfig, SpeedRange


def _base_calibration():
    return CalibrationConfig(
        source_points=[[0, 0], [1, 0], [1, 1], [0, 1]],
        target_width_m=5.0,
        target_height_m=10.0,
    )


def test_camera_id_valid():
    cam = CameraConfig(camera_id="front_yard-1", calibration=_base_calibration())
    assert cam.camera_id == "front_yard-1"


def test_camera_id_path_traversal():
    with pytest.raises(ValidationError, match="camera_id"):
        CameraConfig(camera_id="../../etc/evil", calibration=_base_calibration())


def test_camera_id_spaces():
    with pytest.raises(ValidationError, match="camera_id"):
        CameraConfig(camera_id="has spaces", calibration=_base_calibration())


def test_camera_id_empty():
    with pytest.raises(ValidationError, match="camera_id"):
        CameraConfig(camera_id="", calibration=_base_calibration())


def test_model_override_valid():
    cam = CameraConfig(
        camera_id="test", calibration=_base_calibration(), model_override="yolov8m.pt"
    )
    assert cam.model_override == "yolov8m.pt"


def test_model_override_invalid():
    with pytest.raises(ValidationError, match="model_override"):
        CameraConfig(
            camera_id="test", calibration=_base_calibration(), model_override="/evil/model.pt"
        )


def test_source_points_wrong_count():
    with pytest.raises(ValidationError, match="exactly 4"):
        CalibrationConfig(
            source_points=[[0, 0], [1, 0], [1, 1]],
            target_width_m=5.0,
            target_height_m=10.0,
        )


def test_source_points_wrong_dimensions():
    with pytest.raises(ValidationError, match="x, y"):
        CalibrationConfig(
            source_points=[[0, 0, 0], [1, 0], [1, 1], [0, 1]],
            target_width_m=5.0,
            target_height_m=10.0,
        )


def test_target_width_zero():
    with pytest.raises(ValidationError, match="between 0 and 1000"):
        CalibrationConfig(
            source_points=[[0, 0], [1, 0], [1, 1], [0, 1]],
            target_width_m=0,
            target_height_m=10.0,
        )


def test_target_width_too_large():
    with pytest.raises(ValidationError, match="between 0 and 1000"):
        CalibrationConfig(
            source_points=[[0, 0], [1, 0], [1, 1], [0, 1]],
            target_width_m=5000,
            target_height_m=10.0,
        )


def test_confidence_override_bounds():
    with pytest.raises(ValidationError, match="confidence"):
        CameraConfig(camera_id="t", calibration=_base_calibration(), confidence_override=1.5)


def test_min_detections_too_low():
    with pytest.raises(ValidationError, match="min_detections"):
        CameraConfig(camera_id="t", calibration=_base_calibration(), min_detections=1)


def test_smoothing_window_too_high():
    with pytest.raises(ValidationError, match="smoothing_window"):
        CameraConfig(camera_id="t", calibration=_base_calibration(), smoothing_window=200)


def test_speed_range_negative():
    with pytest.raises(ValidationError, match="Speed"):
        SpeedRange(min_mph=-5.0)


def test_speed_range_too_high():
    with pytest.raises(ValidationError, match="Speed"):
        SpeedRange(max_mph=600.0)
