from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import supervision as sv


def _make_test_image() -> bytes:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 300), (900, 700), (128, 128, 128), -1)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_preview_returns_png(client):
    img_bytes = _make_test_image()
    resp = await client.post(
        "/calibrate/preview",
        data={"camera_id": "test_cam", "detect": "false"},
        files={"file": ("frame.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 100


@pytest.mark.asyncio
async def test_preview_camera_not_found(client):
    img_bytes = _make_test_image()
    resp = await client.post(
        "/calibrate/preview",
        data={"camera_id": "nonexistent", "detect": "false"},
        files={"file": ("frame.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_preview_invalid_image(client):
    resp = await client.post(
        "/calibrate/preview",
        data={"camera_id": "test_cam", "detect": "false"},
        files={"file": ("frame.png", b"not an image", "image/png")},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_warp_returns_png(client):
    img_bytes = _make_test_image()
    resp = await client.post(
        "/calibrate/warp",
        data={"camera_id": "test_cam"},
        files={"file": ("frame.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

    # Decode and check dimensions match expected scale (50px/m)
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    warped = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    expected_w = int(8.0 * 50)  # target_width_m * scale
    expected_h = int(40.0 * 50)  # target_height_m * scale
    assert warped.shape[1] == expected_w
    assert warped.shape[0] == expected_h


@pytest.mark.asyncio
async def test_calibration_ui(client):
    resp = await client.get("/calibrate/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Clockd Calibration Tool" in resp.text


@pytest.mark.asyncio
async def test_extract_frame_invalid_file(client):
    resp = await client.post(
        "/calibrate/extract-frame",
        data={"frame_number": "0"},
        files={"file": ("clip.mp4", b"not a video", "video/mp4")},
    )
    assert resp.status_code == 400


# ── Speed test endpoint ───────────────────────────────────────────────────


def _make_test_video_bytes(tmp_path, fps=30, frames=60, w=640, h=480) -> bytes:
    path = str(tmp_path / "speed_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for _ in range(frames):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()
    with open(path, "rb") as f:
        return f.read()


def _make_detections(boxes, cls_ids, confs):
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(cls_ids, dtype=int),
        confidence=np.array(confs, dtype=np.float32),
    )


def _moving_car_detector(speed_px_per_frame=10):
    """Return a detector mock that produces a car moving across the frame."""
    frame_counter = [0]

    def detect(frame):
        i = frame_counter[0]
        frame_counter[0] += 1
        x = 100 + i * speed_px_per_frame
        if x > 500:
            return sv.Detections.empty()
        return _make_detections(
            [[x, 450, x + 80, 550]],
            [2],
            [0.9],  # class 2 = car
        )

    mock = MagicMock()
    mock.detect = MagicMock(side_effect=detect)
    return mock


@pytest.mark.asyncio
async def test_speed_test_returns_factor(client, tmp_path):
    video_bytes = _make_test_video_bytes(tmp_path)
    detector = _moving_car_detector()

    with patch("clockd.services.pipeline.create_detector", return_value=detector):
        resp = await client.post(
            "/calibrate/speed-test",
            data={
                "camera_id": "test_cam",
                "known_speed": "25.0",
                "unit": "mph",
                "apply": "false",
            },
            files={"file": ("clip.mp4", video_bytes, "video/mp4")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["camera_id"] == "test_cam"
    assert data["known_speed"] == 25.0
    assert "measured_speed" in data
    assert "recommended_factor" in data
    assert data["applied"] is False
    assert data["recommended_factor"] > 0


@pytest.mark.asyncio
async def test_speed_test_no_vehicles(client, tmp_path):
    video_bytes = _make_test_video_bytes(tmp_path)

    # Detector that returns no vehicles
    mock_det = MagicMock()
    mock_det.detect = MagicMock(return_value=sv.Detections.empty())

    with patch("clockd.services.pipeline.create_detector", return_value=mock_det):
        resp = await client.post(
            "/calibrate/speed-test",
            data={
                "camera_id": "test_cam",
                "known_speed": "25.0",
                "unit": "mph",
            },
            files={"file": ("clip.mp4", video_bytes, "video/mp4")},
        )

    assert resp.status_code == 422
    assert "No vehicles" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_speed_test_camera_not_found(client, tmp_path):
    video_bytes = _make_test_video_bytes(tmp_path)

    resp = await client.post(
        "/calibrate/speed-test",
        data={
            "camera_id": "nonexistent",
            "known_speed": "25.0",
            "unit": "mph",
        },
        files={"file": ("clip.mp4", video_bytes, "video/mp4")},
    )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_speed_test_invalid_unit(client, tmp_path):
    video_bytes = _make_test_video_bytes(tmp_path)

    resp = await client.post(
        "/calibrate/speed-test",
        data={
            "camera_id": "test_cam",
            "known_speed": "25.0",
            "unit": "furlongs",
        },
        files={"file": ("clip.mp4", video_bytes, "video/mp4")},
    )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_speed_test_apply_saves_config(client, tmp_path, sample_camera, server_cfg):
    video_bytes = _make_test_video_bytes(tmp_path)
    detector = _moving_car_detector()

    # Ensure cameras_dir exists for save
    os.makedirs(server_cfg.cameras_dir, exist_ok=True)

    with patch("clockd.services.pipeline.create_detector", return_value=detector):
        resp = await client.post(
            "/calibrate/speed-test",
            data={
                "camera_id": "test_cam",
                "known_speed": "25.0",
                "unit": "mph",
                "apply": "true",
            },
            files={"file": ("clip.mp4", video_bytes, "video/mp4")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"] is True

    # Verify the camera config was updated in app state
    from clockd.main import app

    updated_cam = app.state.cameras["test_cam"]
    assert updated_cam.speed_calibration_factor == data["recommended_factor"]
