"""Tests for the /process endpoint with mocked pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from clockd.models import ProcessingResult


def _mock_result():
    return ProcessingResult(
        camera_id="test_cam",
        video_filename="test.mp4",
        fps=30.0,
        total_frames=90,
        duration_s=3.0,
        unit="mph",
        vehicles=[],
        processing_time_s=0.1,
    )


@pytest.mark.asyncio
async def test_process_sync(client, tmp_path):
    """POST /process with sync mode returns result."""
    with (
        patch("clockd.routers.process.stream_upload_to_disk", return_value=str(tmp_path / "f.mp4")),
        patch("clockd.routers.process.process_video", return_value=_mock_result()),
        patch("clockd.routers.process.cleanup"),
    ):
        resp = await client.post(
            "/process",
            data={"camera_id": "test_cam", "unit": "mph"},
            files={"file": ("clip.mp4", b"fake video", "video/mp4")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["camera_id"] == "test_cam"
    assert data["vehicles"] == []


@pytest.mark.asyncio
async def test_process_async(client, tmp_path):
    """POST /process with async_mode=true returns 202 with job_id."""
    with patch(
        "clockd.routers.process.stream_upload_to_disk", return_value=str(tmp_path / "f.mp4")
    ):
        resp = await client.post(
            "/process",
            data={"camera_id": "test_cam", "async_mode": "true"},
            files={"file": ("clip.mp4", b"fake video", "video/mp4")},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_process_queue_full(client, tmp_path):
    """When job queue is full, returns 503."""
    from clockd.main import app

    with (
        patch("clockd.routers.process.stream_upload_to_disk", return_value=str(tmp_path / "f.mp4")),
        patch("clockd.routers.process.cleanup"),
        patch.object(
            app.state.job_manager, "submit", side_effect=RuntimeError("Too many jobs queued")
        ),
    ):
        resp = await client.post(
            "/process",
            data={"camera_id": "test_cam", "async_mode": "true"},
            files={"file": ("clip.mp4", b"fake video", "video/mp4")},
        )

    assert resp.status_code == 503
