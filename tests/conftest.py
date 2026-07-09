from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from clockd.config import CalibrationConfig, CameraConfig, ServerConfig
from clockd.main import app


@pytest.fixture
def sample_camera() -> CameraConfig:
    return CameraConfig(
        camera_id="test_cam",
        description="Test camera",
        calibration=CalibrationConfig(
            source_points=[[100, 400], [700, 400], [900, 700], [50, 700]],
            target_width_m=8.0,
            target_height_m=40.0,
        ),
    )


@pytest.fixture
def server_cfg(tmp_path) -> ServerConfig:
    return ServerConfig(
        upload_dir=str(tmp_path / "uploads"),
        cameras_dir=str(tmp_path / "cameras"),
    )


@pytest.fixture
async def client(sample_camera, server_cfg):
    app.state.server_cfg = server_cfg
    app.state.cameras = {sample_camera.camera_id: sample_camera}

    from clockd.services.metrics import MetricsService

    app.state.metrics = MetricsService(server_cfg.metrics)

    from clockd.services.job_manager import JobManager

    app.state.job_manager = JobManager(max_workers=1, metrics=app.state.metrics)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.state.job_manager.shutdown()
