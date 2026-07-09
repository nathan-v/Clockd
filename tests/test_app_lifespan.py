"""Test the app lifespan and middleware with prometheus enabled."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from clockd.main import app


@pytest.mark.asyncio
async def test_lifespan_initializes_state(tmp_path):
    """App lifespan initializes server_cfg, cameras, metrics, and job_manager."""
    with (
        patch("clockd.main.load_server_config") as mock_cfg,
        patch("clockd.main.load_cameras", return_value={}),
    ):
        from clockd.config import ServerConfig

        mock_cfg.return_value = ServerConfig(
            cameras_dir=str(tmp_path),
            upload_dir=str(tmp_path),
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

            # Verify lifespan set up app state
            assert hasattr(app.state, "server_cfg")
            assert hasattr(app.state, "cameras")
            assert hasattr(app.state, "metrics")
            assert hasattr(app.state, "job_manager")
