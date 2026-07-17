from __future__ import annotations

import pytest

from clockd.main import app


def _camera_body(camera_id: str) -> dict:
    return {
        "camera_id": camera_id,
        "calibration": {
            "source_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "target_width_m": 8.0,
            "target_height_m": 40.0,
        },
    }


@pytest.mark.asyncio
async def test_create_camera_limit_reached(client):
    # The client fixture starts with one camera registered
    app.state.server_cfg.max_cameras = 1
    resp = await client.post("/cameras", json=_camera_body("over_limit"))
    assert resp.status_code == 409
    assert "Camera limit reached" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_camera_under_limit(client):
    app.state.server_cfg.max_cameras = 10
    resp = await client.post("/cameras", json=_camera_body("under_limit"))
    assert resp.status_code == 201
    assert resp.json()["camera_id"] == "under_limit"


@pytest.mark.asyncio
async def test_update_camera_rename_onto_existing_conflicts(client):
    # A second camera exists; renaming test_cam onto its id must not clobber it.
    app.state.server_cfg.max_cameras = 10
    assert (await client.post("/cameras", json=_camera_body("other_cam"))).status_code == 201

    resp = await client.put("/cameras/test_cam", json=_camera_body("other_cam"))
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]

    # Both cameras survive unchanged — no silent data loss.
    assert (await client.get("/cameras/test_cam")).status_code == 200
    assert (await client.get("/cameras/other_cam")).status_code == 200


@pytest.mark.asyncio
async def test_update_camera_rename_to_free_id_succeeds(client):
    resp = await client.put("/cameras/test_cam", json=_camera_body("renamed_cam"))
    assert resp.status_code == 200
    assert resp.json()["camera_id"] == "renamed_cam"
    # Old id is gone, new id resolves.
    assert (await client.get("/cameras/test_cam")).status_code == 404
    assert (await client.get("/cameras/renamed_cam")).status_code == 200
