from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_cameras_list(client):
    resp = await client.get("/cameras")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["camera_id"] == "test_cam"


@pytest.mark.asyncio
async def test_camera_get(client):
    resp = await client.get("/cameras/test_cam")
    assert resp.status_code == 200
    assert resp.json()["camera_id"] == "test_cam"


@pytest.mark.asyncio
async def test_camera_not_found(client):
    resp = await client.get("/cameras/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_process_missing_camera(client):
    resp = await client.post(
        "/process",
        data={"camera_id": "nonexistent"},
        files={"file": ("test.mp4", b"fake", "video/mp4")},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_process_invalid_unit(client):
    resp = await client.post(
        "/process",
        data={"camera_id": "test_cam", "unit": "invalid"},
        files={"file": ("test.mp4", b"fake", "video/mp4")},
    )
    assert resp.status_code == 400


NEW_CAM = {
    "camera_id": "back_yard",
    "description": "Back yard camera",
    "calibration": {
        "source_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "target_width_m": 5.0,
        "target_height_m": 20.0,
    },
}


@pytest.mark.asyncio
async def test_create_camera(client):
    resp = await client.post("/cameras", json=NEW_CAM)
    assert resp.status_code == 201
    assert resp.json()["camera_id"] == "back_yard"

    # Verify it shows in the list
    resp = await client.get("/cameras")
    ids = [c["camera_id"] for c in resp.json()]
    assert "back_yard" in ids


@pytest.mark.asyncio
async def test_create_camera_duplicate(client):
    await client.post("/cameras", json=NEW_CAM)
    resp = await client.post("/cameras", json=NEW_CAM)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_update_camera(client):
    await client.post("/cameras", json=NEW_CAM)
    updated = {**NEW_CAM, "description": "Updated description"}
    resp = await client.put("/cameras/back_yard", json=updated)
    assert resp.status_code == 200
    assert resp.json()["description"] == "Updated description"


@pytest.mark.asyncio
async def test_update_camera_not_found(client):
    resp = await client.put("/cameras/nonexistent", json=NEW_CAM)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_camera(client):
    await client.post("/cameras", json=NEW_CAM)
    resp = await client.delete("/cameras/back_yard")
    assert resp.status_code == 204

    resp = await client.get("/cameras/back_yard")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_camera_not_found(client):
    resp = await client.delete("/cameras/nonexistent")
    assert resp.status_code == 404
