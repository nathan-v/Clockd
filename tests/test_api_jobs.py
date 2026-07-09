from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_list_jobs_empty(client):
    resp = await client.get("/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_job_not_found(client):
    resp = await client.get("/jobs/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_job_not_found(client):
    resp = await client.delete("/jobs/nonexistent")
    assert resp.status_code == 404
