from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_no_prometheus(client):
    """Middleware works when prometheus is disabled (default in tests)."""
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_middleware_records_requests(client):
    """When Prometheus is disabled, middleware still works via record_request."""
    from unittest.mock import MagicMock

    from clockd.main import app

    original_metrics = app.state.metrics
    mock_metrics = MagicMock()
    mock_metrics.prometheus = None
    app.state.metrics = mock_metrics

    resp = await client.get("/health")
    assert resp.status_code == 200

    # Middleware calls record_request even without Prometheus (for InfluxDB)
    mock_metrics.record_request.assert_called_once()

    app.state.metrics = original_metrics
