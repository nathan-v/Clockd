"""Direct unit tests for middleware functions."""

from __future__ import annotations

from unittest.mock import MagicMock

from clockd.middleware import _get_route_template


def test_get_route_template_with_route():
    request = MagicMock()
    route = MagicMock()
    route.path = "/jobs/{job_id}"
    request.scope = {"route": route}
    assert _get_route_template(request) == "/jobs/{job_id}"


def test_get_route_template_no_route():
    request = MagicMock()
    request.scope = {}
    request.url.path = "/some/raw/path"
    assert _get_route_template(request) == "/some/raw/path"


def test_get_route_template_route_no_path():
    request = MagicMock()
    route = MagicMock(spec=[])  # no path attribute
    request.scope = {"route": route}
    request.url.path = "/fallback"
    assert _get_route_template(request) == "/fallback"
