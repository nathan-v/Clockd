from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


def _get_route_template(request: Request) -> str:
    """Get the route template (e.g. /jobs/{job_id}) instead of the raw path to avoid cardinality explosion."""
    route = request.scope.get("route")
    if route and hasattr(route, "path"):
        return route.path
    return request.url.path


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        metrics = getattr(request.app.state, "metrics", None)
        if metrics is None:
            return await call_next(request)

        prom = metrics.prometheus
        method = request.method
        if prom:
            prom.http_requests_in_progress.labels(method=method).inc()

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            if prom:
                prom.http_requests_in_progress.labels(method=method).dec()
            raise

        duration = time.perf_counter() - t0
        path = _get_route_template(request)
        metrics.record_request(method, path, response.status_code, duration)
        if prom:
            prom.http_requests_in_progress.labels(method=method).dec()

        return response
