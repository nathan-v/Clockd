from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from clockd import __version__
from clockd.config import load_cameras, load_server_config
from clockd.middleware import MetricsMiddleware
from clockd.models import HealthResponse
from clockd.routers import calibrate, cameras, jobs, process
from clockd.services.job_manager import JobManager
from clockd.services.metrics import MetricsService


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_server_config()
    if cfg.verbose:
        logging.getLogger("clockd").setLevel(logging.INFO)
    app.state.server_cfg = cfg
    app.state.cameras = load_cameras(cfg.cameras_dir)
    app.state.metrics = MetricsService(cfg.metrics)
    app.state.job_manager = JobManager(
        max_workers=cfg.max_workers,
        ttl_seconds=cfg.job_ttl_seconds,
        metrics=app.state.metrics,
    )

    if cfg.metrics.prometheus.enabled:
        from prometheus_client import make_asgi_app

        metrics_app = make_asgi_app()
        app.mount(cfg.metrics.prometheus.path, metrics_app)

    # Start event sources (Protect poller, etc.)
    from clockd.services.event_sources.manager import EventSourceManager, create_event_source

    esm = EventSourceManager()
    for source_name, source_cfg in cfg.event_sources.items():
        if not source_cfg.enabled:
            continue
        try:
            source = create_event_source(
                source_name, source_cfg, app.state.cameras, cfg, app.state.job_manager
            )
            esm.add(source)
        except Exception:
            logging.getLogger(__name__).exception("Failed to create event source: %s", source_name)
    await esm.start_all()
    app.state.event_sources = esm

    yield

    await app.state.event_sources.stop_all()
    app.state.job_manager.shutdown()


app = FastAPI(title="Clockd", version=__version__, lifespan=lifespan)
app.add_middleware(MetricsMiddleware)

app.include_router(process.router)
app.include_router(jobs.router)
app.include_router(cameras.router)
app.include_router(calibrate.router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=__version__)
