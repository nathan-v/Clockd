from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clockd.config import CameraConfig, ServerConfig, UnifiEventSourceConfig
    from clockd.services.job_manager import JobManager

from clockd.services.event_sources.base import EventSource
from clockd.services.event_sources.unifi_protect import UnifiProtectSource

logger = logging.getLogger(__name__)


def create_event_source(
    source_name: str,
    config: UnifiEventSourceConfig,
    cameras: dict[str, CameraConfig],
    server_cfg: ServerConfig,
    job_manager: JobManager,
) -> EventSource:
    """Factory: create an EventSource from config. Extend this when adding Frigate, etc."""
    if hasattr(config, "unifi"):
        return UnifiProtectSource(source_name, config, cameras, server_cfg, job_manager)
    raise ValueError(f"Unknown event source type for '{source_name}'")


class EventSourceManager:
    """Manages lifecycle of all configured event sources."""

    def __init__(self) -> None:
        self._sources: list[EventSource] = []

    def add(self, source: EventSource) -> None:
        self._sources.append(source)

    async def start_all(self) -> None:
        for source in self._sources:
            try:
                await source.start()
            except Exception:
                logger.exception("Failed to start event source: %s", source.name)

    async def stop_all(self) -> None:
        for source in self._sources:
            try:
                await source.stop()
            except Exception:
                logger.exception("Failed to stop event source: %s", source.name)
