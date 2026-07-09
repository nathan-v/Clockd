from __future__ import annotations

import abc


class EventSource(abc.ABC):
    """Base class for NVR event sources that poll for new vehicle detection
    events and submit them to the Clockd processing pipeline."""

    @abc.abstractmethod
    async def start(self) -> None:
        """Start polling for events. Must not block — should create its own asyncio task."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop polling and clean up."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
