from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid
from http.cookiejar import CookieJar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from clockd.config import CameraConfig, ServerConfig, UnifiEventSourceConfig

from clockd.services.event_sources.base import EventSource

logger = logging.getLogger(__name__)


class UnifiProtectSource(EventSource):
    """Polls UniFi Protect for vehicle smart-detection events, downloads clips,
    and submits them to the Clockd processing pipeline via JobManager."""

    def __init__(
        self,
        source_name: str,
        config: UnifiEventSourceConfig,
        cameras: dict[str, CameraConfig],
        server_cfg: ServerConfig,
        job_manager: "object",
    ) -> None:
        self._source_name = source_name
        self._config = config
        self._protect = config.unifi
        self._cameras = cameras
        self._server_cfg = server_cfg
        self._job_manager = job_manager
        self._seen_event_ids: set[str] = set()
        self._session_cookie: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(server_cfg.max_workers)
        self._ssl_ctx = self._make_ssl_context()

    @property
    def name(self) -> str:
        return f"unifi_protect:{self._source_name}"

    def _make_ssl_context(self) -> ssl.SSLContext:
        ctx = ssl.create_default_context()
        if not self._protect.verify_ssl:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def start(self) -> None:
        self._task = asyncio.create_task(self._poll_loop(), name=f"eventsource-{self.name}")
        logger.info(
            "Event source started: %s (poll every %ds)", self.name, self._protect.poll_interval_s
        )

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Event source stopped: %s", self.name)

    # ── HTTP helpers ───────────────────────────────────────────────────────

    async def _authenticate(self) -> None:
        def _do_auth() -> str:
            url = f"https://{self._protect.host}/api/auth/login"
            payload = json.dumps(
                {"username": self._protect.username, "password": self._protect.password}
            ).encode()
            req = urllib.request.Request(url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            cj = CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(cj),
                urllib.request.HTTPSHandler(context=self._ssl_ctx),
            )
            resp = opener.open(req, timeout=10)
            resp.close()
            return "; ".join(f"{c.name}={c.value}" for c in cj)

        self._session_cookie = await asyncio.get_event_loop().run_in_executor(None, _do_auth)
        logger.info("Authenticated to Protect at %s", self._protect.host)

    async def _api_get(self, path: str, timeout: int = 30) -> bytes:
        if not self._session_cookie:
            await self._authenticate()

        def _do_get() -> bytes:
            url = f"https://{self._protect.host}{path}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Cookie", self._session_cookie)  # type: ignore[arg-type]
            handler = urllib.request.HTTPSHandler(context=self._ssl_ctx)
            opener = urllib.request.build_opener(handler)
            resp = opener.open(req, timeout=timeout)
            data = resp.read()
            resp.close()
            return data

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _do_get)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                self._session_cookie = None
                await self._authenticate()
                return await asyncio.get_event_loop().run_in_executor(None, _do_get)
            raise

    # ── Poll loop ──────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        last_poll_ms = int((time.time() - self._protect.lookback_s) * 1000)
        while True:
            try:
                now_ms = int(time.time() * 1000)
                await self._poll_events(last_poll_ms, now_ms)
                last_poll_ms = now_ms
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Event source %s poll error", self.name)
            await asyncio.sleep(self._protect.poll_interval_s)

    async def _poll_events(self, start_ms: int, end_ms: int) -> None:
        camera_ids = ",".join(self._config.camera_map.keys())
        if not camera_ids:
            return

        path = (
            f"/proxy/protect/api/events"
            f"?start={start_ms}&end={end_ms}"
            f"&cameras={camera_ids}"
            f"&types=smartDetectZone"
        )
        data = await self._api_get(path)
        events = json.loads(data)

        for event in events:
            event_id = event.get("id")
            if not event_id or event_id in self._seen_event_ids:
                continue

            smart_types = event.get("smartDetectTypes", [])
            if not any(t in self._protect.smart_detect_types for t in smart_types):
                continue

            protect_camera_id = event.get("camera")
            clockd_camera_id = self._config.camera_map.get(protect_camera_id)
            if not clockd_camera_id:
                continue

            camera = self._cameras.get(clockd_camera_id)
            if not camera:
                logger.warning(
                    "%s: camera_map references unknown clockd camera '%s'",
                    self.name,
                    clockd_camera_id,
                )
                continue

            self._seen_event_ids.add(event_id)
            asyncio.create_task(
                self._handle_event(event, camera),
                name=f"eventsource-event-{event_id}",
            )

        # Prune to avoid unbounded growth
        if len(self._seen_event_ids) > 10_000:
            self._seen_event_ids = set(list(self._seen_event_ids)[-5_000:])

    # ── Event handling ─────────────────────────────────────────────────────

    async def _handle_event(self, event: dict, camera: "CameraConfig") -> None:
        async with self._semaphore:
            await self._process_event(event, camera)

    async def _process_event(self, event: dict, camera: "CameraConfig") -> None:
        event_id = event["id"]
        protect_camera_id = event["camera"]
        start_ms = event.get("start")
        end_ms = event.get("end")

        if not end_ms:
            end_ms = await self._wait_for_event_end(event_id, protect_camera_id)
            if not end_ms:
                logger.warning("%s: event %s timed out waiting for end", self.name, event_id)
                return

        duration_s = (end_ms - start_ms) / 1000
        logger.info(
            "%s: processing event %s on %s (%.1fs clip)",
            self.name,
            event_id,
            camera.camera_id,
            duration_s,
        )

        # Add buffer around the event
        buffer_ms = 2000
        video_path = await self._download_clip(
            protect_camera_id, start_ms - buffer_ms, end_ms + buffer_ms
        )
        if not video_path:
            return

        try:
            job_id = self._job_manager.submit(  # type: ignore[attr-defined]
                video_path, camera, self._server_cfg, self._config.unit
            )
            logger.info("%s: submitted job %s for event %s", self.name, job_id, event_id)
        except RuntimeError:
            logger.warning("%s: job queue full, dropping event %s", self.name, event_id)
            _safe_remove(video_path)
        except Exception:
            logger.exception("%s: failed to submit event %s", self.name, event_id)
            _safe_remove(video_path)

    async def _wait_for_event_end(self, event_id: str, camera_id: str) -> Optional[int]:
        deadline = time.time() + self._protect.event_end_timeout_s
        while time.time() < deadline:
            await asyncio.sleep(2)
            try:
                now_ms = int(time.time() * 1000)
                lookback_ms = now_ms - 600_000  # 10 minutes
                path = (
                    f"/proxy/protect/api/events"
                    f"?start={lookback_ms}&end={now_ms}"
                    f"&cameras={camera_id}&types=smartDetectZone"
                )
                data = await self._api_get(path)
                for ev in json.loads(data):
                    if ev.get("id") == event_id and ev.get("end"):
                        return ev["end"]
            except Exception:
                logger.debug("Error checking event %s end status", event_id)
        return None

    async def _download_clip(self, camera_id: str, start_ms: int, end_ms: int) -> Optional[str]:
        url_path = (
            f"/proxy/protect/api/video/export?camera={camera_id}&start={start_ms}&end={end_ms}"
        )
        max_bytes = self._server_cfg.max_upload_mb * 1024 * 1024
        upload_dir = self._server_cfg.upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        dest = os.path.join(upload_dir, f"protect_{uuid.uuid4().hex[:12]}.mp4")

        if not self._session_cookie:
            await self._authenticate()

        def _do_download() -> int:
            url = f"https://{self._protect.host}{url_path}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Cookie", self._session_cookie)  # type: ignore[arg-type]
            handler = urllib.request.HTTPSHandler(context=self._ssl_ctx)
            opener = urllib.request.build_opener(handler)
            resp = opener.open(req, timeout=120)
            written = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        resp.close()
                        raise ValueError(f"Clip exceeds {self._server_cfg.max_upload_mb}MB limit")
                    f.write(chunk)
            resp.close()
            return written

        try:
            written = await asyncio.get_event_loop().run_in_executor(None, _do_download)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                self._session_cookie = None
                await self._authenticate()
                try:
                    written = await asyncio.get_event_loop().run_in_executor(None, _do_download)
                except Exception:
                    logger.exception(
                        "%s: failed to download clip for camera %s", self.name, camera_id
                    )
                    _safe_remove(dest)
                    return None
            else:
                logger.exception("%s: failed to download clip for camera %s", self.name, camera_id)
                _safe_remove(dest)
                return None
        except Exception:
            logger.exception("%s: failed to download clip for camera %s", self.name, camera_id)
            _safe_remove(dest)
            return None

        if written < 1000:
            logger.warning("%s: clip too small (%d bytes), skipping", self.name, written)
            _safe_remove(dest)
            return None

        return dest


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
