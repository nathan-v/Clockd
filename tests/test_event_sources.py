"""Tests for the event source system (base, manager, unifi_protect)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clockd.config import (
    CalibrationConfig,
    CameraConfig,
    ServerConfig,
    UnifiEventSourceConfig,
    UnifiProtectConfig,
)
from clockd.services.event_sources.base import EventSource
from clockd.services.event_sources.manager import EventSourceManager, create_event_source
from clockd.services.event_sources.unifi_protect import UnifiProtectSource


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_camera(camera_id: str = "test_cam") -> CameraConfig:
    return CameraConfig(
        camera_id=camera_id,
        calibration=CalibrationConfig(
            source_points=[[100, 400], [700, 400], [900, 700], [50, 700]],
            target_width_m=8.0,
            target_height_m=40.0,
        ),
    )


def _make_server_cfg(tmp_path) -> ServerConfig:
    return ServerConfig(
        upload_dir=str(tmp_path / "uploads"),
        cameras_dir=str(tmp_path / "cameras"),
    )


def _make_source_config(
    camera_map: dict[str, str] | None = None,
    enabled: bool = True,
) -> UnifiEventSourceConfig:
    return UnifiEventSourceConfig(
        enabled=enabled,
        camera_map=camera_map or {"protect_cam_1": "test_cam"},
        unit="mph",
        unifi=UnifiProtectConfig(
            host="10.0.0.1",
            username="user",
            password="pass",
            poll_interval_s=1,
            lookback_s=60,
            event_end_timeout_s=5,
        ),
    )


def _make_protect_event(
    event_id: str = "evt1",
    camera: str = "protect_cam_1",
    start: int = 1000000,
    end: int | None = 1010000,
    smart_types: list[str] | None = None,
) -> dict:
    return {
        "id": event_id,
        "camera": camera,
        "start": start,
        "end": end,
        "type": "smartDetectZone",
        "smartDetectTypes": smart_types or ["vehicle"],
    }


# ── EventSource ABC ───────────────────────────────────────────────────────


class DummySource(EventSource):
    def __init__(self):
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    @property
    def name(self):
        return "dummy"


@pytest.mark.asyncio
async def test_event_source_abc_contract():
    source = DummySource()
    assert source.name == "dummy"
    await source.start()
    assert source.started
    await source.stop()
    assert source.stopped


# ── EventSourceManager ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_manager_start_stop():
    mgr = EventSourceManager()
    s1, s2 = DummySource(), DummySource()
    mgr.add(s1)
    mgr.add(s2)

    await mgr.start_all()
    assert s1.started and s2.started

    await mgr.stop_all()
    assert s1.stopped and s2.stopped


@pytest.mark.asyncio
async def test_manager_start_survives_exception():
    mgr = EventSourceManager()
    bad = DummySource()
    bad.start = AsyncMock(side_effect=RuntimeError("boom"))
    good = DummySource()
    mgr.add(bad)
    mgr.add(good)

    await mgr.start_all()
    assert good.started  # good source still started despite bad one failing


def test_create_event_source_unifi(tmp_path):
    config = _make_source_config()
    cameras = {"test_cam": _make_camera()}
    server_cfg = _make_server_cfg(tmp_path)
    job_manager = MagicMock()

    source = create_event_source("test", config, cameras, server_cfg, job_manager)
    assert isinstance(source, UnifiProtectSource)
    assert source.name == "unifi_protect:test"


def test_create_event_source_unknown_raises(tmp_path):
    config = MagicMock(spec=[])  # no 'unifi' attribute
    with pytest.raises(ValueError, match="Unknown event source"):
        create_event_source("test", config, {}, _make_server_cfg(tmp_path), MagicMock())


# ── UnifiProtectSource ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_protect_source_start_stop(tmp_path):
    config = _make_source_config()
    source = UnifiProtectSource(
        "test",
        config,
        {"test_cam": _make_camera()},
        _make_server_cfg(tmp_path),
        MagicMock(),
    )

    # Patch _poll_loop to avoid real HTTP calls
    with patch.object(source, "_poll_loop", new_callable=AsyncMock):
        await source.start()
        assert source._task is not None
        await source.stop()
        assert source._task.done()


@pytest.mark.asyncio
async def test_protect_poll_events_filters_non_vehicle(tmp_path):
    config = _make_source_config()
    cameras = {"test_cam": _make_camera()}
    source = UnifiProtectSource(
        "test",
        config,
        cameras,
        _make_server_cfg(tmp_path),
        MagicMock(),
    )

    person_event = _make_protect_event(smart_types=["person"])
    vehicle_event = _make_protect_event(event_id="evt2", smart_types=["vehicle"])
    api_response = json.dumps([person_event, vehicle_event]).encode()

    with patch.object(source, "_api_get", new_callable=AsyncMock, return_value=api_response):
        with patch.object(source, "_handle_event", new_callable=AsyncMock):
            await source._poll_events(0, 9999999999999)
            # Give the spawned task a chance to run
            await asyncio.sleep(0.05)

    # Only the vehicle event should have been handled
    assert "evt2" in source._seen_event_ids
    assert "evt1" not in source._seen_event_ids


@pytest.mark.asyncio
async def test_protect_poll_events_skips_duplicates(tmp_path):
    config = _make_source_config()
    cameras = {"test_cam": _make_camera()}
    source = UnifiProtectSource(
        "test",
        config,
        cameras,
        _make_server_cfg(tmp_path),
        MagicMock(),
    )
    source._seen_event_ids.add("evt1")

    event = _make_protect_event(event_id="evt1")
    api_response = json.dumps([event]).encode()

    with patch.object(source, "_api_get", new_callable=AsyncMock, return_value=api_response):
        with patch.object(source, "_handle_event", new_callable=AsyncMock) as mock_handle:
            await source._poll_events(0, 9999999999999)
            await asyncio.sleep(0.05)

    mock_handle.assert_not_called()


@pytest.mark.asyncio
async def test_protect_poll_events_skips_unmapped_camera(tmp_path):
    config = _make_source_config(camera_map={"other_cam": "test_cam"})
    cameras = {"test_cam": _make_camera()}
    source = UnifiProtectSource(
        "test",
        config,
        cameras,
        _make_server_cfg(tmp_path),
        MagicMock(),
    )

    event = _make_protect_event(camera="protect_cam_1")  # not in camera_map
    api_response = json.dumps([event]).encode()

    with patch.object(source, "_api_get", new_callable=AsyncMock, return_value=api_response):
        with patch.object(source, "_handle_event", new_callable=AsyncMock) as mock_handle:
            await source._poll_events(0, 9999999999999)
            await asyncio.sleep(0.05)

    mock_handle.assert_not_called()


@pytest.mark.asyncio
async def test_protect_handle_event_downloads_and_submits(tmp_path):
    config = _make_source_config()
    cameras = {"test_cam": _make_camera()}
    job_manager = MagicMock()
    job_manager.submit.return_value = "job-123"
    source = UnifiProtectSource(
        "test",
        config,
        cameras,
        _make_server_cfg(tmp_path),
        job_manager,
    )

    event = _make_protect_event(start=1000000, end=1010000)

    with patch.object(
        source, "_download_clip", new_callable=AsyncMock, return_value="/tmp/clip.mp4"
    ):
        await source._handle_event(event, cameras["test_cam"])

    job_manager.submit.assert_called_once()
    call_args = job_manager.submit.call_args
    assert call_args[0][0] == "/tmp/clip.mp4"
    assert call_args[0][1].camera_id == "test_cam"


@pytest.mark.asyncio
async def test_protect_handle_event_waits_for_end(tmp_path):
    config = _make_source_config()
    cameras = {"test_cam": _make_camera()}
    job_manager = MagicMock()
    job_manager.submit.return_value = "job-123"
    source = UnifiProtectSource(
        "test",
        config,
        cameras,
        _make_server_cfg(tmp_path),
        job_manager,
    )

    event_no_end = _make_protect_event(start=1000000, end=None)

    async def mock_wait(event_id, camera_id):
        return 1010000  # event ended

    with patch.object(source, "_wait_for_event_end", side_effect=mock_wait):
        with patch.object(
            source, "_download_clip", new_callable=AsyncMock, return_value="/tmp/clip.mp4"
        ):
            await source._handle_event(event_no_end, cameras["test_cam"])

    job_manager.submit.assert_called_once()


@pytest.mark.asyncio
async def test_protect_handle_event_timeout_skips(tmp_path):
    config = _make_source_config()
    cameras = {"test_cam": _make_camera()}
    job_manager = MagicMock()
    source = UnifiProtectSource(
        "test",
        config,
        cameras,
        _make_server_cfg(tmp_path),
        job_manager,
    )

    event_no_end = _make_protect_event(start=1000000, end=None)

    with patch.object(source, "_wait_for_event_end", new_callable=AsyncMock, return_value=None):
        await source._handle_event(event_no_end, cameras["test_cam"])

    job_manager.submit.assert_not_called()


@pytest.mark.asyncio
async def test_protect_handle_event_queue_full(tmp_path):
    config = _make_source_config()
    cameras = {"test_cam": _make_camera()}
    job_manager = MagicMock()
    job_manager.submit.side_effect = RuntimeError("queue full")
    source = UnifiProtectSource(
        "test",
        config,
        cameras,
        _make_server_cfg(tmp_path),
        job_manager,
    )

    event = _make_protect_event()

    with patch.object(
        source, "_download_clip", new_callable=AsyncMock, return_value="/tmp/clip.mp4"
    ):
        with patch("clockd.services.event_sources.unifi_protect._safe_remove") as mock_rm:
            await source._handle_event(event, cameras["test_cam"])
            mock_rm.assert_called_once_with("/tmp/clip.mp4")


def _mock_urlopen(data: bytes):
    """Create a mock that simulates urllib opener.open() returning chunked data."""
    resp = MagicMock()
    chunks = [data[i : i + 65536] for i in range(0, len(data), 65536)] + [b""]
    resp.read = MagicMock(side_effect=chunks)
    resp.close = MagicMock()
    opener = MagicMock()
    opener.open = MagicMock(return_value=resp)
    return opener


@pytest.mark.asyncio
async def test_protect_download_clip_writes_file(tmp_path):
    config = _make_source_config()
    server_cfg = _make_server_cfg(tmp_path)
    source = UnifiProtectSource(
        "test",
        config,
        {},
        server_cfg,
        MagicMock(),
    )
    source._session_cookie = "fake=cookie"

    fake_video = b"\x00" * 5000
    opener = _mock_urlopen(fake_video)

    with patch("urllib.request.build_opener", return_value=opener):
        path = await source._download_clip("cam1", 1000, 2000)

    assert path is not None
    assert path.endswith(".mp4")
    with open(path, "rb") as f:
        assert len(f.read()) == 5000


@pytest.mark.asyncio
async def test_protect_download_clip_rejects_small(tmp_path):
    config = _make_source_config()
    server_cfg = _make_server_cfg(tmp_path)
    source = UnifiProtectSource(
        "test",
        config,
        {},
        server_cfg,
        MagicMock(),
    )
    source._session_cookie = "fake=cookie"

    opener = _mock_urlopen(b"tiny")

    with patch("urllib.request.build_opener", return_value=opener):
        path = await source._download_clip("cam1", 1000, 2000)

    assert path is None


@pytest.mark.asyncio
async def test_protect_seen_ids_pruning(tmp_path):
    config = _make_source_config()
    source = UnifiProtectSource(
        "test",
        config,
        {"test_cam": _make_camera()},
        _make_server_cfg(tmp_path),
        MagicMock(),
    )

    # Fill with 10001 IDs
    source._seen_event_ids = {f"evt_{i}" for i in range(10_001)}

    events = [_make_protect_event(event_id="new_evt")]
    api_response = json.dumps(events).encode()

    with patch.object(source, "_api_get", new_callable=AsyncMock, return_value=api_response):
        with patch.object(source, "_handle_event", new_callable=AsyncMock):
            await source._poll_events(0, 9999999999999)

    assert len(source._seen_event_ids) <= 5_002  # pruned to 5000 + new ones


# ── Config model tests ────────────────────────────────────────────────────


def test_unifi_event_source_config_defaults():
    cfg = UnifiEventSourceConfig()
    assert cfg.enabled is False
    assert cfg.camera_map == {}
    assert cfg.unit == "mph"
    assert cfg.unifi.poll_interval_s == 30
    assert cfg.unifi.smart_detect_types == ["vehicle"]
    assert cfg.unifi.verify_ssl is False


def test_unifi_event_source_config_from_dict():
    cfg = UnifiEventSourceConfig(
        enabled=True,
        camera_map={"abc": "front_yard"},
        unifi={"host": "10.0.0.1", "username": "u", "password": "p"},
    )
    assert cfg.enabled is True
    assert cfg.unifi.host == "10.0.0.1"
    assert cfg.camera_map == {"abc": "front_yard"}


def test_server_config_event_sources():
    cfg = ServerConfig(
        event_sources={
            "nvr1": UnifiEventSourceConfig(
                enabled=True,
                camera_map={"cam1": "yard"},
                unifi=UnifiProtectConfig(host="10.0.0.1"),
            )
        }
    )
    assert "nvr1" in cfg.event_sources
    assert cfg.event_sources["nvr1"].unifi.host == "10.0.0.1"


def test_server_config_no_event_sources_by_default():
    cfg = ServerConfig()
    assert cfg.event_sources == {}
