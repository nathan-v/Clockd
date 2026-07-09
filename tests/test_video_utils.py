from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from clockd.utils.video import (
    cleanup,
    read_upload_with_limit,
    stream_upload_to_disk,
    validate_video,
)


def _make_test_video(tmp_path, fps=30, frames=90, w=320, h=240):
    path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for _ in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_validate_video_valid(tmp_path):
    path = _make_test_video(tmp_path, fps=30, frames=90)
    fps, total = validate_video(path)
    assert fps == 30.0
    assert total == 90


def test_validate_video_not_a_video(tmp_path):
    path = str(tmp_path / "bad.mp4")
    with open(path, "wb") as f:
        f.write(b"not a video")
    with pytest.raises(ValueError, match="Cannot open video"):
        validate_video(path)


def test_validate_video_too_many_frames(tmp_path):
    # We can't easily make a video with 18000+ frames, so patch the constant
    path = _make_test_video(tmp_path, fps=30, frames=100)
    import clockd.utils.video as vid

    original = vid.MAX_FRAMES
    vid.MAX_FRAMES = 50
    try:
        with pytest.raises(ValueError, match="frames"):
            validate_video(path)
    finally:
        vid.MAX_FRAMES = original


def test_validate_video_too_long(tmp_path):
    path = _make_test_video(tmp_path, fps=30, frames=90)
    import clockd.utils.video as vid

    original = vid.MAX_DURATION_S
    vid.MAX_DURATION_S = 1
    try:
        with pytest.raises(ValueError, match="max is"):
            validate_video(path)
    finally:
        vid.MAX_DURATION_S = original


def test_cleanup_existing_file(tmp_path):
    path = str(tmp_path / "to_delete.txt")
    with open(path, "w") as f:
        f.write("test")
    cleanup(path)
    assert not os.path.exists(path)


def test_cleanup_nonexistent():
    cleanup("/nonexistent/path/file.txt")  # should not raise


@pytest.mark.asyncio
async def test_read_upload_with_limit():
    from unittest.mock import AsyncMock, MagicMock

    file = MagicMock()
    data = b"x" * 100
    chunks = [data[i : i + 30] for i in range(0, len(data), 30)]
    chunks.append(b"")
    file.read = AsyncMock(side_effect=chunks)

    result = await read_upload_with_limit(file, 200)
    assert len(result) == 100


@pytest.mark.asyncio
async def test_read_upload_with_limit_exceeds():
    from unittest.mock import AsyncMock, MagicMock

    file = MagicMock()
    file.read = AsyncMock(return_value=b"x" * 1024 * 1024)

    with pytest.raises(HTTPException) as exc_info:
        await read_upload_with_limit(file, 100)
    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_stream_upload_to_disk(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    upload_dir = str(tmp_path / "uploads")
    file = MagicMock()
    file.filename = "test.mp4"
    data = b"x" * 500
    file.read = AsyncMock(side_effect=[data, b""])

    path = await stream_upload_to_disk(file, upload_dir, 1000)
    assert os.path.exists(path)
    assert path.endswith(".mp4")
    with open(path, "rb") as f:
        assert f.read() == data


@pytest.mark.asyncio
async def test_stream_upload_to_disk_exceeds(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    upload_dir = str(tmp_path / "uploads")
    file = MagicMock()
    file.filename = "test.mp4"
    file.read = AsyncMock(return_value=b"x" * 1024 * 1024)

    with pytest.raises(HTTPException) as exc_info:
        await stream_upload_to_disk(file, upload_dir, 100)
    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_stream_upload_bad_extension(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    upload_dir = str(tmp_path / "uploads")
    file = MagicMock()
    file.filename = "malicious.php"
    file.read = AsyncMock(side_effect=[b"data", b""])

    path = await stream_upload_to_disk(file, upload_dir, 1000)
    assert path.endswith(".mp4")  # forced to safe extension
