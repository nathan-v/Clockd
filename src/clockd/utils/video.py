from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2


from fastapi import HTTPException, UploadFile

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts", ".m4v"}

MAX_DURATION_S = 300  # 5 minutes
MAX_FRAMES = 18000  # 10 min @ 30fps


async def read_upload_with_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Stream upload in chunks, abort early if too large. Returns bytes for image uploads."""
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="Upload exceeds size limit")
        chunks.append(chunk)
    return b"".join(chunks)


async def stream_upload_to_disk(file: UploadFile, upload_dir: str, max_bytes: int) -> str:
    """Stream upload directly to disk without holding it all in memory."""
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "upload.mp4").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        ext = ".mp4"
    dest = os.path.join(upload_dir, f"{uuid.uuid4().hex}{ext}")
    total = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="Upload exceeds size limit")
                f.write(chunk)
    except Exception:
        cleanup(dest)
        raise
    return dest


def validate_video(path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or total_frames <= 0:
        raise ValueError("Invalid video: could not read FPS or frame count")
    if total_frames > MAX_FRAMES:
        raise ValueError(f"Video has {total_frames} frames, max is {MAX_FRAMES}")
    duration = total_frames / fps
    if duration > MAX_DURATION_S:
        raise ValueError(f"Video is {duration:.0f}s, max is {MAX_DURATION_S}s")
    return fps, total_frames


def cleanup(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
