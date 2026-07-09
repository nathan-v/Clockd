from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from clockd.models import ProcessingResult
from clockd.services.pipeline import process_video
from clockd.utils.video import cleanup, stream_upload_to_disk

logger = logging.getLogger(__name__)

router = APIRouter(tags=["process"])


@router.post("/process", response_model=ProcessingResult)
async def process_endpoint(
    request: Request,
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    unit: str = Form(None),
    async_mode: bool = Form(False),
) -> ProcessingResult | JSONResponse:
    cameras = request.app.state.cameras
    server_cfg = request.app.state.server_cfg

    cam = cameras.get(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    unit = unit or server_cfg.default_unit
    if unit not in ("mph", "kmh"):
        raise HTTPException(status_code=400, detail="unit must be 'mph' or 'kmh'")

    max_bytes = server_cfg.max_upload_mb * 1024 * 1024
    video_path = await stream_upload_to_disk(file, server_cfg.upload_dir, max_bytes)

    if server_cfg.verbose:
        logger.info(
            "Upload received: camera=%s file=%s async=%s unit=%s",
            camera_id,
            file.filename,
            async_mode,
            unit,
        )

    if async_mode:
        job_mgr = request.app.state.job_manager
        try:
            job_id = job_mgr.submit(video_path, cam, server_cfg, unit)
        except RuntimeError as exc:
            cleanup(video_path)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception:
            cleanup(video_path)
            raise
        if server_cfg.verbose:
            logger.info("Job submitted: job_id=%s camera=%s", job_id, camera_id)
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "pending"},
        )

    try:
        result = process_video(video_path, cam, server_cfg, unit)
    finally:
        cleanup(video_path)

    request.app.state.metrics.record(result)
    return result
