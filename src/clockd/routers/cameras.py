from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from clockd.config import CameraConfig, delete_camera_file, save_camera

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=list[CameraConfig])
async def list_cameras(request: Request) -> list[CameraConfig]:
    cameras: dict[str, CameraConfig] = request.app.state.cameras
    return list(cameras.values())


@router.get("/{camera_id}", response_model=CameraConfig)
async def get_camera(camera_id: str, request: Request) -> CameraConfig:
    cameras: dict[str, CameraConfig] = request.app.state.cameras
    cam = cameras.get(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    return cam


@router.post("", response_model=CameraConfig, status_code=201)
async def create_camera(camera: CameraConfig, request: Request) -> CameraConfig:
    cameras: dict[str, CameraConfig] = request.app.state.cameras
    if camera.camera_id in cameras:
        raise HTTPException(status_code=409, detail=f"Camera '{camera.camera_id}' already exists")
    max_cameras = request.app.state.server_cfg.max_cameras
    if len(cameras) >= max_cameras:
        raise HTTPException(
            status_code=409,
            detail=f"Camera limit reached ({max_cameras}); raise max_cameras in the server config",
        )
    cameras_dir = request.app.state.server_cfg.cameras_dir
    save_camera(cameras_dir, camera)
    cameras[camera.camera_id] = camera
    return camera


@router.put("/{camera_id}", response_model=CameraConfig)
async def update_camera(camera_id: str, camera: CameraConfig, request: Request) -> CameraConfig:
    cameras: dict[str, CameraConfig] = request.app.state.cameras
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    cameras_dir = request.app.state.server_cfg.cameras_dir
    # If camera_id in the path differs from the body, rename the file — but
    # never overwrite a different existing camera in the process.
    if camera.camera_id != camera_id:
        if camera.camera_id in cameras:
            raise HTTPException(
                status_code=409, detail=f"Camera '{camera.camera_id}' already exists"
            )
        delete_camera_file(cameras_dir, camera_id)
        del cameras[camera_id]
    save_camera(cameras_dir, camera)
    cameras[camera.camera_id] = camera
    return camera


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: str, request: Request) -> None:
    cameras: dict[str, CameraConfig] = request.app.state.cameras
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    cameras_dir = request.app.state.server_cfg.cameras_dir
    delete_camera_file(cameras_dir, camera_id)
    del cameras[camera_id]
