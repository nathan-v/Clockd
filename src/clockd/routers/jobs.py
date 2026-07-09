from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from clockd.models import JobInfo

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobInfo])
async def list_jobs(request: Request) -> list[JobInfo]:
    job_mgr = request.app.state.job_manager
    return job_mgr.list_jobs()


@router.get("/{job_id}", response_model=JobInfo)
async def get_job(job_id: str, request: Request) -> JobInfo:
    job_mgr = request.app.state.job_manager
    job = job_mgr.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.delete("/{job_id}")
async def delete_job(job_id: str, request: Request) -> dict:
    job_mgr = request.app.state.job_manager
    job = job_mgr.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if not job_mgr.delete(job_id):
        raise HTTPException(status_code=409, detail="Can only delete completed or failed jobs")
    return {"deleted": job_id}
