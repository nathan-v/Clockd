from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

import logging

from clockd.config import CameraConfig, ServerConfig
from clockd.models import JobInfo, JobStatus
from clockd.services.pipeline import process_video
from clockd.utils.video import cleanup

logger = logging.getLogger(__name__)

MAX_QUEUED_JOBS = 100

if TYPE_CHECKING:
    from clockd.services.metrics import MetricsService


class JobManager:
    def __init__(
        self,
        max_workers: int = 2,
        ttl_seconds: int = 3600,
        metrics: Optional[MetricsService] = None,
    ) -> None:
        self._jobs: dict[str, JobInfo] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._video_paths: dict[str, str] = {}
        self._finished_at: dict[str, float] = {}
        self._ttl = ttl_seconds
        self._metrics = metrics

    def submit(
        self,
        video_path: str,
        camera: CameraConfig,
        server_cfg: ServerConfig,
        unit: str,
    ) -> str:
        job_id = uuid.uuid4().hex
        job = JobInfo(job_id=job_id, status=JobStatus.pending)
        with self._lock:
            self._purge_expired()
            active = sum(
                1 for j in self._jobs.values() if j.status in (JobStatus.pending, JobStatus.running)
            )
            if active >= MAX_QUEUED_JOBS:
                raise RuntimeError("Too many jobs queued, try again later")
            self._jobs[job_id] = job
            self._video_paths[job_id] = video_path
        if self._metrics:
            self._metrics.record_job_submitted()
        self._pool.submit(self._run, job_id, video_path, camera, server_cfg, unit)
        return job_id

    def _run(
        self,
        job_id: str,
        video_path: str,
        camera: CameraConfig,
        server_cfg: ServerConfig,
        unit: str,
    ) -> None:
        verbose = server_cfg.verbose
        with self._lock:
            self._jobs[job_id].status = JobStatus.running

        if verbose:
            logger.info("Job %s running: camera=%s", job_id, camera.camera_id)

        def on_progress(p: float) -> None:
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].progress = round(min(p, 1.0), 3)

        try:
            result = process_video(video_path, camera, server_cfg, unit, progress_cb=on_progress)
            with self._lock:
                self._jobs[job_id].status = JobStatus.completed
                self._jobs[job_id].progress = 1.0
                self._jobs[job_id].result = result
                self._finished_at[job_id] = time.monotonic()
            if self._metrics:
                self._metrics.record(result)
            if verbose:
                logger.info(
                    "Job %s completed: camera=%s vehicles=%d time=%.2fs",
                    job_id,
                    camera.camera_id,
                    len(result.vehicles),
                    result.processing_time_s,
                )
        except Exception:
            logger.exception("Job %s failed", job_id)
            with self._lock:
                self._jobs[job_id].status = JobStatus.failed
                self._jobs[job_id].error = "Processing failed"
                self._finished_at[job_id] = time.monotonic()
        finally:
            if self._metrics:
                self._metrics.record_job_finished()
            cleanup(video_path)

    def get(self, job_id: str) -> Optional[JobInfo]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobInfo]:
        with self._lock:
            self._purge_expired()
            return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in (JobStatus.completed, JobStatus.failed):
                del self._jobs[job_id]
                self._video_paths.pop(job_id, None)
                self._finished_at.pop(job_id, None)
                return True
            return False

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [jid for jid, finished in self._finished_at.items() if now - finished > self._ttl]
        for jid in expired:
            del self._jobs[jid]
            self._video_paths.pop(jid, None)
            self._finished_at.pop(jid, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
