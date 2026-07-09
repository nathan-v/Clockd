from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from clockd.config import CalibrationConfig, CameraConfig, ServerConfig
from clockd.models import JobStatus
from clockd.services.job_manager import MAX_QUEUED_JOBS, JobManager


def _make_camera() -> CameraConfig:
    return CameraConfig(
        camera_id="test",
        calibration=CalibrationConfig(
            source_points=[[0, 0], [1, 0], [1, 1], [0, 1]],
            target_width_m=5.0,
            target_height_m=10.0,
        ),
    )


def _make_cfg(tmp_path) -> ServerConfig:
    return ServerConfig(upload_dir=str(tmp_path), cameras_dir=str(tmp_path))


def _make_video_file(tmp_path) -> str:
    path = str(tmp_path / "test.mp4")
    with open(path, "wb") as f:
        f.write(b"fake video data")
    return path


def test_submit_and_get(tmp_path):
    mgr = JobManager(max_workers=1, ttl_seconds=3600)
    video = _make_video_file(tmp_path)

    with patch("clockd.services.job_manager.process_video") as mock_pv:
        from clockd.models import ProcessingResult

        mock_pv.return_value = ProcessingResult(
            camera_id="test",
            video_filename="test.mp4",
            fps=30.0,
            total_frames=90,
            duration_s=3.0,
            unit="mph",
            vehicles=[],
            processing_time_s=0.1,
        )
        job_id = mgr.submit(video, _make_camera(), _make_cfg(tmp_path), "mph")

    # Wait for completion
    for _ in range(50):
        job = mgr.get(job_id)
        if job and job.status in (JobStatus.completed, JobStatus.failed):
            break
        time.sleep(0.1)

    job = mgr.get(job_id)
    assert job is not None
    assert job.status == JobStatus.completed
    assert job.result is not None
    assert job.result.camera_id == "test"
    mgr.shutdown()


def test_submit_failure_generic_error(tmp_path):
    mgr = JobManager(max_workers=1, ttl_seconds=3600)
    video = _make_video_file(tmp_path)

    with patch("clockd.services.job_manager.process_video", side_effect=ValueError("boom")):
        job_id = mgr.submit(video, _make_camera(), _make_cfg(tmp_path), "mph")

    for _ in range(50):
        job = mgr.get(job_id)
        if job and job.status == JobStatus.failed:
            break
        time.sleep(0.1)

    job = mgr.get(job_id)
    assert job.status == JobStatus.failed
    assert job.error == "Processing failed"  # sanitized, not "boom"
    mgr.shutdown()


def test_list_jobs(tmp_path):
    mgr = JobManager(max_workers=1, ttl_seconds=3600)
    assert mgr.list_jobs() == []

    video = _make_video_file(tmp_path)
    with patch("clockd.services.job_manager.process_video", side_effect=ValueError("fail")):
        mgr.submit(video, _make_camera(), _make_cfg(tmp_path), "mph")

    time.sleep(0.5)
    jobs = mgr.list_jobs()
    assert len(jobs) == 1
    mgr.shutdown()


def test_delete_completed_job(tmp_path):
    mgr = JobManager(max_workers=1, ttl_seconds=3600)
    video = _make_video_file(tmp_path)

    with patch("clockd.services.job_manager.process_video", side_effect=ValueError("fail")):
        job_id = mgr.submit(video, _make_camera(), _make_cfg(tmp_path), "mph")

    time.sleep(0.5)
    assert mgr.delete(job_id)
    assert mgr.get(job_id) is None
    mgr.shutdown()


def test_delete_nonexistent():
    mgr = JobManager()
    assert not mgr.delete("nonexistent")
    mgr.shutdown()


def test_delete_running_job_fails(tmp_path):
    import threading

    block = threading.Event()
    mgr = JobManager(max_workers=1, ttl_seconds=3600)
    video = _make_video_file(tmp_path)

    def blocking_process(*args, **kwargs):
        block.wait(timeout=5)
        raise ValueError("done")

    with patch("clockd.services.job_manager.process_video", side_effect=blocking_process):
        job_id = mgr.submit(video, _make_camera(), _make_cfg(tmp_path), "mph")
        time.sleep(0.2)
        assert not mgr.delete(job_id)  # can't delete running job
        block.set()

    time.sleep(0.5)
    mgr.shutdown()


def test_purge_expired(tmp_path):
    mgr = JobManager(max_workers=1, ttl_seconds=0)  # 0s TTL = expire immediately
    video = _make_video_file(tmp_path)

    with patch("clockd.services.job_manager.process_video", side_effect=ValueError("fail")):
        mgr.submit(video, _make_camera(), _make_cfg(tmp_path), "mph")

    time.sleep(0.5)
    # list_jobs triggers purge
    jobs = mgr.list_jobs()
    assert len(jobs) == 0  # should be purged
    mgr.shutdown()


def test_max_queue_limit(tmp_path):
    import threading

    block = threading.Event()
    mgr = JobManager(max_workers=1, ttl_seconds=3600)

    def blocking_process(*args, **kwargs):
        block.wait(timeout=10)

    with patch("clockd.services.job_manager.process_video", side_effect=blocking_process):
        # Fill up the queue
        for _ in range(MAX_QUEUED_JOBS):
            video = _make_video_file(tmp_path)
            mgr.submit(video, _make_camera(), _make_cfg(tmp_path), "mph")

        # Next submit should fail
        video = _make_video_file(tmp_path)
        with pytest.raises(RuntimeError, match="Too many jobs"):
            mgr.submit(video, _make_camera(), _make_cfg(tmp_path), "mph")

    block.set()
    time.sleep(0.5)
    mgr.shutdown()


def test_metrics_integration(tmp_path):
    from clockd.services.metrics import MetricsService
    from clockd.config import MetricsConfig

    metrics = MetricsService(MetricsConfig())
    mgr = JobManager(max_workers=1, ttl_seconds=3600, metrics=metrics)
    video = _make_video_file(tmp_path)

    with patch("clockd.services.job_manager.process_video", side_effect=ValueError("fail")):
        mgr.submit(video, _make_camera(), _make_cfg(tmp_path), "mph")

    time.sleep(0.5)
    mgr.shutdown()
    # Just verify it didn't crash — metrics with disabled prometheus are no-ops
