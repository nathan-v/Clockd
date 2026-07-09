"""
Local-only validation test against the Roboflow vehicles.mp4 sample video.

This test downloads a real traffic video and runs the full Clockd pipeline
to verify end-to-end speed estimation produces plausible results.

Run with:  pytest -m local -s
"""

from __future__ import annotations

import os
import urllib.request

import pytest

from clockd.config import (
    CalibrationConfig,
    CameraConfig,
    ServerConfig,
    SpeedRange,
)
from clockd.services.pipeline import process_video

# Safety net: skip if someone runs the file directly without -m local
_skip_unless_local = pytest.mark.skipif(
    not os.environ.get("CLOCKD_RUN_LOCAL_TESTS"),
    reason="Local-only test: run with `pytest -m local` or set CLOCKD_RUN_LOCAL_TESTS=1",
)

VIDEO_URL = "https://media.roboflow.com/supervision/video-examples/vehicles.mp4"
VIDEO_CACHE_PATH = "/tmp/clockd_test_vehicles.mp4"

# Known Roboflow calibration for this video
SOURCE_POINTS = [
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159],
]
TARGET_WIDTH_M = 25.0
TARGET_HEIGHT_M = 250.0


def _ensure_video() -> str:
    """Download the sample video if not already cached."""
    if os.path.exists(VIDEO_CACHE_PATH):
        size = os.path.getsize(VIDEO_CACHE_PATH)
        if size > 1_000_000:  # must be at least 1 MB to be valid
            return VIDEO_CACHE_PATH

    print(f"\nDownloading test video from {VIDEO_URL} ...")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_CACHE_PATH)
    print(f"Saved to {VIDEO_CACHE_PATH} ({os.path.getsize(VIDEO_CACHE_PATH):,} bytes)")
    return VIDEO_CACHE_PATH


def _build_camera_config() -> CameraConfig:
    """Build a CameraConfig using the known Roboflow calibration."""
    return CameraConfig(
        camera_id="roboflow-test",
        description="Roboflow vehicles.mp4 validation camera",
        calibration=CalibrationConfig(
            source_points=SOURCE_POINTS,
            target_width_m=TARGET_WIDTH_M,
            target_height_m=TARGET_HEIGHT_M,
        ),
        min_detections=10,
        smoothing_window=5,
        speed_range=SpeedRange(min_mph=5.0, max_mph=120.0),
    )


@pytest.mark.local
@_skip_unless_local
class TestRoboflowVehicleValidation:
    """End-to-end validation against the Roboflow vehicles.mp4 sample video."""

    @pytest.fixture(scope="class")
    def result(self):
        """Run the pipeline once and share the result across all tests in this class."""
        video_path = _ensure_video()
        camera = _build_camera_config()
        server_cfg = ServerConfig()

        print("\nRunning Clockd pipeline on vehicles.mp4 ...")
        processing_result = process_video(
            video_path=video_path,
            camera=camera,
            server_cfg=server_cfg,
            unit="mph",
        )

        # Print summary for manual review
        print(f"\n{'=' * 60}")
        print("VALIDATION RESULTS")
        print(f"{'=' * 60}")
        print(f"  Camera ID:        {processing_result.camera_id}")
        print(f"  Video FPS:        {processing_result.fps}")
        print(f"  Total Frames:     {processing_result.total_frames}")
        print(f"  Duration:         {processing_result.duration_s}s")
        print(f"  Processing Time:  {processing_result.processing_time_s}s")
        print(f"  Vehicles Found:   {len(processing_result.vehicles)}")
        print(f"  Vehicles Filtered:{processing_result.vehicles_filtered}")
        if processing_result.warnings:
            print(f"  Warnings:         {processing_result.warnings}")

        print(
            f"\n  {'ID':>4}  {'Avg':>7}  {'Min':>7}  {'Max':>7}  {'Conf':>5}  {'Dets':>4}  Frames"
        )
        print(f"  {'--':>4}  {'---':>7}  {'---':>7}  {'---':>7}  {'----':>5}  {'----':>4}  ------")
        for v in processing_result.vehicles:
            print(
                f"  {v.track_id:4d}  {v.speed_avg:6.1f}{v.unit}  "
                f"{v.speed_min:6.1f}{v.unit}  {v.speed_max:6.1f}{v.unit}  "
                f"{v.mean_detection_confidence:.3f}  {v.num_detections:4d}  "
                f"{v.first_seen_frame}-{v.last_seen_frame}"
            )
        print(f"{'=' * 60}\n")

        return processing_result

    def test_at_least_one_vehicle_detected(self, result):
        """The video contains visible highway traffic; at least 1 vehicle must be found."""
        assert len(result.vehicles) >= 1, f"Expected at least 1 vehicle, got {len(result.vehicles)}"

    def test_speeds_are_plausible(self, result):
        """All reported speeds must be between 5 and 120 mph (highway plausibility)."""
        for v in result.vehicles:
            assert 5.0 <= v.speed_avg <= 120.0, (
                f"Vehicle {v.track_id} avg speed {v.speed_avg} mph is outside [5, 120]"
            )

    def test_detection_confidence_above_threshold(self, result):
        """All vehicles must have mean detection confidence > 0.3."""
        for v in result.vehicles:
            assert v.mean_detection_confidence > 0.3, (
                f"Vehicle {v.track_id} mean confidence {v.mean_detection_confidence} <= 0.3"
            )

    def test_processing_time_positive(self, result):
        """Pipeline must report a positive processing time."""
        assert result.processing_time_s > 0, (
            f"processing_time_s should be > 0, got {result.processing_time_s}"
        )

    def test_video_metadata_positive(self, result):
        """FPS and total_frames must be positive values."""
        assert result.fps > 0, f"fps should be > 0, got {result.fps}"
        assert result.total_frames > 0, f"total_frames should be > 0, got {result.total_frames}"

    def test_video_within_limits(self, result):
        """The Roboflow video (~1400 frames, ~47s) must be within the configured limits.

        MAX_FRAMES = 18000, MAX_DURATION_S = 300 — the test video should pass easily.
        """
        from clockd.utils.video import MAX_DURATION_S, MAX_FRAMES

        assert result.total_frames <= MAX_FRAMES, (
            f"total_frames {result.total_frames} exceeds MAX_FRAMES {MAX_FRAMES}"
        )
        assert result.duration_s <= MAX_DURATION_S, (
            f"duration_s {result.duration_s} exceeds MAX_DURATION_S {MAX_DURATION_S}"
        )
