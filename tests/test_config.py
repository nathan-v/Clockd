from __future__ import annotations

import yaml

from clockd.config import load_cameras, load_server_config


def test_load_server_config_defaults(tmp_path):
    path = str(tmp_path / "nonexistent.yaml")
    cfg = load_server_config(path)
    assert cfg.port == 8000
    assert cfg.model == "yolo26n.pt"


def test_load_server_config_from_file(tmp_path):
    path = tmp_path / "server.yaml"
    path.write_text(yaml.dump({"port": 9000, "model": "yolov8m.pt"}))
    cfg = load_server_config(str(path))
    assert cfg.port == 9000
    assert cfg.model == "yolov8m.pt"


def test_load_cameras(tmp_path):
    cam_dir = tmp_path / "cameras"
    cam_dir.mkdir()
    (cam_dir / "test.yaml").write_text(
        yaml.dump(
            {
                "camera_id": "test1",
                "description": "Test",
                "calibration": {
                    "source_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "target_width_m": 5.0,
                    "target_height_m": 10.0,
                },
            }
        )
    )
    cameras = load_cameras(str(cam_dir))
    assert "test1" in cameras
    assert cameras["test1"].calibration.target_width_m == 5.0


def test_load_cameras_empty_dir(tmp_path):
    cam_dir = tmp_path / "empty"
    cam_dir.mkdir()
    cameras = load_cameras(str(cam_dir))
    assert cameras == {}
