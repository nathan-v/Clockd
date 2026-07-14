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


def test_env_overrides_yaml(tmp_path, monkeypatch):
    path = tmp_path / "server.yaml"
    path.write_text(yaml.dump({"port": 9000}))
    monkeypatch.setenv("CLOCKD_PORT", "9100")
    cfg = load_server_config(str(path))
    assert cfg.port == 9100


def test_nested_env_secrets_merge_with_yaml(tmp_path, monkeypatch):
    path = tmp_path / "server.yaml"
    path.write_text(
        yaml.dump(
            {
                "metrics": {"influxdb_v2": {"enabled": True, "url": "http://influx:8086"}},
                "event_sources": {"home_nvr": {"enabled": True, "unifi": {"host": "10.0.0.1"}}},
            }
        )
    )
    monkeypatch.setenv("CLOCKD_METRICS__INFLUXDB_V2__TOKEN", "tok123")
    monkeypatch.setenv("CLOCKD_EVENT_SOURCES__HOME_NVR__UNIFI__PASSWORD", "s3cret")
    cfg = load_server_config(str(path))
    # Env-provided secrets land in the right nested fields...
    assert cfg.metrics.influxdb_v2.token == "tok123"
    assert cfg.event_sources["home_nvr"].unifi.password == "s3cret"
    # ...without clobbering the non-secret YAML values around them
    assert cfg.metrics.influxdb_v2.enabled is True
    assert cfg.metrics.influxdb_v2.url == "http://influx:8086"
    assert cfg.event_sources["home_nvr"].enabled is True
    assert cfg.event_sources["home_nvr"].unifi.host == "10.0.0.1"


def test_k8s_service_link_port_ignored(tmp_path, monkeypatch):
    # Kubernetes injects CLOCKD_PORT=tcp://<ip>:<port> when a Service named
    # "clockd" shares the namespace; it must not clobber the YAML port.
    path = tmp_path / "server.yaml"
    path.write_text(yaml.dump({"port": 9000}))
    monkeypatch.setenv("CLOCKD_PORT", "tcp://10.108.181.189:8000")
    cfg = load_server_config(str(path))
    assert cfg.port == 9000


def test_k8s_service_link_port_ignored_without_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOCKD_PORT", "tcp://10.108.181.189:8000")
    cfg = load_server_config(str(tmp_path / "missing.yaml"))
    assert cfg.port == 8000


def test_real_env_port_still_wins(tmp_path, monkeypatch):
    path = tmp_path / "server.yaml"
    path.write_text(yaml.dump({"port": 9000}))
    monkeypatch.setenv("CLOCKD_PORT", "9100")
    cfg = load_server_config(str(path))
    assert cfg.port == 9100
