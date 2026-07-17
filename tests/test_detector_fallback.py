from __future__ import annotations

import numpy as np
import pytest

from clockd.services.detector import (
    Detector,
    DetectorUnavailableError,
    FallbackDetector,
)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


class _StubDetector(Detector):
    """Scripted detector: each call pops the next behavior.

    "ok" returns a sentinel detections object; "fail" raises
    DetectorUnavailableError. The last behavior repeats when exhausted.
    """

    def __init__(self, *script: str, sentinel=None):
        import supervision as sv

        self._script = list(script)
        self.calls = 0
        self.sentinel = sentinel if sentinel is not None else sv.Detections.empty()

    def detect(self, frame):
        self.calls += 1
        behavior = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if behavior == "fail":
            raise DetectorUnavailableError("stub backend down")
        return self.sentinel


def test_immediate_fallback_when_remote_never_succeeded():
    primary = _StubDetector("fail")
    local = _StubDetector("ok")
    det = FallbackDetector(primary, fallback_factory=lambda: local)

    det.detect(FRAME)

    assert det.using_fallback
    assert primary.calls == 1  # single failure, no retry storm
    assert local.calls == 1


def test_fallback_sticky_for_rest_of_job():
    primary = _StubDetector("fail")
    local = _StubDetector("ok")
    det = FallbackDetector(primary, fallback_factory=lambda: local)

    for _ in range(5):
        det.detect(FRAME)

    assert primary.calls == 1  # never consulted again after the switch
    assert local.calls == 5


def test_consecutive_failure_threshold_after_success():
    # ok, then permanent failure: 2 failures return empty, 3rd switches
    primary = _StubDetector("ok", "fail")
    local = _StubDetector("ok")
    det = FallbackDetector(primary, fallback_factory=lambda: local)

    det.detect(FRAME)  # ok
    det.detect(FRAME)  # fail 1 -> empty
    det.detect(FRAME)  # fail 2 -> empty
    assert not det.using_fallback
    assert local.calls == 0

    det.detect(FRAME)  # fail 3 -> switch
    assert det.using_fallback
    assert local.calls == 1
    assert "stub backend down" in (det.fallback_reason or "")


def test_transient_blip_does_not_switch():
    # a single failure between successes resets the counter
    primary = _StubDetector("ok", "fail", "ok", "fail", "ok")
    local = _StubDetector("ok")
    det = FallbackDetector(primary, fallback_factory=lambda: local)

    for _ in range(5):
        det.detect(FRAME)

    assert not det.using_fallback
    assert local.calls == 0


def test_no_fallback_configured_returns_empty():
    primary = _StubDetector("fail")
    det = FallbackDetector(primary, fallback_factory=None)

    detections = det.detect(FRAME)

    assert len(detections) == 0
    assert not det.using_fallback


def test_fallback_factory_called_lazily():
    calls = []

    def factory():
        calls.append(1)
        return _StubDetector("ok")

    primary = _StubDetector("ok")
    det = FallbackDetector(primary, fallback_factory=factory)
    det.detect(FRAME)

    assert calls == []  # healthy remote never constructs the local detector


def test_detection_fallback_config_validation():
    from clockd.config import ServerConfig

    assert ServerConfig(detection_fallback="local").detection_fallback == "local"
    with pytest.raises(ValueError):
        ServerConfig(detection_fallback="gpu")
