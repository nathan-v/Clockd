from __future__ import annotations

from clockd.utils.log import sanitize_for_log


def test_sanitize_strips_control_chars():
    assert sanitize_for_log("evil\nFAKE log line\r\x1b[31m") == "evil?FAKE log line??[31m"


def test_sanitize_none_returns_empty():
    assert sanitize_for_log(None) == ""


def test_sanitize_passthrough():
    assert sanitize_for_log("normal-video.mp4") == "normal-video.mp4"
