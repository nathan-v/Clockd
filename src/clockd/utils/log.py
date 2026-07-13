from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_for_log(value: str | None) -> str:
    """Strip control characters (incl. newlines) so untrusted values can't forge log lines."""
    return _CONTROL_CHARS.sub("?", value or "")
