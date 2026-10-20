"""Read step progress off a line of stable-diffusion.cpp output."""

from __future__ import annotations

import re

# sd.cpp has printed ``step N/M``, ``[N/M]`` and, in current builds, a bar that
# ends ``| N/M - 23.3s/it``. Model-loading bars end in a byte rate
# (``| 19/219 - 250MB/s``) and must not count as sampling steps.
_STEP_RE = re.compile(
    r"(?:step\s+|\[)(\d+)\s*/\s*(\d+)"
    r"|(\d+)\s*/\s*(\d+)\s*-\s*[\d.]+\s*(?:s/it|it/s)"
)


def parse_step(line: str) -> tuple[int, int] | None:
    """``(step, total)`` when ``line`` reports a sampling step, else None."""
    match = _STEP_RE.search(line)
    if match is None:
        return None
    step, total = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
    return int(step), int(total)
