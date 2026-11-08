"""The agent guide (``AGENTS.md``), served three ways: ``GET /api/agent/guide``, the MCP resource
``chaosengine://guide`` and the ``start_here`` prompt.

The file sits at the repo root in a source checkout and in the packaged backend folder (``scripts/stage-runtime.mjs``
copies it next to ``backend_service``), so one path lookup serves both.
"""

from __future__ import annotations

from pathlib import Path

GUIDE_URI = "chaosengine://guide"
GUIDE_NAME = "AGENTS.md"


def guide_path() -> Path:
    return Path(__file__).resolve().parents[2] / GUIDE_NAME


def guide_text() -> str:
    """The guide, or a short pointer when the file is missing from this install."""
    try:
        return guide_path().read_text(encoding="utf-8")
    except OSError:
        return ("The agent guide file is missing from this install. Call system_status, list_models and tools/list, "
                "and run one job at a time with wait_for_job.")
