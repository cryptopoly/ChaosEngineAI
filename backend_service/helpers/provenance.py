"""Which Storyboard project and shot a generation belongs to, for the gallery label.

The Storyboard sets this around its calls into the Image and Video generate functions; the artifact
writers read it and record the project id and name and the shot id next to the other settings. A remote
client's agent tools set it too (project_id / project_name arguments). Names
and ids only, never paths. Outside a Storyboard run nothing is set and nothing is recorded.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_CURRENT: ContextVar[dict[str, str] | None] = ContextVar("generation_provenance", default=None)


@contextmanager
def provenance(project_id: str, project_name: str, shot_id: str = "") -> Iterator[None]:
    token = _CURRENT.set({"projectId": project_id, "projectName": project_name, "shotId": shot_id})
    try:
        yield
    finally:
        _CURRENT.reset(token)


def stamp(artifact: dict[str, Any]) -> dict[str, Any]:
    """The artifact with the current provenance added (unchanged when none is set)."""
    current = _CURRENT.get()
    return {**artifact, **{k: v for k, v in current.items() if v}} if current else artifact
