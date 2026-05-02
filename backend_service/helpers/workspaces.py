"""Phase 3.7: workspace knowledge stack registry.

A workspace is a named bundle of documents that multiple chat
sessions can share. Each session can be assigned to at most one
workspace via `ChatSession.workspaceId`; when the RAG retriever
runs it sees both the session's own docs and the workspace's docs
under one merged corpus.

Persistence: a JSON list at `<dataDir>/workspaces.json`, plus a
per-workspace subdirectory at `<dataDir>/workspaces/<id>/` for
uploaded files.

This is a slim CRUD surface — Workspace metadata only (id, title,
description, doc list, timestamps). Document content stays in the
filesystem under the workspace's directory; the index entries on
the workspace point at filenames.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any


class WorkspaceRegistry:
    """JSON-backed CRUD manager for workspace metadata."""

    def __init__(self, registry_path: Path, workspaces_dir: Path) -> None:
        self._lock = RLock()
        self._path = Path(registry_path)
        self._dir = Path(workspaces_dir)
        self._workspaces: dict[str, dict[str, Any]] = {}
        self.load()

    # -- Persistence --------------------------------------------------

    def load(self) -> None:
        with self._lock:
            if not self._path.is_file():
                self._workspaces = {}
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._workspaces = {}
                return
            if isinstance(raw, list):
                self._workspaces = {
                    str(entry.get("id")): entry
                    for entry in raw
                    if isinstance(entry, dict) and entry.get("id")
                }
            elif isinstance(raw, dict):
                self._workspaces = {
                    str(k): v for k, v in raw.items()
                    if isinstance(v, dict)
                }
            else:
                self._workspaces = {}

    def save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = list(self._workspaces.values())
            self._path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # -- CRUD ---------------------------------------------------------

    def list_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(entry) for entry in self._workspaces.values()]

    def get(self, workspace_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._workspaces.get(workspace_id)
            return dict(entry) if entry else None

    def create(self, title: str, description: str = "") -> dict[str, Any]:
        now = self._now_label()
        workspace_id = uuid.uuid4().hex
        entry: dict[str, Any] = {
            "id": workspace_id,
            "title": title or "Untitled workspace",
            "description": description or "",
            "documents": [],
            "createdAt": now,
            "updatedAt": now,
        }
        with self._lock:
            self._workspaces[workspace_id] = entry
            self.save()
            (self._dir / workspace_id).mkdir(parents=True, exist_ok=True)
        return dict(entry)

    def update(
        self,
        workspace_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            existing = self._workspaces.get(workspace_id)
            if existing is None:
                return None
            if title is not None:
                existing["title"] = title
            if description is not None:
                existing["description"] = description
            existing["updatedAt"] = self._now_label()
            self.save()
            return dict(existing)

    def delete(self, workspace_id: str) -> bool:
        with self._lock:
            if workspace_id not in self._workspaces:
                return False
            del self._workspaces[workspace_id]
            self.save()
            workspace_dir = self._dir / workspace_id
            if workspace_dir.is_dir():
                # Remove the workspace's document directory + contents.
                # We do this last so a save() failure above doesn't lose
                # files from an undeleted workspace.
                for child in workspace_dir.glob("**/*"):
                    if child.is_file():
                        try:
                            child.unlink()
                        except OSError:
                            pass
                try:
                    workspace_dir.rmdir()
                except OSError:
                    # Non-empty (residual subdirs) — leave alone.
                    pass
            return True

    def workspace_dir(self, workspace_id: str) -> Path:
        return self._dir / workspace_id

    @staticmethod
    def _now_label() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
