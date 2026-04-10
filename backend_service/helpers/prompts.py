"""Prompt template library with CRUD, search, and persistence."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any


# ---------------------------------------------------------------------------
# Seed templates
# ---------------------------------------------------------------------------

_SEED_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "builtin-coding-assistant",
        "name": "Coding Assistant",
        "systemPrompt": (
            "You are an expert software engineer. Write clean, well-documented "
            "code. Explain your reasoning step by step when asked. Follow best "
            "practices for the language in use."
        ),
        "tags": ["coding", "engineering", "development"],
        "category": "Development",
        "fewShotExamples": [],
    },
    {
        "id": "builtin-creative-writer",
        "name": "Creative Writer",
        "systemPrompt": (
            "You are a creative writing assistant. Help the user brainstorm, "
            "draft, and polish fiction, poetry, screenplays, and other creative "
            "works. Offer vivid language and varied sentence structure."
        ),
        "tags": ["writing", "creative", "fiction"],
        "category": "Writing",
        "fewShotExamples": [],
    },
    {
        "id": "builtin-data-analyst",
        "name": "Data Analyst",
        "systemPrompt": (
            "You are a data analysis expert. Help the user explore datasets, "
            "write queries, build visualizations, and interpret statistical "
            "results. Prefer clarity and reproducibility."
        ),
        "tags": ["data", "analytics", "statistics"],
        "category": "Data",
        "fewShotExamples": [],
    },
    {
        "id": "builtin-translator",
        "name": "Translator",
        "systemPrompt": (
            "You are a professional translator. Translate text accurately "
            "between languages while preserving tone, idiom, and cultural "
            "nuance. Ask for clarification when the source is ambiguous."
        ),
        "tags": ["translation", "language", "localization"],
        "category": "Language",
        "fewShotExamples": [],
    },
    {
        "id": "builtin-summarizer",
        "name": "Summarizer",
        "systemPrompt": (
            "You are a concise summarization assistant. Distill long documents, "
            "articles, or conversations into clear, accurate summaries. "
            "Highlight key points and preserve important details."
        ),
        "tags": ["summary", "condensing", "research"],
        "category": "Productivity",
        "fewShotExamples": [],
    },
]


class PromptLibrary:
    """CRUD manager for prompt templates backed by a JSON file."""

    def __init__(self, data_dir: str | Path) -> None:
        self._lock = RLock()
        self._data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self._file = self._data_dir / "prompt_templates.json"
        self._templates: dict[str, dict[str, Any]] = {}
        self.load()

    # -- Persistence ---------------------------------------------------------

    def load(self) -> None:
        """Load templates from disk, seeding with built-ins if empty."""
        with self._lock:
            if self._file.is_file():
                try:
                    raw = json.loads(self._file.read_text(encoding="utf-8"))
                    if isinstance(raw, list):
                        self._templates = {t["id"]: t for t in raw if "id" in t}
                    elif isinstance(raw, dict):
                        self._templates = raw
                except (json.JSONDecodeError, OSError):
                    self._templates = {}

            # Seed built-ins if we have no templates at all
            if not self._templates:
                now = time.time()
                for tmpl in _SEED_TEMPLATES:
                    entry = {**tmpl, "createdAt": now, "updatedAt": now}
                    self._templates[entry["id"]] = entry
                self.save()

    def save(self) -> None:
        """Persist templates to disk."""
        with self._lock:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            payload = list(self._templates.values())
            self._file.write_text(
                json.dumps(payload, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )

    # -- CRUD ----------------------------------------------------------------

    def list_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._templates.values())

    def get(self, template_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._templates.get(template_id)

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        entry: dict[str, Any] = {
            "id": data.get("id") or uuid.uuid4().hex,
            "name": data.get("name", "Untitled"),
            "systemPrompt": data.get("systemPrompt", ""),
            "tags": data.get("tags", []),
            "category": data.get("category", "General"),
            "fewShotExamples": data.get("fewShotExamples", []),
            "createdAt": now,
            "updatedAt": now,
        }
        with self._lock:
            self._templates[entry["id"]] = entry
            self.save()
        return entry

    def update(self, template_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            existing = self._templates.get(template_id)
            if existing is None:
                return None
            for key in ("name", "systemPrompt", "tags", "category", "fewShotExamples"):
                if key in data:
                    existing[key] = data[key]
            existing["updatedAt"] = time.time()
            self.save()
            return existing

    def delete(self, template_id: str) -> bool:
        with self._lock:
            if template_id in self._templates:
                del self._templates[template_id]
                self.save()
                return True
            return False

    def search(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Simple in-memory search over templates."""
        results = self.list_all()

        if category:
            cat_lower = category.lower()
            results = [t for t in results if t.get("category", "").lower() == cat_lower]

        if tags:
            tag_set = {t.lower() for t in tags}
            results = [
                t for t in results
                if tag_set & {tg.lower() for tg in t.get("tags", [])}
            ]

        if query:
            q = query.lower()
            results = [
                t for t in results
                if q in t.get("name", "").lower()
                or q in t.get("systemPrompt", "").lower()
                or any(q in tg.lower() for tg in t.get("tags", []))
            ]

        return results
