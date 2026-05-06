"""Prompt template library with CRUD, search, and persistence."""
from __future__ import annotations

import json
import re
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
            # Phase 2.7: variable declarations + preset samplers + preset model
            # default to empty / None so existing templates keep their shape.
            "variables": _normalise_variables(data.get("variables", [])),
            "presetSamplers": data.get("presetSamplers"),
            "presetModelRef": data.get("presetModelRef"),
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
            # Phase 2.7: optional fields — set when present, leave alone otherwise.
            if "variables" in data:
                existing["variables"] = _normalise_variables(data["variables"])
            if "presetSamplers" in data:
                existing["presetSamplers"] = data["presetSamplers"]
            if "presetModelRef" in data:
                existing["presetModelRef"] = data["presetModelRef"]
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


# ---------------------------------------------------------------------------
# Phase 2.7: variable substitution helpers
# ---------------------------------------------------------------------------

# Match `{{name}}` placeholders. Names are alphanumeric + underscore + dash;
# whitespace inside the braces is tolerated so users can write `{{ topic }}`
# in templates and still have it match the declared variable name `topic`.
_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z0-9_\-]+)\s*\}\}")

_VALID_VARIABLE_TYPES: tuple[str, ...] = ("string", "number", "boolean")


def _normalise_variables(raw: Any) -> list[dict[str, Any]]:
    """Coerce a user-supplied variable list into the canonical schema.

    Each entry is `{name: str, type: "string"|"number"|"boolean", default: Any}`.
    Invalid entries are dropped silently rather than raising — the UI
    does the validation work; this layer just keeps storage clean.
    """
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name in seen_names:
            continue
        seen_names.add(name)
        var_type = entry.get("type", "string")
        if var_type not in _VALID_VARIABLE_TYPES:
            var_type = "string"
        cleaned.append({
            "name": name,
            "type": var_type,
            "default": entry.get("default"),
            "description": str(entry.get("description") or "")[:200],
        })
    return cleaned


def extract_placeholders(text: str) -> list[str]:
    """Return the unique placeholder names present in `text`.

    Order is the order of first appearance — the form renderer uses this
    to match declared-variable order with text-occurrence order so
    declarations not present in the text fall to the bottom.
    """
    if not text:
        return []
    seen: list[str] = []
    seen_set: set[str] = set()
    for match in _PLACEHOLDER_PATTERN.finditer(text):
        name = match.group(1)
        if name not in seen_set:
            seen_set.add(name)
            seen.append(name)
    return seen


def apply_variables(text: str, values: dict[str, Any]) -> str:
    """Replace `{{name}}` placeholders with stringified values.

    Missing names stay as the literal placeholder so the user notices
    the gap in the assembled prompt rather than getting a silently
    truncated message. Boolean / numeric values are coerced via str().
    """
    if not text:
        return text

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            return match.group(0)
        value = values[name]
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    return _PLACEHOLDER_PATTERN.sub(_sub, text)
