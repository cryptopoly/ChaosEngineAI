"""Tiny shared helpers used by inference engines + the controller.

Lives in its own module so submodules (engines, controller pieces) can
import without circling through ``backend_service.inference``'s
``__init__``.
"""

from __future__ import annotations

import time
from typing import Any


def _now_label() -> str:
    """ISO-ish timestamp for ``LoadedModelInfo.loadedAt`` and similar fields."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_message_content(content: Any) -> str:
    """Coerce whatever shape the chat API received into a single text string.

    The OpenAI-style payloads can be a plain string, a list of content
    parts (``[{"type": "text", "text": "…"}, …]``), or even ``None``.
    Concrete engines all want a single string for downstream prompt
    assembly, so do that conversion in one place.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return " ".join(parts)
    return str(content or "")
