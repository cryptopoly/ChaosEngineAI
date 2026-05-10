"""Hugging Face date / number label formatters.

Pure helpers — parse ISO timestamps, format the ``Updated <month> <day>``
label used in the catalog, the shorter ``Released <month> <year>`` for
release dates, and the ``1,234 downloads`` style number label. No
filesystem or network deps.

Extracted from ``backend_service/helpers/huggingface.py`` as part of the
v0.8.0 refactor.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_hf_updated_label(value: str | None) -> str | None:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    now = datetime.now(timezone.utc)
    month_label = parsed.strftime("%b")
    if parsed.year == now.year:
        return f"Updated {month_label} {parsed.day}"
    return f"Updated {month_label} {parsed.day}, {parsed.year}"


def _format_release_label(value: str | None) -> str | None:
    """Format a release date / HF ``createdAt`` into a short human label.

    Accepts either a full ISO datetime (``2024-08-01T12:34:56Z`` — HF API)
    or a year-month shorthand (``2024-08`` — curated catalog entries) and
    returns ``"Released Aug 2024"``. Falls back to None when the input
    can't be parsed.
    """
    if not value:
        return None
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        # Try ``YYYY-MM`` or ``YYYY-MM-DD`` shorthand used in curated catalog
        # entries — ``_parse_iso_datetime`` only handles the full datetime form.
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                parsed = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        if parsed is None:
            return None
    return f"Released {parsed.strftime('%b')} {parsed.year}"


def _hf_number_label(value: int, noun: str) -> str:
    return f"{value:,} {noun}"
