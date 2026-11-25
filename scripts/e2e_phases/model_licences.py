"""D0135 check: every row of the text, image and video catalogues carries a structured licence (name, link, commercial
use, gated, one sentence), in the shape the voice and music rows use, so the model cards can show it. No model runs.
Each function returns ``(status, reason, detail)`` for the suite's ``_check``; ``h`` is the suite module.
"""

from __future__ import annotations

from typing import Any

COMMERCIAL = ("yes", "limited", "no")


def licence_problem(licence: Any) -> str | None:
    """What is wrong with one licence object, or None."""
    if not isinstance(licence, dict):
        return "no licence"
    for key in ("id", "name", "url", "summary"):
        if not str(licence.get(key) or "").strip():
            return f"no {key}"
    if licence.get("commercial") not in COMMERCIAL:
        return f"commercial is {licence.get('commercial')!r}"
    if not isinstance(licence.get("gated"), bool):
        return "gated is not a boolean"
    if not str(licence["url"]).startswith("https://"):
        return f"link is not https: {licence['url']}"
    return None


def catalog_licences(h: Any, kind: str) -> tuple[str, str, dict[str, Any]]:
    """Every row of the live ``kind`` catalogue ("text", "image" or "video") has a valid licence."""
    if kind == "text":
        rc, payload, err = h._cli_json("call", "GET", "/api/workspace", timeout=20.0)
        families = (payload or {}).get("featuredModels") if isinstance(payload, dict) else None
    else:
        rc, payload, err = h._cli_json(f"{kind}-catalog", timeout=20.0)
        families = (payload or {}).get("families") if isinstance(payload, dict) else None
    if rc != 0 or not isinstance(families, list):
        return "fail", f"{kind} catalogue fetch failed: {str(err)[:160]}", {}
    rows, problems, commercial = 0, [], {value: 0 for value in COMMERCIAL}
    for family in families:
        for variant in family.get("variants") or []:
            rows += 1
            problem = licence_problem(variant.get("licence"))
            if problem:
                problems.append(f"{variant.get('id')}: {problem}")
            else:
                commercial[variant["licence"]["commercial"]] += 1
    if not rows:
        return "fail", f"no {kind} rows", {}
    if problems:
        return "fail", "; ".join(problems[:5]), {"count": len(problems), "rows": rows}
    return "pass", "", {"rows": rows, **commercial}
