"""``GET /api/chat/studio/tiers`` (FU-241): Draft, Standard and Final for every installed image and video model.

Called from phase 7 of ``scripts/e2e_test_suite.py`` as ``check(suite)``. Draft is no larger than Standard, Standard
no larger than Final, and a time is positive wherever the catalog has one. Skips when no image or video model is
installed.
"""

from __future__ import annotations

import json
from typing import Any


def check(suite: Any):
    code, raw = suite._voice_http("GET", "/api/chat/studio/tiers", timeout=30.0)
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        data = None
    if code != 200 or not isinstance(data, dict) or data.get("tiers") != ["draft", "standard", "final"]:
        return "fail", f"HTTP {code}: tiers missing", {}
    if data.get("default") != "standard":
        return "fail", f"default tier is {data.get('default')!r}", {}
    models = data.get("models") or []
    if not models:
        return "skip", "no image or video model is installed", {}
    for model in models:
        areas = []
        for tier in ("draft", "standard", "final"):
            entry = (model.get("tiers") or {}).get(tier) or {}
            if entry.get("width") and entry.get("height"):
                areas.append(entry["width"] * entry["height"])
            seconds = entry.get("estimatedSeconds")
            if seconds is not None and seconds <= 0:
                return "fail", f"{model.get('id')} {tier}: time {seconds}", {}
        if areas != sorted(areas):
            return "fail", f"{model.get('id')}: sizes not draft <= standard <= final ({areas})", {}
    return "pass", "", {"models": len(models)}
