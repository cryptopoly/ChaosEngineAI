"""E2E phase 15: sound effects (D-27, Stable Audio Open Small in the Music Studio's Sounds mode).

Checks, against a running backend:
1. the catalog lists the sound row: gated, pinned 40-character revisions, the "Powered by Stability AI" line;
2. the runtime reports the sounds pack next to the music pack;
3. a sound model is refused as a music model, and a music model as a sound model;
4. a sound longer than the model's own maximum (11 s Small, 47 s 1.0) is refused before anything loads;
5. while the pack or the gated weights are missing, the request answers 409 with the plain fix;
6. full run when the sounds pack and the weights are installed: a 3 s seamless loop is made, saved to the
   Music Library as kind ``sound`` with the loop flag, and its FLAC carries the machine-generated tag, the
   prompt, the seed and the licence line (never a local path); the item is deleted afterwards.
Nothing is downloaded or installed by this phase (the weights are gated and need the user's own licence
acceptance), and the full run waits its turn on the shared heavy-job queue.
"""

from __future__ import annotations

import json
import re
from typing import Any

SOUND_REPO = "stabilityai/stable-audio-open-small"


def phase_15(cap: Any):
    import e2e_test_suite as suite  # scripts/ is on sys.path when the suite runs

    phase = suite.PhaseResult(phase=15, name="Sound effects")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 15", "skip", reason="backend not reachable"))
        return phase
    http = suite._voice_http
    state: dict[str, Any] = {}

    def _json(method: str, path: str, payload: dict | None = None, timeout: float = 120.0):
        body = json.dumps(payload).encode() if payload is not None else None
        code, raw = http(method, path, body=body, content_type="application/json" if body else None, timeout=timeout)
        try:
            return code, json.loads(raw.decode() or "null")
        except ValueError:
            return code, raw

    def _catalog():
        code, body = _json("GET", "/api/music/catalog")
        if code == 404:
            return "skip", "music routes not in this build", {}
        if code != 200:
            return "fail", f"HTTP {code}", {}
        rows = [v for f in body.get("families") or [] for v in f.get("variants") or [] if v.get("kind") == "sound"]
        if not rows:
            return "skip", "no sound model in this build's catalog", {}
        row = rows[0]
        state["row"] = row
        licence = row.get("licence") or {}
        problems = []
        if row.get("repo") != SOUND_REPO:
            problems.append(f"repo {row.get('repo')}")
        if not licence.get("gated"):
            problems.append("not gated")
        if licence.get("attribution") != "Powered by Stability AI":
            problems.append("attribution missing")
        if not re.fullmatch(r"[0-9a-f]{40}", str(row.get("revision", ""))):
            problems.append("revision not pinned")
        if not re.fullmatch(r"[0-9a-f]{40}", str((row.get("textEncoder") or {}).get("revision", ""))):
            problems.append("text encoder not pinned")
        if problems:
            return "fail", ", ".join(problems), {}
        return "pass", "", {"installed": bool(row.get("availableLocally")), "packInstalled": bool(row.get("packInstalled"))}

    def _full_model_row():
        code, body = _json("GET", "/api/music/catalog")
        rows = [v for f in (body or {}).get("families") or [] for v in f.get("variants") or [] if v.get("kind") == "sound"]
        full = next((v for v in rows if v.get("repo") == "stabilityai/stable-audio-open-1.0"), None)
        if code != 200 or not rows:
            return "skip", "no sound models in this build's catalog", {}
        if full is None:
            return "fail", "Stable Audio Open 1.0 is not in the catalog", {}
        problems = []
        if int(full.get("maxSeconds") or 0) != 47:
            problems.append(f"maxSeconds {full.get('maxSeconds')}")
        if not (full.get("licence") or {}).get("gated"):
            problems.append("not gated")
        if not re.fullmatch(r"[0-9a-f]{40}", str(full.get("revision", ""))):
            problems.append("revision not pinned")
        if (full.get("estimates") or {}).get("basis") == "measured" and not (full.get("estimates") or {}).get("measuredOn"):
            problems.append("measured without a measuredOn")
        code, body = _json("POST", "/api/music/estimate", {"modelId": full["id"], "seconds": 40})
        if code != 200:
            problems.append(f"estimate HTTP {code}")
        code, _ = _json("POST", "/api/music/estimate", {"modelId": rows[0]["id"], "seconds": 40})
        if rows[0]["id"] != full["id"] and code != 400:
            problems.append(f"Small accepted 40 s (HTTP {code})")
        return ("fail", ", ".join(problems), {}) if problems else ("pass", "", {"installed": bool(full.get("availableLocally"))})

    def _runtime():
        if "row" not in state:
            return "skip", "no sound row", {}
        code, body = _json("GET", "/api/music/runtime")
        sound = ((body or {}).get("runtime") or {}).get("sound") if code == 200 else None
        if not sound or "installed" not in sound or "install" not in sound:
            return "fail", f"HTTP {code}, no sounds block", {}
        return "pass", "", {"packInstalled": sound["installed"], "lane": sound.get("lane")}

    def _cross_refusals():
        if "row" not in state:
            return "skip", "no sound row", {}
        code, body = _json("POST", "/api/music/sound/generate", {"caption": "x", "modelId": "ACE-Step/Ace-Step1.5"})
        if code != 400 or "makes music" not in str((body or {}).get("detail")):
            return "fail", f"sound request with a music model: HTTP {code} {body}", {}
        code, body = _json("POST", "/api/music/generate", {"caption": "x", "instrumental": True, "seconds": 20, "modelId": SOUND_REPO})
        if code != 400 or "sound effects" not in str((body or {}).get("detail")):
            return "fail", f"music request with the sound model: HTTP {code} {body}", {}
        return "pass", "", {}

    def _too_long():
        if "row" not in state:
            return "skip", "no sound row", {}
        # One second over the row's own maximum is a 400 naming the model's range (the request model only holds a
        # clip to the widest sound model's maximum, which is a 422).
        over = int(state["row"].get("maxSeconds") or 11) + 1
        code, _ = _json("POST", "/api/music/sound/generate", {"caption": "x", "seconds": over, "modelId": state["row"]["id"]})
        if code not in (400, 422):
            return "fail", f"expected 400 for {over} s on {state['row']['id']}, got {code}", {}
        code, _ = _json("POST", "/api/music/sound/generate", {"caption": "x", "seconds": 600})
        return ("pass", "", {}) if code == 422 else ("fail", f"expected 422 for 600 s, got {code}", {})

    def _ready() -> bool:
        row = state.get("row") or {}
        return bool(row.get("availableLocally") and row.get("packInstalled"))

    def _missing_is_plain():
        if "row" not in state:
            return "skip", "no sound row", {}
        if _ready():
            return "skip", "sounds pack and weights installed (the full run covers this)", {}
        code, body = _json("POST", "/api/music/sound/generate", {"caption": "a door closing", "seconds": 2})
        detail = str((body or {}).get("detail"))
        if code == 409 and ("not installed" in detail or "not downloaded" in detail or "Hugging Face" in detail or "Apple Silicon" in detail):
            return "pass", "", {"detail": detail[:140]}
        return "fail", f"expected 409 with the fix, got {code}: {detail[:160]}", {}

    def _full_run():
        if "row" not in state:
            return "skip", "no sound row", {}
        if not _ready():
            return "skip", "sounds pack or gated weights not installed (won't download them in a test run)", {}
        code, body = _json("POST", "/api/music/sound/generate",
                           {"caption": "wooden door creaking open", "seconds": 3, "loop": True, "seed": 21}, timeout=900.0)
        if code != 200:
            return "fail", f"generate HTTP {code}: {str(body)[:200]}", {}
        item = body["items"][0]
        try:
            if (item.get("kind"), item.get("loop"), item.get("seed")) != ("sound", True, 21):
                return "fail", f"library item wrong: {item}", {}
            code, audio = http("GET", f"/api/music/library/{item['id']}/audio", timeout=60.0)
            if code != 200 or audio[:4] != b"fLaC":
                return "fail", f"audio HTTP {code}", {}
            tag = re.search(rb"chaosengine:(\{.*?\})", audio)
            settings = json.loads(tag.group(1)) if tag else {}
            if not (settings.get("machine_generated") and settings.get("kind") == "sound" and settings.get("loop") is True
                    and settings.get("seed") == 21 and "Powered by Stability AI" in str(settings.get("licence"))
                    and settings.get("prompt")):
                return "fail", f"tag incomplete: {settings}", {}
            if b"/Users/" in tag.group(1) or b"/home/" in tag.group(1):
                return "fail", "tag holds a local path", {}
            return "pass", "", {"seconds": body.get("seconds"), "peakFootprintGb": body.get("peakFootprintGb"),
                                "durationS": item.get("durationS")}
        finally:
            _json("DELETE", f"/api/music/library/{item['id']}")
            _json("POST", "/api/music/unload", {})

    for name, fn in (("sound row pinned and gated", _catalog), ("Stable Audio Open 1.0 row and 47 s limit", _full_model_row),
                     ("sounds pack in runtime", _runtime),
                     ("music and sound models not mixed", _cross_refusals), ("over the maximum refused", _too_long),
                     ("missing pack or weights is a plain 409", _missing_is_plain), ("loop made, tagged, saved", _full_run)):
        phase.checks.append(suite._check(name, fn))
    statuses = {c.status for c in phase.checks}
    phase.status = "fail" if "fail" in statuses else ("pass" if "pass" in statuses else "skip")
    return phase
