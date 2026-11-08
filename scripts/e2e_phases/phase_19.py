"""Phase 19: skills import (D-62): import, off by default, switch on, hostile archives refused, delete.

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)``. Needs no model. Works on a skill with a
unique name made in a temp folder (so a skill the user already has is never touched) and deletes it again.
The backend reads the path itself, so this needs the backend on this machine.
"""

from __future__ import annotations

import json
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=19, name="Skills import")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 19", "skip", reason="backend not reachable"))
        return phase

    def api(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        code, raw = suite._voice_http(method, path, body=body, content_type="application/json" if body else None, timeout=30.0)
        try:
            return code, json.loads(raw.decode("utf-8")) if raw.strip() else None
        except ValueError:
            return code, raw

    tmp = tempfile.TemporaryDirectory()
    name = f"e2e-skill-{uuid.uuid4().hex[:8]}"
    folder = Path(tmp.name) / name
    folder.mkdir()
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: End-to-end check skill.\n---\n\nStep one.\n")
    (folder / "notes.md").write_text("Extra notes.\n")
    (folder / "logo.png").write_bytes(b"\x89PNG\x00\x00")
    state: dict[str, Any] = {}

    def route_present():
        code, data = api("GET", "/api/skills")
        if code == 404:
            return "skip", "skills routes not present in this build", {}
        if code != 200 or not isinstance(data, dict) or "skills" not in data:
            return "fail", f"HTTP {code}", {}
        state["ok"] = True
        return "pass", "", {"skills": data.get("count")}

    def import_folder():
        if not state.get("ok"):
            return "skip", "skills routes not present", {}
        code, data = api("POST", "/api/skills/import", {"path": str(folder)})
        skill = (data or {}).get("skill") if isinstance(data, dict) else None
        if code != 200 or not skill:
            return "fail", f"HTTP {code}: {data}", {}
        files = [f["path"] for f in skill["files"]]
        skipped = [s["path"] for s in skill["skipped"]]
        if skill["enabled"] or files != ["SKILL.md", "notes.md"] or skipped != ["logo.png"]:
            return "fail", f"enabled={skill['enabled']} files={files} skipped={skipped}", {}
        state["id"] = skill["id"]
        return "pass", "", {"id": skill["id"]}

    def switch_on():
        if not state.get("id"):
            return "skip", "no imported skill", {}
        code, data = api("POST", f"/api/skills/{state['id']}/enabled", {"enabled": True})
        if code != 200 or not data["skill"]["enabled"]:
            return "fail", f"HTTP {code}", {}
        code, data = api("POST", "/api/skills/import", {"path": str(folder)})
        if code != 409:
            return "fail", f"a second import answered HTTP {code}, expected 409", {}
        return "pass", "", {}

    def hostile_archives():
        if not state.get("ok"):
            return "skip", "skills routes not present", {}
        bad = Path(tmp.name) / "evil.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("SKILL.md", "---\nname: evil\ndescription: d\n---\n")
            zf.writestr("../escape.md", "x")
        bomb = Path(tmp.name) / "bomb.zip"
        with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", "---\nname: bomb\ndescription: d\n---\n")
            zf.writestr("big.md", "0" * 200_000)
        answers = {p.name: api("POST", "/api/skills/import", {"path": str(p)})[0] for p in (bad, bomb)}
        if any(code != 400 for code in answers.values()):
            return "fail", f"expected 400 for each, got {answers}", {}
        if (Path(tmp.name).parent / "escape.md").exists():
            return "fail", "a traversal entry was written outside", {}
        return "pass", "", answers

    def delete():
        if not state.get("id"):
            return "skip", "no imported skill", {}
        code, _ = api("DELETE", f"/api/skills/{state['id']}")
        listed = api("GET", "/api/skills")[1]
        if code != 200 or any(s["id"] == state["id"] for s in listed["skills"]):
            return "fail", f"HTTP {code}", {}
        state.pop("id")
        return "pass", "", {}

    try:
        for label, fn in [
            ("GET /api/skills answers", route_present),
            ("a folder imports off, with the binary skipped and listed", import_folder),
            ("switch on works and a duplicate import is refused", switch_on),
            ("a traversal zip and a zip bomb are refused", hostile_archives),
            ("delete removes the skill", delete),
        ]:
            phase.checks.append(suite._check(label, fn))
    finally:
        if state.get("id"):
            api("DELETE", f"/api/skills/{state['id']}")
        tmp.cleanup()
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
