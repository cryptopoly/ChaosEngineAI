"""Phase 23: Storyboard guided-flow contracts (UX audit stream A, D0136 to D0138) and the character lock (B1315).

No model runs. Through the backend's API: the ``guide`` block on every project payload and ``GET .../guide``; the
character lock (a reference in every shot, a shot that leaves it out, each shot's effective cast in
``guide.cast``); lip-sync coverage (a song made here as a short WAV, a 1.5 s and a 4 s shot, "Give each shot its
vocal" without a stem: reason codes in the answer, the record stored with the song, the table after a reload);
quality labels (the film's size and the draft size); export readiness (an empty project stops the export, shots
without clips stop it with ``no-clips``); and the stage rail (stage order, the current stage).

Two checks run in this process against a temporary project store, because they need a finished or failed step and
this phase never generates: a still made by a stand-in still maker that puts the cast in the way the studio's does
(``cast.apply_to_still``, no reference-picture slots) must record ``cast: words`` on the take and leave the
opted-out shot without the character; and a step that fails with an out-of-memory error must store the
``out-of-memory`` code beside the raw text. The song is written to a temporary folder the backend reads, so the
song checks need the backend on this machine. Every project made here is deleted at the end.

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PHASE_KEY = 23
ROOT = Path(__file__).resolve().parents[2]
STAGES = ["song", "look", "shots", "drafts", "final", "lipsync", "sound", "check", "export"]


def _in_process_store(base: Path) -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend_service.storyboard.store import ProjectStore

    return ProjectStore(base / "storyboards")


def _png(path: Path) -> Path:
    from e2e_phases.phase_20 import png

    path.write_bytes(png(16, 16, [(200, 30, 30)]))
    return path


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=PHASE_KEY, name="Storyboard guide")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 23", "skip", reason="backend not reachable"))
        return phase

    def api(method: str, path: str, payload: Any = None, timeout: float = 120.0) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        code, data = suite._voice_http(method, path, body=body, content_type="application/json" if body else None,
                                       timeout=timeout)
        try:
            return code, json.loads(data.decode("utf-8")) if data.strip() else None
        except (ValueError, UnicodeDecodeError):
            return code, data

    code, _ = api("GET", "/api/storyboard/projects")
    if code == 404:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 23", "skip", reason="storyboard not present in this build"))
        return phase

    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    made: list[str] = []

    def create(name: str) -> dict[str, Any]:
        code, body = api("POST", "/api/storyboard/projects", {"name": name})
        if code != 200:
            raise RuntimeError(f"create: HTTP {code} {str(body)[:160]}")
        made.append(body["project"]["id"])
        return body

    def op(body: dict[str, Any], name: str, args: dict[str, Any]) -> dict[str, Any]:
        pid = body["project"]["id"]
        code, out = api("POST", f"/api/storyboard/projects/{pid}/ops",
                        {"revision": body["project"]["revision"], "op": name, "args": args})
        if code != 200:
            raise RuntimeError(f"{name}: HTTP {code} {str(out)[:200]}")
        return out

    def _guide_on_payloads():
        body = create("E2E guide")
        keys = {"quality", "coverage", "failures", "readiness", "stages", "cast"}
        if set(body.get("guide") or {}) != keys:
            return "fail", f"payload guide keys {sorted(body.get('guide') or {})}", {}
        code, got = api("GET", f"/api/storyboard/projects/{body['project']['id']}/guide")
        if code != 200 or got.get("revision") != body["project"]["revision"] or set(got["guide"]) != keys:
            return "fail", f"GET guide: HTTP {code} {str(got)[:160]}", {}
        return "pass", "", {}

    def _character_lock_api():
        body = create("E2E character lock")
        body = op(body, "add_reference", {"kind": "character", "name": "Mara", "description": "short red hair",
                                          "inEveryShot": True})
        ref = body["project"]["references"]["characters"][0]["id"]
        body = op(body, "add_shot", {"title": "with Mara", "stillPrompt": "a rooftop at night"})
        body = op(body, "add_shot", {"title": "empty street", "stillPrompt": "an empty street"})
        shots = [s["id"] for s in body["project"]["shots"]]
        body = op(body, "edit_shot", {"shotId": shots[1], "castOut": [ref]})
        cast = body["guide"]["cast"]
        if cast != {shots[0]: [ref], shots[1]: []}:
            return "fail", f"effective cast {cast}", {}
        body = op(body, "edit_reference", {"refId": ref, "inEveryShot": False})
        if body["guide"]["cast"][shots[0]] != []:
            return "fail", "the lock is off but the shot still has the character", {}
        return "pass", "", {"cast": cast}

    def _character_lock_take():
        from backend_service.storyboard import cast as cast_mod
        from backend_service.storyboard.batch import Batch
        from backend_service.storyboard.ops import run_op

        store = _in_process_store(base)
        p = store.create("E2E take cast", None)
        p = run_op(store, p.id, p.revision, "add_reference", {"kind": "character", "name": "Mara",
                                                             "description": "short red hair", "inEveryShot": True})
        ref = p.references.characters[0].id
        for title in ("with Mara", "without"):
            p = run_op(store, p.id, p.revision, "add_shot", {"title": title, "stillPrompt": "a rooftop at night"})
        p = run_op(store, p.id, p.revision, "edit_shot", {"shotId": p.shots[1].id, "castOut": [ref]})
        prompts: dict[str, str] = {}

        def make_still(shot, project):  # the studio's cast step with a model that takes no reference pictures
            _paths, prompt, mode, note = cast_mod.apply_to_still(project, shot, [], 0, shot.stillPrompt)
            prompts[shot.id] = prompt
            settings: dict[str, Any] = {"model": "stand-in", "width": 640, "height": 384}
            if mode != "none":
                settings["cast"] = mode
            if note:
                settings["castNote"] = note
            return str(_png(base / f"{shot.id}.png")), {"settings": settings}

        batch = Batch(store, p.id, "still", make_still, lambda s, q, i: None)
        batch.start(p.revision)
        batch.run()
        p = store.load(p.id)
        first, second = (s.active() for s in p.shots)
        if first is None or second is None:
            return "fail", "a still was not made", {}
        if first.settings.get("cast") != "words" or "Mara" not in prompts[p.shots[0].id]:
            return "fail", f"take cast {first.settings.get('cast')!r}, prompt {prompts[p.shots[0].id]!r}", {}
        if "cast" in second.settings or "Mara" in prompts[p.shots[1].id]:
            return "fail", "the shot that left the character out still has it", {}
        return "pass", "", {"cast": first.settings.get("cast"), "castNote": first.settings.get("castNote")}

    def _failure_code():
        from backend_service.storyboard.batch import Batch
        from backend_service.storyboard.failure import describe

        store = _in_process_store(base)
        p = store.create("E2E failure", None)
        from backend_service.storyboard.ops import run_op

        p = run_op(store, p.id, p.revision, "add_shot", {"title": "big", "stillPrompt": "a city"})

        def make_still(shot, project):
            raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")

        batch = Batch(store, p.id, "still", make_still, lambda s, q, i: None)
        batch.start(p.revision)
        batch.run()
        result = next(r for r in store.load(p.id).batches[-1].results if r.state == "failed")
        if result.code != "out-of-memory" or "CUDA out of memory" not in (result.error or ""):
            return "fail", f"code {result.code!r}, error {result.error!r}", {}
        if describe(result.code)["action"] != "smaller-size":
            return "fail", "out-of-memory does not offer a smaller size", {}
        return "pass", "", {"code": result.code}

    def _coverage():
        from e2e_phases.phase_20 import song_wav

        body = create("E2E lip-sync coverage")
        for seconds in (1.5, 4.0):
            body = op(body, "add_shot", {"title": f"{seconds} s", "seconds": seconds, "trimOut": seconds})
        wav = base / "e2e-guide-song.wav"
        wav.write_bytes(song_wav())
        pid = body["project"]["id"]
        code, body = api("POST", f"/api/storyboard/projects/{pid}/song",
                         {"revision": body["project"]["revision"], "path": str(wav)}, timeout=300)
        if code != 200:
            return "fail", f"song attach: HTTP {code} {str(body)[:200]}", {}
        code, out = api("POST", f"/api/storyboard/projects/{pid}/song/vocals", {"revision": body["project"]["revision"]},
                        timeout=300)
        if code == 409 and isinstance(out, dict) and (out.get("detail") or {}).get("code") == "install-ffmpeg":
            return "skip", "video tools are not installed on the backend", {}
        if code != 200:
            return "fail", f"vocals: HTTP {code} {str(out)[:200]}", {}
        skipped = [(s["number"], s.get("code")) for s in out["vocals"]["skipped"]]
        if skipped != [(1, "too-short")]:
            return "fail", f"skipped {skipped}", {}
        stored = (out["project"]["song"] or {}).get("coverage") or {}
        if [s["code"] for s in stored.get("shots", [])] != ["too-short", None]:
            return "fail", f"stored coverage {stored}", {}
        code, again = api("GET", f"/api/storyboard/projects/{pid}")
        rows = [(r["state"], r["code"], r["fix"]) for r in again["guide"]["coverage"]["shots"]]
        if rows != [("skipped", "too-short", "lengthen"), ("waiting", None, "make-lipsync")]:
            return "fail", f"coverage after a reload {rows}", {}
        return "pass", "", {"rows": rows}

    def _quality_readiness_stages():
        body = create("E2E readiness")
        guide = body["guide"]
        if guide["quality"]["target"] != {"width": 1280, "height": 720, "fps": 24}:
            return "fail", f"quality target {guide['quality']['target']}", {}
        if guide["quality"]["draftSize"] != {"width": 640, "height": 384}:
            return "fail", f"draft size {guide['quality']['draftSize']}", {}
        if [i["code"] for i in guide["readiness"]["stops"]] != ["no-shots"]:
            return "fail", f"empty project stops {guide['readiness']['stops']}", {}
        if [s["id"] for s in guide["stages"]["stages"]] != STAGES or guide["stages"]["current"] != "shots":
            return "fail", f"stages {guide['stages']}", {}
        body = op(body, "add_shot", {"title": "one", "stillPrompt": "sea"})
        guide = body["guide"]
        if [i["code"] for i in guide["readiness"]["stops"]] != ["no-clips"] or guide["readiness"]["ready"]:
            return "fail", f"no clips: {guide['readiness']}", {}
        drafts = next(s for s in guide["stages"]["stages"] if s["id"] == "drafts")
        if (guide["stages"]["current"], drafts["action"]) != ("drafts", "make-drafts"):
            return "fail", f"current {guide['stages']['current']}, drafts {drafts}", {}
        return "pass", "", {}

    try:
        for name, fn in [
            ("every project payload and GET .../guide carry the guide block", _guide_on_payloads),
            ("character lock: in every shot, a shot leaves it out, effective cast in the payload (B1315)", _character_lock_api),
            ("character lock: the take records how the cast went in; the opted-out shot has none (in process)", _character_lock_take),
            ("a failed step stores its failure code beside the raw text (in process)", _failure_code),
            ("lip-sync coverage: reason codes, stored with the song, live table with fixes", _coverage),
            ("quality target, export readiness groups and the stage rail", _quality_readiness_stages),
        ]:
            phase.checks.append(suite._check(name, fn))
    finally:
        for project_id in made:
            code, body = api("GET", f"/api/storyboard/projects/{project_id}")
            if code == 200:
                suite._voice_http("DELETE", f"/api/storyboard/projects/{project_id}?revision={body['project']['revision']}")
        tmp.cleanup()
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
