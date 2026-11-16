"""Phase 20: Storyboard 0.10 (plan, style and brand kit, draft and final passes, redo one half, take compare,
sample film, rough-cut inputs, caption look, stale-state reconcile, the busy-engine field, the Open route and the
Chat plan-card pre-check).

Most checks need no model. Three run real models and skip with the reason when the model is not installed: a
draft still with the fast still model (and, during it, the ``activeGenerations`` entry in ``/api/workspace``),
a draft clip with the project's video model, then a cancelled Redo clip that must keep the take in use (FU-181)
and a Redo still that leaves the clip behind its still. The final pass itself is priced, never run (it uses the
quality still model, minutes per still).

Two checks write into the backend's own project folder (the projects this phase made, nothing else), so they need
the backend on this machine: the reconcile check leaves a project file in the "running" state as a crash would and
opens it, and the Open route check puts a symlink in that project. The folder is the backend's data directory plus
``storyboards`` unless ``CHAOSENGINE_STORYBOARDS_DIR`` is set for the suite as it is for the backend. The plan-card
pre-check runs the Chat gate in this process against a temporary store, because no chat model is needed to prove
that a call that cannot run gets no card; the reload check asks the backend's turn route and runs the gate in this
process the same way. Every project made here is deleted at the end.

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)``.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import time
import urllib.parse
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def png(width: int, height: int, colours: list[tuple[int, int, int]]) -> bytes:
    """A small RGB PNG in vertical bands of ``colours`` (stdlib only), so the palette has something to read."""
    rows = b""
    for _y in range(height):
        row = b"\x00"
        for x in range(width):
            row += bytes(colours[min(len(colours) - 1, x * len(colours) // width)])
        rows += row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    head = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=20, name="Storyboard 0.10")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 20", "skip", reason="backend not reachable"))
        return phase

    def api(method: str, path: str, payload: Any = None, raw: bytes | None = None, timeout: float = 120.0) -> tuple[int, Any]:
        body = raw if raw is not None else (json.dumps(payload).encode("utf-8") if payload is not None else None)
        kind = "application/octet-stream" if raw is not None else ("application/json" if body else None)
        code, data = suite._voice_http(method, path, body=body, content_type=kind, timeout=timeout)
        try:
            return code, json.loads(data.decode("utf-8")) if data.strip() else None
        except (ValueError, UnicodeDecodeError):
            return code, data

    code, _ = api("GET", "/api/storyboard/projects")
    if code == 404:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 20", "skip", reason="storyboard not present in this build"))
        return phase

    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    made: list[str] = []
    state: dict[str, Any] = {}

    class Proj:
        def __init__(self, name: str, defaults: dict[str, Any] | None = None, payload: dict[str, Any] | None = None):
            if payload is None:
                code, payload = api("POST", "/api/storyboard/projects", {"name": name, "defaults": defaults})
                if code != 200:
                    raise RuntimeError(f"create: HTTP {code} {str(payload)[:160]}")
            self.id, self.rev, self.body = payload["project"]["id"], payload["project"]["revision"], payload
            made.append(self.id)

        @property
        def p(self) -> dict[str, Any]:
            return self.body["project"]

        def op(self, op: str, args: dict[str, Any], want: int = 200, revision: int | None = None) -> Any:
            code, body = api("POST", f"/api/storyboard/projects/{self.id}/ops",
                             {"revision": self.rev if revision is None else revision, "op": op, "args": args})
            if code != want:
                raise RuntimeError(f"{op}: HTTP {code}, wanted {want}: {str(body)[:200]}")
            if code == 200:
                self.rev, self.body = body["project"]["revision"], body
            return body

        def step(self, kind: str, want: int = 200) -> Any:
            code, body = api("POST", f"/api/storyboard/projects/{self.id}/{kind}", {"revision": self.rev})
            if code != want:
                raise RuntimeError(f"{kind}: HTTP {code}, wanted {want}: {str(body)[:200]}")
            if code == 200:
                self.rev, self.body = body["project"]["revision"], body
            return body

        def reload(self) -> dict[str, Any]:
            code, body = api("GET", f"/api/storyboard/projects/{self.id}")
            if code != 200:
                raise RuntimeError(f"get: HTTP {code}")
            self.rev, self.body = body["project"]["revision"], body
            return body["project"]

        def shot(self, index: int) -> dict[str, Any]:
            return self.p["shots"][index]

    def store_folder(project_id: str) -> Path | None:
        override = os.environ.get("CHAOSENGINE_STORYBOARDS_DIR")
        if override:
            root = Path(override)
        else:
            code, body = api("GET", "/api/settings")
            data = ((body or {}).get("settings") or {}).get("dataDirectory") if code == 200 else None
            if not data:
                return None
            root = Path(os.path.expanduser(data)) / "storyboards"
        folder = root / project_id
        return folder if (folder / "project.json").is_file() else None

    def upload(data: bytes) -> str:
        code, body = api("POST", "/api/storyboard/uploads", raw=data)
        if code != 200 or not str((body or {}).get("handle", "")).startswith("upload:"):
            raise RuntimeError(f"picture upload: HTTP {code} {str(body)[:160]}")
        return body["handle"]

    def wait_batch(project_id: str, batch_id: str, timeout: float, watch=None) -> dict[str, Any]:
        deadline = time.time() + timeout
        record: dict[str, Any] = {}
        while time.time() < deadline:
            code, body = api("GET", f"/api/storyboard/projects/{project_id}/batches/{batch_id}")
            record = (body or {}).get("batch") or {}
            if watch is not None:
                watch(body or {})
            if record.get("state") not in ("queued", "running"):
                return record
            time.sleep(0.5)
        api("POST", f"/api/storyboard/projects/{project_id}/batches/{batch_id}/cancel")
        raise RuntimeError(f"batch {batch_id} still running after {timeout:.0f} s (cancelled)")

    # ---- model-free checks ----
    def _undo_chain():
        proj = Proj("E2E 0.10 · 灯台 ✨")
        state["main"] = proj
        for index in range(4):
            proj.op("add_shot", {"title": f"shot {index + 1}", "stillPrompt": "a lighthouse at dusk", "seconds": 2})
        first = proj.shot(0)["id"]
        proj.op("set_plan", {"brief": "Ein Leuchtturm bei Nacht · 夜の灯台", "filmSeconds": 30})
        proj.op("set_brand", {"captions": {"colour": "#ff8800", "position": "top"}, "logoCorner": "top-left"})
        proj.op("set_kept", {"shotId": first, "kept": True})
        stale = proj.rev - 1
        proj.op("set_brand", {"styleUse": "words"}, want=409, revision=stale)
        if not (proj.p["plan"]["filmSeconds"] == 30 and proj.p["brand"]["captions"]["position"] == "top" and proj.shot(0)["kept"]):
            return "fail", "an op did not change the project", {}
        proj.step("undo")
        proj.step("undo")
        proj.step("undo")
        if proj.shot(0)["kept"] or proj.p["brand"]["captions"]["position"] != "bottom" or proj.p["plan"]["brief"]:
            return "fail", "three undos did not restore kept, the brand kit and the brief", {"project": proj.p["plan"]}
        for _ in range(3):
            proj.step("redo")
        if not (proj.shot(0)["kept"] and proj.p["brand"]["captions"]["colour"] == "#ff8800"
                and proj.p["plan"]["brief"].endswith("夜の灯台")):
            return "fail", "three redos did not bring the changes back", {}
        return "pass", "", {"projectId": proj.id}

    def _plan_ops():
        proj = state["main"]
        before = [s["id"] for s in proj.p["shots"]]
        proj.op("apply_plan", {"mode": "append", "shots": [
            {"title": "planned a", "stillPrompt": "steam over a cup", "seconds": 3},
            {"title": "planned b", "stillPrompt": "a shop front", "seconds": 4}]})
        if len(proj.p["shots"]) != 6:
            return "fail", f"append left {len(proj.p['shots'])} shots, wanted 6", {}
        proj.op("add_plan_message", {"role": "user", "text": "make it shorter"})
        ids = [s["id"] for s in proj.p["shots"]]
        proj.op("plan_changes", {"edits": [{"shotId": ids[0], "seconds": 1.5}], "remove": [ids[4]],
                                 "add": [{"title": "insert", "stillPrompt": "gull", "after": ids[1]}]})
        titles = [s["title"] for s in proj.p["shots"]]
        if titles[2] != "insert" or "planned a" in titles or proj.shot(0)["seconds"] != 1.5:
            return "fail", f"plan_changes gave {titles}", {}
        proj.step("undo")
        if [s["id"] for s in proj.p["shots"]] != ids or len(proj.p["planChat"]) != 1:
            return "fail", "one undo did not restore the shots, or it removed the chat line", {}
        proj.step("undo")
        if [s["id"] for s in proj.p["shots"]] != before:
            return "fail", "undo of apply_plan did not restore the board", {}
        proj.op("lock", {"shotId": before[1]})
        proj.op("apply_plan", {"mode": "replace", "shots": [{"title": "x", "stillPrompt": "y"}]}, want=400)
        proj.op("lock", {"shotId": before[1], "locked": False})
        return "pass", "", {"shots": len(proj.p["shots"])}

    def _plan_routes():
        proj = state["main"]
        code, status = api("GET", f"/api/storyboard/projects/{proj.id}/plan/status")
        if code != 200 or "chatModel" not in status or "mode" not in (status.get("style") or {}):
            return "fail", f"plan/status: HTTP {code} {str(status)[:160]}", {}
        empty = Proj("E2E 0.10 no brief")
        code, body = api("POST", f"/api/storyboard/projects/{empty.id}/plan/propose", {})
        if code != 400 or (body.get("detail") or {}).get("code") != "no-brief":
            return "fail", f"propose with no brief: HTTP {code} {str(body)[:160]}", {}
        if status["chatModel"] is not None:
            return "pass", "no-model refusal not checked: a chat model is loaded (it is not run here)", {"chatModel": status["chatModel"]}
        code, body = api("POST", f"/api/storyboard/projects/{proj.id}/plan/propose", {"brief": "a tea advert"})
        if code != 409 or (body.get("detail") or {}).get("code") != "no-model":
            return "fail", f"propose with no chat model: HTTP {code} {str(body)[:160]}", {}
        code, body = api("POST", f"/api/storyboard/projects/{proj.id}/plan/direct", {"message": "shorter"})
        if code != 409:
            return "fail", f"direct with no chat model: HTTP {code}", {}
        return "pass", "", {}

    def _style_brand():
        proj = state["main"]
        warm = png(48, 32, [(200, 120, 40), (240, 200, 120), (60, 40, 30)])
        cool = png(48, 32, [(30, 90, 160), (200, 220, 240)])
        handles = [upload(warm), upload(cool)]
        proj.op("add_reference", {"kind": "style", "name": "Look", "images": handles})
        style = proj.p["references"]["styles"][0]
        if len(style["images"]) != 2 or not style.get("palette"):
            return "fail", f"style reference: {len(style['images'])} pictures, palette {style.get('palette')}", {}
        proj.op("set_brand", {"logo": upload(png(16, 16, [(255, 255, 255)])), "logoCorner": "bottom-left",
                              "captions": {"font": "Helvetica", "colour": style["palette"][0], "position": "middle"}})
        brand = proj.p["brand"]
        if not str((brand.get("logo") or {}).get("path", "")).startswith("media/brand/"):
            return "fail", f"logo is {brand.get('logo')}", {}
        code, raw = suite._voice_http("GET", f"/api/storyboard/projects/{proj.id}/files?path={urllib.parse.quote(brand['logo']['path'])}")
        if code != 200 or not raw.startswith(b"\x89PNG"):
            return "fail", f"logo file: HTTP {code}", {}
        code, fonts = api("GET", "/api/storyboard/fonts")
        if code != 200 or not isinstance((fonts or {}).get("fonts"), list):
            return "fail", f"fonts: HTTP {code}", {}
        # the bundle carries the style pictures and the logo
        code, body = api("POST", f"/api/storyboard/projects/{proj.id}/bundle")
        if code != 200:
            return "fail", f"bundle: HTTP {code}", {}
        code, raw = suite._voice_http("GET", f"/api/storyboard/projects/{proj.id}/files?path={urllib.parse.quote(body['file'])}")
        code, copy = api("POST", "/api/storyboard/import", raw=raw)
        if code != 200:
            return "fail", f"bundle import: HTTP {code} {str(copy)[:160]}", {}
        made.append(copy["project"]["id"])
        if copy["project"]["brand"]["logo"] is None or len(copy["project"]["references"]["styles"][0]["images"]) != 2:
            return "fail", "the imported copy lost the logo or a style picture", {}
        return "pass", "", {"palette": style["palette"], "fonts": len(fonts["fonts"])}

    def _passes_free():
        proj = state["main"]
        first = proj.shot(0)["id"]
        code, est = api("POST", f"/api/storyboard/projects/{proj.id}/estimate", {"what": "final"})
        if code != 200 or "passes" not in est or set(est["passes"]) != {"draft", "final"}:
            return "fail", f"estimate final: HTTP {code} {str(est)[:160]}", {}
        if est["passes"]["final"].get("shots") != 1:
            return "fail", f"the final pass counts {est['passes']['final'].get('shots')} kept shots, wanted 1", {}
        code, body = api("POST", f"/api/storyboard/projects/{proj.id}/batches",
                         {"revision": proj.rev, "what": "final", "draft": True})
        if code != 400:
            return "fail", f"a draft final pass answered HTTP {code}, wanted 400", {}
        proj.op("set_kept", {"shotId": first, "kept": False})
        code, body = api("POST", f"/api/storyboard/projects/{proj.id}/batches", {"revision": proj.rev, "what": "final"})
        if code not in (400, 409):
            if code == 200:
                api("POST", f"/api/storyboard/projects/{proj.id}/batches/{body['batchId']}/cancel")
            return "fail", f"a final pass with no kept shot answered HTTP {code}", {}
        proj.op("set_kept", {"shotId": first, "kept": True})
        return "pass", "", {"finalSeconds": est["passes"]["final"].get("seconds")}

    def _redo_free():
        proj = state["main"]
        a, b = base / "a.mp4", base / "s.png"
        a.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 512)
        b.write_bytes(png(16, 16, [(10, 10, 10)]))
        upload_shot = proj.shot(3)["id"]
        proj.op("edit_shot", {"shotId": upload_shot, "startFrom": "upload"})
        proj.op("import_take", {"shotId": upload_shot, "clip": str(a)})
        code, est = api("POST", f"/api/storyboard/projects/{proj.id}/estimate", {"what": "redo-still", "shotIds": [upload_shot]})
        if code != 200 or est.get("stills") != 0:
            return "fail", f"redo-still on an upload shot counts {est.get('stills') if isinstance(est, dict) else est} stills", {}
        code, est = api("POST", f"/api/storyboard/projects/{proj.id}/estimate", {"what": "redo-clip"})
        if code != 200 or "clips" not in est:
            return "fail", f"redo-clip estimate: HTTP {code}", {}
        code, body = api("POST", f"/api/storyboard/projects/{proj.id}/batches",
                         {"revision": proj.rev, "what": "redo-clip", "shotIds": [upload_shot]})
        if code != 400:
            if code == 200:
                api("POST", f"/api/storyboard/projects/{proj.id}/batches/{body['batchId']}/cancel")
            return "fail", f"redo-clip of an upload shot answered HTTP {code}, wanted 400", {}
        return "pass", "", {}

    def _compare():
        proj = state["main"]
        sid = proj.shot(2)["id"]
        takes = []
        for colour in ((200, 30, 30), (30, 200, 30)):
            path = base / f"take-{colour[0]}.png"
            path.write_bytes(png(16, 16, [colour]))
            proj.op("import_take", {"shotId": sid, "still": str(path)})
            takes.append(proj.shot(2)["activeTake"])
        if len(set(takes)) != 2 or len(proj.shot(2)["takes"]) != 2:
            return "fail", "two imported takes did not make two takes", {}
        proj.op("pick_take", {"shotId": sid, "takeId": takes[0]})
        if proj.shot(2)["activeTake"] != takes[0]:
            return "fail", "Use this one (pick_take) did not switch the take in use", {}
        proj.step("undo")
        if proj.shot(2)["activeTake"] != takes[1]:
            return "fail", "undo of pick_take did not switch back", {}
        return "pass", "", {"takes": takes}

    def _sample():
        code, info = api("GET", "/api/storyboard/sample")
        if code != 200:
            return "fail", f"sample info: HTTP {code}", {}
        if not info.get("available"):
            return "skip", "this install carries no sample film", {}
        code, poster = suite._voice_http("GET", "/api/storyboard/sample/poster")
        if code != 200 or not poster[:3] in (b"\xff\xd8\xff", b"\x89PN"):
            return "fail", f"poster: HTTP {code}", {}
        code, body = api("POST", "/api/storyboard/sample", {"name": "E2E sample · サンプル"})
        if code != 200:
            return "fail", f"install: HTTP {code} {str(body)[:160]}", {}
        proj = Proj("", payload=body)
        state["sample"] = proj
        shots = proj.p["shots"]
        if proj.p["name"] != "E2E sample · サンプル" or len(shots) != info["shots"] or any(s["status"] != "done" for s in shots):
            return "fail", f"sample copy: {proj.p['name']!r}, statuses {[s['status'] for s in shots]}", {}
        return "pass", "", {"projectId": proj.id, "shots": len(shots)}

    def _rough_cut():
        proj = state.get("sample")
        if proj is None:
            return "skip", "no sample film to cut", {}
        timeline = proj.body["timeline"]
        shots = timeline["shots"]
        if len(shots) != len(proj.p["shots"]) or not all(s["ready"] for s in shots):
            return "fail", f"timeline shots {len(shots)}, ready {[s['ready'] for s in shots]}", {}
        if any(b["start"] < a["start"] or b["end"] <= b["start"] for a, b in zip(shots, shots[1:])):
            return "fail", "timeline shot starts are not in order", {}
        if abs(timeline["total"] - shots[-1]["end"]) > 0.05:
            return "fail", f"total {timeline['total']} does not match the last shot end {shots[-1]['end']}", {}
        for shot in proj.p["shots"]:
            take = next(t for t in shot["takes"] if t["id"] == shot["activeTake"])
            code, raw = suite._voice_http("GET", f"/api/storyboard/projects/{proj.id}/files?path={urllib.parse.quote(take['clip'])}")
            if code != 200 or b"ftyp" not in raw[:64]:
                return "fail", f"clip {take['clip']}: HTTP {code}", {}
        return "pass", "", {"total": timeline["total"]}

    def _caption_look():
        proj = state.get("sample")
        if proj is None:
            return "skip", "no sample film", {}
        folder = store_folder(proj.id)
        if folder is None:
            return "skip", "the backend's project folder is not on this machine (set CHAOSENGINE_STORYBOARDS_DIR as for the backend)", {}
        proj.op("set_brand", {"captions": {"colour": "#ffcc00", "position": "top", "font": "Helvetica"}})
        proj.op("edit_project", {"export": {"captionsBurn": True}})
        proj.op("add_caption", {"start": 0.2, "end": 1.5, "text": "灯台が光る"})
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from backend_service.storyboard.models import Project
        from backend_service.storyboard.timeline import build_render_plan

        plan = build_render_plan(Project.model_validate(proj.reload()), True, folder)
        look = plan.get("captions_style") or {}
        if (look.get("position"), look.get("colour"), look.get("font")) != ("top", "#FFCC00", "Helvetica"):
            return "fail", f"the render plan's caption look is {look}, not the brand kit's", {}
        return "pass", "", {"look": look}

    def _reconcile():
        proj = Proj("E2E 0.10 crash")
        proj.op("add_shot", {"title": "half made", "stillPrompt": "x"})
        folder = store_folder(proj.id)
        if folder is None:
            return "skip", "the backend's project folder is not on this machine (set CHAOSENGINE_STORYBOARDS_DIR as for the backend)", {}
        path = folder / "project.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        sid = data["shots"][0]["id"]
        data["shots"][0]["status"] = "running"
        data["batches"] = [{"id": "b_e2edead0", "what": "still", "force": False, "draft": False, "state": "running",
                            "createdAt": time.time() - 60, "results": [{"shotId": sid, "kind": "still", "state": "running"}]}]
        path.write_text(json.dumps(data), encoding="utf-8")  # what a power cut mid-batch leaves on disk
        got = proj.reload()
        batch = got["batches"][0]
        if got["shots"][0]["status"] != "idle" or batch["state"] != "interrupted" or batch["results"][0]["state"] != "cancelled":
            return "fail", f"after opening: shot {got['shots'][0]['status']}, batch {batch['state']}, result {batch['results'][0]['state']}", {}
        return "pass", "start-up pass: tests/test_storyboard_reconcile.py (needs a restart)", {}

    def _workspace_field():
        code, body = api("GET", "/api/workspace")
        runs = ((body or {}).get("system") or {}).get("activeGenerations")
        if code != 200 or not isinstance(runs, list):
            return "fail", f"/api/workspace system.activeGenerations is {type(runs).__name__}", {}
        for run in runs:
            if run.get("kind") not in ("image", "video", "music") or "runLabel" not in run:
                return "fail", f"an active generation lacks kind or runLabel: {run}", {}
        return "pass", "", {"running": len(runs)}

    def _open_route():
        def refused(path: str, want: int) -> str | None:
            code, body = api("POST", "/api/files/open", {"path": path})
            return None if code == want else f"{path}: HTTP {code}, wanted {want} ({str(body)[:120]})"

        outside = base / "outside.png"
        outside.write_bytes(png(8, 8, [(1, 2, 3)]))
        script = base / "run.sh"
        script.write_text("echo hi\n")
        problems = [refused(str(outside), 400), refused(str(script), 400), refused(str(base / "gone.png"), 404),
                    refused(str(base / ".." / base.name / "outside.png"), 400)]
        proj = state.get("main")
        folder = store_folder(proj.id) if proj else None
        note = ""
        if folder is not None:
            link = folder / "exports" / "e2e-link.png"
            link.parent.mkdir(exist_ok=True)
            link.symlink_to(outside)
            try:
                problems.append(refused(str(link), 400))  # a link inside the app's folders that points outside
            finally:
                link.unlink()
        else:
            note = "symlink case not run: the backend's project folder is not on this machine"
        problems = [p for p in problems if p]
        return ("fail", "; ".join(problems), {}) if problems else ("pass", note, {})

    def _plan_card_precheck():
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        old = os.environ.get("CHAOSENGINE_STORYBOARDS_DIR")
        os.environ["CHAOSENGINE_STORYBOARDS_DIR"] = str(base / "gate-store")
        try:
            from backend_service.storyboard.ops import run_op
            from backend_service.storyboard.store import ProjectStore
            from backend_service.tools import studio_gate
            from backend_service.tools.studio import precheck

            store = ProjectStore(base / "gate-store")
            project = store.create("Gate check")
            project = run_op(store, project.id, project.revision, "add_shot", {"title": "a", "stillPrompt": "tea"})
            good = {"project_id": project.id, "revision": project.revision, "what": "still"}
            cases = {"stale revision": {**good, "revision": 1}, "draft final": {**good, "what": "final", "draft": True},
                     "unknown shot": {**good, "shot_ids": ["s_nope"]}, "bad kind": {**good, "what": "stills"}}
            for label, args in cases.items():
                if not precheck("storyboard_generate", args):
                    return "fail", f"{label}: the pre-check let a call that cannot run through", {}
            if precheck("storyboard_generate", good):
                return "fail", f"a good call was refused: {precheck('storyboard_generate', good)}", {}
            gate = studio_gate.StudioGate("e2e", poll_seconds=0.01, is_cancelled=lambda: True)
            call = {"id": "c1", "type": "function",
                    "function": {"name": "storyboard_generate", "arguments": json.dumps(cases["stale revision"])}}
            review = gate.review([call])
            events = []
            try:
                while True:
                    events.append(next(review))
            except StopIteration as done:
                answers = done.value or {}
            if any("studio_plan" in e for e in events) or "c1" not in answers:
                return "fail", "a call with a stale revision still got a plan card", {}
            return "pass", "in-process gate, temporary store, no chat model", {}
        finally:
            if old is None:
                os.environ.pop("CHAOSENGINE_STORYBOARDS_DIR", None)
            else:
                os.environ["CHAOSENGINE_STORYBOARDS_DIR"] = old

    def _plan_card_reload():
        """S-J: the turn route the reloaded window asks, a waiting plan listed for its chat and answered there, and
        a refused call's reason kept in the next turn's history."""
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from backend_service.state._helpers import _build_history_with_reasoning
        from backend_service.tools import studio_gate

        code, body = api("GET", "/api/chat/studio/sessions/e2e-no-such-chat/turn")
        if code != 200 or (body or {}).get("running") is not False or (body or {}).get("plans") != []:
            return "fail", f"turn route answered {code} {body}", {}
        gate = studio_gate.StudioGate("e2e-reload", poll_seconds=0.01)
        call = {"id": "c1", "type": "function", "function": {"name": "generate_image", "arguments": json.dumps({"prompt": "rose"})}}
        review = gate.review([call])
        first = next(review)
        listed = studio_gate.pending_for_session("e2e-reload")
        if "studio_plan" not in first or [p["planId"] for p in listed] != [first["studio_plan"]["planId"]]:
            review.close()
            return "fail", "a waiting plan is not listed for its chat", {}
        studio_gate.decide(listed[0]["planId"], {"c1": "approve"})
        try:
            while True:
                next(review)
        except StopIteration as done:
            answers = done.value or {}
        if answers or studio_gate.pending_for_session("e2e-reload"):
            return "fail", f"the answered plan did not let the call run: {answers}", {}
        reason = "'what' must be one of still, clip, both."
        history = _build_history_with_reasoning([{"role": "assistant", "text": "", "toolCalls": [
            {"name": "storyboard_generate", "renderAs": "studio", "data": {"error": reason}}]}], preserve_reasoning=False)
        if reason not in history[0]["text"]:
            return "fail", "a refused call's reason is not in the next turn's history", {}
        return "pass", "turn route on the backend; gate and history in this process, no chat model", {}

    # ---- real models (skip when not installed) ----
    def _models() -> tuple[str | None, bool]:
        if "models" not in state:
            code, body = api("GET", "/api/storyboard/still-models?width=512&height=512")
            fast = (body or {}).get("fast") if code == 200 else None
            probe = Proj("E2E 0.10 probe")
            probe.op("add_shot", {"title": "p", "motionPrompt": "waves", "seconds": 2})
            code, est = api("POST", f"/api/storyboard/projects/{probe.id}/estimate", {"what": "clip"})
            video = code == 200 and not est.get("missing")
            state["models"] = (fast, video)
        return state["models"]

    def _draft_run():
        fast, video = _models()
        if not fast:
            return "skip", "no still model installed (Images > Models)", {}
        proj = Proj("E2E 0.10 drafts", {"width": 512, "height": 512, "resolution": "480p"})
        state["run"] = proj
        proj.op("add_shot", {"title": "lighthouse", "stillPrompt": "a white lighthouse on a cliff at dusk", "motionPrompt": "the beam turns slowly", "seconds": 2})
        what = "both" if video else "still"
        code, body = api("POST", f"/api/storyboard/projects/{proj.id}/batches", {"revision": proj.rev, "what": what, "draft": True})
        if code != 200:
            return "fail", f"draft batch: HTTP {code} {str(body)[:160]}", {}
        seen: list[str] = []

        def watch(_status):
            code, ws = api("GET", "/api/workspace")
            for run in ((ws or {}).get("system") or {}).get("activeGenerations") or []:
                seen.append(run.get("kind"))

        started = time.time()
        record = wait_batch(proj.id, body["batchId"], 900, watch)
        took = time.time() - started
        shot = proj.reload()["shots"][0]
        take = next((t for t in shot["takes"] if t["id"] == shot["activeTake"]), None)
        if record.get("state") != "done" or take is None or take.get("renderPass") != "draft" or not record.get("draft"):
            return "fail", f"batch {record.get('state')}, take {take and take.get('renderPass')}, errors {[r.get('error') for r in record.get('results', [])]}", {}
        if "image" not in seen and took > 8:
            return "fail", f"/api/workspace never listed the image engine during a {took:.0f} s draft", {}
        if video and not take.get("clip"):
            return "fail", "drafts with clips made no clip", {}
        return "pass", "" if video else "no video model: stills only", {"seconds": round(took, 1), "busyKinds": sorted(set(k for k in seen if k))}

    def _cancel_keeps_take():
        proj = state.get("run")
        if proj is None or not _models()[1]:
            return "skip", "needs the draft run and an installed video model", {}
        shot = proj.reload()["shots"][0]
        code, body = api("POST", f"/api/storyboard/projects/{proj.id}/batches", {"revision": proj.rev, "what": "redo-clip"})
        if code != 200:
            return "fail", f"redo-clip: HTTP {code} {str(body)[:160]}", {}
        bid = body["batchId"]
        deadline = time.time() + 120
        while time.time() < deadline:
            _, status = api("GET", f"/api/storyboard/projects/{proj.id}/batches/{bid}")
            if (status or {}).get("batch", {}).get("results", [{}])[0].get("state") == "running":
                time.sleep(3)
                break
            time.sleep(0.5)
        api("POST", f"/api/storyboard/projects/{proj.id}/batches/{bid}/cancel")
        record = wait_batch(proj.id, bid, 120)
        after = proj.reload()["shots"][0]
        if record.get("state") != "cancelled" or after["activeTake"] != shot["activeTake"] or after["status"] in ("queued", "running"):
            return "fail", f"batch {record.get('state')}, take kept {after['activeTake'] == shot['activeTake']}, status {after['status']}", {}
        return "pass", "", {}

    def _redo_still_stale():
        proj = state.get("run")
        if proj is None or not _models()[1]:
            return "skip", "needs the draft run and an installed video model (the rule is about a clip behind its still)", {}
        before = proj.reload()["shots"][0]
        code, body = api("POST", f"/api/storyboard/projects/{proj.id}/batches", {"revision": proj.rev, "what": "redo-still"})
        if code != 200:
            return "fail", f"redo-still: HTTP {code} {str(body)[:160]}", {}
        record = wait_batch(proj.id, body["batchId"], 600)
        shot = proj.reload()["shots"][0]
        take = next(t for t in shot["takes"] if t["id"] == shot["activeTake"])
        old = next(t for t in shot["takes"] if t["id"] == before["activeTake"])
        if record.get("state") != "done" or take["id"] == old["id"] or take.get("clip") != old.get("clip"):
            return "fail", f"redo-still: batch {record.get('state')}, the new take does not keep the old clip", {}
        if shot["status"] != "stale":
            return "fail", f"a clip behind its new still shows {shot['status']}, wanted stale", {}
        code, est = api("POST", f"/api/storyboard/projects/{proj.id}/estimate", {"what": "both"})
        code2, still_est = api("POST", f"/api/storyboard/projects/{proj.id}/estimate", {"what": "still"})
        if est.get("clips") != 1 or est.get("stills") != 0 or still_est.get("stills") != 0:
            return "fail", f"Make would redo stills {est.get('stills')} clips {est.get('clips')}; a stills batch {still_est.get('stills')}", {}
        if take.get("settings", {}).get("stillSeed") == old.get("settings", {}).get("stillSeed") is not None:
            return "fail", "the redone still kept the old seed", {}
        return "pass", "", {"takes": len(shot["takes"])}

    try:
        for name, fn in [
            ("create a project; plan, brand kit and Keep for final undo and redo", _undo_chain),
            ("plan ops: apply a plan, changes in one step, one undo each, the chat kept", _plan_ops),
            ("plan routes: status, no brief, no chat model", _plan_routes),
            ("style pictures give a palette; logo and caption look; the bundle keeps them", _style_brand),
            ("draft and final kinds: the final pass counts kept shots; a draft final is refused", _passes_free),
            ("redo kinds: estimate, nothing to redo on an upload shot", _redo_free),
            ("take compare: Use this one picks a take, undo switches back", _compare),
            ("sample film installs as a finished project", _sample),
            ("rough-cut inputs: the timeline and every clip of the sample", _rough_cut),
            ("the render plan burns captions in the brand kit's look", _caption_look),
            ("a project left running by a crash is reset when opened", _reconcile),
            ("/api/workspace carries activeGenerations", _workspace_field),
            ("the Open route refuses files outside the app folders, non-media and symlinks out", _open_route),
            ("Chat plan card: a call that cannot run gets no card", _plan_card_precheck),
            ("Chat plan card: a waiting card is listed for a reloaded window and answered there", _plan_card_reload),
            ("real draft with the fast still model; the busy engine shows in /api/workspace", _draft_run),
            ("a cancelled Redo clip keeps the take in use (FU-181)", _cancel_keeps_take),
            ("Redo still keeps the old clip and marks it behind its still", _redo_still_stale),
        ]:
            needs_main = fn in (_plan_ops, _plan_routes, _style_brand, _passes_free, _redo_free, _compare)
            if needs_main and "main" not in state:
                phase.checks.append(suite.CheckResult(name, "skip", reason="no project"))
                continue
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
