"""Phase 16: Storyboard v2 (captions, OpenTimelineIO, stems, burned-in captions, resumable render).

No model runs: the clips and the voice line are made with the pinned ffmpeg, captions are typed in (the
speech-to-text path is covered by a unit test with a stand-in and is only run here when
``E2E_CAPTIONS_STT=1`` and a speech model is installed). A render killed with SIGKILL and resumed is covered
by ``tests/test_storyboard_v2.py``; this phase cancels one instead, because it must never stop a process
it did not start.

``run(h, cap)`` is called by ``phase_16`` in ``scripts/e2e_test_suite.py`` with the suite module as ``h``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any


def run(h: Any, cap: Any) -> Any:
    phase = h.PhaseResult(phase=16, name="Storyboard v2")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(h.CheckResult("phase 16", "skip", reason="backend not reachable"))
        return phase

    def call(method: str, path: str, payload: Any = None) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        code, data = h._voice_http(method, f"/api/storyboard{path}", body=body,
                                   content_type="application/json" if body else None, timeout=120.0)
        try:
            return code, json.loads(data.decode("utf-8")) if data.strip() else None
        except ValueError:
            return code, data

    code, _ = call("GET", "/projects")
    if code == 404:
        phase.status = "skip"
        phase.checks.append(h.CheckResult("phase 16", "skip", reason="storyboard not present in this build"))
        return phase
    try:
        import sys

        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)  # the suite runs from scripts/; the pinned ffmpeg lookup lives in the repo
        from backend_service.storyboard.ffmpeg_bin import ffmpeg_exe

        exe = ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        phase.status = "skip"
        phase.checks.append(h.CheckResult("phase 16", "skip", reason=f"no bundled ffmpeg here: {exc}"))
        return phase

    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    state: dict[str, Any] = {}

    def ff(*args: str) -> None:
        subprocess.run([exe, "-y", "-loglevel", "error", "-nostdin", *args], check=True, capture_output=True)

    def op(name: str, args: dict[str, Any]) -> Any:
        code, body = call("POST", f"/projects/{state['id']}/ops", {"revision": state["rev"], "op": name, "args": args})
        if code != 200:
            raise RuntimeError(f"{name}: HTTP {code} {str(body)[:200]}")
        state["rev"] = body["project"]["revision"]
        return body

    def wait_render(timeout: float = 180.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        job: dict[str, Any] = {}
        while time.time() < deadline:
            _, body = call("GET", f"/projects/{state['id']}/render")
            job = (body or {}).get("render") or {}
            if job.get("state") in ("done", "failed", "cancelled", "interrupted"):
                return job
            time.sleep(0.5)
        raise RuntimeError(f"render did not finish: {job}")

    def start_export() -> int:
        code, _ = call("POST", f"/projects/{state['id']}/render",
                       {"revision": state["rev"], "preview": False, "notify": False})
        return code

    def build():
        voice = base / "voice.wav"
        ff("-f", "lavfi", "-i", "sine=f=700:d=3", str(voice))
        code, body = call("POST", "/projects", {"name": "E2E v2 film"})
        if code != 200:
            return "fail", f"create: HTTP {code}", {}
        state["id"], state["rev"] = body["project"]["id"], body["project"]["revision"]
        for i in range(3):
            clip = base / f"c{i}.mp4"
            ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=1.5", "-f", "lavfi", "-i", "sine=f=440:d=1.5",
               "-vf", f"hue=h={i * 90}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip))
            body = op("add_shot", {"title": f"shot {i + 1}", "seconds": 1.5})
            op("import_take", {"shotId": body["project"]["shots"][i]["id"], "clip": str(clip)})
        state["first"] = body["project"]["shots"][0]["id"]
        body = op("add_audio", {"kind": "voice", "path": str(voice), "anchor": {"shotId": state["first"], "offset": 0.0}})
        state["voice"] = body["project"]["audio"][0]["id"]
        return "pass", "", {"projectId": state["id"]}

    def captions():
        op("set_captions", {"clipId": state["voice"], "cues": [{"start": 0.2, "end": 1.6, "text": "First line"},
                                                                {"start": 1.8, "end": 2.8, "text": "Second line"}]})
        code, body = call("POST", f"/projects/{state['id']}/captions/export")
        if code != 200 or body["cues"] != 2:
            return "fail", f"captions export: HTTP {code} {str(body)[:160]}", {}
        code, raw = h._voice_http("GET", f"/api/storyboard/projects/{state['id']}/files?path={urllib.parse.quote(body['file'])}")
        text = raw.decode("utf-8")
        if code != 200 or "00:00:00,200 --> 00:00:01,600" not in text or "Second line" not in text:
            return "fail", f"the .srt is wrong: {text[:160]!r}", {}
        return "pass", "", {"file": body["file"]}

    def otio():
        code, body = call("POST", f"/projects/{state['id']}/otio")
        if code != 200:
            return "fail", f"otio export: HTTP {code} {str(body)[:160]}", {}
        code, raw = h._voice_http("GET", f"/api/storyboard/projects/{state['id']}/files?path={urllib.parse.quote(body['file'])}")
        doc = json.loads(raw.decode("utf-8"))
        names = [t["name"] for t in doc["tracks"]["children"]]
        if doc["OTIO_SCHEMA"] != "Timeline.1" or names[:2] != ["Picture", "Shot sound"] or "Voice" not in names:
            return "fail", f"unexpected OTIO layout {names}", {}
        return "pass", "", {"tracks": names}

    def export_all():
        op("edit_project", {"export": {"captionsBurn": True, "captionsSrt": True, "stems": True}})
        code = start_export()
        if code != 200:
            return "fail", f"render: HTTP {code}", {}
        job = wait_render()
        if job["state"] != "done":
            return "fail", f"render {job['state']}: {job.get('error')}", {}
        extras = sorted(job.get("extras", []))
        want = {"srt": any(e.endswith(".srt") for e in extras),
                "voice stem": any(e.endswith("-stem-voice.wav") for e in extras),
                "shot sound stem": any(e.endswith("-stem-shot-sound.wav") for e in extras)}
        if not all(want.values()):
            return "fail", f"missing outputs {want} in {extras}", {}
        return "pass", "", {"warnings": job.get("warnings", []), "extras": extras}

    def cancel_and_resume():
        op("edit_shot", {"shotId": state["first"], "trimIn": 0.1})  # a new hash: its intermediate is rebuilt
        if start_export() != 200:
            return "fail", "could not start the export", {}
        call("POST", f"/projects/{state['id']}/render/cancel")
        job = wait_render()
        if job["state"] == "done":
            return "skip", "the render finished before it could be cancelled", {}
        if job["state"] != "cancelled" or not job.get("resumable"):
            return "fail", f"after cancel: {job}", {}
        if start_export() != 200:
            return "fail", "could not resume the export", {}
        job = wait_render()
        if job["state"] != "done":
            return "fail", f"resume ended {job['state']}: {job.get('error')}", {}
        return "pass", "", {}

    def stt():
        if os.environ.get("E2E_CAPTIONS_STT") != "1":
            return "skip", "set E2E_CAPTIONS_STT=1 with a speech model installed to run the speech-to-text path", {}
        code, body = call("POST", f"/projects/{state['id']}/captions/generate", {"revision": state["rev"]})
        if code != 200:
            return "fail", f"generate: HTTP {code} {str(body)[:200]}", {}
        return "pass", "", {"notes": body.get("notes")}

    first = "build a 3-shot film with a voice clip"
    try:
        for name, fn in [
            (first, build),
            ("captions: edit and write an .srt", captions),
            ("export for an editor (.otio)", otio),
            ("export with burned-in captions, an .srt and lane stems", export_all),
            ("cancel an export, then resume it", cancel_and_resume),
            ("make captions from the voice clip (speech model)", stt),
        ]:
            if "id" not in state and name != first:
                phase.checks.append(h.CheckResult(name, "skip", reason="no project"))
                continue
            phase.checks.append(h._check(name, fn))
    finally:
        if "id" in state:
            code, body = call("GET", f"/projects/{state['id']}")
            if code == 200:
                h._voice_http("DELETE", f"/api/storyboard/projects/{state['id']}?revision={body['project']['revision']}")
        tmp.cleanup()
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
