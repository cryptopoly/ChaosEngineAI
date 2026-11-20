"""Phase 22: lip sync (D-103, LTX-2.3 audio to video) through POST /api/video/generate with ``audioPath``.

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)``. The contract checks need only a backend; the real
shot needs an NVIDIA card with the 4-bit LTX-2.3 row downloaded (``OzzyGT/LTX-2.3-Distilled-bnb-nf4``), a Kokoro
voice and an installed image model, and skips with the reason when any is missing. It makes one 5 s shot (about two
minutes on an RTX 4090) and checks the mp4 has the vocal as its sound, the planned length and a picture that moves.
Mouth-to-audio sync itself is not scored here (that needs a face landmarker the app does not ship, FU-334).
Two more contract checks need no model: the ``make_lipsync_shot`` agent tool is listed and refuses a missing still in
words (Agent access is switched on for the call and put back), and a Storyboard shot takes a WAV vocal through
``set_vocal_clip``, refuses a text file and clears it (a throwaway project, deleted afterwards).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

CUDA_REPO = "OzzyGT/LTX-2.3-Distilled-bnb-nf4"
LINE = "I never thought the city lights would look this bright tonight, but here we are."


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=22, name="Lip sync (LTX-2.3 audio to video)")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 22", "skip", reason="backend not reachable"))
        return phase
    root = str(Path(__file__).resolve().parents[2])
    import sys

    if root not in sys.path:
        sys.path.insert(0, root)

    def api(method: str, path: str, body: Any = None, timeout: float = 120.0) -> tuple[int, Any]:
        raw_body = json.dumps(body).encode() if body is not None else None
        code, raw = suite._voice_http(method, path, body=raw_body, content_type="application/json" if raw_body else None,
                                      timeout=timeout)
        try:
            return code, json.loads(raw.decode("utf-8")) if raw.strip() else None
        except (ValueError, UnicodeDecodeError):
            return code, raw

    state: dict[str, Any] = {}

    def catalog_row():
        code, data = api("GET", "/api/video/catalog")
        rows = [v for f in (data or {}).get("families", []) for v in f.get("variants", [])] if code == 200 else []
        row = next((v for v in rows if v.get("repo") == CUDA_REPO), None)
        if row is None:
            return "fail", f"HTTP {code}: no {CUDA_REPO} row", {}
        block = row.get("lipSync") or {}
        if block.get("basis") != "measured" or not block.get("secondsPerShot"):
            return "fail", f"lipSync block missing or unmeasured: {block}", {}
        state["row"] = row
        return "pass", "", {"secondsPerShot": block["secondsPerShot"], "installed": bool(row.get("availableLocally"))}

    def refuses_a_row_without_lip_sync():
        code, data = api("POST", "/api/video/generate", {
            "modelId": "diffusers/LTX-2.3-Distilled-Diffusers", "prompt": "x", "numFrames": 121,
            "startImagePath": "nope.png", "audioPath": "nope.wav"})
        detail = str((data or {}).get("detail") if isinstance(data, dict) else data)
        # 409 when that row is not downloaded (checked first), 400 (cannot lip-sync) when it is.
        if code in (400, 409, 404):
            return "pass", "", {"http": code, "detail": detail[:120]}
        return "fail", f"HTTP {code}: {detail[:200]}", {}

    def real_shot():
        row = state.get("row")
        if not row or not row.get("availableLocally"):
            return "skip", f"{CUDA_REPO} is not downloaded", {}
        from backend_service.helpers.gpu import get_device_vram_total_gb

        card = get_device_vram_total_gb()
        if not card or card < 22.8:
            return "skip", f"needs a 24 GB NVIDIA card (found {card or 'none'})", {}
        folder = Path(tempfile.mkdtemp(prefix="e2e-lipsync-"))
        code, wav = suite._voice_http("POST", "/api/voice/synthesize",
                                      body=json.dumps({"text": LINE, "voice": "af_heart"}).encode(),
                                      content_type="application/json", timeout=300.0)
        if code != 200 or not wav.startswith(b"RIFF"):
            return "skip", f"no Kokoro voice (HTTP {code})", {}
        (folder / "line.wav").write_bytes(wav)
        code, models = api("GET", "/api/images/library")
        rows = sorted((models or {}).get("models", []) if code == 200 else [], key=lambda m: float(m.get("sizeGb") or 1e9))
        image_ids = [m.get("id") for m in rows]  # the smallest first: the still only has to show a face
        if not image_ids:
            return "skip", "no image model installed for the start still", {}
        code, img = api("POST", "/api/images/generate", {
            "modelId": image_ids[0], "prompt": "head-and-shoulders portrait of a woman facing the camera, studio light",
            "width": 768, "height": 512, "seed": 78}, timeout=900.0)
        if code != 200:
            return "fail", f"still: HTTP {code} {str(img)[:200]}", {}
        still = img["artifacts"][0]["imagePath"]
        code, res = api("POST", "/api/video/generate", {
            "modelId": CUDA_REPO, "prompt": "A woman talks to the camera, natural head movement, static camera.",
            "width": 768, "height": 512, "numFrames": 121, "fps": 24, "seed": 42,
            "startImagePath": still, "audioPath": str(folder / "line.wav")}, timeout=3600.0)
        if code == 503:
            return "skip", f"memory gate: {res.get('detail') if isinstance(res, dict) else res}", {}
        if code != 200:
            return "fail", f"HTTP {code}: {str(res)[:300]}", {}
        art = res["artifact"]
        if not art.get("hasAudio"):
            return "fail", "the clip has no sound track", {}
        if abs(float(art.get("clipDurationSeconds") or 0) - 121 / 24) > 0.01:
            return "fail", f"length {art.get('clipDurationSeconds')} s, planned {121 / 24:.3f}", {}
        motion = _frame_motion(Path(art["videoPath"]))
        if motion is not None and motion < 0.5:
            return "fail", f"the picture barely changes (mean frame difference {motion:.2f})", {}
        return "pass", "", {"timings": art.get("timings"), "frameMotion": motion}

    def agent_tool():
        # make_lipsync_shot on the agent bridge: listed, and a missing still is refused in words before any job starts.
        code, access = api("GET", "/api/agent/access")
        if code == 404:
            return "skip", "agent bridge not present in this build", {}
        was_on = bool(isinstance(access, dict) and access.get("enabled"))
        api("POST", "/api/agent/access", {"enabled": True})
        try:
            code, listed = api("POST", "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            names = {t["name"] for t in ((listed or {}).get("result") or {}).get("tools", [])} if code == 200 else set()
            if "make_lipsync_shot" not in names:
                return "fail", f"make_lipsync_shot not in tools/list (HTTP {code})", {}
            code, called = api("POST", "/mcp", {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "make_lipsync_shot",
                "arguments": {"prompt": "x", "start_image": "e2e-no-such-still.png", "audio": "e2e-no-such-vocal.wav"}}})
            result = (called or {}).get("result") or {}
            text = str(((result.get("content") or [{}])[0]).get("text"))
            if code != 200 or not result.get("isError"):
                return "fail", f"a missing still did not refuse: HTTP {code} {text[:200]}", {}
            return "pass", "", {"refusal": text[:120]}
        finally:
            api("POST", "/api/agent/access", {"enabled": was_on})

    def vocal_clip_op():
        # Storyboard Shot.vocalClip: set_vocal_clip takes a WAV, refuses a file that is not WAV or FLAC, and clears.
        import wave

        code, made = api("POST", "/api/storyboard/projects", {"name": "E2E lip-sync vocal"})
        if code == 404:
            return "skip", "storyboard not present in this build", {}
        if code != 200:
            return "fail", f"create project: HTTP {code}", {}
        pid, rev = made["project"]["id"], made["project"]["revision"]
        folder = Path(tempfile.mkdtemp(prefix="e2e-vocal-"))
        try:
            def op(name: str, args: dict[str, Any]) -> tuple[int, Any]:
                nonlocal rev
                code, body = api("POST", f"/api/storyboard/projects/{pid}/ops", {"revision": rev, "op": name, "args": args})
                if code == 200:
                    rev = body["project"]["revision"]
                return code, body

            code, body = op("add_shot", {"title": "sung line", "seconds": 4})
            sid = body["project"]["shots"][-1]["id"]
            vocal = folder / "vocal.wav"
            with wave.open(str(vocal), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(b"\x00\x00" * 16000)
            code, body = op("set_vocal_clip", {"shotId": sid, "audio": str(vocal)})
            shot = next((s for s in (body or {}).get("project", {}).get("shots", []) if s["id"] == sid), {}) if code == 200 else {}
            if code != 200 or not str(shot.get("vocalClip") or "").startswith(f"media/{sid}/vocal-"):
                return "fail", f"set_vocal_clip: HTTP {code} vocalClip={shot.get('vocalClip')!r}", {}
            text_file = folder / "notes.txt"
            text_file.write_text("not audio", encoding="utf-8")
            code, body = op("set_vocal_clip", {"shotId": sid, "audio": str(text_file)})
            if code != 400:
                return "fail", f"a text file as the vocal answered HTTP {code}, wanted 400", {}
            code, body = op("set_vocal_clip", {"shotId": sid, "audio": ""})
            shot = next((s for s in (body or {}).get("project", {}).get("shots", []) if s["id"] == sid), {}) if code == 200 else {}
            if code != 200 or shot.get("vocalClip") is not None:
                return "fail", f"clearing the vocal: HTTP {code} vocalClip={shot.get('vocalClip')!r}", {}
            return "pass", "", {"shot": sid}
        finally:
            api("DELETE", f"/api/storyboard/projects/{pid}?revision={rev}")
            for item in folder.iterdir():
                item.unlink(missing_ok=True)
            folder.rmdir()

    for name, fn in [
        ("the 4-bit LTX-2.3 row carries the measured lip-sync figures", catalog_row),
        ("audioPath on a row that cannot lip-sync is refused", refuses_a_row_without_lip_sync),
        ("make_lipsync_shot is on the agent bridge and refuses a missing still in words", agent_tool),
        ("a Storyboard shot takes a WAV vocal, refuses a text file and clears it", vocal_clip_op),
        ("a Kokoro line and a still make a 5 s lip-sync clip with the line as its sound", real_shot),
    ]:
        phase.checks.append(suite._check(name, fn))
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase


def _frame_motion(video: Path) -> float | None:
    """Mean absolute difference between consecutive frames (0-255 scale) at 64x48 grey, or None without ffmpeg."""
    from backend_service.storyboard import ffmpeg_bin

    found = ffmpeg_bin.find_ffmpeg(allow_path=True)
    if found is None:
        return None
    raw = subprocess.run([found[0], "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", "scale=64:48,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, timeout=300).stdout
    size = 64 * 48
    frames = [raw[i:i + size] for i in range(0, len(raw) - size + 1, size)]
    if len(frames) < 2:
        return 0.0
    diffs = [sum(abs(a - b) for a, b in zip(f1, f2)) / size for f1, f2 in zip(frames, frames[1:])]
    return round(sum(diffs) / len(diffs), 3)
