"""Phase 11 checks (FU-139, FU-274): the music runtime's lane matches the host, and hosts without a lane refuse install;
``real_sound`` makes one real Stable Audio 3 clip on CUDA when the sounds pack and its weights are there.

Called from ``phase_11`` in ``scripts/e2e_test_suite.py`` as ``check(call, runtime)`` where ``call(method, path,
payload)`` returns ``(status, json)`` against ``/api/music``. Three paths:

* Apple Silicon: lane ``mlx``, no reason, no code.
* An NVIDIA host (lane ``cuda``, Windows or Linux x86_64): music and sounds have no reason or code (FU-274: the
  sounds lane is ``cuda`` too), the ACE-Step rows and every sound row list ``cuda``, and the three Stable Audio 3
  rows are in the catalog, gated, with a measured CUDA estimate.
* Anything else: the single music sentence and ``music.unsupportedPlatform``, the sounds lane says the same, and
  ``POST /install`` starts nothing (the job answers ``error`` with that reason).

It never starts an install or a download on a host where music runs.
"""

from __future__ import annotations

import platform
from typing import Any, Callable

REASON_START = "Music generation needs an Apple Silicon Mac or an NVIDIA GPU"
CODE = "music.unsupportedPlatform"
SA3_IDS = ("stabilityai/stable-audio-3-small-sfx", "stabilityai/stable-audio-3-small-music", "stabilityai/stable-audio-3-medium")


def _rows(call: Callable[..., tuple[int, Any]]) -> tuple[int, list[dict[str, Any]]]:
    code, body = call("GET", "/catalog")
    return code, [v for f in (body or {}).get("families", []) for v in f.get("variants", [])]


def _cuda(call: Callable[..., tuple[int, Any]], runtime: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if runtime.get("unavailableReason") or runtime.get("unavailableCode"):
        return "fail", f"music is refused on an NVIDIA host: {runtime.get('unavailableReason')!r}", {}
    sound = runtime.get("sound") or {}
    if sound.get("lane") != "cuda" or sound.get("unavailableReason") or sound.get("unavailableCode"):
        return "fail", f"the sounds lane is not open on CUDA: {sound.get('lane')!r} {sound.get('unavailableReason')!r}", {}
    code, rows = _rows(call)
    if code != 200 or any("cuda" not in (v.get("platforms") or []) for v in rows):
        return "fail", f"catalog HTTP {code}; a row does not list cuda", {}
    by_id = {v.get("id"): v for v in rows}
    for model_id in SA3_IDS:
        row = by_id.get(model_id) or {}
        cuda = (row.get("estimates") or {}).get("cuda") or {}
        if not (row.get("licence") or {}).get("gated") or cuda.get("basis") != "measured" or not cuda.get("peakGb"):
            return "fail", f"{model_id}: missing, not gated, or no measured CUDA estimate", {}
    sounds = [v for v in rows if v.get("kind") == "sound"]
    return "pass", "", {"path": "cuda", "rows": len(rows), "soundRows": len(sounds), "installed": runtime.get("installed"),
                        "soundsInstalled": sound.get("installed")}


def real_sound(call: Callable[..., tuple[int, Any]], fetch: Callable[[str], tuple[int, bytes]],
               runtime: dict[str, Any], cleanup: list[str]) -> tuple[str, str, dict[str, Any]]:
    """FU-274: a real 5 s Stable Audio 3 Small SFX clip on CUDA: a 44.1 kHz stereo FLAC of the asked length, a VRAM
    peak under 24 GB, and (when ``soundfile`` is importable here) not silent."""
    sound = runtime.get("sound") or {}
    if sound.get("lane") != "cuda":
        return "skip", "not the CUDA sounds lane", {}
    code, rows = _rows(call)
    row = next((v for v in rows if v.get("id") == SA3_IDS[0]), None)
    if not sound.get("installed") or not row or not row.get("availableLocally"):
        return "skip", "sounds pack or Stable Audio 3 Small SFX weights not installed", {}
    code, body = call("POST", "/sound/generate", {"caption": "e2e check, glass shattering on a stone floor", "seconds": 5,
                                                  "seed": 11, "modelId": SA3_IDS[0], "title": "E2E sound"})
    if code != 200 or not (body or {}).get("items"):
        return "fail", f"sound generate: HTTP {code} {str(body)[:200]}", {}
    item = body["items"][0]
    cleanup.append(item["id"])
    if (item.get("sampleRate"), item.get("channels")) != (44100, 2) or abs(float(item.get("durationS") or 0) - 5) > 0.05:
        return "fail", f"sound facts wrong: {item}", {}
    vram = (body.get("timings") or {}).get("vram_peak_gb")
    if not (isinstance(vram, (int, float)) and 0 < vram <= 24):
        return "fail", f"CUDA VRAM peak missing or above 24 GB: {vram!r}", {}
    code, data = fetch(f"/api/music/library/{item['id']}/audio")
    if code != 200 or not data.startswith(b"fLaC"):
        return "fail", f"audio: HTTP {code}", {}
    rms_db = None
    try:
        import io

        import numpy as np
        import soundfile

        samples, _rate = soundfile.read(io.BytesIO(data), dtype="float32")
        rms_db = round(float(20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-12)), 1)
        if rms_db < -60:
            return "fail", f"the clip is silent (RMS {rms_db} dBFS)", {}
    except ImportError:
        pass
    return "pass", "", {"jobSeconds": body.get("seconds"), "vramPeakGb": vram, "rmsDb": rms_db, "bytes": len(data)}


def check(call: Callable[..., tuple[int, Any]], runtime: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    apple = platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
    if apple:
        if runtime.get("unavailableReason") or runtime.get("unavailableCode"):
            return "fail", f"Apple Silicon reports unavailable: {runtime.get('unavailableReason')}", {}
        return "pass", "", {"lane": runtime.get("lane"), "path": "apple-silicon"}
    if runtime.get("lane") == "cuda":
        return _cuda(call, runtime)

    reason = runtime.get("unavailableReason") or ""
    if not reason.startswith(REASON_START) or runtime.get("unavailableCode") != CODE:
        return "fail", f"runtime reason/code wrong: {reason!r} / {runtime.get('unavailableCode')!r}", {}
    sound = runtime.get("sound") or {}
    if sound.get("unavailableReason") != reason:
        return "fail", f"the sounds lane says something else: {sound.get('unavailableReason')!r}", {}
    code, rows = _rows(call)
    if code != 200 or not rows:
        return "fail", f"catalog HTTP {code}", {}
    code, job = call("POST", "/install")
    if code != 200 or job.get("phase") != "error" or job.get("running") or job.get("error") != reason:
        return "fail", f"install was not refused: HTTP {code} {job}", {}
    return "pass", "", {"path": "unsupported", "rows": len(rows)}
