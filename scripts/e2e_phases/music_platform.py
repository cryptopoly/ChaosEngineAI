"""Phase 11 check (FU-139): the music runtime's lane matches the host, and hosts without a lane refuse install.

Called from ``phase_11`` in ``scripts/e2e_test_suite.py`` as ``check(call, runtime)`` where ``call(method, path,
payload)`` returns ``(status, json)`` against ``/api/music``. Three paths:

* Apple Silicon: lane ``mlx``, no reason, no code.
* An NVIDIA host (lane ``cuda``, Windows or Linux x86_64): music has no reason or code, both ACE-Step rows list
  ``cuda``, the sound rows do not, the sounds lane reports its own sentence and ``music.soundUnsupportedPlatform``,
  and ``POST /download`` of a sound repo is refused (409) without starting anything.
* Anything else: the single music sentence and ``music.unsupportedPlatform``, the sounds lane says the same, and
  ``POST /install`` starts nothing (the job answers ``error`` with that reason).

It never starts an install or a download on a host where music runs.
"""

from __future__ import annotations

import platform
from typing import Any, Callable

REASON_START = "Music generation needs an Apple Silicon Mac or an NVIDIA GPU"
CODE = "music.unsupportedPlatform"
SOUND_CODE = "music.soundUnsupportedPlatform"
SOUND_REPO = "stabilityai/stable-audio-open-small"


def _rows(call: Callable[..., tuple[int, Any]]) -> tuple[int, list[dict[str, Any]]]:
    code, body = call("GET", "/catalog")
    return code, [v for f in (body or {}).get("families", []) for v in f.get("variants", [])]


def _cuda(call: Callable[..., tuple[int, Any]], runtime: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if runtime.get("unavailableReason") or runtime.get("unavailableCode"):
        return "fail", f"music is refused on an NVIDIA host: {runtime.get('unavailableReason')!r}", {}
    sound = runtime.get("sound") or {}
    if sound.get("unavailableCode") != SOUND_CODE or not sound.get("unavailableReason"):
        return "fail", f"the sounds lane does not say why: {sound.get('unavailableReason')!r} / {sound.get('unavailableCode')!r}", {}
    code, rows = _rows(call)
    music = [v for v in rows if v.get("kind") != "sound"]
    sounds = [v for v in rows if v.get("kind") == "sound"]
    if code != 200 or not music or any("cuda" not in (v.get("platforms") or []) for v in music):
        return "fail", f"catalog HTTP {code}; an ACE-Step row does not list cuda", {}
    if any("cuda" in (v.get("platforms") or []) for v in sounds):
        return "fail", "a sound row lists cuda, but the CUDA sound lane is not built", {}
    code, body = call("POST", "/download", {"repo": SOUND_REPO})
    if code != 409 or (body or {}).get("detail") != sound.get("unavailableReason"):
        return "fail", f"a sound download was not refused: HTTP {code} {str(body)[:200]}", {}
    return "pass", "", {"path": "cuda", "musicRows": len(music), "installed": runtime.get("installed")}


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
