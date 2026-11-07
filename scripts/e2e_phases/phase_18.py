"""Phase 18: video tools (ffmpeg) status, the pinned download table, the encoder chooser, a real encode.

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)``. Needs no model and downloads nothing: the
pinned Windows/Linux download itself (real network, 100+ MB) is not exercised here; unit tests cover its
verify, extract and atomic-install steps with fake archives (``tests/test_ffmpeg_fetch.py``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_ENCODERS_LGPL_WIN = " V....D libopenh264 OpenH264\n V....D h264_nvenc NVENC\n V....D h264_amf AMF\n"


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=18, name="Video tools (ffmpeg)")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 18", "skip", reason="backend not reachable"))
        return phase
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

    def api(path: str) -> tuple[int, Any]:
        code, raw = suite._voice_http("GET", path, timeout=60.0)
        try:
            return code, json.loads(raw.decode("utf-8")) if raw.strip() else None
        except ValueError:
            return code, raw

    state: dict[str, Any] = {}

    def status_shape():
        code, data = api("/api/setup/ffmpeg")
        if code == 404:
            return "skip", "video tools route not present in this build", {}
        keys = ("installed", "path", "source", "version", "licence", "encoder", "pin", "canDownload", "job")
        missing = [k for k in keys if not isinstance(data, dict) or k not in data]
        if code != 200 or missing:
            return "fail", f"HTTP {code}, missing {missing}", {}
        state["status"] = data
        return "pass", "", {"installed": data["installed"], "source": data["source"], "encoder": data["encoder"]}

    def pin_table():
        from backend_service.ffmpeg_fetch import pins

        bad = [k for k, p in pins.PINS.items()
               if not (p.url.startswith("https://") and len(p.sha256) == 64 and p.size > 0 and "lgpl" in p.asset)]
        if bad or not pins.PINS_VERIFIED:
            return "fail", f"bad or unverified pins: {bad}", {}
        return "pass", "", {"rows": sorted(pins.PINS)}

    def chooser():
        from backend_service.storyboard import encoders

        found = encoders.parse_encoders(_ENCODERS_LGPL_WIN)
        picks = (encoders.choose_h264(found, "win32", lambda n: n != "h264_nvenc"),
                 encoders.choose_h264(found | {"libx264"}, "linux"),
                 encoders.choose_h264(frozenset(), "linux"))
        if picks != ("h264_amf", "libx264", None):
            return "fail", f"unexpected choices {picks}", {}
        return "pass", "", {}

    def real_encode():
        status = state.get("status")
        if not status or not status.get("installed"):
            return "skip", "no ffmpeg installed on this machine (Setup > Install video tools)", {}
        from backend_service.storyboard import encoders

        encoder = status.get("encoder")
        if not encoder:
            return "fail", "an ffmpeg is installed but no H.264 encoder works", {}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "t.mp4"
            run_args = [status["path"], "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "testsrc=s=160x96:r=24:d=1", "-f", "lavfi", "-i", "sine=d=1",
                        *encoders.codec_args(encoder, True, width=160, height=96, fps=24),
                        "-c:a", "aac", "-shortest", str(out)]
            proc = subprocess.run(run_args, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0 or not out.is_file():
                return "fail", f"{encoder} encode failed: {proc.stderr.strip()[-200:]}", {}
            info = subprocess.run([status["path"], "-hide_banner", "-nostdin", "-i", str(out)],
                                  capture_output=True, text=True, timeout=60).stderr
            if "Video: h264" not in info or "Audio: aac" not in info:
                return "fail", "the MP4 is not H.264 + AAC", {}
        return "pass", "", {"encoder": encoder}

    for name, fn in [
        ("GET /api/setup/ffmpeg has the status shape", status_shape),
        ("pin table rows are complete and LGPL", pin_table),
        ("encoder chooser picks by listing and trial", chooser),
        ("a 1 s clip encodes to H.264 + AAC with the chosen encoder", real_encode),
    ]:
        phase.checks.append(suite._check(name, fn))
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
