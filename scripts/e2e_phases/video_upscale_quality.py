"""Phase 27: the video upscaler's quality tier (FlashVSR+ Turbo, D0145, B1334).

Three checks:

* **catalogue and licence** (in process, always): the ``flashvsr-plus-turbo`` row carries the structured Apache-2.0
  licence (commercial use yes), its weights are pinned by Hugging Face revision and SHA-256, the code by commit and
  tree digest, and the fit gates are the spike's: not offered on a Mac, a CPU-only box or a card under 12 GB; 4K never;
  2560x1440 needs 20 GB.
* **the job around the model** (in process, always; needs only ffmpeg): the quality runner, the worker's own segment
  loop (``quality_worker.process_segment``) with a nearest-neighbour stand-in for the model, through
  ``pipeline.run_job`` on a 101-frame clip with a 12 s sound track, in segments of 30 frames with warm-up: the film
  has exactly the source's frames at the target size, and the stand-in saw the warm-up frames.
* **a real FlashVSR+ run on the running backend**: skips with its reason when no backend answers, the tier is not
  offered on that machine (no NVIDIA card, or one under 12 GB) or its add-on is not installed. Otherwise a 101-frame
  160x120 clip goes to 640x480 through ``/api/video/upscale`` (the add-on's own venv, worker and weights), and the
  phase asserts what phase 25 asserts (done, the source's frame count while running and done, a frames-a-second
  number, PNG first frames), the output size, and that a 4K request is refused with 409. About a minute on an RTX 4090
  (model load included). The film it makes in the default upscale folder is removed.

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)``. Run alone: ``--phases 27``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PHASE_KEY = 27
ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "flashvsr-plus-turbo"
SOURCE_FRAMES = 101
SOURCE_SIZE = (160, 120)
TARGET = (640, 480)


def _path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _make_clip(exe: str, path: Path) -> None:
    """101 frames of ``testsrc2`` at 10 fps with a 12 s tone (longer than the picture), joined with a copy."""
    head = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    picture, sound = path.with_name("q-picture.mp4"), path.with_name("q-sound.m4a")
    w, h = SOURCE_SIZE
    source = ["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=10", "-frames:v", str(SOURCE_FRAMES)]
    run = subprocess.run([*head, *source, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(picture)],
                         capture_output=True, text=True)
    if run.returncode != 0:
        subprocess.run([*head, *source, "-c:v", "mpeg4", "-q:v", "3", str(picture)], capture_output=True, text=True,
                       check=True)
    subprocess.run([*head, "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=12", "-c:a", "aac",
                    str(sound)], capture_output=True, text=True, check=True)
    subprocess.run([*head, "-i", str(picture), "-i", str(sound), "-c", "copy", str(path)], capture_output=True,
                   text=True, check=True)


def _size(exe: str, path: Path) -> tuple[int, int]:
    import re

    text = subprocess.run([exe, "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stderr
    found = re.search(r"Video: .*?(\d{2,5})x(\d{2,5})", text)
    return (int(found.group(1)), int(found.group(2))) if found else (0, 0)


def _catalogue() -> tuple[str, str, dict[str, Any]]:
    _path()
    from types import SimpleNamespace

    from backend_service.catalog import licences
    from backend_service.catalog import video_upscale_models as catalog
    from backend_service.video_upscale import quality_addon

    row = catalog.find_quality_model(MODEL_ID)
    if row is None:
        raise AssertionError(f"no {MODEL_ID} row in the upscale catalogue")
    lic = row.get("licence") or {}
    if lic.get("id") != licences.APACHE_2_0["id"] or lic.get("commercial") != "yes" or lic.get("gated"):
        raise AssertionError(f"the licence is {lic}, expected Apache-2.0, commercial use yes, not gated")
    if len(str(row.get("revision", ""))) != 40 or any(len(str(f.get("sha256", ""))) != 64 for f in row["files"]):
        raise AssertionError("the weights are not pinned by revision and SHA-256")
    if len(quality_addon.FLASHVSR_REF) != 40 or len(quality_addon.CODE_TREE_SHA256) != 64:
        raise AssertionError("the upstream code is not pinned by commit and tree digest")
    if catalog.find_model(MODEL_ID) is not None:
        raise AssertionError("the quality row is in the fast add-on's list (its installer would fetch it)")

    def gate(kind: str, gb: float, size: tuple[int, int] | None = None) -> dict[str, Any]:
        return catalog.quality_fit(SimpleNamespace(kind=kind, usable_gb=gb, total_gb=64.0, unified=False), size)

    checks = {
        "mac": gate("apple", 64.0)["code"] == "notCuda",
        "cpu": gate("cpu", 32.0)["code"] == "notCuda",
        "11 GB card": gate("cuda", 11.0)["code"] == "smallCard",
        "12 GB card to 1080p": gate("cuda", 12.0, (1920, 1080)).get("sizeOk") is True,
        "16 GB card to 1440p": gate("cuda", 16.0, (2560, 1440))["code"] == "sizeNeedsMore",
        "24 GB card to 1440p": gate("cuda", 24.0, (2560, 1440)).get("sizeOk") is True,
        "24 GB card to 4K": gate("cuda", 24.0, (3840, 2160))["code"] == "tooBig",
    }
    wrong = [name for name, ok in checks.items() if not ok]
    if wrong:
        raise AssertionError(f"fit gates wrong for: {', '.join(wrong)}")
    return "pass", "", {"licence": lic["name"], "revision": row["revision"][:7], "ref": quality_addon.FLASHVSR_REF[:7]}


def _in_process(exe: str) -> tuple[str, str, dict[str, Any]]:
    _path()
    import numpy as np

    from backend_service.video_upscale import pipeline, quality, quality_worker

    seen: list[int] = []

    def model(frames: Any, size: tuple[int, int], padded: tuple[int, int], progress: Any = None) -> Any:
        seen.append(int(frames.shape[0]))
        if progress is not None:
            progress(0.5)
        w, h = size
        ys, xs = np.arange(h) * frames.shape[1] // h, np.arange(w) * frames.shape[2] // w
        return frames[:, ys][:, :, xs].copy()

    def send(cmd: str, *, timeout: float, stall_s: float, on_event: Any, cancel_check: Any, job: str, **req: Any):
        return quality_worker.process_segment(req, model, lambda done, total: on_event({"event": "step", "step": done}),
                                              lambda: None)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        clip = root / "q-source.mp4"
        _make_clip(exe, clip)
        spec = pipeline.make_spec(exe, clip, root / "q-out.mp4", model_id=MODEL_ID, weights=root, size=TARGET,
                                  segment_seconds=3.0)
        result = pipeline.run_job(spec, quality.runner(send=send), exe=exe, work=root / "work")
        made = _decoded(exe, root / "q-out.mp4")
        size = _size(exe, root / "q-out.mp4")
    if (result["frames"], made) != (SOURCE_FRAMES, SOURCE_FRAMES):
        raise AssertionError(f"{result['frames']} frames reported, {made} decoded; the source has {SOURCE_FRAMES}")
    if size != TARGET:
        raise AssertionError(f"the film is {size[0]}x{size[1]}, expected {TARGET[0]}x{TARGET[1]}")
    if seen[0] != 30 or any(count != 30 + quality.WARMUP_FRAMES for count in seen[1:-1]):
        raise AssertionError(f"the model saw {seen} frames per segment, expected 30, then 30 + {quality.WARMUP_FRAMES}")
    return "pass", "", {"segments": result["segments"], "modelFrames": seen}


def _decoded(exe: str, path: Path) -> int:
    import re

    run = subprocess.run([exe, "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    found = re.findall(r"frame=\s*(\d+)", run.stderr)
    return int(found[-1]) if found else 0


def _live(suite: Any, exe: str) -> tuple[str, str, dict[str, Any]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from e2e_phases import video_upscale as fast_phase

    def http(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, bytes]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        return suite._voice_http(method, path, body=data, content_type="application/json" if data else None,
                                 timeout=60.0)

    status, raw = http("GET", "/api/setup/video-upscale-quality/status")
    if status == 404:
        return "fail", "the running backend has no quality upscaler routes (an older build)", {}
    addon = fast_phase._json(status, raw)
    if not addon.get("available"):
        return "skip", f"the quality tier does not run on this machine: {addon.get('unavailableReason')}", {}
    rows = fast_phase._json(*http("GET", "/api/video/upscale/models")).get("qualityModels") or []
    if not rows:
        return "fail", "the add-on says it can run here but the models route lists no quality row", {}
    gate = rows[0].get("qualityFit") or {}
    if not gate.get("offered"):
        return "skip", f"the quality tier is not offered on this card: {gate.get('reason')}", {}
    if not addon.get("installed"):
        return "skip", "the quality upscaler add-on is not installed (Upscale tab > Quality > Install)", {}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        clip = root / "q-live-source.mp4"
        _make_clip(exe, clip)
        status, raw = http("POST", "/api/video/upscale", {"inputPath": str(clip), "modelId": MODEL_ID,
                                                           "width": 3840, "height": 2880})
        if status != 409 or b"at most" not in raw:
            raise AssertionError(f"a 4K quality request answered HTTP {status}: {raw[:160]!r}")
        detail = fast_phase._upscale_and_verify(
            http, exe, root, {"modelId": MODEL_ID, "width": TARGET[0], "height": TARGET[1]}, timeout_s=900.0,
            make=_make_clip, name="q-live-source.mp4", expected=SOURCE_FRAMES)
        size = _size(exe, Path(detail["output"]))
        Path(detail["output"]).unlink(missing_ok=True)
    if size != TARGET:
        raise AssertionError(f"the film is {size[0]}x{size[1]}, expected {TARGET[0]}x{TARGET[1]}")
    return "pass", "", {**detail, "model": MODEL_ID, "size": list(size)}


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=PHASE_KEY, name="Video upscale, quality tier (FlashVSR+ Turbo)")
    phase.checks.append(suite._check("the quality row: Apache-2.0 licence, pinned weights and code, the spike's fit "
                                     "gates", _catalogue))
    try:
        _path()
        from backend_service.storyboard import ffmpeg_bin

        exe = ffmpeg_bin.ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 - FfmpegMissing and import errors alike
        phase.checks.append(suite.CheckResult("phase 27 jobs", "skip", reason=f"no ffmpeg to generate the test clip: {exc}"))
    else:
        def live() -> tuple[str, str, dict[str, Any]]:
            if not cap.backend_reachable:
                return "skip", "backend not reachable", {}
            return _live(suite, exe)

        phase.checks.append(suite._check("a quality job in segments with warm-up: every frame, the target size "
                                         "(in process, stand-in model)", lambda: _in_process(exe)))
        phase.checks.append(suite._check("a real FlashVSR+ run on the running backend: 101 frames to 640x480, 4K "
                                         "refused", live))
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
