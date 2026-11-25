"""Phase 25: video upscale jobs (B1316: the frame count came from duration x rate, and ``framesPerSecond`` was null).

A tiny generated clip (101 frames at 10 fps, with a sound track 12 s long, so duration x rate says 120) goes through
the real ``/api/video/upscale`` routes and must come out with exactly the source's frames. The same checks run twice:

* **in process**: the real routes and job manager, a pixel-doubling stand-in for the model, no backend, no GPU, no
  add-on. This is what runs on every machine, so the phase never passes by skipping everything;
* **against the running backend**, with the real model: skipped with its reason when no backend answers, the add-on
  is not installed or no upscale model is downloaded.

A second in-process case (B1328) is an MPEG-TS clip whose timestamps start 1.4 s in (a broadcast capture), 520 frames at
25 fps so the job has three segments: the segments after the first used to seek past frames and the film came back
short with the job "done". It must come out with all 520 frames.

Each run asserts: the job reaches ``done``; ``framesTotal`` and ``framesDone`` are the source's real frame count;
the output decodes to the same number of frames as the source (the tail of the film is neither padded nor dropped);
``framesPerSecond`` is a number once done (and in every status seen while running); and both first-frame routes
(``?which=input`` and ``?which=output``) answer PNGs.

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)``.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from unittest import mock

PHASE_KEY = 25
ROOT = Path(__file__).resolve().parents[2]
SOURCE_FRAMES = 101  # 10 fps; a tail of one frame after a 100-frame segment is what once came back padded
TS_FRAMES = 520  # 25 fps, 250-frame segments: three segments, the last of 20 frames
SOUND_SECONDS = 12  # longer than the picture: duration x rate = 120, the real count is 101
PNG = b"\x89PNG\r\n\x1a\n"
Http = Callable[..., tuple[int, bytes]]  # (method, path, json_body=None) -> (status, body)


def _ffmpeg() -> str:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend_service.storyboard import ffmpeg_bin

    return ffmpeg_bin.ffmpeg_exe()


def _make_clip(exe: str, path: Path) -> None:
    """The picture and the sound are made apart and joined with a copy: ``-frames:v`` would end the sound with it."""
    head = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    picture, sound = path.with_name("picture.mp4"), path.with_name("sound.m4a")
    source = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10", "-frames:v", str(SOURCE_FRAMES)]
    run = subprocess.run([*head, *source, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(picture)],
                         capture_output=True, text=True)
    if run.returncode != 0:  # a build without libx264
        subprocess.run([*head, *source, "-c:v", "mpeg4", "-q:v", "3", str(picture)], capture_output=True, text=True,
                       check=True)
    subprocess.run([*head, "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={SOUND_SECONDS}",
                    "-c:a", "aac", str(sound)], capture_output=True, text=True, check=True)
    subprocess.run([*head, "-i", str(picture), "-i", str(sound), "-c", "copy", str(path)], capture_output=True,
                   text=True, check=True)


def _make_ts_clip(exe: str, path: Path) -> None:
    """An MPEG-TS capture that starts 1.4 s in (the muxer's own 1.4 s start delay comes on top of that)."""
    subprocess.run([exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=64x48:rate=25", "-frames:v", str(TS_FRAMES), "-c:v", "mpeg2video", "-q:v", "3",
                    "-output_ts_offset", "1.4", "-f", "mpegts", str(path)], capture_output=True, text=True, check=True)


def _decoded_frames(exe: str, path: Path) -> int:
    """Frames a player would show: the video stream decoded at its constant rate, repeats included."""
    run = subprocess.run([exe, "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    found = re.findall(r"frame=\s*(\d+)", run.stderr)
    return int(found[-1]) if found else 0


def _json(status: int, body: bytes) -> dict[str, Any]:
    try:
        data = json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise AssertionError(f"HTTP {status}, not JSON: {body[:120]!r}") from exc
    if status >= 400:
        raise AssertionError(f"HTTP {status}: {str(data)[:200]}")
    return data


def _upscale_and_verify(http: Http, exe: str, work: Path, start_body: dict[str, Any], timeout_s: float,
                        make: Callable[[str, Path], None] = _make_clip, name: str = "e2e-upscale-source.mp4",
                        expected: int = SOURCE_FRAMES) -> dict[str, Any]:
    """Start a job on a fresh clip, wait for it, and assert everything in the module docstring. Returns the output."""
    clip = work / name
    make(exe, clip)
    source_frames = _decoded_frames(exe, clip)
    if source_frames != expected:
        raise AssertionError(f"the test clip has {source_frames} frames, expected {expected}")

    probed = _json(*http("POST", "/api/video/upscale/probe", {"inputPath": str(clip)}))
    if probed["frames"] != source_frames:  # the finished job's total is corrected at the end; the probe is not
        raise AssertionError(f"the probe counts {probed['frames']} frames, the source has {source_frames}")

    job = _json(*http("POST", "/api/video/upscale", {"inputPath": str(clip), **start_body}))
    job_id, running_rates, running_totals, deadline = job["id"], [], set(), time.time() + timeout_s
    while job["state"] in ("queued", "running"):
        if time.time() > deadline:
            raise AssertionError(f"the job was still {job['state']} after {timeout_s:.0f} s")
        time.sleep(0.05)
        job = _json(*http("GET", f"/api/video/upscale/jobs/{job_id}"))
        if job["state"] == "running":
            running_rates.append(job["framesPerSecond"])
            running_totals.add(job["framesTotal"])
    if running_totals - {source_frames}:
        raise AssertionError(f"framesTotal read {sorted(running_totals)} while running, the source has {source_frames}")
    if job["state"] != "done":
        raise AssertionError(f"the job ended {job['state']}: {job.get('error')}")
    if (job["framesTotal"], job["framesDone"]) != (expected, expected):
        raise AssertionError(f"frames {job['framesDone']} of {job['framesTotal']}, the source has {expected}")
    if not isinstance(job["framesPerSecond"], (int, float)) or isinstance(job["framesPerSecond"], bool):
        raise AssertionError(f"framesPerSecond is {job['framesPerSecond']!r} on a finished job")
    if any(rate is None for rate in running_rates):
        raise AssertionError(f"framesPerSecond was null in {running_rates.count(None)} status reads while running")

    output = Path(job["output"])
    out_frames = _decoded_frames(exe, output)
    if out_frames != source_frames:
        raise AssertionError(f"the output has {out_frames} frames, the source {source_frames}")

    for which, source in (("input", clip), ("output", output)):
        status, body = http("GET", f"/api/video/upscale/jobs/{job_id}/frame?which={which}")
        if status != 200 or not body.startswith(PNG):
            raise AssertionError(f"the {which} frame route answered HTTP {status}, {body[:40]!r}, not a PNG")
    return {"job": job_id, "frames": out_frames, "framesPerSecond": job["framesPerSecond"], "output": str(output)}


def _in_process(exe: str, *, transport_stream: bool = False) -> tuple[str, str, dict[str, Any]]:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import backend_service.app  # noqa: F401 - the route modules import each other; the app module loads them in order
    from backend_service.helpers.job_queue import HeavyJobQueue
    from backend_service.routes import video_upscale as route
    from backend_service.video_upscale import addon, jobs, pipeline

    def double(frames: Any) -> Any:  # a 2x "model" with a known answer
        return frames.repeat(2, axis=1).repeat(2, axis=2)

    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)
        outputs = root / "outputs"
        outputs.mkdir()
        manager = jobs.UpscaleJobs(runner_factory=lambda: pipeline.local_runner(double), root=lambda: root / "cache",
                                   queue=HeavyJobQueue())
        for patcher in (mock.patch.object(jobs, "JOBS", manager),
                        mock.patch.object(addon, "site_installed", return_value=True),
                        mock.patch.object(addon, "model_installed", return_value=True),
                        mock.patch.object(route, "_fit_refusal", return_value=None),
                        mock.patch.object(route, "output_roots", return_value=[outputs.resolve()]),
                        mock.patch.object(route, "default_output_dir", return_value=outputs / "upscaled")):
            stack.enter_context(patcher)
        app = FastAPI()
        app.include_router(route.router)
        client = TestClient(app)

        def http(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, bytes]:
            resp = client.request(method, path, json=body)
            return resp.status_code, resp.content

        if transport_stream:
            detail = _upscale_and_verify(http, exe, root, {"scale": 2.0}, timeout_s=180.0, make=_make_ts_clip,
                                         name="e2e-upscale-capture.ts", expected=TS_FRAMES)
        else:
            detail = _upscale_and_verify(http, exe, root, {"scale": 2.0}, timeout_s=120.0)
        manager._jobs.clear()  # the threads are done; nothing outlives the temp folder
    return "pass", "", detail


def _live(suite: Any, exe: str) -> tuple[str, str, dict[str, Any]]:
    def http(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, bytes]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        return suite._voice_http(method, path, body=data, content_type="application/json" if data else None, timeout=60.0)

    status, raw = http("GET", "/api/video/upscale/models")
    if status == 404:
        return "fail", "the running backend has no /api/video/upscale routes (an older build)", {}
    catalog = _json(status, raw)
    if not catalog.get("addonInstalled"):
        return "skip", "the video upscale add-on is not installed (Setup > Install video upscale)", {}
    ready = [row for row in catalog.get("models", []) if row.get("installed") and (row.get("fit") or {}).get("status") != "wont-fit"]
    if not ready:
        return "skip", "no video upscale model is downloaded (or none fits this machine)", {}
    model = ready[0]["id"]
    with tempfile.TemporaryDirectory() as tmp:
        detail = _upscale_and_verify(http, exe, Path(tmp), {"modelId": model, "scale": 2.0}, timeout_s=900.0)
        # The film went to the app's default upscale folder; it was this phase's own, so it is removed again.
        Path(detail["output"]).unlink(missing_ok=True)
    return "pass", "", {**detail, "model": model}


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=PHASE_KEY, name="Video upscale")
    try:
        exe = _ffmpeg()
    except Exception as exc:  # noqa: BLE001 - FfmpegMissing and import errors alike
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 25", "skip", reason=f"no ffmpeg to generate the test clip: {exc}"))
        return phase

    def _live_check():
        if not cap.backend_reachable:
            return "skip", "backend not reachable", {}
        return _live(suite, exe)

    for name, fn in [
        ("an upscale job on a clip with a long sound track: done, the source's frame count, a frames-a-second number, "
         "PNG first frames (in process, pixel-doubling stand-in)", lambda: _in_process(exe)),
        ("an MPEG-TS capture that starts 1.4 s in, three segments: every frame comes out and the job is done "
         "(in process, pixel-doubling stand-in)", lambda: _in_process(exe, transport_stream=True)),
        ("the same job on the running backend with the real model", _live_check),
    ]:
        phase.checks.append(suite._check(name, fn))
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
