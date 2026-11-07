#!/usr/bin/env python3
"""Reference-gen wall-time capture for the v0.8.0 refactor.

Reference gens run in subprocess isolation so model loads don't fight for
the same address space. Output is a single JSON blob suitable for diffing
against PERF_BASELINE.md.

* ``text``: mlx_lm, Qwen2.5-0.5B.
* ``image`` / ``video``: diffusers FLUX.1-schnell / Wan2.1-T2V-1.3B on MPS.
* ``image-mflux``: Iris-3B Turbo through the app's mflux engine, one cold
  CLI run then three warm-worker runs (I1).
* ``video-ltx2mlx``: LTX-2.5 q4 distilled through the app's ltx-2-mlx
  engine, one cold CLI clip then three warm-worker clips (V1).
* ``music``: ACE-Step 1.5 turbo, a 60 s song with lyrics, cold then warm
  (nothing is saved to the music library).
* ``storyboard-render``: the storyboard assembler (ffmpeg only, no model)
  joins four synthetic 4 s clips with crossfades and a music bed.

Every image and video label reports the C1 split (``load_seconds``,
``encode_seconds``, ``denoise_seconds``, ``decode_seconds``) next to the
wall time. On the diffusers labels the split comes from step-callback
timestamps (encode and denoise are estimates); the app engines report it
themselves.

Usage:
    .venv/bin/python scripts/perf-baseline.py
    .venv/bin/python scripts/perf-baseline.py --only text
    .venv/bin/python scripts/perf-baseline.py --output /tmp/baseline.json

Each gen is gated on the model being already cached locally — we never
trigger an unwanted multi-GB download from this script. Missing models
are reported but don't fail the run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HF_HUB = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")


def _model_cached(repo: str) -> bool:
    folder = HF_HUB / f"models--{repo.replace('/', '--')}"
    return folder.is_dir()


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ready_always() -> tuple[bool, str]:
    return True, ""


def _run_isolated(label: str, code: str, timeout: float = 600) -> dict:
    """Run ``code`` as a subprocess so each gen gets a fresh interpreter."""
    print(f"\n[{label}] starting subprocess...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    wall = time.time() - t0
    if proc.returncode != 0:
        return {
            "label": label,
            "ok": False,
            "wall_seconds": round(wall, 2),
            "stderr": proc.stderr[-2000:],
        }
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {
            "label": label,
            "ok": False,
            "wall_seconds": round(wall, 2),
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
    result["label"] = label
    result["ok"] = True
    result["wall_seconds"] = round(wall, 2)
    return result


# ---------------------------------------------------------------------------
# Text gen — mlx_lm + Qwen2.5-0.5B-Instruct-4bit
# ---------------------------------------------------------------------------

TEXT_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
TEXT_CODE = f"""
import json, time
from mlx_lm import load, generate
t0 = time.time()
model, tok = load({TEXT_REPO!r})
load_s = time.time() - t0
prompt = 'Explain in two sentences why prompt caching speeds up LLM inference.'
formatted = tok.apply_chat_template([{{"role": "user", "content": prompt}}], add_generation_prompt=True)
t0 = time.time()
text = generate(model, tok, prompt=formatted, max_tokens=128, verbose=False)
gen_s = time.time() - t0
n_tok = len(tok.encode(text))
print(json.dumps({{
    'repo': {TEXT_REPO!r},
    'load_seconds': round(load_s, 3),
    'gen_seconds': round(gen_s, 3),
    'output_tokens': n_tok,
    'tokens_per_second': round(n_tok / gen_s, 1),
}}))
"""


# ---------------------------------------------------------------------------
# Image gen — FLUX.1-schnell, 4 steps, 1024×1024
# ---------------------------------------------------------------------------

# C1 split for a diffusers call from step-callback timestamps: the time before the first callback,
# less one step, is the prompt encode; the time after the last one is the VAE decode.
SPLIT_HELPER = """
def _split(t0, marks, end):
    if not marks:
        return {'encode_seconds': None, 'denoise_seconds': None, 'decode_seconds': None}
    step = (marks[-1] - marks[0]) / (len(marks) - 1) if len(marks) > 1 else marks[0] - t0
    return {
        'encode_seconds': round(max(0.0, marks[0] - t0 - step), 3),
        'denoise_seconds': round(marks[-1] - marks[0] + step, 3),
        'decode_seconds': round(end - marks[-1], 3),
    }

def _mark(marks):
    def callback(pipe, step, timestep, kwargs):
        marks.append(time.time())
        return kwargs
    return callback
"""

IMAGE_REPO = "black-forest-labs/FLUX.1-schnell"
IMAGE_CODE = f"""
import json, time, torch
from diffusers import FluxPipeline
{SPLIT_HELPER}
t0 = time.time()
pipe = FluxPipeline.from_pretrained({IMAGE_REPO!r}, torch_dtype=torch.bfloat16)
pipe = pipe.to('mps' if torch.backends.mps.is_available() else 'cpu')
load_s = time.time() - t0
marks = []
t0 = time.time()
img = pipe('a fluffy cat on a windowsill', num_inference_steps=4, guidance_scale=0.0, height=1024, width=1024,
           callback_on_step_end=_mark(marks)).images[0]
end = time.time()
gen_s = end - t0
print(json.dumps({{
    'repo': {IMAGE_REPO!r},
    'load_seconds': round(load_s, 3),
    'gen_seconds': round(gen_s, 3),
    **_split(t0, marks, end),
    'image_pixels': img.size[0] * img.size[1],
}}))
"""


# ---------------------------------------------------------------------------
# Video gen — Wan2.1-T2V-1.3B, 5 frames, 480x272, 4 steps
# ---------------------------------------------------------------------------

VIDEO_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
VIDEO_CODE = f"""
import json, time, torch
from diffusers import WanPipeline
{SPLIT_HELPER}
t0 = time.time()
pipe = WanPipeline.from_pretrained({VIDEO_REPO!r}, torch_dtype=torch.bfloat16)
pipe = pipe.to('mps' if torch.backends.mps.is_available() else 'cpu')
load_s = time.time() - t0
marks = []
t0 = time.time()
out = pipe('a fluffy cat walking on a windowsill', num_inference_steps=4, num_frames=5, height=272, width=480, guidance_scale=1.0,
           callback_on_step_end=_mark(marks))
end = time.time()
gen_s = end - t0
print(json.dumps({{
    'repo': {VIDEO_REPO!r},
    'load_seconds': round(load_s, 3),
    'gen_seconds': round(gen_s, 3),
    **_split(t0, marks, end),
    'frames': 5,
}}))
"""


# ---------------------------------------------------------------------------
# App engines: one cold CLI run, then three warm-worker runs (I1, V1)
# ---------------------------------------------------------------------------

# Summarise one cold run and the warm runs: the warm median is the gated number (protocol section 9).
ENGINE_SUMMARY = """
import statistics
def _summary(cold, first, warm):
    walls = [w['wall'] for w in warm]
    mid = sorted(warm, key=lambda w: w['wall'])[len(warm) // 2]
    t = mid['timings'] or {}
    return {
        'cold_seconds': round(cold['wall'], 3),
        'cold_timings': cold['timings'],
        'worker_first_seconds': round(first['wall'], 3),
        'gen_seconds': round(statistics.median(walls), 3),
        'warm_runs': [round(w, 3) for w in walls],
        'load_seconds': t.get('loadSeconds'),
        'encode_seconds': t.get('encodeSeconds'),
        'denoise_seconds': t.get('denoiseSeconds'),
        'decode_seconds': t.get('decodeSeconds'),
        'warm': bool(t.get('warm')),
        'same_output_as_cold': first['sha'] == cold['sha'],
    }
"""

MFLUX_REPO = "speridlabs/iris-3b"
MFLUX_CODE = f"""
import hashlib, json, os, time
os.environ['CHAOSENGINE_WARM_WORKERS'] = '0'
from backend_service.image_runtime import mflux_engine
from backend_service.image_runtime.mflux_engine import MfluxImageEngine
from backend_service.image_runtime.types import ImageGenerationConfig
{ENGINE_SUMMARY}
def run(seed):
    cfg = ImageGenerationConfig(modelId='iris', modelName='Iris-3B', repo={MFLUX_REPO!r}, prompt='a lighthouse on a cliff at dusk, oil painting',
                                negativePrompt='', width=768, height=512, steps=4, guidance=1.0, batchSize=1, seed=seed, runtime='mflux')
    t = time.time()
    [img] = MfluxImageEngine().generate(cfg)
    return {{'wall': time.time() - t, 'timings': img.timings, 'sha': hashlib.sha256(img.bytes).hexdigest(), 'seed': seed}}
cold = run(1234)
os.environ['CHAOSENGINE_WARM_WORKERS'] = '1'
first = run(1234)  # worker start and load; same seed as the cold run, so the bytes must match
warm = [run(1235 + i) for i in range(3)]
if mflux_engine._WORKER is not None:
    mflux_engine._WORKER.stop('perf-baseline done')
print(json.dumps({{'repo': {MFLUX_REPO!r}, 'size': '768x512', 'steps': 4, **_summary(cold, first, warm)}}))
"""

LTX2_REPO = "dgrauet/ltx-2.5-mlx-q4"
LTX2_CODE = f"""
import hashlib, json, os, time
os.environ['CHAOSENGINE_WARM_WORKERS'] = '0'
from backend_service import ltx2_mlx_runtime
from backend_service.ltx2_mlx_runtime import Ltx2MlxVideoEngine
from backend_service.video_runtime.types import VideoGenerationConfig
{ENGINE_SUMMARY}
def run(seed):
    cfg = VideoGenerationConfig(modelId='ltx', modelName='LTX-2.5 q4', repo={LTX2_REPO!r},
                                prompt='a lighthouse on a cliff at dusk, waves crashing', negativePrompt='',
                                width=768, height=448, numFrames=49, fps=24, guidance=1.0, seed=seed)
    t = time.time()
    video = Ltx2MlxVideoEngine().generate(cfg)
    return {{'wall': time.time() - t, 'timings': video.timings, 'sha': hashlib.sha256(video.bytes).hexdigest(), 'seed': seed}}
cold = run(4242)
os.environ['CHAOSENGINE_WARM_WORKERS'] = '1'
first = run(4242)  # worker start and load; same seed as the cold run, so the bytes must match
warm = [run(4243 + i) for i in range(3)]
if ltx2_mlx_runtime._WORKER is not None:
    ltx2_mlx_runtime._WORKER.stop('perf-baseline done')
print(json.dumps({{'repo': {LTX2_REPO!r}, 'size': '768x448x49', **_summary(cold, first, warm)}}))
"""


# ---------------------------------------------------------------------------
# Music: ACE-Step 1.5 turbo, 60 s with lyrics (cold worker, then warm)
# ---------------------------------------------------------------------------

MUSIC_CODE = """
import json, time
from backend_service import music_runtime
from backend_service.music_client import get_worker
music_runtime._save_track = lambda *args, **kwargs: {}  # time the model, keep the library clean
request = {'caption': 'warm folk song, acoustic guitar, soft female vocal', 'seconds': 60, 'variations': 1, 'seed': 42,
           'lyrics': '[verse]\\nThe lamp is lit above the sea\\nThe keeper keeps the light for me\\n[chorus]\\nShine on, shine on'}
runs = []
for _ in range(2):
    t = time.time()
    result = music_runtime.generate(dict(request))
    runs.append({'wall': time.time() - t, 'timings': result.get('timings') or {}, 'peak': result.get('peakFootprintGb')})
get_worker().stop()
print(json.dumps({
    'model': music_runtime.resolve_variant(None)['id'],
    'cold_seconds': round(runs[0]['wall'], 3),
    'gen_seconds': round(runs[1]['wall'], 3),
    'timings': runs[1]['timings'],
    'peak_footprint_gb': runs[1]['peak'],
}))
"""


def _music_ready() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from backend_service import music_env
    except Exception as exc:  # noqa: BLE001
        return False, f"music env unreadable: {exc}"
    return (True, "") if music_env.is_installed() else (False, "the music pack is not installed")


# ---------------------------------------------------------------------------
# Storyboard render: ffmpeg assembly of four synthetic clips (no model)
# ---------------------------------------------------------------------------

STORYBOARD_CODE = """
import json, subprocess, tempfile, time
from pathlib import Path
from backend_service.storyboard import ffmpeg_bin
from backend_service.storyboard.assemble import render
from backend_service.storyboard.render_plan import RenderPlan
work = Path(tempfile.mkdtemp(prefix='perf-storyboard-'))
ffmpeg = ffmpeg_bin.ffmpeg_exe()
clips = []
for i in range(4):
    path = work / f'clip{i}.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', f'testsrc2=size=768x448:rate=24:duration=4',
                    '-f', 'lavfi', '-i', f'sine=frequency={220 + 110 * i}:duration=4', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-shortest', str(path)], check=True)
    clips.append(path)
bed = work / 'bed.wav'
subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=110:duration=16', str(bed)], check=True)
plan = RenderPlan.from_dict({
    'width': 768, 'height': 448, 'fps': 24, 'out_path': str(work / 'film.mp4'), 'work_dir': str(work / 'work'),
    'shots': [{'id': f's{i}', 'video': str(p), 'hash': f'h{i}', 'transition_in': {'type': 'crossfade', 'seconds': 0.5} if i else None}
              for i, p in enumerate(clips)],
    'audio': [{'id': 'bed', 'kind': 'music', 'path': str(bed), 'gain_db': -12.0}],
})
t = time.time()
result = render(plan)
gen_s = time.time() - t
print(json.dumps({'gen_seconds': round(gen_s, 3), 'film_seconds': round(result.duration_s, 2), 'shots': len(clips),
                  'ffmpeg': result.ffmpeg_version}))
"""


GENS = [
    ("text", TEXT_REPO, TEXT_CODE),
    ("image", IMAGE_REPO, IMAGE_CODE),
    ("video", VIDEO_REPO, VIDEO_CODE),
    ("image-mflux", MFLUX_REPO, MFLUX_CODE),
    ("video-ltx2mlx", LTX2_REPO, LTX2_CODE),
    ("music", None, MUSIC_CODE),
    ("storyboard-render", None, STORYBOARD_CODE),
]
# Labels whose readiness is not "a Hugging Face repo is cached".
READY_CHECKS = {"music": _music_ready, "storyboard-render": _ready_always}
TIMEOUTS = {"video-ltx2mlx": 1800, "image-mflux": 900}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=[label for label, _, _ in GENS], help="Run a single gen")
    ap.add_argument("--output", default=None, help="Write JSON to this path (default stdout only)")
    ap.add_argument("--allow-missing", action="store_true", help="Run even when models aren't cached (will trigger download)")
    args = ap.parse_args()

    selected = [(label, repo, code) for label, repo, code in GENS if not args.only or label == args.only]
    results = []
    for label, repo, code in selected:
        if label in READY_CHECKS:
            ready, reason = READY_CHECKS[label]()
            if not ready:
                results.append({"label": label, "ok": False, "skipped": True, "reason": reason})
                print(f"[{label}] skipped — {reason}")
                continue
        elif not args.allow_missing and not _model_cached(repo):
            results.append({"label": label, "ok": False, "skipped": True, "reason": f"{repo} not cached locally; pass --allow-missing to download"})
            print(f"[{label}] skipped — {repo} not cached")
            continue
        results.append(_run_isolated(label, code, timeout=TIMEOUTS.get(label, 600)))

    blob = {"capturedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": results}
    payload = json.dumps(blob, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
