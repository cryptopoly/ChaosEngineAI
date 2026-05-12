#!/usr/bin/env python3
"""Reference-gen wall-time capture for the v0.8.0 refactor.

Three reference gens — text (mlx_lm), image (FLUX.1-schnell), and video
(Wan2.1-T2V-1.3B) — run in subprocess isolation so model loads don't
fight for the same address space. Output is a single JSON blob suitable
for diffing against PERF_BASELINE.md.

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


def _run_isolated(label: str, code: str) -> dict:
    """Run ``code`` as a subprocess so each gen gets a fresh interpreter."""
    print(f"\n[{label}] starting subprocess...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
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

IMAGE_REPO = "black-forest-labs/FLUX.1-schnell"
IMAGE_CODE = f"""
import json, time, torch
from diffusers import FluxPipeline
t0 = time.time()
pipe = FluxPipeline.from_pretrained({IMAGE_REPO!r}, torch_dtype=torch.bfloat16)
pipe = pipe.to('mps' if torch.backends.mps.is_available() else 'cpu')
load_s = time.time() - t0
t0 = time.time()
img = pipe('a fluffy cat on a windowsill', num_inference_steps=4, guidance_scale=0.0, height=1024, width=1024).images[0]
gen_s = time.time() - t0
print(json.dumps({{
    'repo': {IMAGE_REPO!r},
    'load_seconds': round(load_s, 3),
    'gen_seconds': round(gen_s, 3),
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
t0 = time.time()
pipe = WanPipeline.from_pretrained({VIDEO_REPO!r}, torch_dtype=torch.bfloat16)
pipe = pipe.to('mps' if torch.backends.mps.is_available() else 'cpu')
load_s = time.time() - t0
t0 = time.time()
out = pipe('a fluffy cat walking on a windowsill', num_inference_steps=4, num_frames=5, height=272, width=480, guidance_scale=1.0)
gen_s = time.time() - t0
print(json.dumps({{
    'repo': {VIDEO_REPO!r},
    'load_seconds': round(load_s, 3),
    'gen_seconds': round(gen_s, 3),
    'frames': 5,
}}))
"""


GENS = [
    ("text", TEXT_REPO, TEXT_CODE),
    ("image", IMAGE_REPO, IMAGE_CODE),
    ("video", VIDEO_REPO, VIDEO_CODE),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["text", "image", "video"], help="Run a single gen")
    ap.add_argument("--output", default=None, help="Write JSON to this path (default stdout only)")
    ap.add_argument("--allow-missing", action="store_true", help="Run even when models aren't cached (will trigger download)")
    args = ap.parse_args()

    selected = [(label, repo, code) for label, repo, code in GENS if not args.only or label == args.only]
    results = []
    for label, repo, code in selected:
        if not args.allow_missing and not _model_cached(repo):
            results.append({"label": label, "ok": False, "skipped": True, "reason": f"{repo} not cached locally; pass --allow-missing to download"})
            print(f"[{label}] skipped — {repo} not cached")
            continue
        results.append(_run_isolated(label, code))

    blob = {"capturedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": results}
    payload = json.dumps(blob, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
