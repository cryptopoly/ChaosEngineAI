#!/usr/bin/env python3
"""Measure the largest image size one model runs at on this machine.

The per-model size limits in the catalog (``maxSide`` / ``maxPixels``) and the
memory gate's pixel term (``backend_service/helpers/memory_gate.py``,
``EXTRA_SHARE_PER_MEGAPIXEL``) are estimates. This script produces measured
numbers to check them against: for one image model it generates one image at
each step of 1x, 1.5x, 2x, 3x and 4x the model's native side with a single
denoising step, and records wall time and peak memory of the backend process
tree (the diffusers pipeline, or the mflux / stable-diffusion.cpp child). It
stops at the first size the backend refuses (the memory gate answers 503), fails
on, or clamps to the model's own limit.

It generates real images, so run it when the GPU is free (nothing else loading a
model), on the hardware the numbers are meant to describe.

Usage (the backend must already be running, e.g. ``chaosengine-cli serve``):

    .venv/bin/python scripts/measure-image-ceiling.py \\
        --model black-forest-labs/FLUX.1-schnell

    # another aspect ratio, other factors, other port
    .venv/bin/python scripts/measure-image-ceiling.py \\
        --model Qwen/Qwen-Image-2.1-Turbo-mflux --aspect 16:9 \\
        --factors 1,1.5,2 --port 8876

``--model`` is a variant id from ``GET /api/images/catalog`` and the model must
be installed. Memory is the macOS physical footprint (``footprint -p``) summed
over the backend process and its children when that command exists, else
resident memory from psutil; on NVIDIA hosts the GPU memory in use
(``nvidia-smi``) is recorded too. Peak is the highest sample, taken every
``--interval`` seconds, so a short spike between samples can be missed: use a
shorter interval for a short run. The generated images are deleted afterwards
unless ``--keep-outputs`` is given.

It writes ``image-ceiling-<model>-<timestamp>.json`` and ``.md`` to
``~/.chaosengine/test-results/`` and prints the table.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FACTORS = (1.0, 1.5, 2.0, 3.0, 4.0)
SIZE_MULTIPLE = 64  # coarser than any engine's own multiple (16), so every step is valid
_FOOTPRINT_RE = re.compile(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB|TB)\b", re.IGNORECASE)
_UNIT_GB = {"B": 1 / 1024 ** 3, "KB": 1 / 1024 ** 2, "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}


# --------------------------------------------------------------------------- pure helpers

def parse_aspect(text: str) -> tuple[int, int]:
    """``"16:9"`` -> ``(16, 9)``."""
    match = re.fullmatch(r"\s*(\d+)\s*[:x/]\s*(\d+)\s*", text)
    if not match or int(match.group(1)) <= 0 or int(match.group(2)) <= 0:
        raise ValueError(f"Bad aspect ratio {text!r}; use e.g. 1:1 or 16:9.")
    return int(match.group(1)), int(match.group(2))


def size_steps(
    native_side: int,
    factors: tuple[float, ...],
    aspect: tuple[int, int] = (1, 1),
) -> list[tuple[float, int, int]]:
    """``[(factor, width, height)]`` with the long edge at ``native_side * factor``.

    Sizes are rounded to multiples of ``SIZE_MULTIPLE``; a factor that rounds to
    the same size as the one before it is skipped.
    """
    ratio_w, ratio_h = aspect
    long_ratio = max(ratio_w, ratio_h)
    steps: list[tuple[float, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for factor in factors:
        long_edge = max(SIZE_MULTIPLE, round(native_side * factor / SIZE_MULTIPLE) * SIZE_MULTIPLE)
        width = max(SIZE_MULTIPLE, round(long_edge * ratio_w / long_ratio / SIZE_MULTIPLE) * SIZE_MULTIPLE)
        height = max(SIZE_MULTIPLE, round(long_edge * ratio_h / long_ratio / SIZE_MULTIPLE) * SIZE_MULTIPLE)
        if (width, height) not in seen:
            seen.add((width, height))
            steps.append((factor, width, height))
    return steps


def exceeds_model_limit(width: int, height: int, max_side: int | None, max_pixels: int | None) -> bool:
    """True when the backend would clamp this size to the model's declared limit."""
    if max_side and max(width, height) > max_side:
        return True
    return bool(max_pixels and width * height > max_pixels)


def parse_footprint(text: str) -> float | None:
    """GB from ``footprint -p <pid>`` output (``Footprint: 18 GB``); None when absent."""
    match = _FOOTPRINT_RE.search(text)
    if not match:
        return None
    return float(match.group(1)) * _UNIT_GB[match.group(2).upper()]


# --------------------------------------------------------------------------- HTTP

class Backend:
    def __init__(self, port: int, token: str | None) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self.token = token

    def request(self, method: str, path: str, body: dict | None = None, timeout: float = 60) -> tuple[int, object]:
        """``(status, parsed JSON or raw text)``; never raises on an HTTP error status."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw, status = response.read().decode("utf-8", "replace"), response.status
        except urllib.error.HTTPError as exc:
            raw, status = exc.read().decode("utf-8", "replace"), exc.code
        try:
            return status, json.loads(raw)
        except ValueError:
            return status, raw


def error_text(payload: object) -> str:
    """The readable message of an error response."""
    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        if isinstance(detail, dict):
            return str(detail.get("message") or detail)
        return str(detail)
    return str(payload)[:500]


# --------------------------------------------------------------------------- memory sampling

def listening_pid(port: int) -> int | None:
    """The process listening on ``port`` (lsof, else psutil)."""
    if shutil.which("lsof"):
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split():
            if line.isdigit():
                return int(line)
    try:
        import psutil

        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port and conn.pid:
                return conn.pid
    except Exception:  # noqa: BLE001 - permissions differ per platform
        pass
    return None


def _process_gb(pid: int, use_footprint: bool) -> float:
    if use_footprint:
        result = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=20)
        value = parse_footprint(result.stdout)
        if value is not None:
            return value
    import psutil

    return psutil.Process(pid).memory_info().rss / 1024 ** 3


def tree_memory_gb(root_pid: int, use_footprint: bool) -> float:
    """Memory of ``root_pid`` and all its children, in GB (processes that vanish mid-sample count as 0)."""
    import psutil

    total = 0.0
    try:
        pids = [root_pid] + [child.pid for child in psutil.Process(root_pid).children(recursive=True)]
    except psutil.Error:
        return 0.0
    for pid in pids:
        try:
            total += _process_gb(pid, use_footprint)
        except (psutil.Error, subprocess.SubprocessError, OSError):
            continue
    return total


def gpu_used_gb() -> float | None:
    """GPU memory in use across NVIDIA cards, in GB; None without nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return sum(float(line) for line in result.stdout.split() if line.strip()) / 1024
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


class PeakSampler(threading.Thread):
    """Samples process-tree memory (and NVIDIA GPU memory) until stopped; keeps the maximum."""

    def __init__(self, pid: int, interval: float, use_footprint: bool) -> None:
        super().__init__(daemon=True)
        self.pid, self.interval, self.use_footprint = pid, interval, use_footprint
        self.peak_gb = 0.0
        self.peak_gpu_gb: float | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak_gb = max(self.peak_gb, tree_memory_gb(self.pid, self.use_footprint))
            gpu = gpu_used_gb()
            if gpu is not None:
                self.peak_gpu_gb = max(self.peak_gpu_gb or 0.0, gpu)
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=self.interval + 30)


# --------------------------------------------------------------------------- the run

def find_variant(backend: Backend, model_id: str) -> dict:
    status, payload = backend.request("GET", "/api/images/catalog", timeout=120)
    if status != 200 or not isinstance(payload, dict):
        raise SystemExit(f"Cannot read the image catalog (HTTP {status}): {error_text(payload)}")
    for family in payload.get("families", []):
        for variant in family.get("variants", []):
            if variant.get("id") == model_id:
                return variant
    raise SystemExit(f"No image model {model_id!r} in the catalog.")


def measure_step(backend: Backend, args, pid: int, use_footprint: bool, width: int, height: int) -> dict:
    row: dict = {"requested": f"{width}x{height}", "status": "ok"}
    baseline = tree_memory_gb(pid, use_footprint)
    sampler = PeakSampler(pid, args.interval, use_footprint)
    sampler.start()
    started = time.monotonic()
    try:
        status, payload = backend.request(
            "POST",
            "/api/images/generate",
            {
                "modelId": args.model,
                "prompt": args.prompt,
                "width": width,
                "height": height,
                "steps": args.steps,
                "seed": 1,
                "batchSize": 1,
            },
            timeout=args.timeout,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        status, payload = 0, f"{type(exc).__name__}: {exc}"
    finally:
        row["seconds"] = round(time.monotonic() - started, 1)
        sampler.stop()
    row["baselineGb"] = round(baseline, 1)
    row["peakGb"] = round(sampler.peak_gb, 1)
    row["peakOverBaselineGb"] = round(max(0.0, sampler.peak_gb - baseline), 1)
    if sampler.peak_gpu_gb is not None:
        row["peakGpuGb"] = round(sampler.peak_gpu_gb, 1)
    if status == 503:
        row.update(status="refused", message=error_text(payload))
    elif status != 200 or not isinstance(payload, dict):
        row.update(status="failed", message=f"HTTP {status}: {error_text(payload)}")
    else:
        artifacts = payload.get("artifacts") or []
        artifact = artifacts[0] if artifacts else {}
        got = (artifact.get("width"), artifact.get("height"))
        row["rendered"] = f"{got[0]}x{got[1]}"
        row["artifactIds"] = [a.get("artifactId") for a in artifacts if a.get("artifactId")]
        if got != (width, height):
            row.update(status="clamped", message=str(artifact.get("runtimeNote") or "the backend changed the size"))
    return row


def render_markdown(model_id: str, variant: dict, rows: list[dict], notes: list[str]) -> str:
    lines = [
        f"# Image size ceiling: {variant.get('name', model_id)}",
        "",
        f"Model `{model_id}`; catalog limits maxSide {variant.get('maxSide')}, maxPixels {variant.get('maxPixels')}, "
        f"nativeSide {variant.get('nativeSide')}. Steps per image: see the JSON.",
        "",
        "| Factor | Size | Result | Seconds | Peak GB | Over baseline GB | GPU GB | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('factor')}x | {row['requested']} | {row['status']} | {row.get('seconds', '')} | "
            f"{row.get('peakGb', '')} | {row.get('peakOverBaselineGb', '')} | {row.get('peakGpuGb', '')} | "
            f"{str(row.get('message', '')).replace('|', '/')[:160]} |"
        )
    lines += ["", *notes]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="image variant id from /api/images/catalog")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--token", default=os.environ.get("CHAOSENGINE_API_TOKEN"), help="API bearer token, if the backend needs one")
    parser.add_argument("--pid", type=int, help="backend process id (default: the process listening on --port)")
    parser.add_argument("--factors", default=",".join(f"{f:g}" for f in DEFAULT_FACTORS), help="multiples of the native side")
    parser.add_argument("--aspect", default="1:1", help="width:height of every step (long edge = native side x factor)")
    parser.add_argument("--steps", type=int, default=1, help="denoising steps per image")
    parser.add_argument("--prompt", default="a lighthouse on a cliff at sunrise, detailed")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between memory samples")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds to wait for one image")
    parser.add_argument("--no-preload", action="store_true", help="skip the preload (the first image then includes the model load)")
    parser.add_argument("--keep-outputs", action="store_true", help="keep the generated images in the gallery")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / ".chaosengine" / "test-results")
    args = parser.parse_args()

    factors = tuple(float(part) for part in args.factors.split(",") if part.strip())
    aspect = parse_aspect(args.aspect)
    backend = Backend(args.port, args.token)

    status, health = backend.request("GET", "/api/health", timeout=10)
    if status != 200:
        raise SystemExit(f"Cannot reach the backend on port {args.port} (HTTP {status}). Is it running?")
    variant = find_variant(backend, args.model)
    pid = args.pid or listening_pid(args.port)
    if pid is None:
        raise SystemExit("Cannot find the backend process; pass --pid.")
    use_footprint = sys.platform == "darwin" and shutil.which("footprint") is not None

    native = int(variant.get("nativeSide") or 1024)
    max_side, max_pixels = variant.get("maxSide"), variant.get("maxPixels")
    print(f"{variant.get('name')}: native {native}px, catalog maxSide {max_side}, maxPixels {max_pixels}")
    print(f"backend pid {pid}; memory via {'footprint' if use_footprint else 'psutil rss'}", flush=True)

    if not args.no_preload:
        status, payload = backend.request("POST", "/api/images/preload", {"modelId": args.model}, timeout=args.timeout)
        if status != 200:
            raise SystemExit(f"Preload failed (HTTP {status}): {error_text(payload)}")

    rows: list[dict] = []
    notes: list[str] = []
    for factor, width, height in size_steps(native, factors, aspect):
        if exceeds_model_limit(width, height, max_side, max_pixels):
            rows.append({"factor": factor, "requested": f"{width}x{height}", "status": "beyond-limit",
                         "message": "above the catalog maxSide / maxPixels; the backend would clamp it"})
            notes.append(f"Stopped before {width}x{height}: above the model's catalog limit.")
            break
        print(f"  {factor:g}x -> {width}x{height} ...", end=" ", flush=True)
        row = measure_step(backend, args, pid, use_footprint, width, height)
        row["factor"] = factor
        rows.append(row)
        print(f"{row['status']} in {row['seconds']}s, peak {row['peakGb']} GB "
              f"(+{row['peakOverBaselineGb']} over baseline)", flush=True)
        if not args.keep_outputs:
            for artifact_id in row.pop("artifactIds", []):
                backend.request("DELETE", f"/api/images/outputs/{artifact_id}", timeout=30)
        if row["status"] != "ok":
            notes.append(f"Stopped at {width}x{height}: {row['status']}. {row.get('message', '')}")
            break

    ran = [row for row in rows if row["status"] == "ok"]
    if ran:
        best = ran[-1]
        notes.append(
            f"Largest size that ran: {best['requested']} in {best['seconds']} s, peak {best['peakGb']} GB "
            f"({best['peakOverBaselineGb']} GB over the {best['baselineGb']} GB before the run)."
        )
    else:
        notes.append("No size ran.")
    notes.append(
        "Wall time includes engine start-up for engines that load per image (mflux, stable-diffusion.cpp). "
        "Peak is the highest sample, so a spike shorter than the interval can be missed."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "model": args.model,
        "variant": {k: variant.get(k) for k in ("name", "maxSide", "maxPixels", "nativeSide", "recommendedResolution", "engine")},
        "steps": args.steps,
        "aspect": args.aspect,
        "memoryMethod": "footprint" if use_footprint else "psutil-rss",
        "platform": sys.platform,
        "rows": rows,
        "notes": notes,
    }
    json_path = args.output_dir / f"image-ceiling-{slug}-{stamp}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(args.model, variant, rows, notes)
    json_path.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Wrote {json_path} and {json_path.with_suffix('.md')}")
    return 0 if ran else 1


if __name__ == "__main__":
    sys.exit(main())
