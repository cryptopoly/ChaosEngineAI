#!/usr/bin/env python3
"""Opt-in real test of Stable Audio Open 1.0 (stabilityai/stable-audio-open-1.0) through a running backend.

NOT part of any automated run: it downloads about 5.7 GB of gated weights and makes two clips that each take
minutes. Do it once, after you have opened https://huggingface.co/stabilityai/stable-audio-open-1.0 while signed
in and accepted the licence there.

It is a plain HTTP client of the app's own routes, the same ones the Models page and the Sounds studio call:

  1. checks the catalog row (gated, pinned revision, 47 s maximum) and the runtime;
  2. installs the sounds pack if it is not installed (about 1 minute);
  3. downloads the weights through the app (uses the Hugging Face token already saved in the app's Settings; this
     script never reads, prints or sends that token, and reports a refused download in the app's own words);
  4. loads the model on its own and times it (explicit preload, so load time is separate from generation);
  5. makes one 20 s and one 45 s clip (8 steps is NOT used: the model's own 100 steps, cfg 7, dpmpp-3m-sde), timing
     each run and sampling the worker's memory footprint once a second while it works;
  6. checks each FLAC with numpy and soundfile if they are importable (RMS, peak, clipped samples, loudness per
     half second near the end, channel correlation) and prints a report in the format of
     docs/research/sound-effects-real-test.md.

Start a backend first, with the throwaway-free data folder you want to test in, for example:

    CHAOSENGINE_API_TOKEN=local-test CHAOSENGINE_PORT=8908 .venv/bin/python -m backend_service.app

then:

    CHAOSENGINE_API_TOKEN=local-test .venv/bin/python scripts/real-test-sound-1.0.py --base-url http://127.0.0.1:8908

``CHAOSENGINE_API_TOKEN`` is the app's own bearer token for its local API, not a Hugging Face token. Options:
``--skip-download`` (weights already there), ``--out FILE`` (also write the report as Markdown), ``--steps`` is
deliberately absent: the model card's settings are what the catalog row ships.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

MODEL = "stabilityai/stable-audio-open-1.0"
CLIPS = (
    ("rain on a tin roof, steady, close microphone", 20.0, 7),
    ("a thunderstorm rolling over a valley, distant thunder and heavy rain", 45.0, 7),
)
POLL_S = 3.0


class Api:
    def __init__(self, base: str, token: str) -> None:
        self.base = base.rstrip("/")
        self.token = token

    def call(self, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.token}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                return json.loads(reply.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except ValueError:
                pass
            raise SystemExit(f"{method} {path} failed with HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise SystemExit(f"Cannot reach the backend at {self.base}: {exc.reason}") from None

    def raw(self, path: str, timeout: float = 120.0) -> bytes:
        request = urllib.request.Request(self.base + path)
        request.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return reply.read()


def say(text: str) -> None:
    print(text, flush=True)


def check_row(api: Api) -> dict[str, Any]:
    families = api.call("GET", "/api/music/catalog")["families"]
    row = next((v for f in families for v in f["variants"] if v["id"] == MODEL), None)
    if row is None:
        raise SystemExit(f"{MODEL} is not in this build's catalog.")
    licence = row.get("licence") or {}
    say(f"Catalog row: {row['name']}, revision {str(row['revision'])[:12]}, up to {row['maxSeconds']} s, "
        f"{row['sizeGb']} GB, gated={licence.get('gated')}, attribution={licence.get('attribution')!r}, "
        f"estimates basis={row['estimates']['basis']}.")
    return row


def install_pack(api: Api) -> float | None:
    runtime = api.call("GET", "/api/music/runtime")["runtime"]
    sound = runtime.get("sound") or {}
    if sound.get("unavailableReason"):
        raise SystemExit(f"Sound effects cannot run here: {sound['unavailableReason']}")
    if sound.get("installed"):
        say("Sounds pack: already installed.")
        return None
    say("Sounds pack: installing ...")
    started = time.monotonic()
    api.call("POST", "/api/music/sound/install")
    while True:
        time.sleep(POLL_S)
        job = api.call("GET", "/api/music/sound/install/status")
        if job.get("error"):
            raise SystemExit(f"Sounds pack install failed: {job['error']}")
        if not job.get("running") and api.call("GET", "/api/music/runtime")["runtime"]["sound"].get("installed"):
            break
    seconds = time.monotonic() - started
    say(f"Sounds pack: installed in {seconds:.0f} s.")
    return seconds


def download(api: Api) -> float | None:
    row = check_row(api)
    if row.get("availableLocally"):
        say("Weights: already downloaded.")
        return None
    say(f"Weights: downloading {row['sizeGb']} GB through the app ...")
    started = time.monotonic()
    api.call("POST", "/api/music/download", {"repo": MODEL})
    last = ""
    while True:
        time.sleep(POLL_S)
        status = next((d for d in api.call("GET", "/api/music/download/status")["downloads"] if d.get("repo") == MODEL), None)
        if status is None:
            continue
        line = f"{status.get('state')} {float(status.get('progress') or 0) * 100:.0f}% " \
               f"({status.get('downloadedGb')} / {status.get('totalGb')} GB)"
        if line != last:
            say("  " + line)
            last = line
        if status.get("state") == "failed":
            raise SystemExit(f"Download failed: {status.get('error')}")
        if status.get("state") == "completed":
            break
    seconds = time.monotonic() - started
    say(f"Weights: downloaded in {seconds:.0f} s.")
    return seconds


class FootprintSampler(threading.Thread):
    """The worker's own footprint (GB) from /api/music/runtime once a second; keeps the largest."""

    def __init__(self, api: Api) -> None:
        super().__init__(daemon=True)
        self.api = api
        self.peak = 0.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(1.0):
            try:
                value = self.api.call("GET", "/api/music/runtime", timeout=10)["runtime"]["sound"].get("footprintGb")
            except SystemExit:
                continue
            if isinstance(value, (int, float)):
                self.peak = max(self.peak, float(value))

    def stop(self) -> float:
        self._stop_event.set()
        self.join(5)
        return self.peak


def objective_checks(path: str, seconds: float) -> dict[str, Any] | None:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None
    samples, rate = sf.read(path, always_2d=True)
    info = sf.info(path)
    peak = float(np.max(np.abs(samples))) or 1e-9
    rms = float(np.sqrt(np.mean(samples**2))) or 1e-9
    half = max(1, rate // 2)
    tail = [20 * np.log10(max(float(np.sqrt(np.mean(samples[i:i + half] ** 2))), 1e-9))
            for i in range(max(0, samples.shape[0] - 3 * half), samples.shape[0], half)]
    corr = float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]) if samples.shape[1] > 1 else 1.0
    return {
        "format": f"{info.format} {info.subtype}, {rate / 1000:g} kHz, {samples.shape[1]} ch",
        "length": samples.shape[0] / rate,
        "rms_db": 20 * np.log10(rms),
        "peak_db": 20 * np.log10(peak),
        "clipped": int(np.sum(np.abs(samples) >= 0.9999)),
        "tail_db": [round(float(v), 1) for v in tail],
        "corr": corr,
    }


def run_clip(api: Api, prompt: str, seconds: float, seed: int) -> dict[str, Any]:
    say(f"Clip: {seconds:g} s, {prompt!r} (this takes minutes) ...")
    sampler = FootprintSampler(api)
    sampler.start()
    started = time.monotonic()
    result = api.call("POST", "/api/music/sound/generate",
                      {"modelId": MODEL, "caption": prompt, "seconds": seconds, "seed": seed}, timeout=3600)
    wall = time.monotonic() - started
    sampled_peak = sampler.stop()
    item = result["items"][0]
    audio_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"sao1-{item['id']}.flac")
    with open(audio_path, "wb") as handle:
        handle.write(api.raw(f"/api/music/library/{item['id']}/audio"))
    checks = objective_checks(audio_path, seconds)
    say(f"  done in {wall:.1f} s (job {result.get('seconds')} s), worker peak {result.get('peakFootprintGb')} GB "
        f"(sampled {sampled_peak:.2f} GB)")
    return {"prompt": prompt, "asked": seconds, "wall": wall, "job": result.get("seconds"), "timings": result.get("timings"),
            "peak_reported": result.get("peakFootprintGb"), "peak_sampled": sampled_peak, "item": item,
            "file": audio_path, "checks": checks}


def report(machine: str, pack_s: float | None, dl_s: float | None, preload: dict[str, Any], clips: list[dict[str, Any]], row: dict[str, Any]) -> str:
    lines = [
        "# Sound effects: Stable Audio Open 1.0 real run",
        "",
        f"Date {time.strftime('%Y-%m-%d')}. Machine: {machine}. Model: `{MODEL}` at the pinned revision "
        f"`{str(row['revision'])[:12]}`, plus `google-t5/t5-base`.",
        "",
        "## What was installed (through the app's own routes)",
        "",
        f"- Sounds pack: {'already installed' if pack_s is None else f'{pack_s:.0f} s'}.",
        f"- Weights: {'already downloaded' if dl_s is None else f'{dl_s:.0f} s'} (catalog says {row['sizeGb']} GB).",
        f"- Explicit model load: {preload['wall']:.1f} s, footprint {preload.get('footprint')} GB.",
        "",
        "## Clips (warm worker, 100 steps, cfg 7, dpmpp-3m-sde, seed 7, default peak -1 dBFS)",
        "",
        "| Prompt | Asked | Got | Job time | Peak footprint (worker) | File | Format |",
        "|---|---|---|---|---|---|---|",
    ]
    for clip in clips:
        item, checks = clip["item"], clip["checks"] or {}
        lines.append(
            f"| {clip['prompt']} | {clip['asked']:g} s | {float(item.get('durationS') or 0):.2f} s | {clip['wall']:.1f} s "
            f"| {clip['peak_reported']} GB (sampled {clip['peak_sampled']:.2f}) | {int(item.get('sizeBytes') or 0) / 1e6:.2f} MB "
            f"| {checks.get('format', 'n/a')} |")
    lines += ["", "## Objective checks (numbers only; listen before judging)", "",
              "| Clip | RMS | Peak | Clipped samples | Last 1.5 s per half second | L/R correlation |", "|---|---|---|---|---|---|"]
    for clip in clips:
        c = clip["checks"]
        if c is None:
            lines.append(f"| {clip['prompt']} | numpy/soundfile not installed | | | | |")
            continue
        lines.append(f"| {clip['prompt']} | {c['rms_db']:.1f} dBFS | {c['peak_db']:.1f} dBFS | {c['clipped']} "
                     f"| {', '.join(str(v) for v in c['tail_db'])} dB | {c['corr']:.2f} |")
    est = row["estimates"]
    measured_clips = ", ".join("%g s clip in %.0f s" % (c["asked"], c["wall"]) for c in clips)
    lines += ["", "## Against the catalog's estimates (to replace them)", "",
              f"- Estimated: {est['wallSecondsAt'][0][1]:g} s per clip warm, {est['coldStartSeconds']:g} s cold start, "
              f"peak {est['peakGb']} GB, idle {est['idleGb']} GB (basis: {est['basis']}).",
              f"- Measured here: {measured_clips}; "
              f"load {preload['wall']:.1f} s; peak {max(float(c['peak_reported'] or 0) for c in clips):.1f} GB; "
              f"idle {preload.get('footprint')} GB.",
              "- Then set `estimates.basis` to `measured`, fill `measuredOn`, `wallSecondsAt`, `secondsPer10sAudio`, "
              "`coldStartSeconds`, `peakGb`, `idleGb` in `backend_service/catalog/music_models.py` from these numbers.",
              "", "Nobody has listened to these clips unless the owner says so below."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8908")
    parser.add_argument("--token", default=os.environ.get("CHAOSENGINE_API_TOKEN", ""), help="the app's local API bearer token (not a Hugging Face token)")
    parser.add_argument("--skip-download", action="store_true", help="the weights are already in place")
    parser.add_argument("--out", help="also write the report to this Markdown file")
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("Set CHAOSENGINE_API_TOKEN (the backend's own API token) or pass --token.")
    api = Api(args.base_url, args.token)
    api.call("GET", "/api/health")
    row = check_row(api)
    pack_s = install_pack(api)
    dl_s = None if args.skip_download else download(api)
    say("Model: loading on its own ...")
    started = time.monotonic()
    api.call("POST", "/api/music/preload", {"modelId": MODEL}, timeout=1800)
    preload = {"wall": time.monotonic() - started}
    preload["footprint"] = (api.call("GET", "/api/music/runtime")["runtime"]["sound"] or {}).get("footprintGb")
    say(f"  loaded in {preload['wall']:.1f} s, footprint {preload['footprint']} GB")
    clips = [run_clip(api, prompt, seconds, seed) for prompt, seconds, seed in CLIPS]
    machine = os.environ.get("MACHINE_LABEL") or " ".join(os.uname()[:1] + os.uname()[4:5])
    text = report(machine, pack_s, dl_s, preload, clips, row)
    say("\n" + text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        say(f"Report written to {args.out}")
    say("Clips are saved in the Music Library; copies for listening: " + ", ".join(c["file"] for c in clips))
    return 0


if __name__ == "__main__":
    sys.exit(main())
