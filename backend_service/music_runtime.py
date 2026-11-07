"""Music runtime: settings, run estimates, load / unload, and one generation into the Music Library.

The model never loads in the backend: ``music_env`` builds the venv, ``music_worker`` runs in it,
``music_client`` talks to it. This module decides which catalog row a request uses, checks the pack and
weights are ready, mirrors the weights into the checkpoints layout, maps the worker's stages onto the
``MUSIC_PROGRESS`` phases (load, encode, compose, decode, save), tags each FLAC (D-34 switch
``musicEmbedTag``) and moves it into the library. ``estimate_run`` reads only the catalog row's
``estimates`` table.
"""

from __future__ import annotations

import json
import random
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from backend_service import music_env

# Worker stage -> progress phase the Music Studio shows.
STAGE_PHASES = {"load": "load", "encode": "encode", "diffusion": "compose", "decode": "decode", "write": "save"}
PHASE_MESSAGES = {
    "load": "Loading the music model",
    "encode": "Reading the prompt",
    "compose": "Composing",
    "decode": "Decoding audio",
    "save": "Saving",
}
LOAD_TIMEOUT_S = 300.0
MAX_SEED = 2**31 - 1

DEFAULT_SETTINGS: dict[str, Any] = {"idleUnloadMinutes": 10}
_SETTINGS_LOCK = threading.Lock()


class MusicNotReady(RuntimeError):
    """The pack, the weights or a required pack is missing (HTTP 409 with the fix in the message)."""


# ---------------------------------------------------------------------------
# Settings (``<music data>/settings.json``) and the D-34 switch (app settings)
# ---------------------------------------------------------------------------

def _settings_path() -> Path:
    return music_env.MUSIC_DATA_DIR / "settings.json"


def load_settings() -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    try:
        stored = json.loads(_settings_path().read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            settings.update({k: v for k, v in stored.items() if k in DEFAULT_SETTINGS})
    except (OSError, ValueError):
        pass
    return settings


def save_settings(patch: dict[str, Any]) -> dict[str, Any]:
    with _SETTINGS_LOCK:
        settings = load_settings()
        settings.update({k: v for k, v in patch.items() if k in DEFAULT_SETTINGS and v is not None})
        path = _settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    from backend_service.music_client import get_worker

    get_worker().idle_minutes = float(settings["idleUnloadMinutes"])
    return settings


def embed_tag_enabled() -> bool:
    """D-34: the machine-generated tag is on unless the user switched ``musicEmbedTag`` off."""
    try:
        from backend_service import app as app_module

        return bool(app_module._load_settings(app_module.SETTINGS_PATH).get("musicEmbedTag", True))
    except Exception:  # noqa: BLE001 - an unreadable settings file keeps the default (on)
        return True


# ---------------------------------------------------------------------------
# Catalog rows and estimates
# ---------------------------------------------------------------------------

def resolve_variant(model_id: str | None) -> dict[str, Any]:
    from backend_service.helpers.music import default_music_variant, find_music_variant

    if not model_id:
        return default_music_variant()
    variant = find_music_variant(model_id)
    if variant is None:
        raise ValueError(f"Unknown music model '{model_id}'.")
    return variant


def _wall_seconds(points: list[list[float]], seconds: float) -> float:
    """Linear interpolation over ``(audio seconds, wall seconds)``; proportional outside the table."""
    table = sorted((float(a), float(w)) for a, w in points)
    if seconds <= table[0][0]:
        return table[0][1] * seconds / table[0][0]
    for (a0, w0), (a1, w1) in zip(table, table[1:]):
        if seconds <= a1:
            return w0 + (w1 - w0) * (seconds - a0) / (a1 - a0)
    (a0, w0), (a1, w1) = table[-2], table[-1]
    return w1 + (w1 - w0) / (a1 - a0) * (seconds - a1)


def estimate_run(model: str | None, seconds: float, variations: int = 1, warm: bool = True) -> dict[str, Any]:
    """Time, peak memory and disk for one run, from the row's ``estimates`` table only."""
    variant = resolve_variant(model)
    est = variant["estimates"]
    count = max(1, int(variations))
    wall = _wall_seconds(est["wallSecondsAt"], float(seconds)) * count
    cold = 0.0 if warm else float(est.get("coldStartSeconds") or 0.0)
    return {
        "model": variant["id"],
        "seconds": round(wall + cold, 1),
        "coldStartSeconds": round(cold, 1),
        "warm": warm,
        "peakGb": est.get("peakGb"),
        "idleGb": est.get("idleGb"),
        "writesMb": round(float(est.get("outputMbPerMinute") or 0.0) * float(seconds) / 60.0 * count, 1),
        "modelDiskGb": est.get("diskGb"),
        "basis": est.get("basis", "estimated"),
        "measuredOn": est.get("measuredOn"),
    }


# ---------------------------------------------------------------------------
# Readiness, load, unload
# ---------------------------------------------------------------------------

def readiness_error(variant: dict[str, Any]) -> str | None:
    """The plain reason this row cannot run now, with the fix; ``None`` when it can."""
    from backend_service.helpers.music import find_music_variant, weights_source

    reason = music_env.lane_unavailable_reason()
    if reason:
        return reason
    if not music_env.is_installed():
        return "The music pack is not installed. Install it from Music > Models."
    if weights_source(variant) is None:
        return f"{variant['name']} is not downloaded yet ({variant['sizeGb']:.1f} GB). Download it from Music > Models."
    for repo in variant.get("requires") or []:
        needed = find_music_variant(repo)
        if needed is not None and weights_source(needed) is None:
            return f"{variant['name']} also needs {needed['name']}. Download it from Music > Models."
    return None


def _prepare(variant: dict[str, Any]) -> None:
    from backend_service.catalog.music_models import music_variants

    error = readiness_error(variant)
    if error:
        raise MusicNotReady(error)
    music_env.prepare_checkpoints(music_variants())


def worker_status() -> dict[str, Any] | None:
    from backend_service.music_client import get_worker

    return get_worker().status()


def loaded_model() -> str | None:
    status = worker_status()
    return str(status["model"]) if status and status.get("model") else None


def load(model_id: str | None) -> dict[str, Any]:
    from backend_service.music_client import get_worker

    variant = resolve_variant(model_id)
    _prepare(variant)
    reply = get_worker().request("load", timeout=LOAD_TIMEOUT_S, engine="acestep", model=variant["engineModel"], lm=None)
    return {"model": variant["id"], "engineModel": variant["engineModel"], "loadSeconds": reply.get("load_s"),
            "footprintGb": reply.get("footprint_gb"), "stepHook": reply.get("step_hook")}


def unload() -> dict[str, Any]:
    """Stop the worker: process exit frees all of its memory (an in-process unload leaves about 1 GB)."""
    from backend_service.music_client import get_worker

    worker = get_worker()
    was_running = worker.running()
    worker.stop()
    return {"stopped": was_running}


def free_idle_worker(state: Any = None, why: str = "another job") -> bool:
    """Stop an idle music worker (about 15 GB) so image or video work gets the memory."""
    from backend_service import music_client

    worker = music_client._WORKER
    if worker is None or not worker.stop_if_idle():
        return False
    if state is not None:
        state.add_log("music", "info", f"Stopped the idle music worker to free memory for {why}.")
    return True


def capabilities() -> dict[str, Any]:
    from backend_service.music_client import get_worker

    worker = get_worker()
    status = worker.status() if worker.running() else None
    return {
        "lane": music_env.lane(),
        "unavailableReason": music_env.lane_unavailable_reason(),
        "installed": music_env.is_installed(),
        "engine": "acestep",
        "engineVersion": music_env.ENGINE_VERSION,
        "engineRef": music_env.ACE_REF,
        "workerPid": worker.pid,
        "workerRunning": worker.running(),
        "loadedModel": (status or {}).get("model"),
        "busy": (status or {}).get("busy"),
        "footprintGb": (status or {}).get("footprint_gb"),
        "idleUnloadMinutes": worker.idle_minutes,
        "embedTag": embed_tag_enabled(),
        "libraryDir": str(music_env.library_dir()),
        "settings": load_settings(),
        "install": music_env.job_status(),
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _encoder_label(variant: dict[str, Any]) -> str:
    from backend_service.helpers.system_hardware import _resolve_app_version

    return f"ChaosEngineAI {_resolve_app_version()} / ACE-Step 1.5 ({music_env.ENGINE_VERSION}, {variant['engineModel']})"


def _save_track(track: dict[str, Any], index: int, variant: dict[str, Any], request: dict[str, Any],
                reply: dict[str, Any], tag: bool) -> dict[str, Any]:
    from backend_service import music_library
    from backend_service.helpers.audio_metadata import embed_flac_tags
    from backend_service.helpers.system_hardware import _resolve_app_version

    path = Path(str(track["path"]))
    tagged = False
    if tag:
        tagged = embed_flac_tags(path, {
            "appVersion": _resolve_app_version(),
            "model": variant["id"],
            "modelName": variant["name"],
            "engine": f"acestep {music_env.ENGINE_VERSION}",
            "seed": track.get("seed"),
            "durationSeconds": track.get("duration_s"),
            "bpm": request.get("bpm"),
            "key": request.get("key"),
            "prompt": request.get("caption"),
            "instrumental": bool(request.get("instrumental")),
            "variation": index + 1 if int(request.get("variations") or 1) > 1 else None,
        }, _encoder_label(variant))
    title = request.get("title") or None
    if title and int(request.get("variations") or 1) > 1:
        title = f"{title} ({index + 1})"
    return music_library.create({
        "title": title,
        "caption": request.get("caption"),
        "lyrics": "" if request.get("instrumental") else request.get("lyrics"),
        "model": variant["id"],
        "modelName": variant["name"],
        "engine": "acestep",
        "seed": track.get("seed"),
        "durationS": track.get("duration_s"),
        "sampleRate": track.get("sample_rate"),
        "channels": track.get("channels"),
        "bpm": request.get("bpm"),
        "key": request.get("key"),
        "instrumental": bool(request.get("instrumental")),
        "variation": index + 1,
        "jobSeconds": reply.get("seconds"),
        "peakFootprintGb": reply.get("peak_footprint_gb"),
        "source": request.get("source") or "ui",
        "tagged": tagged,
    }, path, ".flac")


def generate(
    request: dict[str, Any],
    *,
    progress: Any = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run one text-to-music request; returns the saved library items, the seed and the timings.

    ``request`` keys: model, caption, lyrics, instrumental, seconds, variations, bpm, key, seed,
    vocalLanguage, timeSignature, title, source. ``progress`` is a ``ProgressTracker``.
    """
    from backend_service.music_client import generate_timeout_s, get_worker, stall_limits

    variant = resolve_variant(request.get("model"))
    _prepare(variant)
    worker = get_worker()
    seed = request.get("seed")
    seed = int(seed) if seed is not None else random.randint(0, MAX_SEED)
    seconds = float(request["seconds"])
    variations = int(request.get("variations") or 1)

    def on_event(event: dict[str, Any]) -> None:
        if progress is None:
            return
        if event.get("event") == "stage":
            phase = STAGE_PHASES.get(str(event.get("stage")), str(event.get("stage")))
            progress.set_phase(phase, PHASE_MESSAGES.get(phase, ""))
        elif event.get("event") == "step":
            progress.set_step(int(event.get("step") or 0), int(event.get("total") or 0))

    status = worker.status()
    if not status or status.get("model") != variant["engineModel"]:
        if progress is not None:
            progress.set_phase("load", PHASE_MESSAGES["load"])
        worker.request("load", timeout=LOAD_TIMEOUT_S, cancel_check=cancel_check, engine="acestep",
                       model=variant["engineModel"], lm=None)
    job = f"mus_{uuid.uuid4().hex[:12]}"
    out_dir = music_env.tmp_dir() / job
    defaults = variant.get("defaultSettings") or {}
    try:
        reply = worker.request(
            "generate",
            timeout=generate_timeout_s(seconds, variations, variant["id"]),
            stall_limits=stall_limits(seconds, variations, variant["id"]),
            on_event=on_event,
            cancel_check=cancel_check,
            job=job,
            engine="acestep",
            model=variant["engineModel"],
            task="text2music",
            caption=request.get("caption") or "",
            lyrics="[Instrumental]" if request.get("instrumental") else (request.get("lyrics") or ""),
            instrumental=bool(request.get("instrumental")),
            vocal_language=request.get("vocalLanguage") or defaults.get("vocalLanguage") or "en",
            seconds=seconds,
            bpm=request.get("bpm"),
            key=request.get("key"),
            time_signature=request.get("timeSignature"),
            seed=seed,
            variations=variations,
            steps=int(defaults.get("steps") or 8),
            shift=float(defaults.get("shift") or 3.0),
            planner=False,
            src_audio=None,
            reference_audio=None,
            cover_strength=None,
            repaint=None,
            out_dir=str(out_dir.resolve()),
            format="flac24",
            normalize_peak_db=defaults.get("normalizePeakDb", -1.0),
        )
        if progress is not None:
            progress.set_phase("save", PHASE_MESSAGES["save"])
        tag = embed_tag_enabled()
        items = [_save_track(t, i, variant, request, reply, tag) for i, t in enumerate(reply.get("tracks") or [])]
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    return {
        "items": items,
        "seed": seed,
        "jobId": job,
        "model": variant["id"],
        "seconds": reply.get("seconds"),
        "timings": reply.get("timings") or {},
        "peakFootprintGb": reply.get("peak_footprint_gb"),
    }
