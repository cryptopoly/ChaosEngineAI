"""Music API routes: catalog, runtime, estimate, progress, cancel, preload, unload and generate.

Routes are plain ``def`` so FastAPI runs them in its thread pool; the model runs in the music worker.
``generate`` takes a ``HEAVY_JOBS`` slot (kind ``music``) before the memory gate, reports progress on
``MUSIC_PROGRESS`` and saves every track into the Music Library. A Studio Cancel stops the running job
only when it is the Studio's own. Library, install and download routes are in ``routes/music_library.py``.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend_service.progress import MUSIC_PROGRESS, GenerationCancelled

router = APIRouter()

_NOTES = ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B")
KEYS = tuple(f"{note} {mode}" for note in _NOTES for mode in ("major", "minor"))
MAX_SEED = 2**31 - 1


class MusicGenerateRequest(BaseModel):
    """Text to music. Cover, repaint and reference audio are not exposed yet (unknown fields are refused)."""

    model_config = ConfigDict(extra="forbid")

    modelId: str | None = Field(default=None, max_length=128)
    caption: str = Field(min_length=1, max_length=512)
    lyrics: str = Field(default="", max_length=4096)
    instrumental: bool = False
    seconds: float = Field(default=60, ge=10, le=600)
    variations: Literal[1, 2, 4] = 1
    bpm: int | None = Field(default=None, ge=30, le=300)
    key: str | None = Field(default=None, max_length=16)
    timeSignature: Literal[2, 3, 4, 6] | None = None
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    vocalLanguage: str | None = Field(default=None, pattern=r"^([a-z]{2}|unknown)$")
    task: Literal["text2music"] = "text2music"
    title: str | None = Field(default=None, max_length=200)

    @field_validator("key")
    @classmethod
    def _known_key(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if value not in KEYS:
            raise ValueError("key must be a note and mode such as 'A minor' or 'F# major'")
        return value

    @model_validator(mode="after")
    def _lyrics_or_instrumental(self) -> "MusicGenerateRequest":
        if not self.caption.strip():
            raise ValueError("Describe the music (caption is empty).")
        if not self.instrumental and not self.lyrics.strip():
            raise ValueError("Add lyrics, or switch Instrumental on.")
        return self


class SoundGenerateRequest(BaseModel):
    """Text to sound effect: a short clip (1 to 11 s), optionally made into a seamless loop."""

    model_config = ConfigDict(extra="forbid")

    modelId: str | None = Field(default=None, max_length=128)
    caption: str = Field(min_length=1, max_length=300)
    seconds: float = Field(default=5, ge=1, le=11)
    loop: bool = False
    variations: Literal[1, 2, 4] = 1
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    title: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _described(self) -> "SoundGenerateRequest":
        if not self.caption.strip():
            raise ValueError("Describe the sound (caption is empty).")
        return self


class MusicModelRequest(BaseModel):
    modelId: str | None = Field(default=None, max_length=128)


class MusicEstimateRequest(BaseModel):
    modelId: str | None = Field(default=None, max_length=128)
    seconds: float = Field(default=60, ge=1, le=600)
    variations: Literal[1, 2, 4] = 1


def _variant_or_400(model_id: str | None) -> dict[str, Any]:
    from backend_service.music_runtime import resolve_variant

    try:
        return resolve_variant(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Catalog, runtime, estimate, progress
# ---------------------------------------------------------------------------

@router.get("/api/music/catalog")
def music_catalog(request: Request) -> dict[str, Any]:
    from backend_service.helpers.music import music_model_payloads

    return {"families": music_model_payloads()}


@router.get("/api/music/runtime")
def music_runtime_status(request: Request) -> dict[str, Any]:
    from backend_service.music_runtime import capabilities

    return {"runtime": capabilities()}


@router.post("/api/music/estimate")
def music_estimate(request: Request, body: MusicEstimateRequest) -> dict[str, Any]:
    """The run estimate strip: time, peak memory, disk, basis, fit, and whether the queue is busy."""
    from backend_service.helpers.hardware_class import host_hardware_class
    from backend_service.helpers.job_queue import HEAVY_JOBS
    from backend_service.helpers.music_fit import music_fit
    from backend_service.music_runtime import estimate_run, loaded_model

    variant = _variant_or_400(body.modelId)
    if not float(variant.get("minSeconds") or 10) <= body.seconds <= float(variant.get("maxSeconds") or 600):
        raise HTTPException(status_code=400, detail=f"{variant['name']} makes {variant.get('minSeconds')} to "
                                                    f"{variant.get('maxSeconds')} seconds.")
    warm = loaded_model(variant) == variant["engineModel"]
    hardware = host_hardware_class()
    return {
        "estimate": estimate_run(variant["id"], body.seconds, body.variations, warm=warm),
        "fit": music_fit(variant, hardware),
        "totalGb": round(hardware.total_gb, 1),
        "queue": HEAVY_JOBS.snapshot(),
    }


@router.get("/api/music/progress")
def music_progress() -> dict[str, Any]:
    from backend_service.helpers.job_queue import HEAVY_JOBS

    return {"progress": {**MUSIC_PROGRESS.snapshot(), "queue": HEAVY_JOBS.queue_state("music")}}


@router.post("/api/music/cancel")
def cancel_music(request: Request) -> dict[str, Any]:
    """Stop the Studio's own running job (worker cancel, then a kill after 3 s) and its waiting jobs."""
    from backend_service.helpers.job_queue import HEAVY_JOBS

    signalled = HEAVY_JOBS.cancel_studio("music", MUSIC_PROGRESS.request_cancel)
    if signalled:
        request.app.state.chaosengine.add_log("music", "info", "Cancel signal sent to the running music job.")
    return {"cancelled": signalled}


# ---------------------------------------------------------------------------
# Load and unload
# ---------------------------------------------------------------------------

def _unload_other_heavy(request: Request, why: str) -> list[str]:
    """Free chat, image and video models before the XL model (it needs 47 GB on a 64 GB Mac)."""
    state = request.app.state.chaosengine
    freed: list[str] = []
    for label, unload in (
        ("image", lambda: state.image_runtime.unload()),
        ("video", lambda: state.video_runtime.unload()),
        ("chat", lambda: state.runtime.unload_model()),
    ):
        try:
            unload()
            freed.append(label)
        except Exception as exc:  # noqa: BLE001 - logged; the gate decides afterwards
            state.add_log("music", "warning", f"Could not unload the {label} model before {why}: {exc}")
    state.add_log("music", "info", f"Unloaded other models before {why}.")
    return freed


def _worker_http_error(exc: Exception) -> HTTPException:
    from backend_service.music_client import MusicWorkerError
    from backend_service.music_runtime import MusicNotReady

    if isinstance(exc, MusicNotReady):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, MusicWorkerError):
        return HTTPException(status_code=503, detail=f"Music generation failed: {exc}")
    return HTTPException(status_code=500, detail=f"Music generation failed: {type(exc).__name__}: {exc}")


@router.post("/api/music/preload")
def preload_music(request: Request, body: MusicModelRequest) -> dict[str, Any]:
    from backend_service.helpers.job_queue import HEAVY_JOBS
    from backend_service.music_runtime import capabilities, load

    variant = _variant_or_400(body.modelId)
    with HEAVY_JOBS.slot("music", "ui", f"load {variant['name']}"):
        if variant.get("unloadsOthers"):
            _unload_other_heavy(request, f"loading {variant['name']}")
        try:
            loaded = load(variant["id"])
        except Exception as exc:  # noqa: BLE001 - mapped to 400 / 409 / 503
            raise _worker_http_error(exc) from exc
    request.app.state.chaosengine.add_activity("Music model loaded", variant["name"])
    return {"loaded": loaded, "runtime": capabilities()}


@router.post("/api/music/unload")
def unload_music(request: Request) -> dict[str, Any]:
    from backend_service.music_runtime import MusicBusy, capabilities, unload

    try:
        result = unload()
    except MusicBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result["stopped"]:
        request.app.state.chaosengine.add_activity("Music model unloaded", "Music worker stopped")
    return {**result, "runtime": capabilities()}


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

@router.post("/api/music/generate")
def generate_music(request: Request, body: MusicGenerateRequest) -> dict[str, Any]:
    from backend_service.helpers.job_queue import HEAVY_JOBS

    label = body.title or body.caption[:60]
    try:
        with HEAVY_JOBS.slot("music", "ui", label):
            return generate_music_locked(request, body, source="ui")
    except GenerationCancelled:
        raise HTTPException(status_code=409, detail="cancelled") from None


def generate_music_locked(request: Request, body: MusicGenerateRequest, source: str = "ui") -> dict[str, Any]:
    """The model run; the caller holds the ``HEAVY_JOBS`` slot (the Studio route or an agent job)."""
    from backend_service import music_runtime
    from backend_service.helpers.memory_gate import (
        MUSIC_GENERATION_GB,
        MUSIC_IDLE_GB,
        gate_music_generation,
        snapshot_memory_signals,
    )
    from backend_service.music_client import MusicCancelled

    state = request.app.state.chaosengine
    variant = _variant_or_400(body.modelId)
    if music_runtime.is_sound(variant):
        raise HTTPException(status_code=400, detail=f"{variant['name']} makes sound effects. Use the Sounds mode.")
    not_ready = music_runtime.readiness_error(variant)
    if not_ready:
        raise HTTPException(status_code=409, detail=not_ready)
    if variant.get("unloadsOthers"):
        _unload_other_heavy(request, f"making music with {variant['name']}")
    estimates = variant.get("estimates") or {}
    try:
        available_gb, pressure = snapshot_memory_signals()
        refusal = gate_music_generation(
            available_gb,
            pressure,
            peak_gb=max(MUSIC_GENERATION_GB, float(estimates.get("peakGb") or 0.0)),
            idle_gb=float(estimates.get("idleGb") or MUSIC_IDLE_GB),
            resident=music_runtime.loaded_model() == variant["engineModel"],
            model_name=variant["name"],
            large=bool(variant.get("large")),
        )
    except Exception as exc:  # noqa: BLE001 - a broken memory read never blocks work
        state.add_log("music", "warning", f"Memory gate skipped: {exc}")
        refusal = None
    if refusal is not None:
        state.add_log("music", "warning", f"Memory gate refused music: {refusal['code']}")
        raise HTTPException(status_code=503, detail=refusal["message"])

    fields = {**body.model_dump(), "model": variant["id"], "source": source}
    MUSIC_PROGRESS.begin(run_label=f"{int(body.seconds)} s · {variant['name']}", total_steps=8,
                         phase="encode", message="Starting")
    try:
        result = music_runtime.generate(fields, progress=MUSIC_PROGRESS, cancel_check=MUSIC_PROGRESS.is_cancelled)
    except MusicCancelled:
        state.add_log("music", "info", f"Music generation cancelled ({variant['name']}).")
        raise GenerationCancelled("cancelled") from None
    except Exception as exc:  # noqa: BLE001 - mapped to 400 / 409 / 503 / 500
        state.add_log("music", "error", f"Music generation failed: {type(exc).__name__}: {exc}")
        raise _worker_http_error(exc) from exc
    finally:
        MUSIC_PROGRESS.finish()
    state.add_log("music", "info", f"Made {len(result['items'])} track(s) with {variant['name']} in {result['seconds']} s.")
    state.add_activity("Music generated", f"{variant['name']} · {int(body.seconds)} s")
    return result


# ---------------------------------------------------------------------------
# Sounds (Stable Audio Open)
# ---------------------------------------------------------------------------

def _sound_variant_or_400(model_id: str | None) -> dict[str, Any]:
    """The requested sound row, or the catalog's default sound row; a music row is refused."""
    from backend_service.catalog.music_models import music_variants
    from backend_service.music_runtime import is_sound

    if not model_id:
        found = next((v for v in music_variants() if is_sound(v)), None)
        if found is None:
            raise HTTPException(status_code=400, detail="No sound model in the catalog.")
        return found
    variant = _variant_or_400(model_id)
    if not is_sound(variant):
        raise HTTPException(status_code=400, detail=f"{variant['name']} makes music. Use the Music mode.")
    return variant


@router.post("/api/music/sound/generate")
def generate_sound(request: Request, body: SoundGenerateRequest) -> dict[str, Any]:
    from backend_service.helpers.job_queue import HEAVY_JOBS

    label = body.title or body.caption[:60]
    try:
        with HEAVY_JOBS.slot("music", "ui", label):
            return generate_sound_locked(request, body, source="ui")
    except GenerationCancelled:
        raise HTTPException(status_code=409, detail="cancelled") from None


def generate_sound_locked(request: Request, body: SoundGenerateRequest, source: str = "ui") -> dict[str, Any]:
    """The sound run; the caller holds the ``HEAVY_JOBS`` slot (the Studio route or an agent job)."""
    from backend_service import music_runtime
    from backend_service.helpers.memory_gate import gate_music_generation, snapshot_memory_signals
    from backend_service.music_client import MusicCancelled

    state = request.app.state.chaosengine
    variant = _sound_variant_or_400(body.modelId)
    not_ready = music_runtime.readiness_error(variant)
    if not_ready:
        raise HTTPException(status_code=409, detail=not_ready)
    estimates = variant.get("estimates") or {}
    try:
        available_gb, pressure = snapshot_memory_signals()
        refusal = gate_music_generation(
            available_gb,
            pressure,
            peak_gb=float(estimates.get("peakGb") or 6.0),
            idle_gb=float(estimates.get("idleGb") or 4.0),
            resident=music_runtime.loaded_model(variant) == variant["engineModel"],
            model_name=variant["name"],
            noun="a sound",
        )
    except Exception as exc:  # noqa: BLE001 - a broken memory read never blocks work
        state.add_log("music", "warning", f"Memory gate skipped: {exc}")
        refusal = None
    if refusal is not None:
        state.add_log("music", "warning", f"Memory gate refused a sound: {refusal['code']}")
        raise HTTPException(status_code=503, detail=refusal["message"])

    fields = {**body.model_dump(), "model": variant["id"], "source": source}
    MUSIC_PROGRESS.begin(run_label=f"{body.seconds:g} s sound · {variant['name']}", total_steps=8,
                         phase="encode", message="Starting")
    try:
        result = music_runtime.generate_sound(fields, progress=MUSIC_PROGRESS, cancel_check=MUSIC_PROGRESS.is_cancelled)
    except MusicCancelled:
        state.add_log("music", "info", f"Sound generation cancelled ({variant['name']}).")
        raise GenerationCancelled("cancelled") from None
    except Exception as exc:  # noqa: BLE001 - mapped to 400 / 409 / 503 / 500
        state.add_log("music", "error", f"Sound generation failed: {type(exc).__name__}: {exc}")
        raise _worker_http_error(exc) from exc
    finally:
        MUSIC_PROGRESS.finish()
    state.add_log("music", "info", f"Made {len(result['items'])} sound(s) with {variant['name']} in {result['seconds']} s.")
    state.add_activity("Sound generated", f"{variant['name']} · {body.seconds:g} s")
    return result
