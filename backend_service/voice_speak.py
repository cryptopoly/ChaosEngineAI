"""One way to speak: pick a voice, get audio (D-58).

The caller names a voice (a Kokoro voice, a Qwen3-TTS preset speaker, ``design`` or ``clone:<id>``); this
module picks the model, cleans and chunks the text, runs the voice worker once per chunk, joins the chunks,
and applies speed and pitch. Cloned voices keep going through ``voice_clone.speak`` so the permission rules,
the AI tag, the watermark and the audit log stay in one place; it calls ``render`` here for the audio.

Voice ids:
  ``af_heart`` ...   Kokoro voices (fast, many languages, speed only)
  ``qwen3:Ryan`` ... Qwen3-TTS CustomVoice preset speakers (style instruction on 1.7B)
  ``design``         Qwen3-TTS VoiceDesign: ``style`` holds the written description of the voice
  ``clone:<id>``     a cloned voice from the Voice Library

Speed and pitch: Kokoro does speed itself; every other model has neither, so both are ffmpeg filters on the
finished audio. Pitch is a shift in semitones (rate change, then tempo restored).
"""

from __future__ import annotations

import array
import base64
import io
import subprocess
import sys
import tempfile
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend_service import voice_env
from backend_service.progress import GenerationCancelled

MAX_CHARS = 20_000
CHUNK_CHARS = 400
SENTENCE_GAP_S = 0.15
MIN_SPEED, MAX_SPEED = 0.5, 2.0
MIN_PITCH, MAX_PITCH = -12.0, 12.0
PRESET_PREFIX = "qwen3:"
CLONE_PREFIX = "clone:"
DESIGN_VOICE = "design"
WORKER_TIMEOUT_S = 900.0
# A model copies the loudness of what it is given: a cloned voice from a quiet sample (peak near -28 dBFS on the
# 2026-11-08 run) speaks just as quietly. Speech whose peak is below QUIET_PEAK is lifted to LIFT_PEAK, by at most
# MAX_LIFT times; louder speech is left alone.
QUIET_PEAK = 0.2
LIFT_PEAK = 0.5
MAX_LIFT = 12.0

# Set by ``request_cancel`` (the Speak screen's Cancel); the running speech job stops at the next chunk and the
# worker is ended so a chunk in progress does not run on. ``render`` clears it when a job starts.
CANCEL = threading.Event()

_NO_FFMPEG = "Changing speed or pitch for this voice needs ffmpeg. Install it from Settings, or leave both at their defaults."


@dataclass
class Plan:
    """What to run for one voice."""

    kind: str  # kokoro | preset | design | clone
    voice: str
    model_key: str
    model_name: str
    engine: str = ""
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)  # worker fields for preset / design / clone
    native_speed: bool = False  # the model applies speed itself (Kokoro)
    notes: list[str] = field(default_factory=list)


@dataclass
class Rendered:
    wav: bytes
    duration_s: float
    seconds: float
    chunks: int
    notes: list[str]


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

def _caps(**flags: bool) -> dict[str, bool]:
    return {"speak": False, "style": False, "clone": False, "design": False, **flags}


def _models_of(kind: str) -> list[dict[str, Any]]:
    from backend_service.catalog.voice_models import VOICE_MODELS  # noqa: PLC0415

    return [row for row in VOICE_MODELS if row["kind"] == kind]


def _installed(row: dict[str, Any]) -> bool:
    from backend_service import voice_clone  # noqa: PLC0415

    return voice_clone.available_here() and voice_clone.snapshot_dir(row) is not None


def list_voices(*, include_clones: bool = True) -> list[dict[str, Any]]:
    """Built-in, preset, designed and cloned voices in one list.

    Fields: ``id``, ``name``, ``kind`` (builtin / preset / design / clone), ``language``, ``languageLabel``,
    ``gender`` (when known), ``model`` (the catalog key used by default, ``kokoro`` for built-ins), ``models``
    (every key that can speak it), ``capabilities`` (speak / style / clone / design), ``available`` (the
    voice can be used right now) and ``needs`` (plain words when it cannot).
    """
    from backend_service import voice_runtime  # noqa: PLC0415
    from backend_service.catalog.voice_models import CUSTOM_SPEAKERS  # noqa: PLC0415

    voices: list[dict[str, Any]] = []
    for v in voice_runtime.list_tts_voices():
        voices.append({
            "id": v["id"], "name": v["name"], "kind": "builtin", "language": v["language"],
            "languageLabel": v["languageLabel"], "gender": v["gender"], "model": "kokoro", "models": ["kokoro"],
            "capabilities": _caps(speak=True), "available": bool(v["available"]),
            "needs": None if v["available"] else f"{v['languageLabel']} voices need an extra install.",
        })

    customs = _models_of("custom")
    custom_ready = [row for row in customs if _installed(row)]
    for speaker in CUSTOM_SPEAKERS:
        voices.append({
            "id": PRESET_PREFIX + speaker["id"], "name": speaker["id"].replace("_", " "), "kind": "preset",
            "language": speaker["language"], "languageLabel": speaker["languageLabel"], "gender": speaker["gender"],
            "about": speaker["about"], "model": _pick_custom(customs, False)["key"], "models": [r["key"] for r in customs],
            "capabilities": _caps(speak=True, style=True), "available": bool(custom_ready),
            "needs": None if custom_ready else f"Download {customs[0]['name']} in Voice Models.",
        })

    design = _models_of("design")[0]
    design_ready = _installed(design)
    voices.append({
        "id": DESIGN_VOICE, "name": "Describe a voice", "kind": "design", "language": "auto", "languageLabel": "Any",
        "gender": None, "model": design["key"], "models": [design["key"]],
        "capabilities": _caps(speak=True, style=True, design=True), "available": design_ready,
        "needs": None if design_ready else f"Download {design['name']} in Voice Models.",
    })

    if include_clones:
        from backend_service import voice_clone  # noqa: PLC0415
        base = voice_clone.pick_clone_model("")  # the default, or any downloaded cloning model
        base_ready = _installed(base) and voice_env.watermark_installed()
        for item in voice_clone.list_voices():
            voices.append({
                "id": CLONE_PREFIX + item["id"], "name": item["name"], "kind": "clone", "language": "auto",
                "languageLabel": "Any", "gender": None, "model": base["key"],
                "models": [r["key"] for r in _models_of("base")], "capabilities": _caps(speak=True, clone=True),
                "available": base_ready, "needs": None if base_ready else f"Download {base['name']} in Voice Models.",
                "consent": item["consent"],
            })
    return voices


def _pick_custom(rows: list[dict[str, Any]], wants_style: bool, installed_only: bool = False) -> dict[str, Any]:
    """Style needs the 1.7B model; plain speech prefers the faster 0.6B one."""
    ordered = sorted(rows, key=lambda r: (wants_style and not r["capabilities"]["style"], r["sizeGb"]))
    if installed_only:
        ordered = [r for r in ordered if _installed(r)]
    if not ordered:
        raise LookupError("none")
    return ordered[0]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def _model_for(kind: str, model_key: str, wants_style: bool) -> dict[str, Any]:
    from backend_service.catalog.voice_models import voice_model  # noqa: PLC0415
    from backend_service import voice_clone  # noqa: PLC0415

    rows = _models_of(kind)
    if model_key:
        row = voice_model(model_key)
        if row["kind"] != kind:
            raise ValueError(f"{row['name']} cannot speak this voice.")
        chosen = row
    else:
        try:
            chosen = _pick_custom(rows, wants_style, installed_only=True)
        except LookupError:
            chosen = _pick_custom(rows, wants_style)
    if not voice_clone.available_here():
        raise ValueError("Preset and designed voices run on Apple Silicon Macs for now. Use a built-in voice here.")
    if not voice_env.is_installed():
        raise RuntimeError("The voice pack is not installed. Install it from Voice > Models.")
    if voice_clone.snapshot_dir(chosen) is None:
        raise RuntimeError(f"{chosen['name']} is not downloaded. Download it in Voice > Models.")
    return chosen


def build_plan(voice: str, style: str = "", model_key: str = "") -> Plan:
    """The model and worker fields for a non-cloned voice. Raises ValueError / RuntimeError with plain words."""
    from backend_service import voice_clone  # noqa: PLC0415
    from backend_service.catalog.voice_models import CUSTOM_SPEAKERS  # noqa: PLC0415

    voice = (voice or "").strip()
    style = " ".join((style or "").split())
    if voice.startswith(CLONE_PREFIX):
        raise ValueError("Cloned voices go through speak().")
    if voice.startswith(PRESET_PREFIX):
        wanted = voice[len(PRESET_PREFIX):].lower()
        speaker = next((s for s in CUSTOM_SPEAKERS if s["id"].lower() == wanted), None)
        if speaker is None:
            raise ValueError(f"Unknown voice '{voice}'.")
        row = _model_for("custom", model_key, bool(style))
        plan = Plan("preset", voice, row["key"], row["name"], row["engine"], str(voice_clone.snapshot_dir(row)),
                    {"mode": "preset", "speaker": speaker["id"], "language": "auto"})
        if style:
            if row["capabilities"]["style"]:
                plan.extra["instruct"] = style
            else:
                plan.notes.append("The 0.6B model ignores the style instruction. Download the 1.7B CustomVoice model to use it.")
        return plan
    if voice == DESIGN_VOICE:
        if not style:
            raise ValueError("Describe the voice you want (age, gender, tone, pace) in the style box.")
        row = _model_for("design", model_key, True)
        return Plan("design", voice, row["key"], row["name"], row["engine"], str(voice_clone.snapshot_dir(row)),
                    {"mode": "design", "instruct": style, "language": "auto"})
    from backend_service import voice_runtime  # noqa: PLC0415

    known = {v["id"]: v for v in voice_runtime.list_tts_voices()}
    if voice not in known:
        raise ValueError(f"Unknown voice '{voice}'.")
    plan = Plan("kokoro", voice, "kokoro", "Kokoro", native_speed=True)
    if style:
        plan.notes.append("Kokoro has no style control, so the style text was ignored.")
    return plan


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def wav_info(data: bytes) -> tuple[int, int, int, float]:
    """``(channels, sample width, sample rate, seconds)`` of a PCM WAV."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        rate = handle.getframerate()
        return handle.getnchannels(), handle.getsampwidth(), rate, handle.getnframes() / float(rate or 1)


def join_wavs(parts: list[bytes], gap_s: float = SENTENCE_GAP_S) -> bytes:
    """One WAV from several with the same format, a short silence between them."""
    if len(parts) == 1:
        return parts[0]
    frames: list[bytes] = []
    params = None
    for data in parts:
        with wave.open(io.BytesIO(data), "rb") as handle:
            current = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
            if params is None:
                params = current
            elif current != params:
                raise ValueError("The speech chunks came back in different audio formats.")
            frames.append(handle.readframes(handle.getnframes()))
    assert params is not None
    silence = b"\x00" * (int(gap_s * params[2]) * params[0] * params[1])
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(params[0])
        handle.setsampwidth(params[1])
        handle.setframerate(params[2])
        handle.writeframes(silence.join(frames))
    return out.getvalue()


def lift_quiet(wav: bytes) -> bytes:
    """Raise very quiet 16-bit speech to a normal level (see ``QUIET_PEAK``); anything else is returned as is."""
    with wave.open(io.BytesIO(wav), "rb") as handle:
        params, frames = handle.getparams(), handle.readframes(handle.getnframes())
    if params.sampwidth != 2 or not frames:
        return wav
    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max(max(samples), -min(samples)) / 32768.0
    if peak <= 0.0 or peak >= QUIET_PEAK:
        return wav
    gain = min(LIFT_PEAK / peak, MAX_LIFT)
    lifted = array.array("h", (max(-32768, min(32767, int(v * gain))) for v in samples))
    if sys.byteorder == "big":
        lifted.byteswap()
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setparams(params)
        handle.writeframes(lifted.tobytes())
    return out.getvalue()


def atempo_chain(factor: float) -> list[str]:
    """``atempo`` filters whose product is ``factor``; each one is limited to 0.5 to 2.0."""
    if factor <= 0:
        raise ValueError("Tempo must be above zero.")
    filters: list[str] = []
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    if abs(factor - 1.0) > 1e-6:
        filters.append(f"atempo={factor:.6f}")
    return filters


def filter_graph(sample_rate: int, speed: float, pitch: float) -> str:
    """The ffmpeg audio filter for a speed factor and a pitch shift in semitones; empty when both are neutral."""
    filters: list[str] = []
    if abs(pitch) > 1e-6:
        ratio = 2.0 ** (pitch / 12.0)
        filters.append(f"asetrate={sample_rate * ratio:.3f}")
        filters.append(f"aresample={sample_rate}")
        filters.extend(atempo_chain(1.0 / ratio))
    filters.extend(atempo_chain(speed))
    return ",".join(filters)


def process_audio(wav: bytes, speed: float = 1.0, pitch: float = 0.0) -> bytes:
    """Speed (pitch kept) and pitch (tempo kept) with ffmpeg. 1.0 and 0 return the audio untouched."""
    _channels, _width, rate, _seconds = wav_info(wav)
    graph = filter_graph(rate, speed, pitch)
    if not graph:
        return wav
    from backend_service.storyboard import ffmpeg_bin  # noqa: PLC0415

    exe = ffmpeg_bin.optional_ffmpeg_exe()
    if exe is None:
        raise RuntimeError(_NO_FFMPEG)
    with tempfile.TemporaryDirectory(prefix="chaosengine-speak-") as folder:
        source, target = Path(folder) / "in.wav", Path(folder) / "out.wav"
        source.write_bytes(wav)
        result = subprocess.run(
            [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
             "-af", graph, "-c:a", "pcm_s16le", str(target)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, check=False,
        )
        if result.returncode != 0 or not target.is_file():
            raise RuntimeError(f"ffmpeg could not change the speed or pitch: {result.stderr.strip()[-300:]}")
        return target.read_bytes()


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    """Clean markdown and code out of ``text`` and cut it into sentences of at most ``CHUNK_CHARS``."""
    from backend_service.voice_text import clean_for_speech, split_sentences  # noqa: PLC0415

    return split_sentences(clean_for_speech(text[:MAX_CHARS]), CHUNK_CHARS)


def _speak_chunk(plan: Plan, sentence: str, speed: float, seed: int | None) -> tuple[bytes, float]:
    """WAV bytes and worker seconds for one sentence."""
    if plan.kind == "kokoro":
        from backend_service import voice_runtime  # noqa: PLC0415

        wav, _duration = voice_runtime.synthesize(sentence, plan.voice, speed)
        return wav, 0.0
    from backend_service.voice_client import get_worker  # noqa: PLC0415

    reply = get_worker().request(
        "synthesize", timeout=WORKER_TIMEOUT_S, engine=plan.engine, source=plan.source, text=sentence,
        seed=seed, **plan.extra,
    )
    return base64.b64decode(reply["wavB64"]), float(reply.get("seconds") or 0.0)


def request_cancel() -> bool:
    """Stop the running speech job now. Returns True when a worker was interrupted or the flag was set."""
    from backend_service import voice_client  # noqa: PLC0415

    CANCEL.set()
    voice_client.interrupt_all()
    return True


def render(plan: Plan, text: str, *, speed: float = 1.0, pitch: float = 0.0, seed: int | None = None) -> Rendered:
    """Speak ``text`` with ``plan``: clean, chunk, synthesize, join, then speed and pitch."""
    sentences = chunk_text(text)
    if not sentences:
        raise ValueError("Nothing to speak after tidying the text.")
    CANCEL.clear()  # a Cancel that arrived before this job started belongs to an earlier one
    wavs: list[bytes] = []
    seconds = 0.0
    for sentence in sentences:
        if CANCEL.is_set():
            raise GenerationCancelled("Speech was cancelled.")
        try:
            wav, took = _speak_chunk(plan, sentence, speed if plan.native_speed else 1.0, seed)
        except Exception:
            if CANCEL.is_set():
                raise GenerationCancelled("Speech was cancelled.") from None
            raise
        wavs.append(wav)
        seconds += took
    joined = lift_quiet(join_wavs(wavs))
    final = process_audio(joined, 1.0 if plan.native_speed else speed, pitch)
    return Rendered(final, round(wav_info(final)[3], 2), round(seconds, 3), len(sentences), list(plan.notes))


# ---------------------------------------------------------------------------
# Entry point for ``POST /api/voice/speak``
# ---------------------------------------------------------------------------

def speak(
    text: str,
    voice: str,
    *,
    style: str = "",
    speed: float = 1.0,
    pitch: float = 0.0,
    seed: int | None = None,
    model: str = "",
    origin: str,
    permission_confirmed: bool = False,
) -> dict[str, Any]:
    """Speak and return ``{wav, durationS, seconds, chunks, voice, kind, model, modelName, clonedVoice, libraryId, notes}``.

    A cloned voice is saved to the Library (tagged and watermarked) and its ``libraryId`` is set; other
    voices are not saved here.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Speech takes up to {MAX_CHARS} characters at a time.")
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"Speed must be between {MIN_SPEED} and {MAX_SPEED}.")
    if not MIN_PITCH <= pitch <= MAX_PITCH:
        raise ValueError(f"Pitch must be between {MIN_PITCH:g} and {MAX_PITCH:g} semitones.")
    voice = (voice or "").strip()

    if voice.startswith(CLONE_PREFIX):
        from backend_service import voice_clone, voice_library  # noqa: PLC0415

        item = voice_clone.speak(
            voice[len(CLONE_PREFIX):], text, model, origin=origin, permission_confirmed=permission_confirmed,
            speed=speed, pitch=pitch, seed=seed, max_chars=MAX_CHARS,
        )
        path = voice_library.audio_path(item["id"])
        if path is None:
            raise RuntimeError("The saved speech could not be read back.")
        notes = ["Labelled as AI-made and watermarked."]
        if style.strip():
            notes.append("A cloned voice has no style control, so the style text was ignored.")
        return {
            "wav": path.read_bytes(), "durationS": item.get("durationS"), "seconds": item.get("seconds"),
            "chunks": item.get("chunks"), "voice": voice, "kind": "clone", "model": item.get("model"),
            "modelName": item.get("modelName"), "clonedVoice": True, "libraryId": item["id"], "notes": notes,
        }

    plan = build_plan(voice, style, model)
    rendered = render(plan, text, speed=speed, pitch=pitch, seed=seed)
    return {
        "wav": rendered.wav, "durationS": rendered.duration_s, "seconds": rendered.seconds, "chunks": rendered.chunks,
        "voice": voice, "kind": plan.kind, "model": plan.model_key, "modelName": plan.model_name,
        "clonedVoice": False, "libraryId": None, "notes": rendered.notes,
    }
