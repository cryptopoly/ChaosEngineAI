"""One way to speak: pick a voice, get audio (D-58).

The caller names a voice (a Kokoro voice, a Qwen3-TTS preset speaker, ``design`` or ``clone:<id>``); this
module picks the model, cleans and chunks the text, runs the voice worker once per chunk, joins the chunks,
and applies speed and pitch. Cloned voices keep going through ``voice_clone.speak`` so the permission rules,
the AI tag, the watermark and the audit log stay in one place; it calls ``render`` here for the audio.

Voice ids:
  ``af_heart`` ...   Kokoro voices (fast, many languages, speed only)
  ``qwen3:Ryan`` ... Qwen3-TTS CustomVoice preset speakers (style instruction on 1.7B)
  ``chatterbox-turbo`` Chatterbox-Turbo's built-in voice (sound tags such as ``[laugh]`` in the text)
  ``chatterbox``     Chatterbox Multilingual V3's default voice (``language``, ``expression``, ``guidance``)
  ``orpheus:tara`` ... Orpheus 3B's eight preset voices (sound tags)
  ``design``         a voice from a description in ``style``: Qwen3-TTS VoiceDesign or VoxCPM2 (``model``)
  ``clone:<id>``     a cloned voice from the Voice Library (any cloning model, ``model``)

Sound tags: people write ``[laugh]``; Turbo reads them as they are, Orpheus gets ``<laugh>`` (written after the
text is tidied, so the HTML rule never removes them), every other model has them removed with a note.

Speed and pitch: Kokoro does speed itself; every other model has neither, so both are ffmpeg filters on the
finished audio. Pitch is a shift in semitones (rate change, then tempo restored).
"""

from __future__ import annotations

import base64
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend_service import voice_env
from backend_service.progress import GenerationCancelled
# The audio steps live in their own module; re-exported so callers and tests keep using voice_speak.<name>.
from backend_service.voice_speak_audio import (  # noqa: F401
    LIFT_PEAK,
    MAX_LIFT,
    QUIET_PEAK,
    SENTENCE_GAP_S,
    atempo_chain,
    filter_graph,
    join_wavs,
    lift_quiet,
    process_audio,
    wav_info,
)

MAX_CHARS = 20_000
CHUNK_CHARS = 400
MIN_SPEED, MAX_SPEED = 0.5, 2.0
MIN_PITCH, MAX_PITCH = -12.0, 12.0
PRESET_PREFIX = "qwen3:"
CLONE_PREFIX = "clone:"
DESIGN_VOICE = "design"
TURBO_VOICE = "chatterbox-turbo"
CHATTERBOX_VOICE = "chatterbox"
ORPHEUS_PREFIX = "orpheus:"
WORKER_TIMEOUT_S = 900.0

# Set by ``request_cancel`` (the Speak screen's Cancel); the running speech job stops at the next chunk and the
# worker is ended so a chunk in progress does not run on. ``render`` clears it when a job starts.
CANCEL = threading.Event()



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
    family: str = ""  # catalog family; "" for Kokoro
    chunk_chars: int = CHUNK_CHARS
    tag_style: str = "strip"  # sound tags: keep (Turbo), angle (Orpheus), strip (every other model)
    tags: frozenset[str] = frozenset()


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


def _rows(test: Any) -> list[dict[str, Any]]:
    """Listed catalog rows (Qwen3-TTS and measured wave-2 rows) that pass ``test``; rows that run here first (the
    CUDA row of a family before its MLX row on Windows and Linux)."""
    from backend_service import voice_clone  # noqa: PLC0415
    from backend_service.catalog.voice_models import listed_models  # noqa: PLC0415

    rows = [row for row in listed_models() if test(row)]
    return sorted(rows, key=lambda row: not voice_clone.runs_here(row))


def _models_of(kind: str) -> list[dict[str, Any]]:
    return _rows(lambda row: row["kind"] == kind)


def _family_row(family: str) -> dict[str, Any] | None:
    rows = _rows(lambda row: row["family"] == family)
    return rows[0] if rows else None


def _installed(row: dict[str, Any]) -> bool:
    from backend_service import voice_clone  # noqa: PLC0415

    return voice_clone.available_here() and voice_clone.snapshot_dir(row) is not None


def _first_installed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if _installed(row)), rows[0])


def _needs(row: dict[str, Any]) -> str:
    return f"Download {row['name']} in Voice Models."


def _family_voices() -> list[dict[str, Any]]:
    """Chatterbox-Turbo's built-in voice, Chatterbox V3's default voice and the Orpheus presets (Natural group)."""
    from backend_service.catalog.voice_models import ORPHEUS_VOICES  # noqa: PLC0415

    voices: list[dict[str, Any]] = []
    turbo = _family_row("chatterbox-turbo")
    if turbo is not None:
        ready = _installed(turbo)
        voices.append({
            "id": TURBO_VOICE, "name": "Turbo", "kind": "preset", "family": "chatterbox-turbo", "language": "en",
            "languageLabel": "English", "gender": "male", "about": "Chatterbox-Turbo's own voice; reads sound tags",
            "model": turbo["key"], "models": [turbo["key"]], "capabilities": _caps(speak=True),
            "available": ready, "needs": None if ready else _needs(turbo),
        })
    mtl = _family_row("chatterbox")
    if mtl is not None and mtl["capabilities"]["speak"]:
        ready = _installed(mtl)
        voices.append({
            "id": CHATTERBOX_VOICE, "name": "Chatterbox", "kind": "preset", "family": "chatterbox", "language": "auto",
            "languageLabel": "23 languages", "gender": "female", "about": "Chatterbox's default voice in 23 languages",
            "model": mtl["key"], "models": [mtl["key"]], "capabilities": _caps(speak=True),
            "available": ready, "needs": None if ready else _needs(mtl),
        })
    orpheus = _family_row("orpheus")
    if orpheus is not None:
        ready = _installed(orpheus)
        for voice in ORPHEUS_VOICES:
            voices.append({
                "id": ORPHEUS_PREFIX + voice["id"], "name": voice["id"].capitalize(), "kind": "preset",
                "family": "orpheus", "language": "en", "languageLabel": "English", "gender": voice["gender"],
                "about": voice["about"], "model": orpheus["key"], "models": [orpheus["key"]],
                "capabilities": _caps(speak=True), "available": ready, "needs": None if ready else _needs(orpheus),
            })
    return voices


def list_voices(*, include_clones: bool = True) -> list[dict[str, Any]]:
    """Built-in, preset, designed and cloned voices in one list.

    Fields: ``id``, ``name``, ``kind`` (builtin / preset / design / clone), ``group`` (quick / natural / designed /
    cloned, the picker's groups), ``family`` (the model family that speaks it by default), ``language``,
    ``languageLabel``, ``gender`` (when known), ``model`` (the catalog key used by default, ``kokoro`` for
    built-ins), ``models`` (every key that can speak it), ``capabilities`` (speak / style / clone / design),
    ``available`` (the voice can be used right now) and ``needs`` (plain words when it cannot). Built-in voices also
    carry ``grade`` (the model author's quality grade, None when ungraded) and ``addon`` (the add-on that unlocks
    them, or None); voices that can never run on this lane (Japanese on MLX) are left out.

    ``needsFfmpeg`` is True on every voice when ffmpeg is missing: pitch is always an ffmpeg step, and so is speed
    for every voice except the built-in ones, so the Speak screen can say so before Generate rather than after.
    """
    from backend_service import voice_runtime  # noqa: PLC0415
    from backend_service.catalog.voice_models import CUSTOM_SPEAKERS  # noqa: PLC0415
    from backend_service.storyboard import ffmpeg_bin  # noqa: PLC0415

    no_ffmpeg = ffmpeg_bin.optional_ffmpeg_exe() is None

    voices: list[dict[str, Any]] = []
    for v in voice_runtime.list_tts_voices():
        if v.get("neverAvailable"):
            continue  # Japanese on the MLX lane: nothing can unlock it, so it is not offered at all
        voices.append({
            "id": v["id"], "name": v["name"], "kind": "builtin", "family": "kokoro", "language": v["language"],
            "languageLabel": v["languageLabel"], "gender": v["gender"], "model": "kokoro", "models": ["kokoro"],
            "capabilities": _caps(speak=True), "available": bool(v["available"]),
            "grade": v.get("grade"), "addon": None if v["available"] else v.get("addon"),
            "needs": None if v["available"] else f"{v['languageLabel']} voices need an extra install.",
        })

    customs = _models_of("custom")
    custom_ready = [row for row in customs if _installed(row)]
    for speaker in CUSTOM_SPEAKERS:
        voices.append({
            "id": PRESET_PREFIX + speaker["id"], "name": speaker["id"].replace("_", " "), "kind": "preset",
            "family": "qwen3-tts", "language": speaker["language"], "languageLabel": speaker["languageLabel"],
            "gender": speaker["gender"], "about": speaker["about"],
            "model": (custom_ready and _pick_custom(custom_ready) or _pick_custom(customs))["key"], "models": [r["key"] for r in customs],
            "capabilities": _caps(speak=True, style=True), "available": bool(custom_ready),
            "needs": None if custom_ready else f"Download {customs[0]['name']} in Voice Models.",
        })
    voices.extend(_family_voices())

    designers = _design_rows()
    design = _first_installed(designers)
    design_ready = _installed(design)
    voices.append({
        "id": DESIGN_VOICE, "name": "Describe a voice", "kind": "design", "family": design["family"],
        "language": "auto", "languageLabel": "Any", "gender": None, "model": design["key"],
        "models": [r["key"] for r in designers], "capabilities": _caps(speak=True, style=True, design=True),
        "available": design_ready, "needs": None if design_ready else _needs(design),
    })

    if include_clones:
        from backend_service import voice_clone  # noqa: PLC0415

        cloners = sorted(_rows(lambda row: row["capabilities"]["clone"]), key=lambda row: not row["default"])
        base = voice_clone.pick_clone_model("")  # the default, or any downloaded cloning model
        base_ready = _installed(base) and voice_env.watermark_installed()
        for item in voice_clone.list_voices():
            voices.append({
                "id": CLONE_PREFIX + item["id"], "name": item["name"], "kind": "clone", "family": base["family"],
                "language": "auto", "languageLabel": "Any", "gender": None, "model": base["key"],
                "models": [r["key"] for r in cloners], "capabilities": _caps(speak=True, clone=True),
                "available": base_ready, "needs": None if base_ready else _needs(base),
                "consent": item["consent"], "durationS": item.get("durationS"),
            })
    for entry in voices:
        entry["group"] = _GROUPS[entry["kind"]]
        entry["needsFfmpeg"] = no_ffmpeg
        entry["chatUse"] = chat_uses(entry["kind"])
        entry["chatCost"] = chat_cost(entry["model"])
    return voices


def chat_uses(kind: str) -> dict[str, bool]:
    """Where a voice may speak in Chat (D-72): read-aloud and calls for quick and natural voices; read-aloud only for
    cloned ones (the clone rules still apply per request); neither for "Describe a voice", which needs a description
    each time (saved designed voices are W8)."""
    return {"read": kind in ("builtin", "preset", "clone"), "call": kind in ("builtin", "preset")}


def chat_cost(model_key: str) -> dict[str, Any] | None:
    """Measured first-audio latency and memory for Chat, from the catalog row (or Kokoro's constant); None when the
    row has no chat measurement."""
    from backend_service.catalog.voice_models import KOKORO_CHAT_ESTIMATES, SLOW_FOR_CALLS_S, voice_model  # noqa: PLC0415

    if model_key == "kokoro":
        est: dict[str, Any] = KOKORO_CHAT_ESTIMATES
    else:
        try:
            est = voice_model(model_key).get("estimates") or {}
        except (KeyError, ValueError):
            return None
    if est.get("chatFirstAudioS") is None:
        return None
    return {
        "firstAudioS": est["chatFirstAudioS"], "firstAudioColdS": est.get("chatFirstAudioColdS"),
        "peakGb": est.get("chatPeakGb"), "measuredOn": est.get("chatMeasuredOn"),
        "slowForCalls": float(est["chatFirstAudioS"]) > SLOW_FOR_CALLS_S,
    }


_GROUPS = {"builtin": "quick", "preset": "natural", "design": "designed", "clone": "cloned"}


def _design_rows() -> list[dict[str, Any]]:
    """Rows that make a voice from a description: Qwen3-TTS VoiceDesign first, then VoxCPM2."""
    return _rows(lambda row: row["capabilities"]["design"])


def _pick_custom(rows: list[dict[str, Any]], installed_only: bool = False) -> dict[str, Any]:
    """The style-capable model first, whether or not the style box has text, so typing a style never swaps the model
    (a reload). The 0.6B model is no quicker (FU-148); it runs only when it is the only one installed."""
    ordered = sorted(rows, key=lambda r: (not r["capabilities"]["style"], r["sizeGb"]))
    if installed_only:
        ordered = [r for r in ordered if _installed(r)]
    if not ordered:
        raise LookupError("none")
    return ordered[0]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def worker_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Fields every worker request for ``row`` carries: the family (which call runs) and the pinned local folders
    of its loader companions. Empty for Qwen3-TTS, whose requests are unchanged."""
    from backend_service import voice_clone  # noqa: PLC0415

    if row["family"] == "qwen3-tts":
        return {}
    fields: dict[str, Any] = {"family": row["family"], "companions": voice_clone.companion_paths(row)}
    if row.get("maxTokens"):
        fields["maxTokens"] = int(row["maxTokens"])
    if row["engine"] == "torch":
        fields["device"] = "cuda"
    elif row["engine"] == "llama-gguf":
        # Orpheus as a GGUF: the bundled llama-server makes the audio codes, the SNAC ONNX decoder makes the sound.
        from backend_service.inference.binaries import _resolve_llama_server  # noqa: PLC0415

        folder = voice_clone.snapshot_dir(row)
        decoder = voice_clone.companion_file(row, "snacDecoder")
        fields.update({
            "gguf": str(folder / row["files"][0]) if folder else "",
            "snacDecoderPath": str(decoder) if decoder else "",
            "llamaServerPath": _resolve_llama_server() or "",
        })
    return fields


def _tag_rules(row: dict[str, Any] | None) -> tuple[str, frozenset[str]]:
    """``(style, allowed)`` for ``voice_text.apply_tags``: Turbo keeps its tags, Orpheus gets angle brackets,
    every other model has them removed."""
    if row is None or not row["capabilities"].get("tags"):
        return "strip", frozenset()
    controls = row.get("controls") or {}
    allowed = frozenset([*controls.get("tags", []), *controls.get("styleTags", [])])
    return ("angle" if row["family"] == "orpheus" else "keep"), allowed


def plan_for(kind: str, voice: str, row: dict[str, Any], extra: dict[str, Any]) -> Plan:
    """A plan for a catalog row: its worker fields, chunk size and sound-tag rules. ``voice_clone.speak`` uses this
    too, so a cloned voice follows the same text rules as every other voice."""
    from backend_service import voice_clone  # noqa: PLC0415

    folder = voice_clone.snapshot_dir(row)
    style, allowed = _tag_rules(row)
    return Plan(
        kind, voice, row["key"], row["name"], row["engine"], str(folder) if folder else "",
        {**worker_fields(row), **extra}, family=row["family"], chunk_chars=int(row.get("chunkChars") or CHUNK_CHARS),
        tag_style=style, tags=allowed,
    )


def _model_for(rows: list[dict[str, Any]], model_key: str, *, pick: Any = None) -> dict[str, Any]:
    """The row to run among ``rows``: the one named (it must be one of them), else ``pick`` (default: the first
    installed, else the first). Raises with plain words when it cannot run here."""
    from backend_service.catalog.voice_models import voice_model  # noqa: PLC0415
    from backend_service import voice_clone  # noqa: PLC0415

    if model_key:
        row = voice_model(model_key)
        if row["key"] not in {r["key"] for r in rows}:
            raise ValueError(f"{row['name']} cannot speak this voice.")
        chosen = row
    elif pick is not None:
        chosen = pick(rows)
    else:
        chosen = _first_installed(rows)
    if not voice_clone.available_here():
        raise ValueError(
            "Preset and designed voices need an Apple Silicon Mac, or the speech GPU add-on with an NVIDIA GPU. "
            "Use a built-in voice here."
        )
    if not voice_env.is_installed():
        raise RuntimeError("The voice pack is not installed. Install it from Voice > Models.")
    if voice_clone.snapshot_dir(chosen) is None:
        raise RuntimeError(f"{chosen['name']} is not downloaded. Download it in Voice > Models.")
    return chosen


def _pick_custom_installed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        return _pick_custom(rows, installed_only=True)
    except LookupError:
        return _pick_custom(rows)


def check_controls(language: str, expression: float, guidance: float) -> None:
    """The optional speak fields, validated the same for every voice (old requests send none of them)."""
    if not re.fullmatch(r"auto|[a-z]{2,3}(-[A-Za-z]{2,4})?", language or ""):
        raise ValueError("Language must be 'auto' or a language code such as 'en' or 'de'.")
    for name, value in (("Expression", expression), ("Guidance", guidance)):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")


def chatterbox_language(language: str) -> str:
    """Chatterbox V3's ``lang_code``: one of its 23; ``auto`` means English."""
    from backend_service.catalog.voice_models import CHATTERBOX_LANGUAGES  # noqa: PLC0415

    code = (language or "auto").split("-")[0].lower()
    if code == "auto":
        return "en"
    if code not in CHATTERBOX_LANGUAGES:
        raise ValueError(f"Chatterbox speaks {', '.join(CHATTERBOX_LANGUAGES)}; '{language}' is not one of them.")
    return code


def chatterbox_fields(row: dict[str, Any], language: str, expression: float, guidance: float) -> dict[str, Any]:
    """Chatterbox V3's worker fields; expression is always sent (mlx-audio's own default is 0.1, the card's 0.5)."""
    from backend_service import voice_clone  # noqa: PLC0415

    fields: dict[str, Any] = {
        "language": chatterbox_language(language), "expression": float(expression), "guidance": float(guidance),
    }
    voice_file = voice_clone.default_voice_path(row)
    if voice_file is not None:
        fields["defaultVoicePath"] = str(voice_file)
    return fields


def build_plan(
    voice: str, style: str = "", model_key: str = "", *,
    language: str = "auto", expression: float = 0.5, guidance: float = 0.5,
) -> Plan:
    """The model and worker fields for a non-cloned voice. Raises ValueError / RuntimeError with plain words.

    Never sets a reference clip: the only path that sends one to the worker is ``voice_clone.speak``.
    """
    from backend_service.catalog.voice_models import CUSTOM_SPEAKERS, ORPHEUS_VOICES, QWEN_LANGUAGES  # noqa: PLC0415

    voice = (voice or "").strip()
    style = " ".join((style or "").split())
    check_controls(language, expression, guidance)
    if voice.startswith(CLONE_PREFIX):
        raise ValueError("Cloned voices go through speak().")
    if voice.startswith(PRESET_PREFIX):
        wanted = voice[len(PRESET_PREFIX):].lower()
        speaker = next((s for s in CUSTOM_SPEAKERS if s["id"].lower() == wanted), None)
        if speaker is None:
            raise ValueError(f"Unknown voice '{voice}'.")
        row = _model_for(_models_of("custom"), model_key, pick=_pick_custom_installed)
        plan = plan_for("preset", voice, row, {
            "mode": "preset", "speaker": speaker["id"], "language": QWEN_LANGUAGES.get(language.split("-")[0], "auto"),
        })
        if style:
            if row["capabilities"]["style"]:
                plan.extra["instruct"] = style
            else:
                plan.notes.append("The 0.6B model ignores the style instruction. Download the 1.7B CustomVoice model to use it.")
        return plan
    if voice.startswith(ORPHEUS_PREFIX):
        wanted = voice[len(ORPHEUS_PREFIX):].lower()
        if wanted not in {v["id"] for v in ORPHEUS_VOICES} or _family_row("orpheus") is None:
            raise ValueError(f"Unknown voice '{voice}'.")
        row = _model_for(_rows(lambda r: r["family"] == "orpheus"), model_key)
        plan = plan_for("preset", voice, row, {"speaker": wanted})
        if style:
            plan.notes.append("Orpheus takes sound tags in the text, such as [laugh], not a style.")
        return plan
    if voice == TURBO_VOICE and _family_row("chatterbox-turbo") is not None:
        row = _model_for(_rows(lambda r: r["family"] == "chatterbox-turbo"), model_key)
        plan = plan_for("preset", voice, row, {})
        if style:
            plan.notes.append("Turbo takes sound tags in the text, such as [laugh], not a style.")
        return plan
    if voice == CHATTERBOX_VOICE and (_family_row("chatterbox") or {}).get("capabilities", {}).get("speak"):
        row = _model_for(_rows(lambda r: r["family"] == "chatterbox"), model_key)
        plan = plan_for("preset", voice, row, chatterbox_fields(row, language, expression, guidance))
        if style:
            plan.notes.append("Chatterbox has no style box; use the Expression slider.")
        return plan
    if voice == DESIGN_VOICE:
        if not style:
            raise ValueError("Describe the voice you want (age, gender, tone, pace) in the style box.")
        row = _model_for(_design_rows(), model_key)
        if row["family"] == "voxcpm2":
            return plan_for("design", voice, row, {"mode": "design", "instruct": style})
        return plan_for("design", voice, row, {"mode": "design", "instruct": style, "language": "auto"})
    from backend_service import voice_runtime  # noqa: PLC0415

    known = {v["id"]: v for v in voice_runtime.list_tts_voices()}
    if voice not in known:
        raise ValueError(f"Unknown voice '{voice}'.")
    plan = Plan("kokoro", voice, "kokoro", "Kokoro", native_speed=True)
    if style:
        plan.notes.append("Kokoro has no style control, so the style text was ignored.")
    return plan


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def chunk_text(text: str, plan: Plan | None = None) -> list[str]:
    """Clean markdown and code out of ``text`` and cut it into sentences (``prepare_text`` without the notes)."""
    return prepare_text(text, plan)[0]


TAGS_SKIPPED = "Sound tags such as [laugh] work with Chatterbox-Turbo and Orpheus; this voice skipped them."


def prepare_text(text: str, plan: Plan | None = None) -> tuple[list[str], list[str]]:
    """Sentences to send, one per worker request, and notes for the person.

    Clean markdown and code, cut into sentences of at most the plan's chunk size, keep a sound tag with its
    sentence, then rewrite the tags for the model (``voice_text.apply_tags``). VoxCPM2 reads a leading "(" as a
    voice description, so that bracket is dropped from every chunk (``clean_for_speech`` itself writes
    "(code block omitted)").
    """
    from backend_service.voice_text import (  # noqa: PLC0415
        apply_tags,
        clean_for_speech,
        guard_leading_paren,
        keep_tags_with_sentences,
        split_sentences,
    )

    size = plan.chunk_chars if plan is not None else CHUNK_CHARS
    pieces = keep_tags_with_sentences(split_sentences(clean_for_speech(text[:MAX_CHARS]), size))
    style, allowed = (plan.tag_style, plan.tags) if plan is not None else ("strip", frozenset())
    sentences: list[str] = []
    skipped = False
    for piece in pieces:
        piece, stripped = apply_tags(piece, allowed, style)
        skipped = skipped or stripped
        if plan is not None and plan.family == "voxcpm2":
            piece = guard_leading_paren(piece)
        if piece.strip(" .,!?;:"):
            sentences.append(piece)
    return sentences, [TAGS_SKIPPED] if skipped else []


def _speak_chunk(plan: Plan, sentence: str, speed: float, seed: int | None,
                 extra: dict[str, Any] | None = None) -> tuple[bytes, float]:
    """WAV bytes and worker seconds for one sentence."""
    if plan.kind == "kokoro":
        from backend_service import voice_runtime  # noqa: PLC0415

        wav, _duration = voice_runtime.synthesize(sentence, plan.voice, speed)
        return wav, 0.0
    from backend_service.voice_client import get_worker  # noqa: PLC0415

    reply = get_worker().request(
        "synthesize", timeout=WORKER_TIMEOUT_S, engine=plan.engine, source=plan.source, text=sentence,
        seed=seed, **(plan.extra if extra is None else extra),
    )
    return base64.b64decode(reply["wavB64"]), float(reply.get("seconds") or 0.0)


def _continues(plan: Plan) -> bool:
    """A VoxCPM2 designed voice holds across chunks by continuing from its own first chunk (D-70)."""
    return plan.family == "voxcpm2" and plan.kind == "design"


def request_cancel() -> bool:
    """Stop the running speech job now. Returns True when a worker was interrupted or the flag was set."""
    from backend_service import voice_client  # noqa: PLC0415

    CANCEL.set()
    voice_client.interrupt_all()
    return True


def render(plan: Plan, text: str, *, speed: float = 1.0, pitch: float = 0.0, seed: int | None = None) -> Rendered:
    """Speak ``text`` with ``plan``: clean, chunk, synthesize, join, then speed and pitch."""
    sentences, notes = prepare_text(text, plan)
    if not sentences:
        raise ValueError("Nothing to speak after tidying the text.")
    CANCEL.clear()  # a Cancel that arrived before this job started belongs to an earlier one
    wavs: list[bytes] = []
    seconds = 0.0
    with tempfile.TemporaryDirectory(prefix="chaosengine-speak-") as folder:
        extra: dict[str, Any] | None = None
        for index, sentence in enumerate(sentences):
            if CANCEL.is_set():
                raise GenerationCancelled("Speech was cancelled.")
            try:
                chunk_speed = speed if plan.native_speed else 1.0
                wav, took = (_speak_chunk(plan, sentence, chunk_speed, seed) if extra is None
                             else _speak_chunk(plan, sentence, chunk_speed, seed, extra))
            except Exception:
                if CANCEL.is_set():
                    raise GenerationCancelled("Speech was cancelled.") from None
                raise
            if index == 0 and _continues(plan) and len(sentences) > 1:
                # The model's own first chunk (never a person's clip) carries the designed voice into the rest.
                first = Path(folder) / "first.wav"
                first.write_bytes(wav)
                extra = {k: v for k, v in plan.extra.items() if k not in ("instruct", "mode")}
                extra.update({"mode": "continue", "continueAudioPath": str(first), "continueText": sentence})
            wavs.append(wav)
            seconds += took
    joined = lift_quiet(join_wavs(wavs))
    final = process_audio(joined, 1.0 if plan.native_speed else speed, pitch)
    return Rendered(final, round(wav_info(final)[3], 2), round(seconds, 3), len(sentences), [*plan.notes, *notes])


def render_sentence(plan: Plan, sentence: str, *, speed: float = 1.0) -> tuple[bytes, float]:
    """One sentence already cut by ``prepare_text`` (Chat read-aloud and calls): WAV bytes and audio seconds.

    Kokoro speaks through ``voice_runtime.synthesize`` as before (it applies speed itself); every other model gets the
    same quiet-lift and ffmpeg speed step as ``render``.
    """
    if plan.kind == "kokoro":
        from backend_service import voice_runtime  # noqa: PLC0415

        return voice_runtime.synthesize(sentence, plan.voice, speed)
    wav, _took = _speak_chunk(plan, sentence, 1.0, None)
    final = process_audio(lift_quiet(wav), speed, 0.0)
    return final, round(wav_info(final)[3], 2)


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
    language: str = "auto",
    expression: float = 0.5,
    guidance: float = 0.5,
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
    check_controls(language, expression, guidance)

    if voice.startswith(CLONE_PREFIX):
        from backend_service import voice_clone, voice_library  # noqa: PLC0415

        item = voice_clone.speak(
            voice[len(CLONE_PREFIX):], text, model, origin=origin, permission_confirmed=permission_confirmed,
            speed=speed, pitch=pitch, seed=seed, max_chars=MAX_CHARS, style=style, language=language,
            expression=expression, guidance=guidance,
        )
        path = voice_library.audio_path(item["id"])
        if path is None:
            raise RuntimeError("The saved speech could not be read back.")
        notes = ["Labelled as AI-made and watermarked.", *item.get("notes", [])]
        return {
            "wav": path.read_bytes(), "durationS": item.get("durationS"), "seconds": item.get("seconds"),
            "chunks": item.get("chunks"), "voice": voice, "kind": "clone", "model": item.get("model"),
            "modelName": item.get("modelName"), "clonedVoice": True, "libraryId": item["id"], "notes": notes,
        }

    plan = build_plan(voice, style, model, language=language, expression=expression, guidance=guidance)
    rendered = render(plan, text, speed=speed, pitch=pitch, seed=seed)
    return {
        "wav": rendered.wav, "durationS": rendered.duration_s, "seconds": rendered.seconds, "chunks": rendered.chunks,
        "voice": voice, "kind": plan.kind, "model": plan.model_key, "modelName": plan.model_name,
        "clonedVoice": False, "libraryId": None, "notes": rendered.notes,
    }
