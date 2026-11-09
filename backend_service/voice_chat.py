"""Chat read-aloud and voice calls with any installed voice (D-67 W6, D-72).

``POST /api/voice/speak/stream`` plans through ``voice_speak`` here, then speaks one sentence at a time with
``voice_speak.render_sentence``, so each model's chunking, sound-tag rules and speed step apply. Which voices may
speak where is ``voice_speak.chat_uses``: quick and natural voices for read-aloud and calls, cloned voices for
read-aloud only (the clone rules in ``voice_clone.clone_plan``, then the watermark and AI tag on every sentence,
and an audit line; never from agents), "Describe a voice" for neither.

When the chosen voice cannot speak (its model is not downloaded, fails to load, or the voice is not allowed for this
use) the Kokoro voice reads instead and the stream carries a one-line ``note``: never silence. A voice that failed is
skipped for ``RETRY_AFTER_S`` so a call does not pay for the failed load on every reply.

Memory: there is one voice worker and it holds one speech model at a time (``voice_worker._load_tts`` frees the old
one first); the existing idle unload frees it. So a big call voice next to a loaded chat model costs one speech model,
never two.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from backend_service import voice_speak

USES = ("read", "call")
RETRY_AFTER_S = 120.0
_FAILED: dict[str, tuple[float, str]] = {}
_FAILED_LOCK = threading.Lock()

NOT_FOR_CALLS = "Cloned voices are not used in voice calls."
NEEDS_DESCRIPTION = "Describe a voice needs a written description each time, so it cannot be the chat voice."


class ChatVoiceRefused(PermissionError):
    """A cloned voice asked for by an agent or another computer (403, as on ``/api/voice/speak``)."""


@dataclass
class ChatPlan:
    """What a stream speaks with: the plan, its sentences, notes for the person, and the clone it marks (if any)."""

    plan: voice_speak.Plan
    sentences: list[str]
    voice: str
    notes: list[str] = field(default_factory=list)
    clone_row: dict[str, Any] | None = None
    clone_id: str = ""


def fallback_voice() -> str:
    from backend_service.voice_runtime import DEFAULT_VOICE  # noqa: PLC0415

    return DEFAULT_VOICE


def is_kokoro(voice: str) -> bool:
    from backend_service.voice_runtime import list_tts_voices  # noqa: PLC0415

    return voice in {v["id"] for v in list_tts_voices()}


def check_setting(voice: str, use: str, *, include_clones: bool) -> None:
    """A chat (``read``) or call (``call``) voice setting: an id ``list_voices`` returns that may speak there."""
    entry = next((v for v in voice_speak.list_voices(include_clones=include_clones) if v["id"] == voice), None)
    if entry is None:
        raise ValueError(f"Unknown voice '{voice}'.")
    if not entry["chatUse"][use]:
        raise ValueError(NOT_FOR_CALLS if entry["kind"] == "clone" else NEEDS_DESCRIPTION)


def _recent_failure(voice: str) -> str | None:
    with _FAILED_LOCK:
        failed = _FAILED.get(voice)
        if failed is None:
            return None
        if time.monotonic() - failed[0] > RETRY_AFTER_S:
            _FAILED.pop(voice, None)
            return None
        return failed[1]


def mark_failed(voice: str, reason: str) -> None:
    with _FAILED_LOCK:
        _FAILED[voice] = (time.monotonic(), reason)


def forget_failures() -> None:
    with _FAILED_LOCK:
        _FAILED.clear()


def _kokoro_plan(text: str, voice: str) -> ChatPlan:
    """Kokoro reads the sentences as before (``speech_sentences``), with sound tags such as [laugh] removed so it
    never reads the word aloud."""
    from backend_service.voice_text import speech_sentences  # noqa: PLC0415

    plan = voice_speak.Plan("kokoro", voice, "kokoro", "Kokoro", native_speed=True)
    return ChatPlan(plan, _without_tags(speech_sentences(text)), voice)


def _without_tags(sentences: list[str]) -> list[str]:
    from backend_service.voice_text import apply_tags  # noqa: PLC0415

    plain = (apply_tags(piece, frozenset(), "strip")[0] for piece in sentences)
    return [s for s in plain if s.strip(" .,!?;:")]


def _plan(text: str, voice: str, use: str) -> ChatPlan:
    if is_kokoro(voice):
        return _kokoro_plan(text, voice)
    reason = _recent_failure(voice)
    if reason is not None:
        raise RuntimeError(reason)
    if voice == voice_speak.DESIGN_VOICE:
        raise ValueError(NEEDS_DESCRIPTION)
    if voice.startswith(voice_speak.CLONE_PREFIX):
        if use != "read":
            raise ValueError(NOT_FOR_CALLS)
        from backend_service import voice_clone  # noqa: PLC0415

        clone_id = voice[len(voice_speak.CLONE_PREFIX):]
        plan, row, _record, notes = voice_clone.clone_plan(clone_id, "", origin="ui")
        sentences, more = voice_speak.prepare_text(text, plan)
        from backend_service import voice_clone_safety as safety  # noqa: PLC0415

        safety.audit("speak", origin="ui", use="read-aloud", voiceId=clone_id, model=row["key"], textLength=len(text),
                     libraryId=None)
        return ChatPlan(plan, sentences, voice, [*notes, *more], clone_row=row, clone_id=clone_id)
    plan = voice_speak.build_plan(voice)
    sentences, notes = voice_speak.prepare_text(text, plan)
    return ChatPlan(plan, sentences, voice, [*plan.notes, *notes])


def fallback_note(voice: str, reason: str) -> str:
    if voice.startswith(voice_speak.CLONE_PREFIX):
        name = "The cloned voice"
    elif voice == voice_speak.DESIGN_VOICE:
        name = "Describe a voice"
    else:
        name = {voice_speak.TURBO_VOICE: "Turbo", voice_speak.CHATTERBOX_VOICE: "Chatterbox"}.get(voice, voice.split(":")[-1].capitalize())
    return f"{name} could not speak here ({reason.rstrip('.')}), so the Kokoro voice reads instead."


def plan_stream(text: str, voice: str, *, use: str = "read", origin: str = "ui") -> ChatPlan:
    """The plan for one read-aloud or call request. Raises ``ChatVoiceRefused`` for a cloned voice from an agent;
    any other reason the voice cannot speak falls back to Kokoro with a note (a Kokoro voice that fails still
    raises: there is nothing smaller to fall back to)."""
    voice = (voice or "").strip() or fallback_voice()
    if use not in USES:
        raise ValueError(f"Use must be one of {', '.join(USES)}.")
    if voice.startswith(voice_speak.CLONE_PREFIX) and origin != "ui":
        raise ChatVoiceRefused("Cloned voices read aloud only in the ChaosEngineAI window on this computer.")
    if is_kokoro(voice):
        return _kokoro_plan(text, voice)
    try:
        return _plan(text, voice, use)
    except Exception as exc:  # noqa: BLE001 - every reason ends in the Kokoro voice, never silence
        chat = _kokoro_plan(text, fallback_voice())
        chat.notes.append(fallback_note(voice, str(exc)))
        return chat


def speak_sentence(chat: ChatPlan, sentence: str, speed: float) -> tuple[bytes, float]:
    """One sentence; a cloned voice's sentence gets the watermark and the AI tag, like every cloned clip."""
    wav, duration = voice_speak.render_sentence(chat.plan, sentence, speed=speed)
    if chat.clone_row is not None:
        from backend_service import voice_clone  # noqa: PLC0415

        wav = voice_clone.mark_cloned(wav, chat.clone_row, chat.clone_id)
    return wav, duration


def switch_to_fallback(chat: ChatPlan, remaining: list[str], reason: str) -> ChatPlan:
    """After a sentence failed (the model would not load): Kokoro reads the rest, sound tags removed."""
    mark_failed(chat.voice, reason)
    fallback = _kokoro_plan("", fallback_voice())
    fallback.sentences = _without_tags(remaining)
    fallback.notes = [fallback_note(chat.voice, reason)]
    return fallback
