"""Chat read-aloud and voice calls with any installed voice (D-67 W6, D-72), and the Fallback voice (D-75).

``POST /api/voice/speak/stream`` plans through ``voice_speak`` here, then speaks one sentence at a time with
``voice_speak.render_sentence``, so each model's chunking, sound-tag rules and speed step apply. Which voices may
speak where is ``voice_speak.chat_uses``: quick and natural voices for read-aloud and calls, cloned voices for
read-aloud only (the clone rules in ``voice_clone.clone_plan``, then the watermark and AI tag on every sentence,
and an audit line; never from agents), "Describe a voice" for neither.

When the chosen voice cannot speak (its model is not downloaded, fails to load, Kokoro is not installed, or the voice
is not allowed for this use) the **fallback voice** (the ``fallbackVoice`` setting) reads instead and the stream
carries a one-line ``note``. The fallback must be installed and allowed for the use (a cloned fallback reads aloud
only, in this window); the default is Kokoro's af_heart, which counts only while Kokoro is installed. With no usable
fallback the request ends with ``NO_FALLBACK``, a plain error, instead of switching voice. A voice that failed is
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
NO_FALLBACK = "The chosen voice could not speak and no fallback voice is set. Pick one on Voice Studio > Voices."
FALLBACK_NOT_INSTALLED = "The fallback voice must be installed. Download its model, or pick another voice."


class ChatVoiceRefused(PermissionError):
    """A cloned voice asked for by an agent or another computer (403, as on ``/api/voice/speak``)."""


class NoFallbackVoice(RuntimeError):
    """The chosen voice cannot speak and no usable fallback voice is set; the message is ``NO_FALLBACK``."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(NO_FALLBACK)
        self.reason = reason


@dataclass
class ChatPlan:
    """What a stream speaks with: the plan, its sentences, notes for the person, and the clone it marks (if any)."""

    plan: voice_speak.Plan
    sentences: list[str]
    voice: str
    notes: list[str] = field(default_factory=list)
    clone_row: dict[str, Any] | None = None
    clone_id: str = ""
    # The fallback voice is speaking: nothing further to switch to if it fails too.
    fallback: bool = False
    use: str = "read"
    origin: str = "ui"


def _entry(voice: str, *, include_clones: bool) -> dict[str, Any] | None:
    return next((v for v in voice_speak.list_voices(include_clones=include_clones) if v["id"] == voice), None)


def is_kokoro(voice: str) -> bool:
    from backend_service.voice_runtime import list_tts_voices  # noqa: PLC0415

    return voice in {v["id"] for v in list_tts_voices()}


def check_setting(voice: str, use: str, *, include_clones: bool) -> None:
    """A chat (``read``) or call (``call``) voice setting: an id ``list_voices`` returns that may speak there."""
    entry = _entry(voice, include_clones=include_clones)
    if entry is None:
        raise ValueError(f"Unknown voice '{voice}'.")
    if not entry["chatUse"][use]:
        raise ValueError(NOT_FOR_CALLS if entry["kind"] == "clone" else NEEDS_DESCRIPTION)


def check_fallback_setting(voice: str, *, include_clones: bool) -> None:
    """The ``fallbackVoice`` setting: "" (none), or an installed voice that may read aloud. A cloned fallback is
    allowed; it is never used for calls (``fallback_for``)."""
    if voice == "":
        return
    check_setting(voice, "read", include_clones=include_clones)
    entry = _entry(voice, include_clones=include_clones)
    if entry is not None and not entry["available"]:
        raise ValueError(FALLBACK_NOT_INSTALLED)


def fallback_for(use: str, *, origin: str = "ui", exclude: str = "") -> str | None:
    """The fallback voice for ``use`` right now, or None: the setting, when it is installed, allowed for ``use`` (a
    cloned one only for read-aloud in this window), not failed in the last ``RETRY_AFTER_S`` and not the voice that
    just failed."""
    voice = _usable_setting(use, origin)
    if not voice or voice == exclude or _recent_failure(voice) is not None:
        return None
    return voice


def _usable_setting(use: str, origin: str) -> str:
    """The ``fallbackVoice`` setting when it can speak for ``use`` from ``origin``; "" otherwise."""
    from backend_service.voice_runtime import load_settings  # noqa: PLC0415

    voice = str(load_settings().get("fallbackVoice") or "").strip()
    if not voice:
        return ""
    is_clone = voice.startswith(voice_speak.CLONE_PREFIX)
    if is_clone and (use != "read" or origin != "ui"):
        return ""
    entry = _entry(voice, include_clones=is_clone)
    if entry is None or not entry["available"] or not entry["chatUse"].get(use):
        return ""
    return voice


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
    never reads the word aloud. Raises when the Kokoro quick voices are not installed (D-75)."""
    from backend_service import voice_env  # noqa: PLC0415
    from backend_service.voice_runtime import KOKORO_MISSING  # noqa: PLC0415
    from backend_service.voice_text import speech_sentences  # noqa: PLC0415

    if not voice_env.kokoro_installed():
        raise RuntimeError(KOKORO_MISSING)
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


def voice_name(voice: str) -> str:
    """A short display name for a voice id in a note."""
    if voice.startswith(voice_speak.CLONE_PREFIX):
        return "The cloned voice"
    if voice == voice_speak.DESIGN_VOICE:
        return "Describe a voice"
    named = {voice_speak.TURBO_VOICE: "Turbo", voice_speak.CHATTERBOX_VOICE: "Chatterbox"}
    if voice in named:
        return named[voice]
    if is_kokoro(voice):
        return f"{voice.split('_', 1)[-1].replace('_', ' ').title()} (Kokoro)"
    return voice.split(":")[-1].capitalize()


def fallback_note(voice: str, reason: str, fallback: str) -> str:
    return f"{voice_name(voice)} could not speak here ({reason.rstrip('.')}), so the fallback voice {voice_name(fallback)} reads instead."


def _fallback_plan(text: str, sentences: list[str] | None, use: str, origin: str, failed: str, reason: str) -> ChatPlan:
    """The fallback voice's plan for ``text`` (or the given remaining ``sentences``); raises ``NoFallbackVoice``."""
    fallback = fallback_for(use, origin=origin, exclude=failed)
    if fallback is None:
        if failed and failed == _usable_setting(use, origin):
            raise RuntimeError(reason)  # the fallback voice itself failed: its own reason, not "none is set"
        raise NoFallbackVoice(reason)
    chat = _plan(text if sentences is None else " ".join(sentences), fallback, use)
    if sentences is not None and chat.plan.kind == "kokoro":
        chat.sentences = _without_tags(sentences)
    chat.notes = [*chat.notes, fallback_note(failed, reason, fallback)]
    chat.fallback, chat.use, chat.origin = True, use, origin
    return chat


def plan_stream(text: str, voice: str, *, use: str = "read", origin: str = "ui") -> ChatPlan:
    """The plan for one read-aloud or call request. Raises ``ChatVoiceRefused`` for a cloned voice from an agent and
    ``NoFallbackVoice`` when the voice cannot speak and no usable fallback is set; any other reason the voice cannot
    speak hands over to the fallback voice with a note."""
    if use not in USES:
        raise ValueError(f"Use must be one of {', '.join(USES)}.")
    voice = (voice or "").strip()
    if voice.startswith(voice_speak.CLONE_PREFIX) and origin != "ui":
        raise ChatVoiceRefused("Cloned voices read aloud only in the ChaosEngineAI window on this computer.")
    if not voice:
        fallback = fallback_for(use, origin=origin)
        if fallback is None:
            raise NoFallbackVoice("no voice was chosen")
        voice = fallback
    try:
        chat = _plan(text, voice, use)
    except Exception as exc:  # noqa: BLE001 - every reason ends in the fallback voice or a plain error, never silence
        return _fallback_plan(text, None, use, origin, voice, str(exc))
    chat.use, chat.origin = use, origin
    return chat


def speak_sentence(chat: ChatPlan, sentence: str, speed: float) -> tuple[bytes, float]:
    """One sentence; a cloned voice's sentence gets the watermark and the AI tag, like every cloned clip."""
    wav, duration = voice_speak.render_sentence(chat.plan, sentence, speed=speed)
    if chat.clone_row is not None:
        from backend_service import voice_clone  # noqa: PLC0415

        wav = voice_clone.mark_cloned(wav, chat.clone_row, chat.clone_id)
    return wav, duration


def switch_to_fallback(chat: ChatPlan, remaining: list[str], reason: str) -> ChatPlan:
    """After a sentence failed (the model would not load): the fallback voice reads the rest. Raises
    ``NoFallbackVoice`` when there is none, and re-raises the reason when the fallback itself was speaking."""
    if chat.fallback:
        raise RuntimeError(reason)
    mark_failed(chat.voice, reason)
    return _fallback_plan("", remaining, chat.use, chat.origin, chat.voice, reason)
