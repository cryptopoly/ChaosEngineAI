"""Wave-2 speech models inside the voice worker: VoxCPM2, Chatterbox Multilingual V3, Chatterbox-Turbo, Orpheus 3B.

Runs in the voice venv next to ``voice_worker.py`` and imports nothing from ``backend_service`` (FU-149).

Two jobs:

* **Companion redirect.** Three mlx-audio loaders fetch extra repos by name, with no revision, while loading:
  both Chatterbox loaders fetch ``mlx-community/S3TokenizerV2``, and the Orpheus loader fetches
  ``mlx-community/snac_24khz`` at import and its tokenizer from ``mlx-community/orpheus-3b-0.1-ft-bf16``. The
  backend downloads those at pinned revisions with the model and sends their local folders in the request
  (``companions``: repo id -> folder). ``install_redirect`` points ``huggingface_hub.snapshot_download``,
  ``hf_hub_download`` and transformers' ``AutoTokenizer.from_pretrained`` at those folders for these ids only; a
  known companion that is not in the map fails with a plain error instead of downloading. Every other id passes
  through untouched (Kokoro, Qwen3-TTS).
* **One call per family**, with the arguments the plan fixes (docs/plans/voice-models-wave2.md section 3.1).

Cloning: a reference clip arrives only as ``refAudioPath``, which the backend sets on one path
(``voice_clone.speak``, which also watermarks and tags the result). ``continueAudioPath`` is VoxCPM2's own earlier
output for a designed voice, never a person's clip. Chatterbox-Turbo stores the clip's conditionals on the model
when it clones; they are put back afterwards so the next plain request speaks with the built-in voice again,
never the cloned one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

FAMILIES = ("chatterbox-turbo", "chatterbox", "voxcpm2", "orpheus")

# Repos the loaders fetch by name. Only these are redirected; one missing from the request's map is refused.
COMPANION_IDS = frozenset({
    "mlx-community/S3TokenizerV2",
    "mlx-community/snac_24khz",
    "mlx-community/orpheus-3b-0.1-ft-bf16",
})

_redirect: dict[str, str] = {}
_patched: dict[str, Any] = {}


class CompanionMissing(RuntimeError):
    """A loader asked for a companion repo the backend did not send."""


def resolve(repo_id: Any) -> str | None:
    """The local folder for a known companion repo, None for any other id; raises for a known one not sent."""
    name = str(repo_id or "")
    if name in _redirect:
        return _redirect[name]
    if name in COMPANION_IDS:
        raise CompanionMissing(f"{name} is not downloaded. Download the speech model again in Voice Models.")
    return None


def _wrap_snapshot(original: Callable[..., Any]) -> Callable[..., Any]:
    def snapshot_download(*args: Any, **kwargs: Any) -> Any:
        local = resolve(kwargs.get("repo_id", args[0] if args else None))
        return local if local is not None else original(*args, **kwargs)

    snapshot_download.__wrapped__ = original  # type: ignore[attr-defined]
    return snapshot_download


def _wrap_file(original: Callable[..., Any]) -> Callable[..., Any]:
    def hf_hub_download(*args: Any, **kwargs: Any) -> Any:
        local = resolve(kwargs.get("repo_id", args[0] if args else None))
        if local is None:
            return original(*args, **kwargs)
        filename = kwargs.get("filename", args[1] if len(args) > 1 else None)
        path = Path(local) / str(filename)
        if not path.is_file():
            raise CompanionMissing(f"{path.name} is missing from {local}. Download the speech model again.")
        return str(path)

    hf_hub_download.__wrapped__ = original  # type: ignore[attr-defined]
    return hf_hub_download


def install_redirect(companions: dict[str, str]) -> None:
    """Point the loaders at the pinned local companion folders. Safe to call before every request."""
    _redirect.clear()
    _redirect.update({str(k): str(v) for k, v in (companions or {}).items() if str(k) in COMPANION_IDS})
    if _patched:
        return
    import huggingface_hub  # type: ignore[import-untyped]

    originals = {"snapshot_download": huggingface_hub.snapshot_download, "hf_hub_download": huggingface_hub.hf_hub_download}
    wrapped = {
        "snapshot_download": _wrap_snapshot(originals["snapshot_download"]),
        "hf_hub_download": _wrap_file(originals["hf_hub_download"]),
    }
    for name, function in wrapped.items():
        setattr(huggingface_hub, name, function)
    # Modules imported earlier bound the original functions by name (``from huggingface_hub import ...``).
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("mlx_audio"):
            continue
        for name, function in wrapped.items():
            if getattr(module, name, None) is originals[name]:
                setattr(module, name, function)
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        original_tokenizer = AutoTokenizer.from_pretrained

        def from_pretrained(name: Any, *args: Any, **kwargs: Any) -> Any:
            local = resolve(name)
            return original_tokenizer(local if local is not None else name, *args, **kwargs)

        AutoTokenizer.from_pretrained = staticmethod(from_pretrained)  # type: ignore[method-assign]
        _patched["tokenizer"] = original_tokenizer
    except Exception:  # noqa: BLE001 - transformers is in the voice venv; without it no loader needs the redirect
        pass
    _patched.update(originals)


def remove_redirect() -> None:
    """Undo ``install_redirect`` (tests)."""
    if not _patched:
        return
    import huggingface_hub  # type: ignore[import-untyped]

    for name in ("snapshot_download", "hf_hub_download"):
        current = getattr(huggingface_hub, name)
        setattr(huggingface_hub, name, _patched[name])
        for module_name, module in list(sys.modules.items()):
            if module is not None and module_name.startswith("mlx_audio") and getattr(module, name, None) is current:
                setattr(module, name, _patched[name])
    if "tokenizer" in _patched:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        AutoTokenizer.from_pretrained = _patched["tokenizer"]  # type: ignore[method-assign]
    _patched.clear()
    _redirect.clear()


# ---------------------------------------------------------------------------
# Per-family calls
# ---------------------------------------------------------------------------

def _collect(results: Any) -> Any:
    import numpy as np

    parts = [np.asarray(result.audio, dtype=np.float32).reshape(-1) for result in results]
    if not parts:
        raise RuntimeError("The speech model produced no audio.")
    return np.concatenate(parts)


def _rate(model: Any) -> int:
    return int(getattr(model, "sample_rate", 24_000))


def _seed(seed: Any) -> None:
    if seed is None:
        return
    import mlx.core as mx  # type: ignore[import-untyped]

    mx.random.seed(int(seed))


def _load_default_conds(model: Any, path: str) -> None:
    """Chatterbox V3 ships no default voice; load a ``conds.safetensors`` (the V2 build's) into the model."""
    import mlx.core as mx  # type: ignore[import-untyped]
    from mlx_audio.tts.models.chatterbox.chatterbox import Conditionals, T3Cond  # type: ignore[import-untyped]

    data = mx.load(path)
    speaker = data.get("t3.speaker_emb")
    emotion = data.get("t3.emotion_adv")
    t3 = T3Cond(
        speaker_emb=speaker if speaker is not None else mx.zeros((1, 256)),
        cond_prompt_speech_tokens=data.get("t3.cond_prompt_speech_tokens"),
        emotion_adv=emotion if emotion is not None else mx.ones((1, 1, 1)) * 0.5,
    )
    gen = {k[len("gen."):]: v for k, v in data.items() if k.startswith("gen.")}
    if "prompt_feat_len" not in gen and "prompt_feat" in gen:
        gen["prompt_feat_len"] = mx.array([gen["prompt_feat"].shape[1]])
    model._conds = Conditionals(t3, gen)


def turbo(model: Any, request: dict[str, Any], decode_audio: Callable[..., Any]) -> tuple[Any, int]:
    """Chatterbox-Turbo: tags stay in the text; our chunks are already sentences, so its own split is off."""
    rate = _rate(model)
    kwargs: dict[str, Any] = {
        "temperature": 0.8, "top_p": 0.95, "repetition_penalty": 1.2, "split_pattern": None, "max_tokens": 800,
    }
    saved = getattr(model, "_conds", None)
    try:
        if request.get("refAudioPath"):
            kwargs["ref_audio"] = decode_audio(str(request["refAudioPath"]), rate)
            kwargs["sample_rate"] = rate
        return _collect(model.generate(str(request["text"]), **kwargs)), rate
    finally:
        model._conds = saved  # cloning replaces the built-in voice on the model; put it back


def chatterbox(model: Any, request: dict[str, Any], decode_audio: Callable[..., Any]) -> tuple[Any, int]:
    """Chatterbox Multilingual V3: language, expression (exaggeration) and guidance (cfg_weight) always sent."""
    rate = _rate(model)
    kwargs: dict[str, Any] = {
        "lang_code": str(request.get("language") or "en"),
        "exaggeration": float(request.get("expression", 0.5)),
        "cfg_weight": float(request.get("guidance", 0.5)),
        "temperature": 0.8,
    }
    saved = getattr(model, "_conds", None)
    try:
        if request.get("refAudioPath"):
            import mlx.core as mx  # type: ignore[import-untyped]

            # This loader needs an mx.array at its own rate (a numpy array fails in expand_dims).
            kwargs["ref_audio"] = mx.array(decode_audio(str(request["refAudioPath"]), rate))
        elif saved is None:
            if not request.get("defaultVoicePath"):
                raise ValueError("Chatterbox needs a voice sample here: pick a cloned voice.")
            _load_default_conds(model, str(request["defaultVoicePath"]))
            saved = model._conds
        return _collect(model.generate(str(request["text"]), **kwargs)), rate
    finally:
        model._conds = saved


def voxcpm2(model: Any, request: dict[str, Any], decode_audio: Callable[..., Any]) -> tuple[Any, int]:
    """VoxCPM2: design (``instruct``), exact clone (clip + clip as prompt + transcript), steered clone (clip +
    ``instruct``), and continuation of a designed voice from its own first chunk (``continueAudioPath``)."""
    rate = _rate(model)
    kwargs: dict[str, Any] = {"inference_timesteps": 10, "cfg_value": 2.0}
    mode = str(request.get("mode") or "")
    if request.get("refAudioPath"):
        clip = decode_audio(str(request["refAudioPath"]), rate)
        kwargs["ref_audio"] = clip
        if mode == "steered":
            kwargs["instruct"] = str(request.get("instruct") or "")
        else:  # exact
            kwargs["prompt_audio"] = clip
            kwargs["prompt_text"] = str(request.get("refText") or "")
    elif request.get("continueAudioPath"):
        kwargs["prompt_audio"] = decode_audio(str(request["continueAudioPath"]), rate)
        kwargs["prompt_text"] = str(request.get("continueText") or "")
    elif request.get("instruct"):
        kwargs["instruct"] = str(request["instruct"])
    else:
        raise ValueError("VoxCPM2 needs a voice description or a cloned voice.")
    return _collect(model.generate(str(request["text"]), **kwargs)), rate


def orpheus(model: Any, request: dict[str, Any], _decode_audio: Callable[..., Any]) -> tuple[Any, int]:
    """Orpheus 3B: one of eight preset voices; tags arrive as ``<laugh>``. No cloning."""
    if request.get("refAudioPath"):
        raise ValueError("Orpheus speaks its eight preset voices only; it does not clone.")
    rate = _rate(model)
    results = model.generate(
        str(request["text"]), voice=str(request.get("speaker") or "tara"), temperature=0.6, top_p=0.8,
        repetition_penalty=1.3, split_pattern="\n", max_tokens=int(request.get("maxTokens") or 1200),
    )
    return _collect(results), rate


_CALLS: dict[str, Callable[..., tuple[Any, int]]] = {
    "chatterbox-turbo": turbo, "chatterbox": chatterbox, "voxcpm2": voxcpm2, "orpheus": orpheus,
}


def synthesize(model: Any, request: dict[str, Any], decode_audio: Callable[..., Any]) -> tuple[Any, int]:
    family = str(request.get("family") or "")
    call = _CALLS.get(family)
    if call is None:
        raise ValueError(f"Unknown speech model family {family!r}.")
    _seed(request.get("seed"))
    return call(model, request, decode_audio)
