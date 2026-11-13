"""LTX-2.5 on diffusers: load the transformer and the Gemma 4 text encoder in 4-bit (bitsandbytes NF4) at load time.

The official ``Lightricks/LTX-2.5-Diffusers`` repo is bf16 only: a 38 GB transformer and a 24 GB text encoder, so a
24 GB card cannot hold either. No pre-quantised diffusers pack exists, so the row that asks for it quantises both while
the shards are read. The transformer goes through ``try_load_bnb_nf4_transformer`` (``LTX2VideoTransformer3DModel`` is
registered for it); the text encoder is built here from the class its own ``config.json`` names, with only the
language-model linear layers quantised. Which precision the text encoder gets depends on the GPU memory free now
(``text_encoder_precision``): a card with enough room, or a unified-memory machine, keeps it bf16. Nothing is keyed to a
card name. Every failure returns a note instead of raising, so the caller can say why the text encoder stayed bf16.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

LTX25_REPO = "Lightricks/LTX-2.5-Diffusers"

# Optional pipeline components the app never uses, passed as ``None`` on every LTX-2.5 load (both rows, text and
# image to video). Both serve only prompt enhancement: ``prompt_enhancer`` (10 GB) is not downloaded, and
# ``processor`` (``Gemma4Processor``) needs torchvision for its video processor, which the GPU runtime does not
# ship, so the installed app failed every clip with "Gemma4VideoProcessor requires the Torchvision library".
LTX25_UNUSED_COMPONENTS: dict[str, None] = {"prompt_enhancer": None, "processor": None}

# At or above this much GPU memory free, the 24 GB text encoder stays bf16 (the transformer is still 4-bit: the row
# asked for it). Below it both are 4-bit. Override with CHAOSENGINE_LTX25_TE_BF16_MIN_GB.
TEXT_ENCODER_BF16_MIN_FREE_GB = 48.0

# Modules bitsandbytes must leave alone (matched by name; ``bnb`` skips a module whose dotted name contains one of
# these as a whole segment): the embeddings (token and the per-layer ones), the output head, and the vision and audio
# towers with their projectors. Norms are not linear layers, so they are never quantised in the first place.
TEXT_ENCODER_SKIP_MODULES: tuple[str, ...] = (
    "lm_head",
    "embed_tokens",
    "embed_tokens_per_layer",
    "per_layer_model_projection",
    "per_layer_projection",
    "vision_tower",
    "vision_model",
    "audio_tower",
    "audio_model",
    "embed_vision",
    "embed_audio",
    "multi_modal_projector",
)


def is_ltx25_repo(repo: str) -> bool:
    return repo == LTX25_REPO


def bf16_threshold_gb() -> float:
    raw = os.environ.get("CHAOSENGINE_LTX25_TE_BF16_MIN_GB", "").strip()
    try:
        value = float(raw) if raw else TEXT_ENCODER_BF16_MIN_FREE_GB
    except ValueError:
        return TEXT_ENCODER_BF16_MIN_FREE_GB
    return value if value > 0 else TEXT_ENCODER_BF16_MIN_FREE_GB


def text_encoder_precision(free_gb: float | None, *, unified_memory: bool = False) -> str:
    """``"bf16"`` or ``"nf4"`` for the text encoder, from the GPU memory free now.

    Unified memory (GPU and CPU share one pool, nothing to save by splitting) and a card with at least the threshold
    free keep bf16. An unreadable figure picks 4-bit: it is the one that fits the smaller cards.
    """
    if unified_memory:
        return "bf16"
    if free_gb is None:
        return "nf4"
    return "bf16" if free_gb >= bf16_threshold_gb() else "nf4"


def _read_text_encoder_architecture(local_path: str) -> str | None:
    try:
        config = json.loads((Path(local_path) / "text_encoder" / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    architectures = config.get("architectures") if isinstance(config, dict) else None
    if isinstance(architectures, list) and architectures and isinstance(architectures[0], str):
        return architectures[0]
    return None


def build_text_encoder_quantization_config(torch: Any) -> tuple[Any, str | None]:
    """``(BitsAndBytesConfig, None)`` for the text encoder, or ``(None, note)`` when it cannot be built."""
    if importlib.util.find_spec("bitsandbytes") is None:
        return None, (
            "bitsandbytes is not installed, so the LTX-2.5 text encoder stays bf16 (24 GB in system memory). "
            "Open Settings > Setup, click Install GPU runtime, and restart the backend."
        )
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except ImportError:
        return None, "The installed transformers has no BitsAndBytesConfig, so the LTX-2.5 text encoder stays bf16."
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=list(TEXT_ENCODER_SKIP_MODULES),
    )
    return config, None


def _dtype_kwarg(transformers: Any) -> str:
    """transformers 5 renamed ``torch_dtype`` to ``dtype``."""
    major = str(getattr(transformers, "__version__", "0")).split(".")[0]
    return "dtype" if major.isdigit() and int(major) >= 5 else "torch_dtype"


def try_load_ltx25_text_encoder_4bit(local_path: str, torch: Any, device: str) -> tuple[Any, str | None]:
    """Load ``text_encoder/`` in 4-bit with the class its config names. ``(None, note)`` on any failure."""
    if device != "cuda":
        return None, "A 4-bit text encoder needs CUDA; the LTX-2.5 text encoder stays bf16."
    quant_config, note = build_text_encoder_quantization_config(torch)
    if quant_config is None:
        return None, note
    try:
        import transformers  # type: ignore
    except ImportError:
        return None, "transformers is not installed, so the LTX-2.5 text encoder stays bf16."
    architecture = _read_text_encoder_architecture(local_path)
    if architecture is None:
        return None, "text_encoder/config.json names no architecture, so the LTX-2.5 text encoder stays bf16."
    model_cls = getattr(transformers, architecture, None)
    if model_cls is None:
        return None, (
            f"The installed transformers has no {architecture} (the class the LTX-2.5 text encoder names): "
            "upgrade transformers from Settings > Setup. The text encoder stays bf16."
        )
    try:
        index = int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - mocked or unreadable device
        index = 0
    try:
        encoder = model_cls.from_pretrained(
            str(Path(local_path) / "text_encoder"),
            quantization_config=quant_config,
            device_map={"": index},
            local_files_only=True,
            **{_dtype_kwarg(transformers): torch.bfloat16},
        )
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the bf16 encoder
        return None, f"4-bit text encoder load failed ({type(exc).__name__}: {exc}). The text encoder stays bf16."
    return encoder, f"Text encoder ({architecture}) loaded with NF4 (bitsandbytes), language-model layers only"


def load_ltx25_components(
    repo: str,
    local_path: str,
    torch: Any,
    device: str,
    *,
    free_gb: float | None,
    unified_memory: bool,
    load_transformer: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Pipeline keyword arguments and notes for a 4-bit LTX-2.5 load.

    ``load_transformer`` is the engine's NF4 transformer loader (``(repo, local_path, torch, device) -> (model,
    note)``). The transformer must come back 4-bit: a bf16 one is 38 GB and would be read into system memory, so that
    failure raises. A text encoder that cannot be quantised falls back to bf16 with a note. ``prompt_enhancer`` is
    always ``None``: its 10 GB are not downloaded and the app does not use it.
    """
    kwargs: dict[str, Any] = dict(LTX25_UNUSED_COMPONENTS)
    notes: list[str] = []
    transformer, transformer_note = load_transformer(repo, local_path, torch, device)
    if transformer_note:
        notes.append(transformer_note)
    if transformer is None:
        raise RuntimeError(transformer_note or "The LTX-2.5 transformer could not be loaded in 4-bit.")
    kwargs["transformer"] = transformer
    precision = text_encoder_precision(free_gb, unified_memory=unified_memory)
    if precision == "bf16":
        notes.append(
            f"LTX-2.5 text encoder kept bf16 ({'unified memory' if unified_memory else f'{free_gb:.0f} GB of GPU memory free'})."
        )
        return kwargs, notes
    encoder, encoder_note = try_load_ltx25_text_encoder_4bit(local_path, torch, device)
    if encoder_note:
        notes.append(encoder_note)
    if encoder is not None:
        kwargs["text_encoder"] = encoder
    return kwargs, notes
