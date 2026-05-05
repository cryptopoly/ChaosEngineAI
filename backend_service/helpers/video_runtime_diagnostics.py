"""Translate opaque diffusers / transformers lazy-import errors into actionable
guidance for the Video Studio UI.

Diffusers raises ``RuntimeError("Failed to import diffusers.pipelines.X.Y
because of the following error (look up to see its traceback): Could not
import module 'Z'. Are this object's requirements defined correctly?")``
whenever any pipeline submodule import chain fails. The wrapped message
hides the real cause -- the user just sees a vague "module 'T5EncoderModel'"
hint with no path forward.

This helper:
  * recognises the wrapper text so we know to dig
  * runs targeted in-process probes on the actual chain (transformers,
    torchao, torch, sentencepiece, protobuf) to surface the underlying
    error message
  * formats a one-paragraph reason the UI can show in the row banner

All probes are wrapped in try/except so we never raise from the diagnostics
helper itself -- if probing also fails we fall back to the original wrapped
text rather than masking it.
"""
from __future__ import annotations

import importlib
import importlib.util
import re
from typing import Any


_DIFFUSERS_LAZY_IMPORT_PATTERN = re.compile(
    r"Failed to import (?P<module>diffusers[\w\.]+) because of the following error",
    re.IGNORECASE,
)


def _probe_module_import_error(module_name: str) -> str | None:
    """Return the underlying ImportError message when *module_name* won't load.

    Returns ``None`` when the module imports cleanly. Catches every exception
    type because import-time errors aren't always ImportError -- a partial
    install can raise AttributeError, RuntimeError, OSError, etc.
    """
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _probe_torch_device() -> dict[str, Any]:
    """Inspect the installed torch wheel: version + CUDA availability.

    Returns ``{"installed": False}`` when torch isn't on the path. Otherwise
    returns version + cuda_available + cuda_built_with so the caller can
    flag the "CPU torch on a CUDA host" case explicitly.
    """
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}
    try:
        import torch  # type: ignore
        return {
            "installed": True,
            "version": str(getattr(torch, "__version__", "unknown")),
            "cuda_available": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
            "cuda_built_with": str(getattr(torch.version, "cuda", None) or ""),
        }
    except Exception as exc:
        return {"installed": True, "import_error": f"{type(exc).__name__}: {exc}"}


def _format_torchao_torch_mismatch(torch_info: dict[str, Any]) -> str | None:
    """Return a hint when torchao requires a newer torch than what's installed.

    The specific failure that triggered this helper:
      ``AttributeError: module 'torch.utils._pytree' has no attribute
      'register_constant'``
    Newer torchao (>=0.10) uses ``register_constant`` which only exists from
    torch 2.11. Older torch + newer torchao breaks the entire transformers
    quantizer import chain, which then breaks T5EncoderModel.
    """
    if not torch_info.get("installed"):
        return None
    if importlib.util.find_spec("torchao") is None:
        return None
    torchao_error = _probe_module_import_error("torchao.utils")
    if torchao_error and "register_constant" in torchao_error:
        torch_version = torch_info.get("version", "unknown")
        return (
            "torchao is incompatible with the installed torch wheel "
            f"({torch_version}). torchao >= 0.10 needs torch >= 2.11 -- "
            "the missing torch.utils._pytree.register_constant attribute "
            "breaks the transformers quantizer import chain, which is what "
            "stops the T5 text encoder from loading. Open Settings > Setup "
            "and re-run Install GPU runtime (torch will upgrade) or "
            "uninstall torchao until torch is updated."
        )
    return None


def _format_cpu_torch_on_cuda_host_warning(torch_info: dict[str, Any]) -> str | None:
    """Detect the "you have a 4090 but the GPU bundle installed CPU torch" case.

    The +cpu local-version tag is the canonical marker. If the user has a
    CUDA-capable host (we delegate that probe to nvidia_gpu_present) but
    their torch is CPU-only, video models can technically load but they'll
    run on CPU only -- effectively useless for any modern DiT.
    """
    if not torch_info.get("installed"):
        return None
    version = str(torch_info.get("version") or "")
    if "+cpu" not in version.lower():
        return None
    try:
        from backend_service.helpers.gpu import nvidia_gpu_present
        nvidia_present = nvidia_gpu_present()
    except Exception:
        nvidia_present = False
    if not nvidia_present:
        return None
    return (
        f"The installed torch wheel is CPU-only ({version}) even though an "
        "NVIDIA GPU is present. Video generation will run on CPU, which is "
        "unusable for modern video DiTs. Open Settings > Setup and click "
        "Install CUDA torch (or re-run Install GPU runtime) so the CUDA "
        "wheel replaces the CPU one. After it lands, click Restart Backend."
    )


def diagnose_diffusers_lazy_import_error(error_text: str) -> str | None:
    """Translate a diffusers lazy-import RuntimeError into a friendlier reason.

    Returns ``None`` when the error doesn't match the lazy-import wrapper
    pattern (caller should fall back to the raw text). Otherwise returns a
    paragraph that names the real broken dep and points the user at the
    Setup page action that fixes it.
    """
    if not error_text or not _DIFFUSERS_LAZY_IMPORT_PATTERN.search(error_text):
        return None

    torch_info = _probe_torch_device()

    # Highest-priority signals first: a fundamentally broken torch install
    # invalidates every downstream "missing X" theory, so report it before
    # checking sentencepiece / protobuf.
    cpu_torch_hint = _format_cpu_torch_on_cuda_host_warning(torch_info)
    if cpu_torch_hint:
        return cpu_torch_hint

    torchao_hint = _format_torchao_torch_mismatch(torch_info)
    if torchao_hint:
        return torchao_hint

    # Walk the typical T5EncoderModel dependency chain in import order and
    # report the first concrete failure. We check transformers itself last
    # because its error often comes from a deeper module (quantizers, etc).
    chain = [
        ("torch", "torch"),
        ("sentencepiece", "sentencepiece"),
        ("google.protobuf", "protobuf"),
        ("transformers.quantizers", "transformers (quantizers submodule)"),
        ("transformers", "transformers"),
    ]
    for module_name, friendly_name in chain:
        if importlib.util.find_spec(module_name.split(".")[0]) is None:
            return (
                f"The backend Python is missing {friendly_name}, which "
                "diffusers needs to load the T5 text encoder. Open Settings "
                f"> Setup and click Install {friendly_name.split(' ')[0]} "
                "(or re-run Install GPU runtime to repair the whole stack), "
                "then click Restart Backend."
            )
        probe_error = _probe_module_import_error(module_name)
        if probe_error:
            return (
                f"The backend Python could not import {friendly_name}: "
                f"{probe_error}. This is what's blocking the T5 text encoder "
                "(and therefore CogVideoX, Wan, LTX, and HunyuanVideo). "
                "Open Settings > Setup and re-run Install GPU runtime to "
                "rebuild the dependency chain, then click Restart Backend."
            )

    # Probes all passed but diffusers still failed -- surface the original
    # wrapped error rather than pretending we know what's wrong.
    return None
