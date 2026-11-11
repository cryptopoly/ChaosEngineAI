"""Which low-precision weight formats this GPU can run in hardware (read from the card, never assumed).

- ``fp8``: Ada (compute capability 8.9) and newer. Older cards can store fp8 but compute slower than bf16.
- ``nvfp4``: Blackwell only (compute capability 10.0 and up: B200 / GB200 at 10.x, RTX 50-series at 12.0,
  DGX Spark at 12.1). An RTX 40-series cannot run it natively and must not be offered it.

Pure functions over a ``(major, minor)`` tuple so they are testable without a GPU.
"""

from __future__ import annotations


def fp8_supported(capability: tuple[int, int] | None) -> bool:
    return capability is not None and tuple(capability) >= (8, 9)


def nvfp4_supported(capability: tuple[int, int] | None) -> bool:
    return capability is not None and tuple(capability)[0] >= 10


def device_capability() -> tuple[int, int] | None:
    """``torch.cuda.get_device_capability()`` for device 0, or None without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return int(major), int(minor)
    except Exception:  # noqa: BLE001 - no torch or no driver: no hardware formats
        return None


def hardware_formats(capability: tuple[int, int] | None = None) -> dict[str, bool]:
    cap = capability if capability is not None else device_capability()
    return {"fp8": fp8_supported(cap), "nvfp4": nvfp4_supported(cap)}
