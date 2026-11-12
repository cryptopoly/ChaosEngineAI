"""Keep the backend's system memory down around diffusers pipelines: hand freed memory back after Unload, and load
pipelines on Windows without the memory-mapped path that doubles their commit.

glibc's allocator keeps freed pages in its arenas, so a diffusers pipeline with model CPU offload left the backend
holding about 12 GB of system RAM after Unload on a Linux CUDA box, and the memory gates then refused the next
model. ``malloc_trim(0)`` gives the free pages back. Windows has its own switch (``MIMALLOC_PURGE_DELAY`` in
``backend_service/__init__``); macOS needs nothing. Never raises: trimming is housekeeping.
"""

from __future__ import annotations

import sys
from typing import Any


def release_freed_memory() -> bool:
    """Ask glibc to return freed heap pages to the OS. True when pages were returned (Linux with glibc), else False."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):  # musl, or a libc without malloc_trim
        return False


def pipeline_load_kwargs() -> dict[str, Any]:
    """Extra ``from_pretrained`` keywords for a diffusers pipeline or model on this platform.

    On Windows a memory-mapped safetensors load stays charged to the process's commit for as long as the model
    lives, on top of the weights cast to the load dtype. Wan 2.2 TI2V-5B (fp32 transformer on disk, loaded as bf16)
    held 46.8 GB of private memory after loading; read without the mapping it held 28.5 GB. CUDA allocations also
    count against commit under WDDM, so with model CPU offload the system ran out of commit (64 GB RAM, a page file
    that could not grow on a nearly full disk): ``c10.dll`` access violation, and ``dwm.exe`` died with it.
    """
    return {"disable_mmap": True} if sys.platform == "win32" else {}
