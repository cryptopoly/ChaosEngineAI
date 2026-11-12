"""Hand freed system memory back to the OS after a model is unloaded (Linux).

glibc's allocator keeps freed pages in its arenas, so a diffusers pipeline with model CPU offload left the backend
holding about 12 GB of system RAM after Unload on a Linux CUDA box, and the memory gates then refused the next
model. ``malloc_trim(0)`` gives the free pages back. Windows has its own switch (``MIMALLOC_PURGE_DELAY`` in
``backend_service/__init__``); macOS needs nothing. Never raises: trimming is housekeeping.
"""

from __future__ import annotations

import sys


def release_freed_memory() -> bool:
    """Ask glibc to return freed heap pages to the OS. True when pages were returned (Linux with glibc), else False."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):  # musl, or a libc without malloc_trim
        return False
