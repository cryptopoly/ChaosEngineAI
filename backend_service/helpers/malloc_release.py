"""Keep the backend's system memory down around diffusers pipelines: hand freed memory back after Unload, and load
pipelines on Windows without the memory-mapped path that doubles their commit.

glibc's allocator keeps freed pages in its arenas, so a diffusers pipeline with model CPU offload left the backend
holding about 12 GB of system RAM after Unload on a Linux CUDA box, and the memory gates then refused the next
model. ``malloc_trim(0)`` gives the free pages back. Windows has its own switches (``MIMALLOC_ENV``, applied in
``backend_service/__init__``); macOS needs nothing. Never raises: trimming is housekeeping.
"""

from __future__ import annotations

import os
import sys
from typing import Any


# mimalloc settings for PyTorch's Windows builds (torch links mimalloc into c10.dll). mimalloc reads them when torch
# loads, so ``apply_allocator_env`` runs from ``backend_service/__init__`` before anything imports torch.
#
# * ``PURGE_DELAY`` / ``RESET_DELAY`` 0: hand freed memory back at once instead of keeping it for reuse (both names,
#   for older and newer mimalloc). Without them the backend kept 16 to 26 GB after an Unload.
# * ``ABANDONED_RECLAIM_ON_FREE`` 1: memory allocated by a thread that has since exited (a loader thread that read
#   the weights, a request worker that went idle) sits in "abandoned" segments, and by default mimalloc does not take
#   such a segment back when a block in it is freed, so the purge above never runs for it. Unload frees the weights
#   from another thread, so on some loads the whole pipeline stayed committed: 16.6 GB working set and 22.9 GB
#   private after unloading FLUX.2 Klein on an RTX 4090 PC with no Python object or tensor left alive, and the gate
#   refused the next image or video. Standalone (torch 2.11 and 2.14): tensors made in a thread that exits and freed
#   from another stay committed (2.86 GB of 2.86, 14.1 of 14.1); with this option and the purge delay at 0 the
#   process drops back to its 1.6 GB baseline. Purge delay 0 alone, or abandoned-page purging alone, changes nothing.
MIMALLOC_ENV: dict[str, str] = {
    "MIMALLOC_PURGE_DELAY": "0",
    "MIMALLOC_RESET_DELAY": "0",
    "MIMALLOC_ABANDONED_RECLAIM_ON_FREE": "1",
}


def apply_allocator_env(environ: Any = None, platform: str | None = None) -> dict[str, str]:
    """Set the mimalloc defaults on Windows (a value the user exported wins). Returns what was set."""
    environ = os.environ if environ is None else environ
    if (platform or sys.platform) != "win32":
        return {}
    applied = {}
    for name, value in MIMALLOC_ENV.items():
        if name not in environ:
            environ[name] = value
            applied[name] = value
    return applied


def release_freed_memory() -> bool:
    """Ask glibc to return freed heap pages to the OS. True when pages were returned (Linux with glibc), else False."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):  # musl, or a libc without malloc_trim
        return False


class ModelThread:
    """Run a model engine's load, generation and unload on one thread of its own (Windows; elsewhere a direct call).

    mimalloc returns freed memory only when the block is freed on the thread that allocated it (or that thread has
    exited and the freeing thread reclaims it). Request handlers run on a pool of worker threads, so the thread that
    loaded a pipeline (and, with model CPU offload, moved its weights back to system memory after every run) was
    seldom the one that served Unload: on an RTX 4090 PC the installed backend kept 16.5 GB private after unloading
    FLUX.2 Klein, and in a standalone run Klein loaded on one live thread and unloaded on another kept 16.6 GB (29.9
    after a second cycle), while load and unload on the same thread went back to 3.3 GB. Funnelling every call that
    touches the weights through one thread keeps allocation and free together. ``CHAOSENGINE_MODEL_THREAD=0`` turns
    it off. A call made from that thread runs inline, so nested calls cannot deadlock.
    """

    def __init__(self, name: str) -> None:
        import threading

        self._name = name
        self._lock = threading.Lock()
        self._executor: Any = None
        self._thread_ident: int | None = None

    @staticmethod
    def enabled(platform: str | None = None, environ: Any = None) -> bool:
        environ = os.environ if environ is None else environ
        return (platform or sys.platform) == "win32" and environ.get("CHAOSENGINE_MODEL_THREAD", "1") != "0"

    def _mark(self) -> None:
        import threading

        self._thread_ident = threading.get_ident()

    def run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        import contextvars
        import threading

        if not self.enabled() or threading.get_ident() == self._thread_ident:
            return fn(*args, **kwargs)
        with self._lock:
            if self._executor is None:
                from concurrent.futures import ThreadPoolExecutor

                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=self._name, initializer=self._mark,
                )
            executor = self._executor
        context = contextvars.copy_context()  # provenance / agent job owner travel with the call
        return executor.submit(context.run, fn, *args, **kwargs).result()


def pipeline_load_kwargs() -> dict[str, Any]:
    """Extra ``from_pretrained`` keywords for a diffusers pipeline or model on this platform.

    On Windows a memory-mapped safetensors load stays charged to the process's commit for as long as the model
    lives, on top of the weights cast to the load dtype. Wan 2.2 TI2V-5B (fp32 transformer on disk, loaded as bf16)
    held 46.8 GB of private memory after loading; read without the mapping it held 28.5 GB. CUDA allocations also
    count against commit under WDDM, so with model CPU offload the system ran out of commit (64 GB RAM, a page file
    that could not grow on a nearly full disk): ``c10.dll`` access violation, and ``dwm.exe`` died with it.
    """
    return {"disable_mmap": True} if sys.platform == "win32" else {}
