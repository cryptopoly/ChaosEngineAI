"""ChaosEngineAI desktop backend service."""

import os as _os
import sys as _sys

# PyTorch's Windows builds allocate CPU memory through mimalloc, which keeps freed memory for reuse instead of
# handing it back. A diffusers pipeline with model CPU offload moves its parts between the card and system memory,
# and after an Unload the backend still held 16 GB (FLUX.2 Klein) to 26 GB (plus Wan 2.2 TI2V) of system RAM on an
# RTX 4090 PC, so the memory gates then refused the next model ("needs 26 GB, only 23 GB available"). Purging at
# once gives it back (1.7 GB after the same run). mimalloc reads these when torch loads, so they are set before any
# import of torch; both names, for older and newer mimalloc. A user can still export their own values.
if _sys.platform == "win32":
    _os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    _os.environ.setdefault("MIMALLOC_RESET_DELAY", "0")
