"""ChaosEngineAI desktop backend service."""

# PyTorch's Windows builds allocate CPU memory through mimalloc, which by default keeps freed memory instead of
# handing it back, so after an Unload the backend held 16 to 26 GB of system RAM and the memory gates refused the
# next model. mimalloc reads its settings when torch loads, so they are set here, before any import of torch; the
# settings and the measurements are in ``helpers/malloc_release.MIMALLOC_ENV``. A user can still export their own.
from backend_service.helpers.malloc_release import apply_allocator_env as _apply_allocator_env

_apply_allocator_env()
