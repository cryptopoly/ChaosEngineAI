from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from collections.abc import Callable, Iterator
from typing import Any

from backend_service.reasoning_split import (
    ThinkingStreamResult,
    ThinkingTokenFilter,
    strip_thinking_tokens as _strip_thinking_tokens,
)
from backend_service.model_resolution import resolve_dflash_target_ref

from backend_service.inference._constants import (
    CAPABILITY_CACHE_TTL_SECONDS,
    DEFAULT_LLAMA_TIMEOUT_SECONDS,
    DEFAULT_MLX_TIMEOUT_SECONDS,
    MLX_LOAD_TIMEOUT_SECONDS,
    WORKSPACE_ROOT,
)
from backend_service.inference.base import (
    BackendCapabilities,
    BaseInferenceEngine,
    GenerationResult,
    LoadedModelInfo,
    RepeatedLineGuard,
    StreamChunk,
)
from backend_service.inference._utils import (
    _append_runtime_note,
    _find_open_port,
    _http_json,
    _is_local_target,
    _looks_like_gguf,
    _normalize_message_content,
    _now_label,
    _read_text_tail,
    _resolve_gguf_path,
)
from backend_service.inference.binaries import (
    _CHAOSENGINE_BIN_DIR,
    _LLAMA_FALLBACK_DIRS,
    _json_subprocess,
    _resolve_llama_cli,
    _resolve_llama_server,
    _resolve_llama_server_turbo,
    _resolve_mlx_python,
    _which_with_fallbacks,
)
from backend_service.inference.capabilities import (
    _capability_cache,
    _capability_lock,
    _initial_backend_capabilities,
    _probe_native_backends,
    get_backend_capabilities,
)
from backend_service.inference.conversion import (
    _MLX_LM_SUPPORTED_CACHE,
    _bytes_to_gb,
    _default_conversion_output,
    _mlx_lm_supported_model_types,
    _nearest_supported_arch,
    _path_size_bytes,
    _peek_hf_model_type,
)
from backend_service.inference.jsonrpc import JsonRpcProcess
from backend_service.inference.llama_cpp_engine import (
    LlamaCppEngine,
    _CACHE_TYPE_CACHE,
    _LLAMA_HELP_CACHE,
    _LLAMA_SAMPLER_KEYS,
    _STANDARD_CACHE_TYPES,
    _apply_llama_chat_template_fixes,
    _apply_sampler_kwargs,
    _friendly_llama_error,
    _gguf_startup_fallback_note,
    _llama_server_cache_types,
    _llama_server_help_text,
    _llama_server_supports,
    _resolve_mmproj_path,
)
from backend_service.inference.mlx_engine import MLXWorkerEngine
from backend_service.inference.simple_engines import (
    MockInferenceEngine,
    RemoteOpenAIEngine,
)


# Phase 1b-8: RuntimeController moved to backend_service.inference.controller.
# Re-exported here so existing call sites (state, routes, tests) keep working.
from backend_service.inference.controller import RuntimeController  # noqa: E402,F401
