"""``RuntimeController`` — the cross-engine LLM orchestrator.

Owns the warm pool of inference engines, chooses which engine handles a
given request, manages capability probes, tracks orphaned subprocess
children, runs model load/unload/convert flows, and surfaces a status
payload the routes layer hands back to the UI.

Extracted from ``backend_service/inference/__init__.py`` as part of the
v0.8.0 Phase 1b-8 refactor. Re-exported from ``backend_service.inference``
so existing call sites (``from backend_service.inference import
RuntimeController``) keep working without churn.
"""

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


class RuntimeController:
    # Hard upper bound on the warm pool independently of memory accounting —
    # if psutil isn't available we still want a sane cap.
    MAX_WARM_MODELS = 2
    # Reserve this much physical memory for the OS / UI / unrelated
    # processes when deciding whether a new (or incoming) model fits. Mirrors
    # the headroom used by ``helpers/system.py::spareHeadroomGb``.
    WARM_POOL_MEMORY_HEADROOM_BYTES = 6 * 1024 * 1024 * 1024

    def __init__(self, *, background_probe: bool = False) -> None:
        self.capabilities = _initial_backend_capabilities()
        self.engine: BaseInferenceEngine = MockInferenceEngine(self.capabilities)
        self.loaded_model: LoadedModelInfo | None = None
        self.runtime_note: str | None = None
        # Warm pool: keeps previously loaded engines alive for instant switch-back
        self._warm_pool: dict[str, tuple[BaseInferenceEngine, LoadedModelInfo]] = {}
        self._pool_lock = Lock()
        self._loading_progress: dict[str, Any] | None = None
        self._loading_log_tail: list[str] = []
        self._recent_orphaned_workers: list[dict[str, Any]] = []
        self._capability_probe_thread: Thread | None = None
        self._capability_probe_lock = Lock()
        if background_probe:
            self.start_capability_probe()

    def start_capability_probe(self, *, force: bool = False) -> None:
        with self._capability_probe_lock:
            if (
                self._capability_probe_thread is not None
                and self._capability_probe_thread.is_alive()
                and not force
            ):
                return
            thread = Thread(
                target=self._capability_probe_worker,
                kwargs={"force": force},
                name="chaosengine-capability-probe",
                daemon=True,
            )
            self._capability_probe_thread = thread
            thread.start()

    def _capability_probe_worker(self, *, force: bool = False) -> None:
        try:
            capabilities = get_backend_capabilities(force=force)
        except Exception as exc:
            current = self.capabilities
            capabilities = BackendCapabilities(
                pythonExecutable=current.pythonExecutable,
                mlxAvailable=False,
                mlxLmAvailable=False,
                mlxUsable=False,
                mlxMessage=f"Native backend detection failed: {type(exc).__name__}: {exc}",
                ggufAvailable=current.ggufAvailable,
                llamaCliPath=current.llamaCliPath,
                llamaServerPath=current.llamaServerPath,
                llamaServerTurboPath=current.llamaServerTurboPath,
                converterAvailable=False,
                vllmAvailable=False,
                vllmVersion=None,
                probing=False,
            )
        self.capabilities = capabilities
        if isinstance(self.engine, MockInferenceEngine):
            self.engine.capabilities = capabilities

    @staticmethod
    def _warm_pool_key(
        *,
        model_ref: str | None,
        runtime_target: str | None,
        path: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        fit_model_in_memory: bool,
        context_tokens: int,
        speculative_decoding: bool = False,
    ) -> str:
        target = runtime_target or path or model_ref or ""
        return json.dumps(
            {
                "target": target,
                "cacheStrategy": cache_strategy,
                "cacheBits": cache_bits,
                "fp16Layers": fp16_layers,
                "fusedAttention": fused_attention,
                "fitModelInMemory": fit_model_in_memory,
                "contextTokens": context_tokens,
                "speculativeDecoding": speculative_decoding,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _warm_pool_key_for_loaded(cls, info: LoadedModelInfo) -> str:
        return cls._warm_pool_key(
            model_ref=info.ref,
            runtime_target=info.runtimeTarget,
            path=info.path,
            cache_strategy=info.cacheStrategy,
            cache_bits=info.cacheBits,
            fp16_layers=info.fp16Layers,
            fused_attention=info.fusedAttention,
            fit_model_in_memory=info.fitModelInMemory,
            context_tokens=info.contextTokens,
            speculative_decoding=info.speculativeDecoding,
        )

    @staticmethod
    def _model_identity(
        *,
        model_ref: str | None,
        runtime_target: str | None,
        path: str | None,
    ) -> str:
        return str(runtime_target or path or model_ref or "")

    @classmethod
    def _model_identity_for_loaded(cls, info: LoadedModelInfo) -> str:
        return cls._model_identity(
            model_ref=info.ref,
            runtime_target=info.runtimeTarget,
            path=info.path,
        )

    def _purge_warm_entries_for_identity(
        self,
        model_identity: str,
        *,
        keep_key: str | None = None,
    ) -> None:
        if not model_identity:
            return
        stale_keys = [
            key
            for key, (_, info) in self._warm_pool.items()
            if key != keep_key and self._model_identity_for_loaded(info) == model_identity
        ]
        for key in stale_keys:
            old_engine, _info = self._warm_pool.pop(key)
            try:
                old_engine.unload_model()
            except Exception:
                pass

    def _park_active_engine_or_unload(
        self,
        *,
        requested_identity: str,
        keep_warm_previous: bool = True,
        required_free_bytes: int = 0,
    ) -> None:
        if not self.loaded_model or not self.engine:
            return
        current_key = self._warm_pool_key_for_loaded(self.loaded_model)
        current_identity = self._model_identity_for_loaded(self.loaded_model)
        if current_identity == requested_identity:
            try:
                self.engine.unload_model()
            except Exception:
                pass
            return
        self._purge_warm_entries_for_identity(current_identity, keep_key=current_key)
        if not keep_warm_previous:
            try:
                self.engine.unload_model()
            except Exception:
                pass
            return
        active_bytes = max(
            self._model_resident_bytes(self.loaded_model),
            self._engine_resident_bytes(self.engine),
        )
        self._evict_warm_pool(
            incoming_bytes=active_bytes,
        )
        if not self._can_keep_warm_model(active_bytes, required_free_bytes=required_free_bytes):
            try:
                self.engine.unload_model()
            except Exception:
                pass
            return
        self._warm_pool[current_key] = (self.engine, self.loaded_model)

    def _tracked_process_pids(self) -> set[int]:
        tracked: set[int] = set()
        active_pid = self.engine.process_pid() if self.engine else None
        if active_pid:
            tracked.add(int(active_pid))
        for engine, _info in self._warm_pool.values():
            pid = engine.process_pid()
            if pid:
                tracked.add(int(pid))
        return tracked

    # How long an orphan record stays visible in status() before being
    # dropped. The UI already auto-dismisses sooner; this keeps the backend
    # from hoarding records across an entire session.
    ORPHAN_RECORD_TTL_SECONDS = 45.0
    # Ignore children younger than this — they are almost always in a
    # spawn-in-progress race or a terminate-in-progress race where psutil
    # can see the PID but our engine hasn't finished registering/releasing
    # it. Killing them a second time and reporting them as "orphans" is
    # pure noise.
    ORPHAN_DETECTION_GRACE_SECONDS = 3.0

    def prune_stale_backend_children(self) -> None:
        tracked = self._tracked_process_pids()
        try:
            import psutil

            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=False)
        except Exception:
            self._expire_orphan_records()
            return

        now_mono = time.monotonic()
        now_wall = time.time()
        pruned: list[dict[str, Any]] = []
        for child in children:
            try:
                cmdline = " ".join(child.cmdline()).lower()
                name = (child.name() or "").lower()
                create_time = child.create_time()
            except Exception:
                continue

            is_mlx_worker = "backend_service.mlx_worker" in cmdline or "mlx_worker" in cmdline
            is_llama = "llama-server" in name or "llama-server" in cmdline
            if not (is_mlx_worker or is_llama):
                continue
            if child.pid in tracked:
                continue
            # Grace window: skip transient mid-spawn / mid-terminate
            # children. The previous behaviour counted these as orphans
            # every time an engine was swapped in or out, so a normal
            # model-change session produced 5-10 "orphans" purely from
            # timing races.
            if now_wall - create_time < self.ORPHAN_DETECTION_GRACE_SECONDS:
                continue

            record = {
                "pid": int(child.pid),
                "kind": "mlx_worker" if is_mlx_worker else "llama_server",
                "label": "MLX worker" if is_mlx_worker else "llama-server",
                "action": "terminated",
                "detectedAt": _now_label(),
                # Internal monotonic stamp used for TTL; not serialized.
                "_detectedAtMono": now_mono,
            }
            try:
                child.terminate()
                child.wait(timeout=2)
            except Exception:
                try:
                    child.kill()
                    record["action"] = "killed"
                except Exception:
                    record["action"] = "kill_failed"
            pruned.append(record)

        if pruned:
            self._recent_orphaned_workers = (pruned + self._recent_orphaned_workers)[:8]

        self._expire_orphan_records()

    def _expire_orphan_records(self) -> None:
        if not self._recent_orphaned_workers:
            return
        cutoff = time.monotonic() - self.ORPHAN_RECORD_TTL_SECONDS
        self._recent_orphaned_workers = [
            record
            for record in self._recent_orphaned_workers
            if float(record.get("_detectedAtMono", 0.0)) >= cutoff
        ]

    def _matches_active(self, model_ref: str) -> bool:
        if self.loaded_model is None:
            return False
        candidates = {
            self.loaded_model.ref,
            self.loaded_model.runtimeTarget,
            self.loaded_model.path,
            self.loaded_model.name,
        }
        return model_ref in {c for c in candidates if c}

    def get_engine_for_request(
        self, model_ref: str | None
    ) -> tuple[BaseInferenceEngine, LoadedModelInfo]:
        """Resolve which engine should serve a request.

        - Empty/None model_ref → active engine.
        - Matches active model identifiers → active engine.
        - Matches a warm pool entry by model identifier → that warm engine (without popping).
        - Otherwise → fall back to active engine.
        Raises RuntimeError if no model is loaded at all.
        """
        if self.loaded_model is None:
            raise RuntimeError("Load a model before sending prompts.")
        if not model_ref:
            return self.engine, self.loaded_model
        if self._matches_active(model_ref):
            return self.engine, self.loaded_model
        with self._pool_lock:
            for _, (eng, info) in reversed(list(self._warm_pool.items())):
                if model_ref in {info.ref, info.runtimeTarget, info.path, info.name}:
                    return eng, info
        return self.engine, self.loaded_model

    def unload_warm_model_by_ref(self, ref: str) -> bool:
        """Pop a single entry from the warm pool and unload it. No-op if not found.

        Never touches the active model. Returns True if something was unloaded.
        """
        if not ref:
            return False
        with self._pool_lock:
            match_key: str | None = None
            for key, (_, info) in reversed(list(self._warm_pool.items())):
                if ref in {info.ref, info.runtimeTarget, info.path, info.name}:
                    match_key = key
                    break
            if match_key is None:
                return False
            entry = self._warm_pool.pop(match_key)
        engine, _info = entry
        try:
            engine.unload_model()
        except Exception:
            pass
        self.prune_stale_backend_children()
        return True

    def clear_warm_pool(self) -> int:
        """Unload and forget every parked warm model."""
        with self._pool_lock:
            entries = list(self._warm_pool.values())
            self._warm_pool.clear()
        for engine, _info in entries:
            try:
                engine.unload_model()
            except Exception:
                pass
        if entries:
            self.prune_stale_backend_children()
        return len(entries)

    def refresh_capabilities(self, *, force: bool = False) -> BackendCapabilities:
        if force:
            # Clear cached help text and cache type sets so that newly
            # installed or updated binaries are re-probed.
            with _LLAMA_HELP_LOCK:
                _LLAMA_HELP_CACHE.clear()
            _CACHE_TYPE_CACHE.clear()
        self.capabilities = get_backend_capabilities(force=force)
        if isinstance(self.engine, MockInferenceEngine):
            self.engine.capabilities = self.capabilities
        return self.capabilities

    def _select_engine(
        self,
        *,
        backend: str,
        runtime_target: str | None,
        path: str | None,
    ) -> BaseInferenceEngine:
        hint = (backend or "auto").lower()
        target = runtime_target or path

        if hint in {"remote", "openai", "cloud"}:
            return RemoteOpenAIEngine(self.capabilities)
        if hint == "mlx":
            if self.capabilities.mlxUsable:
                return MLXWorkerEngine(self.capabilities)
            reason = self.capabilities.mlxMessage or "MLX is not available in this environment"
            raise RuntimeError(
                f"MLX backend requested but unavailable: {reason}. "
                f"Use a GGUF model with llama.cpp instead, or check your Python environment."
            )
        if hint in {"gguf", "llama.cpp", "llama-cpp"}:
            if self.capabilities.ggufAvailable:
                return LlamaCppEngine(self.capabilities)
            raise RuntimeError(
                "This model requires llama-server (llama.cpp) which is not installed. "
                "Install with: brew install llama.cpp"
            )
        if hint == "vllm":
            if self.capabilities.vllmAvailable:
                from backend_service.vllm_engine import VLLMEngine
                return VLLMEngine(self.capabilities)
            raise RuntimeError(
                "vLLM backend requested but not installed. "
                "Install with: pip install vllm (Linux + CUDA only)."
            )

        # Auto-detect: try to find a real backend
        if _looks_like_gguf(target):
            if self.capabilities.ggufAvailable:
                return LlamaCppEngine(self.capabilities)
            raise RuntimeError(
                "This is a GGUF model which requires llama-server. "
                "Install with: brew install llama.cpp"
            )
        if self.capabilities.mlxUsable:
            return MLXWorkerEngine(self.capabilities)
        if self.capabilities.ggufAvailable:
            return LlamaCppEngine(self.capabilities)
        raise RuntimeError(
            "No inference backend is available. "
            "Install llama.cpp (brew install llama.cpp) for GGUF models, "
            "or ensure MLX is working for safetensors models on Apple Silicon."
        )

    @staticmethod
    def _display_name(model_ref: str, model_name: str | None = None, path: str | None = None) -> str:
        if model_name:
            return model_name
        if path:
            return Path(path).stem or model_ref
        return model_ref.split("/")[-1]

    def _is_same_loaded_model(self, model_ref: str | None) -> bool:
        if self.loaded_model is None or not model_ref:
            return False
        return model_ref in {self.loaded_model.ref, self.loaded_model.runtimeTarget}

    def warm_models(self) -> list[dict[str, Any]]:
        """Return info about all models in the warm pool (including active)."""
        result = []
        if self.loaded_model:
            result.append({**self.loaded_model.to_dict(), "warm": True, "active": True})
        for ref, (_, info) in self._warm_pool.items():
            if self.loaded_model and ref == self.loaded_model.ref:
                continue
            result.append({**info.to_dict(), "warm": True, "active": False})
        return result

    @staticmethod
    def _model_resident_bytes(info: LoadedModelInfo) -> int:
        """Best-effort estimate of RAM held by a loaded model.

        For local weights we use on-disk size as a proxy — mlx-lm mmaps the
        weights so RSS tracks file size closely; for llama.cpp / GGUF the
        whole file ends up resident once warm. For catalog/no-path entries
        we fall back to 0 (no useful estimate, treat as memory-free).
        """
        return _path_size_bytes(info.path) if info.path else 0

    @staticmethod
    def _target_resident_bytes(*, path: str | None, runtime_target: str | None) -> int:
        for candidate in (path, runtime_target):
            if not candidate:
                continue
            size = _path_size_bytes(candidate)
            if size > 0:
                return size
        return 0

    @staticmethod
    def _engine_resident_bytes(engine: BaseInferenceEngine | None) -> int:
        if engine is None:
            return 0
        pid_getter = getattr(engine, "process_pid", None)
        pid = pid_getter() if callable(pid_getter) else None
        if not isinstance(pid, int):
            return 0
        try:
            import psutil

            return int(psutil.Process(pid).memory_info().rss)
        except Exception:
            return 0

    def _warm_pool_resident_bytes(self) -> int:
        return sum(
            max(self._model_resident_bytes(info), self._engine_resident_bytes(engine))
            for engine, info in self._warm_pool.values()
        )

    def _memory_budget_bytes(self) -> int:
        """Bytes available for warm-pool weights, after OS headroom.

        Returns 0 when psutil isn't usable; callers must fall back to the
        count-based MAX_WARM_MODELS cap in that case.
        """
        try:
            import psutil

            available = int(psutil.virtual_memory().available)
        except Exception:
            return 0
        return max(0, available - self.WARM_POOL_MEMORY_HEADROOM_BYTES)

    def _pop_oldest_warm_entry(self) -> None:
        if not self._warm_pool:
            return
        oldest_key = next(iter(self._warm_pool))
        old_engine, _ = self._warm_pool.pop(oldest_key)
        try:
            old_engine.unload_model()
        except Exception:
            pass

    def _can_keep_warm_model(self, incoming_bytes: int, *, required_free_bytes: int = 0) -> bool:
        budget = self._memory_budget_bytes()
        if budget <= 0:
            return True
        if required_free_bytes > budget:
            return False
        return self._warm_pool_resident_bytes() + incoming_bytes <= budget

    def _evict_warm_pool(self, *, incoming_bytes: int = 0) -> None:
        """Make room for an incoming entry in the warm pool.

        First applies the count cap (MAX_WARM_MODELS) so a flapping budget
        can never grow the pool unboundedly. Then, if ``psutil`` reports a
        live memory budget, evicts oldest entries until the pool plus the
        incoming model fits within ``available - headroom``.

        ``incoming_bytes`` is the resident-byte estimate for the model
        about to enter the pool (typically the model being parked from
        active to warm). Passing 0 still triggers the count cap.
        """
        while len(self._warm_pool) >= self.MAX_WARM_MODELS:
            self._pop_oldest_warm_entry()

        budget = self._memory_budget_bytes()
        if budget <= 0:
            return
        while self._warm_pool and self._warm_pool_resident_bytes() + incoming_bytes > budget:
            self._pop_oldest_warm_entry()

    def load_model(
        self,
        *,
        model_ref: str,
        model_name: str | None = None,
        canonical_repo: str | None = None,
        source: str = "catalog",
        backend: str = "auto",
        path: str | None = None,
        runtime_target: str | None = None,
        cache_strategy: str = "native",
        cache_bits: int = 0,
        fp16_layers: int = 0,
        fused_attention: bool = False,
        fit_model_in_memory: bool = True,
        context_tokens: int = 8192,
        speculative_decoding: bool = False,
        tree_budget: int = 0,
        keep_warm_previous: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LoadedModelInfo:
        self.refresh_capabilities()
        self._loading_progress = None
        self._loading_log_tail = []

        def _internal_progress(progress: dict[str, Any]) -> None:
            try:
                self._loading_progress = dict(progress)
                msg = progress.get("message")
                phase = progress.get("phase")
                if msg or phase:
                    line = f"[{phase}] {msg}" if phase and msg else str(msg or phase)
                    self._loading_log_tail.append(line)
                    if len(self._loading_log_tail) > 5:
                        self._loading_log_tail = self._loading_log_tail[-5:]
            except Exception:
                pass
            if progress_callback is not None:
                try:
                    progress_callback(progress)
                except Exception:
                    pass
        resolved_name = self._display_name(model_ref, model_name=model_name, path=path)
        requested_identity = self._model_identity(
            model_ref=model_ref,
            runtime_target=runtime_target,
            path=path,
        )
        incoming_load_bytes = self._target_resident_bytes(
            path=path,
            runtime_target=runtime_target,
        )

        # Check warm pool first — instant switch if the exact runtime profile is cached
        pool_key = self._warm_pool_key(
            model_ref=model_ref,
            runtime_target=runtime_target,
            path=path,
            cache_strategy=cache_strategy,
            cache_bits=cache_bits,
            fp16_layers=fp16_layers,
            fused_attention=fused_attention,
            fit_model_in_memory=fit_model_in_memory,
            context_tokens=context_tokens,
            speculative_decoding=speculative_decoding,
        )
        if pool_key in self._warm_pool:
            cached_engine, cached_info = self._warm_pool.pop(pool_key)
            self._purge_warm_entries_for_identity(requested_identity)
            self._park_active_engine_or_unload(
                requested_identity=requested_identity,
                keep_warm_previous=keep_warm_previous,
            )
            self.engine = cached_engine
            self.loaded_model = cached_info
            if canonical_repo is not None:
                self.loaded_model.canonicalRepo = canonical_repo
            self.runtime_note = cached_info.runtimeNote
            self.prune_stale_backend_children()
            return cached_info

        selected_engine = self._select_engine(
            backend=backend,
            runtime_target=runtime_target,
            path=path,
        )

        # Never keep multiple warm copies of the same logical model under
        # different runtime profiles; that just burns extra RAM.
        self._purge_warm_entries_for_identity(requested_identity)
        self._park_active_engine_or_unload(
            requested_identity=requested_identity,
            keep_warm_previous=keep_warm_previous,
            required_free_bytes=incoming_load_bytes,
        )

        self.engine = selected_engine
        try:
            loaded = self.engine.load_model(
                model_ref=model_ref,
                model_name=resolved_name,
                canonical_repo=canonical_repo,
                source=source,
                backend=self.engine.engine_name,
                path=path,
                runtime_target=runtime_target,
                cache_strategy=cache_strategy,
                cache_bits=cache_bits,
                fp16_layers=fp16_layers,
                fused_attention=fused_attention,
                fit_model_in_memory=fit_model_in_memory,
                context_tokens=context_tokens,
                speculative_decoding=speculative_decoding,
                tree_budget=tree_budget,
                progress_callback=_internal_progress,
            )
        except Exception:
            self.loaded_model = None
            self.runtime_note = None
            self._loading_progress = None
            self._loading_log_tail = []
            self.prune_stale_backend_children()
            raise

        self.loaded_model = loaded
        self.runtime_note = loaded.runtimeNote
        self._loading_progress = None
        self._loading_log_tail = []
        self.prune_stale_backend_children()
        return loaded

    def update_profile(
        self,
        *,
        canonical_repo: str | None = None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
    ) -> LoadedModelInfo:
        if self.loaded_model is None or self.engine is None:
            raise RuntimeError("No model is loaded.")
        if not isinstance(self.engine, MLXWorkerEngine):
            raise RuntimeError("In-place profile updates are only supported by the MLX runtime.")

        loaded = self.engine.update_profile(
            canonical_repo=canonical_repo,
            cache_strategy=cache_strategy,
            cache_bits=cache_bits,
            fp16_layers=fp16_layers,
            fused_attention=fused_attention,
        )
        self.loaded_model = loaded
        self.runtime_note = loaded.runtimeNote
        self._purge_warm_entries_for_identity(self._model_identity_for_loaded(loaded))
        self.prune_stale_backend_children()
        return loaded

    def loading_progress(self) -> tuple[dict[str, Any] | None, list[str]]:
        return self._loading_progress, list(self._loading_log_tail)

    def unload_model(self) -> None:
        self.engine.unload_model()
        self.loaded_model = None
        self.runtime_note = None
        self.prune_stale_backend_children()

    def generate(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        engine: BaseInferenceEngine | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if self.loaded_model is None:
            raise RuntimeError("Load a model before sending prompts.")

        target_engine = engine or self.engine
        result = target_engine.generate(
            prompt=prompt,
            history=history,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images,
            tools=tools,
            samplers=samplers,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
        )
        if result.runtimeNote is None:
            result.runtimeNote = self.runtime_note
        return result

    def stream_generate(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        engine: BaseInferenceEngine | None = None,
        thinking_mode: str | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        if self.loaded_model is None:
            raise RuntimeError("Load a model before sending prompts.")

        target_engine = engine or self.engine
        yield from target_engine.stream_generate(
            prompt=prompt,
            history=history,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images,
            tools=tools,
            thinking_mode=thinking_mode,
            samplers=samplers,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
        )

    def extract_gguf_metadata(self, path: str) -> dict[str, Any]:
        code, payload, message = _json_subprocess(
            [self.capabilities.pythonExecutable, "-m", "backend_service.mlx_worker", "gguf-metadata", path],
            timeout=15.0,
        )
        if code != 0 or payload is None:
            raise RuntimeError(message or "Failed to read GGUF metadata.")
        return payload

    def convert_model(
        self,
        *,
        source_ref: str | None,
        source_path: str | None,
        output_path: str | None,
        hf_repo: str | None,
        quantize: bool,
        q_bits: int,
        dtype: str,
        q_group_size: int = 64,
    ) -> dict[str, Any]:
        self.refresh_capabilities(force=True)
        if not self.capabilities.converterAvailable:
            raise RuntimeError(self.capabilities.mlxMessage or "MLX conversion is unavailable in this environment.")

        resolved_hf_repo = hf_repo
        gguf_metadata: dict[str, Any] | None = None
        source_label = source_path or source_ref or "model"

        # --hf-path accepts either a valid `owner/name` HF repo identifier
        # OR a local directory/file path. Figure out which one to hand to
        # mlx_lm.convert based on what the caller actually gave us.
        hf_path_arg: str | None = None

        if source_path and _looks_like_gguf(source_path):
            # HF cache layouts point at the repo root, not the .gguf file.
            # Resolve to the concrete .gguf before reading metadata so
            # extract_gguf_metadata never gets handed a directory.
            resolved_gguf_file = _resolve_gguf_path(source_path, source_ref) or source_path
            try:
                gguf_metadata = self.extract_gguf_metadata(resolved_gguf_file)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read GGUF metadata from {resolved_gguf_file}: {exc}"
                ) from exc
            if resolved_hf_repo is None:
                resolved_hf_repo = gguf_metadata.get("baseModelRepo")
            if resolved_hf_repo is None:
                raise RuntimeError(
                    "GGUF-to-MLX conversion needs a base Hugging Face model repo. "
                    "Either pick a source that includes base-model metadata in "
                    "its GGUF header, or provide `hfRepo` explicitly in the "
                    "Conversion page."
                )
            hf_path_arg = resolved_hf_repo
        elif source_path and Path(source_path).exists():
            # Local directory or file (Transformers/HF cache) — hand the
            # path directly to mlx_lm, which accepts local paths for
            # --hf-path. This avoids mis-using the library item's display
            # name as an HF repo identifier (which fails auth at the hub).
            hf_path_arg = source_path
            if resolved_hf_repo is None:
                resolved_hf_repo = source_ref  # purely for display / logs
        elif resolved_hf_repo is None:
            if source_ref and "/" in source_ref:
                resolved_hf_repo = source_ref
                hf_path_arg = source_ref
            else:
                raise RuntimeError(
                    "Conversion source is not a valid target. Provide a "
                    "`owner/name` Hugging Face repo identifier, a local "
                    "model directory, or a GGUF file."
                )
        else:
            hf_path_arg = resolved_hf_repo

        # Sanity-check the HF repo format when we're actually hitting the
        # hub. Catches the common bug of passing a bare model name (e.g.
        # "GLM-4.7-Flash-MLX-6bit") as a repo id.
        if hf_path_arg and not Path(hf_path_arg).exists() and "/" not in hf_path_arg:
            raise RuntimeError(
                f"'{hf_path_arg}' is not a valid Hugging Face repository "
                f"identifier (expected `owner/name`) and no local path "
                f"with that name exists. If this is a local model, make "
                f"sure the library entry has the correct on-disk path."
            )

        # Fail fast if the resolved HF repo is a GGUF-only mirror (e.g.
        # `mistralai/Devstral-Small-2507_gguf` or anything ending in
        # `-GGUF`). mlx_lm.convert requires `config.json` + safetensors,
        # which GGUF-only repos don't have. Without this check, mlx_lm
        # downloads the snapshot then crashes with
        # `FileNotFoundError: config.json`. Point the user at the base
        # Transformers repo instead.
        def _looks_gguf_only_repo(repo: str) -> bool:
            lowered = repo.lower()
            return lowered.endswith("_gguf") or lowered.endswith("-gguf") or "/gguf-" in lowered

        if (
            hf_path_arg
            and not Path(hf_path_arg).exists()
            and "/" in hf_path_arg
            and _looks_gguf_only_repo(hf_path_arg)
        ):
            base_hint = gguf_metadata.get("baseModelRepo") if gguf_metadata else None
            suggestion = (
                f" Try the base Transformers repo '{base_hint}' instead."
                if base_hint and not _looks_gguf_only_repo(base_hint)
                else ""
            )
            raise RuntimeError(
                f"'{hf_path_arg}' looks like a GGUF-only Hugging Face repo, "
                f"but MLX conversion needs the original Transformers checkpoint "
                f"(config.json + safetensors). GGUF repos only contain quantised "
                f"weights and cannot be re-converted to MLX.{suggestion}"
            )

        # Pre-flight architecture check. mlx_lm.convert will happily spend
        # 5+ minutes downloading 20+GB of weights before discovering the
        # model's architecture isn't supported. Catch it first by reading
        # config.json (cheap: one file, ~few KB) and matching model_type
        # against the set of supported model modules in the installed
        # mlx_lm version.
        preflight_model_type = _peek_hf_model_type(hf_path_arg, convert_env=os.environ.copy())
        if preflight_model_type:
            supported = _mlx_lm_supported_model_types(self.capabilities.pythonExecutable)
            if supported is not None and preflight_model_type not in supported:
                nearest = _nearest_supported_arch(preflight_model_type, supported)
                hint = f" The closest supported variant is '{nearest}'." if nearest else ""
                raise RuntimeError(
                    f"mlx-lm {self.capabilities.mlxLmVersion or 'installed'} does "
                    f"not support architecture '{preflight_model_type}'. Update "
                    f"mlx-lm (pip install -U mlx-lm) or pick a model with a "
                    f"supported architecture.{hint}"
                )

        # Resolve the output path to an ABSOLUTE location. A bare name like
        # "TESTCONVERSION-foo" would otherwise be relative to the backend's
        # cwd, which is the embedded-runtime extraction dir under $TMPDIR
        # — that gets purged on reboot. Bare names land under ~/Models so
        # they survive, ~ gets expanded, and any explicit absolute path
        # is left alone.
        if output_path:
            candidate = Path(output_path).expanduser()
            if not candidate.is_absolute():
                candidate = Path.home() / "Models" / output_path
            target_output = str(candidate.resolve(strict=False))
        else:
            target_output = _default_conversion_output(Path(source_label).stem)

        command = [
            self.capabilities.pythonExecutable,
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            hf_path_arg,
            "--mlx-path",
            target_output,
        ]
        if quantize:
            command.append("--quantize")
            command.extend(["--q-bits", str(q_bits)])
            command.extend(["--q-group-size", str(q_group_size)])
        if dtype:
            command.extend(["--dtype", dtype])

        convert_env = os.environ.copy()
        for _tok_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            _tok_val = os.environ.get(_tok_var)
            if _tok_val:
                convert_env[_tok_var] = _tok_val

        try:
            completed = subprocess.run(
                command,
                cwd=str(WORKSPACE_ROOT),
                check=False,
                capture_output=True,
                text=True,
                timeout=3600,
                env=convert_env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(str(exc)) from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            combined = ((completed.stderr or "") + "\n" + (completed.stdout or "")).lower()
            # Specific, low-false-positive markers only. The previous
            # version matched "token" anywhere, which triggered on every
            # tokenizer-related traceback and masked the real error.
            gated_markers = (
                "gatedrepoerror",
                "cannot access gated repo",
                "is a gated repository",
                "access to model",
                "access this repository",
                "401 client error",
                "403 client error",
                "unauthorized",
            )
            notfound_markers = (
                "repositorynotfounderror",
                "404 client error",
                "not found",
                "does not exist on",
            )
            safetensor_markers = (
                "no safetensors found",
                "no model.safetensors",
            )
            exists_markers = (
                "cannot save to the path",
                "already exists",
            )
            if any(marker in combined for marker in gated_markers):
                raise RuntimeError(
                    f"This model is gated on Hugging Face. Accept the licence at "
                    f"https://huggingface.co/{resolved_hf_repo} and set HF_TOKEN in Settings, then retry."
                )
            if any(marker in combined for marker in notfound_markers):
                raise RuntimeError(
                    f"Hugging Face repository not found: {resolved_hf_repo}. "
                    f"Check the spelling / owner prefix, or provide a local path instead."
                )
            if any(marker in combined for marker in safetensor_markers):
                raise RuntimeError(
                    f"{resolved_hf_repo} has no safetensors weights available — "
                    f"mlx_lm can only convert from safetensors. Pick a different "
                    f"source (e.g. the upstream BF16 repo), not a GGUF-only or "
                    f"MLX-only mirror."
                )
            if any(marker in combined for marker in exists_markers):
                raise RuntimeError(
                    f"Output path already exists: {target_output}. Delete it "
                    f"or choose a different Output path and retry."
                )
            raise RuntimeError(detail or "mlx_lm.convert failed.")

        return {
            "sourceRef": source_ref,
            "sourcePath": source_path,
            "sourceLabel": Path(source_label).name,
            "hfRepo": resolved_hf_repo,
            "outputPath": target_output,
            "quantize": quantize,
            "qBits": q_bits,
            "qGroupSize": q_group_size,
            "dtype": dtype,
            "sourceSizeGb": _bytes_to_gb(_path_size_bytes(source_path)) if source_path else None,
            "outputSizeGb": _bytes_to_gb(_path_size_bytes(target_output)),
            "ggufMetadata": gguf_metadata,
            "log": (completed.stdout or "").strip(),
        }

    def status(self, *, active_requests: int = 0, requests_served: int = 0) -> dict[str, Any]:
        self.prune_stale_backend_children()
        self._expire_orphan_records()
        # Strip the internal monotonic timestamp before sending to the UI.
        public_orphans = [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in self._recent_orphaned_workers
        ]
        return {
            "state": "loaded" if self.loaded_model is not None else "idle",
            "engine": self.engine.engine_name,
            "engineLabel": self.engine.engine_label,
            "loadedModel": self.loaded_model.to_dict() if self.loaded_model is not None else None,
            "warmModels": self.warm_models(),
            "supportsGeneration": True,
            "serverReady": self.loaded_model is not None,
            "activeRequests": active_requests,
            "requestsServed": requests_served,
            "runtimeNote": self.runtime_note,
            "nativeBackends": self.capabilities.to_dict(),
            "recentOrphanedWorkers": public_orphans,
        }
