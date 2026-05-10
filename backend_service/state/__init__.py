from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from backend_service.catalog import CATALOG
from backend_service.inference import RuntimeController
from backend_service.state import benchmarks as _benchmarks
from backend_service.state import documents as _docs
from backend_service.state import downloads as _downloads
from backend_service.state import generation as _generation
from backend_service.state import metrics as _metrics
from backend_service.state import openai_compat as _openai
from backend_service.state import payloads as _payloads
from backend_service.state import sessions as _sessions
from backend_service.state import settings_state as _settings
from backend_service.state._helpers import (
    _CATALOG_REF_ALIASES,
    _TITLE_LEADING_PATTERNS,
    _build_history_with_reasoning,
    _build_sampler_overrides,
    _clean_prompt_for_title,
    _compose_chat_system_prompt,
    _legacy_title_from_prompt,
    _normalize_remote_provider_api_base,
    _read_text_tail,
    _spawn_snapshot_download,
    _title_from_prompt,
    _title_variant_pattern,
)
from backend_service.state.logs import LogManager, _time_label

if TYPE_CHECKING:
    from backend_service.image_runtime import ImageRuntimeManager
    from backend_service.video_runtime import VideoRuntimeManager
from backend_service.models import (
    LoadModelRequest,
    ConvertModelRequest,
    UpdateSessionRequest,
    GenerateRequest,
    OpenAIChatCompletionRequest,
    OpenAIEmbeddingsRequest,
    BenchmarkRunRequest,
    UpdateSettingsRequest,
)
from backend_service.helpers.system import (
    _best_fit_recommendation,
    _describe_process,
    _get_disk_usage_for_models,
)
from backend_service.helpers.discovery import (
    _discover_local_models,
    _model_family_payloads,
)
from backend_service.helpers.huggingface import (
    _known_repo_size_gb,
    _HF_REPO_PATTERN,
    _hf_repo_cache_dir,
)
from backend_service.helpers.images import (
    _image_download_validation_error,
    _friendly_image_download_error,
    _image_repo_allow_patterns,
)
from backend_service.helpers.video import (
    _video_download_validation_error,
    _video_repo_allow_patterns,
)
from backend_service.helpers.settings import (
    _save_data_location,
    _migrate_data_directory,
    _normalize_model_directories,
    _normalize_launch_preferences,
)
from backend_service.helpers.persistence import (
    _default_chat_variant,
    _library_cache_fingerprint,
    _load_library_cache,
    _save_library_cache,
    MAX_BENCHMARK_RUNS,
)
from backend_service.model_resolution import infer_hf_repo_from_local_path
from backend_service.helpers.documents import (
    _sanitize_filename,
    _extract_text_from_file,
    _chunk_text,
    _retrieve_relevant_chunks,
)
from backend_service.helpers.formatting import (
    _context_label,
    _parse_context_label,
    _benchmark_label,
    _bytes_to_gb,
)
from backend_service.helpers.network import (
    _local_ipv4_addresses,
)

class ChaosEngineState:
    def __init__(
        self,
        *,
        system_snapshot_provider=None,
        library_provider=None,
        server_port: int | None = None,
        settings_path: Path | None = None,
        benchmarks_path: Path | None = None,
        chat_sessions_path: Path | None = None,
        library_cache_path: Path | None = None,
        background_capability_probe: bool = False,
    ) -> None:
        # Defer imports of module-level constants to avoid circular imports
        from backend_service.app import (
            _build_system_snapshot,
            _load_settings,
            _load_chat_sessions,
            _load_benchmark_runs,
            DEFAULT_PORT,
            SETTINGS_PATH,
            BENCHMARKS_PATH,
            CHAT_SESSIONS_PATH,
            LIBRARY_CACHE_PATH,
        )

        self._lock = RLock()
        self._system_snapshot_provider = system_snapshot_provider or _build_system_snapshot
        self._library_provider = library_provider
        self.server_port = server_port if server_port is not None else DEFAULT_PORT
        self._settings_path = settings_path if settings_path is not None else SETTINGS_PATH
        self._benchmarks_path = benchmarks_path if benchmarks_path is not None else BENCHMARKS_PATH
        self._library_cache_path = library_cache_path if library_cache_path is not None else LIBRARY_CACHE_PATH
        self.settings = _load_settings(self._settings_path)
        self._library_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._library_scan_started: bool = False
        self._library_scan_done: threading.Event = threading.Event()
        self._library_scan_generation: int = 0
        self._library_scan_threads: list[threading.Thread] = []
        self._library_fingerprint: dict[str, float] = {}
        if library_provider is None:
            cached_payload = _load_library_cache(self._library_cache_path)
            if cached_payload is not None:
                fingerprint = _library_cache_fingerprint(self.settings["modelDirectories"])
                stored_fingerprint = cached_payload.get("fingerprint") or {}
                self._library_fingerprint = {str(k): float(v) for k, v in stored_fingerprint.items()}
                items = cached_payload.get("items") or []
                self._library_cache = (float(cached_payload.get("scannedAt") or time.time()), items)
                if self._library_fingerprint == fingerprint:
                    self._library_scan_done.set()
        else:
            self._library_scan_done.set()
        self.runtime = RuntimeController(background_probe=background_capability_probe)
        self._image_runtime: "ImageRuntimeManager | None" = None
        self._video_runtime: "VideoRuntimeManager | None" = None
        self._chat_sessions_path = chat_sessions_path if chat_sessions_path is not None else CHAT_SESSIONS_PATH
        loaded_sessions = _load_chat_sessions(self._chat_sessions_path)
        self.chat_sessions = loaded_sessions
        if self._normalize_auto_generated_session_titles():
            self._persist_sessions()
        self.benchmark_runs = _load_benchmark_runs(self._benchmarks_path)
        self._log_manager = LogManager()
        self.requests_served = 0
        self.active_requests = 0
        self._loading_state: dict[str, Any] | None = None
        self._downloads: dict[str, dict[str, Any]] = {}
        self._download_cancel: dict[str, bool] = {}
        # Cancellation flags for in-flight chat generations, keyed by session id.
        # Set to True via request_cancel_chat(); the streaming loop in
        # generate_stream() checks this flag between events and breaks early.
        # Cleared at the start of each new generation so a stale flag from a
        # prior turn never aborts a fresh request.
        self._chat_cancel: dict[str, bool] = {}
        self._download_processes: dict[str, subprocess.Popen[str]] = {}
        self._download_tokens: dict[str, str] = {}
        self._bootstrap()

    @property
    def image_runtime(self) -> "ImageRuntimeManager":
        if self._image_runtime is None:
            with self._lock:
                if self._image_runtime is None:
                    from backend_service.image_runtime import ImageRuntimeManager
                    self._image_runtime = ImageRuntimeManager()
        return self._image_runtime

    @image_runtime.setter
    def image_runtime(self, value: "ImageRuntimeManager | None") -> None:
        self._image_runtime = value

    @property
    def video_runtime(self) -> "VideoRuntimeManager":
        if self._video_runtime is None:
            with self._lock:
                if self._video_runtime is None:
                    from backend_service.video_runtime import VideoRuntimeManager
                    self._video_runtime = VideoRuntimeManager()
        return self._video_runtime

    @video_runtime.setter
    def video_runtime(self, value: "VideoRuntimeManager | None") -> None:
        self._video_runtime = value

    def _launch_preferences(self) -> dict[str, Any]:
        return dict(self.settings["launchPreferences"])

    def _library(self, *, force: bool = False) -> list[dict[str, Any]]:
        if self._library_provider is not None:
            return self._library_provider()
        if force:
            with self._lock:
                self._library_scan_generation += 1
                generation = self._library_scan_generation
                directories_snapshot = [dict(item) for item in self.settings["modelDirectories"]]
            fingerprint = _library_cache_fingerprint(directories_snapshot)
            library = _discover_local_models(directories_snapshot)
            with self._lock:
                if generation == self._library_scan_generation:
                    self._library_cache = (time.time(), library)
                    self._library_fingerprint = fingerprint
                    self._persist_library_cache(library, fingerprint)
                self._library_scan_done.set()
                self._library_scan_started = False
            return library
        if self._library_cache is not None:
            return self._library_cache[1]
        if self._library_scan_started:
            return []
        return self._library(force=True)

    def _persist_library_cache(
        self,
        library: list[dict[str, Any]],
        fingerprint: dict[str, float],
    ) -> None:
        try:
            _save_library_cache(library, fingerprint, self._library_cache_path)
        except OSError as exc:
            self.add_log("library", "warn", f"Failed to persist library cache: {exc}")

    def _kick_library_scan(self, *, force: bool = False) -> None:
        if self._library_provider is not None:
            return
        with self._lock:
            if self._library_scan_started and not force:
                return
            self._library_scan_started = True
            self._library_scan_generation += 1
            generation = self._library_scan_generation
            directories_snapshot = [dict(item) for item in self.settings["modelDirectories"]]
            if self._library_cache is None:
                self._library_scan_done.clear()
        thread = threading.Thread(
            target=self._scan_library_into_cache,
            args=(directories_snapshot, generation),
            name="chaosengine-library-scan",
            daemon=True,
        )
        thread.start()
        with self._lock:
            self._library_scan_threads = [
                t for t in self._library_scan_threads if t.is_alive()
            ]
            self._library_scan_threads.append(thread)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            threads = list(self._library_scan_threads)
        for t in threads:
            if t.is_alive():
                t.join(timeout=timeout)

    def _scan_library_into_cache(
        self,
        directories_snapshot: list[dict[str, Any]],
        generation: int,
    ) -> None:
        try:
            fingerprint = _library_cache_fingerprint(directories_snapshot)
            library = _discover_local_models(directories_snapshot)
        except Exception as exc:
            self.add_log("library", "error", f"Library scan failed: {exc}")
            with self._lock:
                if generation == self._library_scan_generation:
                    self._library_scan_started = False
                    self._library_scan_done.set()
            return
        with self._lock:
            if generation != self._library_scan_generation:
                return
            self._library_cache = (time.time(), library)
            self._library_fingerprint = fingerprint
            self._library_scan_started = False
            self._persist_library_cache(library, fingerprint)
            self._library_scan_done.set()
        self.add_log("library", "info", f"Discovered {len(library)} local model entries.")
        self.add_activity(
            "Library scan completed",
            f"{len(library)} local entries found across configured model directories.",
        )

    def _settings_payload(self, library: list[dict[str, Any]]) -> dict[str, Any]:
        return _settings.settings_payload(self, library)

    def _system_snapshot(self) -> dict[str, Any]:
        try:
            return self._system_snapshot_provider(capabilities=self.runtime.capabilities)
        except TypeError:
            return self._system_snapshot_provider()

    def _bootstrap(self) -> None:
        from backend_service.app import app_version

        system = self._system_snapshot()
        recommendation = _best_fit_recommendation(system)
        self.add_log("chaosengine", "info", f"Workspace booted in {system['backendLabel']} mode.")
        self.add_log("chaosengine", "info", f"ChaosEngine v{app_version} detected.")
        self.add_activity("Hardware profile refreshed", recommendation["title"])
        self.add_activity(
            "Backend readiness",
            " / ".join(
                [
                    f"MLX installed: {'yes' if system['mlxAvailable'] else 'no'}",
                    f"mlx-lm installed: {'yes' if system['mlxLmAvailable'] else 'no'}",
                    f"MLX usable: {'yes' if system.get('mlxUsable') else 'no'}",
                    f"GGUF runtime: {'yes' if system.get('ggufAvailable') else 'no'}",
                ]
            ),
        )
        self._kick_library_scan()

    @staticmethod
    def _time_label() -> str:
        return _time_label()

    @property
    def logs(self) -> deque[dict[str, Any]]:
        return self._log_manager.logs

    @property
    def activity(self) -> deque[dict[str, Any]]:
        return self._log_manager.activity

    def add_log(self, source: str, level: str, message: str) -> None:
        self._log_manager.add_log(source, level, message)

    def subscribe_logs(self):
        return self._log_manager.subscribe()

    def unsubscribe_logs(self, q) -> None:
        self._log_manager.unsubscribe(q)

    def add_activity(self, title: str, detail: str) -> None:
        self._log_manager.add_activity(title, detail)

    # Cache + profile + metrics helpers — pure delegations to ``state.metrics``.
    # The methods stay on the class so internal call sites that go through
    # ``self._cache_label(...)`` etc. don't need to be touched.

    def _cache_strategy_label(self, bits: int, fp16_layers: int) -> str:
        return _metrics.native_cache_strategy_label(bits, fp16_layers)

    @staticmethod
    def _native_cache_label() -> str:
        return _metrics.native_cache_strategy_label(0, 0)

    def _cache_label(self, *, cache_strategy: str, bits: int, fp16_layers: int) -> str:
        return _metrics.cache_label(cache_strategy=cache_strategy, bits=bits, fp16_layers=fp16_layers)

    def _loaded_model_metrics_fields(self) -> dict[str, Any]:
        return _metrics.loaded_model_metrics_fields(self.runtime)

    def _requested_runtime_metrics_fields(
        self,
        *,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fit_model_in_memory: bool,
        speculative_decoding: bool,
        tree_budget: int,
    ) -> dict[str, Any]:
        return _metrics.requested_runtime_metrics_fields(
            cache_strategy=cache_strategy,
            cache_bits=cache_bits,
            fp16_layers=fp16_layers,
            fit_model_in_memory=fit_model_in_memory,
            speculative_decoding=speculative_decoding,
            tree_budget=tree_budget,
        )

    def _result_runtime_metrics_fields(self, result: Any | None) -> dict[str, Any]:
        return _metrics.result_runtime_metrics_fields(result)

    def _assistant_metrics_payload(
        self,
        result: Any,
        *,
        requested_runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _metrics.assistant_metrics_payload(self.runtime, result, requested_runtime=requested_runtime)

    def _stream_assistant_metrics_payload(
        self,
        *,
        final_chunk: Any,
        tok_s: float,
        response_seconds: float,
        requested_runtime: dict[str, Any] | None = None,
        ttft_seconds: float | None = None,
    ) -> dict[str, Any]:
        return _metrics.stream_assistant_metrics_payload(
            self.runtime,
            final_chunk=final_chunk,
            tok_s=tok_s,
            response_seconds=response_seconds,
            requested_runtime=requested_runtime,
            ttft_seconds=ttft_seconds,
        )

    def _should_reload_for_profile(
        self,
        *,
        model_ref: str | None,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        cache_strategy: str,
        fit_model_in_memory: bool,
        context_tokens: int,
        speculative_decoding: bool,
        tree_budget: int,
    ) -> bool:
        return _metrics.should_reload_for_profile(
            self.runtime,
            model_ref=model_ref,
            cache_bits=cache_bits,
            fp16_layers=fp16_layers,
            fused_attention=fused_attention,
            cache_strategy=cache_strategy,
            fit_model_in_memory=fit_model_in_memory,
            context_tokens=context_tokens,
            speculative_decoding=speculative_decoding,
            tree_budget=tree_budget,
        )

    def _cache_profile_change_reasons(
        self,
        *,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        cache_strategy: str,
    ) -> list[str]:
        return _metrics.cache_profile_change_reasons(
            self.runtime.loaded_model,
            cache_bits=cache_bits,
            fp16_layers=fp16_layers,
            fused_attention=fused_attention,
            cache_strategy=cache_strategy,
        )

    def _runtime_profile_change_reasons(
        self,
        *,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        cache_strategy: str,
        fit_model_in_memory: bool,
        context_tokens: int,
        speculative_decoding: bool,
        tree_budget: int,
    ) -> list[str]:
        return _metrics.runtime_profile_change_reasons(
            self.runtime.loaded_model,
            cache_bits=cache_bits,
            fp16_layers=fp16_layers,
            fused_attention=fused_attention,
            cache_strategy=cache_strategy,
            fit_model_in_memory=fit_model_in_memory,
            context_tokens=context_tokens,
            speculative_decoding=speculative_decoding,
            tree_budget=tree_budget,
        )

    def _append_benchmark_run(self, run: dict[str, Any]) -> None:
        _benchmarks.append_benchmark_run(self, run)

    def _find_catalog_entry(self, model_ref: str) -> dict[str, Any] | None:
        canonical_ref = _CATALOG_REF_ALIASES.get(model_ref, model_ref)
        for entry in CATALOG:
            if (
                entry["id"] == canonical_ref
                or entry["name"] == canonical_ref
                or entry["repo"] == canonical_ref
                or entry["link"] == canonical_ref
                or entry["id"] == model_ref
                or entry["name"] == model_ref
                or entry["repo"] == model_ref
                or entry["link"] == model_ref
            ):
                return entry
        return None

    def _find_library_entry(self, path: str | None, model_ref: str | None) -> dict[str, Any] | None:
        if path is None and model_ref is None:
            return None
        if (
            self._library_provider is None
            and self._library_cache is None
            and not self._library_scan_done.is_set()
        ):
            self._library_scan_done.wait(timeout=10.0)
        for entry in self._library():
            if path and entry["path"] == path:
                return entry
            if model_ref and entry["name"] == model_ref:
                return entry
        return None

    def _resolve_canonical_repo(
        self,
        *,
        model_ref: str | None,
        path: str | None,
        canonical_repo: str | None,
    ) -> str | None:
        from backend_service.app import _hf_repo_from_link

        cleaned = str(canonical_repo or "").strip() or None
        if cleaned is not None:
            return cleaned
        catalog_entry = self._find_catalog_entry(model_ref) if model_ref else None
        if catalog_entry is not None:
            return (
                _hf_repo_from_link(catalog_entry.get("link"))
                or str(catalog_entry.get("repo") or "").strip()
                or None
            )
        return infer_hf_repo_from_local_path(path)

    def _resolve_model_target(
        self,
        *,
        model_ref: str | None,
        path: str | None,
        backend: str,
    ) -> tuple[str | None, str]:
        from backend_service.app import _hf_repo_from_link

        if model_ref in _CATALOG_REF_ALIASES:
            model_ref = _CATALOG_REF_ALIASES[model_ref]
        resolved_backend = backend
        runtime_target = path
        explicit_gguf_path = bool(path and path.lower().endswith(".gguf"))
        catalog_entry = self._find_catalog_entry(model_ref) if model_ref else None
        library_entry = self._find_library_entry(path, model_ref)

        if explicit_gguf_path:
            runtime_target = path
            if backend == "auto":
                resolved_backend = "llama.cpp"
            return runtime_target, resolved_backend

        if catalog_entry is not None:
            runtime_target = _hf_repo_from_link(catalog_entry.get("link")) or runtime_target or model_ref
            if backend == "auto":
                resolved_backend = "llama.cpp" if catalog_entry.get("format") == "GGUF" else "mlx"
        elif library_entry is not None:
            lib_format = library_entry.get("format", "")
            lib_name = library_entry.get("name", "")
            lib_path = library_entry.get("path", "")
            lib_source_kind = library_entry.get("sourceKind", "")
            is_gguf = lib_format == "GGUF" or "gguf" in lib_name.lower() or "gguf" in lib_path.lower()
            if backend == "auto":
                resolved_backend = "llama.cpp" if is_gguf else "mlx"
            if lib_source_kind == "HF cache":
                runtime_target = library_entry["path"] if is_gguf else library_entry["name"]
            else:
                runtime_target = runtime_target or library_entry["path"]
        elif path and path.lower().endswith(".gguf") and backend == "auto":
            resolved_backend = "llama.cpp"

        # Last-resort GGUF detection
        if resolved_backend in {"auto", "mlx"}:
            haystack = " ".join(
                str(value).lower()
                for value in (runtime_target, model_ref, path)
                if value
            )
            if "gguf" in haystack:
                resolved_backend = "llama.cpp"

        return runtime_target or model_ref, resolved_backend

    def _default_session_model(self) -> dict[str, Any]:
        return _sessions.default_session_model(self)

    def _promote_session(self, session: dict[str, Any]) -> None:
        _sessions.promote_session(self, session)

    def _persist_sessions(self) -> None:
        _sessions.persist_sessions(self)

    def _unique_session_title(self, base_title: str, *, exclude_session_id: str | None = None) -> str:
        return _sessions.unique_session_title(self, base_title, exclude_session_id=exclude_session_id)

    def _auto_session_title(self, prompt: str | None, *, exclude_session_id: str | None = None) -> str:
        return _sessions.auto_session_title(self, prompt, exclude_session_id=exclude_session_id)

    def _normalize_auto_generated_session_titles(self) -> bool:
        return _sessions.normalize_auto_generated_session_titles(self)

    def _ensure_session(self, session_id: str | None = None, title: str | None = None) -> dict[str, Any]:
        return _sessions.ensure_session(self, session_id=session_id, title=title)

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        return _sessions.create_session(self, title)

    def add_message_variant(
        self,
        session_id: str,
        message_index: int,
        model_ref: str,
        model_name: str,
        canonical_repo: str | None,
        source: str,
        path: str | None,
        backend: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        return _sessions.add_message_variant(
            self,
            session_id,
            message_index,
            model_ref,
            model_name,
            canonical_repo,
            source,
            path,
            backend,
            max_tokens,
            temperature,
        )

    def delve_message(
        self,
        session_id: str,
        message_index: int,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> dict[str, Any]:
        return _sessions.delve_message(self, session_id, message_index, max_tokens, temperature)

    def fork_session(
        self,
        source_session_id: str,
        fork_at_message_index: int,
        title: str | None = None,
    ) -> dict[str, Any]:
        return _sessions.fork_session(self, source_session_id, fork_at_message_index, title)

    def update_session(self, session_id: str, request: UpdateSessionRequest) -> dict[str, Any]:
        return _sessions.update_session(self, session_id, request)

    def update_settings(self, request: UpdateSettingsRequest) -> dict[str, Any]:
        return _settings.update_settings(self, request)

    def _conversion_details(
        self,
        *,
        request: ConvertModelRequest,
        conversion: dict[str, Any],
    ) -> dict[str, Any]:
        from backend_service.app import compute_cache_preview

        library_entry = self._find_library_entry(request.path, request.modelRef)
        catalog_entry = self._find_catalog_entry(request.modelRef or conversion.get("hfRepo") or "")
        params_b = float(catalog_entry.get("paramsB")) if catalog_entry and catalog_entry.get("paramsB") is not None else None
        launch_preferences = self._launch_preferences()

        preview = (
            compute_cache_preview(
                bits=launch_preferences["cacheBits"],
                fp16_layers=launch_preferences["fp16Layers"],
                context_tokens=launch_preferences["contextTokens"],
                params_b=params_b,
                system_stats=self._system_snapshot(),
            )
            if params_b is not None
            else None
        )

        gguf_metadata = conversion.get("ggufMetadata") or {}
        context_length = gguf_metadata.get("contextLength")
        context_window = (
            _context_label(int(context_length))
            if context_length
            else (catalog_entry.get("contextWindow") if catalog_entry is not None else None)
        )

        return {
            **conversion,
            "sourceFormat": library_entry.get("format") if library_entry is not None else (catalog_entry.get("format") if catalog_entry is not None else None),
            "sourceSizeGb": conversion.get("sourceSizeGb") or (library_entry.get("sizeGb") if library_entry is not None else None),
            "paramsB": params_b,
            "contextWindow": context_window,
            "architecture": gguf_metadata.get("architecture") or gguf_metadata.get("name"),
            "estimatedTokS": preview["estimatedTokS"] if preview is not None else None,
            "baselineCacheGb": preview["baselineCacheGb"] if preview is not None else None,
            "optimizedCacheGb": preview["optimizedCacheGb"] if preview is not None else None,
            "compressionRatio": preview["compressionRatio"] if preview is not None else None,
            "qualityPercent": preview["qualityPercent"] if preview is not None else None,
        }

    def run_benchmark(self, request: BenchmarkRunRequest) -> dict[str, Any]:
        return _benchmarks.run_benchmark(self, request)

    def load_model(
        self,
        request: LoadModelRequest,
        *,
        keep_warm_previous: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            catalog_entry = self._find_catalog_entry(request.modelRef)
            library_entry = self._find_library_entry(request.path, request.modelRef)
            effective_canonical_repo = self._resolve_canonical_repo(
                model_ref=request.modelRef,
                path=request.path,
                canonical_repo=request.canonicalRepo,
            )
            # Reject load requests for models we can't locate. Without this,
            # llama-server's built-in HuggingFace fallback kicks in and tries to
            # fetch the weights at load time, which on Windows fails with an
            # opaque SSL error (the bundled llama-server.exe ships without a
            # CA bundle) and in any case isn't a UX we want to encourage — a
            # pull-through download hides from the Library scan, bypasses the
            # download-progress UI, and surprises users with invisible disk use.
            #
            # Allowed shapes:
            #   - library_entry set → scanner found the model in the library
            #   - request.path is provided → operator pointed us at a custom
            #     location. We trust the caller and let llama.cpp / MLX fail
            #     fast with a clear "path not found" error if it's wrong,
            #     rather than second-guessing non-existent mount points here.
            model_ref_str = (request.modelRef or "").strip()
            has_path = bool((request.path or "").strip())
            if model_ref_str and library_entry is None and not has_path:
                raise RuntimeError(
                    f"Model '{model_ref_str}' isn't downloaded on this machine. "
                    "Open the Discover tab and download it first, then try loading again."
                )
            if library_entry is not None and library_entry.get("broken"):
                reason = library_entry.get("brokenReason") or "incomplete or corrupt"
                raise RuntimeError(
                    f"Cannot load '{library_entry.get('name') or request.modelRef}': {reason}."
                )
            detected_max: int | None = None
            if library_entry is not None:
                detected_max = library_entry.get("maxContext")
            if detected_max is None and catalog_entry is not None:
                detected_max = _parse_context_label(catalog_entry.get("contextWindow"))
            if detected_max is not None and request.contextTokens > detected_max:
                self.add_log(
                    "runtime",
                    "warning",
                    f"Requested context {request.contextTokens} exceeds model max {detected_max}; clamping.",
                )
                try:
                    request.contextTokens = int(detected_max)
                except Exception:
                    pass
            model_name = request.modelName
            if model_name is None and catalog_entry is not None:
                model_name = catalog_entry["name"]
            if model_name is None and library_entry is not None:
                model_name = library_entry["name"]
            runtime_target, resolved_backend = self._resolve_model_target(
                model_ref=request.modelRef,
                path=request.path,
                backend=request.backend,
            )
            display_name = model_name or request.modelRef
            speculative_decoding = bool(getattr(request, "speculativeDecoding", False))
            tree_budget = int(getattr(request, "treeBudget", 0) or 0)

            # When speculative decoding is active, force native cache strategy
            # because DFLASH manages its own KV caches with rollback, which
            # conflicts with compression strategies.
            effective_cache_strategy = request.cacheStrategy
            effective_cache_bits = request.cacheBits
            effective_fp16_layers = request.fp16Layers
            if speculative_decoding:
                effective_cache_strategy = "native"
                effective_cache_bits = 0
                effective_fp16_layers = 0
            exclusive_memory_load = bool(speculative_decoding and resolved_backend == "mlx")

            same_loaded_model = (
                self.runtime.loaded_model is not None
                and (
                    request.modelRef in {
                        self.runtime.loaded_model.ref,
                        self.runtime.loaded_model.runtimeTarget,
                    }
                    or (request.path is not None and request.path == self.runtime.loaded_model.path)
                )
            )
            cache_profile_changes = (
                self._cache_profile_change_reasons(
                    cache_bits=effective_cache_bits,
                    fp16_layers=effective_fp16_layers,
                    fused_attention=request.fusedAttention,
                    cache_strategy=effective_cache_strategy,
                )
                if same_loaded_model
                else []
            )
            profile_changes = (
                self._runtime_profile_change_reasons(
                    cache_bits=effective_cache_bits,
                    fp16_layers=effective_fp16_layers,
                    fused_attention=request.fusedAttention,
                    cache_strategy=effective_cache_strategy,
                    fit_model_in_memory=request.fitModelInMemory,
                    context_tokens=request.contextTokens,
                    speculative_decoding=speculative_decoding,
                    tree_budget=tree_budget,
                )
                if same_loaded_model
                else []
            )
            reload_required_changes = [change for change in profile_changes if change not in cache_profile_changes]
            can_apply_profile_without_reload = bool(
                same_loaded_model
                and cache_profile_changes
                and not reload_required_changes
                and resolved_backend == "mlx"
                and self.runtime.loaded_model is not None
                and self.runtime.loaded_model.engine == "mlx"
            )
            if same_loaded_model and not profile_changes:
                if effective_canonical_repo is not None and self.runtime.loaded_model is not None:
                    self.runtime.loaded_model.canonicalRepo = effective_canonical_repo
                return self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served)
            self._loading_state = {
                "modelName": display_name,
                "stage": "applying" if can_apply_profile_without_reload else "loading",
                "startedAt": time.time(),
                "progress": None,
                "progressPercent": None,
                "progressPhase": None,
                "progressMessage": None,
                "recentLogLines": [],
            }
            if can_apply_profile_without_reload:
                self.add_log(
                    "runtime",
                    "info",
                    f"Applying MLX runtime profile for {display_name} ({', '.join(cache_profile_changes)}).",
                )
            elif profile_changes:
                self.add_log(
                    "runtime",
                    "info",
                    f"Reloading {display_name} because launch settings changed ({', '.join(profile_changes)}).",
                )
            else:
                self.add_log("runtime", "info", f"Loading {display_name}...")

        def _on_load_progress(prog: dict[str, Any]) -> None:
            try:
                with self._lock:
                    if self._loading_state is None:
                        return
                    percent = prog.get("percent")
                    phase = prog.get("phase")
                    message = prog.get("message")
                    self._loading_state["progressPercent"] = percent
                    self._loading_state["progressPhase"] = phase
                    self._loading_state["progressMessage"] = message
                    self._loading_state["progress"] = percent
                    if message or phase:
                        line = f"[{phase}] {message}" if phase and message else str(message or phase)
                        tail = list(self._loading_state.get("recentLogLines") or [])
                        tail.append(line)
                        if len(tail) > 5:
                            tail = tail[-5:]
                        self._loading_state["recentLogLines"] = tail
            except Exception:
                pass

        try:
            if can_apply_profile_without_reload:
                loaded = self.runtime.update_profile(
                    canonical_repo=effective_canonical_repo,
                    cache_strategy=effective_cache_strategy,
                    cache_bits=effective_cache_bits,
                    fp16_layers=effective_fp16_layers,
                    fused_attention=request.fusedAttention,
                )
            else:
                warm_model_count = len(self.runtime.warm_models())
                if exclusive_memory_load and warm_model_count > 0:
                    cleared = self.runtime.clear_warm_pool()
                    if cleared > 0:
                        self.add_log(
                            "runtime",
                            "info",
                            f"Cleared {cleared} warm model{'s' if cleared != 1 else ''} before speculative MLX load.",
                        )
                loaded = self.runtime.load_model(
                    model_ref=request.modelRef,
                    model_name=model_name,
                    canonical_repo=effective_canonical_repo,
                    source=request.source,
                    backend=resolved_backend,
                    path=request.path,
                    runtime_target=runtime_target,
                    cache_strategy=effective_cache_strategy,
                    cache_bits=effective_cache_bits,
                    fp16_layers=effective_fp16_layers,
                    fused_attention=request.fusedAttention,
                    fit_model_in_memory=request.fitModelInMemory,
                    context_tokens=request.contextTokens,
                    speculative_decoding=speculative_decoding,
                    tree_budget=tree_budget,
                    keep_warm_previous=keep_warm_previous and not exclusive_memory_load,
                    progress_callback=_on_load_progress,
                )
        except Exception:
            with self._lock:
                self._loading_state = None
            raise

        with self._lock:
            self._loading_state = None
            loaded_cache_label = self._cache_label(
                cache_strategy=str(loaded.cacheStrategy),
                bits=int(loaded.cacheBits),
                fp16_layers=int(loaded.fp16Layers),
            )
            self.add_log("runtime", "info", f"Model loaded: {loaded.name} via {loaded.engine}.")
            self.add_activity("Model loaded", f"{loaded.name} / {loaded_cache_label}")
            return self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served)

    def unload_model(self, ref: str | None = None) -> dict[str, Any]:
        with self._lock:
            if ref:
                if self.runtime.loaded_model and ref in {
                    self.runtime.loaded_model.ref,
                    self.runtime.loaded_model.runtimeTarget,
                    self.runtime.loaded_model.path,
                    self.runtime.loaded_model.name,
                }:
                    name = self.runtime.loaded_model.name
                    self.runtime.unload_model()
                    self.add_log("runtime", "info", f"Model unloaded: {name}.")
                    self.add_activity("Model unloaded", name)
                else:
                    unloaded = self.runtime.unload_warm_model_by_ref(ref)
                    if unloaded:
                        self.add_log("runtime", "info", f"Warm model unloaded: {ref}.")
                        self.add_activity("Warm model unloaded", ref)
                    else:
                        self.add_log("runtime", "info", f"Unload no-op: {ref} not found.")
            else:
                name = self.runtime.loaded_model.name if self.runtime.loaded_model else "No model"
                self.runtime.unload_model()
                self.add_log("runtime", "info", f"Model unloaded: {name}.")
                self.add_activity("Model unloaded", name)
            return self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served)

    def convert_model(self, request: ConvertModelRequest) -> dict[str, Any]:
        with self._lock:
            runtime_target, _ = self._resolve_model_target(
                model_ref=request.modelRef,
                path=request.path,
                backend="auto",
            )
            conversion = self.runtime.convert_model(
                source_ref=runtime_target if request.path is None else request.modelRef,
                source_path=request.path,
                output_path=request.outputPath,
                hf_repo=request.hfRepo,
                quantize=request.quantize,
                q_bits=request.qBits,
                q_group_size=request.qGroupSize,
                dtype=request.dtype,
            )
            conversion = self._conversion_details(request=request, conversion=conversion)
            self.add_log(
                "conversion",
                "info",
                f"Converted {conversion['sourceLabel']} to MLX at {conversion['outputPath']}.",
            )
            self.add_activity("Model converted", f"{conversion['sourceLabel']} -> {Path(conversion['outputPath']).name}")
            return {
                "conversion": conversion,
                "library": self._library(force=True),
                "runtime": self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served),
            }

    def reveal_model_path(self, path: str) -> dict[str, Any]:
        from backend_service.helpers.discovery import _reveal_path_in_file_manager

        with self._lock:
            target = Path(path).expanduser()
            _reveal_path_in_file_manager(target)
            resolved = str(target.resolve())
            self.add_log("library", "info", f"Revealed model path: {resolved}.")
            return {"revealed": resolved}

    def delete_model_path(self, path: str) -> dict[str, Any]:
        """Delete a local model file or directory on disk."""
        with self._lock:
            target = Path(path).expanduser()
            try:
                resolved = target.resolve(strict=True)
            except (OSError, RuntimeError):
                raise HTTPException(status_code=404, detail=f"Path not found: {path}")

            allowed = False
            for directory in self.settings.get("modelDirectories", []):
                if not directory.get("enabled", True):
                    continue
                root_raw = str(directory.get("path") or "").strip()
                if not root_raw:
                    continue
                try:
                    root = Path(os.path.expanduser(root_raw)).resolve()
                except (OSError, RuntimeError):
                    continue
                if resolved == root:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Refusing to delete a configured model directory. "
                            "Only files/subdirectories inside it may be removed."
                        ),
                    )
                try:
                    resolved.relative_to(root)
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Refusing to delete {resolved}: not inside any "
                        f"configured model directory."
                    ),
                )

            try:
                loaded = getattr(self.runtime, "loaded_model", None)
                if loaded and getattr(loaded, "path", None):
                    loaded_resolved = Path(str(loaded.path)).expanduser().resolve()
                    if loaded_resolved == resolved or loaded_resolved.is_relative_to(resolved):
                        self.runtime.unload_model()
            except (OSError, RuntimeError, AttributeError):
                pass

            try:
                if resolved.is_dir() and not resolved.is_symlink():
                    import shutil as _shutil
                    _shutil.rmtree(resolved)
                else:
                    resolved.unlink()
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete {resolved}: {exc}",
                )

            self.add_log("library", "info", f"Deleted model at {resolved}.")
            return {
                "deleted": str(resolved),
                "library": self._library(force=True),
            }

    def _session_docs_dir(self, session_id: str) -> Path:
        return _docs.session_docs_dir(self, session_id)

    def list_documents(self, session_id: str) -> list[dict[str, Any]]:
        return _docs.list_session_documents(self, session_id)

    def upload_document(self, session_id: str, original_name: str, raw_bytes: bytes) -> dict[str, Any]:
        return _docs.upload_session_document(self, session_id, original_name, raw_bytes)

    def delete_document(self, session_id: str, doc_id: str) -> dict[str, Any]:
        return _docs.delete_session_document(self, session_id, doc_id)

    # -- Phase 3.7: workspace knowledge stack helpers --------------------

    def _workspace_dir(self, workspace_id: str) -> Path:
        return _docs.workspace_docs_dir(self, workspace_id)

    def upload_workspace_document(
        self,
        workspace_id: str,
        filename: str,
        data: bytes,
    ) -> dict[str, Any]:
        return _docs.upload_workspace_document(self, workspace_id, filename, data)

    def delete_workspace_document(self, workspace_id: str, doc_id: str) -> dict[str, Any]:
        return _docs.delete_workspace_document(self, workspace_id, doc_id)

    def delete_session(self, session_id: str) -> dict[str, Any]:
        return _sessions.delete_session(self, session_id)

    def _retrieve_session_context(self, session_id: str, prompt: str, top_k: int = 5) -> tuple[str, list[dict[str, Any]]]:
        return _docs.retrieve_session_context(self, session_id, prompt, top_k)

    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        return _generation.generate(self, request)

    def generate_stream(self, request: GenerateRequest):
        return _generation.generate_stream(self, request)

    def start_download(
        self,
        repo: str,
        allow_patterns: list[str] | None = None,
        validation_error_fn: Callable[[str], str | None] | None = None,
    ) -> dict[str, Any]:
        return _downloads.start_download(self, repo, allow_patterns, validation_error_fn)

    def download_status(self) -> list[dict[str, Any]]:
        return _downloads.download_status(self)

    @staticmethod
    def _loaded_model_matches_repo_cache(loaded: Any, repo: str, repo_cache_dir: Path) -> bool:
        return _downloads.loaded_model_matches_repo_cache(loaded, repo, repo_cache_dir)

    def _unload_repo_from_runtimes(self, repo: str, repo_cache_dir: Path) -> None:
        _downloads.unload_repo_from_runtimes(self, repo, repo_cache_dir)

    def request_cancel_chat(self, session_id: str) -> dict[str, Any]:
        """Mark a chat generation for cancellation.

        The streaming loop in generate_stream() checks this flag between
        events and breaks early, persisting whatever output has accumulated
        so far. Returns metadata about whether the session is currently
        generating so the UI can decide whether to show a "stop" toast.
        """
        with self._lock:
            self._chat_cancel[session_id] = True
            session = next(
                (s for s in self.chat_sessions if s.get("id") == session_id),
                None,
            )
            return {
                "sessionId": session_id,
                "cancelled": True,
                "wasActive": session is not None,
            }

    def is_chat_cancel_requested(self, session_id: str) -> bool:
        with self._lock:
            return bool(self._chat_cancel.get(session_id, False))

    def clear_chat_cancel(self, session_id: str) -> None:
        with self._lock:
            self._chat_cancel.pop(session_id, None)

    def cancel_download(self, repo: str) -> dict[str, Any]:
        return _downloads.cancel_download(self, repo)

    def delete_download(self, repo: str) -> dict[str, Any]:
        return _downloads.delete_download(self, repo)

    def server_status(self) -> dict[str, Any]:
        return _payloads.server_status(self)

    def workspace(self) -> dict[str, Any]:
        return _payloads.workspace(self)

    def openai_models(self) -> dict[str, Any]:
        return _openai.openai_models(self)

    def openai_embeddings(self, request: OpenAIEmbeddingsRequest) -> dict[str, Any]:
        return _openai.openai_embeddings(self, request)

    def openai_chat_completion(
        self, request: OpenAIChatCompletionRequest
    ) -> dict[str, Any] | StreamingResponse:
        return _openai.openai_chat_completion(self, request)
