from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from urllib.parse import urlparse
from collections import deque
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from backend_service.catalog import CATALOG
from backend_service.inference import RuntimeController

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

_CATALOG_REF_ALIASES = {
    # The non-it MLX Gemma variant lacks a chat template and is a poor default
    # for the chat picker. Resolve old saved refs to the instruct checkpoint.
    "mlx-community/gemma-4-26b-a4b-5bit": "mlx-community/gemma-4-26b-a4b-it-5bit",
}


def _compose_chat_system_prompt(system_prompt: str | None, thinking_mode: str | None = None) -> str:
    return (system_prompt or "").strip()


def _build_sampler_overrides(request: Any) -> dict[str, Any]:
    """Phase 2.2: collect the request's sampler overrides into a flat dict
    keyed using the llama-server `/v1/chat/completions` field names.

    The dict contains only fields the user actually set — `None` defaults
    are skipped so the backend's defaults stay in force when the UI sends
    no override. Both engines treat unknown keys as no-ops, so the output
    is forward-compatible across llama-server / mlx-lm versions.
    """
    overrides: dict[str, Any] = {}

    def _put(dst: str, value: Any) -> None:
        if value is not None:
            overrides[dst] = value

    _put("top_p", getattr(request, "topP", None))
    _put("top_k", getattr(request, "topK", None))
    _put("min_p", getattr(request, "minP", None))
    _put("repeat_penalty", getattr(request, "repeatPenalty", None))
    _put("seed", getattr(request, "seed", None))
    mirostat_mode = getattr(request, "mirostatMode", None)
    if mirostat_mode is not None:
        overrides["mirostat"] = mirostat_mode
    _put("mirostat_tau", getattr(request, "mirostatTau", None))
    _put("mirostat_eta", getattr(request, "mirostatEta", None))
    return overrides


def _build_history_with_reasoning(
    messages: list[dict[str, Any]],
    *,
    preserve_reasoning: bool,
) -> list[dict[str, Any]]:
    """Project a session's stored messages into the history list passed to the
    inference layer.

    When `preserve_reasoning` is true and an assistant message has a
    `reasoning` field captured by ThinkingTokenFilter on a previous turn,
    the reasoning is re-emitted inside `<think>...</think>` tags ahead of
    the visible answer. Reasoning-capable models (Qwen3, DeepSeek R1, etc.)
    consume this naturally on follow-up turns; non-reasoning models will
    treat it as inline text. Falsy / missing reasoning is skipped, so this
    is safe to call unconditionally.
    """
    history: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        text = str(message.get("text") or "")
        if (
            preserve_reasoning
            and role == "assistant"
            and message.get("reasoning")
        ):
            reasoning_str = str(message["reasoning"]).strip()
            if reasoning_str:
                text = f"<think>\n{reasoning_str}\n</think>\n\n{text}"
        history.append({"role": role, "text": text})
    return history


def _title_from_prompt(prompt: str | None) -> str:
    words = str(prompt or "").strip().split()
    return " ".join(words[:4]) or "New chat"


def _title_variant_pattern(base_title: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(base_title)}(?: \((\d+)\))?$")


def _read_text_tail(path: str | None, *, limit: int = 4096) -> str:
    if not path:
        return ""
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(content) <= limit:
        return content
    return content[-limit:]


def _spawn_snapshot_download(
    repo: str,
    env: dict[str, str],
    log_handle: Any,
    allow_patterns: list[str] | None = None,
) -> subprocess.Popen[str]:
    from backend_service.app import HF_SNAPSHOT_DOWNLOAD_HELPER

    args = [sys.executable, "-c", HF_SNAPSHOT_DOWNLOAD_HELPER, repo]
    # The helper treats an empty string as "no allowlist"; a JSON-encoded
    # list restricts the download to matching files. This is how we keep
    # diffusers video repos from ballooning to hundreds of GB when the
    # repo ships legacy standalone checkpoints alongside the pipeline
    # layout.
    args.append(json.dumps(allow_patterns) if allow_patterns else "")
    return subprocess.Popen(
        args,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def _normalize_remote_provider_api_base(raw_api_base: str) -> str:
    api_base = raw_api_base.strip().rstrip("/")
    parsed = urlparse(api_base)
    hostname = (parsed.hostname or "").lower()
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Remote provider API base must be a valid absolute URL.")
    if parsed.scheme == "https":
        return api_base
    if parsed.scheme == "http" and hostname in {"127.0.0.1", "localhost"}:
        return api_base
    raise HTTPException(
        status_code=400,
        detail="Remote providers must use HTTPS unless they point to localhost.",
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
        self.logs: deque[dict[str, Any]] = deque(maxlen=120)
        self._log_subscribers: list = []
        self.activity: deque[dict[str, Any]] = deque(maxlen=60)
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
        from backend_service.app import DATA_LOCATION

        model_counts: dict[str, int] = {}
        for item in library:
            directory_id = item.get("directoryId")
            if not directory_id:
                continue
            model_counts[directory_id] = model_counts.get(directory_id, 0) + 1

        directories: list[dict[str, Any]] = []
        for directory in self.settings["modelDirectories"]:
            expanded = Path(os.path.expanduser(str(directory.get("path") or ""))).expanduser()
            directories.append(
                {
                    **directory,
                    "exists": expanded.exists(),
                    "modelCount": model_counts.get(directory["id"], 0),
                }
            )

        # Mask API keys when returning to the frontend
        remote_providers = self.settings.get("remoteProviders") or []
        masked_providers = []
        for p in remote_providers:
            api_key = p.get("apiKey", "")
            masked_providers.append({
                "id": p.get("id"),
                "label": p.get("label"),
                "apiBase": p.get("apiBase"),
                "model": p.get("model"),
                "hasApiKey": bool(api_key),
                "apiKeyMasked": ("\u2022" * 8 + api_key[-4:]) if len(api_key) > 4 else "",
            })

        hf_token_value = str(self.settings.get("huggingFaceToken") or "")
        if len(hf_token_value) > 4:
            hf_token_masked = "\u2022" * 8 + hf_token_value[-4:]
        else:
            hf_token_masked = ""

        return {
            "modelDirectories": directories,
            "preferredServerPort": self.settings["preferredServerPort"],
            "allowRemoteConnections": bool(self.settings.get("allowRemoteConnections", False)),
            "requireApiAuth": bool(self.settings.get("requireApiAuth", True)),
            "autoStartServer": bool(self.settings.get("autoStartServer", False)),
            "launchPreferences": self._launch_preferences(),
            "remoteProviders": masked_providers,
            "huggingFaceToken": hf_token_masked,
            "hasHuggingFaceToken": bool(hf_token_value),
            "dataDirectory": str(DATA_LOCATION.data_dir),
            # Per-modality output overrides (empty == use default under
            # dataDirectory). The frontend uses these to render the picker
            # value and the resolved path used for new artifacts.
            "imageOutputsDirectory": str(self.settings.get("imageOutputsDirectory") or ""),
            "videoOutputsDirectory": str(self.settings.get("videoOutputsDirectory") or ""),
            # Hugging Face cache root override. Empty string means "use the
            # platform default" — the frontend renders the resolved path
            # alongside the override input so users always know where
            # models are actually landing.
            "hfCachePath": str(self.settings.get("hfCachePath") or ""),
        }

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
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _relative_label() -> str:
        return time.strftime("%H:%M")

    def add_log(self, source: str, level: str, message: str) -> None:
        import queue as _queue_mod
        entry = {
            "ts": self._time_label(),
            "source": source,
            "level": level,
            "message": message,
        }
        self.logs.appendleft(entry)
        for q in self._log_subscribers:
            try:
                q.put_nowait(entry)
            except _queue_mod.Full:
                pass

    def subscribe_logs(self):
        import queue as _queue_mod
        q = _queue_mod.Queue(maxsize=200)
        self._log_subscribers.append(q)
        return q

    def unsubscribe_logs(self, q) -> None:
        try:
            self._log_subscribers.remove(q)
        except ValueError:
            pass

    def add_activity(self, title: str, detail: str) -> None:
        self.activity.appendleft(
            {
                "time": "Now",
                "title": title,
                "detail": detail,
            }
        )

    def _cache_strategy_label(self, bits: int, fp16_layers: int) -> str:
        if bits and bits < 16:
            return f"Native {bits}-bit {fp16_layers}+{fp16_layers}"
        return "Native f16 cache"

    @staticmethod
    def _native_cache_label() -> str:
        return "Native f16 cache"

    def _cache_label(self, *, cache_strategy: str, bits: int, fp16_layers: int) -> str:
        from cache_compression import registry as cache_registry

        strategy = cache_registry.get(cache_strategy)
        if strategy is not None:
            return strategy.label(bits, fp16_layers)
        return self._cache_strategy_label(bits, fp16_layers)

    def _loaded_model_metrics_fields(self) -> dict[str, Any]:
        loaded = self.runtime.loaded_model
        return {
            "model": loaded.name if loaded else None,
            "modelRef": loaded.ref if loaded else None,
            "canonicalRepo": loaded.canonicalRepo if loaded else None,
            "backend": loaded.backend if loaded else None,
            "engineLabel": self.runtime.engine.engine_label,
            "cacheLabel": self._cache_label(
                cache_strategy=str(loaded.cacheStrategy) if loaded else "native",
                bits=int(loaded.cacheBits) if loaded else 0,
                fp16_layers=int(loaded.fp16Layers) if loaded else 0,
            ),
            "cacheStrategy": loaded.cacheStrategy if loaded else None,
            "cacheBits": loaded.cacheBits if loaded else None,
            "fp16Layers": loaded.fp16Layers if loaded else None,
            "fusedAttention": loaded.fusedAttention if loaded else None,
            "fitModelInMemory": loaded.fitModelInMemory if loaded else None,
            "speculativeDecoding": loaded.speculativeDecoding if loaded else None,
            "dflashDraftModel": loaded.dflashDraftModel if loaded else None,
            "modelSource": loaded.source if loaded else None,
            "modelPath": loaded.path if loaded else None,
            "contextTokens": loaded.contextTokens if loaded else None,
            "treeBudget": loaded.treeBudget if loaded else 0,
            "generatedAt": self._time_label(),
        }

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
        return {
            "requestedCacheLabel": self._cache_label(
                cache_strategy=cache_strategy,
                bits=cache_bits,
                fp16_layers=fp16_layers,
            ),
            "requestedCacheStrategy": cache_strategy,
            "requestedCacheBits": cache_bits,
            "requestedFp16Layers": fp16_layers,
            "requestedFitModelInMemory": fit_model_in_memory,
            "requestedSpeculativeDecoding": speculative_decoding,
            "requestedTreeBudget": tree_budget,
        }

    def _result_runtime_metrics_fields(self, result: Any | None) -> dict[str, Any]:
        if result is None:
            return {}
        metrics: dict[str, Any] = {}
        cache_strategy = getattr(result, "cache_strategy", None)
        if cache_strategy is not None:
            cache_bits = int(getattr(result, "cache_bits", 0) or 0)
            fp16_layers = int(getattr(result, "fp16_layers", 0) or 0)
            metrics.update({
                "cacheLabel": self._cache_label(
                    cache_strategy=str(cache_strategy),
                    bits=cache_bits,
                    fp16_layers=fp16_layers,
                ),
                "cacheStrategy": str(cache_strategy),
                "cacheBits": cache_bits,
                "fp16Layers": fp16_layers,
            })
        speculative_decoding = getattr(result, "speculative_decoding", None)
        if speculative_decoding is not None:
            metrics["speculativeDecoding"] = bool(speculative_decoding)
            metrics["treeBudget"] = int(getattr(result, "tree_budget", 0) or 0)
            if not speculative_decoding:
                metrics["dflashDraftModel"] = None
        return metrics

    def _assistant_metrics_payload(
        self,
        result: Any,
        *,
        requested_runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            **self._loaded_model_metrics_fields(),
            **self._result_runtime_metrics_fields(result),
            **result.to_metrics(),
            **(requested_runtime or {}),
        }

    def _stream_assistant_metrics_payload(
        self,
        *,
        final_chunk: Any,
        tok_s: float,
        response_seconds: float,
        requested_runtime: dict[str, Any] | None = None,
        ttft_seconds: float | None = None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "finishReason": final_chunk.finish_reason if final_chunk else "stop",
            "promptTokens": final_chunk.prompt_tokens if final_chunk else 0,
            "completionTokens": final_chunk.completion_tokens if final_chunk else 0,
            "totalTokens": (final_chunk.prompt_tokens + final_chunk.completion_tokens) if final_chunk else 0,
            "tokS": tok_s,
            "responseSeconds": response_seconds,
            "runtimeNote": final_chunk.runtime_note if final_chunk else None,
        }
        if final_chunk and getattr(final_chunk, "dflash_acceptance_rate", None) is not None:
            metrics["dflashAcceptanceRate"] = final_chunk.dflash_acceptance_rate
        if ttft_seconds is not None:
            metrics["ttftSeconds"] = ttft_seconds
        return {
            **self._loaded_model_metrics_fields(),
            **self._result_runtime_metrics_fields(final_chunk),
            **metrics,
            **(requested_runtime or {}),
        }

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
        if model_ref and (
            self.runtime.loaded_model is None
            or model_ref not in {self.runtime.loaded_model.ref, self.runtime.loaded_model.runtimeTarget}
        ):
            return True

        if self.runtime.loaded_model is None:
            return True

        return bool(
            self._runtime_profile_change_reasons(
                cache_bits=cache_bits,
                fp16_layers=fp16_layers,
                fused_attention=fused_attention,
                cache_strategy=cache_strategy,
                fit_model_in_memory=fit_model_in_memory,
                context_tokens=context_tokens,
                speculative_decoding=speculative_decoding,
                tree_budget=tree_budget,
            )
        )

    def _cache_profile_change_reasons(
        self,
        *,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        cache_strategy: str,
    ) -> list[str]:
        loaded_model = self.runtime.loaded_model
        if loaded_model is None:
            return []

        changes: list[str] = []
        if loaded_model.cacheStrategy != cache_strategy:
            changes.append(f"cache strategy {loaded_model.cacheStrategy} -> {cache_strategy}")
        if loaded_model.cacheBits != cache_bits:
            changes.append(f"cache bits {loaded_model.cacheBits} -> {cache_bits}")
        if loaded_model.fp16Layers != fp16_layers:
            changes.append(f"fp16 layers {loaded_model.fp16Layers} -> {fp16_layers}")
        if loaded_model.fusedAttention != fused_attention:
            changes.append(
                f"fused attention {'on' if loaded_model.fusedAttention else 'off'} -> {'on' if fused_attention else 'off'}"
            )
        return changes

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
        loaded_model = self.runtime.loaded_model
        if loaded_model is None:
            return []

        changes: list[str] = []
        if loaded_model.contextTokens != context_tokens:
            changes.append(f"context {loaded_model.contextTokens} -> {context_tokens}")
        changes.extend(
            self._cache_profile_change_reasons(
                cache_bits=cache_bits,
                fp16_layers=fp16_layers,
                fused_attention=fused_attention,
                cache_strategy=cache_strategy,
            )
        )
        if loaded_model.fitModelInMemory != fit_model_in_memory:
            changes.append(
                f"fit-in-memory {'on' if loaded_model.fitModelInMemory else 'off'} -> {'on' if fit_model_in_memory else 'off'}"
            )
        if bool(loaded_model.speculativeDecoding) != speculative_decoding:
            changes.append(
                f"speculative decoding {'on' if loaded_model.speculativeDecoding else 'off'} -> {'on' if speculative_decoding else 'off'}"
            )
        if int(loaded_model.treeBudget or 0) != int(tree_budget):
            changes.append(f"tree budget {loaded_model.treeBudget or 0} -> {tree_budget}")
        return changes

    def _append_benchmark_run(self, run: dict[str, Any]) -> None:
        from backend_service.app import _save_benchmark_runs
        self.benchmark_runs = [run, *[item for item in self.benchmark_runs if item["id"] != run["id"]]][:MAX_BENCHMARK_RUNS]
        _save_benchmark_runs(self.benchmark_runs, self._benchmarks_path)

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
        model_info = self.runtime.loaded_model
        launch_preferences = self._launch_preferences()
        if model_info is not None:
            return {
                "model": model_info.name,
                "modelRef": model_info.ref,
                "canonicalRepo": model_info.canonicalRepo,
                "modelSource": model_info.source,
                "modelPath": model_info.path,
                "modelBackend": model_info.backend,
                "cacheLabel": self._cache_label(
                    cache_strategy=str(model_info.cacheStrategy),
                    bits=int(model_info.cacheBits),
                    fp16_layers=int(model_info.fp16Layers),
                ),
                "cacheStrategy": model_info.cacheStrategy,
                "cacheBits": model_info.cacheBits,
                "fp16Layers": model_info.fp16Layers,
                "fusedAttention": model_info.fusedAttention,
                "fitModelInMemory": model_info.fitModelInMemory,
                "contextTokens": model_info.contextTokens,
                "speculativeDecoding": model_info.speculativeDecoding,
                "dflashDraftModel": model_info.dflashDraftModel,
                "treeBudget": model_info.treeBudget,
            }

        # No model is currently loaded. Prefer a model the user actually has
        # downloaded over a catalog default — surfacing a catalog-only entry
        # (e.g. nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF) just produces a
        # confusing "Failed to load … isn't downloaded on this machine"
        # error when the user clicks Load.
        for entry in self._library():
            entry_type = entry.get("modelType")
            if entry_type and entry_type != "text":
                continue
            if entry.get("broken"):
                continue
            return {
                "model": entry["name"],
                "modelRef": entry["name"],
                "canonicalRepo": entry.get("canonicalRepo") or entry.get("repo"),
                "modelSource": "library",
                "modelPath": entry["path"],
                "modelBackend": entry.get("backend", "auto"),
                "cacheLabel": self._cache_label(
                    cache_strategy=str(launch_preferences["cacheStrategy"]),
                    bits=int(launch_preferences["cacheBits"]),
                    fp16_layers=int(launch_preferences["fp16Layers"]),
                ),
                "cacheStrategy": launch_preferences["cacheStrategy"],
                "cacheBits": launch_preferences["cacheBits"],
                "fp16Layers": launch_preferences["fp16Layers"],
                "fusedAttention": launch_preferences["fusedAttention"],
                "fitModelInMemory": launch_preferences["fitModelInMemory"],
                "contextTokens": launch_preferences["contextTokens"],
                "speculativeDecoding": launch_preferences.get("speculativeDecoding", False),
                "dflashDraftModel": None,
                "treeBudget": launch_preferences.get("treeBudget", 0),
            }

        default_variant = _default_chat_variant()
        return {
            "model": default_variant["name"],
            "modelRef": default_variant["id"],
            "canonicalRepo": str(default_variant.get("repo") or "").strip() or None,
            "modelSource": "catalog",
            "modelPath": None,
            "modelBackend": default_variant.get("backend", "auto"),
            "cacheLabel": self._cache_label(
                cache_strategy=str(launch_preferences["cacheStrategy"]),
                bits=int(launch_preferences["cacheBits"]),
                fp16_layers=int(launch_preferences["fp16Layers"]),
            ),
            "cacheStrategy": launch_preferences["cacheStrategy"],
            "cacheBits": launch_preferences["cacheBits"],
            "fp16Layers": launch_preferences["fp16Layers"],
            "fusedAttention": launch_preferences["fusedAttention"],
            "fitModelInMemory": launch_preferences["fitModelInMemory"],
            "contextTokens": launch_preferences["contextTokens"],
            "speculativeDecoding": launch_preferences.get("speculativeDecoding", False),
            "dflashDraftModel": None,
            "treeBudget": launch_preferences.get("treeBudget", 0),
        }

    def _promote_session(self, session: dict[str, Any]) -> None:
        self.chat_sessions = [session, *[item for item in self.chat_sessions if item["id"] != session["id"]]]

    def _persist_sessions(self) -> None:
        from backend_service.app import _save_chat_sessions
        try:
            _save_chat_sessions(self.chat_sessions, self._chat_sessions_path)
        except OSError:
            pass  # Non-critical -- don't crash if disk is full

    def _unique_session_title(self, base_title: str, *, exclude_session_id: str | None = None) -> str:
        base = base_title.strip() or "New chat"
        if base == "New chat":
            return base

        pattern = _title_variant_pattern(base)
        highest_suffix = 0
        for session in self.chat_sessions:
            if exclude_session_id and session.get("id") == exclude_session_id:
                continue
            title = str(session.get("title") or "").strip()
            match = pattern.match(title)
            if not match:
                continue
            suffix = match.group(1)
            highest_suffix = max(highest_suffix, int(suffix) if suffix else 1)

        if highest_suffix == 0:
            return base
        return f"{base} ({highest_suffix + 1})"

    def _auto_session_title(self, prompt: str | None, *, exclude_session_id: str | None = None) -> str:
        return self._unique_session_title(
            _title_from_prompt(prompt),
            exclude_session_id=exclude_session_id,
        )

    def _normalize_auto_generated_session_titles(self) -> bool:
        seen_counts: dict[str, int] = {}
        changed = False

        for session in self.chat_sessions:
            messages = session.get("messages") if isinstance(session.get("messages"), list) else []
            first_user_message = next(
                (
                    message.get("text")
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "user"
                ),
                None,
            )
            base_title = _title_from_prompt(first_user_message)
            if base_title == "New chat":
                continue

            current_title = str(session.get("title") or "").strip()
            if not _title_variant_pattern(base_title).match(current_title):
                continue

            seen_counts[base_title] = seen_counts.get(base_title, 0) + 1
            next_index = seen_counts[base_title]
            normalized_title = base_title if next_index == 1 else f"{base_title} ({next_index})"
            if current_title != normalized_title:
                session["title"] = normalized_title
                changed = True

        return changed

    def _ensure_session(self, session_id: str | None = None, title: str | None = None) -> dict[str, Any]:
        if session_id:
            for session in self.chat_sessions:
                if session["id"] == session_id:
                    return session

        model_defaults = self._default_session_model()
        session = {
            "id": session_id or f"session-{uuid.uuid4().hex[:8]}",
            "title": title or "New chat",
            "updatedAt": self._time_label(),
            "pinned": False,
            "thinkingMode": "off",
            **model_defaults,
            "messages": [],
        }
        self.chat_sessions.insert(0, session)
        self.add_activity("Chat session created", session["title"])
        self._persist_sessions()
        return session

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        with self._lock:
            session = self._ensure_session(title=title)
            return session

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
        """Phase 2.5: generate a sibling variant of an assistant message.

        Truncates the session's message list to the user message that
        produced the target assistant turn (i.e. messages[0..index-1]
        plus the user prompt at index-1), then runs a non-streaming
        generation against the override model. The result is attached
        to ``messages[message_index].variants`` so the frontend can
        render it side-by-side with the original answer.

        The override model must already be loaded as the current
        runtime — callers should preload via the existing My Models
        flow before invoking compare. Raising on misalignment keeps
        the contract simple: variant generation never reloads the
        runtime under the user.

        Returns the updated session dict so the frontend can replace
        its local copy in one round-trip.
        """
        with self._lock:
            session = next(
                (s for s in self.chat_sessions if s.get("id") == session_id),
                None,
            )
            if session is None:
                raise ValueError(f"Session not found: {session_id}")
            messages = session.get("messages") or []
            if message_index < 0 or message_index >= len(messages):
                raise ValueError(
                    f"message_index {message_index} out of range "
                    f"(session has {len(messages)} messages)"
                )
            target = messages[message_index]
            if target.get("role") != "assistant":
                raise ValueError(
                    f"Variants can only be added to assistant messages "
                    f"(message {message_index} role: {target.get('role')})"
                )
            if message_index == 0:
                raise ValueError("Cannot add a variant to the first message — no prompt available")
            user_msg = messages[message_index - 1]
            if user_msg.get("role") != "user":
                raise ValueError(
                    f"Variant prompt must come from a user message at index "
                    f"{message_index - 1}, got role {user_msg.get('role')}"
                )
            history = _build_history_with_reasoning(
                messages[: message_index - 1],
                preserve_reasoning=False,
            )
            user_prompt = str(user_msg.get("text") or "")

            if self.runtime.loaded_model is None:
                raise ValueError("Load the override model before requesting a variant")
            loaded = self.runtime.loaded_model
            # Sanity check the runtime is the requested model. We don't
            # auto-reload because the user explicitly wants to compare
            # against an already-warm choice.
            if loaded.ref != model_ref and loaded.runtimeTarget != model_ref:
                raise ValueError(
                    f"Loaded runtime is {loaded.ref}, but variant requested {model_ref}. "
                    "Load the desired model first via My Models, then retry."
                )

            started_at = time.perf_counter()
            try:
                result = self.runtime.generate(
                    prompt=user_prompt,
                    history=history,
                    system_prompt=_compose_chat_system_prompt(None),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except RuntimeError as exc:
                raise ValueError(f"Variant generation failed: {exc}") from exc
            elapsed = round(time.perf_counter() - started_at, 2)

            metrics = self._stream_assistant_metrics_payload(
                final_chunk=type("Chunk", (), {
                    "finish_reason": result.finishReason,
                    "prompt_tokens": result.promptTokens,
                    "completion_tokens": result.completionTokens,
                    "tok_s": result.tokS,
                    "runtime_note": result.runtimeNote,
                    "dflash_acceptance_rate": getattr(result, "dflashAcceptanceRate", None),
                })(),
                tok_s=result.tokS,
                response_seconds=elapsed,
            )
            metrics["model"] = model_name
            metrics["modelRef"] = model_ref
            metrics["canonicalRepo"] = canonical_repo
            metrics["modelSource"] = source
            metrics["modelPath"] = path
            metrics["backend"] = backend

            variant = {
                "modelRef": model_ref,
                "modelName": model_name,
                "text": result.text,
                "metrics": metrics,
                "generatedAt": self._time_label(),
            }
            target.setdefault("variants", []).append(variant)
            session["updatedAt"] = self._time_label()
            self._persist_sessions()
            return session

    def fork_session(
        self,
        source_session_id: str,
        fork_at_message_index: int,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Phase 2.4: branch a thread at a specific message.

        Creates a new session containing a deep copy of the source's
        messages up to (and including) `fork_at_message_index`, plus
        the source's runtime profile (model, cache, thinking mode) so
        the fork resumes exactly where the user diverged. The new
        session carries `parentSessionId` and `forkedAtMessageIndex`
        metadata so the sidebar can render a relationship hint and
        future features (compare-vs-parent, merge) have the linkage.

        Raises ``ValueError`` when the source session doesn't exist
        or the fork index is out of range.
        """
        import copy

        with self._lock:
            source = next(
                (s for s in self.chat_sessions if s.get("id") == source_session_id),
                None,
            )
            if source is None:
                raise ValueError(f"Source session not found: {source_session_id}")
            messages = source.get("messages") or []
            if fork_at_message_index < 0 or fork_at_message_index >= len(messages):
                raise ValueError(
                    f"fork_at_message_index {fork_at_message_index} out of range "
                    f"(session has {len(messages)} messages)"
                )

            fork_title = title or f"{source.get('title', 'Chat')} (fork)"
            new_id = f"session-{uuid.uuid4().hex[:8]}"
            new_session: dict[str, Any] = {
                "id": new_id,
                "title": fork_title,
                "updatedAt": self._time_label(),
                "pinned": False,
                # Carry the runtime profile so the fork resumes on the
                # same model + cache config as the parent.
                "model": source.get("model"),
                "modelRef": source.get("modelRef"),
                "canonicalRepo": source.get("canonicalRepo"),
                "modelSource": source.get("modelSource"),
                "modelPath": source.get("modelPath"),
                "modelBackend": source.get("modelBackend"),
                "thinkingMode": source.get("thinkingMode") or "off",
                "cacheLabel": source.get("cacheLabel"),
                "cacheStrategy": source.get("cacheStrategy"),
                "cacheBits": source.get("cacheBits"),
                "fp16Layers": source.get("fp16Layers"),
                "fusedAttention": source.get("fusedAttention"),
                "fitModelInMemory": source.get("fitModelInMemory"),
                "contextTokens": source.get("contextTokens"),
                "speculativeDecoding": source.get("speculativeDecoding"),
                "dflashDraftModel": source.get("dflashDraftModel"),
                "treeBudget": source.get("treeBudget"),
                # Branching linkage so the UI can render the
                # parent-child relationship and so future features
                # (diff, merge) have the tie.
                "parentSessionId": source_session_id,
                "forkedAtMessageIndex": fork_at_message_index,
                "messages": copy.deepcopy(messages[: fork_at_message_index + 1]),
            }
            self.chat_sessions.insert(0, new_session)
            self.add_activity(
                "Chat session forked",
                f"{source.get('title', 'Chat')} → {fork_title}",
            )
            self._persist_sessions()
            return new_session

    def update_session(self, session_id: str, request: UpdateSessionRequest) -> dict[str, Any]:
        with self._lock:
            session = self._ensure_session(session_id=session_id)
            fields_set = getattr(request, "model_fields_set", set())
            if request.title is not None and request.title.strip():
                session["title"] = request.title.strip()
            if request.model is not None:
                session["model"] = request.model
            if "modelRef" in fields_set:
                session["modelRef"] = request.modelRef
            if "canonicalRepo" in fields_set:
                session["canonicalRepo"] = request.canonicalRepo
            if "modelSource" in fields_set:
                session["modelSource"] = request.modelSource
            if "modelPath" in fields_set:
                session["modelPath"] = request.modelPath
            if "modelBackend" in fields_set:
                session["modelBackend"] = request.modelBackend
            if "thinkingMode" in fields_set:
                session["thinkingMode"] = request.thinkingMode
            if "pinned" in fields_set:
                session["pinned"] = request.pinned
            if "cacheStrategy" in fields_set:
                session["cacheStrategy"] = request.cacheStrategy
            if "cacheBits" in fields_set:
                session["cacheBits"] = request.cacheBits
            if "fp16Layers" in fields_set:
                session["fp16Layers"] = request.fp16Layers
            if "fusedAttention" in fields_set:
                session["fusedAttention"] = request.fusedAttention
            if "fitModelInMemory" in fields_set:
                session["fitModelInMemory"] = request.fitModelInMemory
            if "contextTokens" in fields_set:
                session["contextTokens"] = request.contextTokens
            if "speculativeDecoding" in fields_set:
                session["speculativeDecoding"] = request.speculativeDecoding
            if "treeBudget" in fields_set:
                session["treeBudget"] = request.treeBudget
            if "dflashDraftModel" in fields_set:
                session["dflashDraftModel"] = request.dflashDraftModel
            if request.messages is not None:
                session["messages"] = request.messages
            session["updatedAt"] = self._time_label()
            self._promote_session(session)
            self.add_activity("Thread updated", session["title"])
            self._persist_sessions()
            return session

    def update_settings(self, request: UpdateSettingsRequest) -> dict[str, Any]:
        """Returns ``{"settings": ..., "restartRequired"?: bool, "migrationSummary"?: dict}``."""
        from backend_service.app import (
            _default_settings,
            _save_settings,
            DATA_LOCATION,
            DEFAULT_HOST,
        )

        with self._lock:
            next_settings = _default_settings()
            next_settings["modelDirectories"] = [dict(entry) for entry in self.settings["modelDirectories"]]
            next_settings["preferredServerPort"] = self.settings["preferredServerPort"]
            next_settings["allowRemoteConnections"] = bool(self.settings.get("allowRemoteConnections", False))
            next_settings["requireApiAuth"] = bool(self.settings.get("requireApiAuth", True))
            next_settings["launchPreferences"] = self._launch_preferences()
            next_settings["remoteProviders"] = list(self.settings.get("remoteProviders") or [])
            next_settings["huggingFaceToken"] = str(self.settings.get("huggingFaceToken") or "")
            next_settings["imageOutputsDirectory"] = str(self.settings.get("imageOutputsDirectory") or "")
            next_settings["videoOutputsDirectory"] = str(self.settings.get("videoOutputsDirectory") or "")
            next_settings["hfCachePath"] = str(self.settings.get("hfCachePath") or "")

            if request.modelDirectories is not None:
                next_settings["modelDirectories"] = _normalize_model_directories(
                    [entry.model_dump() for entry in request.modelDirectories]
                )
            if request.preferredServerPort is not None:
                next_settings["preferredServerPort"] = request.preferredServerPort
            if request.allowRemoteConnections is not None:
                next_settings["allowRemoteConnections"] = request.allowRemoteConnections
            if request.requireApiAuth is not None:
                next_settings["requireApiAuth"] = request.requireApiAuth
            if request.autoStartServer is not None:
                next_settings["autoStartServer"] = request.autoStartServer
            if request.launchPreferences is not None:
                next_settings["launchPreferences"] = _normalize_launch_preferences(request.launchPreferences.model_dump())
            if request.remoteProviders is not None:
                existing_by_id = {p.get("id"): p for p in (self.settings.get("remoteProviders") or [])}
                normalized = []
                for provider in request.remoteProviders:
                    api_base = _normalize_remote_provider_api_base(provider.apiBase)
                    api_key = provider.apiKey.strip()
                    existing_provider = existing_by_id.get(provider.id) or {}
                    existing_api_base = str(existing_provider.get("apiBase") or "").strip().rstrip("/")
                    existing_api_key = str(existing_provider.get("apiKey") or "")
                    if not api_key and provider.id in existing_by_id:
                        if not existing_api_key:
                            api_key = ""
                        elif existing_api_base == api_base:
                            api_key = existing_api_key
                        else:
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    f"Provider {provider.id} changed its API base. "
                                    "Re-enter the API key before saving this change."
                                ),
                            )
                    normalized.append({
                        "id": provider.id,
                        "label": provider.label,
                        "apiBase": api_base,
                        "apiKey": api_key,
                        "model": provider.model,
                    })
                next_settings["remoteProviders"] = normalized

            if request.huggingFaceToken is not None:
                previous_token_value = str(next_settings.get("huggingFaceToken") or "")
                token_value = request.huggingFaceToken.strip()
                next_settings["huggingFaceToken"] = token_value
                if token_value:
                    os.environ["HF_TOKEN"] = token_value
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = token_value
                else:
                    os.environ.pop("HF_TOKEN", None)
                    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
                if token_value != previous_token_value:
                    from backend_service.helpers.huggingface import _clear_huggingface_caches
                    from backend_service.helpers.images import _clear_image_discover_caches

                    _clear_huggingface_caches()
                    _clear_image_discover_caches()

            # Output directory overrides. Empty string clears the override.
            # Anything non-empty must be absolute or ~-relative — same rule as
            # dataDirectory — so we don't silently end up writing artifacts to
            # the working directory of whoever launched the backend.
            for field_name, label in (
                ("imageOutputsDirectory", "imageOutputsDirectory"),
                ("videoOutputsDirectory", "videoOutputsDirectory"),
                ("hfCachePath", "hfCachePath"),
            ):
                raw_value = getattr(request, field_name, None)
                if raw_value is None:
                    continue
                cleaned = raw_value.strip()
                # Accept Windows-style absolute paths (``D:\...``) alongside
                # POSIX and ``~``-relative paths. Bare relative paths are
                # rejected — silently writing artifacts to the backend's cwd
                # is exactly the "where did my models go?" class of bug we
                # want to avoid.
                if cleaned and not (
                    cleaned.startswith("/")
                    or cleaned.startswith("~")
                    or (len(cleaned) >= 2 and cleaned[1] == ":")
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"{label} must be an absolute path or start with ~.",
                    )
                next_settings[field_name] = cleaned

            data_migration: dict[str, Any] | None = None
            restart_required_for_data_dir = False
            if request.dataDirectory is not None:
                raw_dir = request.dataDirectory.strip()
                if raw_dir:
                    if not (raw_dir.startswith("/") or raw_dir.startswith("~")):
                        raise HTTPException(
                            status_code=400,
                            detail="dataDirectory must be an absolute path or start with ~.",
                        )
                    new_dir = Path(os.path.expanduser(raw_dir)).resolve()
                    if new_dir != DATA_LOCATION.data_dir:
                        try:
                            data_migration = _migrate_data_directory(
                                DATA_LOCATION.data_dir, new_dir
                            )
                            _save_data_location(new_dir)
                            restart_required_for_data_dir = True
                        except RuntimeError as exc:
                            raise HTTPException(status_code=400, detail=str(exc)) from exc

            _save_settings(next_settings, self._settings_path)
            self.settings = next_settings
            self._library_cache = None
            library = self._library(force=True)

            self.add_log(
                "settings",
                "info",
                f"Saved settings with {len(self.settings['modelDirectories'])} model directories, preferred API port {self.settings['preferredServerPort']}, and remote access {'enabled' if self.settings['allowRemoteConnections'] else 'disabled'}.",
            )
            if self.settings["preferredServerPort"] != self.server_port:
                self.add_log(
                    "server",
                    "info",
                    f"Preferred API port changed to {self.settings['preferredServerPort']}. Restart the API service to apply it.",
                )
            if bool(self.settings.get("allowRemoteConnections", False)) != (DEFAULT_HOST != "127.0.0.1"):
                self.add_log(
                    "server",
                    "info",
                    "Remote connection setting changed. Restart the API service to apply the new bind mode.",
                )
            self.add_activity(
                "Settings updated",
                f"{len(library)} models discovered across {len(self.settings['modelDirectories'])} configured directories.",
            )
            payload = self._settings_payload(library)
            response: dict[str, Any] = {"settings": payload}
            if restart_required_for_data_dir:
                response["restartRequired"] = True
            if data_migration is not None:
                response["migrationSummary"] = data_migration
            return response

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
        from backend_service.app import compute_cache_preview

        with self._lock:
            default_variant = _default_chat_variant()
            effective_model_ref = (
                request.modelRef
                or (self.runtime.loaded_model.ref if self.runtime.loaded_model is not None else None)
                or default_variant["id"]
            )
            catalog_entry = self._find_catalog_entry(effective_model_ref)
            library_entry = self._find_library_entry(request.path, effective_model_ref)
            model_name = request.modelName
            if model_name is None and library_entry is not None:
                model_name = str(library_entry.get("name") or "")
            if model_name is None and catalog_entry is not None:
                model_name = str(catalog_entry.get("name") or "")
            if model_name is None:
                model_name = str(effective_model_ref or default_variant["name"])

            if library_entry is not None and library_entry.get("broken"):
                reason = library_entry.get("brokenReason") or "incomplete or corrupt"
                raise RuntimeError(
                    f"Cannot benchmark '{library_entry.get('name') or effective_model_ref}': {reason}."
                )
            effective_source = request.source or ("library" if library_entry is not None else "catalog")
            effective_path = request.path if request.path is not None else (library_entry.get("path") if library_entry is not None else None)
            effective_backend = request.backend or (
                "llama.cpp"
                if (library_entry and library_entry.get("format") == "GGUF") or (catalog_entry and catalog_entry.get("format") == "GGUF")
                else "mlx"
            )

        load_seconds = 0.0
        effective_cache_strategy = "native" if request.speculativeDecoding else request.cacheStrategy
        effective_cache_bits = 0 if request.speculativeDecoding else request.cacheBits
        effective_fp16_layers = 0 if request.speculativeDecoding else request.fp16Layers

        if self._should_reload_for_profile(
            model_ref=effective_model_ref,
            cache_bits=effective_cache_bits,
            fp16_layers=effective_fp16_layers,
            fused_attention=request.fusedAttention,
            cache_strategy=effective_cache_strategy,
            fit_model_in_memory=request.fitModelInMemory,
            context_tokens=request.contextTokens,
            speculative_decoding=request.speculativeDecoding,
            tree_budget=0,
        ):
            load_started = time.perf_counter()
            self.load_model(
                LoadModelRequest(
                    modelRef=str(effective_model_ref),
                    modelName=model_name,
                    canonicalRepo=self._resolve_canonical_repo(
                        model_ref=str(effective_model_ref),
                        path=effective_path,
                        canonical_repo=None,
                    ),
                    source=effective_source,
                    backend=effective_backend,
                    path=effective_path,
                    cacheStrategy=request.cacheStrategy,
                    cacheBits=request.cacheBits,
                    fp16Layers=request.fp16Layers,
                    fusedAttention=request.fusedAttention,
                    fitModelInMemory=request.fitModelInMemory,
                    contextTokens=request.contextTokens,
                    speculativeDecoding=request.speculativeDecoding,
                )
            )
            load_seconds = round(time.perf_counter() - load_started, 2)

        with self._lock:
            params_b = float(catalog_entry.get("paramsB")) if catalog_entry and catalog_entry.get("paramsB") is not None else 7.0
            preview = compute_cache_preview(
                bits=request.cacheBits if request.cacheBits else 4,
                fp16_layers=request.fp16Layers,
                context_tokens=request.contextTokens,
                params_b=params_b,
                system_stats=self._system_snapshot(),
            )
            use_compressed = request.cacheBits > 0
            cache_gb = preview["optimizedCacheGb"] if use_compressed else preview["baselineCacheGb"]
            baseline_cache_gb = preview["baselineCacheGb"]
            compression = round(baseline_cache_gb / cache_gb, 1) if use_compressed and cache_gb else 1.0
            quality = int(round(preview["qualityPercent"])) if use_compressed else 100
            cache_label = self._cache_label(
                cache_strategy=request.cacheStrategy,
                bits=request.cacheBits,
                fp16_layers=request.fp16Layers,
            )

        base_run: dict[str, Any] = {
            "id": f"bench-{uuid.uuid4().hex[:8]}",
            "mode": request.mode,
            "model": model_name,
            "modelRef": effective_model_ref,
            "backend": self.runtime.loaded_model.backend if self.runtime.loaded_model else effective_backend,
            "engineLabel": self.runtime.engine.engine_label,
            "source": effective_source,
            "measuredAt": self._time_label(),
            "bits": request.cacheBits if request.cacheBits > 0 else 16,
            "fp16Layers": request.fp16Layers,
            "cacheStrategy": request.cacheStrategy,
            "cacheLabel": cache_label,
            "cacheGb": cache_gb,
            "baselineCacheGb": baseline_cache_gb,
            "compression": compression,
            "contextTokens": request.contextTokens,
            "maxTokens": request.maxTokens,
            "loadSeconds": load_seconds,
        }

        if request.mode == "perplexity":
            eval_result = self.runtime.engine.eval_perplexity(
                dataset=request.perplexityDataset,
                num_samples=request.perplexityNumSamples,
                seq_length=request.perplexitySeqLength,
                batch_size=request.perplexityBatchSize,
            )
            run = {
                **base_run,
                "label": request.label or f"{model_name} / Perplexity / {request.perplexityDataset}",
                "perplexity": eval_result["perplexity"],
                "perplexityStdError": eval_result["standardError"],
                "perplexityDataset": eval_result["dataset"],
                "perplexityNumSamples": eval_result["numSamples"],
                "evalTokensPerSecond": eval_result["evalTokensPerSecond"],
                "evalSeconds": eval_result["evalSeconds"],
                "quality": quality,
                "tokS": eval_result["evalTokensPerSecond"],
                "responseSeconds": eval_result["evalSeconds"],
                "totalSeconds": round(load_seconds + eval_result["evalSeconds"], 2),
                "promptTokens": 0,
                "completionTokens": 0,
                "totalTokens": 0,
                "notes": f"Perplexity: {eval_result['perplexity']:.2f} \u00b1 {eval_result['standardError']:.2f} on {eval_result['dataset']} ({eval_result['numSamples']} samples)",
            }
        elif request.mode == "task_accuracy":
            eval_result = self.runtime.engine.eval_task_accuracy(
                task_name=request.taskName,
                limit=request.taskLimit,
                num_shots=request.taskNumShots,
            )
            accuracy_pct = round(eval_result["accuracy"] * 100, 1)
            run = {
                **base_run,
                "label": request.label or f"{model_name} / {request.taskName.upper()} / {eval_result['correct']}/{eval_result['total']}",
                "taskName": eval_result["taskName"],
                "taskAccuracy": eval_result["accuracy"],
                "taskCorrect": eval_result["correct"],
                "taskTotal": eval_result["total"],
                "taskNumShots": eval_result["numShots"],
                "evalSeconds": eval_result["evalSeconds"],
                "quality": quality,
                "tokS": 0,
                "responseSeconds": eval_result["evalSeconds"],
                "totalSeconds": round(load_seconds + eval_result["evalSeconds"], 2),
                "promptTokens": 0,
                "completionTokens": 0,
                "totalTokens": 0,
                "notes": f"{request.taskName.upper()}: {accuracy_pct}% ({eval_result['correct']}/{eval_result['total']}) {eval_result['numShots']}-shot",
            }
        else:
            prompt = request.prompt or (
                "Summarize the practical trade-offs of this runtime profile for a local desktop user in six short bullets."
            )
            result = self.runtime.generate(
                prompt=prompt,
                history=[],
                system_prompt="Return a concise but complete answer so ChaosEngineAI can benchmark response speed consistently.",
                max_tokens=request.maxTokens,
                temperature=request.temperature,
            )
            run = {
                **base_run,
                "label": request.label
                or _benchmark_label(
                    model_name,
                    cache_strategy=request.cacheStrategy,
                    bits=request.cacheBits,
                    fp16_layers=request.fp16Layers,
                    context_tokens=request.contextTokens,
                ),
                "tokS": round(result.tokS, 1),
                "quality": quality,
                "responseSeconds": round(result.responseSeconds, 2),
                "totalSeconds": round(load_seconds + result.responseSeconds, 2),
                "promptTokens": result.promptTokens,
                "completionTokens": result.completionTokens,
                "totalTokens": result.totalTokens,
                "notes": result.runtimeNote,
            }

        with self._lock:
            self._append_benchmark_run(run)
            mode_label = {"perplexity": "Perplexity", "task_accuracy": "Task accuracy"}.get(request.mode, "Throughput")
            self.add_log("benchmark", "info", f"{mode_label} benchmark completed for {model_name}: {run.get('notes', '')}")
            self.add_activity("Benchmark completed", run["label"])
            return {
                "result": run,
                "benchmarks": self.benchmark_runs,
                "runtime": self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served),
            }

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
        from backend_service.app import DOCUMENTS_DIR
        safe_id = re.sub(r"[^\w\-]", "_", session_id)
        return DOCUMENTS_DIR / safe_id

    def list_documents(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            session = self._ensure_session(session_id)
            return list(session.get("documents", []))

    def upload_document(self, session_id: str, original_name: str, raw_bytes: bytes) -> dict[str, Any]:
        from backend_service.app import MAX_DOC_SIZE_BYTES, MAX_SESSION_DOCS_BYTES, DOC_ALLOWED_EXTENSIONS

        if len(raw_bytes) > MAX_DOC_SIZE_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_DOC_SIZE_BYTES // (1024*1024)}MB limit.")
        sanitized = _sanitize_filename(original_name)
        ext = Path(sanitized).suffix.lower()
        if ext not in DOC_ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type not supported: {ext}")

        with self._lock:
            session = self._ensure_session(session_id)
            existing = session.get("documents") or []
            current_total = sum(d.get("sizeBytes", 0) for d in existing)
            if current_total + len(raw_bytes) > MAX_SESSION_DOCS_BYTES:
                raise HTTPException(status_code=413, detail="Session document quota exceeded (200MB).")

            doc_id = f"doc-{uuid.uuid4().hex[:12]}"
            session_dir = self._session_docs_dir(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            try:
                session_dir.chmod(0o700)
            except OSError:
                pass

            doc_path = session_dir / f"{doc_id}{ext}"
            doc_path.write_bytes(raw_bytes)
            try:
                doc_path.chmod(0o600)
            except OSError:
                pass

        try:
            text = _extract_text_from_file(doc_path)
        except RuntimeError as exc:
            doc_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        chunks = _chunk_text(text)
        chunks_path = session_dir / f"{doc_id}.chunks.json"
        chunks_path.write_text(
            json.dumps([{"index": i, "text": c} for i, c in enumerate(chunks)], indent=2),
            encoding="utf-8",
        )

        with self._lock:
            session = self._ensure_session(session_id)
            doc_meta = {
                "id": doc_id,
                "filename": doc_path.name,
                "originalName": sanitized,
                "sizeBytes": len(raw_bytes),
                "chunkCount": len(chunks),
                "uploadedAt": self._time_label(),
            }
            session.setdefault("documents", []).append(doc_meta)
            session["updatedAt"] = self._time_label()
            self.add_log("chat", "info", f"Document uploaded to session {session_id}: {sanitized} ({len(chunks)} chunks)")
            self._persist_sessions()
            return doc_meta

    def delete_document(self, session_id: str, doc_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._ensure_session(session_id)
            docs = session.get("documents") or []
            target = next((d for d in docs if d.get("id") == doc_id), None)
            if not target:
                raise HTTPException(status_code=404, detail="Document not found.")
            session["documents"] = [d for d in docs if d.get("id") != doc_id]
            session["updatedAt"] = self._time_label()
            session_dir = self._session_docs_dir(session_id)
            for f in session_dir.glob(f"{doc_id}*"):
                try:
                    f.unlink()
                except OSError:
                    pass
            self.add_log("chat", "info", f"Document removed: {target.get('originalName')}")
            self._persist_sessions()
            return {"deleted": doc_id}

    def delete_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            target = next((s for s in self.chat_sessions if s.get("id") == session_id), None)
            if not target:
                raise HTTPException(status_code=404, detail="Session not found.")
            self.chat_sessions = [s for s in self.chat_sessions if s.get("id") != session_id]
            self.add_log("chat", "info", f"Session deleted: {target.get('title', session_id)}")
            self._persist_sessions()
            return {"deleted": session_id}

    def _retrieve_session_context(self, session_id: str, prompt: str, top_k: int = 5) -> tuple[str, list[dict[str, Any]]]:
        """Retrieve relevant document chunks for a prompt.

        Returns (context_text, citations) where citations is a list of
        dicts with docId, docName, chunkIndex, page, preview keys.

        Phase 2.6: when an llama-embedding binary + embedding GGUF are
        both discoverable via env vars or `<dataDir>/embeddings/`,
        retrieval uses semantic cosine similarity blended with BM25
        (70/30) instead of TF-IDF + BM25. The embedding client is
        resolved per-call so newly-installed models pick up without a
        restart, and the legacy lexical path remains the fallback when
        anything goes wrong.
        """
        from backend_service.helpers.documents import DocumentIndex
        from backend_service.rag import resolve_embedding_client

        session_dir = self._session_docs_dir(session_id)
        if not session_dir.exists():
            return "", []

        # Embedding client discovery: env vars override path; if no
        # CHAOSENGINE_EMBEDDING_MODEL is set we look under
        # `<documents-parent>/embeddings/*.gguf`. Returns None when
        # nothing is wired, in which case retrieval transparently
        # falls back to TF-IDF + BM25.
        from backend_service.app import DOCUMENTS_DIR

        embedding_client = resolve_embedding_client(DOCUMENTS_DIR.parent)

        # Build a temporary index from all session documents. When the
        # embedding client is available, chunks are embedded as they're
        # added so the search call below routes through cosine + BM25.
        index = DocumentIndex()
        for chunk_file in session_dir.glob("*.chunks.json"):
            try:
                doc_chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
                doc_name = chunk_file.stem.replace(".chunks", "")
                full_text = "\n\n".join(c.get("text", "") for c in doc_chunks)
                if full_text.strip():
                    index.add_document(
                        full_text,
                        doc_id=doc_name,
                        doc_name=doc_name,
                        embedding_client=embedding_client,
                    )
            except (OSError, json.JSONDecodeError):
                continue

        results = index.search(prompt, top_k=top_k, embedding_client=embedding_client)
        if not results:
            return "", []

        context = "\n\n---\n\n".join(r["text"] for r in results)
        citations = [r["citation"] for r in results]
        return context, citations

    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        with self._lock:
            session = self._ensure_session(request.sessionId, request.title)
            launch_preferences = self._launch_preferences()
            effective_model_ref = request.modelRef or session.get("modelRef")
            effective_model_name = request.modelName or session.get("model")
            effective_canonical_repo = request.canonicalRepo or session.get("canonicalRepo")
            effective_source = request.source or session.get("modelSource") or "catalog"
            effective_path = request.path if request.path is not None else session.get("modelPath")
            effective_backend = request.backend or session.get("modelBackend") or "auto"
            effective_thinking_mode = request.thinkingMode or session.get("thinkingMode") or "off"
            desired_cache_strategy = (
                request.cacheStrategy if request.cacheStrategy is not None
                else session.get("cacheStrategy") or launch_preferences["cacheStrategy"]
            )
            desired_cache_bits = (
                request.cacheBits if request.cacheBits is not None
                else session.get("cacheBits") if session.get("cacheBits") is not None
                else launch_preferences["cacheBits"]
            )
            desired_fp16_layers = (
                request.fp16Layers if request.fp16Layers is not None
                else session.get("fp16Layers") if session.get("fp16Layers") is not None
                else launch_preferences["fp16Layers"]
            )
            desired_fused_attention = (
                request.fusedAttention if request.fusedAttention is not None
                else session.get("fusedAttention") if session.get("fusedAttention") is not None
                else launch_preferences["fusedAttention"]
            )
            desired_fit_model = (
                request.fitModelInMemory if request.fitModelInMemory is not None
                else session.get("fitModelInMemory") if session.get("fitModelInMemory") is not None
                else launch_preferences["fitModelInMemory"]
            )
            desired_context_tokens = (
                request.contextTokens if request.contextTokens is not None
                else session.get("contextTokens") if session.get("contextTokens") is not None
                else launch_preferences["contextTokens"]
            )
            desired_speculative_decoding = (
                request.speculativeDecoding if request.speculativeDecoding is not None
                else session.get("speculativeDecoding") if session.get("speculativeDecoding") is not None
                else launch_preferences["speculativeDecoding"]
            )
            desired_tree_budget = (
                request.treeBudget if request.treeBudget is not None
                else session.get("treeBudget") if session.get("treeBudget") is not None
                else launch_preferences["treeBudget"]
            )
            requested_runtime = self._requested_runtime_metrics_fields(
                cache_strategy=str(desired_cache_strategy),
                cache_bits=int(desired_cache_bits),
                fp16_layers=int(desired_fp16_layers),
                fit_model_in_memory=bool(desired_fit_model),
                speculative_decoding=bool(desired_speculative_decoding),
                tree_budget=int(desired_tree_budget),
            )
            effective_cache_strategy = "native" if desired_speculative_decoding else desired_cache_strategy
            effective_cache_bits = 0 if desired_speculative_decoding else desired_cache_bits
            effective_fp16_layers = 0 if desired_speculative_decoding else desired_fp16_layers

            should_reload_model = self._should_reload_for_profile(
                model_ref=effective_model_ref,
                cache_bits=effective_cache_bits,
                fp16_layers=effective_fp16_layers,
                fused_attention=desired_fused_attention,
                cache_strategy=effective_cache_strategy,
                fit_model_in_memory=desired_fit_model,
                context_tokens=desired_context_tokens,
                speculative_decoding=desired_speculative_decoding,
                tree_budget=desired_tree_budget,
            )

            if effective_model_ref and should_reload_model:
                self.load_model(
                    LoadModelRequest(
                        modelRef=effective_model_ref,
                        modelName=effective_model_name,
                        canonicalRepo=effective_canonical_repo,
                        source=effective_source,
                        backend=effective_backend,
                        path=effective_path,
                        cacheStrategy=desired_cache_strategy,
                        cacheBits=desired_cache_bits,
                        fp16Layers=desired_fp16_layers,
                        fusedAttention=desired_fused_attention,
                        fitModelInMemory=desired_fit_model,
                        contextTokens=desired_context_tokens,
                        speculativeDecoding=desired_speculative_decoding,
                        treeBudget=desired_tree_budget,
                    )
                )

            if self.runtime.loaded_model is None:
                raise HTTPException(status_code=409, detail="Load a model before sending prompts.")

            if effective_canonical_repo and self.runtime.loaded_model.canonicalRepo != effective_canonical_repo:
                self.runtime.loaded_model.canonicalRepo = effective_canonical_repo

            history = _build_history_with_reasoning(
                session["messages"],
                preserve_reasoning=(effective_thinking_mode == "auto"),
            )
            session["messages"].append({"role": "user", "text": request.prompt, "metrics": None})
            session["updatedAt"] = self._time_label()
            # Phase 2.12: if `oneTurnOverride` is set, skip persisting the
            # active runtime's model identity onto the session so the
            # session default (the previously-loaded model) sticks for
            # the next plain message. Other session metadata (cache
            # strategy, context, thinking mode) still updates so the
            # picked model's runtime profile is reflected on this turn.
            if not getattr(request, "oneTurnOverride", False):
                session["model"] = self.runtime.loaded_model.name
                session["modelRef"] = self.runtime.loaded_model.ref
                session["canonicalRepo"] = self.runtime.loaded_model.canonicalRepo
                session["modelSource"] = self.runtime.loaded_model.source
                session["modelPath"] = self.runtime.loaded_model.path
                session["modelBackend"] = self.runtime.loaded_model.backend
            session["thinkingMode"] = effective_thinking_mode
            session["cacheLabel"] = self._cache_label(
                cache_strategy=str(self.runtime.loaded_model.cacheStrategy),
                bits=int(self.runtime.loaded_model.cacheBits),
                fp16_layers=int(self.runtime.loaded_model.fp16Layers),
            )
            session["cacheStrategy"] = self.runtime.loaded_model.cacheStrategy
            session["cacheBits"] = self.runtime.loaded_model.cacheBits
            session["fp16Layers"] = self.runtime.loaded_model.fp16Layers
            session["fusedAttention"] = self.runtime.loaded_model.fusedAttention
            session["fitModelInMemory"] = self.runtime.loaded_model.fitModelInMemory
            session["contextTokens"] = self.runtime.loaded_model.contextTokens
            session["speculativeDecoding"] = self.runtime.loaded_model.speculativeDecoding
            session["dflashDraftModel"] = self.runtime.loaded_model.dflashDraftModel
            session["treeBudget"] = self.runtime.loaded_model.treeBudget
            if session["title"] == "New chat":
                requested_title = str(request.title or "").strip()
                session["title"] = (
                    requested_title
                    if requested_title and requested_title != "New chat"
                    else self._auto_session_title(request.prompt, exclude_session_id=session["id"])
                )
            model_tag = self.runtime.loaded_model.name if self.runtime.loaded_model else "unknown"
            msg_count = len(history) + 1
            self.add_log("chat", "info", f"[{model_tag}] Running chat completion on conversation with {msg_count} messages.")
            self.add_log("chat", "info", f"[{model_tag}] Generating response...")
            self.active_requests += 1
            effective_system_prompt = _compose_chat_system_prompt(request.systemPrompt, effective_thinking_mode)
            doc_context, rag_citations = self._retrieve_session_context(session["id"], request.prompt)
            if doc_context:
                rag_preamble = (
                    "You have access to the following document context retrieved from the user's uploaded files. "
                    "Use it to answer their questions when relevant.\n\n--- DOCUMENT CONTEXT ---\n"
                    + doc_context
                    + "\n--- END CONTEXT ---"
                )
                effective_system_prompt = (rag_preamble + "\n\n" + effective_system_prompt).strip()
                self.add_log("chat", "info", f"[{model_tag}] Injected {len(doc_context)} chars of document context.")

        gen_start = time.perf_counter()
        try:
            if request.enableTools:
                from backend_service.agent import run_agent_loop
                agent_result = run_agent_loop(
                    generate_fn=self.runtime.generate,
                    prompt=request.prompt,
                    history=history,
                    system_prompt=effective_system_prompt,
                    max_tokens=request.maxTokens,
                    temperature=request.temperature,
                    images=request.images,
                    available_tools=request.availableTools,
                )
                # Synthesize a GenerationResult-like object for metrics
                class _AgentResultProxy:
                    text = agent_result.text
                    finishReason = "stop"
                    promptTokens = agent_result.total_prompt_tokens
                    completionTokens = agent_result.total_completion_tokens
                    totalTokens = agent_result.total_prompt_tokens + agent_result.total_completion_tokens
                    tokS = 0.0
                    runtimeNote = f"Agent loop: {agent_result.iterations} iterations, {len(agent_result.tool_calls)} tool calls"
                    responseSeconds = 0.0
                    tool_calls = None
                result = _AgentResultProxy()
                tool_call_payloads = [
                    {
                        "id": tc.tool_call_id,
                        "name": tc.tool_name,
                        "arguments": tc.arguments,
                        "result": tc.result,
                        "elapsed": tc.elapsed_seconds,
                        # Phase 2.8: forward structured output hint +
                        # data through to the frontend `ToolCallInfo`.
                        # When `render_as` is None the frontend falls
                        # back to the legacy collapsible-JSON view.
                        "renderAs": tc.render_as,
                        "data": tc.data,
                    }
                    for tc in agent_result.tool_calls
                ]
            else:
                result = self.runtime.generate(
                    prompt=request.prompt,
                    history=history,
                    system_prompt=effective_system_prompt,
                    max_tokens=request.maxTokens,
                    temperature=request.temperature,
                    images=request.images,
                    samplers=_build_sampler_overrides(request),
                    reasoning_effort=request.reasoningEffort,
                    json_schema=request.jsonSchema,
                )
                tool_call_payloads = []
        except RuntimeError as exc:
            with self._lock:
                if (session["messages"]
                        and session["messages"][-1].get("role") == "user"
                        and session["messages"][-1].get("text") == request.prompt):
                    session["messages"].pop()
                    session["updatedAt"] = self._time_label()
                    self._persist_sessions()
                self.active_requests = max(0, self.active_requests - 1)
                self.add_log("chat", "error", f"[{model_tag}] Generation failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        gen_elapsed = round(time.perf_counter() - gen_start, 2)
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.requests_served += 1
            metrics = self._assistant_metrics_payload(result, requested_runtime=requested_runtime)
            if tool_call_payloads:
                metrics["toolCalls"] = tool_call_payloads
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "text": result.text,
                "metrics": metrics,
            }
            if tool_call_payloads:
                assistant_message["toolCalls"] = tool_call_payloads
            if rag_citations:
                assistant_message["citations"] = rag_citations
            session["messages"].append(assistant_message)
            session["updatedAt"] = self._time_label()
            self._promote_session(session)
            self.add_log(
                "chat", "info",
                f"[{model_tag}] Finished response -- {result.completionTokens} tokens in {gen_elapsed}s "
                f"({result.tokS} tok/s, {result.promptTokens} prompt tokens).",
            )
            self.add_activity("Chat completion", session["title"])
            self._persist_sessions()

            return {
                "session": session,
                "assistant": assistant_message,
                "runtime": self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served),
            }

    def generate_stream(self, request: GenerateRequest):
        """SSE streaming version of generate(). Returns a StreamingResponse."""
        with self._lock:
            session = self._ensure_session(request.sessionId, request.title)
            launch_preferences = self._launch_preferences()
            effective_model_ref = request.modelRef or session.get("modelRef")
            effective_model_name = request.modelName or session.get("model")
            effective_canonical_repo = request.canonicalRepo or session.get("canonicalRepo")
            effective_source = request.source or session.get("modelSource") or "catalog"
            effective_path = request.path if request.path is not None else session.get("modelPath")
            effective_backend = request.backend or session.get("modelBackend") or "auto"
            effective_thinking_mode = request.thinkingMode or session.get("thinkingMode") or "off"
            desired_cache_strategy = (
                request.cacheStrategy if request.cacheStrategy is not None
                else session.get("cacheStrategy") or launch_preferences["cacheStrategy"]
            )
            desired_cache_bits = (
                request.cacheBits if request.cacheBits is not None
                else session.get("cacheBits") if session.get("cacheBits") is not None
                else launch_preferences["cacheBits"]
            )
            desired_fp16_layers = (
                request.fp16Layers if request.fp16Layers is not None
                else session.get("fp16Layers") if session.get("fp16Layers") is not None
                else launch_preferences["fp16Layers"]
            )
            desired_fused_attention = (
                request.fusedAttention if request.fusedAttention is not None
                else session.get("fusedAttention") if session.get("fusedAttention") is not None
                else launch_preferences["fusedAttention"]
            )
            desired_fit_model = (
                request.fitModelInMemory if request.fitModelInMemory is not None
                else session.get("fitModelInMemory") if session.get("fitModelInMemory") is not None
                else launch_preferences["fitModelInMemory"]
            )
            desired_context_tokens = (
                request.contextTokens if request.contextTokens is not None
                else session.get("contextTokens") if session.get("contextTokens") is not None
                else launch_preferences["contextTokens"]
            )
            desired_speculative_decoding = (
                request.speculativeDecoding if request.speculativeDecoding is not None
                else session.get("speculativeDecoding") if session.get("speculativeDecoding") is not None
                else launch_preferences["speculativeDecoding"]
            )
            desired_tree_budget = (
                request.treeBudget if request.treeBudget is not None
                else session.get("treeBudget") if session.get("treeBudget") is not None
                else launch_preferences["treeBudget"]
            )
            requested_runtime = self._requested_runtime_metrics_fields(
                cache_strategy=str(desired_cache_strategy),
                cache_bits=int(desired_cache_bits),
                fp16_layers=int(desired_fp16_layers),
                fit_model_in_memory=bool(desired_fit_model),
                speculative_decoding=bool(desired_speculative_decoding),
                tree_budget=int(desired_tree_budget),
            )
            effective_cache_strategy = "native" if desired_speculative_decoding else desired_cache_strategy
            effective_cache_bits = 0 if desired_speculative_decoding else desired_cache_bits
            effective_fp16_layers = 0 if desired_speculative_decoding else desired_fp16_layers

            should_reload = self._should_reload_for_profile(
                model_ref=effective_model_ref, cache_bits=effective_cache_bits,
                fp16_layers=effective_fp16_layers, fused_attention=desired_fused_attention,
                cache_strategy=effective_cache_strategy, fit_model_in_memory=desired_fit_model,
                context_tokens=desired_context_tokens,
                speculative_decoding=desired_speculative_decoding,
                tree_budget=desired_tree_budget,
            )
            if effective_model_ref and should_reload:
                self.load_model(LoadModelRequest(
                    modelRef=effective_model_ref, modelName=effective_model_name,
                    canonicalRepo=effective_canonical_repo,
                    source=effective_source, backend=effective_backend, path=effective_path,
                    cacheStrategy=desired_cache_strategy, cacheBits=desired_cache_bits,
                    fp16Layers=desired_fp16_layers,
                    fusedAttention=desired_fused_attention,
                    fitModelInMemory=desired_fit_model, contextTokens=desired_context_tokens,
                    speculativeDecoding=desired_speculative_decoding,
                    treeBudget=desired_tree_budget,
                ))

            if self.runtime.loaded_model is None:
                raise HTTPException(status_code=409, detail="Load a model before sending prompts.")

            if effective_canonical_repo and self.runtime.loaded_model.canonicalRepo != effective_canonical_repo:
                self.runtime.loaded_model.canonicalRepo = effective_canonical_repo

            history = _build_history_with_reasoning(
                session["messages"],
                preserve_reasoning=(effective_thinking_mode == "auto"),
            )
            session["messages"].append({"role": "user", "text": request.prompt, "metrics": None})
            session["updatedAt"] = self._time_label()
            # Phase 2.12: if `oneTurnOverride` is set, skip persisting the
            # active runtime's model identity onto the session so the
            # session default (the previously-loaded model) sticks for
            # the next plain message. Other session metadata (cache
            # strategy, context, thinking mode) still updates so the
            # picked model's runtime profile is reflected on this turn.
            if not getattr(request, "oneTurnOverride", False):
                session["model"] = self.runtime.loaded_model.name
                session["modelRef"] = self.runtime.loaded_model.ref
                session["canonicalRepo"] = self.runtime.loaded_model.canonicalRepo
                session["modelSource"] = self.runtime.loaded_model.source
                session["modelPath"] = self.runtime.loaded_model.path
                session["modelBackend"] = self.runtime.loaded_model.backend
            session["thinkingMode"] = effective_thinking_mode
            session["cacheLabel"] = self._cache_label(
                cache_strategy=str(self.runtime.loaded_model.cacheStrategy),
                bits=int(self.runtime.loaded_model.cacheBits),
                fp16_layers=int(self.runtime.loaded_model.fp16Layers),
            )
            session["cacheStrategy"] = self.runtime.loaded_model.cacheStrategy
            session["cacheBits"] = self.runtime.loaded_model.cacheBits
            session["fp16Layers"] = self.runtime.loaded_model.fp16Layers
            session["fusedAttention"] = self.runtime.loaded_model.fusedAttention
            session["fitModelInMemory"] = self.runtime.loaded_model.fitModelInMemory
            session["contextTokens"] = self.runtime.loaded_model.contextTokens
            session["speculativeDecoding"] = self.runtime.loaded_model.speculativeDecoding
            session["dflashDraftModel"] = self.runtime.loaded_model.dflashDraftModel
            session["treeBudget"] = self.runtime.loaded_model.treeBudget
            if session["title"] == "New chat":
                requested_title = str(request.title or "").strip()
                session["title"] = (
                    requested_title
                    if requested_title and requested_title != "New chat"
                    else self._auto_session_title(request.prompt, exclude_session_id=session["id"])
                )
            model_tag = self.runtime.loaded_model.name
            self.add_log("chat", "info", f"[{model_tag}] Streaming response...")
            self.active_requests += 1
            # Hotfix (2026-05-01 v2): vision input has no working path
            # on either runtime today. The MLX worker subprocess never
            # wired images, and `_resolve_gguf_path` strips mmproj
            # projector files so llama-server never gets `--mmproj`.
            # Until mmproj wiring lands (Phase 2.6+ work), the
            # `visionEnabled` flag on LoadedModelInfo stays False on
            # every load and we strip + warn loudly here. The capability
            # resolver also demotes vision via this same flag so the
            # composer hides the attach button — this branch is the
            # belt-and-braces for legacy clients that bypass the gate.
            if request.images and not self.runtime.loaded_model.visionEnabled:
                engine_label = self.runtime.loaded_model.engine or "current"
                self.add_log(
                    "chat", "warning",
                    f"[{model_tag}] Stripped {len(request.images)} attached "
                    f"image(s): the {engine_label} runtime has no mmproj "
                    "vision projector wired up, so images would be silently "
                    "dropped and the model would hallucinate. Vision support "
                    "lands with the mmproj loader.",
                )
                request.images = None
            effective_system_prompt = _compose_chat_system_prompt(request.systemPrompt, effective_thinking_mode)
            doc_context, stream_rag_citations = self._retrieve_session_context(session["id"], request.prompt)
            if doc_context:
                rag_preamble = (
                    "You have access to the following document context retrieved from the user's uploaded files. "
                    "Use it to answer their questions when relevant.\n\n--- DOCUMENT CONTEXT ---\n"
                    + doc_context
                    + "\n--- END CONTEXT ---"
                )
                effective_system_prompt = (rag_preamble + "\n\n" + effective_system_prompt).strip()
                self.add_log("chat", "info", f"[{model_tag}] Injected {len(doc_context)} chars of document context.")

        chaosengine = self
        enable_tools = request.enableTools
        available_tools = request.availableTools
        gen_start = time.perf_counter()
        # Reset any stale cancellation flag from a prior turn so this fresh
        # generation isn't aborted before it starts.
        chaosengine.clear_chat_cancel(session["id"])
        session_id_for_cancel = session["id"]

        def _sse_stream():
            full_text = ""
            full_reasoning = ""
            final_chunk = None
            agent_tool_calls: list[dict[str, Any]] = []
            cancelled = False
            # Phase 2.0: track prompt-eval → generating phase transition so the
            # client can render an explicit "Processing prompt..." indicator
            # instead of a blank flashing cursor while the model is still
            # ingesting the prompt. The OpenAI-compat streaming endpoint
            # exposes nothing until the first decoded token, so phase here is
            # binary (prompt_eval | generating) plus a TTFT measurement on
            # transition.
            phase_first_output_seen = False
            ttft_seconds: float | None = None

            # Phase 2.0.5-B: pre-flight memory gate. Refuse the generation
            # before it starts when the host is already memory-starved, so
            # the user gets an actionable error instead of a silent OOM /
            # swap-thrash that wedges the laptop. The gate is conservative
            # — it does not predict working-set size, just bails when the
            # available-memory floor or pressure ceiling is breached.
            try:
                from backend_service.helpers.memory_gate import (
                    gate_chat_generation,
                    snapshot_memory_signals,
                )

                available_gb, pressure_percent = snapshot_memory_signals()
                refusal = gate_chat_generation(available_gb, pressure_percent)
                if refusal is not None:
                    chaosengine.add_log(
                        "chat", "warning",
                        f"[{model_tag}] Memory gate refused generation: "
                        f"{refusal['code']} (avail={available_gb:.1f} GB, "
                        f"pressure={pressure_percent:.0f}%).",
                    )
                    with chaosengine._lock:
                        # Roll back the optimistic user message we appended
                        # earlier so the refusal looks like the request never
                        # happened, matching the existing RuntimeError path.
                        if (session["messages"]
                                and session["messages"][-1].get("role") == "user"
                                and session["messages"][-1].get("text") == request.prompt):
                            session["messages"].pop()
                            session["updatedAt"] = chaosengine._time_label()
                            chaosengine._persist_sessions()
                        chaosengine.active_requests = max(0, chaosengine.active_requests - 1)
                    yield f"data: {json.dumps({'error': refusal['message']})}\n\n"
                    return
            except Exception as exc:
                # Gate failure must not block legitimate generations. Log and
                # continue — better to risk a possible OOM than to refuse
                # everything when psutil glitches.
                chaosengine.add_log(
                    "chat", "warning",
                    f"[{model_tag}] Memory gate skipped due to error: {exc}",
                )

            yield f"data: {json.dumps({'phase': 'prompt_eval'})}\n\n"

            # Phase 2.0.5-D: output-length runaway guard. Abort the generation
            # if accumulated visible text exceeds the user's max_tokens budget
            # by 1.5×, which catches decoder loops that ignore the EOS token
            # (a known failure mode on certain quantised models). Char count
            # is a fast proxy — average ~4 chars per token across English +
            # markdown code, so the threshold is `max_tokens * 6` chars.
            runaway_char_budget = max(2000, int(request.maxTokens) * 6)
            runaway_triggered = False
            runaway_loop_reason: str | None = None

            # Phase 2.0.5-F: per-stream repetition / reasoning-loop guard for
            # the llama.cpp path. The MLX worker has run this guard inside the
            # subprocess for a while; the llama-server REST stream had no
            # equivalent and a runaway model could decode tokens indefinitely
            # against a paused UI. Same RunawayGuard module both paths use.
            from backend_service.runaway_guard import RunawayGuard as _RunawayGuard

            llama_path_guard = _RunawayGuard()

            # Phase 2.0.5-C: tok/s floor monitor. After the model has
            # produced output for a 30-second window, check the rolling
            # decode rate. Falling below 0.3 tok/s for that long usually
            # means thermal throttle, GPU stall, or a corrupted model
            # state — none of which recovers on its own. Abort with a
            # diagnostic so the user can switch model / cool down /
            # restart the worker.
            TOKS_FLOOR_WINDOW_S = 30.0
            TOKS_FLOOR_MIN = 0.3
            window_started_at: float | None = None
            window_tokens = 0
            stall_triggered = False

            # Phase 2.0.5-G: in-stream panic monitor. While a generation
            # is in flight, sample memory every PANIC_SAMPLE_INTERVAL_S
            # and emit a `panic` SSE event when free RAM crosses the
            # critical floor or pressure goes critical. The front-end
            # renders a non-blocking banner offering Cancel / Unload
            # warm / Continue. Generation is NOT auto-cancelled here —
            # that's the user's call. The stricter pre-flight gate
            # (Phase 2.0.5-B) blocks tight starts, this catches mid-
            # flight degradation as KV cache or other activity grows.
            PANIC_SAMPLE_INTERVAL_S = 5.0
            PANIC_AVAILABLE_FLOOR_GB = 0.5
            PANIC_PRESSURE_CEILING = 96.0
            last_panic_sample_at: float | None = None
            panic_emitted = False
            # Phase 2.0.5-I: thermal pressure watch. `pmset -g therm` on
            # macOS reports warning levels when CPU/GPU is throttling.
            # We surface the first transition to "critical" via a SSE
            # event so the user sees why decode just slowed. Linux /
            # Windows: read returns None and this watch is a no-op.
            thermal_warning_emitted = False

            def _maybe_emit_generating_phase() -> str:
                nonlocal phase_first_output_seen, ttft_seconds
                if phase_first_output_seen:
                    return ""
                phase_first_output_seen = True
                ttft_seconds = round(time.perf_counter() - gen_start, 3)
                return f"data: {json.dumps({'phase': 'generating', 'ttftSeconds': ttft_seconds})}\n\n"

            try:
                if enable_tools:
                    from backend_service.agent import run_agent_loop_streaming
                    for event in run_agent_loop_streaming(
                        generate_fn=chaosengine.runtime.generate,
                        stream_generate_fn=chaosengine.runtime.stream_generate,
                        prompt=request.prompt, history=history,
                        system_prompt=effective_system_prompt,
                        max_tokens=request.maxTokens, temperature=request.temperature,
                        images=request.images,
                        available_tools=available_tools,
                    ):
                        if chaosengine.is_chat_cancel_requested(session_id_for_cancel):
                            cancelled = True
                            break
                        if "token" in event:
                            phase_event = _maybe_emit_generating_phase()
                            if phase_event:
                                yield phase_event
                            full_text += event["token"]
                            yield f"data: {json.dumps({'token': event['token']})}\n\n"
                            if len(full_text) > runaway_char_budget:
                                runaway_triggered = True
                                cancelled = True
                                break
                        elif "tool_call_start" in event:
                            phase_event = _maybe_emit_generating_phase()
                            if phase_event:
                                yield phase_event
                            yield f"data: {json.dumps({'toolCallStart': event['tool_call_start']})}\n\n"
                        elif "tool_call_result" in event:
                            agent_tool_calls.append(event["tool_call_result"])
                            yield f"data: {json.dumps({'toolCallResult': event['tool_call_result']})}\n\n"
                        elif event.get("done"):
                            # Agent loop finished — handled below
                            pass
                else:
                    for chunk in chaosengine.runtime.stream_generate(
                        prompt=request.prompt, history=history,
                        system_prompt=effective_system_prompt,
                        max_tokens=request.maxTokens, temperature=request.temperature,
                        images=request.images,
                        thinking_mode=effective_thinking_mode,
                        samplers=_build_sampler_overrides(request),
                        reasoning_effort=request.reasoningEffort,
                        json_schema=request.jsonSchema,
                    ):
                        if chaosengine.is_chat_cancel_requested(session_id_for_cancel):
                            cancelled = True
                            break
                        if chunk.reasoning:
                            phase_event = _maybe_emit_generating_phase()
                            if phase_event:
                                yield phase_event
                            full_reasoning += chunk.reasoning
                            yield f"data: {json.dumps({'reasoning': chunk.reasoning})}\n\n"
                        if chunk.reasoning_done:
                            yield f"data: {json.dumps({'reasoningDone': True})}\n\n"
                        if chunk.text:
                            phase_event = _maybe_emit_generating_phase()
                            if phase_event:
                                yield phase_event
                            full_text += chunk.text
                            yield f"data: {json.dumps({'token': chunk.text})}\n\n"
                            if len(full_text) > runaway_char_budget:
                                runaway_triggered = True
                                cancelled = True
                                break
                            # Phase 2.0.5-F: feed loop / repetition guard.
                            try:
                                llama_path_guard.feed(chunk.text)
                            except RuntimeError as guard_exc:
                                runaway_triggered = True
                                runaway_loop_reason = str(guard_exc)
                                cancelled = True
                                break
                            # Phase 2.0.5-C: tok/s floor sampling. Each
                            # chunk roughly maps to one token from the
                            # SSE stream; chunk count is a workable proxy.
                            now = time.perf_counter()
                            if window_started_at is None:
                                window_started_at = now
                                window_tokens = 0
                            window_tokens += 1
                            if now - window_started_at >= TOKS_FLOOR_WINDOW_S:
                                rate = window_tokens / max(1e-6, now - window_started_at)
                                if rate < TOKS_FLOOR_MIN:
                                    stall_triggered = True
                                    cancelled = True
                                    runaway_loop_reason = (
                                        f"Decode stalled at {rate:.2f} tok/s "
                                        f"for {TOKS_FLOOR_WINDOW_S:.0f}s — "
                                        "likely thermal throttle, GPU stall, "
                                        "or worker deadlock. Aborting."
                                    )
                                    break
                                window_started_at = now
                                window_tokens = 0
                            # Phase 2.0.5-G + I: panic + thermal monitors.
                            # Sampled at PANIC_SAMPLE_INTERVAL_S together to
                            # keep subprocess / psutil cost bounded. Each
                            # emits at most once per turn.
                            if (
                                (not panic_emitted or not thermal_warning_emitted)
                                and (
                                    last_panic_sample_at is None
                                    or now - last_panic_sample_at >= PANIC_SAMPLE_INTERVAL_S
                                )
                            ):
                                last_panic_sample_at = now
                                if not panic_emitted:
                                    try:
                                        from backend_service.helpers.memory_gate import (
                                            snapshot_memory_signals as _panic_snapshot,
                                        )
                                        p_avail, p_pressure = _panic_snapshot()
                                        if (
                                            p_avail < PANIC_AVAILABLE_FLOOR_GB
                                            or p_pressure > PANIC_PRESSURE_CEILING
                                        ):
                                            panic_emitted = True
                                            chaosengine.add_log(
                                                "chat", "warning",
                                                f"[{model_tag}] Panic: avail="
                                                f"{p_avail:.1f} GB, "
                                                f"pressure={p_pressure:.0f}%.",
                                            )
                                            yield (
                                                "data: "
                                                + json.dumps({
                                                    "panic": True,
                                                    "availableGb": p_avail,
                                                    "pressurePercent": p_pressure,
                                                    "message": (
                                                        "System memory critical mid-"
                                                        "generation. Consider cancelling "
                                                        "this turn or unloading warm "
                                                        "models before retrying."
                                                    ),
                                                })
                                                + "\n\n"
                                            )
                                    except Exception as panic_exc:
                                        chaosengine.add_log(
                                            "chat", "warning",
                                            f"[{model_tag}] Panic sample skipped: {panic_exc}",
                                        )
                                if not thermal_warning_emitted:
                                    try:
                                        from backend_service.helpers.thermal import (
                                            read_thermal_state,
                                        )
                                        thermal_state = read_thermal_state()
                                        if thermal_state == "critical":
                                            thermal_warning_emitted = True
                                            chaosengine.add_log(
                                                "chat", "warning",
                                                f"[{model_tag}] Thermal warning: critical.",
                                            )
                                            yield (
                                                "data: "
                                                + json.dumps({
                                                    "thermalWarning": True,
                                                    "state": thermal_state,
                                                    "message": (
                                                        "System is thermally throttling. "
                                                        "Decode speed will drop until the "
                                                        "machine cools. Consider pausing "
                                                        "and retrying after a cooldown."
                                                    ),
                                                })
                                                + "\n\n"
                                            )
                                    except Exception as thermal_exc:
                                        chaosengine.add_log(
                                            "chat", "warning",
                                            f"[{model_tag}] Thermal sample skipped: {thermal_exc}",
                                        )
                        if chunk.done:
                            final_chunk = chunk
            except RuntimeError as exc:
                with chaosengine._lock:
                    if (session["messages"]
                            and session["messages"][-1].get("role") == "user"
                            and session["messages"][-1].get("text") == request.prompt):
                        session["messages"].pop()
                        session["updatedAt"] = chaosengine._time_label()
                        chaosengine._persist_sessions()
                    chaosengine.active_requests = max(0, chaosengine.active_requests - 1)
                    chaosengine.add_log("chat", "error", f"[{model_tag}] Streaming failed: {exc}")
                chaosengine.clear_chat_cancel(session_id_for_cancel)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return
            finally:
                chaosengine.clear_chat_cancel(session_id_for_cancel)

            if cancelled:
                yield f"data: {json.dumps({'cancelled': True})}\n\n"
                if runaway_loop_reason is not None:
                    chaosengine.add_log(
                        "chat", "warning",
                        f"[{model_tag}] {runaway_loop_reason} "
                        f"(after {len(full_text)} chars).",
                    )
                elif runaway_triggered:
                    chaosengine.add_log(
                        "chat", "warning",
                        f"[{model_tag}] Output runaway guard tripped at "
                        f"{len(full_text)} chars (budget {runaway_char_budget}); "
                        "stream aborted to prevent decoder loop.",
                    )
                else:
                    chaosengine.add_log("chat", "info", f"[{model_tag}] Generation cancelled by user.")

            gen_elapsed = round(time.perf_counter() - gen_start, 2)
            with chaosengine._lock:
                chaosengine.active_requests = max(0, chaosengine.active_requests - 1)
                chaosengine.requests_served += 1

                tok_s = final_chunk.tok_s if final_chunk else 0
                prompt_tokens = final_chunk.prompt_tokens if final_chunk else 0
                completion_tokens = final_chunk.completion_tokens if final_chunk else 0
                if (not tok_s or tok_s == 0) and completion_tokens > 0 and gen_elapsed > 0:
                    tok_s = round(completion_tokens / gen_elapsed, 1)

                metrics = chaosengine._stream_assistant_metrics_payload(
                    final_chunk=final_chunk,
                    tok_s=tok_s,
                    response_seconds=gen_elapsed,
                    requested_runtime=requested_runtime,
                    ttft_seconds=ttft_seconds,
                )
                if agent_tool_calls:
                    metrics["toolCalls"] = agent_tool_calls

                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "text": full_text,
                    "metrics": metrics,
                }
                if full_reasoning:
                    assistant_message["reasoning"] = full_reasoning
                if agent_tool_calls:
                    assistant_message["toolCalls"] = agent_tool_calls
                if stream_rag_citations:
                    assistant_message["citations"] = stream_rag_citations
                session["messages"].append(assistant_message)
                session["updatedAt"] = chaosengine._time_label()
                chaosengine._promote_session(session)
                chaosengine.add_log(
                    "chat", "info",
                    f"[{model_tag}] Finished streaming -- {completion_tokens} tokens in {gen_elapsed}s ({tok_s} tok/s).",
                )
                chaosengine._persist_sessions()

                done_payload = {
                    "done": True,
                    "session": session,
                    "assistant": assistant_message,
                    "runtime": chaosengine.runtime.status(
                        active_requests=chaosengine.active_requests,
                        requests_served=chaosengine.requests_served,
                    ),
                }
                if cancelled:
                    done_payload["cancelled"] = True
            yield f"data: {json.dumps(done_payload)}\n\n"

        return StreamingResponse(
            _sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    def start_download(self, repo: str, allow_patterns: list[str] | None = None) -> dict[str, Any]:
        from backend_service.helpers.huggingface import (
            _friendly_hf_download_error,
            _hf_repo_downloaded_bytes,
            _hf_repo_preflight_size_gb,
        )

        if not _HF_REPO_PATTERN.match(repo):
            raise HTTPException(status_code=400, detail="Invalid repo format. Expected 'owner/model-name'.")
        if repo in self._downloads and self._downloads[repo].get("state") == "downloading":
            return self._downloads[repo]

        total_gb = _known_repo_size_gb(repo)
        downloaded_gb = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
        try:
            preflight_total_gb = _hf_repo_preflight_size_gb(repo)
        except Exception as exc:
            friendly_error = _friendly_image_download_error(
                repo,
                _friendly_hf_download_error(repo, str(exc)),
            )
            failed_status = {
                "repo": repo,
                "state": "failed",
                "progress": 0.0,
                "downloadedGb": downloaded_gb,
                "totalGb": total_gb,
                "error": friendly_error,
            }
            with self._lock:
                self._downloads[repo] = failed_status
                self._download_cancel.pop(repo, None)
                self._download_processes.pop(repo, None)
                self._download_tokens.pop(repo, None)
            self.add_log("library", "error", f"Download preflight failed for {repo}: {friendly_error}")
            return failed_status
        if isinstance(preflight_total_gb, (int, float)) and preflight_total_gb > 0:
            total_gb = float(preflight_total_gb)

        initial_progress = 0.0
        if isinstance(total_gb, (int, float)) and total_gb > 0 and downloaded_gb > 0:
            initial_progress = min(0.99, downloaded_gb / float(total_gb))
        elif downloaded_gb > 0:
            initial_progress = 0.01
        download_token = uuid.uuid4().hex
        self._downloads[repo] = {
            "repo": repo,
            "state": "downloading",
            "progress": initial_progress,
            "downloadedGb": downloaded_gb,
            "totalGb": total_gb,
            "error": None,
        }
        self._download_cancel[repo] = False
        self._download_tokens[repo] = download_token
        self.add_log("library", "info", f"{'Resuming' if downloaded_gb > 0 else 'Starting'} download: {repo}")

        def _download_worker():
            stop_progress = threading.Event()
            process: subprocess.Popen[str] | None = None
            process_log_path: str | None = None

            def _progress_worker() -> None:
                while not stop_progress.wait(1.0):
                    downloaded_bytes = _hf_repo_downloaded_bytes(repo)
                    downloaded_gb = _bytes_to_gb(downloaded_bytes)
                    with self._lock:
                        current = self._downloads.get(repo)
                        if (
                            current is None
                            or current.get("state") != "downloading"
                            or self._download_tokens.get(repo) != download_token
                        ):
                            return
                        current["downloadedGb"] = downloaded_gb
                        total = current.get("totalGb")
                        if isinstance(total, (int, float)) and total > 0:
                            current["progress"] = min(0.99, downloaded_gb / float(total))
                        elif downloaded_gb > 0:
                            current["progress"] = max(float(current.get("progress") or 0.0), 0.01)

            monitor = threading.Thread(target=_progress_worker, daemon=True)
            monitor.start()
            try:
                with self._lock:
                    if self._download_tokens.get(repo) != download_token:
                        return
                env = os.environ.copy()
                env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                env.setdefault("PYTHONUNBUFFERED", "1")
                # Force the standard Hub download path for app-managed downloads.
                # Xet-backed transfers can keep most activity outside the per-repo
                # cache tree we use for progress tracking, which makes large repos
                # look permanently stuck at 0-1% in the UI.
                env["HF_HUB_DISABLE_XET"] = "1"
                temp_log = tempfile.NamedTemporaryFile(
                    prefix="chaosengine-download-",
                    suffix=".log",
                    delete=False,
                )
                process_log_path = temp_log.name
                temp_log.close()
                with open(process_log_path, "w", encoding="utf-8", errors="replace") as process_log:
                    # Diffusers repos (image + video) get a component-folder
                    # allowlist so we skip legacy single-file checkpoints the
                    # pipelines never load. Both helpers return None for repos
                    # outside their catalog, so only one ever applies.
                    effective_allow_patterns = allow_patterns or (
                        _video_repo_allow_patterns(repo)
                        or _image_repo_allow_patterns(repo)
                    )
                    process = _spawn_snapshot_download(
                        repo,
                        env,
                        process_log,
                        allow_patterns=effective_allow_patterns,
                    )
                    with self._lock:
                        if self._download_tokens.get(repo) == download_token:
                            self._download_processes[repo] = process

                    while True:
                        with self._lock:
                            cancel_requested = self._download_cancel.get(repo, False)
                            token_matches = self._download_tokens.get(repo) == download_token
                        if not token_matches:
                            return
                        if cancel_requested:
                            if process.poll() is None:
                                try:
                                    process.terminate()
                                    process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait(timeout=5)
                            break
                        if process.poll() is not None:
                            break
                        time.sleep(0.5)

                returncode = process.returncode if process.returncode is not None else process.wait()
                stderr_output = _read_text_tail(process_log_path)

                with self._lock:
                    if self._download_tokens.get(repo) != download_token:
                        return
                    cancelled = self._download_cancel.get(repo, False)
                if cancelled:
                    downloaded_gb = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
                    with self._lock:
                        current = self._downloads.get(repo)
                        if current is None or self._download_tokens.get(repo) != download_token:
                            return
                        current["state"] = "cancelled"
                        current["error"] = None
                        current["downloadedGb"] = downloaded_gb
                        total = current.get("totalGb")
                        if isinstance(total, (int, float)) and total > 0:
                            current["progress"] = min(0.99, downloaded_gb / float(total))
                        elif downloaded_gb > 0:
                            current["progress"] = max(float(current.get("progress") or 0.0), 0.01)
                    return

                if returncode != 0:
                    raise RuntimeError(stderr_output or f"snapshot_download exited with status {returncode}")

                # Image catalog validation first; fall through to video so
                # a successful video download isn't flagged for missing image
                # shape. Each validator returns None for repos outside its
                # catalog.
                validation_error = (
                    _image_download_validation_error(repo)
                    or _video_download_validation_error(repo)
                )
                if validation_error:
                    with self._lock:
                        if self._download_tokens.get(repo) != download_token:
                            return
                        self._downloads[repo]["state"] = "failed"
                        self._downloads[repo]["error"] = validation_error
                        self.add_log("library", "error", validation_error)
                    return
                downloaded_gb = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
                with self._lock:
                    if self._download_tokens.get(repo) != download_token:
                        return
                    self._downloads[repo]["state"] = "completed"
                    self._downloads[repo]["progress"] = 1.0
                    self._downloads[repo]["downloadedGb"] = downloaded_gb
                    if downloaded_gb > 0:
                        current_total = self._downloads[repo].get("totalGb")
                        if not isinstance(current_total, (int, float)) or current_total <= 0:
                            self._downloads[repo]["totalGb"] = downloaded_gb
                        else:
                            self._downloads[repo]["totalGb"] = max(float(current_total), downloaded_gb)
                    self._library_cache = None
                    self.add_log("library", "info", f"Download completed: {repo}")
            except Exception as exc:
                with self._lock:
                    if self._download_tokens.get(repo) != download_token:
                        return
                    self._downloads[repo]["state"] = "failed"
                    friendly_error = _friendly_image_download_error(
                        repo,
                        _friendly_hf_download_error(repo, str(exc)),
                    )
                    self._downloads[repo]["error"] = friendly_error
                    self.add_log("library", "error", f"Download failed for {repo}: {friendly_error}")
            finally:
                stop_progress.set()
                monitor.join(timeout=1.0)
                with self._lock:
                    if process is not None and self._download_processes.get(repo) is process:
                        self._download_processes.pop(repo, None)
                    if self._download_tokens.get(repo) == download_token and self._downloads.get(repo, {}).get("state") != "downloading":
                        self._download_tokens.pop(repo, None)
                        self._download_cancel.pop(repo, None)
                if process_log_path:
                    try:
                        os.unlink(process_log_path)
                    except OSError:
                        pass

        t = threading.Thread(target=_download_worker, daemon=True)
        t.start()
        return self._downloads[repo]

    def download_status(self) -> list[dict[str, Any]]:
        return list(self._downloads.values())

    @staticmethod
    def _loaded_model_matches_repo_cache(loaded: Any, repo: str, repo_cache_dir: Path) -> bool:
        if loaded is None:
            return False
        if repo in {
            getattr(loaded, "ref", None),
            getattr(loaded, "runtimeTarget", None),
        }:
            return True
        path_value = getattr(loaded, "path", None)
        if not path_value:
            return False
        try:
            resolved = Path(str(path_value)).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        try:
            return resolved == repo_cache_dir or resolved.is_relative_to(repo_cache_dir)
        except AttributeError:
            return False

    def _unload_repo_from_runtimes(self, repo: str, repo_cache_dir: Path) -> None:
        from contextlib import nullcontext

        try:
            loaded = getattr(self.runtime, "loaded_model", None)
            if self._loaded_model_matches_repo_cache(loaded, repo, repo_cache_dir):
                self.runtime.unload_model()
        except Exception:
            pass

        try:
            self.image_runtime.unload(repo)
        except Exception:
            pass

        try:
            self.video_runtime.unload(repo)
        except Exception:
            pass

        warm_pool = getattr(self.runtime, "_warm_pool", None)
        if not isinstance(warm_pool, dict):
            return
        pool_lock = getattr(self.runtime, "_pool_lock", None)
        pool_context = pool_lock if hasattr(pool_lock, "__enter__") and hasattr(pool_lock, "__exit__") else nullcontext()
        with pool_context:
            stale_keys = [
                key
                for key, (_engine, info) in warm_pool.items()
                if self._loaded_model_matches_repo_cache(info, repo, repo_cache_dir)
            ]
            for key in stale_keys:
                engine, _info = warm_pool.pop(key)
                try:
                    engine.unload_model()
                except Exception:
                    pass

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
        from backend_service.helpers.huggingface import _hf_repo_downloaded_bytes

        with self._lock:
            current = self._downloads.get(repo)
            if current is None:
                return {"repo": repo, "state": "not_found"}
            if current.get("state") == "completed":
                return current
            self._download_cancel[repo] = True
            process = self._download_processes.get(repo)

        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                pass

        downloaded_gb = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
        with self._lock:
            current = self._downloads.get(repo)
            if current is None:
                return {"repo": repo, "state": "not_found"}
            current["state"] = "cancelled"
            current["error"] = None
            current["downloadedGb"] = downloaded_gb
            total = current.get("totalGb")
            if isinstance(total, (int, float)) and total > 0:
                current["progress"] = min(0.99, downloaded_gb / float(total))
            elif downloaded_gb > 0:
                current["progress"] = max(float(current.get("progress") or 0.0), 0.01)
            self.add_log("library", "info", f"Download paused: {repo}")
            return current
        return {"repo": repo, "state": "not_found"}

    def delete_download(self, repo: str) -> dict[str, Any]:
        if not _HF_REPO_PATTERN.match(repo):
            raise HTTPException(status_code=400, detail="Invalid repo format. Expected 'owner/model-name'.")

        repo_cache_dir = _hf_repo_cache_dir(repo)
        with self._lock:
            current = self._downloads.get(repo)
            process = self._download_processes.get(repo)
            if current is not None:
                self._download_cancel[repo] = True

        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                pass

        with self._lock:
            self._unload_repo_from_runtimes(repo, repo_cache_dir)

            removed_local_data = False
            try:
                if repo_cache_dir.exists():
                    import shutil as _shutil

                    _shutil.rmtree(repo_cache_dir)
                    removed_local_data = True
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete cached download for {repo}: {exc}",
                ) from exc

            removed_status = repo in self._downloads
            self._downloads.pop(repo, None)
            self._download_cancel.pop(repo, None)
            self._download_processes.pop(repo, None)
            self._download_tokens.pop(repo, None)
            self._library_cache = None

            if removed_local_data or removed_status:
                action = "download data" if removed_local_data else "download record"
                self.add_log("library", "info", f"Deleted {action}: {repo}")
                return {"repo": repo, "state": "deleted"}

            return {"repo": repo, "state": "not_found"}

    def server_status(self) -> dict[str, Any]:
        from backend_service.app import DEFAULT_HOST

        runtime = self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served)
        loaded = runtime["loadedModel"]
        recent_orphaned_workers = runtime.get("recentOrphanedWorkers") or []
        recent_server_logs = [
            entry["message"] for entry in list(self.logs) if entry["source"] in {"runtime", "chat", "server"}
        ][:3]
        status = "running" if runtime["serverReady"] else "idle"
        remote_enabled = DEFAULT_HOST != "127.0.0.1"
        localhost_url = f"http://127.0.0.1:{self.server_port}/v1"
        lan_urls = [f"http://{address}:{self.server_port}/v1" for address in _local_ipv4_addresses()] if remote_enabled else []
        base_url = localhost_url
        preferred_port = self.settings["preferredServerPort"]
        port_note = (
            f"Preferred API port is {preferred_port}. Restart the API service to apply it."
            if preferred_port != self.server_port
            else (
                "Remote access is enabled for local-network clients. Allow incoming connections in your firewall if prompted."
                if remote_enabled
                else "Third-party tools on this machine can target the displayed localhost URL."
            )
        )
        loading = None
        if self._loading_state is not None:
            elapsed = time.time() - self._loading_state["startedAt"]
            loading = {
                "modelName": self._loading_state["modelName"],
                "stage": self._loading_state["stage"],
                "elapsedSeconds": round(elapsed, 1),
                "progress": self._loading_state.get("progress"),
                "progressPercent": self._loading_state.get("progressPercent"),
                "progressPhase": self._loading_state.get("progressPhase"),
                "progressMessage": self._loading_state.get("progressMessage"),
                "recentLogLines": list(self._loading_state.get("recentLogLines") or []),
            }

        return {
            "status": status,
            "baseUrl": base_url,
            "localhostUrl": localhost_url,
            "lanUrls": lan_urls,
            "bindHost": DEFAULT_HOST,
            "remoteAccessActive": remote_enabled,
            "port": self.server_port,
            "activeConnections": runtime["activeRequests"],
            "concurrentRequests": runtime["activeRequests"],
            "requestsServed": runtime["requestsServed"],
            "loadedModelName": loaded["name"] if loaded else None,
            "loading": loading,
            "recentOrphanedWorkers": recent_orphaned_workers,
            "logTail": recent_server_logs or [
                "Load a model to make the OpenAI-compatible local API ready for external tools.",
                "Ports and concurrency are configurable in Settings.",
                port_note,
            ],
        }

    def workspace(self) -> dict[str, Any]:
        from backend_service.app import compute_cache_preview

        system_stats = self._system_snapshot()
        try:
            loaded_name = self.runtime.loaded_model.name if self.runtime.loaded_model else None
            loaded_engine = self.runtime.engine.engine_name if self.runtime.engine else None
            warm_entries = [
                (engine.engine_name, info.name)
                for engine, info in self.runtime._warm_pool.values()
            ]
            procs = list(system_stats.get("runningLlmProcesses") or [])
            seen_pids = {
                int(pid)
                for pid in (proc.get("pid") for proc in procs)
                if isinstance(pid, int)
            }
            proc_by_pid = {
                int(proc["pid"]): proc
                for proc in procs
                if isinstance(proc.get("pid"), int)
            }

            tracked_runtime_processes: list[tuple[int, str]] = []
            active_pid_getter = getattr(self.runtime.engine, "process_pid", None) if self.runtime.engine else None
            active_pid = active_pid_getter() if callable(active_pid_getter) else None
            if isinstance(active_pid, int):
                active_kind = "other"
                if loaded_engine == "mlx":
                    active_kind = "mlx_worker"
                elif loaded_engine == "llama.cpp":
                    active_kind = "llama_server"
                tracked_runtime_processes.append((active_pid, active_kind))

            for engine, _info in self.runtime._warm_pool.values():
                pid_getter = getattr(engine, "process_pid", None)
                warm_pid = pid_getter() if callable(pid_getter) else None
                if not isinstance(warm_pid, int) or warm_pid in seen_pids:
                    continue
                warm_kind = "other"
                if engine.engine_name == "mlx":
                    warm_kind = "mlx_worker"
                elif engine.engine_name == "llama.cpp":
                    warm_kind = "llama_server"
                tracked_runtime_processes.append((warm_pid, warm_kind))

            for pid, kind in tracked_runtime_processes:
                described = _describe_process(pid, kind_hint=kind, owner_hint="ChaosEngineAI")
                if described is None:
                    continue
                existing = proc_by_pid.get(pid)
                if existing is not None:
                    model_name = existing.get("modelName")
                    model_status = existing.get("modelStatus")
                    existing.clear()
                    existing.update(described)
                    if model_name:
                        existing["modelName"] = model_name
                    if model_status:
                        existing["modelStatus"] = model_status
                    continue
                procs.append(described)
                seen_pids.add(pid)
                proc_by_pid[pid] = described

            mlx_workers = [p for p in procs if p.get("kind") == "mlx_worker"]
            llama_servers = [p for p in procs if p.get("kind") == "llama_server"]

            assigned_loaded = False
            if loaded_name and loaded_engine == "mlx" and mlx_workers:
                mlx_workers[0]["modelName"] = loaded_name
                mlx_workers[0]["modelStatus"] = "active"
                assigned_loaded = True
            elif loaded_name and loaded_engine == "llama.cpp" and llama_servers:
                llama_servers[0]["modelName"] = loaded_name
                llama_servers[0]["modelStatus"] = "active"
                assigned_loaded = True

            if loaded_name and not assigned_loaded:
                for proc in procs:
                    if proc.get("owner") == "ChaosEngineAI" and not proc.get("modelName"):
                        proc["modelName"] = loaded_name
                        proc["modelStatus"] = "active"
                        break

            warm_mlx = [
                name for engine, name in warm_entries
                if engine == "mlx" and name != loaded_name
            ]
            warm_llama = [
                name for engine, name in warm_entries
                if engine == "llama.cpp" and name != loaded_name
            ]
            for proc in mlx_workers[1:]:
                if warm_mlx and not proc.get("modelName"):
                    proc["modelName"] = warm_mlx.pop(0)
                    proc["modelStatus"] = "warm"
            for proc in llama_servers[1:]:
                if warm_llama and not proc.get("modelName"):
                    proc["modelName"] = warm_llama.pop(0)
                    proc["modelStatus"] = "warm"

            def _proc_rank(proc: dict[str, Any]) -> tuple[int, float, float]:
                status = proc.get("modelStatus")
                if status == "active":
                    priority = 0
                elif status == "warm":
                    priority = 1
                else:
                    priority = 2
                return (
                    priority,
                    -float(proc.get("memoryGb", 0.0)),
                    -float(proc.get("cpuPercent", 0.0)),
                )

            procs.sort(key=_proc_rank)
            system_stats["runningLlmProcesses"] = procs[:12]
        except Exception:
            pass

        try:
            disk_info = _get_disk_usage_for_models(self.settings)
            if disk_info:
                system_stats["diskFreeGb"] = disk_info["freeGb"]
                system_stats["diskTotalGb"] = disk_info["totalGb"]
                system_stats["diskUsedGb"] = disk_info["usedGb"]
                system_stats["diskPath"] = disk_info.get("path")
        except Exception:
            pass
        library = self._library()
        recommendation = _best_fit_recommendation(system_stats)
        launch_preferences = self._launch_preferences()
        return {
            "system": system_stats,
            "recommendation": recommendation,
            "featuredModels": _model_family_payloads(system_stats, library),
            "library": library,
            "libraryStatus": "ready" if self._library_scan_done.is_set() else "scanning",
            "settings": self._settings_payload(library),
            "chatSessions": self.chat_sessions,
            "runtime": self.runtime.status(
                active_requests=self.active_requests,
                requests_served=self.requests_served,
            ),
            "server": self.server_status(),
            "benchmarks": self.benchmark_runs,
            "logs": [entry for entry in self.logs if entry.get("level") != "debug"],
            "activity": list(self.activity),
            "preview": compute_cache_preview(
                bits=launch_preferences["cacheBits"],
                fp16_layers=launch_preferences["fp16Layers"],
                context_tokens=launch_preferences["contextTokens"],
                system_stats=system_stats,
            ),
            "quickActions": [
                "Online Models",
                "New Thread",
                "Start Server",
                "Convert to MLX",
                "Run Benchmark",
                "Open Logs",
            ],
        }

    def openai_models(self) -> dict[str, Any]:
        runtime = self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served)
        loaded = runtime["loadedModel"]
        if loaded is None:
            return {"object": "list", "data": []}
        created = int(time.time())
        seen: set[str] = set()
        data: list[dict[str, Any]] = []
        for model_id in (loaded["ref"], loaded.get("runtimeTarget")):
            if model_id and model_id not in seen:
                seen.add(model_id)
                data.append({
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "chaosengine",
                })
        return {"object": "list", "data": data}

    def openai_embeddings(self, request: OpenAIEmbeddingsRequest) -> dict[str, Any]:
        """Phase 2.13: OpenAI-compatible embeddings endpoint.

        Routes through the bundled GGUF embedding model (Phase 2.6).
        Returns a 503 when no embedding client is available; returns
        the OpenAI-shaped response shape on success so external
        scripts can drop us in for OpenAI without code changes.
        """
        from backend_service.app import DOCUMENTS_DIR
        from backend_service.rag import resolve_embedding_client
        from backend_service.rag.embedding_client import EmbeddingClientUnavailable

        client = resolve_embedding_client(DOCUMENTS_DIR.parent)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No embedding model is configured. Set CHAOSENGINE_EMBEDDING_MODEL "
                    "or drop a *.gguf into <dataDir>/embeddings/."
                ),
            )

        if isinstance(request.input, str):
            inputs = [request.input]
        else:
            inputs = list(request.input)

        if not inputs:
            raise HTTPException(status_code=400, detail="`input` must be a non-empty string or list of strings.")

        try:
            vectors = client.embed_batch(inputs)
        except EmbeddingClientUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        # Truncate per OpenAI's `dimensions` parameter when set. We don't
        # re-normalise after truncation; the bundled model is already
        # L2-normalised end-to-end, so cosine similarity stays well-defined.
        if request.dimensions is not None:
            vectors = [vec[: request.dimensions] for vec in vectors]

        prompt_tokens = sum(max(1, len(text.split())) for text in inputs)
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": vec,
                    "index": idx,
                }
                for idx, vec in enumerate(vectors)
            ],
            "model": request.model or "chaosengine-embed",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    def openai_chat_completion(self, request: OpenAIChatCompletionRequest) -> dict[str, Any] | StreamingResponse:
        if not request.messages:
            raise HTTPException(status_code=400, detail="At least one message is required.")

        last_user = None
        last_user_images: list[str] = []
        history: list[dict[str, Any]] = []
        system_prompt = None
        for message in request.messages:
            if isinstance(message.content, list):
                text_parts = []
                for part in message.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(str(part.get("text", "")))
                        elif part.get("type") == "image_url":
                            url = (part.get("image_url") or {}).get("url", "")
                            if url.startswith("data:") and ";base64," in url:
                                last_user_images.append(url.split(";base64,", 1)[1])
                content = " ".join(text_parts) if text_parts else ""
            else:
                content = str(message.content) if message.content is not None else ""

            if message.role == "system" and system_prompt is None:
                system_prompt = content
            elif message.role == "user":
                last_user = content
                history.append({"role": "user", "text": content})
            elif message.role == "assistant":
                if message.tool_calls:
                    history.append({"role": "assistant", "text": content, "tool_calls": message.tool_calls})
                else:
                    history.append({"role": "assistant", "text": content})
            elif message.role == "tool":
                history.append({"role": "tool", "text": content, "tool_call_id": message.tool_call_id})

        if last_user is None:
            raise HTTPException(status_code=400, detail="A user message is required.")

        msg_count = len(request.messages)

        with self._lock:
            launch_preferences = self._launch_preferences()
            if self.runtime.loaded_model is None and request.model:
                self.add_log("server", "info", f"[{request.model}] Auto-loading model for /v1/chat/completions...")
                self.load_model(
                    LoadModelRequest(
                        modelRef=request.model,
                        modelName=request.model,
                        canonicalRepo=self._resolve_canonical_repo(
                            model_ref=request.model,
                            path=None,
                            canonical_repo=None,
                        ),
                        source="openai",
                        backend="auto",
                        cacheStrategy=launch_preferences["cacheStrategy"],
                        cacheBits=launch_preferences["cacheBits"],
                        fp16Layers=launch_preferences["fp16Layers"],
                        fusedAttention=launch_preferences["fusedAttention"],
                        fitModelInMemory=launch_preferences["fitModelInMemory"],
                        contextTokens=launch_preferences["contextTokens"],
                    )
                )
            if self.runtime.loaded_model is None:
                raise HTTPException(status_code=409, detail="Load a model before calling /v1/chat/completions.")

            try:
                target_engine, target_info = self.runtime.get_engine_for_request(request.model)
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            self.active_requests += 1
            model_ref = target_info.ref
            model_tag = target_info.name
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            self.add_log("server", "info", f"[{model_tag}] Running chat completion on conversation with {msg_count} messages.")

        # Phase 2.13: build a sampler dict from OpenAI-shaped fields. The
        # runtime accepts the same llama-server key names so we map field
        # → key here once and pass the dict to both stream + non-stream
        # paths. None values drop out so they don't override server
        # defaults.
        oai_samplers: dict[str, Any] = {}
        if request.top_p is not None:
            oai_samplers["top_p"] = request.top_p
        if request.top_k is not None:
            oai_samplers["top_k"] = request.top_k
        if request.frequency_penalty is not None:
            oai_samplers["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            oai_samplers["presence_penalty"] = request.presence_penalty
        if request.seed is not None:
            oai_samplers["seed"] = request.seed
        if request.stop is not None:
            oai_samplers["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]

        # Phase 2.13: pull a JSON schema out of OpenAI's response_format
        # envelope so the constrained-decode path lights up. Anything
        # other than `json_schema` → no constraint (json_object would
        # require a different code path llama-server already handles
        # via response_format= but we don't surface that here).
        oai_json_schema: dict[str, Any] | None = None
        if isinstance(request.response_format, dict):
            rf_type = request.response_format.get("type")
            if rf_type == "json_schema":
                schema_envelope = request.response_format.get("json_schema") or {}
                schema_obj = schema_envelope.get("schema")
                if isinstance(schema_obj, dict):
                    oai_json_schema = schema_obj

        if request.stream:
            chaosengine = self

            def _stream_chunks():
                stream_start = time.perf_counter()
                with chaosengine._lock:
                    chaosengine.add_log("server", "info", f"[{model_tag}] Streaming response...")
                token_count = 0
                prompt_tokens = 0
                try:
                    first = True
                    for chunk in chaosengine.runtime.stream_generate(
                        prompt=last_user,
                        history=history[:-1],
                        system_prompt=system_prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        images=last_user_images or None,
                        tools=request.tools,
                        engine=target_engine,
                        samplers=oai_samplers or None,
                        json_schema=oai_json_schema,
                    ):
                        if chunk.text:
                            token_count += 1
                            delta = {"content": chunk.text}
                            if first:
                                delta["role"] = "assistant"
                                first = False
                            sse_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_ref,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(sse_chunk)}\n\n"
                        if chunk.done:
                            if hasattr(chunk, "prompt_tokens") and chunk.prompt_tokens:
                                prompt_tokens = chunk.prompt_tokens
                            if hasattr(chunk, "completion_tokens") and chunk.completion_tokens:
                                token_count = chunk.completion_tokens
                            done_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_ref,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": chunk.finish_reason or "stop",
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(done_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except RuntimeError as exc:
                    with chaosengine._lock:
                        chaosengine.add_log("server", "error", f"[{model_tag}] Streaming failed: {exc}")
                finally:
                    elapsed = round(time.perf_counter() - stream_start, 2)
                    tok_s = round(token_count / elapsed, 1) if elapsed > 0 else 0
                    with chaosengine._lock:
                        chaosengine.active_requests = max(0, chaosengine.active_requests - 1)
                        chaosengine.requests_served += 1
                        chaosengine.add_log(
                            "server", "info",
                            f"[{model_tag}] Finished streaming response -- {token_count} tokens in {elapsed}s "
                            f"({tok_s} tok/s{f', {prompt_tokens} prompt tokens' if prompt_tokens else ''}).",
                        )

            return StreamingResponse(
                _stream_chunks(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        with self._lock:
            self.add_log("server", "info", f"[{model_tag}] Generating response...")
        gen_start = time.perf_counter()
        try:
            result = self.runtime.generate(
                prompt=last_user,
                history=history[:-1],
                system_prompt=system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                images=last_user_images or None,
                tools=request.tools,
                engine=target_engine,
                samplers=oai_samplers or None,
                json_schema=oai_json_schema,
            )
        except RuntimeError as exc:
            with self._lock:
                self.active_requests = max(0, self.active_requests - 1)
                self.add_log("server", "error", f"[{model_tag}] Generation failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        gen_elapsed = round(time.perf_counter() - gen_start, 2)
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.requests_served += 1
            self.add_log(
                "server", "info",
                f"[{model_tag}] Finished response -- {result.completionTokens} tokens in {gen_elapsed}s "
                f"({result.tokS} tok/s, {result.promptTokens} prompt tokens).",
            )

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_ref,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": result.finishReason,
                        "message": {
                            "role": "assistant",
                            "content": result.text,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": result.promptTokens,
                    "completion_tokens": result.completionTokens,
                    "total_tokens": result.totalTokens,
                },
            }
