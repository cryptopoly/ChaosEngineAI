"""Model lifecycle (load / unload / convert / reveal / delete) for ``ChaosEngineState``.

Five helpers lifted out of ``state/__init__.py``:

* ``load_model`` (~228 LOC) — resolve catalog + library entry, validate
  the model is downloadable / not broken, decide whether to apply a
  cache profile change in-place or trigger a full reload, evict warm
  pool when speculative decoding is requested, and dispatch through
  ``runtime.load_model`` (or ``runtime.update_profile``) with the
  progress callback that mirrors phase / percent / message into the
  loading state for the UI.
* ``unload_model`` — drop the active model or evict a warm entry by
  ``ref``.
* ``convert_model`` — drive a HF → MLX conversion through the runtime
  controller and refresh the library cache afterwards.
* ``reveal_model_path`` — open the model's containing directory in the
  OS file manager.
* ``delete_model_path`` — unload the active model when the path is in
  scope, then ``rmtree`` (or ``unlink``) the file/directory inside one
  of the configured model directories. Refuses to touch directory
  roots themselves.

All take the ``ChaosEngineState`` instance as the first argument so
the class methods stay 1-line wrappers.

Extracted as part of the v0.8.0 Phase 1a-12 refactor.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from backend_service.helpers.formatting import _parse_context_label
from backend_service.models import ConvertModelRequest, LoadModelRequest


if TYPE_CHECKING:
    from backend_service.state import ChaosEngineState


def load_model(
    state: ChaosEngineState,
    request: LoadModelRequest,
    *,
    keep_warm_previous: bool = True,
) -> dict[str, Any]:
    with state._lock:
        catalog_entry = state._find_catalog_entry(request.modelRef)
        library_entry = state._find_library_entry(request.path, request.modelRef)
        effective_canonical_repo = state._resolve_canonical_repo(
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
            # When the caller passed an explicit ``request.path`` that
            # exists on disk, trust them — the broken library entry
            # likely refers to a stale or incomplete snapshot elsewhere
            # (e.g. an empty HF cache stub) while the user is pointing
            # at the real weights. Drop the broken match and let
            # resolution fall through to the path-based code below.
            trust_path = False
            if has_path:
                try:
                    trust_path = Path(os.path.expanduser(request.path.strip())).exists()
                except OSError:
                    trust_path = False
            if trust_path:
                library_entry = None
            else:
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
            state.add_log(
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
        runtime_target, resolved_backend = state._resolve_model_target(
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
            state.runtime.loaded_model is not None
            and (
                request.modelRef in {
                    state.runtime.loaded_model.ref,
                    state.runtime.loaded_model.runtimeTarget,
                }
                or (request.path is not None and request.path == state.runtime.loaded_model.path)
            )
        )
        cache_profile_changes = (
            state._cache_profile_change_reasons(
                cache_bits=effective_cache_bits,
                fp16_layers=effective_fp16_layers,
                fused_attention=request.fusedAttention,
                cache_strategy=effective_cache_strategy,
            )
            if same_loaded_model
            else []
        )
        profile_changes = (
            state._runtime_profile_change_reasons(
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
            and state.runtime.loaded_model is not None
            and state.runtime.loaded_model.engine == "mlx"
        )
        if same_loaded_model and not profile_changes:
            if effective_canonical_repo is not None and state.runtime.loaded_model is not None:
                state.runtime.loaded_model.canonicalRepo = effective_canonical_repo
            return state.runtime.status(active_requests=state.active_requests, requests_served=state.requests_served)
        state._loading_state = {
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
            state.add_log(
                "runtime",
                "info",
                f"Applying MLX runtime profile for {display_name} ({', '.join(cache_profile_changes)}).",
            )
        elif profile_changes:
            state.add_log(
                "runtime",
                "info",
                f"Reloading {display_name} because launch settings changed ({', '.join(profile_changes)}).",
            )
        else:
            state.add_log("runtime", "info", f"Loading {display_name}...")

    def _on_load_progress(prog: dict[str, Any]) -> None:
        try:
            with state._lock:
                if state._loading_state is None:
                    return
                percent = prog.get("percent")
                phase = prog.get("phase")
                message = prog.get("message")
                state._loading_state["progressPercent"] = percent
                state._loading_state["progressPhase"] = phase
                state._loading_state["progressMessage"] = message
                state._loading_state["progress"] = percent
                if message or phase:
                    line = f"[{phase}] {message}" if phase and message else str(message or phase)
                    tail = list(state._loading_state.get("recentLogLines") or [])
                    tail.append(line)
                    if len(tail) > 5:
                        tail = tail[-5:]
                    state._loading_state["recentLogLines"] = tail
        except Exception:
            pass

    try:
        if can_apply_profile_without_reload:
            loaded = state.runtime.update_profile(
                canonical_repo=effective_canonical_repo,
                cache_strategy=effective_cache_strategy,
                cache_bits=effective_cache_bits,
                fp16_layers=effective_fp16_layers,
                fused_attention=request.fusedAttention,
            )
        else:
            warm_model_count = len(state.runtime.warm_models())
            if exclusive_memory_load and warm_model_count > 0:
                cleared = state.runtime.clear_warm_pool()
                if cleared > 0:
                    state.add_log(
                        "runtime",
                        "info",
                        f"Cleared {cleared} warm model{'s' if cleared != 1 else ''} before speculative MLX load.",
                    )
            loaded = state.runtime.load_model(
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
        with state._lock:
            state._loading_state = None
        raise

    with state._lock:
        state._loading_state = None
        loaded_cache_label = state._cache_label(
            cache_strategy=str(loaded.cacheStrategy),
            bits=int(loaded.cacheBits),
            fp16_layers=int(loaded.fp16Layers),
        )
        state.add_log("runtime", "info", f"Model loaded: {loaded.name} via {loaded.engine}.")
        state.add_activity("Model loaded", f"{loaded.name} / {loaded_cache_label}")
        return state.runtime.status(active_requests=state.active_requests, requests_served=state.requests_served)


def unload_model(state: ChaosEngineState, ref: str | None = None) -> dict[str, Any]:
    with state._lock:
        if ref:
            if state.runtime.loaded_model and ref in {
                state.runtime.loaded_model.ref,
                state.runtime.loaded_model.runtimeTarget,
                state.runtime.loaded_model.path,
                state.runtime.loaded_model.name,
            }:
                name = state.runtime.loaded_model.name
                state.runtime.unload_model()
                state.add_log("runtime", "info", f"Model unloaded: {name}.")
                state.add_activity("Model unloaded", name)
            else:
                unloaded = state.runtime.unload_warm_model_by_ref(ref)
                if unloaded:
                    state.add_log("runtime", "info", f"Warm model unloaded: {ref}.")
                    state.add_activity("Warm model unloaded", ref)
                else:
                    state.add_log("runtime", "info", f"Unload no-op: {ref} not found.")
        else:
            name = state.runtime.loaded_model.name if state.runtime.loaded_model else "No model"
            state.runtime.unload_model()
            state.add_log("runtime", "info", f"Model unloaded: {name}.")
            state.add_activity("Model unloaded", name)
        return state.runtime.status(active_requests=state.active_requests, requests_served=state.requests_served)


def convert_model(state: ChaosEngineState, request: ConvertModelRequest) -> dict[str, Any]:
    with state._lock:
        runtime_target, _ = state._resolve_model_target(
            model_ref=request.modelRef,
            path=request.path,
            backend="auto",
        )
        conversion = state.runtime.convert_model(
            source_ref=runtime_target if request.path is None else request.modelRef,
            source_path=request.path,
            output_path=request.outputPath,
            hf_repo=request.hfRepo,
            quantize=request.quantize,
            q_bits=request.qBits,
            q_group_size=request.qGroupSize,
            dtype=request.dtype,
        )
        conversion = state._conversion_details(request=request, conversion=conversion)
        state.add_log(
            "conversion",
            "info",
            f"Converted {conversion['sourceLabel']} to MLX at {conversion['outputPath']}.",
        )
        state.add_activity("Model converted", f"{conversion['sourceLabel']} -> {Path(conversion['outputPath']).name}")
        return {
            "conversion": conversion,
            "library": state._library(force=True),
            "runtime": state.runtime.status(active_requests=state.active_requests, requests_served=state.requests_served),
        }


def reveal_model_path(state: ChaosEngineState, path: str) -> dict[str, Any]:
    from backend_service.helpers.discovery import _reveal_path_in_file_manager

    with state._lock:
        target = Path(path).expanduser()
        _reveal_path_in_file_manager(target)
        resolved = str(target.resolve())
        state.add_log("library", "info", f"Revealed model path: {resolved}.")
        return {"revealed": resolved}


def delete_model_path(state: ChaosEngineState, path: str) -> dict[str, Any]:
    """Delete a local model file or directory on disk."""
    with state._lock:
        target = Path(path).expanduser()
        try:
            resolved = target.resolve(strict=True)
        except (OSError, RuntimeError):
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")

        allowed = False
        for directory in state.settings.get("modelDirectories", []):
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
            loaded = getattr(state.runtime, "loaded_model", None)
            if loaded and getattr(loaded, "path", None):
                loaded_resolved = Path(str(loaded.path)).expanduser().resolve()
                if loaded_resolved == resolved or loaded_resolved.is_relative_to(resolved):
                    state.runtime.unload_model()
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

        state.add_log("library", "info", f"Deleted model at {resolved}.")
        return {
            "deleted": str(resolved),
            "library": state._library(force=True),
        }
