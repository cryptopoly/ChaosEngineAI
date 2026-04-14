from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, RLock, Thread
from collections.abc import Callable, Iterator
from typing import Any

from backend_service.reasoning_split import (
    ThinkingStreamResult,
    ThinkingTokenFilter,
    strip_thinking_tokens as _strip_thinking_tokens,
)
from backend_service.model_resolution import resolve_dflash_target_ref

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLX_TIMEOUT_SECONDS = 120.0
# Loading large MLX models (30B+) can take much longer than a normal request,
# especially on a first-time pull from Hugging Face. Allow a generous ceiling.
MLX_LOAD_TIMEOUT_SECONDS = 1800.0
DEFAULT_LLAMA_TIMEOUT_SECONDS = 120.0
CAPABILITY_CACHE_TTL_SECONDS = 10.0
_LLAMA_HELP_CACHE: dict[str, str] = {}
_LLAMA_HELP_LOCK = RLock()


def _now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return " ".join(parts)
    return str(content or "")


def _read_text_tail(path: Path | None, limit: int = 40) -> str:
    if path is None or not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


def _friendly_llama_error(logs: str | None) -> str:
    """Translate known llama.cpp startup failures into actionable messages.

    Falls back to the original log tail when nothing matches.
    """
    if not logs:
        return "llama.cpp server exited during startup."
    lower = logs.lower()
    if "unknown model architecture" in lower:
        match = re.search(r"unknown model architecture:\s*'([^']+)'", logs)
        arch = match.group(1) if match else "this model"
        return (
            f"llama.cpp does not recognise architecture '{arch}'. Your "
            f"llama.cpp build may be too old for this model. Update it "
            f"by installing a newer llama-server binary."
        )
    if "failed to allocate" in lower or "out of memory" in lower:
        return (
            "llama.cpp ran out of memory loading this model. Try a smaller "
            "quantisation, reduce the context window, or close other apps "
            "using the GPU."
        )
    return logs


class RepeatedLineGuard:
    """Abort obviously runaway streams that emit the same long line repeatedly."""

    def __init__(self, *, min_line_length: int = 48, max_repeats: int = 6) -> None:
        self.min_line_length = min_line_length
        self.max_repeats = max_repeats
        self._buffer = ""
        self._last_line: str | None = None
        self._repeat_count = 0

    def feed(self, text: str) -> None:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._check_line(line)

    def flush(self) -> None:
        if self._buffer:
            self._check_line(self._buffer)
            self._buffer = ""

    def _check_line(self, line: str) -> None:
        normalized = " ".join(line.strip().lower().split())
        if len(normalized) < self.min_line_length:
            self._last_line = None
            self._repeat_count = 0
            return

        if normalized == self._last_line:
            self._repeat_count += 1
        else:
            self._last_line = normalized
            self._repeat_count = 1

        if self._repeat_count >= self.max_repeats:
            raise RuntimeError(
                "Stopped runaway generation after repeated identical output from the model."
            )


def _append_runtime_note(existing: str | None, extra: str) -> str:
    if not existing:
        return extra
    if extra in existing:
        return existing
    return f"{existing} {extra}"


def _llama_server_help_text(binary_path: str | None) -> str:
    if not binary_path:
        return ""
    with _LLAMA_HELP_LOCK:
        cached = _LLAMA_HELP_CACHE.get(binary_path)
        if cached is not None:
            return cached

    try:
        completed = subprocess.run(
            [binary_path, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8.0,
        )
        help_text = "\n".join(part for part in (completed.stdout, completed.stderr) if part).lower()
    except (OSError, subprocess.TimeoutExpired):
        help_text = ""

    with _LLAMA_HELP_LOCK:
        _LLAMA_HELP_CACHE[binary_path] = help_text
    return help_text


def _llama_server_supports(binary_path: str | None, option: str) -> bool:
    return option.lower() in _llama_server_help_text(binary_path)


def _json_subprocess(
    command: list[str],
    *,
    timeout: float = 15.0,
    cwd: Path = WORKSPACE_ROOT,
) -> tuple[int, dict[str, Any] | None, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (-1, None, str(exc))

    payload: dict[str, Any] | None = None
    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
    return (completed.returncode, payload, stderr or stdout)


def _find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _resolve_mlx_python() -> str:
    override = os.getenv("CHAOSENGINE_MLX_PYTHON")
    if override:
        return override
    candidate = WORKSPACE_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


# Common install locations for llama.cpp binaries that may not be in PATH
# when launched from a GUI app (Tauri doesn't inherit the user's shell profile).
_LLAMA_FALLBACK_DIRS = [
    "/opt/homebrew/bin",         # macOS ARM Homebrew
    "/usr/local/bin",            # macOS Intel Homebrew / manual
    "/usr/bin",                  # system
    str(Path.home() / ".local" / "bin"),  # pip --user installs
]


def _which_with_fallbacks(name: str) -> str | None:
    """Like shutil.which but also checks common install directories."""
    found = shutil.which(name)
    if found:
        return found
    for d in _LLAMA_FALLBACK_DIRS:
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _resolve_llama_server() -> str | None:
    override = os.getenv("CHAOSENGINE_LLAMA_SERVER")
    if override:
        return override
    return _which_with_fallbacks("llama-server")


def _resolve_llama_cli() -> str | None:
    override = os.getenv("CHAOSENGINE_LLAMA_CLI")
    if override:
        return override
    return _which_with_fallbacks("llama-cli")


def _http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _looks_like_gguf(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return lowered.endswith(".gguf") or "gguf" in lowered


def _resolve_gguf_path(path: str | None, runtime_target: str | None) -> str | None:
    """Find a concrete .gguf file from a path or HF-cache directory.

    When a user loads an HF-cache GGUF repo, the path points to the repo
    directory (e.g. ``models--lmstudio-community--Qwen3.5-9B-GGUF``), not a
    specific file.  We scan for the best .gguf file inside it, excluding
    vision projectors (mmproj) and picking the largest non-projector file.
    """
    for candidate in (path, runtime_target):
        if not candidate:
            continue
        p = Path(candidate)
        if p.is_file() and p.suffix.lower() == ".gguf":
            return str(p)
        if p.is_dir():
            gguf_files = sorted(p.rglob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
            # Filter out vision projector files
            model_files = [f for f in gguf_files if "mmproj" not in f.name.lower()]
            if model_files:
                return str(model_files[0])
            if gguf_files:
                return str(gguf_files[0])
    return None


def _is_local_target(candidate: str | None) -> bool:
    if not candidate:
        return False
    expanded = os.path.expanduser(candidate)
    path = Path(expanded)
    return (
        path.exists()
        or expanded.startswith(("/", "~/", "./", "../"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", expanded))
    )


def _gguf_startup_fallback_note(strategy_name: str) -> str:
    return (
        f"GGUF startup failed with {strategy_name} cache, so ChaosEngineAI retried "
        f"with the standard f16 KV cache."
    )


_MLX_LM_SUPPORTED_CACHE: tuple[str | None, frozenset[str] | None] = (None, None)


def _mlx_lm_supported_model_types(python_executable: str) -> frozenset[str] | None:
    """List of model_type strings supported by the installed mlx-lm version.

    We enumerate `mlx_lm.models.<module>` files and return the set of
    module names. mlx_lm's `_get_classes` does a direct
    `importlib.import_module(f"mlx_lm.models.{model_type}")` so matching
    module name is the correct compatibility check.

    Cached per python_executable. Returns None on any failure (meaning
    "we couldn't check, don't block the conversion").
    """
    global _MLX_LM_SUPPORTED_CACHE
    cached_key, cached_val = _MLX_LM_SUPPORTED_CACHE
    if cached_key == python_executable and cached_val is not None:
        return cached_val

    probe = (
        "import os, pkgutil, json, importlib.util;"
        "spec = importlib.util.find_spec('mlx_lm.models');"
        "paths = spec.submodule_search_locations if spec else [];"
        "names = sorted({m.name for p in paths for m in pkgutil.iter_modules([p]) if not m.name.startswith('_') and not m.ispkg});"
        "print(json.dumps(names))"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", probe],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            return None
        names = json.loads(result.stdout.strip())
        if not isinstance(names, list):
            return None
        supported = frozenset(n for n in names if isinstance(n, str))
        _MLX_LM_SUPPORTED_CACHE = (python_executable, supported)
        return supported
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None


def _peek_hf_model_type(
    hf_path_arg: str | None,
    *,
    convert_env: dict[str, str] | None = None,
) -> str | None:
    """Read `config.json.model_type` without downloading any weights.

    - Local directory: read config.json directly from disk.
    - HF repo id: use huggingface_hub.hf_hub_download for JUST config.json
      (few KB). Honours HF_TOKEN / HUGGING_FACE_HUB_TOKEN for gated repos.
    - HF cache directory (models--owner--name/snapshots/<rev>/config.json):
      walk the snapshot dir.

    Returns None on any failure — callers must treat None as "could not
    pre-flight, proceed optimistically".
    """
    if not hf_path_arg:
        return None

    def _read(p: Path) -> str | None:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        mt = data.get("model_type")
        if isinstance(mt, str) and mt.strip():
            return mt.strip()
        return None

    candidate = Path(hf_path_arg)
    if candidate.exists():
        if candidate.is_file() and candidate.name == "config.json":
            return _read(candidate)
        if candidate.is_dir():
            direct = candidate / "config.json"
            if direct.is_file():
                return _read(direct)
            # HF cache layout: models--owner--name/snapshots/<rev>/config.json
            snapshots = candidate / "snapshots"
            if snapshots.is_dir():
                for rev in sorted(snapshots.iterdir(), reverse=True):
                    cfg = rev / "config.json"
                    if cfg.is_file():
                        return _read(cfg)
        return None

    # Remote HF repo id — pull just config.json.
    if "/" not in hf_path_arg:
        return None
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        return None
    env = dict(convert_env or os.environ)
    token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    try:
        cfg_path = hf_hub_download(
            repo_id=hf_path_arg,
            filename="config.json",
            token=token,
        )
        return _read(Path(cfg_path))
    except Exception:
        return None


def _nearest_supported_arch(requested: str, supported: frozenset[str]) -> str | None:
    """Suggest the closest supported architecture for a given model_type.

    Extracts the base family (letters only) and picks the highest-numbered
    supported variant of that family, preferring plain names (no suffixes
    like _text / _moe / _vl) so "gemma4" → "gemma3" not "gemma3n".
    """
    if not requested or not supported:
        return None
    import re as _re
    base_match = _re.match(r"^([a-z]+)", requested.lower())
    if not base_match:
        return None
    base = base_match.group(1)
    candidates = [s for s in supported if _re.match(rf"^{base}\d*$", s.lower())]
    if not candidates:
        # Fall back to any family member (including suffixed variants).
        candidates = [s for s in supported if s.lower().startswith(base)]
    if not candidates:
        return None

    def _score(name: str) -> tuple[int, int, str]:
        m = _re.search(r"(\d+)", name)
        num = int(m.group(1)) if m else 0
        # Prefer plain names (no underscore suffix) over variant names.
        plain = 1 if "_" not in name else 0
        return (plain, num, name)

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _default_conversion_output(source_label: str) -> str:
    home_models = Path.home() / "Models"
    home_models.mkdir(parents=True, exist_ok=True)
    base = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in source_label)
    base = base.strip("-_") or "model"
    candidate = home_models / f"{base}-mlx"
    suffix = 2
    while candidate.exists():
        candidate = home_models / f"{base}-mlx-{suffix}"
        suffix += 1
    return str(candidate)


def _bytes_to_gb(value: int | float) -> float:
    """Convert a byte count to gigabytes, rounded to 2 decimal places."""
    try:
        return round(float(value) / (1024 ** 3), 2)
    except (TypeError, ValueError):
        return 0.0


def _path_size_bytes(path: str | Path | None) -> int:
    if path is None:
        return 0

    target = Path(path).expanduser()
    if not target.exists():
        return 0
    if target.is_file():
        try:
            return int(target.stat().st_size)
        except OSError:
            return 0

    total = 0
    try:
        for child in target.rglob("*"):
            if child.is_file():
                total += int(child.stat().st_size)
    except OSError:
        return total
    return total


@dataclass
class BackendCapabilities:
    pythonExecutable: str
    mlxAvailable: bool
    mlxLmAvailable: bool
    mlxUsable: bool
    mlxVersion: str | None = None
    mlxLmVersion: str | None = None
    mlxMessage: str | None = None
    ggufAvailable: bool = False
    llamaCliPath: str | None = None
    llamaServerPath: str | None = None
    converterAvailable: bool = False
    vllmAvailable: bool = False
    vllmVersion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pythonExecutable": self.pythonExecutable,
            "mlxAvailable": self.mlxAvailable,
            "mlxLmAvailable": self.mlxLmAvailable,
            "mlxUsable": self.mlxUsable,
            "mlxVersion": self.mlxVersion,
            "mlxLmVersion": self.mlxLmVersion,
            "mlxMessage": self.mlxMessage,
            "ggufAvailable": self.ggufAvailable,
            "llamaCliPath": self.llamaCliPath,
            "llamaServerPath": self.llamaServerPath,
            "converterAvailable": self.converterAvailable,
            "vllmAvailable": self.vllmAvailable,
            "vllmVersion": self.vllmVersion,
        }


_capability_cache: tuple[float, BackendCapabilities] | None = None
_capability_lock = RLock()


def _probe_native_backends() -> BackendCapabilities:
    python_executable = _resolve_mlx_python()
    llama_server_path = _resolve_llama_server()
    llama_cli_path = _resolve_llama_cli()

    code, payload, message = _json_subprocess(
        [python_executable, "-m", "backend_service.mlx_worker", "probe"],
        timeout=12.0,
    )

    if payload is None:
        payload = {}

    mlx_available = bool(payload.get("mlxAvailable", False))
    mlx_lm_available = bool(payload.get("mlxLmAvailable", False))
    mlx_usable = bool(payload.get("mlxUsable", False) and code == 0)
    probe_message = payload.get("message")
    if probe_message is None and code != 0:
        probe_message = message or f"probe exited with code {code}"

    from backend_service.vllm_engine import _vllm_importable, _vllm_version

    return BackendCapabilities(
        pythonExecutable=python_executable,
        mlxAvailable=mlx_available,
        mlxLmAvailable=mlx_lm_available,
        mlxUsable=mlx_usable,
        mlxVersion=payload.get("mlxVersion"),
        mlxLmVersion=payload.get("mlxLmVersion"),
        mlxMessage=probe_message,
        ggufAvailable=bool(llama_server_path),
        llamaCliPath=llama_cli_path,
        llamaServerPath=llama_server_path,
        converterAvailable=mlx_usable,
        vllmAvailable=_vllm_importable(),
        vllmVersion=_vllm_version(),
    )


def get_backend_capabilities(*, force: bool = False) -> BackendCapabilities:
    global _capability_cache
    with _capability_lock:
        now = time.time()
        if not force and _capability_cache is not None:
            cached_at, cached = _capability_cache
            if (now - cached_at) < CAPABILITY_CACHE_TTL_SECONDS:
                return cached

        capabilities = _probe_native_backends()
        _capability_cache = (now, capabilities)
        return capabilities


@dataclass
class LoadedModelInfo:
    ref: str
    name: str
    backend: str
    source: str
    engine: str
    cacheStrategy: str
    cacheBits: int
    fp16Layers: int
    fusedAttention: bool
    fitModelInMemory: bool
    contextTokens: int
    loadedAt: str
    canonicalRepo: str | None = None
    path: str | None = None
    runtimeTarget: str | None = None
    runtimeNote: str | None = None
    speculativeDecoding: bool = False
    dflashDraftModel: str | None = None
    treeBudget: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "name": self.name,
            "canonicalRepo": self.canonicalRepo,
            "backend": self.backend,
            "source": self.source,
            "engine": self.engine,
            "cacheStrategy": self.cacheStrategy,
            "cacheBits": self.cacheBits,
            "fp16Layers": self.fp16Layers,
            "fusedAttention": self.fusedAttention,
            "fitModelInMemory": self.fitModelInMemory,
            "contextTokens": self.contextTokens,
            "loadedAt": self.loadedAt,
            "path": self.path,
            "runtimeTarget": self.runtimeTarget,
            "runtimeNote": self.runtimeNote,
            "speculativeDecoding": self.speculativeDecoding,
            "dflashDraftModel": self.dflashDraftModel,
            "treeBudget": self.treeBudget,
        }


@dataclass
class GenerationResult:
    text: str
    finishReason: str
    promptTokens: int
    completionTokens: int
    totalTokens: int
    tokS: float
    responseSeconds: float
    runtimeNote: str | None = None
    dflashAcceptanceRate: float | None = None
    cache_strategy: str | None = None
    cache_bits: int | None = None
    fp16_layers: int | None = None
    speculative_decoding: bool | None = None
    tree_budget: int | None = None

    def to_metrics(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "finishReason": self.finishReason,
            "promptTokens": self.promptTokens,
            "completionTokens": self.completionTokens,
            "totalTokens": self.totalTokens,
            "tokS": self.tokS,
            "responseSeconds": self.responseSeconds,
            "runtimeNote": self.runtimeNote,
        }
        if self.dflashAcceptanceRate is not None:
            d["dflashAcceptanceRate"] = self.dflashAcceptanceRate
        return d


@dataclass
class StreamChunk:
    text: str | None = None
    reasoning: str | None = None
    reasoning_done: bool = False
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tok_s: float = 0.0
    runtime_note: str | None = None
    dflash_acceptance_rate: float | None = None
    cache_strategy: str | None = None
    cache_bits: int | None = None
    fp16_layers: int | None = None
    speculative_decoding: bool | None = None
    tree_budget: int | None = None
    done: bool = False


class BaseInferenceEngine:
    engine_name = "base"
    engine_label = "Base runtime"

    def load_model(
        self,
        *,
        model_ref: str,
        model_name: str,
        canonical_repo: str | None,
        source: str,
        backend: str,
        path: str | None,
        runtime_target: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        fit_model_in_memory: bool,
        context_tokens: int,
        speculative_decoding: bool = False,
        tree_budget: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LoadedModelInfo:
        raise NotImplementedError

    def update_profile(
        self,
        *,
        canonical_repo: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
    ) -> LoadedModelInfo:
        raise RuntimeError(f"{self.engine_name} does not support in-place profile updates.")

    def unload_model(self) -> None:
        raise NotImplementedError

    def process_pid(self) -> int | None:
        return None

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
    ) -> GenerationResult:
        raise NotImplementedError

    def eval_perplexity(
        self,
        *,
        dataset: str = "wikitext-2",
        num_samples: int = 64,
        seq_length: int = 512,
        batch_size: int = 4,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"Perplexity evaluation is not supported by the {self.engine_name} backend.")

    def eval_task_accuracy(
        self,
        *,
        task_name: str = "mmlu",
        limit: int = 100,
        num_shots: int = 5,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"Task accuracy evaluation is not supported by the {self.engine_name} backend.")

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
    ) -> Iterator[StreamChunk]:
        result = self.generate(
            prompt=prompt,
            history=history,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        yield StreamChunk(text=result.text)
        yield StreamChunk(
            done=True,
            finish_reason=result.finishReason,
            prompt_tokens=result.promptTokens,
            completion_tokens=result.completionTokens,
            total_tokens=result.totalTokens,
            tok_s=result.tokS,
            runtime_note=result.runtimeNote,
        )


class RemoteOpenAIEngine(BaseInferenceEngine):
    engine_name = "remote"
    engine_label = "Remote OpenAI-compatible API"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.loaded_model: LoadedModelInfo | None = None
        self.api_base: str = ""
        self.api_key: str = ""
        self.remote_model: str = ""

    def load_model(
        self, *, model_ref, model_name, canonical_repo, source, backend, path, runtime_target,
        cache_strategy, cache_bits, fp16_layers, fused_attention,
        fit_model_in_memory, context_tokens, speculative_decoding=False,
        tree_budget=0,
        progress_callback=None,
    ) -> LoadedModelInfo:
        # The model_ref encodes the remote provider config: "remote:<base>|<key>|<model>"
        if not model_ref.startswith("remote:"):
            raise RuntimeError("Remote engine requires a remote:<base>|<key>|<model> ref.")
        try:
            _, payload = model_ref.split("remote:", 1)
            parts = payload.split("|", 2)
            if len(parts) != 3:
                raise ValueError("malformed remote ref")
            self.api_base, self.api_key, self.remote_model = parts
        except ValueError as exc:
            raise RuntimeError(f"Invalid remote model ref: {exc}") from exc

        if not self.api_base.startswith("https://") and not self.api_base.startswith("http://127.0.0.1"):
            raise RuntimeError("Remote API must use HTTPS (or localhost http://127.0.0.1).")

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name or self.remote_model,
            canonicalRepo=canonical_repo,
            source=source,
            backend="remote",
            engine=self.engine_name,
            cacheStrategy="native",
            cacheBits=0,
            fp16Layers=0,
            fusedAttention=False,
            fitModelInMemory=False,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            path=None,
            runtimeTarget=self.remote_model,
            runtimeNote=f"Remote API at {self.api_base}",
        )
        return self.loaded_model

    def unload_model(self) -> None:
        self.loaded_model = None

    def _request(self, *, prompt, history, system_prompt, max_tokens, temperature, stream=False):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for m in history:
            role = m.get("role")
            if role in {"user", "assistant", "system"}:
                messages.append({"role": role, "content": m.get("text", "")})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.remote_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        url = self.api_base.rstrip("/") + "/chat/completions"
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        return urllib.request.urlopen(req, timeout=120.0)

    def generate(self, *, prompt, history, system_prompt, max_tokens, temperature,
                 images=None, tools=None) -> GenerationResult:
        if self.loaded_model is None:
            raise RuntimeError("Remote model not configured.")
        started = time.perf_counter()
        try:
            resp = self._request(
                prompt=prompt, history=history, system_prompt=system_prompt,
                max_tokens=max_tokens, temperature=temperature, stream=False,
            )
            data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Remote API error: {detail or exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Remote connection failed: {exc.reason}") from exc

        elapsed = max(time.perf_counter() - started, 1e-6)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        completion = int(usage.get("completion_tokens") or 0)
        prompt_t = int(usage.get("prompt_tokens") or 0)
        return GenerationResult(
            text=str(msg.get("content") or ""),
            finishReason=str(choice.get("finish_reason") or "stop"),
            promptTokens=prompt_t,
            completionTokens=completion,
            totalTokens=prompt_t + completion,
            tokS=round(completion / elapsed, 1) if completion else 0.0,
            responseSeconds=round(elapsed, 2),
            runtimeNote=f"Generated by remote API ({self.api_base})",
        )


class MockInferenceEngine(BaseInferenceEngine):
    """Placeholder engine used only as the initial default before any model is loaded.

    All methods raise ``RuntimeError`` — this engine never produces fake results.
    If ``_select_engine()`` is implemented correctly, the mock engine is never
    chosen for real model loads; these raises are a safety net.
    """

    engine_name = "mock"
    engine_label = "No backend"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.loaded_model: LoadedModelInfo | None = None

    def _missing_backend_message(self) -> str:
        hints: list[str] = []
        if not self.capabilities.mlxUsable:
            hints.append("MLX is not available" + (f" ({self.capabilities.mlxMessage})" if self.capabilities.mlxMessage else ""))
        if not self.capabilities.ggufAvailable:
            hints.append("llama-server not found (install with: brew install llama.cpp)")
        if not self.capabilities.vllmAvailable:
            hints.append("vLLM not installed")
        return (
            "No inference backend is available. " + " | ".join(hints) + ". "
            "Install llama.cpp for GGUF models or ensure MLX is working for safetensors models."
        )

    def load_model(self, **kwargs: Any) -> LoadedModelInfo:
        raise RuntimeError(self._missing_backend_message())

    def unload_model(self) -> None:
        self.loaded_model = None

    def generate(self, **kwargs: Any) -> GenerationResult:
        raise RuntimeError(self._missing_backend_message())

    def stream_generate(self, **kwargs: Any) -> Iterator[StreamChunk]:
        raise RuntimeError(self._missing_backend_message())


class JsonRpcProcess:
    def __init__(self, command: list[str], *, timeout: float = DEFAULT_MLX_TIMEOUT_SECONDS) -> None:
        self.command = command
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self._stdout_queue: Queue[str | None] = Queue()
        self._reader_thread: Thread | None = None
        self._lock = Lock()

    def _pump_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self._stdout_queue.put(line.rstrip("\n"))
        self._stdout_queue.put(None)

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        self.process = subprocess.Popen(
            self.command,
            cwd=str(WORKSPACE_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_queue = Queue()
        self._reader_thread = Thread(target=self._pump_stdout, daemon=True)
        self._reader_thread.start()

    def close(self, *, force: bool = False) -> None:
        if self.process is None:
            return

        try:
            if self.process.stdin is not None and self.process.poll() is None:
                self.process.stdin.close()
        except OSError:
            pass

        if self.process.poll() is None:
            if force:
                self.process.kill()
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

    def is_alive(self) -> bool:
        """Return True if the worker process is running."""
        return self.process is not None and self.process.poll() is None

    @staticmethod
    def _exit_detail(stderr: str, return_code: int | None) -> str:
        if stderr:
            return stderr
        if return_code == -9:
            return (
                "worker was SIGKILLed by the OS. This usually means memory pressure "
                "during MLX model load or startup."
            )
        return f"worker exited with code {return_code}"

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return self.request_with_progress(payload, on_progress=None, timeout=timeout)

    def request_with_progress(
        self,
        payload: dict[str, Any],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        effective_timeout = timeout if timeout is not None else self.timeout
        with self._lock:
            self.start()
            assert self.process is not None and self.process.stdin is not None

            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
            except OSError as exc:
                self.close()
                raise RuntimeError(f"Native worker stdin failed: {exc}") from exc

            while True:
                try:
                    line = self._stdout_queue.get(timeout=effective_timeout)
                except Empty as exc:
                    self.close()
                    raise RuntimeError(
                        f"Timed out waiting for the MLX worker after {effective_timeout:.0f}s."
                    ) from exc

                if line is None:
                    stderr = ""
                    if self.process is not None and self.process.stderr is not None:
                        try:
                            stderr = self.process.stderr.read().strip()
                        except OSError:
                            stderr = ""
                    return_code = self.process.poll() if self.process else None
                    self.close()
                    detail = self._exit_detail(stderr, return_code)
                    raise RuntimeError(f"MLX worker exited unexpectedly: {detail}")

                try:
                    response = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.close()
                    raise RuntimeError(f"MLX worker returned invalid JSON: {line}") from exc

                if not response.get("ok", False):
                    raise RuntimeError(str(response.get("error") or "MLX worker returned an unknown error."))

                # Intermediate progress message — keep reading.
                if "result" not in response and "progress" in response:
                    if on_progress is not None:
                        try:
                            on_progress(response.get("progress") or {})
                        except Exception:
                            pass
                    continue

                result = response.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("MLX worker returned an invalid result payload.")
                return result

    def stream_request(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self._lock:
            self.start()
            assert self.process is not None and self.process.stdin is not None

            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
            except OSError as exc:
                self.close()
                raise RuntimeError(f"Native worker stdin failed: {exc}") from exc

        while True:
            try:
                line = self._stdout_queue.get(timeout=self.timeout)
            except Empty as exc:
                self.close()
                raise RuntimeError("Timed out waiting for the MLX worker.") from exc

            if line is None:
                stderr = ""
                if self.process is not None and self.process.stderr is not None:
                    try:
                        stderr = self.process.stderr.read().strip()
                    except OSError:
                        stderr = ""
                return_code = self.process.poll() if self.process else None
                self.close()
                detail = self._exit_detail(stderr, return_code)
                raise RuntimeError(f"MLX worker exited unexpectedly: {detail}")

            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self.close()
                raise RuntimeError(f"MLX worker returned invalid JSON: {line}") from exc

            if not response.get("ok", False):
                raise RuntimeError(str(response.get("error") or "MLX worker returned an unknown error."))

            yield response
            if response.get("done"):
                break


class MLXWorkerEngine(BaseInferenceEngine):
    engine_name = "mlx"
    engine_label = "MLX"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.worker = JsonRpcProcess(
            [self.capabilities.pythonExecutable, "-m", "backend_service.mlx_worker", "serve"],
            timeout=DEFAULT_MLX_TIMEOUT_SECONDS,
        )
        self.loaded_model: LoadedModelInfo | None = None

    def _base_runtime_note(self) -> str:
        return (
            f"Using {Path(self.capabilities.pythonExecutable).name} with MLX {self.capabilities.mlxVersion or 'unknown'} "
            f"and mlx-lm {self.capabilities.mlxLmVersion or 'unknown'}."
        )

    def _compose_runtime_note(
        self,
        *,
        worker_note: str | None,
        dflash_target_ref: str | None,
        requested_speculative: bool,
        actual_speculative: bool,
        actual_draft_model: str | None,
        actual_tree_budget: int,
    ) -> str:
        runtime_note = self._base_runtime_note()
        if worker_note:
            runtime_note = f"{runtime_note} {worker_note}"
        elif actual_speculative and actual_draft_model:
            if actual_tree_budget > 0:
                runtime_note = (
                    f"{runtime_note} DDTree speculative decoding active "
                    f"(budget={actual_tree_budget}, draft: {actual_draft_model})."
                )
            else:
                runtime_note = f"{runtime_note} DFLASH speculative decoding active (draft: {actual_draft_model})."
        elif requested_speculative:
            resolved_ref = dflash_target_ref or (self.loaded_model.ref if self.loaded_model else None) or "unknown target"
            runtime_note = (
                f"{runtime_note} DFLASH unavailable for '{resolved_ref}': no compatible draft model is registered."
            )
        return runtime_note

    def load_model(
        self,
        *,
        model_ref: str,
        model_name: str,
        canonical_repo: str | None,
        source: str,
        backend: str,
        path: str | None,
        runtime_target: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        fit_model_in_memory: bool,
        context_tokens: int,
        speculative_decoding: bool = False,
        tree_budget: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LoadedModelInfo:
        if not self.capabilities.mlxUsable:
            raise RuntimeError(self.capabilities.mlxMessage or "MLX is not available.")

        # Resolve DFLASH draft model when speculative decoding is requested
        draft_model: str | None = None
        dflash_target_ref = resolve_dflash_target_ref(
            canonical_repo=canonical_repo,
            path=path,
            model_ref=model_ref,
        )
        if speculative_decoding:
            try:
                from dflash import get_draft_model, is_mlx_available
                if is_mlx_available():
                    draft_model = get_draft_model(dflash_target_ref or model_ref)
            except ImportError:
                pass

        target = runtime_target or path or model_ref
        result = self.worker.request_with_progress(
            {
                "op": "load_model",
                "target": target,
                "cacheStrategy": cache_strategy,
                "cacheBits": cache_bits,
                "fp16Layers": fp16_layers,
                "fusedAttention": fused_attention,
                "contextTokens": context_tokens,
                "speculativeDecoding": speculative_decoding and draft_model is not None,
                "dflashDraftModel": draft_model,
                "treeBudget": tree_budget if speculative_decoding and draft_model else 0,
            },
            on_progress=progress_callback,
            timeout=MLX_LOAD_TIMEOUT_SECONDS,
        )
        actual_cache_strategy = str(result.get("cacheStrategy") or cache_strategy)
        actual_cache_bits = int(result.get("cacheBits") if result.get("cacheBits") is not None else cache_bits)
        actual_fp16_layers = int(result.get("fp16Layers") if result.get("fp16Layers") is not None else fp16_layers)
        actual_fused_attention = bool(
            result.get("fusedAttention") if result.get("fusedAttention") is not None else fused_attention
        )
        actual_speculative = bool(result.get("speculativeDecoding"))
        actual_tree_budget = int(result.get("treeBudget") or 0)
        actual_draft_model = (
            str(result.get("dflashDraftModel"))
            if result.get("dflashDraftModel")
            else (draft_model if actual_speculative else None)
        )
        runtime_note = self._compose_runtime_note(
            worker_note=str(result.get("note") or "").strip() or None,
            dflash_target_ref=dflash_target_ref,
            requested_speculative=speculative_decoding,
            actual_speculative=actual_speculative,
            actual_draft_model=actual_draft_model,
            actual_tree_budget=actual_tree_budget,
        )

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name,
            backend=backend,
            source=source,
            engine=self.engine_name,
            cacheStrategy=actual_cache_strategy,
            cacheBits=actual_cache_bits,
            fp16Layers=actual_fp16_layers,
            fusedAttention=actual_fused_attention,
            fitModelInMemory=fit_model_in_memory,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            canonicalRepo=canonical_repo,
            path=path,
            runtimeTarget=target,
            runtimeNote=runtime_note,
            speculativeDecoding=actual_speculative,
            dflashDraftModel=actual_draft_model,
            treeBudget=actual_tree_budget,
        )
        return self.loaded_model

    def update_profile(
        self,
        *,
        canonical_repo: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
    ) -> LoadedModelInfo:
        if self.loaded_model is None:
            raise RuntimeError("No MLX model is loaded.")
        if not self.worker.is_alive():
            self.loaded_model = None
            raise RuntimeError(
                "The MLX worker process exited and the model is no longer loaded. "
                "Please reload the model from My Models."
            )

        result = self.worker.request_with_progress(
            {
                "op": "update_profile",
                "cacheStrategy": cache_strategy,
                "cacheBits": cache_bits,
                "fp16Layers": fp16_layers,
                "fusedAttention": fused_attention,
            },
            on_progress=None,
            timeout=DEFAULT_MLX_TIMEOUT_SECONDS,
        )

        self.loaded_model.cacheStrategy = str(result.get("cacheStrategy") or cache_strategy)
        self.loaded_model.cacheBits = int(result.get("cacheBits") if result.get("cacheBits") is not None else cache_bits)
        self.loaded_model.fp16Layers = int(result.get("fp16Layers") if result.get("fp16Layers") is not None else fp16_layers)
        self.loaded_model.fusedAttention = bool(
            result.get("fusedAttention") if result.get("fusedAttention") is not None else fused_attention
        )
        if canonical_repo is not None:
            self.loaded_model.canonicalRepo = canonical_repo
        dflash_target_ref = resolve_dflash_target_ref(
            canonical_repo=self.loaded_model.canonicalRepo,
            path=self.loaded_model.path,
            model_ref=self.loaded_model.ref,
        )
        self.loaded_model.runtimeNote = self._compose_runtime_note(
            worker_note=str(result.get("note") or "").strip() or None,
            dflash_target_ref=dflash_target_ref,
            requested_speculative=self.loaded_model.speculativeDecoding,
            actual_speculative=self.loaded_model.speculativeDecoding,
            actual_draft_model=self.loaded_model.dflashDraftModel,
            actual_tree_budget=self.loaded_model.treeBudget,
        )
        return self.loaded_model

    def unload_model(self) -> None:
        self.loaded_model = None
        # Skip the in-worker cleanup RPC (gc.collect / mx.metal.clear_cache)
        # and kill the process immediately.  The OS reclaims all process memory
        # on exit, so the manual cleanup is redundant and was the main source
        # of multi-second blocking during unload.
        self.worker.close(force=True)

    def process_pid(self) -> int | None:
        process = self.worker.process
        if process is None or process.poll() is not None:
            return None
        return int(process.pid)

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
    ) -> GenerationResult:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        # Detect worker process restart: if the process died, the model is gone.
        if not self.worker.is_alive():
            self.loaded_model = None
            raise RuntimeError(
                "The MLX worker process was restarted and the model is no longer loaded. "
                "Please reload the model from My Models."
            )

        started_at = time.perf_counter()
        payload: dict[str, Any] = {
            "op": "generate",
            "prompt": prompt,
            "history": history,
            "systemPrompt": system_prompt,
            "maxTokens": max_tokens,
            "temperature": temperature,
        }
        if images:
            payload["images"] = images
        if tools:
            payload["tools"] = tools
        result = self.worker.request(payload)
        elapsed = max(time.perf_counter() - started_at, 1e-6)
        return GenerationResult(
            text=str(result.get("text") or ""),
            finishReason=str(result.get("finishReason") or "stop"),
            promptTokens=int(result.get("promptTokens") or 0),
            completionTokens=int(result.get("completionTokens") or 0),
            totalTokens=int(result.get("totalTokens") or 0),
            tokS=float(result.get("tokS") or 0.0),
            responseSeconds=round(float(result.get("responseSeconds") or elapsed), 2),
            runtimeNote=str(result.get("runtimeNote") or self.loaded_model.runtimeNote),
            dflashAcceptanceRate=result.get("dflashAcceptanceRate"),
            cache_strategy=str(result.get("cacheStrategy")) if result.get("cacheStrategy") is not None else None,
            cache_bits=int(result.get("cacheBits")) if result.get("cacheBits") is not None else None,
            fp16_layers=int(result.get("fp16Layers")) if result.get("fp16Layers") is not None else None,
            speculative_decoding=(
                bool(result.get("speculativeDecoding"))
                if result.get("speculativeDecoding") is not None
                else None
            ),
            tree_budget=int(result.get("treeBudget")) if result.get("treeBudget") is not None else None,
        )

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
    ) -> Iterator[StreamChunk]:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        if not self.worker.is_alive():
            self.loaded_model = None
            raise RuntimeError(
                "The MLX worker process exited and the model is no longer loaded. "
                "Please reload the model from My Models."
            )

        payload: dict[str, Any] = {
            "op": "stream_generate",
            "prompt": prompt,
            "history": history,
            "systemPrompt": system_prompt,
            "maxTokens": max_tokens,
            "temperature": temperature,
        }
        if images:
            payload["images"] = images
        if tools:
            payload["tools"] = tools
        try:
            request_iter = self.worker.stream_request(payload)
        except RuntimeError as exc:
            if "No MLX model is loaded" in str(exc):
                self.loaded_model = None
                raise RuntimeError(
                    "The MLX worker lost the loaded model. "
                    "Please reload the model from My Models."
                ) from exc
            raise
        try:
            for response in request_iter:
                chunk = response.get("chunk")
                if chunk:
                    if chunk.get("reasoning"):
                        yield StreamChunk(reasoning=chunk["reasoning"])
                    if chunk.get("reasoningDone"):
                        yield StreamChunk(reasoning_done=True)
                    if chunk.get("text"):
                        yield StreamChunk(text=chunk["text"])
                if response.get("done"):
                    result = response.get("result") or {}
                    yield StreamChunk(
                        done=True,
                        finish_reason=str(result.get("finishReason") or "stop"),
                        prompt_tokens=int(result.get("promptTokens") or 0),
                        completion_tokens=int(result.get("completionTokens") or 0),
                        total_tokens=int(result.get("totalTokens") or 0),
                        tok_s=float(result.get("tokS") or 0.0),
                        runtime_note=str(result.get("runtimeNote") or self.loaded_model.runtimeNote),
                        dflash_acceptance_rate=result.get("dflashAcceptanceRate"),
                        cache_strategy=str(result.get("cacheStrategy")) if result.get("cacheStrategy") is not None else None,
                        cache_bits=int(result.get("cacheBits")) if result.get("cacheBits") is not None else None,
                        fp16_layers=int(result.get("fp16Layers")) if result.get("fp16Layers") is not None else None,
                        speculative_decoding=(
                            bool(result.get("speculativeDecoding"))
                            if result.get("speculativeDecoding") is not None
                            else None
                        ),
                        tree_budget=int(result.get("treeBudget")) if result.get("treeBudget") is not None else None,
                    )
        except RuntimeError as exc:
            if "No MLX model is loaded" in str(exc):
                self.loaded_model = None
                raise RuntimeError(
                    "The MLX worker lost the loaded model. "
                    "Please reload the model from My Models."
                ) from exc
            raise

    def eval_perplexity(
        self,
        *,
        dataset: str = "wikitext-2",
        num_samples: int = 64,
        seq_length: int = 512,
        batch_size: int = 4,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        return self.worker.request_with_progress(
            {
                "op": "eval_perplexity",
                "dataset": dataset,
                "numSamples": num_samples,
                "seqLength": seq_length,
                "batchSize": batch_size,
            },
            on_progress=progress_callback,
            timeout=600,
        )

    def eval_task_accuracy(
        self,
        *,
        task_name: str = "mmlu",
        limit: int = 100,
        num_shots: int = 5,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        return self.worker.request_with_progress(
            {
                "op": "eval_task_accuracy",
                "taskName": task_name,
                "limit": limit,
                "numShots": num_shots,
            },
            on_progress=progress_callback,
            timeout=900,
        )


class LlamaCppEngine(BaseInferenceEngine):
    engine_name = "llama.cpp"
    engine_label = "llama.cpp + GGUF"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.loaded_model: LoadedModelInfo | None = None
        self.process: subprocess.Popen[str] | None = None
        self.port: int | None = None
        self.log_path: Path | None = None
        self.log_handle: Any = None

    def _server_url(self, path: str) -> str:
        if self.port is None:
            raise RuntimeError("llama.cpp server is not running.")
        return f"http://127.0.0.1:{self.port}{path}"

    def _cleanup_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            # llama-server now shares the Python backend's process group,
            # so we target just the single process. killpg would take the
            # entire backend down with it.
            try:
                self.process.terminate()
            except (ProcessLookupError, OSError):
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        self.process = None
        self.port = None
        if self.log_handle is not None:
            try:
                self.log_handle.close()
            except OSError:
                pass
        self.log_handle = None

    def process_pid(self) -> int | None:
        if self.process is None or self.process.poll() is not None:
            return None
        return int(self.process.pid)

    def _build_command(
        self,
        *,
        path: str | None,
        runtime_target: str | None,
        cache_strategy: str,
        cache_bits: int,
        context_tokens: int,
        fit_enabled: bool,
        is_fallback: bool,
    ) -> tuple[list[str], str | None]:
        if not self.capabilities.llamaServerPath:
            raise RuntimeError("llama-server was not found on this machine.")

        from compression import registry as _strategy_registry
        strategy = _strategy_registry.get(cache_strategy) or _strategy_registry.default()

        runtime_note = None
        self.port = _find_open_port()
        command = [
            self.capabilities.llamaServerPath,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--parallel",
            "1",
            "--ctx-size",
            str(max(256, context_tokens)),
            "--jinja",
        ]
        if _llama_server_supports(self.capabilities.llamaServerPath, "--reasoning-format"):
            command.extend(["--reasoning-format", "deepseek"])
        if _llama_server_supports(self.capabilities.llamaServerPath, "--reasoning"):
            command.extend(["--reasoning", "off"])
        if fit_enabled:
            command.extend(["--fit", "on"])
        else:
            command.extend(["--fit", "off"])

        try:
            cache_flags = strategy.llama_cpp_cache_flags(cache_bits)
        except NotImplementedError:
            cache_flags = ["--cache-type-k", "f16", "--cache-type-v", "f16"]
            runtime_note = f"Cache strategy '{strategy.name}' does not support llama.cpp yet; using native f16 cache."

        if is_fallback:
            cache_flags = ["--cache-type-k", "f16", "--cache-type-v", "f16"]
            runtime_note = (
                f"GGUF startup failed with {strategy.name} cache, so ChaosEngineAI retried with the standard f16 KV cache."
            )

        command.extend(cache_flags)

        target = runtime_target or path
        resolved_gguf = _resolve_gguf_path(path, target)
        if resolved_gguf:
            command.extend(["--model", resolved_gguf])
        elif path:
            command.extend(["--model", path])
        elif target:
            command.extend(["--hf-repo", target])
        else:
            raise RuntimeError("GGUF loading requires a local model path or a Hugging Face GGUF repository.")

        return command, runtime_note

    def _wait_for_server(self) -> None:
        deadline = time.time() + DEFAULT_LLAMA_TIMEOUT_SECONDS
        last_error = "llama.cpp server did not become ready."

        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                logs = _read_text_tail(self.log_path)
                raise RuntimeError(_friendly_llama_error(logs))

            try:
                _http_json(self._server_url("/health"), timeout=2.0)
                models = _http_json(self._server_url("/v1/models"), timeout=2.0)
                if isinstance(models, dict):
                    return
            except Exception as exc:
                last_error = str(exc)
            time.sleep(1.0)

        logs = _read_text_tail(self.log_path)
        raise RuntimeError(_friendly_llama_error(logs) if logs else last_error)

    def load_model(
        self,
        *,
        model_ref: str,
        model_name: str,
        canonical_repo: str | None,
        source: str,
        backend: str,
        path: str | None,
        runtime_target: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        fit_model_in_memory: bool,
        context_tokens: int,
        speculative_decoding: bool = False,
        tree_budget: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LoadedModelInfo:
        if not self.capabilities.ggufAvailable:
            raise RuntimeError("llama.cpp support is unavailable on this machine.")

        if _is_local_target(path) or _is_local_target(runtime_target):
            resolved_preflight = _resolve_gguf_path(path, runtime_target)
            if resolved_preflight is None:
                raise RuntimeError(
                    f"No .gguf weights found inside {path or runtime_target}. "
                    f"The download may be incomplete or corrupt. Re-download the model, "
                    f"or pick a specific .gguf file from the source directory."
                )

        self.unload_model()
        runtime_note = None
        actual_strategy = cache_strategy
        actual_fit = fit_model_in_memory
        from compression import registry as _strategy_registry
        failed_strategy_name: str | None = None

        # Try the requested strategy first, then fall back to native.
        attempts: list[tuple[str, bool, bool]] = [(cache_strategy, fit_model_in_memory, False)]
        if cache_strategy != "native":
            attempts.append(("native", False, True))
        last_error: str | None = None

        for strategy_id, fit_enabled, is_fallback in attempts:
            strategy = _strategy_registry.get(strategy_id) or _strategy_registry.default()
            command, attempt_note = self._build_command(
                path=path,
                runtime_target=runtime_target,
                cache_strategy=strategy_id,
                cache_bits=cache_bits,
                context_tokens=context_tokens,
                fit_enabled=fit_enabled,
                is_fallback=is_fallback,
            )

            temp_log = tempfile.NamedTemporaryFile(prefix="chaosengine-llama-", suffix=".log", delete=False)
            temp_log.close()
            self.log_path = Path(temp_log.name)
            self.log_handle = self.log_path.open("a", encoding="utf-8")
            self.process = subprocess.Popen(
                command,
                cwd=str(WORKSPACE_ROOT),
                stdout=self.log_handle,
                stderr=self.log_handle,
                text=True,
            )

            try:
                self._wait_for_server()
                runtime_note = attempt_note
                actual_strategy = strategy_id
                actual_fit = fit_enabled
                if is_fallback and failed_strategy_name is not None:
                    runtime_note = _gguf_startup_fallback_note(failed_strategy_name)
                break
            except RuntimeError as exc:
                last_error = str(exc)
                if not is_fallback:
                    failed_strategy_name = strategy.name
                self._cleanup_process()
        else:
            raise RuntimeError(last_error or "llama.cpp server failed to start.")

        strat = _strategy_registry.get(actual_strategy) or _strategy_registry.default()
        actual_cache_bits = cache_bits if actual_strategy != "native" else 0
        actual_fp16_layers = fp16_layers if actual_strategy != "native" else 0
        if runtime_note is None:
            runtime_note = (
                f"GGUF generation is running through the local llama.cpp server with "
                f"{strat.label(actual_cache_bits, actual_fp16_layers)} cache."
            )

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name,
            canonicalRepo=canonical_repo,
            backend=backend,
            source=source,
            engine=self.engine_name,
            cacheStrategy=actual_strategy,
            cacheBits=actual_cache_bits,
            fp16Layers=actual_fp16_layers,
            fusedAttention=fused_attention,
            fitModelInMemory=actual_fit,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            path=path,
            runtimeTarget=runtime_target or path,
            runtimeNote=runtime_note,
        )
        return self.loaded_model

    def unload_model(self) -> None:
        self._cleanup_process()
        self.loaded_model = None

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
    ) -> GenerationResult:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        if self.process is None or self.process.poll() is not None:
            logs = _read_text_tail(self.log_path)
            raise RuntimeError(logs or "The llama.cpp server is not running.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            messages.append({"role": role, "content": _normalize_message_content(message.get("text", ""))})
        # Build user message with optional images
        if images:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img_b64 in images:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": prompt})

        started_at = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.loaded_model.ref,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        try:
            response = _http_json(
                self._server_url("/v1/chat/completions"),
                payload=payload,
                timeout=DEFAULT_LLAMA_TIMEOUT_SECONDS,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(detail or str(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

        elapsed = max(time.perf_counter() - started_at, 1e-6)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = response.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        text = _strip_thinking_tokens(str(message.get("content") or ""))

        return GenerationResult(
            text=text,
            finishReason=str(choice.get("finish_reason") or "stop"),
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=total_tokens,
            tokS=round(completion_tokens / elapsed, 1) if completion_tokens else 0.0,
            responseSeconds=round(elapsed, 2),
            runtimeNote=self.loaded_model.runtimeNote,
        )

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
    ) -> Iterator[StreamChunk]:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        if self.process is None or self.process.poll() is not None:
            logs = _read_text_tail(self.log_path)
            raise RuntimeError(logs or "The llama.cpp server is not running.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            messages.append({"role": role, "content": _normalize_message_content(message.get("text", ""))})
        if images:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img_b64 in images:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.loaded_model.ref,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        url = self._server_url("/v1/chat/completions")
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(request, timeout=DEFAULT_LLAMA_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(detail or str(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        stream_start = time.perf_counter()
        first_token_time: float | None = None
        runtime_note = self.loaded_model.runtimeNote
        think_filter = ThinkingTokenFilter()
        runaway_guard = RepeatedLineGuard()
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_str = line[len("data: "):]
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    split = think_filter.feed(str(content))
                    if split.reasoning:
                        yield StreamChunk(reasoning=split.reasoning)
                    if split.reasoning_done:
                        yield StreamChunk(reasoning_done=True)
                    if split.text:
                        runaway_guard.feed(split.text)
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        completion_tokens += 1
                        yield StreamChunk(text=split.text)
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            flushed = think_filter.flush()
            if flushed.reasoning:
                yield StreamChunk(reasoning=flushed.reasoning)
            if flushed.reasoning_done:
                yield StreamChunk(reasoning_done=True)
            if flushed.text:
                runaway_guard.feed(flushed.text)
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                yield StreamChunk(text=flushed.text)
            runaway_guard.flush()
        except RuntimeError as exc:
            runtime_note = _append_runtime_note(runtime_note, str(exc))
            finish_reason = "stop"
        finally:
            resp.close()

        # Measure generation speed from first token to completion
        end_time = time.perf_counter()
        gen_elapsed = max(end_time - (first_token_time or stream_start), 1e-6)
        tok_s = round(completion_tokens / gen_elapsed, 1) if completion_tokens > 0 else 0.0

        yield StreamChunk(
            done=True,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tok_s=tok_s,
            runtime_note=runtime_note,
        )


class RuntimeController:
    MAX_WARM_MODELS = 2

    def __init__(self) -> None:
        self.capabilities = get_backend_capabilities()
        self.engine: BaseInferenceEngine = MockInferenceEngine(self.capabilities)
        self.loaded_model: LoadedModelInfo | None = None
        self.runtime_note: str | None = None
        # Warm pool: keeps previously loaded engines alive for instant switch-back
        self._warm_pool: dict[str, tuple[BaseInferenceEngine, LoadedModelInfo]] = {}
        self._pool_lock = Lock()
        self._loading_progress: dict[str, Any] | None = None
        self._loading_log_tail: list[str] = []
        self._recent_orphaned_workers: list[dict[str, Any]] = []

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

    def _park_active_engine_or_unload(self, *, requested_identity: str) -> None:
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
        self._evict_warm_pool()
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

    def prune_stale_backend_children(self) -> None:
        tracked = self._tracked_process_pids()
        try:
            import psutil

            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=False)
        except Exception:
            return

        pruned: list[dict[str, Any]] = []
        for child in children:
            try:
                cmdline = " ".join(child.cmdline()).lower()
                name = (child.name() or "").lower()
            except Exception:
                continue

            is_mlx_worker = "backend_service.mlx_worker" in cmdline or "mlx_worker" in cmdline
            is_llama = "llama-server" in name or "llama-server" in cmdline
            if not (is_mlx_worker or is_llama):
                continue
            if child.pid in tracked:
                continue

            record = {
                "pid": int(child.pid),
                "kind": "mlx_worker" if is_mlx_worker else "llama_server",
                "label": "MLX worker" if is_mlx_worker else "llama-server",
                "action": "terminated",
                "detectedAt": _now_label(),
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

    def refresh_capabilities(self, *, force: bool = False) -> BackendCapabilities:
        self.capabilities = get_backend_capabilities(force=force)
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

    def _evict_warm_pool(self) -> None:
        """Remove the oldest entry from the warm pool if at capacity."""
        while len(self._warm_pool) >= self.MAX_WARM_MODELS:
            oldest_key = next(iter(self._warm_pool))
            old_engine, _ = self._warm_pool.pop(oldest_key)
            try:
                old_engine.unload_model()
            except Exception:
                pass

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
            self._park_active_engine_or_unload(requested_identity=requested_identity)
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
        self._park_active_engine_or_unload(requested_identity=requested_identity)

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
            "recentOrphanedWorkers": list(self._recent_orphaned_workers),
        }
