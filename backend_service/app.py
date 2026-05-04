from __future__ import annotations

import os
import ipaddress
import secrets
import signal
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from backend_service.models import ImageGenerationRequest, VideoGenerationRequest
from backend_service.routes import register_routes
from backend_service.state import ChaosEngineState

if TYPE_CHECKING:
    from backend_service.image_runtime import ImageRuntimeManager
    from backend_service.video_runtime import VideoRuntimeManager

# ---------------------------------------------------------------------------
# Helper modules -- extracted from this file for maintainability.
# ---------------------------------------------------------------------------
from backend_service.helpers.system import (
    _build_system_snapshot as _build_system_snapshot_impl,
    _resolve_app_version,
)
from backend_service.helpers.images import (
    _load_image_outputs as _load_image_outputs_impl,
    _save_image_artifact as _save_image_artifact_impl,
    _find_image_output as _find_image_output_impl,
    _delete_image_output as _delete_image_output_impl,
)
from backend_service.helpers.video import (
    _load_video_outputs as _load_video_outputs_impl,
    _save_video_artifact as _save_video_artifact_impl,
    _find_video_output as _find_video_output_impl,
    _delete_video_output as _delete_video_output_impl,
)
from backend_service.helpers.settings import (
    DataLocation,
    _default_settings as _default_settings_impl,
    _load_settings as _load_settings_impl,
    _save_settings as _save_settings_impl,
)
from backend_service.helpers.cache import (
    compute_cache_preview as compute_cache_preview_impl,
)
from backend_service.helpers.persistence import (
    _load_benchmark_runs as _load_benchmark_runs_impl,
    _save_benchmark_runs as _save_benchmark_runs_impl,
    _load_chat_sessions as _load_chat_sessions_impl,
    _save_chat_sessions as _save_chat_sessions_impl,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
APP_STARTED_AT = time.time()
HF_SNAPSHOT_DOWNLOAD_HELPER = (
    "import json, sys\n"
    "from huggingface_hub import snapshot_download\n"
    "repo_id = sys.argv[1]\n"
    "raw_allow = sys.argv[2] if len(sys.argv) > 2 else ''\n"
    "allow_patterns = json.loads(raw_allow) if raw_allow else None\n"
    "kwargs = {'repo_id': repo_id, 'resume_download': True}\n"
    "if allow_patterns:\n"
    "    kwargs['allow_patterns'] = allow_patterns\n"
    "snapshot_download(**kwargs)\n"
)
DEFAULT_PORT = int(os.getenv("CHAOSENGINE_PORT", "8876"))
DEFAULT_HOST = os.getenv("CHAOSENGINE_HOST", "127.0.0.1")

app_version = _resolve_app_version()

DATA_LOCATION = DataLocation()
# Backwards-compat aliases captured at import time.
SETTINGS_DIR = DATA_LOCATION.data_dir
SETTINGS_PATH = DATA_LOCATION.settings_path
BENCHMARKS_PATH = DATA_LOCATION.benchmarks_path
CHAT_SESSIONS_PATH = DATA_LOCATION.chat_sessions_path
LIBRARY_CACHE_PATH = DATA_LOCATION.data_dir / "library_cache.json"
DOCUMENTS_DIR = DATA_LOCATION.documents_dir
WORKSPACES_PATH = DATA_LOCATION.workspaces_path
WORKSPACES_DIR = DATA_LOCATION.workspaces_dir
IMAGE_OUTPUTS_DIR = DATA_LOCATION.image_outputs_dir
VIDEO_OUTPUTS_DIR = DATA_LOCATION.video_outputs_dir
MAX_DOC_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB per file
MAX_SESSION_DOCS_BYTES = 200 * 1024 * 1024  # 200 MB per session
DOC_ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml", ".toml",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".html", ".css", ".sh",
}
DEFAULT_ALLOWED_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5174",
    "http://localhost:5174",
    "http://127.0.0.1:1420",
    "http://localhost:1420",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
)
EXEMPT_AUTH_PATHS = frozenset({
    "/api/health",
    "/api/auth/session",
    "/api/system/gpu-status",
})


# ---------------------------------------------------------------------------
# Thin wrappers that bind module-level constants to helper functions whose
# extracted signatures require them explicitly.
# ---------------------------------------------------------------------------

def _build_system_snapshot(*, capabilities: Any | None = None) -> dict[str, Any]:
    return _build_system_snapshot_impl(app_version, APP_STARTED_AT, capabilities=capabilities)


def _default_settings() -> dict[str, Any]:
    return _default_settings_impl(DEFAULT_PORT, DATA_LOCATION.data_dir)


def _load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    return _load_settings_impl(path, DEFAULT_PORT, DATA_LOCATION.data_dir)


def _save_settings(settings: dict[str, Any], path: Path = SETTINGS_PATH) -> None:
    return _save_settings_impl(settings, path)


def _load_benchmark_runs(path: Path = BENCHMARKS_PATH) -> list[dict[str, Any]]:
    return _load_benchmark_runs_impl(path)


def _save_benchmark_runs(runs: list[dict[str, Any]], path: Path = BENCHMARKS_PATH) -> None:
    return _save_benchmark_runs_impl(runs, path)


def _load_chat_sessions(path: Path = CHAT_SESSIONS_PATH) -> list[dict[str, Any]]:
    return _load_chat_sessions_impl(path)


def _save_chat_sessions(sessions: list[dict[str, Any]], path: Path = CHAT_SESSIONS_PATH) -> None:
    return _save_chat_sessions_impl(sessions, path)


def _resolve_output_dir_override(raw: str, default: Path) -> Path:
    """Return the user-chosen output directory, or the default.

    Empty / whitespace-only strings restore the default. A non-empty value is
    expanded (``~`` → home), resolved to an absolute path, and the directory is
    created if missing. If creation fails (path is unwritable, on a missing
    volume, etc.) we transparently fall back to ``default`` so generation never
    crashes just because the user pointed at a stale Dropbox folder.
    """
    value = (raw or "").strip()
    if not value:
        return default
    try:
        candidate = Path(os.path.expanduser(value)).resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except OSError:
        return default


def _current_image_outputs_dir() -> Path:
    # The module-level ``IMAGE_OUTPUTS_DIR`` is the install-time default and
    # the override target tests use to redirect output into a tempdir. Anything
    # the user typed in Settings takes precedence — but only when actually set,
    # so test patches still win when no setting is configured.
    settings = _load_settings()
    return _resolve_output_dir_override(
        str(settings.get("imageOutputsDirectory") or ""),
        IMAGE_OUTPUTS_DIR,
    )


def _current_video_outputs_dir() -> Path:
    settings = _load_settings()
    return _resolve_output_dir_override(
        str(settings.get("videoOutputsDirectory") or ""),
        VIDEO_OUTPUTS_DIR,
    )


def _load_image_outputs() -> list[dict[str, Any]]:
    return _load_image_outputs_impl(_current_image_outputs_dir())


def _save_image_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return _save_image_artifact_impl(artifact, _current_image_outputs_dir())


def _find_image_output(artifact_id: str) -> dict[str, Any] | None:
    return _find_image_output_impl(artifact_id, _current_image_outputs_dir())


def _delete_image_output(artifact_id: str) -> bool:
    return _delete_image_output_impl(artifact_id, _current_image_outputs_dir())


def _load_video_outputs() -> list[dict[str, Any]]:
    return _load_video_outputs_impl(_current_video_outputs_dir())


def _save_video_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return _save_video_artifact_impl(artifact, _current_video_outputs_dir())


def _find_video_output(artifact_id: str) -> dict[str, Any] | None:
    return _find_video_output_impl(artifact_id, _current_video_outputs_dir())


def _delete_video_output(artifact_id: str) -> bool:
    return _delete_video_output_impl(artifact_id, _current_video_outputs_dir())


def compute_cache_preview(
    *,
    bits: int = 3,
    fp16_layers: int = 4,
    num_layers: int = 32,
    num_heads: int = 32,
    num_kv_heads: int | None = None,
    hidden_size: int = 4096,
    context_tokens: int = 8192,
    params_b: float = 7.0,
    system_stats: dict[str, Any] | None = None,
    strategy: str = "native",
) -> dict[str, Any]:
    return compute_cache_preview_impl(
        bits=bits,
        fp16_layers=fp16_layers,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        context_tokens=context_tokens,
        params_b=params_b,
        system_stats=system_stats,
        strategy=strategy,
        build_system_snapshot=_build_system_snapshot,
    )


def _allowed_cors_origins() -> list[str]:
    configured = [
        item.strip()
        for item in os.getenv("CHAOSENGINE_ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for origin in (*DEFAULT_ALLOWED_ORIGINS, *configured):
        if origin in seen:
            continue
        seen.add(origin)
        ordered.append(origin)
    return ordered


def _resolve_api_token(explicit_token: str | None = None) -> str:
    token = (explicit_token or os.getenv("CHAOSENGINE_API_TOKEN") or "").strip()
    return token or secrets.token_urlsafe(32)


def _resolve_require_api_auth(settings: dict[str, Any]) -> bool:
    # Env var wins — useful for CI / headless scripts that need to drop
    # the bearer requirement without touching settings.json. Accepts any
    # of "0", "false", "no", "off" (case-insensitive) to disable.
    env_override = os.getenv("CHAOSENGINE_REQUIRE_AUTH")
    if env_override is not None:
        return env_override.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(settings.get("requireApiAuth", True))


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.strip().lower()
    if normalized in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _extract_auth_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.headers.get("x-chaosengine-token", "").strip()


# Functions that remain in app.py because they couple tightly to module-level
# state or route-handler logic.

def _hf_repo_from_link(link: str | None) -> str | None:
    if not link or "huggingface.co/" not in link:
        return None
    repo = link.split("huggingface.co/", 1)[1].strip("/")
    if not repo:
        return None
    return repo.split("/tree/", 1)[0].split("/blob/", 1)[0].strip("/")


def _get_cache_strategies() -> list[dict[str, Any]]:
    from cache_compression import registry
    return registry.available()


def _apply_draft_resolution(width: int, height: int) -> tuple[int, int]:
    """Scale w/h so the long edge is 512px, preserving aspect, div-by-8.

    Draft mode is a speed toggle: users iterate prompts at 512px where the
    transformer attention is ~3-4x faster (attention scales quadratically
    with spatial tokens), then switch off Preview for the full-res final
    render. We snap to multiples of 8 because the VAE's 8x downsampling
    factor requires it — non-divisible dims produce padding artifacts at
    the right/bottom edges on every supported pipeline.
    """
    longest = max(width, height)
    if longest <= 512:
        return width, height
    scale = 512 / longest
    scaled_w = max(256, (int(round(width * scale)) // 8) * 8)
    scaled_h = max(256, (int(round(height * scale)) // 8) * 8)
    return scaled_w, scaled_h


def _generate_image_artifacts(
    request: ImageGenerationRequest,
    variant: dict[str, Any],
    runtime_manager: ImageRuntimeManager | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import logging
    from backend_service.image_runtime import ImageGenerationConfig, ImageRuntimeManager

    logger = logging.getLogger("chaosengine.images")
    effective_width, effective_height = (
        _apply_draft_resolution(request.width, request.height)
        if request.draftMode
        else (request.width, request.height)
    )
    logger.info("Generating image: model=%s repo=%s size=%dx%d steps=%d draft=%s",
                variant.get("name"), variant.get("repo"), effective_width, effective_height, request.steps, request.draftMode)
    runtime_manager = runtime_manager or ImageRuntimeManager()
    # FU-019: variant-declared defaults override schema defaults only
    # when the user hasn't moved the slider. Schema defaults (24 steps,
    # CFG 5.5) come from ImageGenerationRequest in models/__init__.py.
    SCHEMA_DEFAULT_STEPS = 24
    SCHEMA_DEFAULT_GUIDANCE = 5.5
    effective_steps = request.steps
    effective_guidance = request.guidance
    variant_default_steps = variant.get("defaultSteps")
    variant_cfg_override = variant.get("cfgOverride")
    if variant_default_steps is not None and request.steps == SCHEMA_DEFAULT_STEPS:
        effective_steps = int(variant_default_steps)
    if variant_cfg_override is not None and abs(request.guidance - SCHEMA_DEFAULT_GUIDANCE) < 1e-3:
        effective_guidance = float(variant_cfg_override)

    rendered_images, runtime_status = runtime_manager.generate(
        ImageGenerationConfig(
            modelId=request.modelId,
            modelName=str(variant["name"]),
            repo=str(variant["repo"]),
            prompt=request.prompt,
            negativePrompt=request.negativePrompt or "",
            width=effective_width,
            height=effective_height,
            steps=effective_steps,
            guidance=effective_guidance,
            batchSize=request.batchSize,
            seed=request.seed,
            qualityPreset=request.qualityPreset,
            sampler=request.sampler,
            ggufRepo=(variant.get("ggufRepo") or None),
            ggufFile=(variant.get("ggufFile") or None),
            runtime=(variant.get("engine") or None),
            cacheStrategy=request.cacheStrategy,
            cacheRelL1Thresh=request.cacheRelL1Thresh,
            cfgDecay=request.cfgDecay,
            previewVae=request.previewVae,
            # FU-019: variant-declared LoRA + step / guidance overrides.
            # When the catalog variant pins a Hyper-SD / FLUX-Turbo /
            # lightx2v LoRA, the engine fuses it into the pipeline at
            # load time. ``defaultSteps`` / ``cfgOverride`` substitute
            # only when the user kept the schema defaults — explicit
            # slider tweaks survive untouched.
            loraRepo=(variant.get("loraRepo") or None),
            loraFile=(variant.get("loraFile") or None),
            loraScale=(variant.get("loraScale") if variant.get("loraScale") is not None else None),
            defaultSteps=(variant.get("defaultSteps") if variant.get("defaultSteps") is not None else None),
            cfgOverride=(variant.get("cfgOverride") if variant.get("cfgOverride") is not None else None),
        )
    )
    created_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    artifacts: list[dict[str, Any]] = []
    for rendered in rendered_images:
        artifact = {
            "artifactId": f"img-{uuid.uuid4().hex[:12]}",
            "modelId": request.modelId,
            "modelName": variant["name"],
            "prompt": request.prompt,
            "negativePrompt": request.negativePrompt or "",
            "width": effective_width,
            "height": effective_height,
            "steps": request.steps,
            "guidance": request.guidance,
            "seed": rendered.seed,
            "createdAt": created_at,
            "durationSeconds": rendered.durationSeconds,
            "imageBytes": rendered.bytes,
            "imageMimeType": rendered.mimeType,
            "imageExtension": rendered.extension,
            "runtimeLabel": rendered.runtimeLabel,
            "runtimeNote": rendered.runtimeNote,
            "qualityPreset": request.qualityPreset,
            "draftMode": request.draftMode,
        }
        artifacts.append(_save_image_artifact(artifact))
    return artifacts, runtime_status


def _generate_video_artifact(
    request: VideoGenerationRequest,
    variant: dict[str, Any],
    runtime_manager: VideoRuntimeManager,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a single video generation and persist it to the outputs dir.

    Returns ``(artifact_dict, runtime_status_dict)``. Unlike the image path,
    there is no placeholder fallback — if the runtime isn't ready or the
    generation fails, the caller sees the exception and surfaces a proper
    HTTP error rather than a fake clip.
    """
    import logging
    from backend_service.video_runtime import VideoGenerationConfig

    logger = logging.getLogger("chaosengine.video")
    logger.info(
        "Generating video: model=%s repo=%s size=%dx%d frames=%d steps=%d",
        variant.get("name"),
        variant.get("repo"),
        request.width,
        request.height,
        request.numFrames,
        request.steps,
    )

    # FU-019: variant-declared step / CFG defaults override schema
    # defaults only when the user kept the schema defaults — explicit
    # slider movement on the frontend is preserved untouched. The
    # video schema default is steps=50 (see VideoGenerationRequest).
    SCHEMA_DEFAULT_STEPS = 50
    SCHEMA_DEFAULT_GUIDANCE = 3.0
    effective_steps = request.steps
    effective_guidance = request.guidance
    variant_default_steps = variant.get("defaultSteps")
    variant_cfg_override = variant.get("cfgOverride")
    if variant_default_steps is not None and request.steps == SCHEMA_DEFAULT_STEPS:
        effective_steps = int(variant_default_steps)
    if variant_cfg_override is not None and abs(request.guidance - SCHEMA_DEFAULT_GUIDANCE) < 1e-3:
        effective_guidance = float(variant_cfg_override)

    video, runtime_status = runtime_manager.generate(
        VideoGenerationConfig(
            modelId=request.modelId,
            modelName=str(variant["name"]),
            repo=str(variant["repo"]),
            prompt=request.prompt,
            negativePrompt=request.negativePrompt or "",
            width=request.width,
            height=request.height,
            numFrames=request.numFrames,
            fps=request.fps,
            steps=effective_steps,
            guidance=effective_guidance,
            seed=request.seed,
            ggufRepo=(variant.get("ggufRepo") or None),
            ggufFile=(variant.get("ggufFile") or None),
            interpolationFactor=request.interpolationFactor,
            scheduler=request.scheduler,
            useNf4=request.useNf4,
            enableLtxRefiner=request.enableLtxRefiner,
            enhancePrompt=request.enhancePrompt,
            cfgDecay=request.cfgDecay,
            stgScale=request.stgScale,
            previewVae=request.previewVae,
            # FU-019: variant-declared LoRA + override metadata.
            loraRepo=(variant.get("loraRepo") or None),
            loraFile=(variant.get("loraFile") or None),
            loraScale=(variant.get("loraScale") if variant.get("loraScale") is not None else None),
            defaultSteps=(variant.get("defaultSteps") if variant.get("defaultSteps") is not None else None),
            cfgOverride=(variant.get("cfgOverride") if variant.get("cfgOverride") is not None else None),
            # Phase 3 / Wan2.2-Distill 4-step: catalog-pinned distilled
            # transformers replace both Wan A14B experts at pipeline load.
            distillTransformerRepo=(variant.get("distillTransformerRepo") or None),
            distillTransformerHighNoiseFile=(variant.get("distillTransformerHighNoiseFile") or None),
            distillTransformerLowNoiseFile=(variant.get("distillTransformerLowNoiseFile") or None),
            distillTransformerPrecision=(variant.get("distillTransformerPrecision") or None),
        )
    )

    created_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    clip_duration = round(video.frameCount / max(1, video.fps), 3)
    artifact = {
        "artifactId": f"vid-{uuid.uuid4().hex[:12]}",
        "modelId": request.modelId,
        "modelName": variant["name"],
        "prompt": request.prompt,
        "negativePrompt": request.negativePrompt or "",
        "width": video.width,
        "height": video.height,
        "numFrames": video.frameCount,
        "fps": video.fps,
        "steps": video.effectiveSteps if video.effectiveSteps is not None else request.steps,
        "guidance": video.effectiveGuidance if video.effectiveGuidance is not None else request.guidance,
        "seed": video.seed,
        "createdAt": created_at,
        "durationSeconds": video.durationSeconds,
        "clipDurationSeconds": clip_duration,
        "videoBytes": video.bytes,
        "videoMimeType": video.mimeType,
        "videoExtension": video.extension,
        "runtimeLabel": video.runtimeLabel,
        "runtimeNote": video.runtimeNote,
    }
    return _save_video_artifact(artifact), runtime_status


def create_app(
    state: ChaosEngineState | None = None,
    api_token: str | None = None,
) -> FastAPI:
    app = FastAPI(title="ChaosEngineAI Sidecar", version="0.2.0")
    allowed_origins = _allowed_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Authorization", "Content-Type", "X-ChaosEngine-Token"],
    )
    app.state.chaosengine = state or ChaosEngineState(
        server_port=DEFAULT_PORT,
        background_capability_probe=True,
    )
    app.state.chaosengine_api_token = _resolve_api_token(api_token)
    app.state.chaosengine_allowed_origins = frozenset(allowed_origins)
    # Bearer-token enforcement toggle. Reads from (in order) env override,
    # then saved settings, defaulting to True (keep the existing secure
    # default). Mutated live by state.update_settings so the user doesn't
    # need to restart the server to toggle it.
    app.state.chaosengine_require_api_auth = _resolve_require_api_auth(
        app.state.chaosengine.settings,
    )

    # Shutdown hook: kill any running llama-server / MLX worker children
    # on backend exit. Runs on clean shutdown (uvicorn SIGTERM), Ctrl-C,
    # and normal Python exit. Prevents orphan llama-server processes from
    # surviving across dev sessions.
    import atexit as _atexit
    import signal as _signal

    def _shutdown_children(*_args: Any) -> None:
        try:
            runtime = getattr(app.state.chaosengine, "runtime", None)
            if runtime is None:
                return
            engine = getattr(runtime, "engine", None)
            # LlamaCppEngine exposes _cleanup_process; MLXWorkerEngine has
            # a worker with close(); remote engines have nothing to do.
            if engine is not None:
                cleanup = getattr(engine, "_cleanup_process", None)
                if callable(cleanup):
                    try:
                        cleanup()
                    except Exception:
                        pass
                worker = getattr(engine, "worker", None)
                if worker is not None and hasattr(worker, "close"):
                    try:
                        worker.close()
                    except Exception:
                        pass
            # Also close any warm-pool engines.
            warm_pool = getattr(runtime, "_warm_pool", None)
            if isinstance(warm_pool, dict):
                for key, entry in list(warm_pool.items()):
                    try:
                        warm_engine = entry[0] if isinstance(entry, tuple) else entry
                        cleanup = getattr(warm_engine, "_cleanup_process", None)
                        if callable(cleanup):
                            cleanup()
                        worker = getattr(warm_engine, "worker", None)
                        if worker is not None and hasattr(worker, "close"):
                            worker.close()
                    except Exception:
                        pass
        except Exception:
            pass

    _atexit.register(_shutdown_children)
    # Also catch SIGTERM explicitly (uvicorn's normal shutdown signal).
    try:
        _signal.signal(_signal.SIGTERM, lambda *a: _shutdown_children())
    except (ValueError, OSError):
        pass  # not in main thread or signal not available

    @app.middleware("http")
    async def require_api_auth(request: Request, call_next):
        path = request.url.path
        if (
            request.method == "OPTIONS"
            or path in EXEMPT_AUTH_PATHS
            or not (path.startswith("/api/") or path.startswith("/v1/"))
            or not getattr(app.state, "chaosengine_require_api_auth", True)
        ):
            return await call_next(request)

        provided_token = _extract_auth_token(request)
        expected_token = str(app.state.chaosengine_api_token)
        if not provided_token or not secrets.compare_digest(provided_token, expected_token):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Unauthorized. Supply the ChaosEngineAI API token as a Bearer token.",
                },
            )
        return await call_next(request)

    @app.middleware("http")
    async def log_requests(request, call_next):
        path = request.url.path
        # Skip noisy internal polling endpoints that flood the log
        _quiet_paths = {
            "/api/server/logs/stream",
            "/api/health",
            "/api/workspace",
            "/api/runtime",
            "/api/cache/preview",
        }
        skip = path in _quiet_paths
        if not skip:
            # Don't log routine HTTP requests — they flood the log.
            # Meaningful operations (model load, chat, etc.) log themselves.
            pass
        response = await call_next(request)
        if not skip and response.status_code >= 400:
            app.state.chaosengine.add_log(
                "server", "warn",
                f"{request.method} {path} -> {response.status_code}",
            )
        return response

    register_routes(app)

    # Deliberately DO NOT call start_torch_warmup() here. Warmup eagerly
    # imports torch into the backend process, which on Windows pins every
    # torch/lib/*.dll into the process handle table. That blocks
    # /api/setup/install-gpu-bundle from pip-installing a new torch (pip's
    # --upgrade --target rmtree can't remove DLLs held by another process).
    # Warmup still exists for callers that want pre-priming (preload() in
    # the video/image runtimes triggers it) — it just isn't automatic.
    return app


app = create_app()


def _watch_parent_and_exit():
    """Kill ourselves and every child when the Tauri parent dies.

    Fires when the desktop shell crashes, gets force-closed from Task
    Manager / Activity Monitor, or is killed via Ctrl+C in dev. Without
    this, subprocess children we spawned (llama-server, llama-server-turbo,
    MLX worker) get re-parented to init/launchd and become multi-GB
    memory ghosts — the exact pattern the user reported where two
    llama-server.exe processes survived at 28 GB each.

    Platform semantics:
      - Unix (macOS / Linux): the backend was started inside its own
        session via setsid() in Tauri's pre_exec hook, so all our
        descendants share our process group. killpg signals the whole
        tree atomically. We send SIGTERM then SIGKILL 300ms later as a
        belt-and-braces — SIGTERM gives llama-server a chance to flush
        caches / release GPU handles cleanly, SIGKILL catches anything
        that was ignoring SIGTERM.
      - Windows: no killpg equivalent. os.kill(self, SIGTERM) just
        kills Python — the llama-server grandchildren still leak.
        The real fix on Windows is a Job Object created by the Tauri
        shell with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (implemented in
        src-tauri/src/lib.rs). When the Tauri process exits, Windows
        kernel kills the whole job. This watchdog runs first as a
        fast-path termination trigger; the Job Object is the safety
        net for the case where Python itself crashed before the
        watchdog fires.
    """
    import threading
    initial_ppid = os.getppid()
    if initial_ppid <= 1:
        return  # Already orphaned or running standalone

    def _watcher():
        while True:
            time.sleep(0.5)
            current_ppid = os.getppid()
            if current_ppid != initial_ppid or current_ppid == 1:
                try:
                    if hasattr(os, "killpg"):
                        # Unix: SIGTERM the whole process group, give
                        # llama-server a moment to release GPU VRAM,
                        # then SIGKILL as backup. killpg(pgrp, SIGKILL)
                        # kills us too since we're in the group, so the
                        # os._exit below is only reached if SIGKILL was
                        # somehow ignored (e.g. PID 1 protections).
                        os.killpg(os.getpgrp(), signal.SIGTERM)
                        time.sleep(0.3)
                        try:
                            os.killpg(os.getpgrp(), signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # group gone already, fine
                    else:
                        # Windows fallback. The Job Object in the Tauri
                        # shell is the real mechanism for Windows orphan
                        # prevention; this just makes sure Python itself
                        # exits fast so the Job handle closes promptly.
                        os.kill(os.getpid(), signal.SIGTERM)
                except Exception:
                    pass
                os._exit(0)

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()


def main() -> None:
    import uvicorn

    # Watch for parent death so we don't orphan ourselves
    _watch_parent_and_exit()

    uvicorn.run(
        "backend_service.app:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
