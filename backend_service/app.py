from __future__ import annotations

import os
import ipaddress
import secrets
import signal
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from backend_service.image_runtime import (
    ImageGenerationConfig,
    ImageRuntimeManager,
)
from backend_service.video_runtime import (
    VideoGenerationConfig,
    VideoRuntimeManager,
)
from backend_service.models import ImageGenerationRequest, VideoGenerationRequest
from backend_service.routes import register_routes
from backend_service.state import ChaosEngineState

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
DOCUMENTS_DIR = DATA_LOCATION.documents_dir
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
})


# ---------------------------------------------------------------------------
# Thin wrappers that bind module-level constants to helper functions whose
# extracted signatures require them explicitly.
# ---------------------------------------------------------------------------

def _build_system_snapshot() -> dict[str, Any]:
    return _build_system_snapshot_impl(app_version, APP_STARTED_AT)


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


def _generate_image_artifacts(
    request: ImageGenerationRequest,
    variant: dict[str, Any],
    runtime_manager: ImageRuntimeManager | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import logging
    logger = logging.getLogger("chaosengine.images")
    logger.info("Generating image: model=%s repo=%s size=%dx%d steps=%d",
                variant.get("name"), variant.get("repo"), request.width, request.height, request.steps)
    runtime_manager = runtime_manager or ImageRuntimeManager()
    rendered_images, runtime_status = runtime_manager.generate(
        ImageGenerationConfig(
            modelId=request.modelId,
            modelName=str(variant["name"]),
            repo=str(variant["repo"]),
            prompt=request.prompt,
            negativePrompt=request.negativePrompt or "",
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance=request.guidance,
            batchSize=request.batchSize,
            seed=request.seed,
            qualityPreset=request.qualityPreset,
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
            "width": request.width,
            "height": request.height,
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
            steps=request.steps,
            guidance=request.guidance,
            seed=request.seed,
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
        "steps": request.steps,
        "guidance": request.guidance,
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
    app.state.chaosengine = state or ChaosEngineState(server_port=DEFAULT_PORT)
    app.state.chaosengine_api_token = _resolve_api_token(api_token)
    app.state.chaosengine_allowed_origins = frozenset(allowed_origins)

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
    return app


app = create_app()


def _watch_parent_and_exit():
    """Exit if our parent process dies (e.g. Tauri shell killed via Ctrl+C).

    This prevents orphaned backend + MLX worker processes from holding
    GPU memory after the desktop app shuts down.
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
                # Parent died — kill ourselves and any subprocess children
                try:
                    if hasattr(os, "killpg"):
                        # Unix: kill our entire process group (includes MLX worker children)
                        os.killpg(os.getpgrp(), signal.SIGTERM)
                    else:
                        # Windows: terminate our own process
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
