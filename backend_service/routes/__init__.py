"""Route registration for the ChaosEngineAI backend API."""

from __future__ import annotations

from fastapi import FastAPI


def register_routes(app: FastAPI) -> None:
    from .auth import router as auth_router
    from .health import router as health_router
    from .models import router as models_router
    from .chat import router as chat_router
    from .images import router as images_router
    from .video import router as video_router
    from .benchmarks import router as benchmarks_router
    from .cache import router as cache_router
    from .server import router as server_router
    from .settings import router as settings_router
    from .setup import router as setup_router
    from .openai_compat import router as openai_compat_router
    from .compare import router as compare_router
    from .metrics import router as metrics_router
    from .plugins import router as plugins_router
    from .finetuning import router as finetuning_router
    from .prompts import router as prompts_router
    from .diagnostics import router as diagnostics_router
    from .storage import router as storage_router
    from .workspaces import router as workspaces_router

    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(chat_router)
    app.include_router(compare_router)
    app.include_router(images_router)
    app.include_router(video_router)
    app.include_router(benchmarks_router)
    app.include_router(cache_router)
    app.include_router(server_router)
    app.include_router(settings_router)
    app.include_router(setup_router)
    app.include_router(openai_compat_router)
    app.include_router(metrics_router)
    app.include_router(plugins_router)
    app.include_router(finetuning_router)
    app.include_router(prompts_router)
    app.include_router(diagnostics_router)
    app.include_router(storage_router)
    app.include_router(workspaces_router)
