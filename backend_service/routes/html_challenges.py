"""HTML Challenge generation and persistence routes."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend_service.routes.compare import (
    COMPARE_SLOT_IDS,
    CompareModelRequest,
    resolve_compare_models,
)


class HtmlChallengeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    prompt: str = Field(min_length=1)
    models: list[CompareModelRequest] = Field(min_length=2, max_length=4)
    systemPrompt: str | None = None


router = APIRouter()


def _utc_label() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify(value: str, fallback: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80].strip("-") or fallback


def _challenge_root() -> Path:
    from backend_service.app import DATA_LOCATION

    root = DATA_LOCATION.data_dir / "html-challenges"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _challenge_dir(challenge_id: str) -> Path:
    root = _challenge_root().resolve()
    candidate = (root / challenge_id).resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(status_code=400, detail="Invalid challenge id.")
    return candidate


def _manifest_path(challenge_id: str) -> Path:
    return _challenge_dir(challenge_id) / "manifest.json"


def _format_token_setting(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    if value >= 1024:
        return f"{round(value / 1024)}K"
    return str(int(value))


def _format_size_gb(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    return f"{value:.1f} GB"


def _launch_summary(settings: dict[str, Any] | None) -> str:
    if not isinstance(settings, dict):
        return ""
    cache_strategy = str(settings.get("cacheStrategy") or "native")
    cache_bits = settings.get("cacheBits")
    cache_label = "Native f16" if cache_strategy == "native" else f"{cache_strategy} {cache_bits}-bit"
    parts = [cache_label]
    context = _format_token_setting(settings.get("contextTokens"))
    max_tokens = _format_token_setting(settings.get("maxTokens"))
    if context:
        parts.append(f"{context} ctx")
    if max_tokens:
        parts.append(f"{max_tokens} max")
    temperature = settings.get("temperature")
    if isinstance(temperature, (int, float)):
        parts.append(f"temp {temperature:.1f}")
    if settings.get("fusedAttention"):
        parts.append("Fused attention")
    if settings.get("speculativeDecoding"):
        tree_budget = settings.get("treeBudget")
        parts.append(f"DDTree {tree_budget}" if isinstance(tree_budget, int) and tree_budget > 0 else "DFlash")
    return " · ".join(parts)


def _write_model_settings(folder: Path, manifest: dict[str, Any]) -> None:
    lines = [
        str(manifest.get("title") or "HTML Challenge"),
        f"Created: {manifest.get('createdAt') or ''}",
        f"Folder: {manifest.get('folderPath') or folder}",
        "",
        "Prompt:",
        str(manifest.get("prompt") or ""),
        "",
        "Models:",
    ]
    for slot in manifest.get("slots", []):
        if not isinstance(slot, dict):
            continue
        lines.extend([
            "",
            str(slot.get("label") or f"Model {str(slot.get('slotId') or '').upper()}").strip(),
            str(slot.get("displayLabel") or slot.get("modelName") or slot.get("modelRef") or "").strip(),
        ])
        for key in ("format", "quantization"):
            value = str(slot.get(key) or "").strip()
            if value:
                lines.append(value)
        size_label = _format_size_gb(slot.get("sizeGb"))
        if size_label:
            lines.append(size_label)
        context_window = str(slot.get("contextWindow") or "").strip()
        if context_window:
            lines.append(context_window)
        launch = _launch_summary(slot.get("settings"))
        if launch:
            lines.append(launch)

    settings_path = folder / str(manifest.get("settingsFilename") or "model-settings.txt")
    tmp = settings_path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    os.replace(str(tmp), str(settings_path))


def _write_manifest(folder: Path, manifest: dict[str, Any]) -> None:
    _write_model_settings(folder, manifest)
    path = folder / "manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _read_manifest(challenge_id: str) -> dict[str, Any]:
    path = _manifest_path(challenge_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"HTML challenge '{challenge_id}' not found.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Challenge manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Challenge manifest is invalid.")
    return payload


def _challenge_file_path(challenge_id: str, slot_id: str) -> Path:
    manifest = _read_manifest(challenge_id)
    slot = next((item for item in manifest.get("slots", []) if item.get("slotId") == slot_id), None)
    if not isinstance(slot, dict) or not slot.get("filename"):
        raise HTTPException(status_code=404, detail=f"Challenge slot '{slot_id}' has no saved file.")
    folder = _challenge_dir(challenge_id).resolve()
    candidate = (folder / str(slot["filename"])).resolve()
    if folder not in candidate.parents and candidate != folder:
        raise HTTPException(status_code=400, detail="Invalid challenge file path.")
    if not candidate.exists():
        raise HTTPException(status_code=410, detail=f"Challenge file for slot '{slot_id}' is missing.")
    return candidate


def _extract_html_document(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    fence_matches = re.findall(r"```(?:html)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence_matches:
        preferred = next((match for match in fence_matches if "<html" in match.lower() or "<!doctype" in match.lower()), None)
        stripped = (preferred or fence_matches[0]).strip()

    lower = stripped.lower()
    starts = [index for index in (lower.find("<!doctype"), lower.find("<html")) if index >= 0]
    valid = bool(starts)
    if starts:
        start = min(starts)
        end = lower.rfind("</html>")
        if end >= start:
            return stripped[start:end + len("</html>")].strip(), True
        return stripped[start:].strip(), True
    return stripped, valid


def _html_system_prompt(extra: str | None) -> str:
    base = (
        "You are participating in an HTML Challenge. Return only a complete, "
        "standalone HTML document for the user's prompt. Include all CSS and "
        "JavaScript inline in that single document. Do not use Markdown fences, "
        "do not explain the code, and do not reference external network assets."
    )
    cleaned = (extra or "").strip()
    return f"{cleaned}\n\n{base}" if cleaned else base


def _settings_payload(launch: Any) -> dict[str, Any]:
    return {
        "temperature": launch.temperature,
        "maxTokens": launch.maxTokens,
        "cacheStrategy": launch.cacheStrategy,
        "cacheBits": launch.cacheBits,
        "fp16Layers": launch.fp16Layers,
        "fusedAttention": launch.fusedAttention,
        "fitModelInMemory": launch.fitModelInMemory,
        "contextTokens": launch.contextTokens,
        "speculativeDecoding": launch.speculativeDecoding,
        "treeBudget": launch.treeBudget,
    }


def _model_display_payload(model: CompareModelRequest) -> dict[str, Any]:
    return {
        "displayLabel": model.displayLabel or model.modelName or model.modelRef,
        "displayDetail": model.displayDetail or "",
        "format": model.format,
        "quantization": model.quantization,
        "sizeGb": model.sizeGb,
        "contextWindow": model.contextWindow,
    }


def _requested_runtime_payload(state: Any, launch: Any) -> dict[str, Any]:
    return state._requested_runtime_metrics_fields(
        cache_strategy=launch.cacheStrategy,
        cache_bits=launch.cacheBits,
        fp16_layers=launch.fp16Layers,
        fit_model_in_memory=launch.fitModelInMemory,
        speculative_decoding=launch.speculativeDecoding,
        tree_budget=launch.treeBudget,
    )


def _loaded_model_metrics(state: Any) -> dict[str, Any]:
    metrics = state._loaded_model_metrics_fields().copy()
    metrics.pop("model", None)
    return metrics


def _done_runtime_payload(
    state: Any,
    *,
    final_chunk: Any,
    elapsed_seconds: float,
    requested_runtime: dict[str, Any],
) -> dict[str, Any]:
    completion_tokens = final_chunk.completion_tokens if final_chunk else 0
    prompt_tokens = final_chunk.prompt_tokens if final_chunk else 0
    tok_s = final_chunk.tok_s or (
        completion_tokens / max(elapsed_seconds, 0.01) if completion_tokens else 0
    )
    payload = {
        **_loaded_model_metrics(state),
        **state._result_runtime_metrics_fields(final_chunk),
        **requested_runtime,
        "finishReason": final_chunk.finish_reason if final_chunk else "stop",
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "totalTokens": prompt_tokens + completion_tokens,
        "tokS": round(tok_s, 1),
        "responseSeconds": elapsed_seconds,
        "runtimeNote": (
            final_chunk.runtime_note
            if final_chunk and getattr(final_chunk, "runtime_note", None) is not None
            else state.runtime.loaded_model.runtimeNote if state.runtime.loaded_model else None
        ),
    }
    if final_chunk and getattr(final_chunk, "dflash_acceptance_rate", None) is not None:
        payload["dflashAcceptanceRate"] = final_chunk.dflash_acceptance_rate
    return payload


def _load_model(state: Any, model: CompareModelRequest) -> None:
    from backend_service.models import LoadModelRequest

    launch = model.launch
    state.load_model(
        LoadModelRequest(
            modelRef=model.modelRef,
            modelName=model.modelName,
            canonicalRepo=model.canonicalRepo,
            source=model.source,
            path=model.path,
            backend=model.backend,
            cacheStrategy=launch.cacheStrategy,
            cacheBits=launch.cacheBits,
            fp16Layers=launch.fp16Layers,
            fusedAttention=launch.fusedAttention,
            fitModelInMemory=launch.fitModelInMemory,
            contextTokens=launch.contextTokens,
            speculativeDecoding=launch.speculativeDecoding,
            treeBudget=launch.treeBudget,
        ),
        keep_warm_previous=False,
    )


def _unload_active_model(state: Any) -> None:
    try:
        state.unload_model()
    except Exception as exc:
        state.add_log(
            "runtime",
            "warning",
            f"HTML Challenge could not unload active model after slot: {type(exc).__name__}: {exc}",
        )


@router.get("/api/chat/html-challenges")
def list_html_challenges() -> dict[str, Any]:
    challenges: list[dict[str, Any]] = []
    root = _challenge_root()
    for manifest_path in root.glob("*/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            challenges.append(payload)
    challenges.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return {"challenges": challenges}


@router.get("/api/chat/html-challenges/{challenge_id}")
def get_html_challenge(challenge_id: str) -> dict[str, Any]:
    return {"challenge": _read_manifest(challenge_id)}


@router.get("/api/chat/html-challenges/{challenge_id}/files/{slot_id}")
def get_html_challenge_file(challenge_id: str, slot_id: str) -> HTMLResponse:
    html = _challenge_file_path(challenge_id, slot_id).read_text(encoding="utf-8")
    headers = {
        "Content-Security-Policy": "default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline';",
        "X-Content-Type-Options": "nosniff",
    }
    return HTMLResponse(content=html, headers=headers)


@router.post("/api/chat/html-challenges")
def run_html_challenge(request: Request, body: HtmlChallengeRequest) -> StreamingResponse:
    state = request.app.state.chaosengine
    models = resolve_compare_models(body)
    created_at = _utc_label()
    title_slug = _slugify(body.title, "html-challenge")
    challenge_id = f"{title_slug}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    folder = _challenge_dir(challenge_id)
    folder.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "id": challenge_id,
        "title": body.title.strip(),
        "prompt": body.prompt,
        "systemPrompt": body.systemPrompt or "",
        "createdAt": created_at,
        "updatedAt": created_at,
        "folderPath": str(folder),
        "settingsFilename": "model-settings.txt",
        "settingsPath": str(folder / "model-settings.txt"),
        "slots": [
            {
                "slotId": COMPARE_SLOT_IDS[index],
                "label": f"Model {COMPARE_SLOT_IDS[index].upper()}",
                "status": "queued",
                "modelRef": model.modelRef,
                "modelName": model.modelName or model.modelRef,
                **_model_display_payload(model),
                "canonicalRepo": model.canonicalRepo,
                "source": model.source,
                "backend": model.backend,
                "path": model.path,
                "settings": _settings_payload(model.launch),
            }
            for index, model in enumerate(models)
        ],
    }
    _write_manifest(folder, manifest)

    def _sse_event(data: dict[str, Any]) -> str:
        return f"data: {json.dumps(data)}\n\n"

    def _update_slot(slot_id: str, patch: dict[str, Any]) -> None:
        for slot in manifest["slots"]:
            if slot["slotId"] == slot_id:
                slot.update(patch)
                break
        manifest["updatedAt"] = _utc_label()
        _write_manifest(folder, manifest)

    def _sse_stream():
        cleared_warm_models = state.runtime.clear_warm_pool()
        if cleared_warm_models:
            state.add_log(
                "runtime",
                "info",
                f"HTML Challenge cleared {cleared_warm_models} warm model(s) before exclusive loading.",
            )
        yield _sse_event({"challengeStarted": True, "challenge": manifest})

        for index, model in enumerate(models):
            slot_id = COMPARE_SLOT_IDS[index]
            model_label = model.modelName or model.modelRef
            requested_runtime = _requested_runtime_payload(state, model.launch)
            _update_slot(slot_id, {"status": "loading"})
            yield _sse_event({
                "model": slot_id,
                "loading": True,
                "message": f"Loading {model_label}...",
                "challenge": manifest,
            })

            load_start = time.perf_counter()
            try:
                _load_model(state, model)
                load_seconds = round(time.perf_counter() - load_start, 2)
                _update_slot(slot_id, {"status": "running", "loadSeconds": load_seconds})
                yield _sse_event({
                    "model": slot_id,
                    "loaded": True,
                    "loadSeconds": load_seconds,
                    **_loaded_model_metrics(state),
                    **requested_runtime,
                })
            except Exception as exc:
                _update_slot(slot_id, {"status": "error", "error": str(exc)})
                yield _sse_event({"model": slot_id, "error": str(exc), "challenge": manifest})
                yield _sse_event({"challengeDone": True, "challenge": manifest})
                return

            full_text = ""
            final_chunk = None
            gen_start = time.perf_counter()
            try:
                for chunk in state.runtime.stream_generate(
                    prompt=body.prompt,
                    history=[],
                    system_prompt=_html_system_prompt(body.systemPrompt),
                    max_tokens=model.launch.maxTokens,
                    temperature=model.launch.temperature,
                ):
                    if chunk.reasoning:
                        yield _sse_event({"model": slot_id, "reasoning": chunk.reasoning})
                    if chunk.reasoning_done:
                        yield _sse_event({"model": slot_id, "reasoningDone": True})
                    if chunk.text:
                        full_text += chunk.text
                        yield _sse_event({"model": slot_id, "token": chunk.text})
                    if chunk.done:
                        final_chunk = chunk
            except Exception as exc:
                _update_slot(slot_id, {"status": "error", "error": str(exc)})
                yield _sse_event({"model": slot_id, "error": str(exc), "challenge": manifest})
            else:
                elapsed = round(time.perf_counter() - gen_start, 2)
                html, valid_html = _extract_html_document(full_text)
                model_slug = _slugify(model_label, f"model-{index + 1}")
                filename = f"{slot_id}-{model_slug}.html"
                html_path = folder / filename
                html_path.write_text(html, encoding="utf-8")
                file_bytes = html_path.stat().st_size
                metrics = _done_runtime_payload(
                    state,
                    final_chunk=final_chunk,
                    elapsed_seconds=elapsed,
                    requested_runtime=requested_runtime,
                )
                slot_patch = {
                    "status": "done",
                    "filename": filename,
                    "filePath": str(html_path),
                    "fileBytes": file_bytes,
                    "validHtmlDocument": valid_html,
                    "metrics": metrics,
                    "responseSeconds": elapsed,
                    "loadSeconds": load_seconds,
                    "totalSeconds": round(load_seconds + elapsed, 2),
                }
                _update_slot(slot_id, slot_patch)
                yield _sse_event({
                    "model": slot_id,
                    "done": True,
                    "text": full_text,
                    "html": html,
                    "filename": filename,
                    "filePath": str(html_path),
                    "fileBytes": file_bytes,
                    "validHtmlDocument": valid_html,
                    "loadSeconds": load_seconds,
                    "totalSeconds": round(load_seconds + elapsed, 2),
                    "challenge": manifest,
                    **metrics,
                })
            finally:
                _unload_active_model(state)
                state.runtime.clear_warm_pool()

        yield _sse_event({"challengeDone": True, "challenge": manifest})

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
