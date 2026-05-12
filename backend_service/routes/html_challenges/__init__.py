"""HTML Challenge generation and persistence routes.

The package exposes ``router`` (FastAPI ``APIRouter``) plus the request
Pydantic models. Heavy lifting (manifest I/O, HTML validation, slot
streaming) lives in ``_helpers.py`` so this file stays focused on the
endpoint surface.

Re-exports the underscore helpers tests historically mutated/inspected
via the package namespace (``_challenge_root``, ``_extract_html_document``,
``_validate_html_document``, etc.) — older tests still reach in for those.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend_service.routes.compare import (
    COMPARE_SLOT_IDS,
    CompareModelRequest,
    resolve_compare_models,
)

from backend_service.routes.html_challenges._helpers import (
    _challenge_asset_path,
    _challenge_dir,
    _challenge_file_path,
    _challenge_root,
    _clear_slot_result_payload,
    _extract_html_document,
    _find_manifest_slot,
    _html_validation_payload,
    _model_sampler_payload,
    _normalized_reasoning_effort,
    _normalized_thinking_mode,
    _open_default_app,
    _read_manifest,
    _repair_prompt,
    _sampler_overrides,
    _slot_manifest_payload,
    _slugify,
    _sse_event,
    _strip_model_names_from_title,
    _stream_html_challenge_slot,
    _update_manifest_slot,
    _utc_label,
    _validate_html_document,
    _write_manifest,
)


class HtmlChallengeModelRequest(CompareModelRequest):
    thinkingMode: str | None = Field(default=None, pattern="^(off|auto)$")
    reasoningEffort: str | None = Field(default=None, pattern="^(low|medium|high)$")
    seed: int | None = Field(default=None, ge=0, le=2147483647)


class HtmlChallengeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    prompt: str = Field(min_length=1)
    models: list[HtmlChallengeModelRequest] = Field(min_length=2, max_length=4)
    systemPrompt: str | None = None
    # Backwards compatibility for older clients. New clients send these per
    # model so each slot can test a different reasoning setting.
    thinkingMode: str | None = Field(default="off", pattern="^(off|auto)$")
    reasoningEffort: str | None = Field(default=None, pattern="^(low|medium|high)$")


class HtmlChallengeRetryRequest(BaseModel):
    model: HtmlChallengeModelRequest
    systemPrompt: str | None = None
    thinkingMode: str | None = Field(default=None, pattern="^(off|auto)$")
    reasoningEffort: str | None = Field(default=None, pattern="^(low|medium|high)$")


class HtmlChallengeRepairRequest(BaseModel):
    mode: str = Field(pattern="^(continue|repair)$")
    model: HtmlChallengeModelRequest
    systemPrompt: str | None = None
    thinkingMode: str | None = Field(default=None, pattern="^(off|auto)$")
    reasoningEffort: str | None = Field(default=None, pattern="^(low|medium|high)$")


class HtmlChallengeValidationUpdateRequest(BaseModel):
    status: str = Field(pattern="^(valid|partial|script-error|blank-render|no-html)$")
    message: str | None = Field(default=None, max_length=500)
    issues: list[str] = Field(default_factory=list, max_length=12)
    source: str | None = Field(default="runtime", max_length=40)


class HtmlChallengeOpenFileRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


router = APIRouter()


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


@router.delete("/api/chat/html-challenges/{challenge_id}")
def delete_html_challenge(challenge_id: str) -> dict[str, Any]:
    folder = _challenge_dir(challenge_id)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"HTML challenge '{challenge_id}' not found.")
    if not (folder / "manifest.json").exists():
        raise HTTPException(status_code=404, detail=f"HTML challenge '{challenge_id}' not found.")
    # Soft-delete: move the challenge folder into a sibling `.trash/` so the
    # user can restore it manually from disk if they regret the click. No
    # native-OS-trash dependency required; works the same on macOS / Linux /
    # Windows. Append a timestamp suffix when the trash already holds an
    # entry with the same id (e.g. user re-created and re-deleted).
    trash_root = _challenge_root() / ".trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    target_name = challenge_id
    target = trash_root / target_name
    if target.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        target_name = f"{challenge_id}-{suffix}"
        target = trash_root / target_name
    try:
        os.replace(str(folder), str(target))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not move HTML challenge to trash: {exc}") from exc
    return {"deleted": challenge_id, "trashedAs": str(target)}


@router.get("/api/chat/html-challenges/{challenge_id}/files/{slot_id}")
def get_html_challenge_file(challenge_id: str, slot_id: str) -> HTMLResponse:
    html = _challenge_file_path(challenge_id, slot_id).read_text(encoding="utf-8")
    headers = {
        "Content-Security-Policy": "default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline';",
        "X-Content-Type-Options": "nosniff",
    }
    return HTMLResponse(content=html, headers=headers)


@router.post("/api/chat/html-challenges/open-file")
def open_html_challenge_file(body: HtmlChallengeOpenFileRequest) -> dict[str, Any]:
    path = _challenge_asset_path(body.path)
    try:
        _open_default_app(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not open challenge file: {exc}") from exc
    return {"opened": str(path)}


@router.post("/api/chat/html-challenges/{challenge_id}/slots/{slot_id}/retry")
def retry_html_challenge_slot(
    challenge_id: str,
    slot_id: str,
    request: Request,
    body: HtmlChallengeRetryRequest,
) -> StreamingResponse:
    slot_id = slot_id.lower()
    if slot_id not in COMPARE_SLOT_IDS:
        raise HTTPException(status_code=400, detail="Invalid challenge slot.")

    state = request.app.state.chaosengine
    folder = _challenge_dir(challenge_id)
    manifest = _read_manifest(challenge_id)
    manifest_slot = _find_manifest_slot(manifest, slot_id)
    if manifest_slot is None:
        raise HTTPException(status_code=404, detail=f"Challenge slot '{slot_id}' was not found.")
    prompt = str(manifest.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Challenge prompt is missing.")

    default_thinking_mode = (
        body.thinkingMode
        if body.thinkingMode is not None
        else manifest_slot.get("thinkingMode") or manifest.get("thinkingMode") or "off"
    )
    default_reasoning_effort = (
        body.reasoningEffort
        if body.reasoningEffort is not None
        else manifest_slot.get("reasoningEffort") or manifest.get("reasoningEffort")
    )
    slot_payload = _slot_manifest_payload(
        slot_id,
        body.model,
        status="queued",
        default_thinking_mode=default_thinking_mode,
        default_reasoning_effort=default_reasoning_effort,
    )
    thinking_mode = slot_payload["thinkingMode"]
    reasoning_effort = slot_payload.get("reasoningEffort")
    sampler_overrides = _sampler_overrides(body.model, manifest_slot=manifest_slot)
    slot_payload.update(_model_sampler_payload(body.model, manifest_slot=manifest_slot))

    _update_manifest_slot(
        folder,
        manifest,
        slot_id,
        {
            **slot_payload,
            **_clear_slot_result_payload(),
        },
    )

    def _sse_stream():
        cleared_warm_models = state.runtime.clear_warm_pool()
        if cleared_warm_models:
            state.add_log(
                "runtime",
                "info",
                f"HTML Challenge cleared {cleared_warm_models} warm model(s) before exclusive loading.",
            )
        yield _sse_event({"challengeStarted": True, "challenge": manifest})
        yield from _stream_html_challenge_slot(
            state=state,
            manifest=manifest,
            folder=folder,
            slot_id=slot_id,
            model=body.model,
            prompt=prompt,
            system_prompt=body.systemPrompt if body.systemPrompt is not None else manifest.get("systemPrompt"),
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort if thinking_mode == "auto" else None,
            sampler_overrides=sampler_overrides,
        )
        yield _sse_event({"challengeDone": True, "challenge": manifest})

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/api/chat/html-challenges/{challenge_id}/slots/{slot_id}/repair")
def repair_html_challenge_slot(
    challenge_id: str,
    slot_id: str,
    request: Request,
    body: HtmlChallengeRepairRequest,
) -> StreamingResponse:
    slot_id = slot_id.lower()
    if slot_id not in COMPARE_SLOT_IDS:
        raise HTTPException(status_code=400, detail="Invalid challenge slot.")

    state = request.app.state.chaosengine
    folder = _challenge_dir(challenge_id)
    manifest = _read_manifest(challenge_id)
    manifest_slot = _find_manifest_slot(manifest, slot_id)
    if manifest_slot is None:
        raise HTTPException(status_code=404, detail=f"Challenge slot '{slot_id}' was not found.")
    original_prompt = str(manifest.get("prompt") or "").strip()
    if not original_prompt:
        raise HTTPException(status_code=422, detail="Challenge prompt is missing.")
    partial_html = _challenge_file_path(challenge_id, slot_id).read_text(encoding="utf-8")
    if not partial_html.strip():
        raise HTTPException(status_code=422, detail="Challenge slot has no partial HTML to repair.")

    default_thinking_mode = (
        body.thinkingMode
        if body.thinkingMode is not None
        else manifest_slot.get("thinkingMode") or manifest.get("thinkingMode") or "off"
    )
    default_reasoning_effort = (
        body.reasoningEffort
        if body.reasoningEffort is not None
        else manifest_slot.get("reasoningEffort") or manifest.get("reasoningEffort")
    )
    slot_payload = _slot_manifest_payload(
        slot_id,
        body.model,
        status="queued",
        default_thinking_mode=default_thinking_mode,
        default_reasoning_effort=default_reasoning_effort,
    )
    thinking_mode = slot_payload["thinkingMode"]
    reasoning_effort = slot_payload.get("reasoningEffort")
    sampler_overrides = _sampler_overrides(body.model, manifest_slot=manifest_slot)
    slot_payload.update(_model_sampler_payload(body.model, manifest_slot=manifest_slot))

    _update_manifest_slot(
        folder,
        manifest,
        slot_id,
        {
            **slot_payload,
            **_clear_slot_result_payload(),
            "repairMode": body.mode,
        },
    )

    def _sse_stream():
        cleared_warm_models = state.runtime.clear_warm_pool()
        if cleared_warm_models:
            state.add_log(
                "runtime",
                "info",
                f"HTML Challenge cleared {cleared_warm_models} warm model(s) before exclusive loading.",
            )
        yield _sse_event({"challengeStarted": True, "challenge": manifest})
        yield from _stream_html_challenge_slot(
            state=state,
            manifest=manifest,
            folder=folder,
            slot_id=slot_id,
            model=body.model,
            prompt=_repair_prompt(original_prompt, partial_html, body.mode),
            system_prompt=body.systemPrompt if body.systemPrompt is not None else manifest.get("systemPrompt"),
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort if thinking_mode == "auto" else None,
            sampler_overrides=sampler_overrides,
        )
        yield _sse_event({"challengeDone": True, "challenge": manifest})

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.patch("/api/chat/html-challenges/{challenge_id}/slots/{slot_id}/validation")
def update_html_challenge_slot_validation(
    challenge_id: str,
    slot_id: str,
    body: HtmlChallengeValidationUpdateRequest,
) -> dict[str, Any]:
    slot_id = slot_id.lower()
    if slot_id not in COMPARE_SLOT_IDS:
        raise HTTPException(status_code=400, detail="Invalid challenge slot.")

    folder = _challenge_dir(challenge_id)
    manifest = _read_manifest(challenge_id)
    slot = _find_manifest_slot(manifest, slot_id)
    if slot is None:
        raise HTTPException(status_code=404, detail=f"Challenge slot '{slot_id}' was not found.")

    existing = slot.get("htmlValidation") if isinstance(slot.get("htmlValidation"), dict) else {}
    checks = existing.get("checks") if isinstance(existing, dict) and isinstance(existing.get("checks"), dict) else {}
    issues = [item.strip() for item in body.issues if item.strip()]
    if body.message and body.message.strip() and body.message.strip() not in issues:
        issues.insert(0, body.message.strip())
    validation = _html_validation_payload(
        body.status,
        issues,
        checks=checks,
        source=body.source or "runtime",
    )
    _update_manifest_slot(
        folder,
        manifest,
        slot_id,
        {
            "htmlValidation": validation,
            "validHtmlDocument": body.status == "valid",
        },
    )
    return {"challenge": manifest}


@router.post("/api/chat/html-challenges")
def run_html_challenge(request: Request, body: HtmlChallengeRequest) -> StreamingResponse:
    state = request.app.state.chaosengine
    models = resolve_compare_models(body)
    created_at = _utc_label()
    challenge_title = _strip_model_names_from_title(body.title, body.models)
    title_slug = _slugify(challenge_title, "html-challenge")
    challenge_id = f"{title_slug}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    folder = _challenge_dir(challenge_id)
    folder.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "id": challenge_id,
        "title": challenge_title,
        "prompt": body.prompt,
        "systemPrompt": body.systemPrompt or "",
        "thinkingMode": body.thinkingMode or "off",
        "reasoningEffort": body.reasoningEffort if body.thinkingMode == "auto" else None,
        "createdAt": created_at,
        "updatedAt": created_at,
        "folderPath": str(folder),
        "settingsFilename": "model-settings.txt",
        "settingsPath": str(folder / "model-settings.txt"),
        "slots": [
            _slot_manifest_payload(
                COMPARE_SLOT_IDS[index],
                model,
                default_thinking_mode=body.thinkingMode or "off",
                default_reasoning_effort=body.reasoningEffort,
            )
            for index, model in enumerate(models)
        ],
    }
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
            manifest_slot = _find_manifest_slot(manifest, slot_id) or {}
            thinking_mode = _normalized_thinking_mode(
                manifest_slot.get("thinkingMode"),
                body.thinkingMode or "off",
            )
            reasoning_effort = _normalized_reasoning_effort(
                manifest_slot.get("reasoningEffort"),
                body.reasoningEffort,
            )
            sampler_overrides = _sampler_overrides(model, manifest_slot=manifest_slot)
            loaded = yield from _stream_html_challenge_slot(
                state=state,
                manifest=manifest,
                folder=folder,
                slot_id=slot_id,
                model=model,
                prompt=body.prompt,
                system_prompt=body.systemPrompt,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort if thinking_mode == "auto" else None,
                sampler_overrides=sampler_overrides,
            )
            if not loaded:
                yield _sse_event({"challengeDone": True, "challenge": manifest})
                return

        yield _sse_event({"challengeDone": True, "challenge": manifest})

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
