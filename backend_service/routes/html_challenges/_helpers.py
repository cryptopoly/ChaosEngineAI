"""HTML Challenge helpers — manifest I/O, HTML validation, slot streaming.

Extracted from ``routes/html_challenges.py`` as part of the v0.8.0 refactor.
The package's ``__init__`` re-exports the underscore helpers tests reach
into via ``backend_service.routes.html_challenges._<name>`` so existing
test imports keep working.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend_service.routes.compare import CompareModelRequest


def _sse_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _utc_label() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify(value: str, fallback: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80].strip("-") or fallback


def _model_title_fragments(model: Any) -> list[str]:
    fragments: list[str] = []
    for value in (model.displayLabel, model.modelName, model.modelRef):
        if not value:
            continue
        candidates = [str(value).strip()]
        if "/" in str(value):
            candidates.append(str(value).rsplit("/", 1)[-1].strip())
        for candidate in candidates:
            if len(candidate) >= 4 and candidate not in fragments:
                fragments.append(candidate)
    return fragments


def _strip_model_names_from_title(title: str, models: list[Any]) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip()
    earliest_model_index: int | None = None
    lowered = cleaned.lower()
    for model in models:
        for fragment in _model_title_fragments(model):
            index = lowered.find(fragment.lower())
            if index > 0 and (earliest_model_index is None or index < earliest_model_index):
                earliest_model_index = index
    if earliest_model_index is None:
        return cleaned

    candidate = cleaned[:earliest_model_index].strip(" \t-–—·:|,/+&")
    candidate = re.sub(r"(?i)(?:\s+(?:vs|versus|and))+$", "", candidate).strip(" \t-–—·:|,/+&")
    return candidate or cleaned


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


def _thinking_summary(thinking_mode: Any, reasoning_effort: Any = None) -> str:
    if thinking_mode != "auto":
        return "Thinking off"
    if reasoning_effort in {"low", "medium", "high"}:
        return f"Thinking {str(reasoning_effort)}"
    return "Thinking auto"


def _normalized_thinking_mode(value: Any, fallback: Any = "off") -> str:
    candidate = value if value in {"off", "auto"} else fallback
    return "auto" if candidate == "auto" else "off"


def _normalized_reasoning_effort(value: Any, fallback: Any = None) -> str | None:
    candidate = value if value in {"low", "medium", "high"} else fallback
    return candidate if candidate in {"low", "medium", "high"} else None


def _model_thinking_payload(
    model: Any,
    *,
    default_thinking_mode: Any = "off",
    default_reasoning_effort: Any = None,
) -> dict[str, Any]:
    thinking_mode = _normalized_thinking_mode(
        getattr(model, "thinkingMode", None),
        default_thinking_mode,
    )
    reasoning_effort = _normalized_reasoning_effort(
        getattr(model, "reasoningEffort", None),
        default_reasoning_effort,
    )
    return {
        "thinkingMode": thinking_mode,
        "reasoningEffort": reasoning_effort if thinking_mode == "auto" else None,
    }


def _model_sampler_payload(model: Any, *, manifest_slot: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_slot = manifest_slot or {}
    seed = getattr(model, "seed", None)
    if seed is None and isinstance(manifest_slot.get("seed"), int):
        seed = manifest_slot.get("seed")
    return {"seed": seed if isinstance(seed, int) else None}


def _sampler_overrides(model: Any, *, manifest_slot: dict[str, Any] | None = None) -> dict[str, Any] | None:
    payload = _model_sampler_payload(model, manifest_slot=manifest_slot)
    if payload["seed"] is None:
        return None
    return {"seed": payload["seed"]}


def _sampler_summary(slot: dict[str, Any]) -> list[str]:
    if isinstance(slot.get("seed"), int):
        return [f"Seed {slot['seed']}"]
    return []


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
        lines.append(_thinking_summary(
            slot.get("thinkingMode") or manifest.get("thinkingMode") or "off",
            slot.get("reasoningEffort") or manifest.get("reasoningEffort"),
        ))
        lines.extend(_sampler_summary(slot))

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


def _find_manifest_slot(manifest: dict[str, Any], slot_id: str) -> dict[str, Any] | None:
    return next(
        (item for item in manifest.get("slots", []) if isinstance(item, dict) and item.get("slotId") == slot_id),
        None,
    )


def _update_manifest_slot(folder: Path, manifest: dict[str, Any], slot_id: str, patch: dict[str, Any]) -> None:
    slot = _find_manifest_slot(manifest, slot_id)
    if slot is None:
        raise HTTPException(status_code=404, detail=f"Challenge slot '{slot_id}' was not found.")
    slot.update(patch)
    manifest["updatedAt"] = _utc_label()
    _write_manifest(folder, manifest)


def _challenge_file_path(challenge_id: str, slot_id: str) -> Path:
    manifest = _read_manifest(challenge_id)
    slot = _find_manifest_slot(manifest, slot_id)
    if not isinstance(slot, dict) or not slot.get("filename"):
        raise HTTPException(status_code=404, detail=f"Challenge slot '{slot_id}' has no saved file.")
    folder = _challenge_dir(challenge_id).resolve()
    candidate = (folder / str(slot["filename"])).resolve()
    if folder not in candidate.parents and candidate != folder:
        raise HTTPException(status_code=400, detail="Invalid challenge file path.")
    if not candidate.exists():
        raise HTTPException(status_code=410, detail=f"Challenge file for slot '{slot_id}' is missing.")
    return candidate


def _challenge_asset_path(path: str) -> Path:
    root = _challenge_root().resolve()
    candidate = Path(path).expanduser().resolve()
    if root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Path is outside the HTML Challenge folder.")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Challenge file does not exist.")
    return candidate


def _open_default_app(path: Path) -> None:
    system_name = platform.system()
    if system_name == "Darwin":
        command = ["open", str(path)]
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    if system_name == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    command = ["xdg-open", str(path)]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


HTML_VALIDATION_LABELS = {
    "valid": "Valid",
    "partial": "Partial",
    "script-error": "Script error",
    "blank-render": "Blank render",
    "no-html": "No HTML",
}


def _html_validation_payload(
    status: str,
    issues: list[str] | None = None,
    *,
    checks: dict[str, Any] | None = None,
    source: str = "static",
) -> dict[str, Any]:
    return {
        "status": status,
        "label": HTML_VALIDATION_LABELS.get(status, status),
        "issues": [issue for issue in (issues or []) if issue],
        "checks": checks or {},
        "source": source,
        "updatedAt": _utc_label(),
    }


def _body_content(html: str) -> str:
    match = re.search(r"<body\b[^>]*>(.*?)</body>", html, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    start = re.search(r"<body\b[^>]*>", html, flags=re.IGNORECASE)
    if not start:
        return ""
    return html[start.end():]


def _visible_body_text(body: str) -> str:
    cleaned = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", body, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _looks_truncated(html: str) -> bool:
    stripped = html.rstrip()
    if re.search(r"<[^>\n]*$", stripped):
        return True
    tail = stripped[-80:].lower()
    return tail.endswith(("{", "(", "[", ",", ".", "=", "=>", "return", "function", "const", "let", "var"))


def _validate_html_document(
    source_text: str,
    html: str,
    final_chunk: Any = None,
) -> dict[str, Any]:
    raw = source_text.strip()
    candidate = html.strip()
    finish_reason = str(getattr(final_chunk, "finish_reason", "") or "").lower()
    lower = candidate.lower()
    has_doctype = "<!doctype" in lower
    has_html_tag = bool(re.search(r"<html\b", lower))
    has_body_open = bool(re.search(r"<body\b", lower))
    has_body_close = "</body>" in lower
    has_html_close = "</html>" in lower
    body = _body_content(candidate)
    body_text = _visible_body_text(body)
    has_visual_markup = bool(re.search(
        r"<(canvas|svg|main|section|article|div|p|h[1-6]|button|input|form|table|ul|ol|img|video)\b",
        body,
        flags=re.IGNORECASE,
    ))
    checks = {
        "hasDoctype": has_doctype,
        "hasHtmlTag": has_html_tag,
        "hasBodyOpen": has_body_open,
        "hasBodyClose": has_body_close,
        "hasHtmlClose": has_html_close,
        "bodyTextLength": len(body_text),
        "hasVisualMarkup": has_visual_markup,
        "finishReason": finish_reason or None,
    }

    if not raw or not candidate:
        return _html_validation_payload("no-html", ["No output was produced."], checks=checks)
    if not has_html_tag and not has_doctype:
        return _html_validation_payload("no-html", ["No <html> document was found."], checks=checks)
    if has_body_open and not body.strip():
        return _html_validation_payload("no-html", ["The <body> is empty."], checks=checks)

    issues: list[str] = []
    if not has_html_tag:
        issues.append("Missing <html> tag.")
    if not has_body_open:
        issues.append("Missing <body> tag.")
    if not has_body_close:
        issues.append("Missing </body> closing tag.")
    if not has_html_close:
        issues.append("Missing </html> closing tag.")
    if not body_text and not has_visual_markup:
        issues.append("The body has no visible content.")
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        issues.append("Generation stopped at the token limit.")
    if _looks_truncated(candidate):
        issues.append("Output appears to end mid-tag or mid-statement.")

    if issues:
        return _html_validation_payload("partial", issues, checks=checks)
    return _html_validation_payload("valid", [], checks=checks)


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


def _repair_prompt(original_prompt: str, partial_html: str, mode: str) -> str:
    action = "continue and complete" if mode == "continue" else "repair and complete"
    guidance = (
        f"The previous HTML Challenge output was incomplete or failed validation. "
        f"Please {action} it without changing the user's original intent. "
        "Return only one complete standalone HTML document with inline CSS and JavaScript. "
        "Do not include Markdown fences, explanations, or external network assets."
    )
    return (
        f"{guidance}\n\n"
        f"Original user prompt:\n{original_prompt.strip()}\n\n"
        f"Partial HTML file:\n```html\n{partial_html.strip()}\n```"
    )


def _html_system_prompt(extra: str | None, thinking_mode: str | None = None) -> str:
    base = (
        "You are participating in an HTML Challenge. Return only a complete, "
        "standalone HTML document for the user's prompt. Include all CSS and "
        "JavaScript inline in that single document. Do not use Markdown fences, "
        "do not explain the code, and do not reference external network assets."
    )
    if thinking_mode != "auto":
        base = (
            "Do not think step by step. Do not output hidden reasoning, planning, "
            "analysis notes, or <think> tags. "
            + base
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


def _slot_label(slot_id: str) -> str:
    return f"Model {slot_id.upper()}"


def _slot_manifest_payload(
    slot_id: str,
    model: CompareModelRequest,
    status: str = "queued",
    *,
    default_thinking_mode: Any = "off",
    default_reasoning_effort: Any = None,
) -> dict[str, Any]:
    return {
        "slotId": slot_id,
        "label": _slot_label(slot_id),
        "status": status,
        "modelRef": model.modelRef,
        "modelName": model.modelName or model.modelRef,
        **_model_display_payload(model),
        "canonicalRepo": model.canonicalRepo,
        "source": model.source,
        "backend": model.backend,
        "path": model.path,
        "settings": _settings_payload(model.launch),
        **_model_sampler_payload(model),
        **_model_thinking_payload(
            model,
            default_thinking_mode=default_thinking_mode,
            default_reasoning_effort=default_reasoning_effort,
        ),
    }


def _clear_slot_result_payload() -> dict[str, Any]:
    return {
        "filename": None,
        "filePath": None,
        "fileBytes": None,
        "validHtmlDocument": None,
        "htmlValidation": None,
        "responseSeconds": 0,
        "loadSeconds": 0,
        "totalSeconds": 0,
        "error": None,
        "metrics": None,
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


def _stream_html_challenge_slot(
    *,
    state: Any,
    manifest: dict[str, Any],
    folder: Path,
    slot_id: str,
    model: CompareModelRequest,
    prompt: str,
    system_prompt: str | None,
    thinking_mode: str | None,
    reasoning_effort: str | None,
    sampler_overrides: dict[str, Any] | None = None,
) -> Any:
    model_label = model.modelName or model.modelRef
    requested_runtime = _requested_runtime_payload(state, model.launch)
    _update_manifest_slot(
        folder,
        manifest,
        slot_id,
        {
            "status": "loading",
            "error": None,
            "filename": None,
            "filePath": None,
            "fileBytes": None,
            "validHtmlDocument": None,
            "htmlValidation": None,
        },
    )
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
        _update_manifest_slot(folder, manifest, slot_id, {"status": "running", "loadSeconds": load_seconds})
        yield _sse_event({
            "model": slot_id,
            "loaded": True,
            "loadSeconds": load_seconds,
            **_loaded_model_metrics(state),
            **requested_runtime,
        })
    except Exception as exc:
        _unload_active_model(state)
        state.runtime.clear_warm_pool()
        _update_manifest_slot(folder, manifest, slot_id, {"status": "error", "error": str(exc)})
        yield _sse_event({"model": slot_id, "error": str(exc), "challenge": manifest})
        return False

    full_text = ""
    final_chunk = None
    gen_start = time.perf_counter()
    try:
        for chunk in state.runtime.stream_generate(
            prompt=prompt,
            history=[],
            system_prompt=_html_system_prompt(system_prompt, thinking_mode),
            max_tokens=model.launch.maxTokens,
            temperature=model.launch.temperature,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort if thinking_mode == "auto" else None,
            samplers=sampler_overrides,
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
        _update_manifest_slot(
            folder,
            manifest,
            slot_id,
            {"status": "error", "error": str(exc), "loadSeconds": load_seconds},
        )
        yield _sse_event({"model": slot_id, "error": str(exc), "challenge": manifest})
    else:
        elapsed = round(time.perf_counter() - gen_start, 2)
        html, valid_html = _extract_html_document(full_text)
        html_validation = _validate_html_document(full_text, html, final_chunk)
        valid_html = html_validation["status"] == "valid"
        model_slug = _slugify(model_label, f"model-{slot_id}")
        filename = f"{slot_id}-{model_slug}.html"
        html_path = folder / filename
        html_path.write_text(html, encoding="utf-8")
        file_bytes = html_path.stat().st_size
        # Drop the previous slot file when a model swap means the new
        # filename differs (e.g. user changed model on a completed slot).
        # Without this the folder accumulates orphan HTML files keyed to
        # old model names while only the new file is referenced.
        previous_slot = _find_manifest_slot(manifest, slot_id) or {}
        previous_filename = str(previous_slot.get("filename") or "")
        if previous_filename and previous_filename != filename:
            previous_path = folder / previous_filename
            try:
                if previous_path.exists() and previous_path.resolve().parent == folder.resolve():
                    previous_path.unlink()
            except OSError:
                # Stale file is harmless — only swallow filesystem errors.
                pass
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
            "htmlValidation": html_validation,
            "metrics": metrics,
            "responseSeconds": elapsed,
            "loadSeconds": load_seconds,
            "totalSeconds": round(load_seconds + elapsed, 2),
            "error": None,
        }
        _update_manifest_slot(folder, manifest, slot_id, slot_patch)
        yield _sse_event({
            "model": slot_id,
            "done": True,
            "text": full_text,
            "html": html,
            "filename": filename,
            "filePath": str(html_path),
            "fileBytes": file_bytes,
            "validHtmlDocument": valid_html,
            "htmlValidation": html_validation,
            "loadSeconds": load_seconds,
            "totalSeconds": round(load_seconds + elapsed, 2),
            "challenge": manifest,
            **metrics,
        })
    finally:
        _unload_active_model(state)
        state.runtime.clear_warm_pool()

    return True
