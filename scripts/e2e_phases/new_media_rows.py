"""E2E checks for the image and video rows added on 2026-11-19 (FU-259): FLUX 3 Image Klein on diffusers (CUDA) and
on stable-diffusion.cpp, Qwen-Image-2.1 Turbo on stable-diffusion.cpp, and Kandinsky 6.0 Lite (video with sound).

Each check asserts the row is listed and routed to its engine, then, when the row's weights are on this machine,
renders one small picture or clip through the app's own route. A row whose weights are absent, or whose engine is
missing here (no NVIDIA GPU, no ``sd`` binary), skips with the reason. ``h`` is the suite module; each function
returns ``(status, reason, detail)`` for the suite's ``_check``.
"""

from __future__ import annotations

import json
from typing import Any

IMAGE_ROWS = {
    # id: (engine, expected runtime marker in the response, needs NVIDIA)
    "black-forest-labs/flux-3-image-klein": (None, "Diffusers", True),
    "leejet/flux-3-image-klein-GGUF-q8": ("sdcpp", "stable-diffusion.cpp", False),
    "unsloth/Qwen-Image-2.1-Turbo-GGUF-q4km": ("sdcpp", "stable-diffusion.cpp", False),
}
VIDEO_ROW = "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers"


def _call(h: Any, method: str, path: str, payload: Any = None, timeout: float = 60.0) -> tuple[int, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    code, data = h._voice_http(method, path, body=body, content_type="application/json" if body else None, timeout=timeout)
    try:
        return code, json.loads(data.decode("utf-8")) if data.strip() else None
    except ValueError:
        return code, data.decode("utf-8", "replace")


def _rows(h: Any, route: str) -> dict[str, dict[str, Any]] | str:
    code, catalog = _call(h, "GET", route)
    if code != 200 or not isinstance(catalog, dict):
        return f"{route}: HTTP {code}"
    return {v.get("id"): v for f in catalog.get("families") or [] for v in f.get("variants") or []}


def _nvidia(h: Any) -> bool:
    code, check = _call(h, "GET", "/api/models/memory-check?weights_gb=1&context_tokens=1024")
    return code == 200 and isinstance(check, dict) and check.get("memoryKind") == "gpu"


def image_rows(h: Any) -> tuple[str, str, dict[str, Any]]:
    rows = _rows(h, "/api/images/catalog")
    if isinstance(rows, str):
        return "fail", rows, {}
    missing = [rid for rid in IMAGE_ROWS if rid not in rows]
    if missing:
        return "fail", f"rows not listed: {missing}", {}
    for rid, (engine, _marker, _cuda) in IMAGE_ROWS.items():
        row = rows[rid]
        if row.get("engine") != engine:
            return "fail", f"{rid} routed to {row.get('engine')!r}, expected {engine!r}", {}
        if not isinstance(row.get("estimateRef"), dict) or not row.get("estimatedGenerationSeconds"):
            return "fail", f"{rid} has no time estimate", {}
    rendered, skipped = [], []
    nvidia = _nvidia(h)
    for rid, (_engine, marker, needs_cuda) in IMAGE_ROWS.items():
        if not rows[rid].get("availableLocally"):
            skipped.append(f"{rid}: weights not downloaded")
            continue
        if needs_cuda and not nvidia:
            skipped.append(f"{rid}: needs an NVIDIA GPU")
            continue
        code, body = _call(h, "POST", "/api/images/generate",
                           {"modelId": rid, "prompt": "a red fox in the snow, golden hour", "width": 512, "height": 512,
                            "seed": 42}, timeout=1800.0)
        text = json.dumps(body) if not isinstance(body, str) else body
        if code != 200:
            if h._is_memory_refusal(text):
                skipped.append(f"{rid}: host memory gate fired")
                continue
            if "sd.cpp" in text and ("binary" in text or "not staged" in text or "unavailable" in text):
                skipped.append(f"{rid}: sd.cpp engine unavailable")
                continue
            return "fail", f"{rid}: HTTP {code} {text[:240]}", {}
        artifact = (body.get("artifacts") or [{}])[0]
        if marker not in str(artifact.get("runtimeLabel") or ""):
            return "fail", f"{rid} did not run on {marker}: {artifact.get('runtimeLabel')!r}", {}
        rendered.append(rid)
        _call(h, "POST", "/api/images/unload", {})
    if not rendered:
        return "skip", "; ".join(skipped), {}
    return "pass", "", {"rendered": rendered, "skipped": skipped}


def video_row(h: Any) -> tuple[str, str, dict[str, Any]]:
    rows = _rows(h, "/api/video/catalog")
    if isinstance(rows, str):
        return "fail", rows, {}
    row = rows.get(VIDEO_ROW)
    if row is None:
        return "fail", f"{VIDEO_ROW} not listed", {}
    if row.get("engine") or not row.get("cudaOnly"):
        return "fail", f"{VIDEO_ROW} should run on diffusers, NVIDIA only: engine={row.get('engine')!r}", {}
    if not row.get("availableLocally"):
        return "skip", "Kandinsky 6.0 Lite weights not downloaded (27 GB)", {}
    if not _nvidia(h):
        return "skip", "needs an NVIDIA GPU", {}
    code, body = _call(h, "POST", "/api/video/generate",
                       {"modelId": VIDEO_ROW, "prompt": "waves on a rocky coast at dusk. Audio: surf and wind.",
                        "width": 512, "height": 320, "numFrames": 49, "seed": 7}, timeout=2400.0)
    text = json.dumps(body) if not isinstance(body, str) else body
    if code != 200:
        if h._is_memory_refusal(text):
            return "skip", "host memory gate fired: " + text[:160], {}
        return "fail", f"HTTP {code} {text[:240]}", {}
    artifact = body.get("artifact") or {}
    if not artifact.get("videoPath"):
        return "fail", "no videoPath in the response", {}
    if artifact.get("hasAudio") is not True:
        return "fail", f"the clip should carry sound: {artifact.get('runtimeNote')}", {}
    _call(h, "POST", "/api/video/unload", {})
    return "pass", "", {"seconds": artifact.get("durationSeconds")}
