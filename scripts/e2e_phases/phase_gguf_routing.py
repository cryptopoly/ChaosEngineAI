"""E2E phase 213: GGUF or MLX is read from the model's files, never from "gguf" in its name.

Checks, against a running backend, with throwaway symlinks in a temporary folder (the real model
folders are only linked, never copied, moved or written):
1. an MLX folder linked as ``my-gguf-notes-mlx`` loads on an MLX engine, not llama.cpp, and answers a prompt;
2. the same link with an explicit ``mlx`` backend stays on MLX;
3. a GGUF file linked inside a folder called ``mlx`` loads on llama.cpp and answers a prompt;
4. a folder with only ``config.json`` and a name saying "gguf" is refused with a plain message that names
   the file it found, before any engine starts.
Uses the smallest local MLX and GGUF text models from the library; a check skips when none is on disk.
The model is unloaded and the temporary folder removed at the end.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _snapshot(path: Path) -> Path:
    """The folder that holds the files: an HF cache repo's current snapshot, else the path itself."""
    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return path
    ref = path / "refs" / "main"
    if ref.is_file():
        candidate = snapshots / ref.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return candidate
    folders = sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return folders[0] if folders else path


def _model_gguf(path: Path) -> Path | None:
    if path.is_file():
        return path if path.suffix.lower() == ".gguf" else None
    files = [p for p in _snapshot(path).rglob("*.gguf") if "mmproj" not in p.name.lower()]
    return max(files, key=lambda p: p.stat().st_size) if files else None


def phase_gguf_routing(cap: Any):
    import e2e_test_suite as suite  # scripts/ is on sys.path when the suite runs

    phase = suite.PhaseResult(phase=213, name="GGUF routing from files, not names")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 213", "skip", reason="backend not reachable"))
        return phase
    http = suite._voice_http
    state: dict[str, Any] = {}
    workdir = Path(tempfile.mkdtemp(prefix="e2e-gguf-routing-"))

    def _json(method: str, path: str, payload: dict | None = None, timeout: float = 120.0):
        body = json.dumps(payload).encode() if payload is not None else None
        code, raw = http(method, path, body=body, content_type="application/json" if body else None, timeout=timeout)
        try:
            return code, json.loads(raw.decode() or "null")
        except ValueError:
            return code, raw

    def _detail(body: Any) -> str:
        detail = body.get("detail") if isinstance(body, dict) else body
        if isinstance(detail, dict):
            detail = detail.get("message")
        return str(detail)

    def _smallest(fmt: str) -> dict[str, Any] | None:
        if "library" not in state:
            code, body = _json("GET", "/api/workspace", timeout=60.0)
            library = (body or {}).get("library") if code == 200 and isinstance(body, dict) else []
            state["library"] = [i for i in library or [] if i.get("modelType") == "text" and not i.get("broken")]
        rows = sorted((i for i in state["library"] if i.get("format") == fmt), key=lambda i: float(i.get("sizeGb") or 0))
        return rows[0] if rows else None

    def _load_and_chat(name: str, path: Path, backend: str, chat: bool = True):
        payload = {"modelRef": name, "modelName": name, "source": "library", "backend": backend,
                   "path": str(path), "contextTokens": 2048}
        code, body = _json("POST", "/api/models/load", payload, timeout=600.0)
        if code != 200:
            return "fail", f"load HTTP {code}: {_detail(body)[:200]}", {}
        runtime = (body or {}).get("runtime") or {}
        engine = runtime.get("engine")
        detail = {"engine": engine, "path": str(path)}
        if not chat:
            return "pass", "", detail
        code, gen = _json("POST", "/api/chat/generate", {
            "prompt": "Name the capital of France in one word.", "maxTokens": 24, "temperature": 0.0,
            "modelRef": name, "modelName": name, "source": "library", "backend": backend, "path": str(path),
            "thinkingMode": "off",
        }, timeout=300.0)
        if code != 200 or not isinstance(gen, dict):
            return "fail", f"generate HTTP {code}: {_detail(gen)[:200]}", detail
        try:
            text = str(gen["session"]["messages"][-1]["text"])
        except (KeyError, IndexError, TypeError):
            return "fail", "no reply in the session", detail
        detail["reply"] = text[:80]
        if not text.strip():
            return "fail", "empty reply", detail
        return "pass", "", detail

    def _unload() -> None:
        _json("POST", "/api/models/unload", {})

    def _mlx_named_gguf(backend: str, chat: bool):
        def check():
            item = _smallest("MLX")
            if item is None:
                return "skip", "no local MLX text model", {}
            link = workdir / "my-gguf-notes-mlx"
            if not link.exists():
                os.symlink(_snapshot(Path(item["path"])), link, target_is_directory=True)
            try:
                status, reason, detail = _load_and_chat("my-gguf-notes-mlx", link, backend, chat)
            finally:
                _unload()
            detail["model"] = item["name"]
            if status == "pass" and detail.get("engine") == "llama.cpp":
                return "fail", "an MLX folder named gguf loaded on llama.cpp", detail
            return status, reason, detail
        return check

    def _gguf_in_mlx_folder():
        item = _smallest("GGUF")
        if item is None:
            return "skip", "no local GGUF text model", {}
        source = _model_gguf(Path(item["path"]))
        if source is None:
            return "skip", f"no .gguf file under {item['path']}", {}
        folder = workdir / "mlx"
        folder.mkdir(exist_ok=True)
        link = folder / source.name
        if not link.exists():
            os.symlink(source, link)
        try:
            status, reason, detail = _load_and_chat(source.stem, link, "auto")
        finally:
            _unload()
        detail["model"] = item["name"]
        if status == "pass" and detail.get("engine") != "llama.cpp":
            return "fail", f"a .gguf file loaded on {detail.get('engine')}", detail
        return status, reason, detail

    def _no_weights():
        folder = workdir / "half-downloaded-gguf"
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        code, body = _json("POST", "/api/models/load", {
            "modelRef": "half-downloaded-gguf", "source": "library", "backend": "auto", "path": str(folder),
            "contextTokens": 2048,
        }, timeout=120.0)
        message = _detail(body)
        if code == 200:
            _unload()
            return "fail", "a folder without weights loaded", {}
        if "Traceback" in message or "No model weights found" not in message or "config.json" not in message:
            return "fail", f"HTTP {code}: {message[:200]}", {}
        return "pass", "", {"message": message[:160]}

    try:
        for name, fn in (
            ("MLX folder named gguf, auto backend: loads on MLX and answers", _mlx_named_gguf("auto", True)),
            ("MLX folder named gguf, explicit mlx backend: stays on MLX", _mlx_named_gguf("mlx", False)),
            ("GGUF file in a folder named mlx: loads on llama.cpp and answers", _gguf_in_mlx_folder),
            ("Folder without weights: plain error naming what it holds", _no_weights),
        ):
            phase.checks.append(suite._check(name, fn))
    finally:
        for entry in sorted(workdir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            elif entry.is_dir():
                entry.rmdir()
        workdir.rmdir()
    statuses = {c.status for c in phase.checks}
    phase.status = "fail" if "fail" in statuses else ("pass" if "pass" in statuses else "skip")
    return phase
