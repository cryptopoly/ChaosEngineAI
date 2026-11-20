"""E2E checks for features built on ``test/rtx4090`` that had unit tests but no E2E check (coverage audit 2026-11-20,
``docs/review/test-coverage-2026-11-20.md``).

Each function is one check for the suite's ``_check``: it takes the suite module ``h`` and returns
``(status, reason, detail)``. They are registered in the phase their feature belongs to (``scripts/e2e_test_suite.py``).
Nothing here downloads or installs: a check that would need weights, a GPU or another platform skips with the reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _call(h: Any, method: str, path: str, payload: Any = None, timeout: float = 60.0) -> tuple[int, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    code, data = h._voice_http(method, path, body=body, content_type="application/json" if body else None, timeout=timeout)
    try:
        return code, json.loads(data.decode("utf-8")) if data.strip() else None
    except (ValueError, UnicodeDecodeError):
        return code, data.decode("utf-8", "replace")


def _detail(body: Any) -> str:
    detail = body.get("detail") if isinstance(body, dict) else body
    if isinstance(detail, dict):
        detail = detail.get("message") or detail
    return str(detail)


def _repo_root_on_path() -> None:
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)


# ---------------------------------------------------------------------------
# Phase 0: chat catalogue follow-ups (FU-263 to FU-269)
# ---------------------------------------------------------------------------

CHAT_ROWS = {
    # repo: (ggufFile, what the row must also carry)
    "ggml-org/Qwen3.6-35B-A3B-GGUF": "Qwen3.6-35B-A3B-Q4_K_M.gguf",  # FU-263: re-pinned to Q4_K_M for a 24 GB card
    "prism-ml/Ternary-Bonsai-2-27B-gguf": "Ternary-Bonsai-2-27B-Q2_0.gguf",  # FU-269: Q2_0 on stock llama.cpp
}
LIGHTNING_PREFIX = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B"


def chat_followup_rows(h: Any) -> tuple[str, str, dict[str, Any]]:
    """The 4090 chat rows are listed with their pinned files, and every Nemotron 3.5 Lightning GGUF row carries the
    16,384-token minimum context its Studio tool list needs (FU-265). The load-time raise itself is
    ``tests/test_backend_service.py::test_a_row_with_a_min_context_loads_larger_and_says_so``."""
    code, payload = _call(h, "GET", "/api/workspace", timeout=30.0)
    if code != 200 or not isinstance(payload, dict):
        return "fail", f"/api/workspace HTTP {code}", {}
    rows = [v for f in payload.get("featuredModels") or [] for v in f.get("variants") or []]
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_repo.setdefault(str(row.get("repo")), []).append(row)
    problems = []
    for repo, gguf in CHAT_ROWS.items():
        files = [r.get("ggufFile") for r in by_repo.get(repo, [])]
        if gguf not in files:
            problems.append(f"{repo}: no row pinning {gguf} (found {files})")
    lightning = [r for r in rows if str(r.get("ggufFile") or "").startswith(LIGHTNING_PREFIX)]
    if not lightning:
        problems.append("no Nemotron 3.5 Lightning GGUF rows")
    short = [r.get("ggufFile") for r in lightning if int(r.get("minContextTokens") or 0) < 16384]
    if short:
        problems.append(f"Lightning rows without a 16,384 minimum context: {short}")
    if problems:
        return "fail", "; ".join(problems)[:300], {"problems": problems}
    return "pass", "", {"lightningRows": len(lightning)}


# ---------------------------------------------------------------------------
# Phase 1: /v1/chat/completions streams tool calls (FU-258)
# ---------------------------------------------------------------------------

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def _sse_choices(raw: bytes) -> list[dict[str, Any]] | str:
    lines = [line[6:] for line in raw.decode("utf-8", "replace").splitlines() if line.startswith("data: ")]
    if not lines or lines[-1].strip() != "[DONE]":
        return f"the stream did not end with [DONE] ({len(lines)} events)"
    try:
        return [json.loads(line)["choices"][0] for line in lines[:-1] if json.loads(line).get("choices")]
    except (ValueError, KeyError, IndexError) as exc:
        return f"unreadable stream event: {exc}"


def stream_tool_calls(h: Any, cap: Any) -> tuple[str, str, dict[str, Any]]:
    """The smallest local GGUF loads on llama.cpp, then a streamed ``/v1/chat/completions`` with one tool must send the
    call as a ``tool_calls`` delta and finish with ``tool_calls`` (it sent the call as nothing before FU-258). A model
    that answers in words instead of calling the tool is a skip: that is the model's choice, not the route's."""
    if not cap.gguf_available:
        return "skip", "llama-server not available", {}
    if not cap.local_gguf_files:
        return "skip", "no GGUF on disk", {}
    ref, path = cap.local_gguf_files[0]
    rc, loaded, err = h._cli_json("load", ref, "--path", path, "--backend", "gguf", "--cache-strategy", "native",
                                  "--context", "4096", "--timeout", "900", timeout=960.0)
    if rc != 0 or not isinstance(loaded, dict) or loaded.get("state") != "loaded":
        h._cli("unload", timeout=60.0)
        return "fail", f"load {ref}: rc={rc} {(err or str(loaded))[:200]}", {}
    try:
        body = {"model": ref, "stream": True, "max_tokens": 256, "temperature": 0.0, "tools": [WEATHER_TOOL],
                "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}]}
        code, raw = h._voice_http("POST", "/v1/chat/completions", body=json.dumps(body).encode("utf-8"),
                                  content_type="application/json", timeout=300.0)
        if code != 200:
            return "fail", f"/v1/chat/completions stream HTTP {code}: {raw[:200]!r}", {}
        choices = _sse_choices(raw)
        if isinstance(choices, str):
            return "fail", choices, {}
        calls = [c for choice in choices for c in (choice.get("delta") or {}).get("tool_calls") or []]
        finish = next((c.get("finish_reason") for c in reversed(choices) if c.get("finish_reason")), None)
        detail = {"model": ref, "events": len(choices), "finish": finish, "calls": [c.get("function", {}).get("name") for c in calls]}
        if not calls:
            text = "".join(str((c.get("delta") or {}).get("content") or "") for c in choices)
            if finish == "tool_calls":
                return "fail", "finish_reason tool_calls but no tool_calls delta was sent", detail
            return "skip", f"{ref} answered in words instead of calling the tool: {text[:80]!r}", detail
        if finish != "tool_calls":
            return "fail", f"tool_calls sent but finish_reason is {finish!r}", detail
        if any((c.get("function") or {}).get("name") != "get_weather" for c in calls if (c.get("function") or {}).get("name")):
            return "fail", f"unexpected tool name in {detail['calls']}", detail
        return "pass", "", detail
    finally:
        h._cli("unload", timeout=60.0)


# ---------------------------------------------------------------------------
# Phase 4: a Nunchaku row that is not on disk refuses plainly (no placeholder picture)
# ---------------------------------------------------------------------------


def nunchaku_refusal(h: Any) -> tuple[str, str, dict[str, Any]]:
    """A Nunchaku INT4 FLUX row whose pack or base parts are missing answers 409 with what to download, instead of
    HTTP 200 with a placeholder picture and a raw Hub error (RTX 4090, 2026-11-19). Skips when the row is ready."""
    code, catalog = _call(h, "GET", "/api/images/catalog")
    if code != 200 or not isinstance(catalog, dict):
        return "fail", f"/api/images/catalog HTTP {code}", {}
    rows = [v for f in catalog.get("families") or [] for v in f.get("variants") or [] if v.get("nunchakuRepo")]
    if not rows:
        return "fail", "no Nunchaku row in the image catalog", {}
    missing = [r for r in rows if not r.get("availableLocally")]
    if not missing:
        return "skip", "every Nunchaku row is downloaded here; its live run is the generate check", {}
    row = missing[0]
    code, body = _call(h, "POST", "/api/images/generate", {
        "modelId": row["id"], "prompt": "e2e nunchaku refusal", "width": 256, "height": 256, "steps": 1, "seed": 1,
    }, timeout=120.0)
    detail = _detail(body)
    if code == 403:
        return "skip", f"refused by another gate first: {detail[:120]}", {}
    if code != 409:
        return "fail", f"{row['id']}: HTTP {code} (wanted 409): {detail[:200]}", {}
    if "download" not in detail.lower():
        return "fail", f"409 without a plain fix: {detail[:200]}", {}
    return "pass", "", {"row": row["id"], "detail": detail[:160]}


# ---------------------------------------------------------------------------
# Phase 6: vLLM and TriAttention refuse at once on native Windows
# ---------------------------------------------------------------------------


def windows_install_refusal(h: Any) -> tuple[str, str, dict[str, Any]]:
    """On native Windows the install route answers 400 at once for ``vllm`` and ``triattention`` (no Windows builds)
    and names the way that works. Skipped elsewhere: there the same call would really install the package."""
    if sys.platform != "win32":
        return "skip", "native Windows only (elsewhere the call would install the package)", {}
    seen = {}
    for package, needle in (("vllm", "WSL"), ("triattention", "triton")):
        code, body = _call(h, "POST", "/api/setup/install-package", {"package": package}, timeout=30.0)
        detail = _detail(body)
        if code != 400 or needle not in detail:
            return "fail", f"{package}: HTTP {code} {detail[:200]}", {}
        seen[package] = detail[:80]
    return "pass", "", seen


# ---------------------------------------------------------------------------
# Phase 7: Studio staging (FU-231 part 2), in process
# ---------------------------------------------------------------------------


def studio_staging(h: Any) -> tuple[str, str, dict[str, Any]]:
    """A Chat turn carries the Storyboard tools only when it is about a film (FU-231): a plain question gets the core
    Studio tools, a film word in the message or the recent thread, an earlier ``storyboard_*`` call or a Keep going
    run brings them back. In process (the tool list a turn gets is not visible over HTTP)."""
    _repo_root_on_path()
    from backend_service.tools.studio import STUDIO_TOOL_NAMES, wants_storyboard

    storyboard = [n for n in STUDIO_TOOL_NAMES if n.startswith("storyboard_")]
    if not storyboard:
        return "fail", "no storyboard_* tools in the Studio list", {}
    cases = {
        "plain question": (wants_storyboard("What is 12 times 7?", []), False),
        "film in the message": (wants_storyboard("Make a 30 second film about a lighthouse", []), True),
        "film earlier in the thread": (wants_storyboard("Make it darker", [{"text": "Plan the storyboard for my trailer"}]), True),
        "earlier storyboard call": (wants_storyboard("ok", [{"text": "", "toolCalls": [{"name": "storyboard_shot"}]}]), True),
        "Keep going run": (wants_storyboard("next", [], keep_going=True), True),
    }
    wrong = [name for name, (got, want) in cases.items() if got != want]
    if wrong:
        return "fail", f"wrong answer for: {wrong}", {}
    return "pass", "", {"storyboardTools": len(storyboard)}


def studio_price(h: Any) -> tuple[str, str, dict[str, Any]]:
    """``POST /api/chat/studio/price`` (the plan card's Expert fields, FU-240/241) prices a single render as typed: a
    tool other than generate_image / generate_video is refused (422), the answer carries the fields the card reads,
    and with an installed image model a larger size costs more time than a smaller one."""
    code, _ = _call(h, "POST", "/api/chat/studio/price", {"tool": "make_music", "arguments": {}})
    if code == 404:
        return "skip", "price route not present in this build", {}
    if code != 422:
        return "fail", f"make_music was not refused (HTTP {code})", {}
    code, plain = _call(h, "POST", "/api/chat/studio/price", {"tool": "generate_image", "arguments": {"prompt": "x"}})
    keys = {"model", "size", "frames", "fps", "steps", "estimatedSeconds", "peakGb"}
    if code != 200 or not isinstance(plain, dict) or keys - set(plain):
        return "fail", f"price answer HTTP {code}: {str(plain)[:200]}", {}
    code, tiers = _call(h, "GET", "/api/chat/studio/tiers", timeout=30.0)
    images = [m for m in (tiers or {}).get("models", []) if m.get("kind") == "image"] if code == 200 else []
    if not images:
        return "pass", "route shape only: no image model installed to price", {}
    model = images[0].get("id")

    def seconds(size: int) -> Any:
        _, body = _call(h, "POST", "/api/chat/studio/price", {"tool": "generate_image", "arguments": {
            "prompt": "x", "model": model, "width": size, "height": size}})
        return (body or {}).get("estimatedSeconds"), (body or {}).get("size")

    (small, small_size), (big, big_size) = seconds(512), seconds(1024)
    detail = {"model": model, "512": small, "1024": big}
    if small_size != "512x512" or big_size != "1024x1024":
        return "fail", f"the price did not keep the typed size ({small_size}, {big_size})", detail
    if small is None or big is None:
        return "pass", "this model has no time estimate; size kept as typed", detail
    if big <= small:
        return "fail", f"1024x1024 priced at {big} s, not more than 512x512 at {small} s", detail
    return "pass", "", detail


# ---------------------------------------------------------------------------
# Phase 8: Voice on a PC (D-101, D-102)
# ---------------------------------------------------------------------------


def aligner_row(h: Any) -> tuple[str, str, dict[str, Any]]:
    """``/api/voice/runtime`` carries the forced aligner as the "Word timings" download exactly where the GPU
    dictation rows are offered (D-101), and nowhere else."""
    code, runtime = _call(h, "GET", "/api/voice/runtime", timeout=30.0)
    if code != 200 or not isinstance(runtime, dict):
        return "fail", f"/api/voice/runtime HTTP {code}", {}
    runtime = runtime.get("runtime", runtime)
    if "aligner" not in runtime:
        return "fail", "the runtime has no aligner field", {}
    gpu_rows = [m for m in runtime.get("sttModels") or [] if m.get("backend") == "transformers-asr"]
    aligner = runtime["aligner"]
    if not gpu_rows:
        if aligner is not None:
            return "fail", "an aligner row without GPU dictation rows", {"aligner": aligner}
        return "pass", "no GPU dictation rows here, so no Word timings row", {"lane": runtime.get("lane")}
    if not isinstance(aligner, dict):
        return "fail", "GPU dictation rows but no Word timings row", {}
    wanted = {"key": "qwen3-forced-aligner-0.6b", "id": "Qwen/Qwen3-ForcedAligner-0.6B-hf"}
    wrong = {k: aligner.get(k) for k, v in wanted.items() if aligner.get(k) != v}
    if wrong or not aligner.get("licence") or not float(aligner.get("sizeGb") or 0) or not isinstance(aligner.get("installed"), bool):
        return "fail", f"aligner row fields: {aligner}", {}
    return "pass", "", {"installed": aligner["installed"], "gpuRows": len(gpu_rows)}


def speech_rows_carry_lanes(h: Any) -> tuple[str, str, dict[str, Any]]:
    """Every speech model row says which lanes run it, so a PC can fold the Mac-only rows behind one line (D-102 A);
    on a non-MLX lane at least one listed row is not MLX-only."""
    code, runtime = _call(h, "GET", "/api/voice/runtime", timeout=30.0)
    lane = (runtime or {}).get("runtime", runtime or {}).get("lane") if isinstance(runtime, dict) else None
    code, payload = _call(h, "GET", "/api/voice/clone/models", timeout=30.0)
    if code == 404:
        return "skip", "speech model list not present in this build", {}
    if code != 200 or not isinstance(payload, dict):
        return "fail", f"/api/voice/clone/models HTTP {code}", {}
    rows = payload.get("models") or []
    if not rows:
        return "fail", "no speech model rows", {}
    without = [r.get("key") for r in rows if not isinstance(r.get("lanes"), list) or not r.get("lanes")]
    if without:
        return "fail", f"rows without lanes: {without}", {}
    mac_only = [r.get("key") for r in rows if r.get("lanes") == ["mlx"]]
    detail = {"lane": lane, "rows": len(rows), "macOnly": len(mac_only)}
    if lane and lane != "mlx" and len(mac_only) == len(rows):
        return "fail", f"on the {lane} lane every row is Mac-only", detail
    return "pass", "", detail


# ---------------------------------------------------------------------------
# Phase 5: MiniMax H3 licence terms (D-99, FU-340)
# ---------------------------------------------------------------------------


def h3_notice_terms(h: Any) -> tuple[str, str, dict[str, Any]]:
    """The H3 gate model carries the live licence link, a tick sentence that accepts the licence and its Acceptable
    Use Policy, and the hosted-service notice (D-99). H3 stays unreleased (no row); the machine-generated tag on its
    outputs is ``tests/test_licence_gate.py``."""
    code, status = _call(h, "GET", "/api/licence-gate")
    if code == 404:
        return "skip", "licence gate not present in this build", {}
    if code != 200 or not isinstance(status, dict):
        return "fail", f"/api/licence-gate HTTP {code}", {}
    model = next((m for m in status.get("models") or [] if m.get("key") == "minimax-h3"), None)
    note = ""
    if model is None:
        # Unreleased (H3_RELEASED False): the route lists nothing, so read the terms it will send, in process.
        _repo_root_on_path()
        from backend_service import licence_gate

        if licence_gate.is_released("minimax-h3"):
            return "fail", "H3 is released but the gate route does not list it", {}
        model = dict(licence_gate.TERRITORY_MODELS["minimax-h3"])
        note = "H3 unreleased: the route lists nothing; checked the terms it will send (in process)"
    problems = []
    if not str(model.get("liveLicenceUrl") or "").startswith("https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/"):
        problems.append(f"liveLicenceUrl {model.get('liveLicenceUrl')!r}")
    if "Acceptable Use Policy" not in str(model.get("confirmSentence")):
        problems.append("the tick sentence does not accept the Acceptable Use Policy")
    if "hosted service" not in str(model.get("note")):
        problems.append("the note has no hosted-service notice")
    if problems:
        return "fail", "; ".join(problems), {}
    return "pass", note, {"confirmed": model.get("confirmed")}


# ---------------------------------------------------------------------------
# Phase 11: Copy now for music weights (D-100, FU-276)
# ---------------------------------------------------------------------------


def weight_copy_status(h: Any) -> tuple[str, str, dict[str, Any]]:
    """The weight-copy job routes answer with the fields the Copy now panel reads; Cancel with no copy running leaves
    it not running. The copy itself is never started here (it would copy gigabytes): tests/test_music_weight_copy.py."""
    code, status = _call(h, "GET", "/api/music/weights/copy/status")
    if code == 404:
        return "skip", "weight copy not present in this build", {}
    if code != 200 or not isinstance(status, dict):
        return "fail", f"status HTTP {code}", {}
    keys = {"phase", "running", "bytesDone", "totalBytes", "fileCount", "filesDone", "currentFile", "error", "etaSeconds"}
    if keys - set(status):
        return "fail", f"status lacks {sorted(keys - set(status))}", {}
    if status["running"]:
        return "skip", "a weight copy is running on this machine; Cancel not tried", {"phase": status["phase"]}
    code, after = _call(h, "POST", "/api/music/weights/copy/cancel", {})
    if code != 200 or not isinstance(after, dict) or after.get("running"):
        return "fail", f"cancel with nothing running: HTTP {code} {str(after)[:160]}", {}
    return "pass", "", {"phase": status["phase"]}
