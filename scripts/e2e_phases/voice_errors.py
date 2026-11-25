"""Phase 24: voice error codes (voice studio UX audit, change 6, D0141 to D0143).

No model runs and no worker starts. A failed voice request keeps its HTTP status and ``detail`` (the raw text) and
gains ``code``, ``message`` (one plain sentence), ``action`` (the one button) and ``extra``. The checks run in this
process against an app with the real voice routes and a stubbed worker, so they need no backend and no GPU:

* a worker failing with out of memory answers 503 with ``gpu_memory`` / ``smaller-model``;
* the setup failures (runtime not installed, add-on missing, model not downloaded, ffmpeg missing) each answer
  with their code and action, ``extra`` naming the add-on to tick;
* a bad language and a refused clone keep 400 and 403 and gain their codes;
* a failure inside the read-aloud stream carries the same fields on its error line;
* every code has a plain one-sentence message and an action the screen knows.

One more check asks the running backend (skipped when none answers): a voice request that fails before any worker
starts (an unknown voice) comes back with the fields, which shows the backend runs this version.

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)``.
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

PHASE_KEY = 24
ROOT = Path(__file__).resolve().parents[2]
FIELDS = {"detail", "code", "message", "action", "extra"}


def _client() -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import backend_service.app  # noqa: F401 - the route modules import each other; the app module loads them in order
    from backend_service.routes import register_routes

    app = FastAPI()
    register_routes(app)
    return TestClient(app)


def _expect(resp: Any, status: int, code: str, action: str, extra: str | None = None) -> tuple[str, str, dict[str, Any]]:
    body = resp.json()
    if resp.status_code != status:
        return "fail", f"HTTP {resp.status_code}, wanted {status}: {resp.text[:160]}", {}
    if not FIELDS <= set(body):
        return "fail", f"fields {sorted(body)} lack {sorted(FIELDS - set(body))}", {}
    if (body["code"], body["action"], body["extra"]) != (code, action, extra):
        return "fail", f"got {body['code']} / {body['action']} / {body['extra']}, wanted {code} / {action} / {extra}", {}
    if not isinstance(body["detail"], str) or not body["detail"]:
        return "fail", "detail is not the raw text", {}
    if not body["message"] or body["message"] == body["detail"]:
        return "fail", "message is missing or only repeats the raw text", {}
    return "pass", "", {}


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=PHASE_KEY, name="Voice error codes")
    try:
        client = _client()
    except Exception as exc:  # noqa: BLE001
        phase.status = "fail"
        phase.checks.append(suite.CheckResult("phase 24", "fail", reason=f"voice routes not importable: {exc}"))
        return phase

    from backend_service import voice_client, voice_errors, voice_runtime, voice_speak

    def transcribe(error: Exception):
        with mock.patch.object(voice_runtime, "transcribe_file", side_effect=error):
            return client.post("/api/voice/transcribe", files={"audio": ("a.wav", b"RIFF", "audio/wav")})

    def speak(error: Exception, **body: Any):
        with mock.patch.object(voice_speak, "speak", side_effect=error):
            return client.post("/api/voice/speak", json={"text": "Hello", "voice": "af_heart", **body})

    def _out_of_memory():
        return _expect(transcribe(voice_client.VoiceWorkerError("OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB")),
                       503, "gpu_memory", "smaller-model")

    def _setup_failures():
        wanted = [
            (RuntimeError("The voice runtime is not installed. Install it from Voice > Models."),
             "runtime_missing", "install-voice-runtime", None),
            (RuntimeError(voice_runtime.KOKORO_MISSING), "addon_missing", "install-addon", "kokoro"),
            (RuntimeError("Qwen3-TTS 1.7B is not downloaded. Download it in Voice > Models."),
             "model_missing", "download-model", None),
            (RuntimeError("Changing speed or pitch for this voice needs ffmpeg. Install it from Settings."),
             "ffmpeg_missing", "install-ffmpeg", None),
            (voice_client.VoiceWorkerError("The voice worker exited. See worker.log."), "worker_stopped", "retry", None),
        ]
        for error, code, action, extra in wanted:
            status, reason, detail = _expect(speak(error), 503, code, action, extra)
            if status != "pass":
                return status, f"{code}: {reason}", detail
        return "pass", "", {}

    def _client_mistakes_keep_their_status():
        status, reason, detail = _expect(
            speak(ValueError("Chatterbox speaks en, de; 'xx' is not one of them."), language="xx"),
            400, "language_unsupported", "pick-another-language")
        if status != "pass":
            return status, f"language: {reason}", detail
        from backend_service import voice_clone_safety

        with mock.patch.object(voice_clone_safety, "clone_allowed", return_value=False):
            refused = client.post("/api/voice/speak", json={"text": "Hello", "voice": "clone:abc"})
        status, reason, detail = _expect(refused, 403, "cloning_not_allowed", "none")
        return (status, f"refused clone: {reason}", detail) if status != "pass" else ("pass", "", {})

    def _stream_error_line():
        with mock.patch.object(voice_runtime, "synthesize", side_effect=RuntimeError("OutOfMemoryError: CUDA out of memory.")):
            resp = client.post("/api/voice/speak/stream", json={"text": "One. Two."})
        line = json.loads(resp.text.splitlines()[0])
        if (line.get("code"), line.get("action")) != ("gpu_memory", "smaller-model") or "out of memory" not in line.get("error", ""):
            return "fail", f"error line {line}", {}
        return "pass", "", {}

    def _every_code_is_plain():
        bad = [code for code, (message, action) in voice_errors._TABLE.items()
               if action not in voice_errors.ACTIONS or not message.endswith(".") or ". " in message]
        return ("fail", f"codes without a plain sentence or a known action: {bad}", {}) if bad else ("pass", "", {})

    def _running_backend():
        if not cap.backend_reachable:
            return "skip", "backend not reachable", {}
        body = json.dumps({"text": "Hello", "voice": "no_such_voice_zz"}).encode("utf-8")
        code, data = suite._voice_http("POST", "/api/voice/speak", body=body, content_type="application/json")
        try:
            got = json.loads(data.decode("utf-8"))
        except ValueError:
            return "fail", f"HTTP {code}, not JSON", {}
        if code < 400 or code >= 500 or not {"code", "message", "action"} <= set(got):
            return "fail", f"HTTP {code}: {str(got)[:160]}", {}
        return "pass", "", {"code": got["code"]}

    # Settings in a throwaway folder and the Kokoro voices counted as installed: the stream needs a speakable voice.
    from backend_service import voice_env

    stack = contextlib.ExitStack()
    data = stack.enter_context(tempfile.TemporaryDirectory())
    stack.enter_context(mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(data)))
    stack.enter_context(mock.patch.object(voice_env, "kokoro_installed", return_value=True))
    for name, fn in [
        ("worker out of memory: 503 with gpu_memory and the smaller-model action (in process)", _out_of_memory),
        ("setup failures: runtime, add-on (extra), model, ffmpeg and a dead worker each get a code and action (in process)", _setup_failures),
        ("a bad language stays 400 and a refused clone stays 403, each with its code (in process)", _client_mistakes_keep_their_status),
        ("a failure inside the read-aloud stream carries the same fields (in process)", _stream_error_line),
        ("every code has one plain sentence and an action the screen knows", _every_code_is_plain),
        ("the running backend answers a failed voice request with the fields", _running_backend),
    ]:
        phase.checks.append(suite._check(name, fn))
    stack.close()
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
