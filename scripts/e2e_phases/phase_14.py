"""E2E phase 14: voice cloning (D-28, D-45).

Checks, against a running backend:
1. the cloning model rows are listed with pinned 40-character revisions;
2. the Agent access token cannot reach the cloning routes;
3. a consent recording with the wrong code is refused (needs Kokoro + STT);
4. full run when a cloning model is downloaded: Kokoro reads the consent sentence, the voice is
   saved, one sentence is spoken, the WAV and the MP3 carry the AI tag, then the voice, the speech
   and the worker are cleaned up. Nothing is downloaded by this phase.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = "----chaosengine-e2e-clone"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data) in files.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            "Content-Type: audio/wav\r\n\r\n".encode() + data + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def phase_14(cap: Any):
    import e2e_test_suite as suite  # scripts/ is on sys.path when the suite runs

    phase = suite.PhaseResult(phase=14, name="Voice cloning")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 14", "skip", reason="backend not reachable"))
        return phase
    http = suite._voice_http
    state: dict[str, Any] = {}

    def _json(method: str, path: str, payload: dict | None = None, timeout: float = 120.0):
        body = json.dumps(payload).encode() if payload is not None else None
        code, raw = http(method, path, body=body, content_type="application/json" if body else None, timeout=timeout)
        try:
            return code, json.loads(raw.decode() or "null")
        except ValueError:
            return code, raw

    def _models():
        code, body = _json("GET", "/api/voice/clone/models")
        if code == 404:
            return "skip", "voice cloning routes not in this build", {}
        if code != 200:
            return "fail", f"HTTP {code}", {}
        state["models"] = body
        rows = body.get("models") or []
        bad = [r["key"] for r in rows if not re.fullmatch(r"[0-9a-f]{40}", r.get("revision", ""))]
        if len(rows) < 2 or bad:
            return "fail", f"expected two pinned Qwen3-TTS rows, bad: {bad}", {}
        return "pass", "", {"installed": [r["key"] for r in rows if r.get("installed")], "available": body.get("available")}

    def _agent_refused():
        code, access = _json("GET", "/api/agent/access")
        token = (access or {}).get("token") if code == 200 and isinstance(access, dict) else None
        if not token:
            return "skip", "agent access token not readable", {}
        import urllib.error
        import urllib.request

        request = urllib.request.Request(f"http://{suite._HOST}:{suite._PORT}/api/voice/clone/voices",
                                         headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        return ("pass", "", {"status": status}) if status in (401, 403) else ("fail", f"agent token got HTTP {status}", {})

    def _say(text: str) -> bytes | None:
        code, wav = http("POST", "/api/voice/synthesize", body=json.dumps({"text": text, "voice": "am_adam", "speed": 1.0}).encode(),
                         content_type="application/json", timeout=180.0)
        return wav if code == 200 and wav[:4] == b"RIFF" else None

    def _voice_ready() -> str | None:
        code, runtime = _json("GET", "/api/voice/runtime")
        if code != 200 or not runtime.get("ttsAvailable") or not runtime.get("sttAvailable"):
            return "voice pack (Kokoro + STT) not ready"
        if not any(m.get("installed") for m in runtime.get("sttModels") or []):
            return "no STT model downloaded"
        if not (state.get("models") or {}).get("available"):
            return "cloning not supported on this machine"
        return None

    def _wrong_code_refused():
        reason = _voice_ready()
        if reason:
            return "skip", reason, {}
        code, challenge = _json("POST", "/api/voice/clone/consent", {"speakerName": "Test Speaker", "basis": "own"})
        if code != 200:
            return "fail", f"consent start HTTP {code}", {}
        wrong = "".join(str((int(d) + 1) % 10) for d in challenge["code"])
        wav = _say(challenge["sentence"].replace(" ".join(challenge["code"]), " ".join(wrong)))
        if wav is None:
            return "fail", "Kokoro could not read the sentence", {}
        body, ctype = _multipart({"challengeId": challenge["id"], "statementAccepted": "true"}, {"consent": ("c.wav", wav)})
        code, raw = http("POST", "/api/voice/clone/voices", body=body, content_type=ctype, timeout=300.0)
        if code == 400 and b"code" in raw:
            return "pass", "", {}
        return "fail", f"expected 400 about the code, got {code}: {raw[:160]!r}", {}

    def _full_clone():
        reason = _voice_ready()
        if reason:
            return "skip", reason, {}
        installed = [r for r in state["models"]["models"] if r.get("installed")]
        if not installed:
            return "skip", "no cloning model downloaded (won't pull 2.5 GB in a test run)", {}
        model = installed[0]["key"]
        code, challenge = _json("POST", "/api/voice/clone/consent", {"speakerName": "Test Speaker", "basis": "own"})
        wav = _say(challenge["sentence"])
        if wav is None:
            return "fail", "Kokoro could not read the sentence", {}
        body, ctype = _multipart({"challengeId": challenge["id"], "statementAccepted": "true", "name": "E2E voice"},
                                 {"consent": ("c.wav", wav)})
        code, raw = http("POST", "/api/voice/clone/voices", body=body, content_type=ctype, timeout=300.0)
        if code != 200:
            return "fail", f"create HTTP {code}: {raw[:200]!r}", {}
        voice = json.loads(raw)
        speech_id = None
        try:
            code, item = _json("POST", "/api/voice/clone/speak", {"voiceId": voice["id"], "text": "This is an end to end check.", "model": model}, timeout=900.0)
            if code != 200:
                return "fail", f"speak HTTP {code}: {str(item)[:200]}", {}
            speech_id = item["id"]
            code, audio = http("GET", f"/api/voice/library/{speech_id}/audio")
            code_mp3, mp3 = http("GET", f"/api/voice/library/{speech_id}/audio?format=mp3", timeout=300.0)
            wav_tagged = b"chaosengine:" in audio and b'"clonedVoice":true' in audio
            mp3_tagged = code_mp3 == 200 and b'"clonedVoice":true' in mp3
            if not (item.get("clonedVoice") and wav_tagged and mp3_tagged):
                return "fail", f"AI tag missing (item={item.get('clonedVoice')}, wav={wav_tagged}, mp3={mp3_tagged})", {}
            return "pass", "", {"model": model, "durationS": item.get("durationS")}
        finally:
            if speech_id:
                http("DELETE", f"/api/voice/library/{speech_id}")
            http("DELETE", f"/api/voice/clone/voices/{voice['id']}")
            http("POST", "/api/voice/worker/stop")

    for name, fn in (("clone models pinned", _models), ("agent token refused", _agent_refused),
                     ("wrong consent code refused", _wrong_code_refused), ("clone, speak, AI tag", _full_clone)):
        phase.checks.append(suite._check(name, fn))
    statuses = {c.status for c in phase.checks}
    phase.status = "fail" if "fail" in statuses else ("pass" if "pass" in statuses else "skip")
    return phase
