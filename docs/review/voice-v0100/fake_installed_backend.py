"""Review-only backend: the voice runtime LOOKS installed on this Windows PC (no venv, no model, nothing runs).

Reads scenario.json next to this file on every call: {"extras": [...], "speech": [catalog keys], "stt": [repo ids]}.
Started with HOME/USERPROFILE pointing at a throwaway folder.
"""
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ["REVIEW_REPO"])
sys.path.insert(0, str(REPO))


def scenario():
    try:
        return json.loads((HERE / "scenario.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


from backend_service import voice_clone, voice_env, voice_env_asr, voice_runtime, voice_runtime_disk  # noqa: E402

voice_env.is_installed = lambda: True
voice_env.read_marker = lambda: {"lane": "portable", "cuda": True, "installedAt": "2026-11-19T15:42:00",
                                 "extras": scenario().get("extras", [])}
voice_env.installed_extras = lambda: list(scenario().get("extras", []))
voice_env.tts_data_installed = lambda: "kokoro" in scenario().get("extras", [])
voice_env.kokoro_installed = voice_env.tts_data_installed
voice_env.vad_installed = lambda: True
voice_env.diarize_installed = lambda: True
voice_env.missing_features = lambda: []
voice_env.qwen_installed = lambda: True
voice_env.watermark_installed = lambda: True
voice_env_asr.asr_site_installed = lambda: True
voice_clone.snapshot_dir = lambda row: (HERE / "nowhere") if row["key"] in scenario().get("speech", []) else None
voice_runtime._hf_snapshot_path = lambda repo: (HERE / "nowhere") if repo in scenario().get("stt", []) else None


def parts():
    extras = scenario().get("extras", [])
    packages = 2.3 + (5.7 if "speech-gpu" in extras else 0) + (2.9 if "stt-gpu" in extras else 0)
    return [{"id": "packages", "sizeGb": packages}, {"id": "kokoro", "sizeGb": 0.33 if "kokoro" in extras else 0},
            {"id": "turns", "sizeGb": 0.002}, {"id": "speakers", "sizeGb": 0.036}]


voice_runtime_disk.pack_parts = parts
sys.argv = ["backend_service.app"]
runpy.run_module("backend_service.app", run_name="__main__", alter_sys=True)
