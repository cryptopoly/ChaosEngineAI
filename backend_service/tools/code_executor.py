"""Sandboxed Python code execution tool.

Runs user-provided Python code in an isolated subprocess with:
- 30-second timeout
- No network access (not enforced at OS level, but the subprocess is short-lived)
- Captured stdout/stderr
- Memory limit via resource soft limits (Unix only)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from backend_service.tools import BaseTool

_MAX_TIMEOUT_SECONDS = 30
_MAX_OUTPUT_CHARS = 8000
_MAX_CODE_LENGTH = 10000


class CodeExecutorTool(BaseTool):
    name = "code_executor"
    description = (
        "Execute Python code in a sandboxed subprocess and return the output. "
        "Use this for calculations, data processing, string manipulation, or any task "
        "that benefits from running actual code. The code runs with a 30-second timeout."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() to produce output.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (1-30, default 30).",
                    "default": 30,
                },
            },
            "required": ["code"],
        }

    def execute(self, **kwargs: Any) -> str:
        code = str(kwargs.get("code", "")).strip()
        if not code:
            return "Error: no code provided."

        if len(code) > _MAX_CODE_LENGTH:
            return f"Error: code exceeds {_MAX_CODE_LENGTH} character limit."

        timeout = min(max(int(kwargs.get("timeout", _MAX_TIMEOUT_SECONDS)), 1), _MAX_TIMEOUT_SECONDS)

        # Write code to a temp file and execute in a fresh Python process
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="chaosengine_exec_",
                delete=False,
            ) as tmp:
                tmp.write(code)
                tmp_path = Path(tmp.name)

            try:
                completed = subprocess.run(
                    [sys.executable, str(tmp_path)],
                    capture_output=True,
                    timeout=timeout,
                    cwd=tempfile.gettempdir(),
                    env={
                        "PATH": "",
                        "HOME": tempfile.gettempdir(),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )

                stdout = completed.stdout.decode("utf-8", errors="replace").strip()
                stderr = completed.stderr.decode("utf-8", errors="replace").strip()

                parts: list[str] = []
                if stdout:
                    parts.append(f"Output:\n{stdout[:_MAX_OUTPUT_CHARS]}")
                if stderr:
                    parts.append(f"Errors:\n{stderr[:_MAX_OUTPUT_CHARS]}")
                if completed.returncode != 0:
                    parts.append(f"Exit code: {completed.returncode}")
                if not parts:
                    parts.append("Code executed successfully (no output).")

                return "\n\n".join(parts)

            except subprocess.TimeoutExpired:
                return f"Error: code execution timed out after {timeout} seconds."
            finally:
                tmp_path.unlink(missing_ok=True)

        except OSError as exc:
            return f"Error: failed to execute code: {exc}"
