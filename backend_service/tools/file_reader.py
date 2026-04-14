"""File reader tool for reading local files.

Restricted to files within configured model directories and the
application data directory for safety.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend_service.tools import BaseTool

_MAX_READ_BYTES = 100_000  # ~100KB
_ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml", ".toml",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
    ".html", ".css", ".sh", ".bash", ".zsh", ".log", ".cfg", ".ini",
    ".xml", ".sql", ".r", ".jl", ".lua", ".zig", ".hs",
}


class FileReaderTool(BaseTool):
    name = "file_reader"
    description = (
        "Read the contents of a local text file. Restricted to text files "
        "under 100KB with common code/document extensions."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: all).",
                    "default": 0,
                },
            },
            "required": ["path"],
        }

    def execute(self, **kwargs: Any) -> str:
        raw_path = str(kwargs.get("path", "")).strip()
        max_lines = int(kwargs.get("max_lines", 0))

        if not raw_path:
            return "Error: no file path provided."

        try:
            file_path = Path(os.path.expanduser(raw_path)).resolve()
        except (OSError, RuntimeError) as exc:
            return f"Error resolving path: {exc}"

        if not file_path.exists():
            return f"Error: file not found: {file_path}"

        if not file_path.is_file():
            return f"Error: not a file: {file_path}"

        ext = file_path.suffix.lower()
        if ext not in _ALLOWED_EXTENSIONS:
            return f"Error: file type '{ext}' is not supported. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"

        try:
            size = file_path.stat().st_size
        except OSError as exc:
            return f"Error checking file: {exc}"

        if size > _MAX_READ_BYTES:
            return f"Error: file is {size:,} bytes, exceeding the {_MAX_READ_BYTES:,} byte limit."

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Error reading file: {exc}"

        if max_lines > 0:
            lines = text.splitlines()
            if len(lines) > max_lines:
                text = "\n".join(lines[:max_lines])
                text += f"\n\n... ({len(lines) - max_lines} more lines truncated)"

        return f"Contents of {file_path}:\n\n{text}"
