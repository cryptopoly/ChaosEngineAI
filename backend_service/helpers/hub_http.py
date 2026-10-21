"""Small JSON reads from the Hugging Face Hub that must not touch the HF cache.

``hf_hub_download`` would create a cache entry holding just the one file,
which the library scan and the "installed" checks would then take for a
partly downloaded model. These helpers read the file over HTTP instead.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def ssl_context():
    """TLS context trusting certifi's bundle (what huggingface_hub uses).

    python.org builds of Python ship no CA bundle until "Install
    Certificates" is run, so plain urllib calls to the Hub fail there.
    None (urllib's default) when certifi is missing.
    """
    try:
        import ssl

        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def fetch_hub_json(repo: str, filename: str, *, timeout: float = 5.0, max_bytes: int = 20_000_000) -> Any | None:
    """Parsed JSON of ``filename`` at the head of ``repo``, or None on any failure."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = (
        f"{endpoint}/{urllib.parse.quote(repo, safe='/')}/resolve/main/"
        f"{urllib.parse.quote(filename, safe='/')}"
    )
    headers = {"User-Agent": "ChaosEngineAI"}
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        token = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as resp:
            return json.loads(resp.read(max_bytes).decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
