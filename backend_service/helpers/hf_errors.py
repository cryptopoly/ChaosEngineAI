"""Hugging Face error → friendly message translators.

Walks Python tracebacks emitted by ``huggingface_hub.snapshot_download``
and similar paths, condenses them to the actionable line, and rewrites
the common failure modes (missing PyYAML, gated / private repos, 404 /
auth refusal, DNS failures) into surface-ready copy that points users
at the right Settings → Setup action.

Extracted from ``backend_service/helpers/huggingface.py`` as part of
the v0.8.0 refactor. Re-exported from ``helpers.huggingface``.
"""

from __future__ import annotations


def _condense_hf_error(error: str) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    ignored_prefixes = (
        "traceback",
        "file ",
        "raise ",
        "for more information",
        "for more details",
    )
    ignored_substrings = (
        "userwarning",
        "warnings.warn",
        "httpstatuserror:",
        "repositorynotfounderror:",
    )
    for line in reversed(lines):
        lowered = line.lower()
        if lowered.startswith(ignored_prefixes):
            continue
        if any(fragment in lowered for fragment in ignored_substrings):
            continue
        return line
    return lines[-1] if lines else str(error).strip()


def _friendly_hf_download_error(repo_id: str, error: str) -> str:
    lowered = str(error).lower()
    # The snapshot_download subprocess imports ``huggingface_hub`` which
    # transitively imports ``yaml``. A partially-installed PyYAML in the
    # extras dir surfaces as ``ModuleNotFoundError: No module named 'yaml...'``
    # rather than a network or repo error. Translate it into actionable
    # guidance (open Setup → install pyyaml) instead of a cryptic Python
    # traceback rendered as the download status.
    if "no module named 'yaml" in lowered or "no module named yaml" in lowered:
        return (
            "The backend Python is missing PyYAML, which huggingface_hub needs "
            "to read model cards. Open Settings > Setup and click Install "
            "pyyaml (or re-run Install GPU runtime) to repair the runtime, "
            f"then retry the download for {repo_id}."
        )
    if "modulenotfounderror" in lowered and "huggingface_hub" in lowered:
        return (
            "The backend Python could not import huggingface_hub. Open Settings > "
            "Setup and click Install GPU runtime to repair the runtime, then "
            f"retry the download for {repo_id}."
        )
    if (
        "repository not found" in lowered
        or "repo not found" in lowered
        or "404 client error" in lowered
    ):
        return (
            f"{repo_id} was not found on Hugging Face. "
            "This repo may have moved or the catalog entry may be stale."
        )
    if (
        "refused access" in lowered
        or "http 401" in lowered
        or "http 403" in lowered
        or "invalid username or password" in lowered
        or "authentication required" in lowered
        or "cannot access gated repo" in lowered
        or "gated repo" in lowered
        or ("access to model" in lowered and "restricted" in lowered)
    ):
        return (
            f"Hugging Face refused access to {repo_id}. "
            "If the repo is gated or private, make sure your account has access "
            "and add a read-enabled HF_TOKEN in Settings."
        )
    if (
        "connecterror" in lowered
        or "name or service not known" in lowered
        or "nodename nor servname provided" in lowered
        or "temporary failure in name resolution" in lowered
        or "timed out" in lowered
    ):
        return (
            f"ChaosEngineAI could not reach Hugging Face while checking {repo_id}. "
            "Check the backend network connection and retry."
        )
    condensed = _condense_hf_error(error)
    return condensed or f"Download failed for {repo_id}."
