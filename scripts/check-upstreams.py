#!/usr/bin/env python3
"""Report every pinned upstream against its newest release (D-78, FU-169).

Reads each pin from the file that owns it and asks the upstream for its newest
release or commit. Prints one line per pin: ``current``, ``behind`` or
``unknown``. It reports only: the exit code is 0 even when pins are behind or a
host is unreachable, so it never blocks a build. ``--json`` prints the same
rows as JSON. ``GITHUB_TOKEN`` (optional) lifts the GitHub API rate limit.

Needs network. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Row:
    name: str
    pinned: str
    newest: str
    state: str  # current | behind | unknown
    note: str = ""


def read_pin(relative: str, pattern: str) -> str | None:
    try:
        text = (ROOT / relative).read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else None


def http_json(url: str, timeout: float = 20.0):
    # curl, not urllib: a python.org Python has no CA bundle and fails every https call.
    cmd = ["curl", "-fsSL", "--max-time", str(int(timeout)), "-H", "User-Agent: ChaosEngineAI-upstream-check",
           "-H", "Accept: application/vnd.github+json"]
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    try:
        out = subprocess.run([*cmd, url], capture_output=True, text=True, timeout=timeout + 5, check=False).stdout
        return json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def ls_remote(repo: str, ref: str) -> list[tuple[str, str]]:
    try:
        out = subprocess.run(
            ["git", "ls-remote", repo, ref], capture_output=True, text=True, timeout=60, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        sha, _, name = line.partition("\t")
        if sha and name:
            rows.append((sha, name))
    return rows


def version_key(tag: str) -> tuple:
    return tuple(int(part) if part.isdigit() else 0 for part in re.findall(r"\d+|[a-z]+", tag.lower()))


def newest_tag(repo: str, pattern: str) -> str | None:
    tags = [n.removeprefix("refs/tags/").removesuffix("^{}") for _, n in ls_remote(repo, "refs/tags/*")]
    tags = sorted({t for t in tags if re.fullmatch(pattern, t)}, key=version_key)
    return tags[-1] if tags else None


def head_sha(repo: str, ref: str = "HEAD") -> str | None:
    rows = ls_remote(repo, ref)
    return rows[0][0] if rows else None


def row(name: str, pinned: str | None, newest: str | None, *, same=lambda a, b: a == b, note: str = "") -> Row:
    if not pinned:
        return Row(name, "?", newest or "?", "unknown", "pin not found in the repo")
    if not newest:
        return Row(name, pinned, "?", "unknown", "upstream unreachable")
    return Row(name, pinned, newest, "current" if same(pinned, newest) else "behind", note)


def sha_same(pinned: str, newest: str) -> bool:
    return newest.startswith(pinned) or pinned.startswith(newest)


def check_llama_cpp() -> Row:
    pinned = read_pin("scripts/update-llama-cpp.sh", r'LLAMA_CPP_REF="\$\{LLAMA_CPP_REF:-(b\d+)\}"')
    releases = http_json("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30")
    newest = None
    for release in releases if isinstance(releases, list) else []:
        names = [a.get("name", "") for a in release.get("assets", [])]
        # Only a release whose prebuilt archives are published can be pinned.
        if any("macos-arm64" in n for n in names) and any("win-cuda" in n for n in names):
            newest = release.get("tag_name")
            break
    return row("llama.cpp", pinned, newest, note="newest tag that has its prebuilt archives")


def check_tensorfold() -> Row:
    pinned = read_pin("backend_service/inference/_tensorfold_native.py", r'^NATIVE_VERSION = "([\d.]+)"')
    tag = newest_tag("https://github.com/ashhart/TensorFold.git", r"v1\.\d+\.\d+")
    return row("TensorFold (native 1.x)", pinned, tag.removeprefix("v") if tag else None)


def check_mflux() -> Row:
    pinned = read_pin("scripts/install-mflux.sh", r'MFLUX_VERSION="\$\{MFLUX_VERSION:-([\d.]+)\}"')
    data = http_json("https://pypi.org/pypi/mflux/json")
    return row("mflux", pinned, (data or {}).get("info", {}).get("version"))


def check_commit_pin(name: str, repo: str, pinned: str | None, branch: str = "HEAD") -> Row:
    return row(name, pinned[:8] if pinned else None, (head_sha(repo, branch) or "")[:8] or None, same=sha_same)


def check_ltx2() -> Row:
    pinned = read_pin("scripts/install-ltx2-mlx.sh", r'LTX2_MLX_REF="\$\{LTX2_MLX_REF:-([0-9a-f]{40})\}"')
    result = check_commit_pin("ltx-2-mlx", "https://github.com/dgrauet/ltx-2-mlx.git", pinned)
    if result.state == "behind" and pinned:
        diff = http_json(f"https://api.github.com/repos/dgrauet/ltx-2-mlx/compare/{pinned[:12]}...HEAD")
        if isinstance(diff, dict) and "files" in diff:
            runtime = [f["filename"] for f in diff["files"] if not f["filename"].startswith(".github/")]
            result.note = (
                f"{diff.get('ahead_by')} commit(s); "
                + (f"code changed: {', '.join(runtime[:3])}" if runtime else "CI files only, nothing to bump")
            )
            if not runtime:
                result.state = "current"
    return result


def check_dflash() -> Row:
    pinned = read_pin("pyproject.toml", r"dflash-mlx\.git@([0-9a-f]{40})")
    repo = "https://github.com/bstnxbt/dflash-mlx.git"
    tag = newest_tag(repo, r"v\d+(\.\d+)+")
    # Pinned by commit: current when it is the commit the newest tag points at (the peeled ^{} row).
    tag_commits = {sha for sha, _ in ls_remote(repo, f"refs/tags/{tag}*")} if tag and pinned else set()
    result = row("dflash-mlx", pinned[:8] if pinned else None, tag,
                 same=lambda a, b: pinned in tag_commits)
    if result.state == "behind":
        result.note = "pinned by commit; newest is a release tag (FU-057)"
    return result


def check_playwright() -> Row:
    pinned = read_pin("e2e/package.json", r'"@playwright/test": "([\d.]+)"')
    data = http_json("https://registry.npmjs.org/@playwright%2ftest/latest")
    return row("Playwright (dev only)", pinned, (data or {}).get("version"))


def unlisted_families() -> tuple[str, list[str]] | None:
    """(binary version, model_types the installed TensorFold binary serves that no table row lists), or None."""
    sys.path.insert(0, str(ROOT))
    try:
        from backend_service.inference import _tensorfold, _tensorfold_native as native

        binary = native.find_binary()
        caps = native.probe_capabilities(binary) if binary else None
        if caps is None:
            return None
        listed = {mt for fam in _tensorfold.TENSORFOLD_FAMILIES for mt in fam.model_types}
        listed |= {model.family for model in native.NATIVE_MODELS}
        return caps.version or "?", sorted(set(caps.families) - listed)
    except Exception:  # noqa: BLE001 - a report must never fail the build
        return None


def collect() -> list[Row]:
    return [
        check_llama_cpp(),
        check_tensorfold(),
        check_ltx2(),
        check_mflux(),
        check_dflash(),
        check_commit_pin(
            "ACE-Step 1.5", "https://github.com/ace-step/ACE-Step-1.5.git",
            read_pin("backend_service/music_env.py", r'^ACE_REF = "([0-9a-f]{40})"'),
        ),
        check_commit_pin(
            "TriAttention", "https://github.com/WeianMao/triattention.git",
            read_pin("pyproject.toml", r"triattention\.git@([0-9a-f]{40})"),
        ),
        check_commit_pin(
            "Chatterbox", "https://github.com/resemble-ai/chatterbox.git",
            read_pin("backend_service/voice_env.py", r'^CHATTERBOX_REF = "([0-9a-f]{7,40})"'),
        ),
        check_playwright(),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = parser.parse_args(argv)
    rows = collect()
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return 0
    width = max(len(r.name) for r in rows)
    print("Pinned upstreams against their newest release (report only, never blocks):")
    for r in rows:
        mark = {"current": "ok    ", "behind": "BEHIND", "unknown": "?     "}[r.state]
        extra = f"  ({r.note})" if r.note else ""
        print(f"  {mark} {r.name.ljust(width)}  pinned {r.pinned}  newest {r.newest}{extra}")
    families = unlisted_families()
    if families and families[1]:
        print(f"TensorFold {families[0]} serves {len(families[1])} model families no catalog row lists (FU-168): "
              f"{', '.join(families[1])}")
    behind = [r.name for r in rows if r.state == "behind"]
    if behind:
        print(f"{len(behind)} behind: {', '.join(behind)}. Bump each, or note why not in FOLLOW_UPS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
