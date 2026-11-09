#!/usr/bin/env python3
"""Check that every Hugging Face repo and file the model catalogs point at exists.

A dead pin (a repo that was renamed or never existed, a quant file name the repo
does not have, a revision that was deleted) only shows up when someone presses
Download. This walks the text, image, video, music and speech catalogs, asks the
Hub for each repo's file list (at the pinned revision when the row pins one), and
reports

  * repos (or pinned revisions) that cannot be reached, and
  * pinned files (GGUF quants, LoRAs, distill weights, sd.cpp companion files,
    speech-model companion files) that are not in their repo.

No token is ever sent (``token=False``): every repo here must be readable
without a login, and the owner's token must never be read by this script.

Usage (needs network; only metadata is fetched, no weights):

    .venv/bin/python scripts/audit-catalog-repos.py

Exits 1 when anything is dead, so it can gate a release by hand. It always
imports the catalogs from this checkout, never from a copy installed in the
environment.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]


class Ref(NamedTuple):
    catalog: str
    variant: str
    repo: str
    filename: str | None
    role: str
    revision: str | None = None

    @property
    def target(self) -> tuple[str, str | None]:
        return self.repo, self.revision


def collect_refs(catalogs: dict[str, list[dict[str, Any]]]) -> list[Ref]:
    """Every (repo, file) a catalog variant depends on."""
    refs: list[Ref] = []

    def add(catalog: str, variant: str, repo: Any, filename: Any, role: str) -> None:
        if repo:
            refs.append(Ref(catalog, variant, str(repo), str(filename) if filename else None, role))

    for catalog, families in catalogs.items():
        for family in families:
            for variant in family.get("variants", []):
                vid = str(variant.get("id"))
                if catalog == "text":
                    # Text rows pin a GGUF file inside their own repo.
                    add(catalog, vid, variant.get("repo"), variant.get("ggufFile"), "repo")
                    continue
                add(catalog, vid, variant.get("repo"), None, "repo")
                add(catalog, vid, variant.get("ggufRepo"), variant.get("ggufFile"), "gguf")
                add(catalog, vid, variant.get("loraRepo"), variant.get("loraFile"), "lora")
                add(catalog, vid, variant.get("nunchakuRepo"), variant.get("nunchakuFile"), "nunchaku")
                add(catalog, vid, variant.get("textEncoderRepo"), None, "textEncoder")
                distill = variant.get("distillTransformerRepo")
                add(catalog, vid, distill, variant.get("distillTransformerHighNoiseFile"), "distill-high")
                add(catalog, vid, distill, variant.get("distillTransformerLowNoiseFile"), "distill-low")
                for flag, source in (variant.get("sdcppAux") or {}).items():
                    add(catalog, vid, source.get("repo"), source.get("file"), f"sdcppAux {flag}")
    return refs


def collect_pinned_refs(catalog: str, rows: list[dict[str, Any]]) -> list[Ref]:
    """Rows that pin a full revision (music variants, speech models) and their companion repos' files."""
    refs: list[Ref] = []
    for row in rows:
        vid = str(row.get("key") or row.get("id"))
        if row.get("repo"):
            refs.append(Ref(catalog, vid, str(row["repo"]), None, "repo", row.get("revision")))
        for companion in row.get("companions", []):
            for name in companion["files"]:
                refs.append(Ref(catalog, vid, companion["repo"], name, f"companion {companion.get('role', '')}".strip(),
                                companion["revision"]))
    return refs


def find_problems(
    refs: list[Ref], listings: dict[Any, set[str] | None]
) -> tuple[dict[str, list[Ref]], list[Ref]]:
    """Split ``refs`` into unreachable repos (with who points at them) and missing files.

    ``listings`` maps ``(repo, revision)`` (or a bare repo id for unpinned refs) to its file names, or None when
    it could not be fetched.
    """
    dead: dict[str, list[Ref]] = {}
    missing: list[Ref] = []
    for ref in refs:
        files = listings.get(ref.target, listings.get(ref.repo)) if ref.revision is None else listings.get(ref.target)
        if files is None:
            label = f"{ref.repo}@{ref.revision[:8]}" if ref.revision else ref.repo
            dead.setdefault(label, []).append(ref)
        elif ref.filename and ref.filename not in files:
            missing.append(ref)
    return dead, missing


def fetch_listings(targets: list[tuple[str, str | None]]) -> dict[tuple[str, str | None], set[str] | None]:
    from huggingface_hub import HfApi

    api = HfApi()

    def fetch(target: tuple[str, str | None]) -> tuple[tuple[str, str | None], set[str] | None]:
        repo, revision = target
        try:
            info = api.model_info(repo, revision=revision, token=False)
            return target, {sibling.rfilename for sibling in info.siblings}
        except Exception:  # noqa: BLE001 — a missing, private or renamed repo all count as dead
            return target, None

    with ThreadPoolExecutor(8) as pool:
        return dict(pool.map(fetch, targets))


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from backend_service.catalog.image_models import IMAGE_MODEL_FAMILIES
    from backend_service.catalog.music_models import MUSIC_MODEL_FAMILIES
    from backend_service.catalog.text_models import MODEL_FAMILIES
    from backend_service.catalog.video_models import VIDEO_MODEL_FAMILIES
    from backend_service.catalog.voice_models import VOICE_MODELS

    refs = collect_refs({"text": MODEL_FAMILIES, "image": IMAGE_MODEL_FAMILIES, "video": VIDEO_MODEL_FAMILIES})
    refs += collect_pinned_refs("music", [v for family in MUSIC_MODEL_FAMILIES for v in family["variants"]])
    refs += collect_pinned_refs("voice", VOICE_MODELS)
    targets = sorted({ref.target for ref in refs}, key=lambda t: (t[0], t[1] or ""))
    print(f"Checking {len(targets)} repos ({len(refs)} references) against the Hub ...")
    dead, missing = find_problems(refs, fetch_listings(targets))

    for repo, users in sorted(dead.items()):
        print(f"DEAD REPO  {repo}")
        for ref in users:
            print(f"    <- [{ref.catalog}] {ref.variant} ({ref.role})")
    for ref in missing:
        print(f"MISSING FILE  [{ref.catalog}] {ref.variant}: {ref.repo} / {ref.filename} ({ref.role})")
    if not dead and not missing:
        print("All catalog repos and pinned files exist.")
        return 0
    print(f"{len(dead)} dead repos, {len(missing)} missing files.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
