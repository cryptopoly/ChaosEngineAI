#!/usr/bin/env python3
"""Check that every Hugging Face repo and file the model catalogs point at exists.

A dead pin (a repo that was renamed or never existed, a quant file name the repo
does not have) only shows up when someone presses Download. This walks the text,
image and video catalogs, asks the Hub for each repo's file list, and reports

  * repos that cannot be reached, and
  * pinned files (GGUF quants, LoRAs, distill weights, sd.cpp companion files)
    that are not in their repo.

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


def find_problems(
    refs: list[Ref], listings: dict[str, set[str] | None]
) -> tuple[dict[str, list[Ref]], list[Ref]]:
    """Split ``refs`` into unreachable repos (with who points at them) and missing files.

    ``listings`` maps a repo id to its file names, or None when the repo could
    not be fetched.
    """
    dead: dict[str, list[Ref]] = {}
    missing: list[Ref] = []
    for ref in refs:
        files = listings.get(ref.repo)
        if files is None:
            dead.setdefault(ref.repo, []).append(ref)
        elif ref.filename and ref.filename not in files:
            missing.append(ref)
    return dead, missing


def fetch_listings(repos: list[str]) -> dict[str, set[str] | None]:
    from huggingface_hub import HfApi

    api = HfApi()

    def fetch(repo: str) -> tuple[str, set[str] | None]:
        try:
            return repo, {sibling.rfilename for sibling in api.model_info(repo).siblings}
        except Exception:  # noqa: BLE001 — a missing, private or renamed repo all count as dead
            return repo, None

    with ThreadPoolExecutor(8) as pool:
        return dict(pool.map(fetch, repos))


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from backend_service.catalog.image_models import IMAGE_MODEL_FAMILIES
    from backend_service.catalog.text_models import MODEL_FAMILIES
    from backend_service.catalog.video_models import VIDEO_MODEL_FAMILIES

    refs = collect_refs({"text": MODEL_FAMILIES, "image": IMAGE_MODEL_FAMILIES, "video": VIDEO_MODEL_FAMILIES})
    repos = sorted({ref.repo for ref in refs})
    print(f"Checking {len(repos)} repos ({len(refs)} references) against the Hub ...")
    dead, missing = find_problems(refs, fetch_listings(repos))

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
