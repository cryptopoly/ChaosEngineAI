# Changelog

The canonical changelog lives at
[`CHANGELOG.md`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/CHANGELOG.md)
in the repo root. It's kept tag-by-tag with one entry per release.

Current top entry (as of v0.8.0):

## v0.8.0 — 2026-05-10

Multi-week pass through the largest backend / frontend modules to land the
v0.8.0 modularisation goal. Zero feature regressions — 1,302 Python tests +
340 TypeScript tests pass before and after every commit; all type checks
clean.

Highlights:

- **MLX worker memory leak fix.** `JsonRpcProcess.close()` now captures and
  nulls `self.process` up-front + wraps the post-kill `wait()` in a 1 s
  timeout, mirroring `LlamaCppEngine._cleanup_process`. Fixes the
  two-47 GB-workers bug under memory pressure.
- **Major shrinks across the four biggest backend modules.**
  `state/__init__.py` 4418 → 860 LOC (-81%). `inference/__init__.py`
  3574 → 97 LOC (-97%). `mlx_worker.py` 2115 → 318 LOC (-85%).
  `image_runtime/__init__.py` 2097 → 992 LOC (-53%). All via sibling
  modules + re-exports; no external import paths changed.
- **FU-030 / FU-031** — ChaosEngine + RotorQuant cache strategy slots
  dropped (eclipsed by KVTC at ICLR 2026); persisted configs that
  reference them coerce silently to `turboquant`. DFlash registry
  extended for Gemma 4 + new z-lab drafters.
- **FU-034 / FU-037 / FU-038 / FU-039 / FU-040 / FU-041** — UX polish
  cluster: hide unrecoverable launch-modal options, per-tab error
  boundary, MallocStackLogging spam suppression, tool-call `arguments:
  null` fix, tool-call XML parsing widened to open-only + array shapes,
  Qwen3.6-27B vs Coder-Next canonicalisation.
- **FU-042** — i18n Phase 0 — infra scaffold + IME composition fix.

For tagged releases see the
[GitHub Releases page](https://github.com/cryptopoly/ChaosEngineAI/releases).

## Tracking follow-ups

Deferred work and upstream conditions to re-check periodically are tracked
in CLAUDE.md's [Follow-Ups Tracker](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/CLAUDE.md).
The tracker is the canonical "what's queued, what's done" record. Items are
deleted once shipped (with a strikethrough preserved for archaeology) or
declared no longer relevant.

## Version policy

- **MAJOR** — backwards-incompatible changes to the on-disk model layout,
  settings format, or persistent session schema.
- **MINOR** — new features, new cache strategies, new engines, new
  catalog rows.
- **PATCH** — bug fixes, UI polish, dependency bumps.

Pre-1.0 releases may still introduce breaking changes between minor
versions — the README's "Work in progress" warning applies.

## Release cadence

Tag-driven. Push `vX.Y.Z` to the staging branch and the GitHub Actions
release workflow builds signed bundles for macOS, Linux, and Windows in
parallel, generates the `latest.json` updater manifest, and stages a
draft release.

## See also

- [Pre-build check](../testing/pre-build-check.md) — the gate every
  release runs.
- [Adding a feature](../contributing/adding-a-feature.md).
