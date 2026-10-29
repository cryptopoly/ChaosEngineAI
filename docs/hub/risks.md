---
title: Risks and assumptions
order: 8
summary: What we assume, what could go wrong, and what we do about it.
---
# Assumptions and risks

Review at the end of every release (the five-line retro goes in [status](status.md)).

## Assumptions

- Apple Silicon with 64 GB is the primary dev and test machine; a result measured there is only a guess for 16 GB Macs and for CUDA.
- Upstream projects we pin (llama.cpp, dflash-mlx, mflux, ltx-2-mlx, TensorFold) keep their command lines and file names between pinned versions.
- Hugging Face repos named in the catalog stay available; `scripts/audit-catalog-repos.py` is the check.

## Risks

| Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|
| A feature works on the Mac and breaks on Windows or Linux (the tracker lists many "not run" items) | high | P1 | Write cross-platform from the first line; CI runs tests on all three; a Windows and Linux UAT pass before big releases | open |
| Release bundle misses a file the source checkout has (happened with `dflash/`, installer scripts, llama-server) | medium | P1 | `validateBundledProjectImports`, staging tests, check the manifest per release ([FU-085](tracker/closed.md), [FU-104](tracker/closed.md), [FU-107](tracker/closed.md)) | open |
| Pinned upstream drifts or breaks (dflash-mlx API rewrite, llama.cpp architectures) | high | P2 | Pins plus sync asserts; migration rows ([FU-057](tracker/open.md)) | open |
| A dependency has a security advisory | medium | P1 | `scripts/audit-dependencies.py` in the pre-release gate and the release workflow ([FU-120](tracker/open.md)) | in progress |
| Silent wrong output instead of an error (MPS bf16 corruption, a speculative path that quietly falls back) | medium | P1 | Probe notes, contract tests against the installed library, measure before and after | open |
| Memory estimates wrong and a load swaps or crashes | medium | P2 | Memory gate and load check are advisory and calibrated on measured runs ([FU-110](tracker/open.md)) | in progress |
| CI minutes burn on every push | high | P3 | Hold pushes until tested ([rules](rules.md)) | open |
| Model licences restrict use (Qwen Research non-commercial, LTX community licence, gated repos) | medium | P2 | Licence stated on each model row; consider an acceptance prompt ([FU-096](tracker/open.md)) | open |

## Third-party assets and licences

The ledger is `THIRD_PARTY_NOTICES.md` (bundled and vendored dependencies with licence types); new dependencies must be MIT, Apache 2.0, BSD or similar and are added there first. Model weights are not bundled; each row's note states the licence.
