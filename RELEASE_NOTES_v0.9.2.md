# ChaosEngineAI v0.9.2

*Released 2026-05-16*

Two flagship features: a full-surface CLI for headless automation, and native Multi-Token Prediction speculative decoding on Apple Silicon. Plus a phased E2E test suite, a documentation site, and a clutch of fixes around library hygiene, vision routing, and TurboQuant on hybrid-attention MoE models.

---

## Headline

### chaosengine-cli — full headless automation

Ship the desktop app, drive it from a terminal. The new `chaosengine-cli` wrapper covers **100% of backend route surface**: 95 typed shortcuts for the common verbs (chat, prompt, compare, html-challenge, image, video, benchmarks, setup, diagnostics, server control), plus a generic dispatcher for anything else.

- Bundled with every install — no extra step.
- Works without the desktop GUI. Start `chaosengine-cli serve` from any shell and the backend comes up the same way the Tauri shell would launch it.
- Useful for: automation scripts, CI pipelines, remote dev boxes, batch generation runs, regression sweeps, and headless benchmarking.

```bash
chaosengine-cli serve                  # boot backend
chaosengine-cli prompt "hello world"   # one-shot completion
chaosengine-cli image generate ...     # studio without the GUI
chaosengine-cli diagnostics snapshot   # full system report
```

### MTPLX — native MTP speculative decoding on Apple Silicon

Multi-Token Prediction (MTP) drafters get a lossless 1.5–2.2× speedup on models trained with MTP heads. v0.9.2 wires the Apache 2.0 `mtplx` package into the Setup tab as a one-click install.

- Auto-installs to an isolated venv at `~/.chaosengine/mtplx-venv/` (~500 MB).
- Supported model families today: Qwen3.5, Qwen3.6, DeepSeek V3 / R1, Qwen3-Coder-Next, and the Youssofal MTPLX-Optimized variants.
- Default draft depth is now `3` (matches MTPLX's own default).
- No more browser pop-up on first model load — MTPLX now starts in headless `quickstart` mode.

---

## What's new

### GGUF MTP via llama.cpp

Apple Silicon isn't the only path anymore. The llama.cpp engine now accepts `--spec-type draft-mtp --spec-draft-n-max N`, wired through cleanly from the launch settings panel. Catalog entries shipped: `ggml-org/Qwen3.6-27B-MTP-GGUF` and `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF`.

Requires a `llama-server` binary built from upstream master ≥ 2026-05-16. Older binaries fall back to standard decode with a clear in-UI note explaining why and how to upgrade.

### Phased E2E test suite

`scripts/e2e_test_suite.py` — 8 phases, 32+ checks, ~2–3 minutes on a warm cache. It drives the CLI through every major surface (chat, compare, html-challenge, image studio, video studio, setup probes, diagnostics) and is now **required before every release**. Catches regressions that unit tests don't see because they live across the FastAPI / runtime / worker boundary.

### Full documentation site

A real MkDocs Material site at **`chaosengineai.com/docs/`** — ~42 pages auto-deployed from the `docs/` tree. Covers install, CLI reference, feature deep-dives, troubleshooting, architecture, and contributing. Pull requests against docs build a preview automatically.

---

## Bug fixes

- **TurboQuant cache + hybrid-attention MoE models** — generations against Qwen3.5 / Qwen3.6 A3B no longer crash immediately after load. ArraysCache slots are now preserved for hybrid-attn layers under TurboQuant.
- **Library hygiene** — entries whose backing directory was deleted on disk are now pruned automatically. An explicit `--path` is also trusted over a stale broken-library-entry for the same ref, so you can load directly past a corrupt registry record.
- **GGUF vision mmproj scoping** — the mmproj resolver no longer reaches into unrelated model directories. Fixes `llama-server` startup crashes on text-only Gemma-4 and other mismatched-model loads.
- **CLI metrics** — `prompt` and `bench` subcommands now report live tokens/sec + token counts from the real response shape (previously surfaced as nulls).
- **MTPLX first-run** — no more stray browser tab popping on first model load. Draft depth bumped from 1 → 3.

---

## Under the hood

- Pre-build gate (`scripts/pre-build-check.sh`) now runs the E2E smoke + a draft-mtp readiness probe and flags a stale bundled `llama-server` against the GGUF MTP requirement.
- Test counts: **1,418 Python tests pass** (up from 1,355 at v0.9.0), **371 TypeScript tests**, plus a new 39-test CLI smoke suite, a new 10-test MTPLX integration suite, and 5 new MTP unit tests.
- **New feature gate** in CLAUDE.md: every user-visible feature must ship with an E2E check from now on.

---

## Upgrade notes & known caveats

- **GGUF MTP needs a fresh `llama-server`.** Homebrew users: `brew upgrade llama.cpp`. Bundled-binary release builds pick this up automatically on the next stage-runtime cycle.
- **MTPLX install is opt-in.** ~500 MB. Install only when you actually want MTP speculative decoding — skip it if you're staying on DFlash or running CUDA.
- **MTPLX vs plain MLX throughput.** On M5 hardware, MTPLX currently caps at ~95% of plain MLX baseline tokens/sec on a 27B model. The MTP head gains are real, but per-token roundtrip latency through the isolated venv server eats some of the win. Optimisation work continues; the value prop for now is having the path + future-proofing for the GGUF lane.
- **No breaking changes.** v0.9.0 settings, sessions, gallery entries, library, and chat threads all carry forward unchanged. No API breakage.

---

## Credits / upstream

- **llama.cpp PR #22673** (am17an) — the upstream MTP merge that made the GGUF lane possible.
- **`youssofal/mtplx`** (Apache 2.0) — Apple Silicon native MTP runtime.
- **Model authors** who shipped MTP-bearing weights: Qwen team, DeepSeek, Youssofal.

---

## Installation

**Desktop app** — download from `chaosengineai.com/#download` or the GitHub Releases page once tagged.

**CLI-only (headless / server / CI)**:

```bash
git clone https://github.com/cryptopoly/ChaosEngineAI
cd ChaosEngineAI
python3 -m venv .venv
.venv/bin/pip install -e .
./scripts/chaosengine-cli serve
```

Then use `chaosengine-cli` from any shell.

**Docs**: `chaosengineai.com/docs/`.
