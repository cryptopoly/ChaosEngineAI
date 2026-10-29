---
title: Decisions
order: 2
summary: Open decisions with a recommendation, and the dated log of decisions made.
---
# Decisions

Open decisions come first, in priority order. Decided ones move to the collapsed log at the bottom with a date. A decision raised in chat is logged here the same turn, with a recommendation and a reason.

Priority: P1 blocks launch or reworks built work; P2 is needed before the next milestone; P3 is later.

## Open decisions

### P1: decide soon

1. **Push the Windows test fix to PR 119.** One local commit (a test compared `/pack` with `\pack`). **Explainer:** every push runs GitHub builds, which cost Actions minutes, so pushes are held until the work is tested. **Recommendation:** push once the DDTree 27B and LTX-2.5 runs are done and committed, so CI runs once. **Why:** the other two platforms already pass; nothing else is blocked on it.
2. **When to re-tag `v0.9.9`.** The tag still points at an old commit and the release workflow reuses the existing draft. **Recommendation:** re-tag only after PR 119 is merged and CI is green; commands are in [status](status.md) history. **Why:** one 30-minute build instead of two.

### P2: before the next milestone

3. **Make LTX-2.5 the default LTX row.** Newer and bundles its own text encoder, but it is gated and unmeasured here. **Recommendation:** switch the default after one live clip matches LTX-2.3 on speed and memory. **Why:** your rule of thumb is newer and faster, but a default that fails a gated download is worse than a slower one. Runs are in [progress](progress/runs-2026-10-29.md).
4. **Names of other apps in the import-models strings** (`importModels.ollama`, `importModels.lmstudio`). **Explainer:** the editorial rule in [rules](rules.md) forbids naming other desktop AI apps in UI strings. **Recommendation:** rename them to "Other app model folders" style generic text and keep the folder paths. **Why:** keeps the rule without removing the feature.

### P3: later

5. **oMLX as an MLX engine lane** ([FU-132](tracker/open.md)). An Apache-2.0 MLX server with continuous batching and an SSD-persistent KV cache, over HTTP only. **Explainer:** it already works as a remote provider (Settings, OpenAI-compatible base URL) with no code; a real lane means another MLX stack in its own venv and it cannot use our DFlash or TurboQuant. **Recommendation:** do not build a lane yet; measure it against our MLX lane first (needs your yes to install it in an isolated venv, about 45 minutes). **Why:** our MLX path already reuses prompt caches; the SSD cache and concurrent requests are the only clear gains, and they matter more for agents and the API server than for one chat.
6. **Merge Discover and My Models into one Models page per area** ([FU-129](tracker/open.md)). **Recommendation:** after v0.9.9, image and video first, redirects for the old routes, 2 to 4 days. **Why:** the screens are nearly identical, but it touches about 4,000 lines and the release should not wait for it.
7. **Local agents building a game through the app** ([FU-126](tracker/open.md)). Banked by you; revisit when a local coding model clearly matches a hosted frontier model. No action now.

<details>
<summary>Decision log (made decisions, newest section last)</summary>

## 2026-10-28 and 2026-10-29

- **Prefer newer and faster models when choosing defaults.** Your rule of thumb. Saved to memory and used for the LTX and Wan choices.
- **Wan 2.2 TI2V 5B is replaced by LTX-2.3 on its own MLX engine** (`ltx-2-mlx`), on Claude's recommendation. The 5B family and its caches are removed; Wan 2.2 A14B and Wan 2.1 rows stay. See [FU-128](tracker/closed.md).
- **Agent game-dev idea is banked** ([FU-126](tracker/open.md)). Decided by Dan.
- **Linear DFlash block size 8 on MLX** (was 16). Decided by Dan on Claude's recommendation; checked on Qwen3-4B, to be re-checked on a 27B target.
- **Sound in Video Studio: no toggle; show the sound** (icon, unmuted result window, prompt hint). Decided by Dan on Claude's recommendation ([FU-130](tracker/open.md)).
- **A project hub for ChaosEngineAI, generated from `CLAUDE.md` and `docs/`** ([FU-131](tracker/open.md)). Decided by Dan on Claude's recommendation.
- **LTX-2.5 licence accepted on Hugging Face** by Dan; Claude adds the engine and catalog support.
- **Re-tag `v0.9.9` is held** until PR 119 merges. Decided by Dan ("hold pushes until tested").

</details>
