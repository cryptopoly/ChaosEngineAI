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
4. **Other apps' names that remain after the import-panel rename.** The import panel now labels groups by folder and the two string keys are gone ([FU-119](tracker/open.md)). Still naming other apps: the import API field names (`ollama`, `lmstudio`) and code identifiers, and the Server tab label "Ollama-compatible apps" (it names an API protocol clients speak). **Recommendation:** keep the protocol label (users need it to connect their tools) but reword it as "Apps that speak the Ollama API"; rename the API fields and identifiers in a later cleanup. **Why:** the editorial rule is about comparing against other products, not about naming a wire protocol; renaming API fields is churn with no user benefit.

5. **Speculative decoding slows creative prose 2 to 4 times** ([FU-134](tracker/open.md)). Found by the 27B-class bench: code and reasoning gain up to 1.7 times, prose loses. **Recommendation:** add a launch-modal note now, an acceptance gate (fall back to plain steps when acceptance is low) in our own DDTree loop next, upstream's adaptive block with the dflash-mlx migration later. **Why:** a note is free and honest; the gate is about a day and measurable.


### P3: later

6. **oMLX as an MLX engine lane** ([FU-132](tracker/open.md)). An Apache-2.0 MLX server with continuous batching and an SSD-persistent KV cache, over HTTP only. **Explainer:** it already works as a remote provider (Settings, OpenAI-compatible base URL) with no code; a real lane means another MLX stack in its own venv and it cannot use our DFlash or TurboQuant. **Recommendation:** do not build a lane yet; measure it against our MLX lane first (needs your yes to install it in an isolated venv, about 45 minutes). **Why:** our MLX path already reuses prompt caches; the SSD cache and concurrent requests are the only clear gains, and they matter more for agents and the API server than for one chat.
7. **"Best fit for your hardware" badges and recommended settings** ([FU-133](tracker/open.md)). Fit badge on every model row (fits well, tight, will not fit, plus a best-fit pick by newest-that-fits) and a "recommended for this Mac" row of settings in Image and Video Studio (TeaCache, threshold, CFG decay, steps) with an Apply button. **Recommendation:** build after v0.9.9 with the Models page merge below; the badge first (about 1.5 days, the data exists), settings second (about 2 days, needs speed and quality measurements on 3 to 4 families). **Why:** only the M4 Max can be measured here, so other hardware tiers must be labelled as estimates.
8. **Merge Discover and My Models into one Models page per area** ([FU-129](tracker/open.md)). **Recommendation:** after v0.9.9, image and video first, redirects for the old routes, 2 to 4 days. **Why:** the screens are nearly identical, but it touches about 4,000 lines and the release should not wait for it.
9. **Local agents building a game through the app** ([FU-126](tracker/open.md)). Banked by you; revisit when a local coding model clearly matches a hosted frontier model. No action now.

<details>
<summary>Decision log (made decisions, newest section last)</summary>

## 2026-10-28 and 2026-10-29

- **DFlash on hybrid Qwen models: fixed, not switched off.** The cause was found the same day (the adapter read the output head from the wrong object under mlx-lm 0.32, [FU-135](tracker/open.md)); a load-time self-check now protects the lane. Claude's call, within the "no null results" brief.
- **Prefer newer and faster models when choosing defaults.** Your rule of thumb. Saved to memory and used for the LTX and Wan choices.
- **Wan 2.2 TI2V 5B is replaced by LTX-2.3 on its own MLX engine** (`ltx-2-mlx`), on Claude's recommendation. The 5B family and its caches are removed; Wan 2.2 A14B and Wan 2.1 rows stay. See [FU-128](tracker/closed.md).
- **Agent game-dev idea is banked** ([FU-126](tracker/open.md)). Decided by Dan.
- **Linear DFlash block size 8 on MLX** (was 16). Decided by Dan on Claude's recommendation; checked on Qwen3-4B, to be re-checked on a 27B target.
- **Sound in Video Studio: no toggle; show the sound** (icon, unmuted result window, prompt hint). Decided by Dan on Claude's recommendation ([FU-130](tracker/open.md)).
- **A project hub for ChaosEngineAI, generated from `CLAUDE.md` and `docs/`** ([FU-131](tracker/open.md)). Decided by Dan on Claude's recommendation.
- **LTX-2.5 licence accepted on Hugging Face** by Dan; Claude adds the engine and catalog support.
- **Re-tag `v0.9.9` is held** until PR 119 merges. Decided by Dan ("hold pushes until tested").

</details>
