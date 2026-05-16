# Upstream Research — 2026-05-16

Investigation into recent activity on spec-decoding and KV cache compression
upstreams that affect ChaosEngineAI. Sources cited inline; no claims taken
on faith.

---

## 1. Executive Summary

- **llama.cpp PR #22673 (MTP support) merged today, 2026-05-16T12:06:24Z** —
  merge commit `2555826`, author am17an. Adds `--spec-type draft-mtp
  --spec-draft-n-max N`. Tested upstream on Qwen3.6-27B and Qwen3.6-35B-A3B;
  works on any model with baked-in MTP heads. Acceptance ~72%, ~2× tok/s.
  Source: <https://github.com/ggml-org/llama.cpp/pull/22673>.
- **Canonical MTP GGUFs published** under the `ggml-org/` org:
  `ggml-org/Qwen3.6-27B-MTP-GGUF` (BF16 + Q8\_0 + mmproj) and
  `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF` (BF16 + Q8\_0 + mmproj). Both verified
  HTTP 200 + sibling list via HF API.
- **FU-028's blocker on the GGUF side is now resolved.** Our existing MLX
  path (MTPLX subprocess engine in `mtplx_engine.py` + `MTP_MODEL_MAP` in
  `_mtp.py`) was already shipping; the GGUF lane was deferred waiting on
  exactly this PR.
- **`turboquant-mlx-full` is still at 0.3.0** on PyPI (`pip index versions`
  confirmed). No new release since FU-001 closed. Upstream `manjunathshiva/turboquant-mlx`
  HEAD is `5644286` (2026-05-03), all CI/license/dependabot — no functional
  changes since the 0.3.0 tag we already consume.
- **`TheTom/turboquant_plus` is alive and growing fast** — 6802 stars,
  908 forks, last push 2026-05-09. **Not on PyPI**, distributed via `pip
  install -e .` from the repo. Still the experimental research workspace
  flagged in FU-032; trip-wires from that row are now partially met (stars,
  forks, multi-hardware validation).
- **The "TurboQuant dev solved TriAttention" claim resolves to interpretation
  (a)** — Tom Turney has built a **C++ TriAttention V3 hybrid policy
  inside the `TheTom/llama-cpp-turboquant` fork** (branch
  `experiment/triattention-integration`), not a new MLX implementation.
  V3 = paper-faithful trig scoring + 128-token prefix protect + per-segment
  eviction quota. On Qwen2.5-7B 32K it claims ~baseline PPL + needle-pass at
  start/middle/end. Source: `docs/papers/triattention-v3.md` in the
  turboquant_plus repo.
- **WeianMao/triattention upstream HEAD is unchanged** since our 2026-04-23
  pin (`c3744ee6`). Latest tag remains `v0.2.0` (SGLang + LongLive + multi-
  backend). No new MLX work since FU-031 pinned this commit. Our FU-002
  shipping wiring is current.
- **The X tweet at `leftcurvedev_/status/2055652467027628472` is gated
  behind X's auth wall (HTTP 402)** — unable to verify content. Treating the
  technical PRs above as the source of truth, per task instructions.

---

## 2. llama.cpp MTP — Merge Status, Flag Surface, Action Plan

### 2.1 Merge facts (verified via `gh pr view 22673 -R ggml-org/llama.cpp`)

| Field | Value |
|---|---|
| PR number | #22673 |
| Title | "llama + spec: MTP Support" |
| State | MERGED |
| Merged at | 2026-05-16T12:06:24Z |
| Merge commit | `255582687b8dd211fdbc582e43ab842491554e94` |
| Author | am17an (Aman Gupta) |
| Minimum llama.cpp version | tagged `master-fff0e0e` (current HEAD) — any
build pulled today |

### 2.2 Flag surface (from PR body, verbatim)

```bash
# Single-model MTP (the recommended default)
llama-server -hf [model-with-mtp] --spec-type draft-mtp --spec-draft-n-max 2

# Optional: disable vision projector to save memory
llama-server ... --no-mmproj

# Combinable with ngram speculative decoding (advanced; non-CUDA only)
llama-server -hf [model-with-mtp] \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --spec-type ngram-mod --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64

# Or just enable both via the default preset
llama-server -hf [model-with-mtp] --spec-default
```

The flag is **`--spec-type draft-mtp`** (not `--spec-type mtp` as the
original FU-028 row guessed). `--spec-draft-n-max N` controls how many
draft tokens per step; upstream-reported sweet spot is `N=2` or `N=3`.

### 2.3 Supported models (PR-cited GGUFs that exist today)

Both verified HTTP 200 + siblings inspected via the HF API:

| Repo | Files |
|---|---|
| `ggml-org/Qwen3.6-27B-MTP-GGUF` | `Qwen3.6-27B-MTP-BF16.gguf`, `Qwen3.6-27B-MTP-Q8_0.gguf`, `mmproj-Qwen3.6-27B-Q8_0.gguf` |
| `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF` | `Qwen3.6-35B-A3B-MTP-BF16.gguf`, `Qwen3.6-35B-A3B-MTP-Q8_0.gguf`, `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` |

The author's own `am17an/Qwen3.6-{27B,35BA3B}-MTP-GGUF` mirrors also exist
(verified HTTP 200) — these were the dev preview, the `ggml-org/` org is
the upstream canonical.

PR says MTP "in principle should work for any MTP model"; ChaosEngineAI's
existing `_mtp.py::MTP_MODEL_MAP` already enumerates the families
(Qwen3.5/3.6, DeepSeek V3/R1, Coder-Next, Youssofal optimised quants),
so the GGUF lane just needs the existing map gated on `gguf_format ==
True`.

### 2.4 Where it lands in our repo — diff sketch

**`backend_service/inference/llama_cpp_engine.py`** — `_build_command`
currently accepts `speculative_decoding` + `tree_budget` in
`load_model()` but does not emit any spec-dec flags into the
llama-server CLI. Extension lives in `_build_command`:

```python
# inside _build_command, after the cache_flags block, before mmproj:
from backend_service.inference._mtp import has_mtp_heads, get_mtp_draft_n

if speculative_decoding:
    repo_for_mtp = canonical_repo or runtime_target or path or ""
    # Detect canonical-MTP GGUF repos by name match against MTP_MODEL_MAP
    # canonicals + the new ggml-org/Qwen3.6-*-MTP-GGUF mirrors.
    if _is_mtp_gguf_repo(repo_for_mtp):
        n_max = get_mtp_draft_n(repo_for_mtp) or 2
        if _llama_server_supports(binary, "--spec-type"):
            command.extend(["--spec-type", "draft-mtp",
                            "--spec-draft-n-max", str(n_max)])
        else:
            runtime_note = (
                "MTP speculative decoding requires llama-server built from "
                "ggml-org/llama.cpp master ≥ 2026-05-16 (PR #22673). "
                "Using standard decode."
            )
```

The `_llama_server_supports` helper already exists for `--reasoning-format`
gating — same pattern.

Add a helper:

```python
def _is_mtp_gguf_repo(repo: str) -> bool:
    """True when *repo* points to a GGUF mirror with baked-in MTP heads."""
    # Direct hit on the canonical MTP GGUF repos
    _MTP_GGUF_REPOS = {
        "ggml-org/Qwen3.6-27B-MTP-GGUF",
        "ggml-org/Qwen3.6-35B-A3B-MTP-GGUF",
        "am17an/Qwen3.6-27B-MTP-GGUF",
        "am17an/Qwen3.6-35BA3B-MTP-GGUF",
    }
    if repo in _MTP_GGUF_REPOS:
        return True
    # Heuristic: any repo whose name contains "-MTP-GGUF" and whose
    # canonical alias is in MTP_MODEL_MAP.
    return "-MTP-GGUF" in repo and has_mtp_heads(_canonical_for_mtp_gguf(repo))
```

**`backend_service/inference/_mtp.py`** — extend `_MTP_ALIASES` with the
new GGUF repos so `get_mtp_draft_n("ggml-org/Qwen3.6-27B-MTP-GGUF")`
returns the right `N`:

```python
"ggml-org/Qwen3.6-27B-MTP-GGUF": "Qwen/Qwen3.6-27B",
"ggml-org/Qwen3.6-35B-A3B-MTP-GGUF": "Qwen/Qwen3.6-35B-A3B",
"am17an/Qwen3.6-27B-MTP-GGUF": "Qwen/Qwen3.6-27B",
"am17an/Qwen3.6-35BA3B-MTP-GGUF": "Qwen/Qwen3.6-35B-A3B",
```

**`backend_service/inference/capabilities.py`** — add an MTP-on-GGUF
capability flag. Today the capability resolver detects the standard +
turbo binaries and the MTPLX venv; the UI keys MTP-toggle visibility off
`mtplxAvailable`. With GGUF MTP shipping we need a separate
`ggufMtpAvailable` (detected by probing `--spec-type` in
`_llama_server_supports`) so the frontend can show the MTP affordance
for GGUF models too.

**`backend_service/catalog/text_models.py`** — add catalog entries for
`ggml-org/Qwen3.6-27B-MTP-GGUF` and `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF`
as new variants under the existing `qwen3.6` family. Sizes for the
Q8\_0 variants: ~29 GB (27B) and ~37 GB (35B-A3B MoE). Vision via the
mmproj sibling (already auto-detected by `_resolve_mmproj_path`).

**`tests/test_inference.py`** — new test cases for:
1. `_build_command` emits `--spec-type draft-mtp --spec-draft-n-max 2`
   when the canonical repo is in `MTP_MODEL_MAP` and the binary supports
   `--spec-type`.
2. Same call when binary lacks `--spec-type` produces a runtimeNote and
   no spec-dec flags.
3. `_is_mtp_gguf_repo` returns True for the 4 canonical repos and
   False for stock `lmstudio-community/Qwen3.6-27B-GGUF`.

**`scripts/stage-runtime.mjs`** — the `llama-server` we ship from
homebrew or the staged binary needs to be ≥ `master-fff0e0e` (today's
HEAD). Add an assert in the staging script that the bundled binary
emits `--spec-type` in its `--help` output; fail the build otherwise.
Mirror in `scripts/pre-build-check.sh`.

**`CLAUDE.md`** — flip FU-028 from "deferred" to "shipped (GGUF lane)";
the MLX-side via MTPLX was already partially shipped (the row body
should reflect both lanes are now active).

**E2E suite** — add a `gguf+mtp` cell to `scripts/cache-strategy-matrix.py`'s
spec-dec axis.

### 2.5 Known caveats (cited from PR body)

- "Prompt processing (PP) speed typically takes a negative hit when MTP is
  enabled mainly due to Device-To-Host (D2H) embedding transfers." — surface
  this in the runtimeNote shown to the user the first time they enable MTP.
- "Parallel decoding with MTP is supported, but not fully optimized yet." —
  our `--parallel 1` default is fine, no change needed.
- Compatible with vision input (so the mmproj wiring keeps working) and
  with tensor/pipeline parallelism (irrelevant for our single-GPU lanes).

---

## 3. TurboQuant — Status and Recommended Action

### 3.1 Evidence

- `pip index versions turboquant-mlx-full` → only `0.3.0` is published; this is
  the version we already pin (`pyproject.toml:43`).
- `manjunathshiva/turboquant-mlx` recent commits (last 10) are all
  Apache-2.0 relicense, NOTICE/CITATION files, Dependabot bumps, and CI for
  PyPI Trusted Publishing. No new kernels or APIs since `v0.3.0`
  (2026-05-03).
- `TheTom/turboquant_plus` is a research workspace with 6802 stars / 908
  forks. README claims V compression to 2 bits is free, "Boundary V" layer-
  adaptive trick recovers 37–91% of quality gap, and validates up to 104B
  Command-R+ at 128K on a 128 GB M5 Max. Not packaged for PyPI; consumption
  would be `pip install -e .` from a vendored fork.

### 3.2 Recommendation

**No version bump action.** `turboquant-mlx-full>=0.3.0` is current.

**`TheTom/turboquant_plus` does not yet meet FU-032's trip-wires** —
trip-wires require either a tagged v1.0 release on the repo or a public
benchmark beating `turboquant-mlx-full` 0.3.0 on a head-to-head MLX
workload. Neither has happened; the benchmarks in turboquant_plus are
all C++/Metal/llama.cpp side, not MLX. **Re-evaluate next quarter.**

**However**, the `experiment/turbo4-quality-investigation` branch on
`TheTom/llama-cpp-turboquant` shows a turbo4 (4-bit) PolarQuant kernel
landing with q8\_0-comparable PPL (+0.23%) at 3.8× compression. Our
shipped llama-server-turbo binary still pins `feature/turboquant-kv-cache`
at `5aeb2fdb` (2026-05-09). The turbo3/turbo4 cache types are already
exposed by that binary. **No pin change recommended** without a public
M-series benchmark comparing turbo3 vs turbo4 on our representative
models (Qwen3.6-27B + Qwen3.5-9B); flagging as a "watch closely" item.

---

## 4. TriAttention — Upstream Stance vs. TurboQuant+ Claim

### 4.1 WeianMao/triattention HEAD

Verified via `git ls-remote` + `gh api`:

- HEAD commit: `c3744ee6a50522a1559a577f85aef2b165a344f2` (2026-04-23)
- Latest tag: `v0.2.0` (2026-04-22) — same release as when FU-031 was
  pinned.
- Our pyproject pin is **identical to upstream HEAD**.

No new MLX work has landed since FU-002 shipped the MLX integration on
2026-05-03. The single recent commit since `v0.2.0` is a README polish
(2026-04-23). Our pin is current.

### 4.2 TurboQuant dev's claim

Tom Turney (TheTom) has built **TriAttention V3** — a *fork* of the
paper's algorithm with two structural additions:

1. **Hard prefix protection** — first 128 tokens never evictable.
2. **Per-segment eviction quota** — eviction spread evenly across K
   buckets rather than concentrating where trig score is lowest.

This lives in C++ in `TheTom/llama-cpp-turboquant` on branch
`experiment/triattention-integration` (HEAD `8ebbace3`,
2026-04-09 — note: predates the V3 paper write-up, so the branch may
contain the work without the full V3 documentation).

The paper reports V3 + TurboQuant+ stack on Qwen2.5-7B 32K: 0.84% PPL
hit vs f16, needle retrieval passes at start/middle/end. The paper-
faithful V1 implementation gave +1.2% PPL at only 10% savings + silent
needle drop near end-of-context — i.e. **TheTom's V3 is a fix for
TriAttention's paper-faithful version misbehaving in production**, not
a "TurboQuant subsumes TriAttention" story.

**Verifiability:** the paper is a single-author writeup published in
the turboquant_plus repo; benchmarks include the raw command lines and
PPL numbers but no replication outside the author's M5 Max box for the
V3-specific results. Strict interpretation: V3 is **plausible** and the
methodology is sound, but **not yet independently reproduced**. Strong
interpretation should wait for an outside replication on a different
hardware/model combination.

**Action:** none. WeianMao's MLX TriAttention pin is current and our
FU-002 shipping integration uses upstream's apply_triattention_mlx with
norm-only scoring on Apple Silicon. TheTom's V3 work is a separate C++
implementation against a different binary; adopting it would mean
bumping our `feature/turboquant-kv-cache` pin to merge in
`experiment/triattention-integration`, which is not what those branches
are for. **Watch closely** in case the V3 changes get upstreamed into
WeianMao's repo (the PR-via-issue thread is the canonical surface).

---

## 5. Recommended PR Sequence

Ordered by combined urgency + risk-reduction. All effort estimates
assume a single developer familiar with the codebase.

### PR 1 — Wire GGUF MTP spec-dec (`feat: GGUF MTP via llama.cpp #22673`)

- **Trigger:** PR #22673 merged today; canonical MTP GGUFs exist on HF.
- **Files:**
  - `backend_service/inference/llama_cpp_engine.py` (~40 LOC: helper
    `_is_mtp_gguf_repo`, `--spec-type draft-mtp` flag emission in
    `_build_command`, runtimeNote on unsupported binary, runtimeNote
    on PP slowdown caveat)
  - `backend_service/inference/_mtp.py` (~6 LOC: 4 new alias entries)
  - `backend_service/inference/capabilities.py` (~15 LOC:
    `ggufMtpAvailable` flag via `--spec-type` probe)
  - `backend_service/catalog/text_models.py` (~50 LOC: 2 new
    catalog variants under qwen3.6 family)
  - `tests/test_inference.py` (~80 LOC: 4 new test cases)
  - `scripts/pre-build-check.sh` + `scripts/pre-build-check.mjs`
    (~10 LOC: assert bundled llama-server has `--spec-type`)
  - `scripts/cache-strategy-matrix.py` (~5 LOC: add `gguf+mtp` cell)
  - `src/components/runtimeSupport.ts` + `src/components/RuntimeControls.tsx`
    (~30 LOC: surface MTP toggle for GGUF models when
    `ggufMtpAvailable && model is in MTP_MODEL_MAP`)
- **Test impact:** new tests for spec-dec flag emission; no regressions
  expected on existing tests since the new flags only emit when
  `speculative_decoding=True` and a canonical MTP repo is selected.
- **Effort:** 4–6 h.

### PR 2 — CLAUDE.md FU-028 status flip + new FU row if turbo4 worth tracking

- **Trigger:** completion of PR 1.
- **Files:** `CLAUDE.md` — strikethrough FU-028, note both MLX (MTPLX)
  and GGUF (PR #22673) lanes shipped. If we decide turbo4 is worth a
  watch-row, add a new FU-NNN for the M-series benchmark.
- **Effort:** 0.5 h.

### PR 3 — turbo4 M-series benchmark (optional, watch-closely)

- **Trigger:** opportunistic; no upstream change required.
- **Files:** `scripts/benchmark-turbo-types.py` (new ~120 LOC), output
  to `~/.chaosengine/test-results/turbo-types-YYYYMMDD.md`.
- **Outcome:** decide whether to default to turbo3 or turbo4 in our
  cache strategy presets. **No code change in `cache_compression/`
  expected** — the strategies already pass through to the binary's
  supported types.
- **Effort:** 2 h.

### PR 4 — TheTom TriAttention V3 monitoring (no code yet)

- **Trigger:** WeianMao/triattention upstream merges or tags a v0.3
  release that includes the V3 prefix-protect + per-segment-quota
  logic; OR a second-party replicates V3 numbers on non-Apple
  hardware.
- **Files:** none.
- **Effort:** 0.

---

## 6. Open Questions

Track these with upstream maintainers / public threads:

1. **PR #22673 caveats** — the "negative PP hit due to D2H transfers"
   note in the PR body has no follow-up issue. **Ask am17an** whether
   the PP regression is consistent across model sizes or only kicks in
   above some threshold (e.g. >32K context). Drives whether we should
   default-on MTP for Qwen3.6-27B or surface it as opt-in.
2. **Recurrent-state save/load with partial rollback** is in the merged
   PR but `Fix partial rollback for batch size > 1 + n_rs_seq` is on
   the post-merge TODO list. **Confirm via gh issue tracker** that
   single-parallel (`--parallel 1`, our default) is unaffected.
3. **TurboQuant+ V3 reproducibility** — would be valuable to have a
   second M-series box (community contributor?) re-run the Qwen2.5-7B
   32K PPL + needle suite from `docs/papers/triattention-v3.md`. Not
   blocking any PR; informs whether we follow on the C++ TriAttention
   side at all.
4. **`turboquant_plus` v1.0 tag** — FU-032's adoption trigger.
   Subscribe to upstream releases. The repo has the activity to suggest
   a release tag is plausible in the next quarter.
5. **`ggml-org/` org owns the MTP GGUF mirrors** — these are not just
   community uploads, they are the official mirrors. Whether they're
   committed to maintaining future MTP GGUFs (Qwen3.7, DeepSeek-V4)
   would inform whether our catalog ships against `ggml-org/` or the
   model author's namespace.

---

## 7. Sources

- llama.cpp PR #22673: <https://github.com/ggml-org/llama.cpp/pull/22673>
  (verified via `gh pr view 22673 -R ggml-org/llama.cpp`).
- Merge commit: `255582687b8dd211fdbc582e43ab842491554e94`
  (verified via `gh api repos/ggml-org/llama.cpp/commits/...`).
- MTP GGUF mirrors: `ggml-org/Qwen3.6-27B-MTP-GGUF`,
  `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF` (verified HTTP 200 + HF API
  siblings).
- TurboQuant-MLX PyPI: `pip index versions turboquant-mlx-full` →
  0.3.0 (no newer release).
- TurboQuant+ workspace: <https://github.com/TheTom/turboquant_plus>
  (HEAD `1224fef3`, 6802 stars, 2026-05-09 push).
- TriAttention V3 paper:
  <https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/triattention-v3.md>
  (single-author, M5 Max only, not externally reproduced).
- TriAttention upstream: <https://github.com/WeianMao/triattention>
  (HEAD `c3744ee6` = our pin, 2026-04-23, v0.2.0 released 2026-04-22).
- `leftcurvedev_/status/2055652467027628472` — unable to verify
  (HTTP 402 on WebFetch, X auth-walled).
