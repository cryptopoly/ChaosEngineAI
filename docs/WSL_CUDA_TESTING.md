# WSL + CUDA Testing Plan

Live-validated 2026-05-18 against WSL2 Ubuntu 24.04 + RTX 4090. All
phases ran end-to-end; the gotchas section at the bottom captures the
non-obvious bits that surfaced during the dry run so the next operator
doesn't trip on them.

This document describes how to exercise ChaosEngineAI's CUDA inference
lanes from a Windows host using WSL2 — useful both for validating the
Linux build on a Windows dev box and for catching regressions in code
paths that never run on Apple Silicon (vLLM, Nunchaku, SageAttention,
DFlash CUDA, TriAttention, FP8 layerwise, stable-diffusion.cpp with
cuBLAS).

The plan is layered: each phase is independently runnable and reports
its own pass/fail so you can stop early if a lower layer breaks. The
Windows host is the dev box; WSL holds the Linux test surface; the
NVIDIA driver lives on Windows and is shared with WSL via the
`/dev/dxg` GPU-passthrough device.

## Why this matters

The standard pytest sweep already runs cleanly on WSL CPU once the
Apple-Silicon-only and Windows-only test bugs are fixed (see the
2026-05-18 cross-platform pass — 1510/3/1 pre-fixes, 0/0/N skips after).
What it does **not** exercise:

| Lane | Apple Silicon (dev box) | Windows native | WSL CPU | WSL + CUDA |
|---|---|---|---|---|
| MLX / mlx-lm / mlx-vlm | ✅ | — | — | — |
| MTPLX subprocess engine | ✅ (POSIX) | skip | ✅ (POSIX) | ✅ (POSIX) |
| llama.cpp + cuBLAS | — | partial | — | **target** |
| vLLM | — | — | — | **target** |
| TriAttention (vLLM mode) | — | — | — | **target** |
| Nunchaku CUDA SVDQuant (FU-023) | — | — | — | **target** |
| SageAttention (FU-016) | — | — | — | **target** |
| FP8 layerwise (FU-024) | — | — | — | **target** |
| DFlash CUDA + kvpress (FU-027) | — | — | — | **target** |
| stable-diffusion.cpp + cuBLAS | — | — | — | **target** |
| Cache-strategy matrix CUDA cells | — | — | — | **target** |

Without a WSL + CUDA path, the eight "target" rows have no automated
coverage on a Windows dev box. Linux users (and Windows users running
the Linux build under WSL) are flying blind on those lanes until either
a Linux CI box exists or this plan runs.

## Preconditions

Already satisfied on this machine (verified 2026-05-18):

| Requirement | Status | Where |
|---|---|---|
| Windows 11 22H2 or later (WSL2 GPU passthrough) | ✅ Build 26200 | host |
| NVIDIA driver ≥ 535 | ✅ 591.86 | host (visible via `nvidia-smi.exe`) |
| WSL2 + Ubuntu distro | ✅ Ubuntu 24.04 | `wsl --list -v` |
| Python 3.10+ inside WSL | ✅ 3.12.3 | `wsl -- python3 --version` |
| CUDA Toolkit 12.x inside WSL | ✅ 12.6.85 (nvcc) | `/usr/local/cuda/bin/nvcc --version` |
| libcudart / libcublas / libcuda passthrough | ✅ | `ldconfig -p | grep libcu` |
| Build tools (gcc 13 / cmake 3.28+) | ✅ | `gcc --version`, `cmake --version` |
| ≥ 100 GB free in WSL filesystem | ✅ ~920 GB | `df -h ~` |

Not yet satisfied — install in Phase A:
- `nvcc` not on `PATH` (only at `/usr/local/cuda/bin/nvcc`)
- CUDA pip wheels (`vllm`, `bitsandbytes`, `sageattention`, `nunchaku`)
- llama.cpp Linux build with cuBLAS
- stable-diffusion.cpp Linux build with cuBLAS
- Node.js + `npm install` for the i18n + vitest probes

## Phase A — Environment bootstrap

One-shot script, idempotent (safe to rerun). All commands run inside
WSL Ubuntu (`wsl -d Ubuntu-24.04 -- bash -lc '<cmd>'`).

```bash
# A1. Add nvcc to PATH for the user
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# A2. Sync the repo from the Windows side (preserve existing venv).
#     The .tmp-py312-deps/ exclude avoids a Windows ACL leak that
#     surfaced live on 2026-05-18.
rsync -a \
  --exclude='.venv' --exclude='node_modules' \
  --exclude='.chaosengine' --exclude='.tmp-py312-deps' \
  /mnt/c/Users/Dan/ChaosEngineAI/ ~/ChaosEngineAI/

# A3. Python venv + base extras (~150 MB, ~30s)
cd ~/ChaosEngineAI
test -d .venv || python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev,desktop,images,diffusion-accel]'

# A4. CUDA wheels (heavier — ~5 GB, ~5 min)
#     vllm pulls a compatible torch (2.11.0+cu130 as of 2026-05-18 with
#     vllm 0.21.0) so we DON'T pin torch first — let vllm drive the
#     resolution. bitsandbytes/sageattention/nunchaku build against
#     whichever torch vllm landed on.
.venv/bin/pip install vllm
.venv/bin/pip install 'nunchaku>=0.16.0' sageattention bitsandbytes
# ALSO: flashinfer (transitive dep of vllm) JIT-compiles a kernel via
# ninja on first sample. ninja must be on PATH. The pip wheel installs
# the binary at .venv/bin/ninja but that's not on PATH unless the venv
# is activated — see Phase D startup env for the workaround.
.venv/bin/pip install ninja
.venv/bin/pip install -e '.[triattention]'  # vendored, builds against vllm's torch

# A5. Node toolchain — for vitest + i18n + pre-build gate.
#     fnm avoids the sudo apt path that fails on unattended hosts; it
#     installs a single static binary under ~/.local/share/fnm and
#     drops Node 20 in user-space.
if ! command -v fnm > /dev/null 2>&1; then
  curl -fsSL https://fnm.vercel.app/install | bash -s -- --skip-shell
fi
export FNM_DIR="$HOME/.local/share/fnm"
export PATH="$FNM_DIR:$PATH"
eval "$($FNM_DIR/fnm env --shell bash)"
fnm install 20 && fnm use 20
cd ~/ChaosEngineAI && npm install --no-audit --no-fund

# A6. CRLF strip — rsync from /mnt/c brings Windows line endings on
# every text file. The `#!/usr/bin/env python3\r` shebangs break with
# `No such file or directory` on Linux. Use tr (not sed) — sed's
# escaping pitfalls cost an hour of debugging last time.
find scripts tests backend_service -type f \( -name "*.py" -o -name "*.sh" -o -name "chaosengine-cli" \) \
  | while read -r f; do
      if file "$f" 2>/dev/null | grep -q CRLF; then
        tr -d '\r' < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
      fi
    done
```

**Validation gate**: after A5, all of these should return non-empty:

```bash
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
# → True NVIDIA GeForce RTX 4090

.venv/bin/python -c 'import vllm; print(vllm.__version__)'
.venv/bin/python -c 'import sageattention'
.venv/bin/python -c 'import nunchaku'
```

If any fail, stop and resolve before moving to Phase B — every later
phase assumes the wheels imported cleanly.

## Phase B — Build the CUDA binaries

`llama-server` and `sd-cli` need CUDA-aware builds on Linux. Both build
scripts already exist in the repo and detect WSL automatically.

```bash
# B1. llama.cpp with cuBLAS (~10 min on RTX 4090)
cd ~/llama.cpp 2>/dev/null || git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp && git pull
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j$(nproc)
mkdir -p ~/.chaosengine/bin
cp build/bin/llama-server ~/.chaosengine/bin/
cp build/bin/llama-cli ~/.chaosengine/bin/

# B2. llama-server-turbo (TurboQuant fork)
cd ~/ChaosEngineAI
LLAMA_TURBO_DIR=~/llama-cpp-turboquant \
  LLAMA_TURBO_BRANCH=feature/turboquant-kv-cache \
  bash scripts/build-llama-turbo.sh

# B3. stable-diffusion.cpp with cuBLAS (~15 min)
cd ~/ChaosEngineAI && bash scripts/build-sdcpp.sh
# (script detects WSL CUDA and passes -DSD_CUBLAS=ON automatically)
```

**Validation gate**:
```bash
~/.chaosengine/bin/llama-server --version
~/.chaosengine/bin/llama-server-turbo --version
~/.chaosengine/bin/sd --help | head -5
```

## Phase C — Repeat the Python test sweep

With CUDA + all extras installed, the test surface widens — cells that
previously skipped on "vllm not installed" now run.

```bash
cd ~/ChaosEngineAI
.venv/bin/python -m pytest tests/ -q --tb=line
```

Expected delta vs the pure-CPU WSL run (2026-05-18):
- ~5–10 cells move from `skip` to `pass` (accelerator capability probes)
- 0 new failures — if any appear, they're CUDA-specific bugs worth filing

The TriAttention adapter + Nunchaku transformer + SageAttention helpers
import lazily at runtime, so the unit tests stay fast even when CUDA is
present.

## Phase D — Backend E2E sweep against WSL CUDA

The Windows desktop app holds port 8876 on the host. WSL needs its own
backend on a different port so the two coexist.

**Critical**: the backend MUST be started with three things set right
or the test scripts won't authenticate / find tools:
- `CHAOSENGINE_REQUIRE_AUTH=0` — the headless e2e/matrix scripts don't
  carry a Bearer token; without this they get 401 on every call.
- `.venv/bin` on PATH — flashinfer JIT-compiles via `subprocess.run(["ninja", ...])`
  and won't find ninja if only the venv's Python is referenced.
- `CHAOSENGINE_LLAMA_SERVER` pointing at the cuBLAS llama-server you
  built in Phase B1.

```bash
# D1. Pick a small chat model so the model-dependent phases actually run.
#     Qwen3-0.6B is ~600 MB on disk, downloads in ~2 min via the
#     project's own huggingface_hub (now pinned).
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-0.6B')
"

# D2. Start the WSL backend (foreground in its own terminal/tmux pane).
cd ~/ChaosEngineAI
export PATH="$HOME/ChaosEngineAI/.venv/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CHAOSENGINE_LLAMA_SERVER="$HOME/.chaosengine/bin/llama-server"
export CHAOSENGINE_REQUIRE_AUTH=0
.venv/bin/python -m backend_service.app --port 8877

# D3. In a separate WSL shell, wait for ready then run the smoke.
#     The suite reads CHAOSENGINE_PORT (not CHAOSENGINE_E2E_PORT —
#     see scripts/e2e_test_suite.py:45). The matrix runner takes
#     `--port` as a CLI arg.
cd ~/ChaosEngineAI
until curl -fsS --max-time 1 http://127.0.0.1:8877/api/workspace > /dev/null; do sleep 1; done
CHAOSENGINE_PORT=8877 .venv/bin/python scripts/e2e_test_suite.py --smoke

# D4. Then the full sweep.
CHAOSENGINE_PORT=8877 .venv/bin/python scripts/e2e_test_suite.py
```

Reports land in `~/.chaosengine/test-results/e2e-<timestamp>.{json,md}`.

Expected outcome with a model present:
- Phase 0–7 all pass except where the suite explicitly requires a
  larger model (Wan2.2 for video, FLUX for image — both ~30 GB+)
- Phase 1 GGUF lane runs through `llama-server` cuBLAS — should be
  measurably faster than the Apple Silicon MLX path

## Phase E — Cross-strategy matrix (CUDA cells)

The matrix runner sweeps every cache × spec-dec × model combo. With
CUDA wheels in place, the previously-skipped cells now produce real
numbers:

```bash
cd ~/ChaosEngineAI
.venv/bin/python scripts/cache-strategy-matrix.py --port 8877 --quick

# Once smoke is green, run the full sweep (15-20 min):
.venv/bin/python scripts/cache-strategy-matrix.py --port 8877
```

Live result on a fresh M4 Windows + WSL2 + RTX 4090 (2026-05-18,
backend on :8877 with vllm wheel + Qwen3-0.6B staged):

| Cell | Result |
|---|---|
| native MLX (smoke) | SKIP — Apple-only |
| turboquant MLX | SKIP — `turboquant` strategy unavailable (need `[turboquant]` extra, Apple-only path) |
| triattention MLX | SKIP — strategy unavailable |
| dflash/ddtree spec-dec | SKIP — `[dflash]` extra not installed |
| mtplx spec-dec | SKIP — MTPLX runtime needs separate venv |
| native GGUF | SKIP — model not in library (catalog expects `lmstudio-community/Qwen3-0.6B-GGUF`) |
| turboquant GGUF | SKIP — turbo binary missing (Phase B2 covers, optional) |
| gguf MTP | SKIP — `ggml-org/Qwen3.6-27B-MTP-GGUF` not in library |
| **vllm native (Qwen3-0.6B)** | **PASS** — SHA `d18c2b8cb410`, runtimeNote=`Applied Native f16 vLLM patches` |
| vllm turboquant | SKIP — strategy unavailable (no CUDA TurboQuant adapter shipped today) |
| vllm triattention | SKIP — strategy unavailable (TriAttention CUDA needs `[triattention]` extra w/ vllm >=0.21) |
| vllm dflash | SKIP — `[dflash]` extra not installed |

The vLLM native cell is the canary — once that's green, the CUDA path
(driver → libcuda → vllm → backend) is healthy end-to-end. The other
CUDA cells require additional `pip install -e .[extra]` invocations
(see Phase A4 for the list).

CUDA-specific cells to watch:
- `vllm native (Qwen3-0.6B)` — baseline throughput on vLLM
- `vllm turboquant (Qwen3-0.6B)` — CUDA TurboQuant variant
- `vllm triattention (Qwen3-0.6B)` — TriAttention through vLLM
- `vllm dflash (Qwen3.5-4B)` — speculative decoding via dflash-cuda

The runner asserts the FU-030 legacy-id coercion (`chaosengine` /
`rotorquant` → `turboquant`). Any regression there exits code 2.

## Phase F — Pre-build gate

Same gate the macOS dev workflow runs. On WSL the npm-side checks
also exercise the i18n locale catalogs + vitest + tsc.

```bash
cd ~/ChaosEngineAI
bash scripts/pre-build-check.sh
# or
node scripts/pre-build-check.mjs
```

Expected:
- All [1/N] Python tests pass (post-fix baseline)
- All [N/N] frontend tests pass (once `npm install` ran in Phase A5)
- i18n 100% across 10 locales
- WARN for `llama-server-turbo update available` is fine — it's
  informational, not a build blocker

## Phase G — Smoke a real CUDA workload

Once everything passes, exercise one workload end-to-end through the UI
or via curl to confirm CUDA actually fires:

```bash
# Load a 0.6B Qwen on the WSL backend
curl -X POST http://127.0.0.1:8877/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"modelRef": "Qwen/Qwen3-0.6B", "backend": "vllm",
       "cacheStrategy": "native", "contextTokens": 4096}'

# Stream a chat turn
curl -N -X POST http://127.0.0.1:8877/api/chat/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"sessionId": "smoke", "message": "Write one short sentence.",
       "maxTokens": 64}'

# Watch GPU utilisation in another pane
nvidia-smi -l 1
```

If `nvidia-smi` shows the Python process consuming VRAM during the
stream, the CUDA path is alive end-to-end. If it stays at 0%, the
backend silently fell back to the CPU lane — check the runtime notes
in `/api/diagnostics/snapshot`.

## Time + disk budget

| Phase | Time | Disk | One-time? |
|---|---|---|---|
| A. Bootstrap | ~10 min | ~6 GB | yes |
| B. CUDA binaries | ~30 min | ~3 GB | yes |
| C. Python sweep | ~3 min | — | every test |
| D. E2E full | ~5 min | ~1 GB (model) | every test |
| E. Matrix --full | ~20 min | varies (per-model) | every release |
| F. Pre-build gate | ~5 min | — | every release |
| G. Smoke workload | ~2 min | — | sanity check |

**Total cold path**: ~75 min for one full run from scratch.
**Warm path** (re-run after rsync + venv unchanged): ~35 min.

## Trip-wires — when to add a new phase

Add a new phase here whenever a CUDA-only feature lands without
corresponding coverage:

- New cache strategy that only exists on CUDA → matrix runner row
- New CUDA-only accelerator (e.g. SageAttention v2, new Nunchaku
  wrapper) → Phase A install step + Phase C import gate
- New CUDA-only model family in the catalog → Phase D test model
- New `[xxx]` extra in `pyproject.toml` that pulls a CUDA wheel →
  Phase A install step

Avoid duplicating Apple-Silicon-only checks here — those live in the
Mac dev workflow.

## Known gotchas

These all surfaced live on the 2026-05-18 dry run — keep this list
current as a tripwire for the next operator.

1. **WSL2 default disk size is 1 TB sparse, max 1 TB**. If you stage
   FLUX + Wan + several quants, watch `df -h ~`. Expand via
   `wsl --shutdown` then edit `~/.wslconfig` if needed.
2. **`/mnt/c` is slow** (DrvFs through the 9P protocol). Always copy
   the repo to `~/ChaosEngineAI/` inside WSL — don't run pytest
   directly against `/mnt/c/...`.
3. **NVIDIA driver lives on Windows, not in WSL**. `apt-get install
   nvidia-driver-XXX` will break WSL2 GPU passthrough. Only ever
   update the driver via Windows Update or NVIDIA's Windows installer.
4. **WSL `localhost` is *not* the same as Windows `localhost`** (in
   WSL2 they share a NAT'd loopback per session). If a Windows-side
   curl can't reach the WSL backend on `:8877`, use the WSL IP from
   `ip addr show eth0`.
5. **First `import torch` after CUDA install takes ~15 s** while it
   inventories the device. Subsequent imports are sub-second. Don't
   mistake the cold-import for a hang.
6. **`vllm` wheels are huge** (~2 GB). Allow extra disk + a longer
   pip install timeout.
7. **Don't pre-pin torch before installing vllm.** As of vllm 0.21.0
   the resolver pulls a specific torch (2.11.0+cu130). If you pin
   torch first, you'll hit "Cannot install vllm because these package
   versions have conflicting dependencies." Let vllm pick the torch.
8. **CRLF stripping needs `tr -d '\r'`, not `sed`.** `sed 's/\r$//'`
   under WSL's bash-inside-PowerShell-inside-Claude-Code quote chain
   can lose its backslash and become `s/r$//` — which strips trailing
   `r` characters from every line. That ate the `r` off `import
   mlx_video_wan_installer`, breaking `test_mlx_video_wan_installer.py`
   with a cryptic `cannot import name 'mlx_video_wan_installe'`.
   `tr` has no regex pitfalls — use it.
9. **`CHAOSENGINE_REQUIRE_AUTH=0` is mandatory for headless scripts.**
   The default secure mode generates a random Bearer token at startup;
   `scripts/e2e_test_suite.py` and `scripts/cache-strategy-matrix.py`
   don't read it. Without the env override every call returns
   `HTTP 401 Unauthorized`. The Tauri desktop app injects the token
   via its own session machinery — that path doesn't apply here.
10. **`ninja` needs to be on PATH at backend launch time.** `vllm` ↔
    `flashinfer` JIT-compile a sampler kernel on first call and shell
    out to `ninja`. The pip-installed ninja lives at `.venv/bin/ninja`
    but that's not on PATH unless the venv is activated. Either
    activate the venv before launching the backend, or export
    `PATH="$HOME/ChaosEngineAI/.venv/bin:$PATH"` in the launch wrapper.
11. **`--backend auto` may pick MLX on safetensors models even when
    MLX isn't available**, returning a confusing "MLX backend
    requested but unavailable" 500. Workaround for now: pass
    `--backend vllm` explicitly when loading from a HF safetensors
    repo on CUDA. (FU-063 candidate — the auto picker should fall
    through to vllm when MLX is gated out by platform.)
12. **fnm puts node in `~/.local/share/fnm/aliases/default/bin`** —
    not on the default PATH. Either source `fnm env --shell bash`
    in your shell rc, or use `npm` via its absolute path. The
    `pre-build-check.sh` script's i18n probe assumes `node` is on
    `PATH` and will FAIL with `node: command not found` otherwise
    (FU-064 candidate — make pre-build-check.sh aware of fnm).
13. **Matrix runner shows `0.0 tok/s` for vLLM cells** even when the
    cell PASSes with a real SHA hash. The pass/fail assertion works
    (compares decoded SHA-12 of generated text), but vLLM's metrics
    aren't piped back through the matrix runner's parser. The cell
    result is correctness-correct; treat the throughput as zero
    until that gap is closed.

## What we won't get from WSL CUDA testing

- True production wall-time numbers (DrvFs + 9P + GPU passthrough adds
  5–15 % overhead vs native Linux). Use for correctness; benchmark on
  bare metal.
- Multi-GPU paths — WSL2 GPU passthrough is single-device.
- Driver-level debugging (no `cuda-gdb` against the Windows driver).
