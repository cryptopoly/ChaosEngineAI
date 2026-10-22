# Progress: fix/ram-estimates (paused 2026-10-21)

Worktree `/Users/dan/ChaosEngineAI-ram`, branch `fix/ram-estimates`, 47 commits ahead of `origin/staging`. Nothing pushed.

## Done and verified
- RAM / speed estimates rebuilt (FU-110), release constraints, mflux 0.23, new catalog rows, TensorFold 0.6.6, llama.cpp b11786, DeepSeek-V4 row removal.
- mlx 0.32 stack for releases (FU-113), MLX DFlash repair and DDTree withdrawal (FU-112), chat-path fixes (FU-114), overlay guard, voice fixes.
- Full pytest green on both stacks (mlx 0.31 dev `.venv`, mlx 0.32 scratch venv); vitest 529, tsc, i18n clean; E2E smoke green on both; quick cache-strategy matrix green.

## Next, in order
1. Big-model support beyond 64 GB Macs: DGX Spark, Windows / Linux CUDA, A100-class. Add rows with correct sizes where a runtime can load them (Kolibri Q4/Q8 GGUF, Qwen-Image-2.1 diffusers lane on diffusers 0.41, vLLM rows), and check estimates / gates do not refuse large-memory hosts.
2. Done 2026-10-22 (FU-116): vlm-only types route to mlx-vlm 0.7.10.
3. Ask the user, then push, open the PR into staging, merge, re-tag v0.9.8 and rebuild the draft release.
4. After merge: upgrade the dev `.venv` with `.venv/bin/pip install -c release-constraints.txt mlx mlx-lm mlx-vlm transformers diffusers`.

## Needs the user
- Accept the FLUX 3 Image Klein licence on Hugging Face (not run here).
- Discard the stale copies of `capabilities.py`, `controller.py`, `ModelLaunchModal.tsx` in the main checkout before pulling.
- Settings has remote access on with API auth off.

## Test scaffolding (scratchpad, not in repo)
- New-stack venv: `<scratchpad>/venv32` (mlx 0.32.4 etc.). b11786 llama-server: `<scratchpad>/llama-b11786/llama-b11786/llama-server`.
- Backends: port 8890 (dev stack) and 8891 (new stack) with `CHAOSENGINE_EXTRAS_SITE_PACKAGES=/tmp/ce-no-extras CHAOSENGINE_REQUIRE_AUTH=0`; both stopped.
