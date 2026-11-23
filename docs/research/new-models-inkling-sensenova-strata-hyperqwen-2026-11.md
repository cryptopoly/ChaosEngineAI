# Inkling, SenseNova Loop, "Strata", HyperQwen: support check and plan (2026-11-23)

The owner presented four items at a conference as upcoming ChaosEngineAI features and asked whether we support them and, if not, what the work is on Windows, macOS and Linux. This page is research only: no app code was changed. Sources were read on 2026-11-23 (Hugging Face model cards, GitHub PRs and READMEs). The repo was checked with grep on branch `test/rtx4090`. Speed figures are the publishers' own unless marked "ours".

## Summary

| Item | What it is | Licence | Supported today? | Engine | Effort | Recommendation |
|---|---|---|---|---|---|---|
| **Inkling 2 Mini** (Thinking Machines, 2026-11-20) | 26.4B MoE, 3.4B active, looped middle layers, text + image, 1M context | Apache-2.0 (plus Thinking Machines' acceptable use policy) | **No.** No catalogue row; the bundled llama.cpp (b12441) has the `inkling2` arch but not the 2026-11 attention-mask fix | llama.cpp GGUF (all three platforms); vLLM on Blackwell later | **S to M**, about 1 day after the llama.cpp bump (FU-510) | **Do first.** Fits the 4090 (UD-Q4_K_M 16.6 GB) and a 32 GB Mac |
| **Inkling 2 Small** (2026-11-12 weights) | 594B MoE, 17B active | Apache-2.0 | No | llama.cpp GGUF (171.7 GB and up) | S (one row) | Add one row, gated by fit, marked untested (256 to 512 GB Macs only) |
| **Inkling / Inkling-Small** (first generation, 2026-07) | 975B / 266B MoE, text + image + audio | Apache-2.0 | No | llama.cpp only through draft PR #25731 (not merged); community MLX 2-bit builds | S once merged | Wait for upstream; Inkling 2 replaces these |
| **SenseNova-Loop-9B** (SenseTime, 2026-11-17) | 9B looped decoder (32 physical, 56 logical layers), text + image, 512K context | Apache-2.0 | **No.** Upstream llama.cpp cannot load it yet | llama.cpp after PR #32588 merges (opened 2026-11-19); vLLM and SGLang only on branches | **S** after merge (half a day); M if we ship a fork (not recommended) | **Wait for the upstream merge**, then add 3 GGUF rows on all platforms |
| **"Strata"** (most likely `Niko1221/Strata`, about 75% confidence) | MIT-licensed local engine that runs the 125B Qwen3.8-Flash-Next MoE on a 12 GB GPU + 64 GB RAM by keeping hot experts on the GPU and the rest in RAM | MIT | **The model: partly** (rows need 111 GB or more of GPU or unified memory). **The trick: no** (our llama.cpp engine has no MoE expert offload) | Our own llama.cpp engine with `--n-cpu-moe` | **M**, 2 to 3 days | **Build the capability ourselves** (expert offload + smaller quant rows); do not integrate the engine |
| **HyperQwen** (`syv-ai/HyperQwen`) | A vLLM 0.33.0 patch series and requantisation pipeline for Qwen3.8-27B: MTP + DFlash2 drafting, prompt drafting, int8 kernels tuned for RTX 3090 | Apache-2.0 | **The model: yes** (Qwen3.8-27B on every lane, DFlash2 drafter wired). **HyperQwen itself: no** | vLLM in Docker (Linux; Windows only through WSL2); no Mac | S to measure (1 day); L to integrate | **Measure it against ours on the 4090; borrow prompt drafting; do not bundle** |

The headline "1,035 tok/s" for HyperQwen is the total over 64 simultaneous requests on one RTX 3090, not one chat. One chat runs at 127 tok/s, or 381 tok/s when the reply copies text from a long prompt. Ours on the 4090 is 49.6 tok/s plain and 132 to 206 tok/s with TensorFold drafting (`docs/research/windows-model-landscape-2026-11-19.md`). So the single-chat claim is already within reach of what ships.

## Do we support them today (repo evidence)

`git grep -niE "inkling|sensenova|strata|hyperqwen"` over `*.py *.ts *.md *.mjs *.sh` finds:

- `CLAUDE.md`, the upstream table: "b12441 adds Inkling 2". This is release-note text for the llama.cpp pin. No catalogue row uses it.
- `docs/progress/v0100-bump-llama.md`: the same pin note.
- `docs/research/minimax-h3-and-new-video-image-2026-11-10.md`: `sensenova_u1`, an image model, not Loop.
- `src/utils/platform.ts`: false match.

Nothing in `backend_service/catalog/`, `backend_service/inference/` (TensorFold family rows included), `cache_compression/` or `dflash/` names any of the four.

What we do have:

- **Qwen3.8-27B:** `backend_service/catalog/text_models.py` family `qwen-3-8` has MLX, TensorFold, MTPLX and GGUF rows. `dflash/__init__.py` `DFLASH2_DRAFT_MODEL_MAP` holds `z-lab/Qwen3.8-27B-DFlash2`, used by vLLM and the WSL bridge (`allow_dflash2=True`). `helpers/text_gguf.py` pairs the GGUF row with the DFlash2 GGUF drafter. `_tensorfold.py` and `_tensorfold_native.py` carry Qwen3.8-27B + DFlash2.
- **Qwen3.8-Flash-Next:** the same family has `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (TensorFold, 113 GB, 128 GB Mac) and `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL (llama.cpp, 111 GB).
- **No MoE expert offload:** `grep -rniE "n-cpu-moe|cpu-moe|override-tensor"` over `backend_service` finds nothing.
- **Looped models:** `_tensorfold.py` already serves a looped 4B (Nemotron Labs Dory) through TensorFold. Neither the llama.cpp nor the MLX lane has a looped model.

## 1. Inkling 2 (Thinking Machines)

**What.** The second Inkling generation. Two checkpoints matter.

- **Inkling-2-Mini**, released 2026-11-20. 26.4B total parameters, 3.4B active. 24 layers; the middle 12 run three times with shared weights, giving 48 layer applications per token (about 5.4B effective active). MoE with 8 of 128 experts per token. Mixed local and global attention. Vision encoder. Up to 1M context. The official checkpoint is about 20 GB of mixed NVFP4 and MXFP8 safetensors. The card names Apple silicon with 32 GB or more, a DGX Spark, or 32 GB of VRAM. Engines named on the card: Transformers, vLLM, SGLang, llama.cpp. No MLX.
- **Inkling-2-Small**, weights released 2026-11-12. 594B total, 17B active, 48 layers, 8 of 384 experts, text + image, 1M context. Native precision is NVFP4, with an MXFP8 repo beside it. It needs 500 GB of VRAM or more (for example 2x B300).

**Licence.** Apache-2.0 on both cards. The Mini GGUF card also links Thinking Machines' Model Acceptable Use Policy. Read it before we feature the model; we do not bundle weights, the user downloads them.

**Formats.**

- GGUF, Mini: `unsloth/Inkling-2-Mini-GGUF`. UD-IQ1_S 7.8 GB, UD-Q2_K_XL 9.1, UD-Q3_K_XL 13.6, UD-IQ4_XS 15.7, UD-Q4_K_M 16.6, UD-Q4_K_XL 17.6, UD-Q8_K_XL 22.9 GB. The card says to use the files in its `Mainline_Support` folder for upstream llama.cpp. GGUF arch `inkling2`.
- GGUF, Small: `unsloth/Inkling-2-Small-GGUF`. UD-IQ1_S 171.7 GB up to BF16 484.4 GB; mmproj-F16 658 MB.
- MLX: none for Inkling 2. The first-generation Inkling has `mlx-community/Inkling-mlx-2bit` and `Inkling-Small-mlx-2bit`.

**Engines.**

- llama.cpp: PR #32217 "add Inkling 2 support (text + vision)" landed in b12406, so our b12441 has the arch. Its notes say "real checkpoint conversion remains untested". PR #32429 (b12549) fixes the Inkling 2 flash-attention mask and graph reuse. FU-510 already tracks the bump; the newest tag with prebuilt archives is b12576.
- Not verified: whether mainline handles Mini's looped layers. Mini was released after both PRs. The `Mainline_Support` folder suggests it does. **Load-test before shipping.**
- First-generation Inkling: llama.cpp PR #25731 is still a draft, so v1 does not run on upstream.
- vLLM and SGLang: day-0 support for Inkling 2 Small on Hopper and Blackwell. NVFP4 needs Blackwell for full speed; the 4090 (Ada) has no FP4 tensor cores.

**Speed.** No publisher figures for any variant. The Mini card only says speed for looped models "varies by runtime and hardware".

**Plan per platform.**

| Platform | Work |
|---|---|
| Windows (RTX 4090 CUDA) | 1. Bump llama.cpp to b12549 or later (FU-510; brings the mask fix). 2. New family `inkling-2` in `text_models.py` with Mini rows UD-Q4_K_M (16.6 GB, the default) and UD-Q3_K_XL (13.6 GB) for 16 GB cards, `backend: "llama.cpp"`, `ggufFile` taken from the `Mainline_Support` folder, mmproj if the folder has one, capabilities `reasoning, coding, agents, tool-use, vision`. 3. Run it on the 4090 and record tok/s. |
| macOS (Apple Silicon) | The same GGUF rows through Metal llama.cpp, gated to 32 GB or more. UD-Q8_K_XL (22.9 GB) as the quality row for 48 GB or more. No MLX path exists; watch mlx-lm / mlx-vlm for `inkling2` (FU-545). Run on the M4 Max. |
| Linux CUDA | The same GGUF rows. Later, vLLM serving of the official NVFP4/MXFP8 checkpoint on Blackwell (RTX 50, DGX Spark) through the vLLM lane (FU-546). |
| All | **Memory estimate:** `helpers/model_arch.py` sizes the KV cache from `block_count`. A looped model keeps its own KV for each logical layer (48 for Mini), so the estimate may undercount by 2x. Check the GGUF metadata and fix before the fit badge is trusted (FU-542). **Tests:** a catalogue test that the rows resolve, a fit test, and a `model_arch` test with looped metadata. **E2E:** a phase 1 (Chat) check that loads Mini when it is present locally and skips otherwise. |

**Inkling 2 Small.** One GGUF row, UD-IQ2_XXS (190.6 GB), gated by fit to 256 GB or larger unified memory and marked untested, following the "support the best models, not just the 4090's" rule (FU-543). The first-generation Inkling waits on PR #25731 (FU-544).

## 2. SenseNova-Loop-9B (SenseTime)

**What.** An open 9B multimodal model released 2026-11-17 (preview 2026-10-31). 32 physical layers; layers 4 to 27 run twice, giving 56 executions per token (about 14.6B "inference-equivalent" parameters). Two-stream manifold-constrained hyper-connections (mHC). Attention repeats three linear-attention layers then one full-attention layer. Text + image in, text out, 512K context. Publisher benchmarks put it near Qwen3.8-27B on agent tasks (NL2Repo 41.36 against 44.60; AIDABench 65.31 against 65.01).

**Licence.** Apache-2.0.

**Formats.**

- BF16 safetensors with custom modelling code: `transformers==5.18.0`, `trust_remote_code=True`.
- An official AWQ 4-bit (about 4.9 GB) with an MTP draft.
- Official GGUFs, `sensenova/SenseNova-Loop-9B-GGUF`: Q4_K_M 5.82 GB, Q5_K_M 6.67, Q8_0 9.8, BF16 18.4. They need the fork `OpenSenseNova/llama.cpp` at commit `24ea7bb`; no mmproj or MTP draft GGUF is listed yet.
- MLX: only community ports of the Preview (`netprojects/SenseNova-Loop-9B-Preview-MLX-4bit` / `-8bit`). One blog says stock mlx-lm loads them; not verified, and doubtful given mHC.

**Engines.**

- llama.cpp: issue #31571 is open. PR #32588 "model : add SenseNova Loop support" (arch `sensenova_loop`) was opened 2026-11-19 by the SenseNova team and is under review. It covers text + image (Qwen3-VL vision code) and optional MTP, with no new CUDA or Metal kernels. Users on the PR report 17 to 23 tok/s on an RTX 5060 Ti and about 16 tok/s on Vulkan, with a Vulkan output divergence still open.
- vLLM PR #64832 and SGLang PR #47281 are branches only.

**Speed.** No publisher figure. A community test reports about half the speed of a regular 9B, which follows from running 56 layers instead of 32.

**Plan per platform.** Nothing to build until PR #32588 merges, then (FU-547):

| Platform | Work |
|---|---|
| Windows (RTX 4090 CUDA) | Bump llama.cpp to the first tag with `sensenova_loop` (with FU-510 if it has not shipped yet). Family `sensenova-loop` with Q4_K_M (default), Q5_K_M and Q8_0 rows. Check that the official GGUFs (made with the fork) load in upstream; the arch name or metadata may differ, so we may need to point at a reconverted repo. Measure tok/s. |
| macOS | The same GGUF rows on Metal (the PR adds no Metal kernels, so measure; mHC ops may fall back). MLX: test the Preview ports in our MLX worker; add a row only if stock mlx-lm really loads the final model (FU-548). |
| Linux CUDA | The same GGUF rows. vLLM AWQ through the vLLM lane once PR #64832 merges (FU-549). |
| All | The same looped-KV estimate check as Inkling 2 Mini (FU-542): KV per logical layer, 56 here. The card gives about 10 GiB at 64K context with 8-bit KV, a good test value. Catalogue + fit tests; E2E phase 1 check that skips when absent. |

**Why not ship the fork now.** It is an out-of-tree llama.cpp build at another commit. It would need its own binary, staged per platform, beside both of ours, and users report it is slower than the PR branch (about 10 against 18 tok/s on CUDA). The upstream PR is authored by SenseTime and actively reviewed, so a merge is likely soon.

## 3. "Strata": which one the owner most likely means

**Most likely (about 75%): `Niko1221/Strata`,** an MIT-licensed local inference engine and server (about 47k GitHub stars; newest release 0.2.20 on the installer names; 0.2.16 on 2026-11-18 replaced its llama.cpp kernels with its own). It runs the 125B Qwen3.8-Flash-Next MoE on consumer hardware.

- **How:** all experts live in RAM and the GPU caches the most used ones; the CPU runs the rest. It uses MTP self-drafting and 8-bit / 4-bit KV.
- **Reference PC:** RTX 5070 12 GB, 64 GB DDR5.
- **Self-reported speed:** Q2_0 pack, about 87 tok/s output at 1K context and about 60 at 262K; 536 tok/s prompt at 1K.
- **Other models:** Qwen3.6-35B-A3B, Tencent Hy4-Flash and **Inkling 2 Mini**. That overlap with item 1 is why this reading is the most likely one.
- **Platforms:** CUDA, HIP, Metal and SYCL on Windows, Linux and macOS.
- It ships its own desktop installer and chat page. Under the CLAUDE.md editorial rule it counts as another desktop app, so it must not be named in code, UI strings or commit messages unless we ship it as a dependency (D-122).

**Alternatives.**

- (about 15%) *Strata: Hierarchical Context Caching for Long Context LLM Serving*, USENIX OSDI 2026, built on SGLang. A data-centre KV-cache technique, relevant only as an idea for our `cache_compression/` work.
- (about 5%) `ro99/strata`, a small C++/CUDA engine for models larger than VRAM.
- (about 5%) Something not public yet. No model family called Strata exists on Hugging Face.

**Supported today?** The model is in the catalogue, but only for machines with 111 GB or more of GPU or unified memory. The trick that makes it run on a gaming PC (expert offload) is **not** in our llama.cpp engine.

**Plan (build the capability, not the engine; FU-550):**

| Platform | Work |
|---|---|
| Windows (RTX 4090 + 64 GB RAM, this PC) | 1. Add MoE expert offload to `inference/llama_cpp_engine.py`: pass `--n-cpu-moe N` (or `-ot` patterns) when a MoE GGUF does not fit in VRAM; choose N from free VRAM after the attention and shared weights. 2. Teach the fit gate to count VRAM + system RAM for MoE rows. 3. Add a smaller Qwen3.8-Flash-Next row (an upstream-loadable 2-bit or IQ2 GGUF around 66 to 68 GB). Check that the packs the other engine uses (ISTA-DASLab "Q2_0" / GSQ) load in mainline llama.cpp; if they do not, use the `unsloth` UD-IQ2 / Q2_K_XL files. 4. Measure tok/s at 1K and 32K context with MTP on. |
| macOS | No offload needed (unified memory). The existing rows stay; a 2-bit row lets a 96 GB Mac run it. |
| Linux CUDA | The same offload code as Windows. |
| All | Unit tests for the `--n-cpu-moe` argument choice and the fit sum; E2E phase 1 check (skip when the GGUF is absent or RAM is under 64 GB). Effort M (2 to 3 days). The same code helps every large MoE row (Inkling 2 Small on a big-RAM PC, Qwen3.6-35B-A3B on 12 GB cards). |

## 4. HyperQwen

**What.** `syv-ai/HyperQwen` (about 2.8k stars, Apache-2.0, no tagged releases). A patch series on vLLM 0.33.0 plus a one-time requantisation of **Qwen3.8-27B** (about 20 GB written on first start). It has two modes, "single" (one or a few users) and "batch" (an API backend). Parts:

- MTP speculation plus a requantised DFlash2 drafter (40k-token draft vocabulary; 7 drafts per pass, 15 tokens verified per step).
- Drafting from the prompt itself.
- int8 Marlin GEMMs with tune tables measured on sm86.
- An int8-QK prefill kernel and int8 or fp8 KV.
- A ported KVarN 4/2-bit KV cache.

It is a speculative-decoding plus kernel stack for one model, not a model or a fine-tune.

**Claimed speed** (one RTX 3090 at 250 W, the repo's `bench/run_benchmarks.sh`):

- Single mode: 127 tok/s one stream; 381 tok/s when the answer quotes a 25K prompt (another write-up notes context drafts give 0.65% of accepted tokens on prose); prefill about 1,440 tok/s at 1K input.
- Batch mode: about 1,035 tok/s total at 64 concurrent requests (128 in / 512 out), but 46 tok/s for a single stream.
- The README says speculation wins below about 8 concurrent users and batching wins above.

**Hardware and platforms.**

- One 24 GB card or two smaller ones. sm86 is the most tested; a 4090 and a 5090 have community reports; sm80 has an open speculation fault.
- Linux through Docker + NVIDIA Container Toolkit or a bare venv. Windows only through Docker Desktop on WSL2. No Mac.

**Supported today?** The model and the main idea are already in the app:

- Qwen3.8-27B rows on MLX, TensorFold, MTPLX and GGUF.
- The DFlash2 drafter on llama.cpp (GGUF drafter pairing in `helpers/text_gguf.py`), vLLM and the WSL bridge (`dflash.get_draft_model(..., allow_dflash2=True)`), and TensorFold native.
- Our 4090 numbers: 49.6 tok/s plain, 132 to 206 tok/s with TensorFold drafting, 1.8 to 2.6x with llama.cpp DFlash2.

HyperQwen itself is not supported.

**Plan (FU-551).**

| Platform | Work |
|---|---|
| Windows (4090) | Measure first, about 1 day. Run HyperQwen single mode in WSL2 Docker on the 4090, using our `scripts/perf-baseline.py` prompts (code, prose, quote-the-prompt). Compare with TensorFold native and llama.cpp DFlash2 on the same prompts. If prompt drafting wins on agent and code-edit workloads, add it to our lanes: llama.cpp has `ngram-mod` in `--spec-type`, so check whether it can sit alongside `draft-dflash2`. Do not bundle HyperQwen: it is a vLLM fork pinned to one version and one model, tuned for sm86, Docker-only on Windows, and writes 20 GB of requantised weights. |
| macOS | Nothing to port (CUDA-only kernels). MTPLX and TensorFold already give 40 to 60 tok/s on Qwen3.8-27B on the M4 Max. |
| Linux CUDA | Same measurement. If batch throughput matters for the app's OpenAI-compatible server, the vLLM lane is the place; consider the KVarN KV cache as a `cache_compression/` strategy candidate (licence check needed). |

**Message for the owner.** If the conference said "1,000 tok/s", that is 64 users at once on a server, not one chat.

## Not verified

- Whether upstream llama.cpp (b12441 or b12576) runs Inkling 2 **Mini**'s looped layers and its vision projector. No run was made; PR #32217 predates Mini.
- Inkling 2 Mini GGUF exact file names inside `Mainline_Support`, and whether an mmproj is there.
- Whether the official SenseNova GGUFs load in upstream once PR #32588 merges.
- Whether the netprojects MLX ports run in stock mlx-lm.
- Every speed figure above except ours. None of the four was run on the 4090 or the M4 Max for this note.
- Which "Strata" the owner meant. Ask once.
- TensorFold 1.3.1 (FU-511) was not checked for new families; recheck its `capabilities --json` for `inkling2` or a looped family when bumping.

## Sources

- Inkling 2 Mini card: https://huggingface.co/thinkingmachines/Inkling-2-Mini
- Inkling 2 Small card: https://huggingface.co/thinkingmachines/Inkling-2-Small
- Inkling (v1) card: https://huggingface.co/thinkingmachines/Inkling
- Mini GGUF: https://huggingface.co/unsloth/Inkling-2-Mini-GGUF
- Small GGUF guide: https://unsloth.ai/docs/models/inkling-2
- llama.cpp b12406 (PR #32217): https://newreleases.io/project/github/ggml-org/llama.cpp/release/b12406
- llama.cpp b12549 (PR #32429): https://newreleases.io/project/github/ggml-org/llama.cpp/release/b12549
- Inkling v1 draft PR: https://github.com/ggml-org/llama.cpp/pull/25731
- SenseNova-Loop-9B card: https://huggingface.co/sensenova/SenseNova-Loop-9B
- SenseNova GGUF: https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF
- Issue #31571: https://github.com/ggml-org/llama.cpp/issues/31571
- PR #32588: https://github.com/ggml-org/llama.cpp/pull/32588
- Strata engine: https://github.com/Niko1221/Strata (docs/DETAILS.md)
- OSDI 2026 Strata paper: https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang
- HyperQwen: https://github.com/syv-ai/HyperQwen
