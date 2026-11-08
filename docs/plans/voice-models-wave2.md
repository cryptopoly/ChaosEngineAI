# Voice models wave 2: VoxCPM2, Chatterbox, Chatterbox-Turbo, Orpheus 3B

Date: 2026-11-08. Decision D-70, tracker FU-149. Branch `plan/voice-models-wave2`, cut from `release/v0.10.0`. Research and plan only; no source changed. Verified facts, repo shas and sources: [voice-models-wave2/research.md](voice-models-wave2/research.md). Builds after the D-67 voice UX packages W1 to W4 (the picker this plan feeds).

## 1. Headline

1. **On Apple Silicon all four run on what we already ship.** mlx-audio 0.6.4, already in the owner's voice venv, has loaders for VoxCPM2 (`voxcpm2`), Chatterbox Multilingual V3 (`chatterbox`), Chatterbox-Turbo (`chatterbox_turbo`) and Orpheus (`llama`), and they import nothing the venv lacks. The Mac work is catalog rows, worker calls and screen controls, not new dependencies.
2. **Three loaders download extra repos, unpinned, inside the worker.** Both Chatterbox loaders fetch `mlx-community/S3TokenizerV2`; the Orpheus loader fetches `mlx-community/snac_24khz` at import and its tokenizer from the bf16 repo by name. We must download these at pinned revisions with the model and send the loaders to the local copies.
3. **Chatterbox's own watermark is not there on the Mac.** The PerTh watermark is added by the upstream PyTorch package only; mlx-audio adds none. Our AudioSeal watermark, AI tag and audit log on cloned speech are the only marks on the MLX lane, and they must apply to every model that can clone (VoxCPM2, Chatterbox, Turbo).
4. **Orpheus upstream is gated.** `canopylabs/orpheus-3b-0.1-ft` is gated and 56.7 GB (training state included). Use `mlx-community/orpheus-3b-0.1-ft-8bit` on the Mac and `unsloth/orpheus-3b-0.1-ft` (or its GGUF) as the ungated mirror elsewhere.
5. **Five fixes our own code needs before any of them sounds right** (research section 4): Orpheus `<laugh>` tags are stripped by `clean_for_speech`; a chunk that starts with `(` turns into a VoxCPM2 voice description; Orpheus's default `max_tokens` cuts a 400-character chunk short; sharded repos never count as installed; one licence constant (`CLONE_LICENCE`) is used for every model.
6. **CUDA is a separate, larger job.** `chatterbox-tts` pins torch 2.6, transformers 5.2 and gradio 6.8; `voxcpm` pulls gradio, funasr, modelscope and datasets; the official Orpheus package needs vLLM (Linux only). Recommended: an on-demand speech GPU add-on with hand-picked dependencies, and Orpheus through our own `llama-server` plus a SNAC ONNX decoder. Phase 6, after the Mac rows have measured numbers.

Recommended order: Chatterbox-Turbo, VoxCPM2, Chatterbox Multilingual V3, Orpheus 3B, then the CUDA lane (section 8).

## 2. Proposed catalog rows

Facts are from the research file. Memory is a guess (weights plus about 0.5 to 1 GB for codec, activations and Python) until Phase 0 measures it. All rows `lanes: ["mlx"]` until Phase 6.

| Key | Repo (pinned sha) | Upstream | Licence for the catalog | Download | Memory (guess) | Out rate | Languages | Capabilities |
|---|---|---|---|---|---|---|---|---|
| `chatterbox-turbo` | `mlx-community/chatterbox-turbo-8bit` @ `2f2e21a0` + `S3TokenizerV2` @ `e0c9886f` | `ResembleAI/chatterbox-turbo` | `mit` (mlx card says apache-2.0; show upstream) | 0.71 + 0.49 GB | ~1.5 to 2 GB | 24 kHz | English | speak (default voice), tags, clone |
| `voxcpm2` | `mlx-community/VoxCPM2-8bit` @ `d5272589` | `openbmb/VoxCPM2` | `apache-2.0` | 3.23 GB | ~4 to 4.5 GB | 48 kHz | 30, auto-detected | speak, design, style, clone (two modes) |
| `chatterbox-mtl` | `mlx-community/chatterbox-multilingual-v3` @ `03565773` + `S3TokenizerV2` + a default-voice `conds.safetensors` (Phase 0 question) | `ResembleAI/chatterbox` | `mit` | 2.71 + 0.49 GB | ~3.5 to 4 GB | 24 kHz | 23, chosen | speak (default voice), expression, clone |
| `orpheus-3b` | `mlx-community/orpheus-3b-0.1-ft-8bit` @ `c8521bf4` + `snac_24khz` @ `556af1cd` + tokenizer files from `orpheus-3b-0.1-ft-bf16` @ `3786aac5` | `canopylabs/orpheus-3b-0.1-ft` (gated; mirror `unsloth/orpheus-3b-0.1-ft`) | `apache-2.0` (card also lists base model `meta-llama/Llama-3.2-3B-Instruct`; open question 5) | 3.54 + 0.08 GB | ~4.5 GB | 24 kHz | English | speak (8 preset voices), tags |

Variant choice: 8-bit everywhere, the same choice FU-148 measured for Qwen3-TTS (smaller and no slower). Two exceptions: Chatterbox V3 has no quantised MLX build (fp16 only), and VoxCPM2's bf16 build is sharded. A 4-bit Turbo (0.42 GB, the most downloaded) is the fallback if the 8-bit sounds no better in the listening check.

Total for all four on the Mac: about 11.2 GB of downloads.

## 3. How each control maps to the Speak screen and `POST /api/voice/speak`

Today's request: `{text, voice, style, speed, pitch, seed, model, permissionConfirmed}` (`backend_service/routes/voice.py:36`). The plan keeps one request and adds three optional fields whose defaults change nothing for existing callers:

| New field | Type, range, default | Used by | Screen |
|---|---|---|---|
| `language` | ISO code or `auto`, default `auto` (already listed in FU-141) | Chatterbox V3 (`lang_code`, required, 23 codes); Qwen3-TTS (`language`); others ignore it | language select, shown only when the voice's model takes one |
| `expression` | 0.0 to 1.0, default 0.5 | Chatterbox V3 `exaggeration` (always sent: mlx defaults to 0.1, the card to 0.5) | "Expression" slider, calm to dramatic, Chatterbox only |
| `guidance` | 0.0 to 1.0, default 0.5 | Chatterbox V3 `cfg_weight` | under Advanced: "Follow the voice sample closely" with the card's tips (0.3 for fast speakers, 0 when the clip is in another language) |

The MCP `speak` tool and `tests/golden/mcp_tools_list.json` gain the same three fields on purpose (regenerated in the phase that adds them).

Voice ids (extends the scheme in `voice_speak.py`):

| Id | Meaning |
|---|---|
| `chatterbox-turbo` | Turbo's built-in voice (from its `conds.safetensors`) |
| `chatterbox` | Chatterbox V3's default voice |
| `orpheus:tara` ... `orpheus:zoe` | Orpheus presets, eight voices |
| `design` | voice design; `model` picks Qwen3-TTS VoiceDesign or VoxCPM2 (default: whichever is installed; both: the last used) |
| `clone:<id>` | a cloned voice; `model` picks any downloaded clone-capable model (Qwen3 Base, VoxCPM2, Chatterbox V3, Turbo) |

`model` stays the way to name a specific catalog row; `voice_speak._model_for` and `speakLogic.chooseModel` learn the new kinds.

### 3.1 Per model

**Chatterbox-Turbo.** Text with tags; default voice; speed and pitch through ffmpeg; no style box (greyed with "Turbo takes tags in the text, not a style"); no language select. Tag chips under the text box insert the nine documented sound tags at the cursor: laugh, chuckle, sigh, gasp, cough, clear throat, sniff, groan, shush. The ten style tokens in its tokenizer (`[angry]`, `[whispering]`, `[happy]` and others) are undocumented; leave them off the chips until the listening check says they work. Worker call: `model.generate(text, temperature=0.8, top_p=0.95, repetition_penalty=1.2, split_pattern=None, max_tokens=800)` (we chunk, so turn its own split off), plus `ref_audio` only on the clone path.

**VoxCPM2.** Three uses on one model:

- Design: voice `design` with model `voxcpm2`; the style box is the description (same label and hint as Qwen VoiceDesign). Worker passes `instruct=description`, never pastes it into the text itself.
- Plain speech without a description: allowed through the same `design` voice with an empty box? No: an empty description gives a random voice every chunk. The `design` voice keeps its rule that the box is required.
- Cloning, two modes chosen automatically, no new field:
  - style box empty: **exact** ("ultimate" in the card): `ref_audio=clip, prompt_audio=clip, prompt_text=<stored transcript>`. We already store the transcript with every cloned voice.
  - style box filled: **steered**: `ref_audio=clip, instruct=style` (the card's controllable cloning). The note under the box says the steered mode trades a little likeness for the style.
- Languages: 30, detected from the text; no select. The note lists them in the model's detail.
- Long text: every chunk would design a new voice. Plan: speak chunk one with the description, then speak each later chunk in continuation mode (`prompt_audio`=chunk one's audio, `prompt_text`=chunk one's text) so the voice holds. Phase 0 confirms continuation keeps the voice; if not, warn above about 400 characters and point to design-then-clone (D-67 W8).
- Guard: strip or escape a leading `(` from every chunk so ordinary text is never read as a description (`clean_for_speech` itself makes "(code block omitted)").
- Fixed for v1: `inference_timesteps=10`, `cfg_value=2.0` (card defaults). A "higher quality" Advanced switch (20 steps) waits for measured cost.

**Chatterbox Multilingual V3.** Default voice `chatterbox`; language select (23 codes, defaulting to the UI language when it is one of them, else English); Expression slider; Guidance under Advanced; style box greyed ("use Expression"); no tags (V3 has no tag tokens; the backend strips bracket tags for it). Worker: `model.generate(text, lang_code=language, exaggeration=expression, cfg_weight=guidance, temperature=0.8)`.

**Orpheus 3B.** Eight preset cards in the picker (tara, leah, jess, leo, dan, mia, zac, zoe; the authors rank them in that order); tag chips for its eight tags (laugh, chuckle, sigh, cough, sniffle, groan, yawn, gasp); no style box, no language select, no cloning. Worker: `model.generate(text, voice=name, temperature=0.6, top_p=0.8, repetition_penalty=1.3, split_pattern="\n", max_tokens=<enough for one chunk>)`; Phase 0 sets the chunk size and token budget together.

**One tag syntax for people.** The text box always uses square brackets (`[laugh]`). The backend, per model: keeps them for Turbo; turns them into angle brackets for Orpheus (and `[sniff]` into `<sniffle>`) after `clean_for_speech` has run, so the HTML-tag rule never sees them; strips them, with a note, for every model without tags (Kokoro, Qwen3-TTS, VoxCPM2, Chatterbox V3), so no voice reads "laugh" aloud. A tag the model does not know is stripped the same way. Sentence splitting must keep a tag with its sentence (test).

### 3.2 Cloning: the same rules for every model that can clone

Cloning-capable after this plan: Qwen3-TTS Base (both), VoxCPM2, Chatterbox V3, Chatterbox-Turbo. Orpheus is not offered for cloning (the fine-tuned model is meant for its eight voices); the backend refuses a reference clip for it.

Rules, unchanged from D-47 and `voice_clone.speak`, and now model-independent:

- The **only** path that sends a reference clip to the worker is `voice_clone.speak`. `voice_speak.build_plan` never sets `refAudioPath`, `ref_audio` or `prompt_audio` from a clone; the VoxCPM2 continuation for designed voices uses audio the model just made, never a user clip. Test: no plan built outside `voice_clone` carries a reference path.
- Permission record required (`get_voice` refuses a clone without `consent`); agents and other computers (`origin != "ui"`) must send `permission_confirmed`; the agent-cloning switch still gates the route (`safety.clone_allowed`).
- After rendering: speed and pitch, then the AudioSeal watermark (`safety.apply_watermark`), then the AI tag (`safety.tag_wav`, which records the model repo), then the Library save with `clonedVoice: true`, then the audit line. Same order for every model.
- The watermark component is installed with any clone-capable model download (today only for Qwen Base rows, because the check reads `capabilities.clone`, which the new rows set).
- Reference length: the 3 to 30 s rule stays; per-model minimum where the loader asks for more (Turbo: 5 s per its docstring; the form says so when Turbo is the selected model).
- `CLONE_MODELS` becomes every row with `capabilities.clone`; `clone_model("")` still returns the Qwen3 0.6B Base default until the owner picks otherwise (open question 2).

## 4. Audio path: sample rate, post-processing, marks

| Step | Kokoro, Qwen3, Chatterbox, Turbo, Orpheus | VoxCPM2 |
|---|---|---|
| Worker output | 24 kHz mono float, written as 16-bit WAV (`wav_bytes`) | 48 kHz mono |
| Join chunks (`join_wavs`) | same format inside one job, fine | same; the 0.15 s gap scales with the rate |
| Lift quiet speech (`lift_quiet`) | unchanged | unchanged |
| Speed (ffmpeg `atempo`) | post-processing for all of them (none does speed natively; mlx Chatterbox ignores `speed`) | same |
| Pitch (`asetrate` + `aresample` + `atempo`) | post-processing | same; uses the file's own rate, so 48 kHz needs no change |
| Cloned only: AudioSeal | 16 kHz watermark signal resampled back and added (`voice_worker_watermark.embed`) | same; check detection on a 48 kHz file in Phase 3 |
| Cloned only: AI tag | WAV INFO chunk, MP3 comment on export | same |
| Library and MP3 | unchanged | 48 kHz WAVs are twice the size (about 5.8 MB per minute) |

Chatterbox's PerTh watermark:

- MLX lane: absent (mlx-audio has no watermark code). Nothing to strip, nothing to keep. Plain Chatterbox speech carries no mark; cloned Chatterbox speech carries ours.
- CUDA lane (Phase 6): `chatterbox-tts` adds PerTh to every output before our ffmpeg speed and pitch run. Resemble says it survives editing; Phase 6 checks it with `perth.PerthImplicitWatermarker().get_watermark` after speed 1.3 and pitch +2. Cloned output then carries PerTh and AudioSeal, both inaudible; the AI tag records `"builtinWatermark": "perth"` so the record is honest. Keep PerTh on (removing it means patching the package).
- Adding PerTh ourselves on the Mac for parity is possible (`resemble-perth` is MIT and needs torch and librosa, which the watermark component already brings) but is not proposed: AudioSeal already marks every cloned clip.

## 5. Install path and dependencies

### 5.1 Apple Silicon: the voice venv, nothing new

- Raise the floor in `voice_env._MLX_PACKAGES` from `mlx-audio>=0.6.2,<0.7` to `>=0.6.4,<0.7`, the version every fact here was read from. A pack built earlier gets the existing "update the voice pack" prompt by adding a feature name (`"speech-wave2"`) to `voice_env.FEATURES`.
- No new package: the four loaders import only mlx, numpy, transformers, tqdm, librosa and scipy, all present.
- Companion repos are catalog data: each row gets `companions: [{repo, revision, files}]`. `voice_clone.start_download` fetches them with the model at their pinned shas; `snapshot_dir` (renamed `model_ready` or kept) counts a row as installed only when the model and every companion are on disk, and accepts `model.safetensors.index.json` with its shards.
- The worker sends the loaders to the local copies: before `load_model`, a small redirect in `voice_worker.py` maps the known companion repo ids (`mlx-community/S3TokenizerV2`, `mlx-community/snac_24khz`, `mlx-community/orpheus-3b-0.1-ft-bf16`) to the pinned local snapshot paths it receives in the request, by wrapping `huggingface_hub.snapshot_download` / `hf_hub_download` and transformers' tokenizer lookup for those ids only. The SNAC import side effect means the redirect must be in place before `mlx_audio.tts.models.llama` is imported. A request for a companion that is not in the map fails with a plain error instead of downloading.
- The catalog audit gains the voice and music catalogs (rows and companions, pinned revision must exist), with the no-token rule written into the script (`token=False`).

### 5.2 Windows, Linux, NVIDIA (Phase 6)

Today the portable lane has faster-whisper and kokoro-onnx only, and cloning is Mac-only (`voice_clone.available_here`). Proposal: a "speech GPU add-on" that installs into the voice venv on the portable lane when an NVIDIA GPU is present, the same way `EXTRAS["chinese"]` works, with its own pinned list:

| Model | Route | Notes |
|---|---|---|
| Chatterbox V3, Turbo | `chatterbox-tts==0.1.7 --no-deps` plus a hand-picked set: torch and torchaudio (CUDA 12 build), transformers, safetensors, librosa, `resemble-perth`, `s3tokenizer`, `conformer`, `pyloudnorm`, `omegaconf`, `diffusers` (what its model code imports; to be confirmed by import test) | skips gradio and the exact pins; loads from our pinned snapshot (upstream Turbo `from_pretrained` asks for a login) |
| VoxCPM2 | `voxcpm==2.0.3 --no-deps` plus torch, torchaudio, transformers, einops, inflect, addict, wetext, soundfile, librosa, safetensors, pydantic, huggingface_hub (to be confirmed by import test) | skips gradio, funasr, modelscope, datasets, matplotlib, spaces; `load_denoiser=False` |
| Orpheus | GGUF (`unsloth/orpheus-3b-0.1-ft-GGUF`, Q8_0 3.52 GB or Q4_K_M 2.09 GB) through the bundled `llama-server`, codes decoded by SNAC ONNX (`onnx-community/snac_24khz-ONNX`) on onnxruntime | no PyTorch, no vLLM; works on CPU, CUDA, Windows and Linux, and could replace the MLX path later if faster; unproven, needs a spike |

Install size of the add-on: PyTorch with CUDA 12 libraries is several GB (unmeasured; measure on the 4090 box). CPU-only machines: Turbo and Orpheus-on-llama-server are the realistic ones; VoxCPM2 and V3 on CPU are expected to be slow (unmeasured). Linux ARM64 (DGX Spark) needs its own check, as `voice_uses_cuda()` already notes for CTranslate2.

### 5.3 Licences for `THIRD_PARTY_NOTICES.md`

Add under "Voice cloning and watermark", or a new "More speech models (on demand)" section: VoxCPM2 weights and `voxcpm` (Apache-2.0); Chatterbox V3 and Turbo weights and `chatterbox-tts` (MIT); `resemble-perth` (MIT, CUDA lane only); `s3tokenizer` (Apache-2.0) and the S3TokenizerV2 MLX weights (repo has no licence field; upstream code Apache-2.0, list as such with the repo link); Orpheus 3B weights (licence field apache-2.0; see open question 5); SNAC code and weights (MIT); mlx-audio (MIT, already listed). Phase 6 adds each hand-picked CUDA dependency with its licence, checked one by one (MIT, Apache-2.0, BSD only).

## 6. Measurement protocol

Same measures as the Qwen3-TTS rows (`estimates` in `backend_service/catalog/voice_models.py`), so the picker can compare like with like. A new `scripts/measure-voice-models.py` drives the running backend's worker, never port 8876's text server; downloads happen only when the owner starts them.

For each row, on a fresh worker with weights already on disk:

| Measure | How |
|---|---|
| `coldStartSeconds` | time from the first `synthesize` request to its reply, minus that reply's own speech time (load cost) |
| `wallSecondsShortLine` | one 12-word line, second request (warm) |
| `secondsPer10sAudio` | a fixed 600-character English text (the one used for Qwen3), wall time divided by output seconds times 10 |
| `peakGb`, `idleGb` | worker RSS (psutil) sampled every 100 ms during the long text, and 5 s after it ends; on CUDA also `nvidia-smi --query-gpu=memory.used` peak |
| Extra per model | Turbo: with and without three tags; VoxCPM2: design, exact clone, steered clone, and a 1,200-character design with continuation (voice holds?); Chatterbox V3: English, German and Japanese lines, expression 0.3 / 0.5 / 0.8; Orpheus: each of the eight voices once, a 400-character chunk (cut short?) |
| Listening check (owner) | does it sound better than Qwen3 1.7B CustomVoice for English; do tags land; does the cloned voice resemble the speaker (own voice only) |
| Marks | cloned output: `GET /api/voice/clone/watermark/{id}` reports the AudioSeal message; AI tag present; on CUDA, PerTh detected after speed and pitch |

M4 Max (64 GB): all four MLX rows. RTX 4090 box: Phase 6 rows, plus the same text through each so the numbers sit side by side; record driver, CUDA and torch versions. Add the four models to `docs/plans/rtx4090-test.md`. Each catalog row ships with `basis: "measured"` and `measuredOn`; until then the picker shows "not measured yet" (D-70 says no row ships without numbers).

## 7. What the D-67 picker needs from the backend

The picker (D-67 W1 to W3) already reads `capabilities`, `installed`, `sizeGb` and, after W1, `estimates`. The new rows need:

| Field on `/api/voice/models` rows | Purpose |
|---|---|
| `licence` per row (`{id, name, commercial, url}`) | replaces the shared `CLONE_LICENCE` |
| `family` (`qwen3-tts`, `voxcpm2`, `chatterbox`, `chatterbox-turbo`, `orpheus`, `kokoro`) | grouping and the model line |
| `capabilities` adds `expression`, `tags`, `presets`, `languageSelect` next to `speak`, `style`, `clone`, `design` | which controls appear |
| `controls`: `{expression: {min, max, default}, guidance: {...}, languages: [codes], tags: [names]}` | the screen draws sliders, the language list and tag chips from data, no per-model code |
| `sampleRate` | the 48 kHz note and file size estimate |
| `downloadGb` (model plus companions) and `estimates` | cost chip: download, memory, seconds per 10 s, measured on |
| `minReferenceS` | clone form hint per model |
| `builtinWatermark` (`perth` on the CUDA Chatterbox rows, else null) | honest note under the result |

On `/api/voice/voices` entries: `group` for the picker (D-67 groups: Recommended, Quick, Natural, Cloned, Designed, More; Orpheus presets and the Chatterbox and Turbo default voices go under Natural, labelled by family), `about` for the Orpheus voices (rank order and gender), and `models` listing every row that can speak it (for `design` and `clone:*`, more than one).

Recommended-card rule (W2) gets one input: per-family English quality rank from the listening check, kept in the catalog (`rank`), so "best installed natural voice" can prefer, say, VoxCPM2 or Orpheus over Qwen3 presets once measured.

## 8. Phases

Sizes: S under half a day, M one to two days, L three days or more, tests included.

| # | Phase | Size | Files | Tests |
|---|---|---|---|---|
| P0 | Spike and measurement harness on the M4 Max: download the four rows with the owner's go-ahead (about 11.2 GB); answer the open checks (V2 `conds.safetensors` as V3's default voice, Orpheus token budget, VoxCPM2 continuation, `mx.random.seed` effect on each); fill `estimates` | S | new `scripts/measure-voice-models.py`; notes appended to the research file | script unit test for the timing and RSS maths (`tests/test_measure_voice_models.py`) |
| P1 | Framework: catalog schema (per-row licence, family, companions, controls, sampleRate, minReferenceS, rank); readiness with shards and companions; companion download; worker redirect and `family` dispatch; tag translation and stripping after `clean_for_speech`; VoxCPM2 leading-paren guard; request fields `language`, `expression`, `guidance`; `CLONE_MODELS` from capabilities; audit script covers voice and music catalogs | M | `backend_service/catalog/voice_models.py`, `voice_clone.py`, `voice_speak.py`, `voice_text.py` (tag-safe cleaning helper), `voice_worker.py`, `voice_env.py` (floor, FEATURES), `routes/voice.py` (`SpeakRequest`), `mcp_bridge/tools.py` (`speak` schema), `tests/golden/mcp_tools_list.json`, `scripts/audit-catalog-repos.py`, `src/types/voice.ts` | pytest `tests/test_voice_speak.py` (tags kept, translated, stripped per family; tag stays with its sentence; leading paren guarded; new fields validated; defaults unchanged), `tests/test_voice_clone.py` (every clone-capable row is in `CLONE_MODELS`; per-row licence; no non-clone plan carries a reference path), `tests/test_voice_io.py` or a new `tests/test_voice_worker_models.py` (redirect maps only known ids, refuses others; dispatch by family with a fake mlx-audio module), `tests/test_audit_catalog_repos.py` (voice and music rows collected, companions included); `tests/test_voice_io.py` (FEATURES update prompt) |
| P2 | Chatterbox-Turbo: row, default voice, tag chips, clone with 5 s minimum | S | catalog row, `voice_speak.py` (`chatterbox-turbo` id), `voice_worker.py` (Turbo call), `VoicePicker.tsx`, `SpeakScreen.tsx` (tag chips from `controls.tags`), `speakLogic.ts`, `NewVoiceForm.tsx` (minimum per model), `src/locales/en/studio.json` | pytest (plan, worker call arguments with a fake model, clone path watermarks and tags a Turbo clip); vitest `speakLogic.test.ts` (tag chip insertion, model choice); e2e `14-voice-speak` (Turbo card, chips insert `[laugh]`), `07-voice-clone` (Turbo selectable as clone model; permission tick still required) |
| P3 | VoxCPM2: row, design through `model`, exact and steered clone, continuation for long designed text, 48 kHz | M | catalog row, `voice_speak.py`, `voice_clone.py` (mode choice), `voice_worker.py`, `SpeakScreen.tsx` (model choice for design; steered-mode note), `speakLogic.ts` | pytest (mode selection: empty style exact, style steered; continuation used from chunk two; 48 kHz join and pitch filter use the file rate; watermark detect on a 48 kHz WAV with the real AudioSeal in `test_voice_watermark.py` when installed, else skipped); vitest (design model choice); e2e `14-voice-speak` (design with VoxCPM2 selected sends `model: "voxcpm2"`) |
| P4 | Chatterbox V3: row, default voice, language select, Expression and Guidance | M | catalog row, `voice_speak.py`, `voice_worker.py`, `SpeakScreen.tsx` (sliders and language from `controls`), `speakLogic.ts` | pytest (expression always sent; language required and validated against the 23 codes; tags stripped with a note); vitest (controls drawn from data; language default from the UI language); e2e `14-voice-speak` (request body carries `expression`, `guidance`, `language`) |
| P5 | Orpheus: eight presets, tags translated, chunk size and token budget, no clone | S | catalog row, `CUSTOM_SPEAKERS`-like `ORPHEUS_VOICES`, `voice_speak.py`, `voice_worker.py`, `VoicePicker.tsx` | pytest (`[laugh]` becomes `<laugh>` after cleaning; `[sniff]` becomes `<sniffle>`; clone refused; chunk size honoured); vitest (presets grouped under Natural); e2e `14-voice-speak` (an Orpheus preset card) |
| P6 | CUDA and CPU lane: speech GPU add-on, Chatterbox and VoxCPM2 through PyTorch, Orpheus through `llama-server` plus SNAC ONNX, PerTh check, cloning opened to the portable lane once AudioSeal installs there | L | `voice_env.py` (add-on list, CUDA wheels index), `voice_worker.py` (torch engines), new `backend_service/voice_orpheus_llama.py` (llama-server driver), catalog rows with `lanes: ["cuda"]`, `voice_clone.available_here`, `THIRD_PARTY_NOTICES.md`, `docs/plans/rtx4090-test.md` | pytest with fake torch modules (dispatch, PerTh flag in the AI tag), `tests/test_voice_io.py` (add-on install command list per OS); real run on the 4090 (section 6) |
| P7 | Docs and wrap-up: `docs/usage/voice.md` (new models, tags, Expression, clone modes), THIRD_PARTY_NOTICES, i18n placeholders in the nine other locales, FOLLOW_UPS rows (FU-149 shipped parts, FU-141 `language` field done), hub rebuild | S | listed | `npm run i18n:validate`; `pytest tests/test_hub_tracker.py` |

E2E gate (`scripts/e2e_test_suite.py`, the voice phase with `_runtime`, `_speak_stream` and friends): add "speech model catalog" (every wave-2 row from `GET /api/voice/models` has a revision, licence, family, controls; companions listed) and "speak with each installed wave-2 model" (`POST /api/voice/speak` with one short line per installed row; WAV comes back at the row's `sampleRate`; skip with the reason when the row is not downloaded, so a fresh CI box skips honestly), plus "clone refused without permission for agents" against one new clone-capable row. Catalog audit: `.venv/bin/python scripts/audit-catalog-repos.py` passes with the voice rows included (needs network).

Order and why: Turbo first (smallest download, the largest step up for English chat-style speech, and it exercises tags and the generalised clone path); VoxCPM2 second (design and cloning in 30 languages, and an English female voice that keeps between runs via continuation); Chatterbox V3 third (Expression dial and 23 languages, after the default-voice question is answered); Orpheus fourth (largest, English only, overlaps Turbo's tags, gated upstream). P1 lands with P2 in one PR; P3, P4, P5 one PR each; P6 its own PR after the Mac rows are measured. Everything waits for D-67 W1 to W4.

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Companion repos fetched unpinned at load | a silent upstream change alters output; offline machines fail | pinned companion download plus the worker redirect (5.1); audit covers companions |
| Chatterbox V3 MLX build has no default voice | no plain-speech Chatterbox until solved | P0 tries V2's `conds.safetensors` (same voice encoder and tokenizer); fallback: ship V3 as clone-only and use Turbo for plain English |
| mlx-community licence fields differ from upstream (apache-2.0 vs mit) | wrong licence shown | catalog shows the upstream value and URL; audit test asserts each row's licence id |
| Orpheus tags stripped, chunks cut short | tags read as nothing, sentences truncated | P1 tag handling after cleaning; P0 sets chunk size and token budget; tests |
| VoxCPM2 voice drifts across chunks; parentheses in text become descriptions | long designed speech changes voice mid-way; odd voices | continuation from chunk one; leading-paren guard; test |
| Memory: Orpheus (~4.5 GB) or VoxCPM2 next to a loaded chat model | swap on smaller Macs | measured `peakGb` feeds the existing fit check (`studio_gate`); one TTS model resident at a time (`_free_tts` already unloads before the next load) |
| mlx-audio API drift in 0.6.x (we allow up to <0.7) | a loader argument renamed | floor 0.6.4; worker calls covered by fake-module tests; the e2e speak check per installed row |
| CUDA dependency pins (torch 2.6 in `chatterbox-tts`; newer GPUs need newer torch) | install conflicts, Blackwell cards unsupported on 2.6 | `--no-deps` with our own tested set; measure on the 4090 first |
| PerTh weakened by our ffmpeg speed and pitch | Resemble's mark harder to detect | P6 measures detection after processing; AI tag records which marks were applied |
| Clone abuse through a new model | cloning without consent | one clone path for every model, reference paths refused elsewhere (test), watermark component installed with any clone-capable model |

## 10. Open questions for the owner

1. **Chatterbox V3 default voice.** If V2's default voice file does not work with V3 weights in the spike, ship V3 as clone-only (Turbo covers plain English), or add the 0.93 GB V2 8-bit build just for its default voice? Recommendation: clone-only V3; Turbo is the plain-speech Chatterbox.
2. **Default cloning model.** Keep Qwen3-TTS 0.6B Base as the default, or make VoxCPM2 the default once measured (exact mode uses the transcript we already store, 30 languages, 48 kHz)? Recommendation: decide after the P0 listening check; default to whichever the owner prefers by ear, keep the other selectable.
3. **Turbo's undocumented style tokens** (`[angry]`, `[whispering]`, `[happy]` and seven more exist in its tokenizer). Offer them as chips? Recommendation: no in the first release; add the ones that pass the listening check.
4. **CUDA lane scope.** Build P6 in this wave, or ship the Mac rows first and schedule P6 with the next RTX 4090 test run? Recommendation: Mac rows first; P6 straight after, starting with Orpheus through `llama-server` (no PyTorch) and Turbo.
5. **Orpheus licence row.** The Hub licence field says apache-2.0 and the card lists `meta-llama/Llama-3.2-3B-Instruct` as a base model. Which licence entry should the catalog row and the notices show? Recommendation: the owner confirms before P5 ships; until then the row shows the Hub field with both repo links.

## 11. Not verified

- No model was downloaded, loaded or heard. Memory figures in section 2 are guesses; speed figures exist only as upstream claims (VoxCPM2 RTF ~0.3 on a 4090; Orpheus ~200 ms streaming latency).
- That `mx.random.seed` makes each new model repeatable.
- That V2's `conds.safetensors` works with V3 weights; that VoxCPM2 continuation holds a designed voice; the exact Orpheus tokens per second.
- That mlx-audio 0.6.2 or 0.6.3 have these loaders (only 0.6.4 was read).
- AudioSeal detection on a 48 kHz file; PerTh survival after our ffmpeg steps.
- The CUDA routes (hand-picked dependency sets, Orpheus through `llama-server` plus SNAC ONNX), their install sizes, and Windows behaviour. Nothing was run on the RTX 4090.
- Quality of Turbo's ten undocumented style tokens.
