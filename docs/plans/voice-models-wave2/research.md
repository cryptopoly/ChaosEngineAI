# Voice models wave 2: verified facts

Date: 2026-11-08. Companion to [../voice-models-wave2.md](../voice-models-wave2.md) (D-70, FU-149).

How these were checked:

- **Hub facts** (revision sha, gated flag, licence field, total size, file list) come from the public Hub API (`/api/models/<repo>?blobs=true`) called with no token and no `Authorization` header, on 2026-11-08. The script is a scratch copy of the `scripts/audit-catalog-repos.py` idea (list a repo, check its files); nothing was downloaded beyond JSON metadata and model cards.
- **MLX facts** (loader names, `generate()` arguments, defaults, sample rates, what is downloaded at load time) come from reading the mlx-audio **0.6.4** source that is already installed in the owner's voice venv (`~/.chaosengine/voice-venv`, Python 3.11, the version `voice_env._MLX_PACKAGES` resolved to). Earlier 0.6.x patch releases were not checked.
- **PyPI facts** (version, pinned dependencies, licence metadata) come from `https://pypi.org/pypi/<name>/json`. GitHub licences come from the GitHub repo API.
- **Not run:** no model was downloaded, loaded or listened to. Every speed and memory figure below is either an upstream claim (named as such) or a file size. RTX 4090 and M4 Max numbers are still to be measured (plan section 6).

## 1. VoxCPM2 (OpenBMB)

| Fact | Value | Source |
|---|---|---|
| Upstream repo | `openbmb/VoxCPM2`, sha `32279effe8c19989596f05d353d1447f51d9e915`, not gated, licence field `apache-2.0`, 4.96 GB (one `model.safetensors` + `audiovae.pth`) | Hub API |
| Code | `OpenBMB/VoxCPM`, Apache-2.0; PyPI `voxcpm` 2.0.3 (2026-05-11) | GitHub API, PyPI |
| Size | 2B parameters (MiniCPM-4 backbone); LM token rate 6.25 Hz; max sequence 8192 tokens; bf16 | model card |
| Output | **48 kHz** mono (AudioVAE V2 encodes at 16 kHz, decodes at 48 kHz) | model card; mlx `sample_rate` = `audio_vae_config.out_sample_rate` |
| Languages | 30, **no language tag**: Arabic, Burmese, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese; plus nine Chinese dialects | model card |
| Modes | plain text; voice design `(description)text`; controllable cloning (reference clip, optional `(style)` prefix); "ultimate" cloning (reference clip + same clip as prompt + its exact transcript); streaming | model card |
| CUDA claims | ~8 GB VRAM; RTF ~0.30 on an RTX 4090 (~0.13 with a separate Nano-vLLM build) | model card (claims, unmeasured) |
| Card caveat | design and style results vary between runs ("generate 1-3 times"); instability possible on very long or very expressive input | model card |

MLX builds (all `mlx-community`, licence field `apache-2.0`, not gated):

| Repo | sha | Size | Files |
|---|---|---|---|
| `mlx-community/VoxCPM2-bf16` | `a7fb80d9056983135a54ca3f71fcc5a637ccddf5` | 4.96 GB | **sharded** (`model-00001/00002.safetensors` + index) |
| `mlx-community/VoxCPM2-8bit` | `d52725898a0675703f7f9ddc5a4d1a3cdbb99032` | 3.23 GB | single `model.safetensors` |
| `mlx-community/VoxCPM2-4bit` | `dc9e5c187858da5f4a13dc4c247e297339216381` | 2.30 GB | single `model.safetensors` |

mlx-audio 0.6.4 (`tts/models/voxcpm2/voxcpm2.py`), model type `voxcpm2`:

```python
model = load_model(snapshot)                     # mlx_audio.tts.utils
model.generate(text, max_tokens=2000, ref_audio=None, prompt_text=None, prompt_audio=None,
               inference_timesteps=10, cfg_value=2.0, instruct=None, ...)  # yields GenerationResult(.audio at 48 kHz)
```

- `instruct="..."` is turned into `f"({instruct}){text}"` inside `generate`. Design and style both use it.
- Modes as coded: text only; `instruct`; `ref_audio` (controllable clone); `prompt_audio`+`prompt_text` (continuation); `ref_audio`+`prompt_audio`+`prompt_text` ("combined", the card's ultimate cloning).
- **`ref_text` is accepted and ignored.** Ultimate cloning needs `prompt_audio=<clip>, prompt_text=<transcript>, ref_audio=<clip>`.
- `cfg_value` has a floor of 2.0 when passed through the `cfg_scale` alias. `max_tokens=2000` patches at 6.25 Hz is about 320 s per call, so our 400-character chunks are well inside it.
- No `speed`, no pitch, no seed argument (use `mx.random.seed`, as we do for Qwen3-TTS; unverified that it fixes the diffusion sampler).
- No extra repo is fetched at load time.

CUDA / CPU path: `pip install voxcpm` (Python >= 3.10, PyTorch >= 2.5, CUDA >= 12 per the card). Its declared dependencies are heavy and mostly unneeded for inference: `gradio<7,>=6`, `funasr`, `modelscope>=1.22`, `datasets<4`, `matplotlib`, `spaces`, `torchcodec`, `wetext`, `argbind`, among others. `VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)` avoids the optional denoiser download. Windows support is not stated on the card (Linux + CUDA is what is documented).

## 2. Chatterbox Multilingual V3 and Chatterbox-Turbo (Resemble AI)

### Upstream

| Fact | Value | Source |
|---|---|---|
| Multilingual repo | `ResembleAI/chatterbox`, sha `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`, not gated, licence field `mit`, 13.87 GB total (English, V2 and V3 checkpoints, `.pt` and `.safetensors` copies side by side) | Hub API |
| Turbo repo | `ResembleAI/chatterbox-turbo`, sha `749d1c1a46eb10492095d68fbcf55691ccf137cd`, not gated, `mit`, 4.04 GB | Hub API |
| Code | `resemble-ai/chatterbox`, MIT; PyPI `chatterbox-tts` 0.1.7 (2026-03-26) | GitHub API, PyPI |
| Sizes | Multilingual V3 0.5B (Llama backbone); Turbo 350M (GPT-2 medium backbone, one-step distilled mel decoder); a 110M Chatterbox-Nano also exists (CPU-oriented, not in D-70) | model cards |
| Languages | V3: 23 (`ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh`), language passed as `language_id`; Turbo: English only | model card; mlx `SUPPORTED_LANGUAGES` |
| Controls | Multilingual and original: `exaggeration` (card default 0.5; 0.7+ dramatic) and `cfg_weight` (default 0.5; ~0.3 for fast references; 0 for cross-language to drop the reference accent). Higher exaggeration speeds speech up; lower `cfg_weight` slows it. Turbo: not used | model card tips |
| Turbo tags | card names `[cough]`, `[laugh]`, `[chuckle]` "and more". The tokenizer (`added_tokens.json`) has 19: `[laugh] [chuckle] [sigh] [gasp] [cough] [clear throat] [sniff] [groan] [shush]` (the nine the mlx-community card documents as sound events) and `[angry] [fear] [surprised] [whispering] [advertisement] [dramatic] [narration] [crying] [happy] [sarcastic]` (style tokens, not documented) | Hub raw file |
| Watermark | upstream package adds Resemble's PerTh watermark to **every** output; `perth.PerthImplicitWatermarker().get_watermark(audio, sample_rate=sr)` detects it. `resemble-ai/Perth` is MIT | model card, GitHub API |
| Output | 24 kHz (`S3GEN_SR`) for both | mlx source |
| Reference clip | upstream example uses ~10 s; mlx Turbo docstring says "> 5 seconds"; decoder uses up to 10 s, encoder up to 6 s (V3) / 15 s (Turbo) | card, mlx source |

### MLX builds

| Repo | sha | Licence field | Size | What it is |
|---|---|---|---|---|
| `mlx-community/chatterbox-multilingual-v3` | `03565773edd72e949572557597af8063bb49a18a` | mit | 2.71 GB | V3 (`t3_model: "v3"`), fp16; **no `conds.safetensors`**, so no default voice |
| `mlx-community/chatterbox-fp16` | `4923fcca09086356aeab5191a2348c5a17a23694` | apache-2.0 (upstream is mit) | 2.58 GB | multilingual V2 (converted with mlx-audio 0.2.7), has `conds.safetensors` (default voice) |
| `mlx-community/chatterbox-8bit` | `9617d61b596a03d1bed766a28c341680e993a1b9` | apache-2.0 | 0.93 GB | V2 8-bit, has conds; also carries a stray nested `models--ResembleAI--chatterbox/...` file |
| `mlx-community/Chatterbox-TTS-fp16` / `-8bit` | `77c7c8f9...` / `0d4905ca...` | none | 2.70 / 1.28 GB | English original; no licence field, skip |
| `mlx-community/chatterbox-turbo-fp16` | `b2d0a13aa7cfff0a06d9acb247ae91c8f19a6d75` | apache-2.0 (upstream mit) | 2.99 GB | Turbo, has conds |
| `mlx-community/chatterbox-turbo-8bit` | `2f2e21a03863f86a1274d1060dcc188e7cde77e1` | apache-2.0 | 0.71 GB | Turbo 8-bit, has conds |
| `mlx-community/chatterbox-turbo-4bit` | `c63817725071d7b5269c7b558772d6e8cbf59cec` | apache-2.0 | 0.42 GB | Turbo 4-bit, has conds (228k downloads, the most used) |
| `mlx-community/S3TokenizerV2` | `e0c9886f0e1c35ae85b1f27277416fb19fc72bec` | none | 0.49 GB | speech tokenizer **fetched at load time** by both Chatterbox loaders |

Licence field mismatch: the mlx-community Chatterbox conversions say `apache-2.0` while every Resemble AI repo and the code say MIT. The catalog should show the upstream value (`mit`) and link the upstream card, as the Qwen3-TTS rows do.

mlx-audio 0.6.4:

```python
# model_type "chatterbox" (tts/models/chatterbox/chatterbox.py)
model.generate(text, exaggeration=0.1, cfg_weight=0.5, temperature=0.8, repetition_penalty=1.2,
               min_p=0.05, top_p=1.0, max_new_tokens=1000, ref_audio=None, lang_code="en", ...)
# model_type "chatterbox_turbo" (tts/models/chatterbox_turbo/chatterbox_turbo.py)
model.generate(text, ref_audio=None, sample_rate=None, temperature=0.8, top_k=1000, top_p=0.95,
               repetition_penalty=1.2, norm_loudness=True, split_pattern=r"(?<=[.!?])\s+", max_tokens=800, ...)
```

- **mlx `exaggeration` defaults to 0.1, not the card's 0.5.** We must always pass the value we want.
- `speed`, `voice`, `stream` are ignored by Chatterbox (stated in the docstring). Turbo ignores `exaggeration` and `cfg_weight`.
- With `ref_audio` the loader builds conditionals from the clip; without it, it uses `conds.safetensors` in the model folder, and raises "No conditionals available" when that file is missing (V3).
- `post_load_hook` calls `snapshot_download("mlx-community/S3TokenizerV2")` with **no revision**; Turbo calls `hf_hub_download` for the same repo. Both happen inside the worker at first load.
- **mlx-audio adds no watermark** (no PerTh or other watermark code in either model folder). The PerTh watermark only appears on the CUDA path through `chatterbox-tts`.
- `max_new_tokens=1000` at 25 Hz is about 40 s per chunk; Turbo's 800 is about 32 s and it splits on sentence ends itself.

CUDA / CPU path: PyPI `chatterbox-tts` 0.1.7 pins exact versions for Python < 3.14: `torch==2.6.0`, `torchaudio==2.6.0`, `transformers==5.2.0`, `diffusers==0.29.0`, `gradio==6.8.0`, `safetensors==0.5.3`, `librosa==0.11.0`, `numpy<2`, plus `resemble-perth`, `s3tokenizer` (Apache-2.0), `conformer`, `spacy-pkuseg`, `pykakasi`, `pyloudnorm`, `omegaconf`. Tested by the authors on Python 3.11 / Debian 11. Devices in the upstream example: `cuda`, `cpu`, `mps`. Upstream Turbo `from_pretrained` passes `token=os.getenv("HF_TOKEN") or True` to `snapshot_download`, which fails without a login; download the pinned snapshot ourselves and load from the local folder. ONNX exports exist (`ResembleAI/chatterbox-turbo-ONNX`, MIT, 7.39 GB for all precisions; `onnx-community/chatterbox-multilingual-ONNX`, MIT, 4.98 GB) and are a possible CPU or Windows route that avoids PyTorch.

## 3. Orpheus 3B (Canopy Labs)

| Fact | Value | Source |
|---|---|---|
| Upstream repo | `canopylabs/orpheus-3b-0.1-ft`, sha `4206a56e5a68cf6cf96900a8a78acd3370c02eb6`, **gated (`auto`)**, licence field `apache-2.0`, 56.7 GB (training state included: optimizer, FSDP, RNG files) | Hub API |
| Ungated mirrors | `unsloth/orpheus-3b-0.1-ft` (sha `eae2b6e5e429c81b95ac42a883ac64f126583d43`, 6.63 GB bf16 safetensors, `apache-2.0`); `unsloth/orpheus-3b-0.1-ft-GGUF` (sha `e2b00302c46af8205f521f60016600aa25a068e6`; Q4_K_M 2.09 GB, Q8_0 3.52 GB, F16 6.61 GB) | Hub API |
| Base model | card metadata lists `meta-llama/Llama-3.2-3B-Instruct` and `canopylabs/orpheus-3b-0.1-pretrained` | Hub API cardData |
| Code | `canopyai/Orpheus-TTS`, Apache-2.0; PyPI `orpheus-speech` 0.1.0 depends on `vllm` and `snac` | GitHub API, PyPI |
| Voices | `tara, leah, jess, leo, dan, mia, zac, zoe` (in the authors' order of conversational realism); prompt format `{name}: text` | GitHub README |
| Tags | `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>` | GitHub README |
| Sampling | `repetition_penalty >= 1.1` required for stable output; higher penalty and temperature speak faster | GitHub README |
| Output | 24 kHz through the SNAC 24 kHz codec (`hubertsiuzdak/snac_24khz`, MIT) | README, mlx source |
| Languages | English; a separate multilingual research release (7 language pairs) is not part of this plan | README |
| Latency claim | ~200 ms streaming (~100 ms with input streaming) on the authors' GPU setup | README (claim) |

MLX builds (all `mlx-community`, `apache-2.0`, not gated, converted with mlx-audio 0.0.3):

| Repo | sha | Size |
|---|---|---|
| `mlx-community/orpheus-3b-0.1-ft-bf16` | `3786aac51797181edddeb0d89a8253deaee99aab` | 6.63 GB (sharded) |
| `mlx-community/orpheus-3b-0.1-ft-8bit` | `c8521bf4ea42f8dd96df099219d7871fc9873142` | 3.54 GB |
| `mlx-community/orpheus-3b-0.1-ft-6bit` | `14d41853b1bf1bbee6920a52d66ef877f66986b5` | 2.71 GB |
| `mlx-community/orpheus-3b-0.1-ft-4bit` | `b30aed4ecfa03c0177a5419b51919d14a105f183` | 1.89 GB |
| `mlx-community/snac_24khz` | `556af1cd3b1c5f2d294f6aa9bb886245d7b716ac` (`mit`) | 0.08 GB |

mlx-audio 0.6.4 (`tts/models/llama/llama.py`, model type `llama`):

```python
model.generate(text, voice=None, temperature=0.6, top_p=0.8, split_pattern="\n", max_tokens=1200,
               ref_audio=None, ref_text=None, stream=False, ...)  # repetition_penalty via kwargs, default 1.3
```

- `voice="tara"` is prepended as `"tara: text"`.
- **Two unpinned downloads happen inside the worker:** `SNAC.from_pretrained("mlx-community/snac_24khz")` runs at module import, and `AutoTokenizer.from_pretrained(config.tokenizer_name)` loads the tokenizer from `mlx-community/orpheus-3b-0.1-ft-bf16` by name (the default in `ModelConfig`) even when the 8-bit repo is loaded.
- **`max_tokens=1200` per segment is short:** mlx-audio's streaming code counts 137.5 tokens per second of audio, and SNAC's 7 codes per 12 Hz frame give about 84, so 1200 tokens is roughly 9 to 15 s of speech, while one of our 400-character chunks is about 25 s. Either pass a larger `max_tokens` or cut Orpheus chunks smaller (to be confirmed in the spike).
- Cloning exists in mlx-audio (`ref_audio`+`ref_text` build a SNAC "zeroprompt"), but the README says the fine-tuned model is meant for its eight voices; we do not expose Orpheus cloning (plan 3.4).

CUDA / CPU path: the official package uses vLLM (Linux; its README pins `vllm==0.7.3` after a bad release). Since Orpheus is a plain Llama model that emits audio-code tokens, two other routes exist: transformers `generate` + `snac` (PyTorch, any OS, slowest), or the GGUF files through the `llama-server` we already ship on every platform, decoding the codes with SNAC (`onnx-community/snac_24khz-ONNX`, MIT, sha `f317cdff7efb7b2a63148c9e68bbb0320fe629a1`, decoder 52.6 MB fp32 / 26.3 MB fp16) on onnxruntime, which the portable voice venv already has. The second route is unproven here.

## 4. Things every new model shares

- None of the four does speed or pitch natively. mlx Chatterbox ignores `speed`; VoxCPM2 and Orpheus have none. Speed and pitch stay ffmpeg post-processing (`voice_speak.process_audio`).
- No new pip package is needed on Apple Silicon: mlx-audio 0.6.4, transformers, librosa and scipy are already in the voice venv; every loader above imports only those.
- The audit script (`scripts/audit-catalog-repos.py`) covers the text, image and video catalogs only; the voice and music catalogs, their pinned revisions, and companion repos are not checked by it today.
- `voice_clone.snapshot_dir()` treats a model as ready only when `config.json` and `model.safetensors` exist, so sharded repos (`VoxCPM2-bf16`, `orpheus-...-bf16`) would never show as installed; the 8-bit rows are single-file.
- `voice_text.clean_for_speech()` removes anything shaped like an HTML tag (`</?[a-zA-Z][^>]*>`), so Orpheus `<laugh>` tags are stripped before they reach the model. `[laugh]` survives (the link pattern needs `](`).
- A chunk that starts with `(` is read by VoxCPM2 as a voice description. `clean_for_speech` produces such chunks itself ("(code block omitted)").

## Sources

- Hub API, no token: `https://huggingface.co/api/models/<repo>?blobs=true` and `/api/models?author=<org>&search=<term>` for every repo named above (2026-11-08).
- Model cards: https://huggingface.co/openbmb/VoxCPM2 , https://huggingface.co/ResembleAI/chatterbox , https://huggingface.co/ResembleAI/chatterbox-turbo (and its `added_tokens.json`), https://huggingface.co/unsloth/orpheus-3b-0.1-ft , the mlx-community cards for each MLX repo.
- GitHub: https://github.com/OpenBMB/VoxCPM , https://github.com/resemble-ai/chatterbox , https://github.com/resemble-ai/Perth , https://github.com/canopyai/Orpheus-TTS , https://github.com/hubertsiuzdak/snac , https://github.com/xingchensong/S3Tokenizer , https://github.com/Blaizzy/mlx-audio .
- PyPI JSON: `voxcpm`, `chatterbox-tts`, `orpheus-speech`, `resemble-perth`, `snac`, `s3tokenizer`, `mlx-audio`.
- Local source: `~/.chaosengine/voice-venv/lib/python3.11/site-packages/mlx_audio/` 0.6.4 (`tts/models/{voxcpm2,chatterbox,chatterbox_turbo,llama}`, `tts/utils.py`, `codec/models/snac`).

## 5. P0 spike results (2026-11-09)

Run on the M4 Max (64 GB) with `scripts/measure-voice-models.py --spike`, weights already on disk, its own voice worker (no app, no port). WAVs in `~/.chaosengine/voice/wave2-listening/`.

| Row | s per 10 s | peak GB | idle GB | cold start s | 12-word line s |
|---|---|---|---|---|---|
| chatterbox-turbo (built-in voice) | 2.1 | 1.72 | 1.72 | 2.4 | 0.8 |
| voxcpm2 (design) | 5.22 | 3.8 | 3.8 | 2.0 | 1.9 |
| chatterbox-mtl (default voice) | 3.63 | 3.67 | 3.66 | 2.9 | 1.4 |
| orpheus-3b (tara) | 7.22 | 4.19 | 4.19 | 3.6 | 3.3 |

- **V3 default voice:** the V2 8-bit build's `conds.safetensors` loads into the V3 weights and speaks English, German and Japanese lines (STT gives the English and German back word for word). Shipped as a `defaultVoice` companion (only that 105 KB file is fetched).
- **Orpheus token budget:** a 400-character chunk gives 14.6 s of audio at `max_tokens` 1200 (cut short), 25.4 s at 2400 and 25.2 s at 4000. Shipped: 250-character chunks with 2400 tokens.
- **VoxCPM2 continuation:** median pitch per chunk (librosa pYIN) stays within about 7% when later chunks continue from chunk one (171 to 183 Hz; 142 to 174 Hz with another seed) and jumps when each chunk is designed again (149 to 243 Hz; 142 to 207 Hz). Every chunk's words came back by STT in both cases. Kept. Speaker-embedding similarity (CAM++) was also tried and swung between runs on the same files, so it is not used as evidence.
- **Seeds:** `mx.random.seed` repeats Turbo, VoxCPM2 and Orpheus byte for byte; Chatterbox V3 does not repeat.
- **Turbo clone safety:** mlx-audio's Turbo stores a clone's conditionals on the model, so the next plain request would speak in the cloned voice without a watermark. The worker puts the built-in conditionals back after every call; the built-in voice after a clone is byte-identical to a fresh worker's.
- **Chatterbox V3 clone:** the loader needs the reference as an `mx.array` (a numpy array fails in `expand_dims`).
- **Upstream pin moved:** `openbmb/VoxCPM2` is now at `7936080b` (was `32279eff` on 2026-11-08); the CUDA row pins the new sha.
