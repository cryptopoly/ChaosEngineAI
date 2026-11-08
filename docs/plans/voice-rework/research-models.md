# Voice rework: model and capability research

Date: 2026-11-08. Sources are model cards, the Hugging Face Hub API, the Qwen3-TTS repo, and the mlx-audio 0.6.1 source installed in `.venv` (the ground truth for what our Mac path can call). Licence ids are copied from the repos. Nothing here was run on the owner's Mac: every speed and RAM figure is marked measured (from upstream) or **unmeasured**.

## 1. Qwen3-TTS (Qwen / Alibaba)

Released 2026-01-22 (paper arXiv 2601.15621). Source: https://github.com/QwenLM/Qwen3-TTS . All Apache-2.0. No newer release listed; 25 Hz variants appear in the paper tables but are not released ("in the near future").

### Variants and repo ids

| Variant | Upstream repo | What it does | MLX builds (mlx-community) |
|---|---|---|---|
| 0.6B Base | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | clone from a short clip; fine-tune base | `-0.6B-Base-{bf16,8bit,6bit,5bit,4bit}` |
| 1.7B Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | clone, better likeness | `-1.7B-Base-{bf16,8bit,6bit,5bit,4bit}` |
| 0.6B CustomVoice | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 9 preset speakers | `-0.6B-CustomVoice-{bf16,8bit,6bit,5bit,4bit}` |
| 1.7B CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 preset speakers + natural-language style `instruct` | `-1.7B-CustomVoice-{bf16,8bit,6bit,5bit,4bit}` |
| 1.7B VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | invent a voice from a text description (no 0.6B VoiceDesign exists) | `-1.7B-VoiceDesign-{bf16,8bit,6bit,5bit,4bit}` |
| Tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | codec encode/decode; bundled inside the MLX repos | n/a |

All 25 mlx-community repos: `mlx-community/Qwen3-TTS-12Hz-<size>-<Variant>-<quant>`, licence apache-2.0 (Hub API, 2026-11-08). Real repo sizes from the Hub: 0.6B Base bf16 2.52 GB; 0.6B CustomVoice bf16 2.50 / 8bit 1.97 / 4bit 1.69 GB; 1.7B CustomVoice and VoiceDesign bf16 4.52 / 8bit 3.08 / 4bit 2.31 GB; 1.7B Base 8bit 3.10 GB. (We already pin the two Base bf16 repos in `backend_service/catalog/voice_models.py`.)

Everything is also available as PyTorch (`pip install -U qwen-tts`, Python 3.12 recommended, FlashAttention 2 optional): that is the CUDA path. vLLM-Omni supports offline inference only.

### Shared facts
- 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian. Any speaker can speak any language; each speaker is best in its native one.
- Dialects: Beijing (Dylan), Sichuan (Eric).
- Streaming: yes on all variants. Upstream claims first packet after one character, ~97 ms end-to-end latency (CUDA claim, unmeasured here). mlx-audio: `stream=True, streaming_interval=0.32` (12.5 frames/s, so 0.32 s is about 4 tokens); default interval is 2.0 s.
- Max length: none stated; evaluation used `max_new_tokens=2048` (at 12.5 Hz that is about 164 s of audio). mlx-audio default `max_tokens` per segment comes from the checkpoint config (function default 4096). Long text is chunked by `split_pattern="\n"` in mlx-audio, so **we must chunk by sentence ourselves**.
- Sampling (mlx-audio): `temperature` (default 0.9), `top_k` (50), `top_p` (1.0), `repetition_penalty` (1.05; forced to at least 1.5 in clone/ICL mode), `max_tokens`. Unset values come from each checkpoint's `generation_config.json`. No `seed` argument: call `mx.random.seed(n)` before generating (our inference, not documented). Upstream PyTorch passes any HF `generate` kwargs.
- **Speed and pitch: not supported by Qwen3-TTS.** mlx-audio's `speed` is documented "not directly supported yet" and the fast path is disabled when it is not 1.0. The only rate control is natural language in `instruct` (CustomVoice/VoiceDesign) or post-processing (section 4).
- Multi-speaker: no multi-speaker script mode. Use one call per line with different speakers (mlx-audio `batch_generate` does this in one batched pass; README bench, 6-bit, short prompt: batch 1 = 20.8 tok/s = 1.67x realtime, 3.9 GB; batch 4 = 4.26x; batch 8 = 5.45x, 4.1 GB. Measured by mlx-audio authors on unspecified hardware).
- Fine-tuning: Base models are the fine-tune base (`finetuning/` folder upstream). Not relevant yet.

### Built-in speakers (CustomVoice, both sizes)
Vivian (bright, slightly edgy young female, zh), Serena (warm gentle young female, zh), Uncle_Fu (seasoned male, low mellow, zh), Dylan (youthful Beijing male, zh-Beijing), Eric (lively Chengdu male, slightly husky, zh-Sichuan), Ryan (dynamic male, strong rhythmic drive, en), Aiden (sunny American male, clear midrange, en), Ono_Anna (playful female, ja), Sohee (warm female, rich emotion, ko). Only two native English voices, both male: that is the main weakness of CustomVoice for English "speak" use, and the reason to consider VoiceDesign (see below) or Kokoro for English female voices.

### How each variant is called

Upstream PyTorch (`qwen_tts`):
```python
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16)
wavs, sr = model.generate_custom_voice(text=..., language="English", speaker="Ryan", instruct="...")   # instruct optional
wavs, sr = model.generate_voice_design(text=..., language="English", instruct="Female, 30s, warm, low pitch ...")  # VoiceDesign model
wavs, sr = model.generate_voice_clone(text=..., language="English", ref_audio="clip.wav", ref_text="transcript")    # Base model
prompt = model.create_voice_clone_prompt(ref_audio=..., ref_text=..., x_vector_only_mode=False)   # reusable clone
```
Lists are accepted for batching. `ref_audio` takes a path, URL, base64 or `(numpy, sr)`. `model.get_supported_speakers()` / `get_supported_languages()` exist.

mlx-audio 0.6.1 (what we ship on Mac; verified in `.venv/.../tts/models/qwen3_tts/`):
```python
from mlx_audio.tts.utils import load_model
m = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")
for r in m.generate_custom_voice(text=, speaker="Ryan", language="English", instruct="Very happy and excited.", temperature=, top_k=, top_p=, repetition_penalty=, max_tokens=, stream=, streaming_interval=): r.audio  # mx.array, 24 kHz
for r in m.generate_voice_design(text=, language="English", instruct="A cheerful young female voice with high pitch ...")
for r in m.generate(text=, ref_audio=path_or_mx, ref_text="exact transcript")   # Base
```
`m.generate(...)` also routes by model type: VoiceDesign needs `instruct`; CustomVoice needs `voice=` (speaker); Base needs `ref_audio`+`ref_text` and rejects a `voice`. `lang_code` is "auto" or a language name. `m.supported_speakers` lists speakers. `batch_generate(texts=, voices=, instructs=, ...)` for several lines at once.

### Instruct (style/emotion) rules
- 1.7B CustomVoice: free-text `instruct` ("Very happy and excited", "speak in a very angry tone", pace, whisper, etc.). Chinese example in the card: "用特别愤怒的语气说".
- **0.6B CustomVoice: mlx-audio silently sets `instruct = None`** ("For 0.6B models, instruct is not supported", source line ~2167). The upstream card shows no `instruct` support for 0.6B either. So style control needs the 1.7B.
- VoiceDesign: `instruct` is the voice description (age, gender, pitch, tone, accent, pace, emotion), required.
- Base: no `instruct` and no presets.

### Cloning
- Needs `ref_audio` + `ref_text` (exact transcript) by default. mlx-audio only enters the high-quality "ICL" mode when both are given and the speech tokenizer has an encoder; with audio only, upstream offers `x_vector_only_mode=True` (speaker embedding only, lower quality). Our `voice_worker.synthesize_cloned` already passes both and no `voice`.
- Card says "3-second" cloning; our catalog note says good likeness from 10-20 s. Both are claims, quality is **unmeasured** here.
- Design-then-clone: make a VoiceDesign clip once, then clone from it with Base for a stable reusable voice (documented upstream). This is the way to get a persistent "designed" voice, since VoiceDesign itself re-rolls the voice every call.

### Answer for the owner's "Qwen3-TTS for normal speech" request
Yes, CustomVoice 1.7B (or 0.6B) is plain TTS with preset voices plus optional style text; VoiceDesign gives unlimited voices. Catch: English presets are only Ryan and Aiden (male). For plain English speech with a female voice, either use VoiceDesign with a fixed description, or keep Kokoro as the fast lane.

## 2. Kokoro (hexgrad/Kokoro-82M)

https://huggingface.co/hexgrad/Kokoro-82M . Apache-2.0, 82M params (model file ~330 MB; fp32), v1.0 (2025-01-27). Our app uses it via mlx-audio on Mac and kokoro-onnx elsewhere.
- Languages (9 via `lang_code`): `a` American English, `b` British English, `j` Japanese, `z` Mandarin, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `p` Brazilian Portuguese. 54 voices total.
- Voices (name prefix = language+gender): American 20 (11F/9M) top graded `af_heart` A, `af_bella` A-, `af_nicole` B-; others C+ or below. British 8 (`bf_emma` B-). Japanese 5, Mandarin 8 (all D), Spanish 3, French 1 (`ff_siwis` B-), Hindi 4, Italian 2, Portuguese 3. The grades come from VOICES.md (https://huggingface.co/hexgrad/Kokoro-82M/raw/main/VOICES.md). Many voices are weak; a curated shortlist (af_heart, af_bella, af_nicole, bf_emma, am_michael, am_fenrir, am_puck, bm_george, bm_fable, ff_siwis) beats showing all 54.
- Controls: `voice`, `speed` (mlx-audio divides predicted durations by speed; usable roughly 0.5 to 2.0), `lang_code`, `split_pattern`. No emotion, style, pitch or cloning controls.
- Voice blending: voices are style tensors, so blending is a weighted average of two voice tensors (the official pipeline accepts a list/comma-separated voices in the reference implementation; mlx-audio `voice.py` loads by name). Not documented on the model card; **needs a test** before offering it.
- Phoneme override: markdown-style `[word](/IPA/)` in the text (card example `[Kokoro](/kˈOkəɹO/)`), stress marks ˈ ˌ. The pipeline also supports `[text](+1)`/`(-1)` stress adjustments in the misaki G2P (from memory of misaki docs, unverified).
- Per-chunk limit: roughly 510 phoneme tokens per forward; the pipeline splits long text on `split_pattern`. Sentence-chunk upstream of it.
- Why it sounds robotic: 82M param model, no prosody conditioning, one flat style per voice. Better voices and speed tuning help only a little. The fix is a bigger model, not Kokoro settings.
- Speed: very fast (far faster than realtime on M-series; **unmeasured on this Mac**). RAM under 1 GB.

## 3. Other open TTS models (permissive licence only)

Licence ids from the Hub API or card. "Mac" = an MLX conversion loadable by mlx-audio 0.6.1 (the local install has a loader for each).

| Model | Licence | Size | Emotion/style | Cloning | Languages | Mac | Verdict |
|---|---|---|---|---|---|---|---|
| Chatterbox (ResembleAI/chatterbox, multilingual V3) | mit | ~0.5B | `exaggeration` (0.5 default; 0.7+ for dramatic) and `cfg_weight` (0.3 for fast refs; 0 for cross-language) | zero-shot, `audio_prompt_path`, no transcript | 23 | `mlx-community/chatterbox-fp16`, `Chatterbox-TTS-{fp16,4bit}` | Best emotion dial among MIT models; cloning without a transcript is easier for users. Adds PerTh watermark to all output. Strong candidate for "style" lane. |
| Chatterbox-Turbo (ResembleAI/chatterbox-turbo) | mit | ~350M | paralinguistic tags `[laugh]`, `[cough]`, `[chuckle]` (more unnamed) | zero-shot, ~10 s clip | English only | `mlx-community/chatterbox-turbo-fp16` | Fast English with laughter/cough tags. Good small default for chatty English. |
| Orpheus 3B (canopylabs/orpheus-3b-0.1-ft) | apache-2.0 | 3B (gated repo) | tags `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>`; 8 voices tara, leah, jess, leo, dan, mia, zac, zoe | weak (prompt examples) | English (multilingual research preview, 7 pairs) | `mlx-community/orpheus-3b-0.1-ft-bf16` | Most natural conversational English presets; heavy (3B ~6 GB bf16) and the official repo is gated (use the mlx-community mirror). |
| Dia 1.6B (nari-labs/Dia-1.6B-0626) | apache-2.0 | ~2B | `[S1]/[S2]` dialogue + `(laughs) (sighs) (coughs) (gasps) (clears throat)` etc. | audio + transcript prefix | English only | mlx-audio loader exists (Hub id for the mlx-community build not confirmed) | Two-speaker dialogue is unique; English only, ~10 GB VRAM on CUDA. Dia2-2B (apache-2.0) also exists. |
| VoxCPM2 (openbmb/VoxCPM2) | apache-2.0 | 2.3B | voice design from a description in parentheses before the text; style guidance in controllable cloning | controllable (clip + optional style) and "ultimate" (clip + transcript) | 30, no language tag | `mlx-community/VoxCPM2-bf16` | The only permissive model besides Qwen with voice design and cloning; 48 kHz output, streaming; ~8 GB VRAM on CUDA, RTF ~0.3 on a 4090 (claims, unmeasured here). Worth a catalog look as the second design model. |
| Soprano 1.1 (ekwek/Soprano-1.1-80M) | apache-2.0 | 80M | none | no | English | `mlx-community/Soprano-80M-bf16` | Tiny and fast; alternative to Kokoro, not a quality jump. |
| KittenTTS mini 0.8 | apache-2.0 | small | none | no | English | via mlx-audio | CPU-friendly fallback only. |
| IndexTTS-2 (IndexTeam/IndexTTS-2) | custom "bilibili Model Use License Agreement" (no SPDX id; its older Index-TTS is apache-2.0) | n/a | strongest emotion control: emotion reference clip, 8-value emotion vector, emotion from text; duration control in 2.5 (`duration_factor` 0.5 to 2.0) | zero-shot | en, zh (2.5 adds ja, es, ar) | `mlx-community/index-tts2-mlx` | Not an SPDX permissive licence: flag for the owner, exclude from default catalog. |
| Higgs Audio v2 (bosonai) | "other" (custom) | 6B | scene description, `[SPEAKER*]` multi-speaker | yes | 4 | mlx-audio loader exists | Custom licence and 6B size; skip. |
| VibeVoice 1.5B / Realtime-0.5B (microsoft) | mit | 3B / ~1B | none documented | not offered (out of scope per card) | en, zh (Realtime: en) | mlx-audio has VibeVoice (ASR confirmed) | Long multi-speaker (4 speakers, 90 min) podcasts; no cloning. Skip for now. |
| Sesame CSM-1B | apache-2.0 | 2B | none; speaker id `[0]`, conversational context | not offered | English | `mlx-community/csm-1b` | Context-aware but no controls; skip. |
| Kyutai TTS 1.6B (en/fr) / pocket-tts | cc-by-4.0 | 1.8B / small | none | only pre-made voice embeddings | en, fr (pocket: 6) | `mlx-community/pocket-tts` | CC-BY needs attribution; streaming-first. Low priority. |
| OmniVoice (k2-fsa) | model weights `cc-by-nc` (non-commercial) | 0.6B | voice design attributes, `[laughter]` | yes | 600+ | n/a | Excluded: non-commercial weights. |
| Fish / OpenAudio S1-mini, S2-pro | cc-by-nc-sa-4.0 / fish-audio-research-license | 0.5B / 4.6B | rich `(angry) (excited) (whispering)` markers | yes | 13+ | mlx-audio has Fish Drama 3 | Excluded: non-commercial licences. Its marker syntax is the model for UI hints though. |
| Spark-TTS 0.5B | cc-by-nc-sa-4.0 | 0.5B | pitch/speed tokens | yes | en, zh | `mlx-community/Spark-TTS-0.5B-bf16` | Excluded: non-commercial. |
| Voxtral-4B-TTS (Mistral) | cc-by-nc-4.0 | 4B | presets | yes | 9 | `mlx-community/Voxtral-4B-TTS-2603-mlx-*` | Excluded: non-commercial. |
| F5-TTS (SWivid) | cc-by-nc-4.0 weights | n/a | none | needs ref text | en, zh | n/a | Excluded: non-commercial. |
| XTTS v2 (Coqui) | CPML (non-commercial) | n/a | none | yes | 17 | n/a | Excluded; stated from memory, not re-checked. |

## 4. Speech-to-text and audio post-processing

### STT (what mlx-audio 0.6.1 lists; our app already runs Nemotron 3.5 ASR streaming via `voice_worker.py`)
Whisper and Distil-Whisper, Qwen3-ASR (has `language=`), Qwen3-ForcedAligner (word-level timings given text: useful for lip sync, FU-140, and for karaoke highlight), Parakeet (`stream=True`), Nemotron 3.5 (streaming), Voxtral and Voxtral Realtime (`transcription_delay_ms`), VibeVoice-ASR (hotword `context`, speaker-aware), Canary, Moonshine, MMS, Granite Speech, MOSS-Transcribe-Diarize (diarisation). Source: https://github.com/Blaizzy/mlx-audio . On CUDA, faster-whisper/Whisper remain the usual path (not re-researched here).
Useful for clone UX: auto-fill `ref_text` by transcribing the uploaded reference clip (Qwen3-TTS needs the exact transcript). The app already has a transcript path (`voice_transcript.py`).

### Post controls the app could expose cheaply
Qwen3-TTS has no speed or pitch, and Kokoro has no pitch, so these are post-processing steps on the generated waveform:

| Control | Cheap way | Notes |
|---|---|---|
| Speed (time stretch, pitch preserved) | ffmpeg `atempo` (0.5 to 2.0 per filter, chain for more) | Present in Homebrew ffmpeg 8 on this Mac. Fine for 0.8 to 1.3x; artefacts beyond. |
| Pitch shift (tempo preserved) | ffmpeg `rubberband` filter (`pitch=`), or `asetrate=sr*f,aresample=sr,atempo=1/f` | Homebrew ffmpeg here has **no** rubberband filter (checked); the asetrate+atempo chain works anywhere. Python `librosa.effects.pitch_shift`/`pyrubberband` are alternatives (extra dependency). Keep to about +/-3 semitones before voices sound artificial. |
| Loudness normalise | ffmpeg `loudnorm` (EBU R128, I=-16) or peak normalise in numpy | `loudnorm` present locally. One-pass numpy peak/RMS normalise is enough for clips. |
| Trim silence / fades | ffmpeg `silenceremove`, `afade` | Hides clipped starts and ends from chunked synthesis. |
| Sentence gaps | insert N ms silence when joining chunks | Do it in numpy; controls pacing for long text. |
All of these are filter graphs PyAV can also run inside the voice venv (the worker already uses `av`); the main app venv has no `av`. See `docs/plans/ffmpeg-download.md` for the ffmpeg packaging decision.

## 5. Capability matrix

Y = native, ~ = partial/indirect, - = no. "Mac" = MLX build exists. All Qwen cells verified from the card and mlx-audio source.

| Model | Plain speak | Preset voices | Style instruction | Emotion tags | Voice design | Cloning | Speed | Pitch | Multilingual | Streaming | Long text chunking | Mac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kokoro 82M | Y | Y (54) | - | - | - | - | Y | - | Y (9) | ~ (per chunk) | caller (split_pattern) | Y |
| Qwen3 0.6B Base | ~ (needs a ref) | - | - | - | - | Y (clip+text) | post only | post only | Y (10) | Y | caller | Y |
| Qwen3 1.7B Base | ~ | - | - | - | - | Y (clip+text) | post only | post only | Y (10) | Y | caller | Y |
| Qwen3 0.6B CustomVoice | Y | Y (9) | - (dropped in mlx-audio) | - | - | - | post only | post only | Y (10) | Y | caller | Y |
| Qwen3 1.7B CustomVoice | Y | Y (9) | Y (free text) | via text | - | - | post only (or ask in text) | post only | Y (10) | Y | caller | Y |
| Qwen3 1.7B VoiceDesign | Y | - (designed) | Y (description) | via text | Y | - (use design-then-clone) | post only | in description | Y (10) | Y | caller | Y |
| Chatterbox MTL | Y | default voice | ~ (exaggeration/cfg) | ~ | - | Y (no transcript) | ~ (via cfg) | post only | Y (23) | ~ | caller | Y |
| Chatterbox Turbo | Y | default voice | - | Y (`[laugh]`...) | - | Y | post only | post only | - (en) | ~ | caller | Y |
| Orpheus 3B | Y | Y (8) | - | Y (8 tags) | - | ~ | post only | post only | ~ (en + preview) | Y | caller | Y |
| Dia 1.6B | Y | - | - | Y (nonverbals) | - | Y | post only | post only | - (en) | - | caller | ~ |
| VoxCPM2 | Y | - | Y (clone style) | ~ | Y | Y | post only | in description | Y (30) | Y | caller | Y |
| Soprano / Kitten | Y | few | - | - | - | - | ~ | - | - (en) | ~ | caller | Y |

## 6. Recommendations

Defaults per job (all Apache-2.0 or MIT, all with MLX builds; RAM numbers are file sizes plus runtime overhead; speed is **unmeasured on this Mac** except where cited):

| Job | Recommended default | Alternative | RAM (weights) | Speed |
|---|---|---|---|---|
| Speak (plain, fast) | Kokoro 82M (keep, with a curated shortlist) | Qwen3 0.6B CustomVoice 8bit (1.97 GB) for multilingual with more natural prosody | <1 GB / ~2 GB | Kokoro: very fast, unmeasured. Qwen3: mlx-audio bench shows about 1.7x realtime at batch 1 for 1.7B 6-bit (their hardware); 0.6B should be faster; both unmeasured here. |
| Style (emotion/instruction) | Qwen3 1.7B CustomVoice 8bit (3.08 GB; bf16 4.52 GB) | Chatterbox (exaggeration dial) or Chatterbox-Turbo (laugh tags) for English | ~3.5 to 5 GB resident | ~1 to 2x realtime est. on M-series, unmeasured. |
| Clone | Qwen3 0.6B Base bf16 default, 1.7B Base for quality (as today) | Chatterbox (no transcript needed; MIT) | 2.5 / 4.5 GB | already in app; unmeasured here |
| Design | Qwen3 1.7B VoiceDesign 8bit (3.08 GB) | VoxCPM2 (2.3B, 48 kHz) | ~3.5 GB | unmeasured |

Things the owner most needs to know:
1. Qwen3-TTS has three families, not one: Base (clone), CustomVoice (9 presets + style), VoiceDesign (invent a voice). We ship only Base. The MLX repos already exist for all of them in 4/5/6/8-bit and bf16, all apache-2.0, so adding speak, style and design is mostly catalog plus a worker call per mode (`generate_custom_voice`, `generate_voice_design`).
2. Style `instruct` works only on 1.7B CustomVoice and VoiceDesign. The 0.6B CustomVoice accepts it but mlx-audio drops it silently, so the UI must grey the style box on 0.6B.
3. Qwen3-TTS has no speed, pitch or seed controls (mlx-audio `speed` is a no-op). Offer speed/pitch/normalise as post-processing (ffmpeg `atempo`/`asetrate`/`loudnorm`; Homebrew ffmpeg here lacks `rubberband`). Long text must be chunked by us; stream with `streaming_interval`.
4. English presets are weak: only Ryan and Aiden (male). For a good English female voice use VoiceDesign (save the result via design-then-clone) or keep Kokoro `af_heart`/`af_bella`. Kokoro sounds robotic because of model size, and has no style knobs beyond speed.
5. Excluded for licence ids that are not permissive: Fish/OpenAudio, Spark, Voxtral-TTS, F5, OmniVoice weights, XTTS, Higgs v2 and IndexTTS-2 (custom). Worth adding after Qwen3: Chatterbox (MIT, emotion dial) and VoxCPM2 (Apache, design and cloning, 30 languages).

Open checks before building: confirm Kokoro voice blending works on our mlx-audio path; measure real tokens/s and RSS for 0.6B/1.7B CustomVoice on the M4 Max; check whether the CUDA path (`qwen-tts` package) needs a separate venv (current catalog says CUDA is a follow-up).

## Sources
- https://github.com/QwenLM/Qwen3-TTS and https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
- https://huggingface.co/api/models?author=mlx-community&search=Qwen3-TTS (licences, sizes via `?blobs=true`)
- https://github.com/Blaizzy/mlx-audio ; local `.venv/lib/python*/site-packages/mlx_audio/tts/models/qwen3_tts/{README.md,qwen3_tts.py}` (mlx-audio 0.6.1)
- https://huggingface.co/hexgrad/Kokoro-82M and /raw/main/VOICES.md
- https://huggingface.co/ResembleAI/chatterbox , /chatterbox-turbo
- https://huggingface.co/canopylabs/orpheus-3b-0.1-ft , https://github.com/canopyai/Orpheus-TTS
- https://huggingface.co/nari-labs/Dia-1.6B-0626
- https://huggingface.co/openbmb/VoxCPM2 , https://huggingface.co/k2-fsa/OmniVoice
- https://github.com/index-tts/index-tts , https://huggingface.co/IndexTeam/IndexTTS-2
- https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base , https://huggingface.co/microsoft/VibeVoice-1.5B
- https://huggingface.co/fishaudio/openaudio-s1-mini , https://huggingface.co/sesame/csm-1b , https://huggingface.co/kyutai/tts-1.6b-en_fr
- Hub API licence checks for Soprano, Kitten, Spark, Voxtral-TTS, pocket-tts, Dia2, s2-pro (2026-11-08)
