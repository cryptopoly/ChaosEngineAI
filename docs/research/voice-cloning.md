# Voice cloning: which model, and what the law expects

Internal research, 2026-11-07, for D-28 (Qwen3-TTS cloning) and D-34 (AI labelling).
**Not legal advice.** This is an engineer's map of the rules so a lawyer can check it quickly.
Licences were read from the Hugging Face API and model cards on the day; laws from the sources linked.

## ELI5

Cloning means: give the app a short recording of someone, and it can say new things in that voice.
- **Model:** ship **Qwen3-TTS Base** (Apache 2.0, runs on Mac through mlx-audio, which we already use).
  It needs about 3 s of audio plus the words spoken in it; we can fill those words in with our own speech-to-text.
  **VoxCPM2** (Apache 2.0, 48 kHz) is the second choice; **Chatterbox** (MIT) is the one with a hidden watermark built in.
- **Law, short version:** making the tool is legal in the UK, EU and US. Misusing it (fraud, fake endorsements,
  sexual deepfakes, election tricks) is the user's offence, *unless* we build or market the tool for misuse.
  From 2 August 2026 the **EU AI Act Art. 50** makes the *maker* of a generative AI system (us, even as free
  open source) mark its audio as AI-made in a machine-readable way. The EU code of practice reads that as
  **signed metadata plus an imperceptible watermark**. Our metadata tag covers half; we should add a watermark
  to cloned voices.
- **Safeguards that are customary and, together, a good defence:** the speaker reads a consent sentence on
  the microphone, the app checks it, consent is stored with the voice, every output is tagged and watermarked,
  cloning is off for remote agents, no celebrity presets, a local log, and clear terms.

## Part 1: models

| Model | Licence (commercial?) | Mac / MLX | Reference audio | Quality notes | Watermark |
|---|---|---|---|---|---|
| **Qwen3-TTS 12Hz Base 0.6B / 1.7B** | Apache 2.0, yes | **mlx-audio** (`mlx-community/Qwen3-TTS-12Hz-*-Base-bf16`), `ref_audio` + `ref_text` | "3-second rapid voice clone"; 10 to 20 s clean speech is better | 10 languages, streaming; strong similarity; already the D-28 pick | None built in |
| **VoxCPM2** (OpenBMB, 2B) | Apache 2.0, yes | mlx-audio (`mlx-community/VoxCPM2-bf16`, 8bit) | Short clip; "ultimate" mode with transcript | 30 languages, 48 kHz output, voice design from text | None built in |
| **Chatterbox Multilingual V3 / Turbo** (Resemble) | MIT, yes | mlx-audio (`mlx-community/chatterbox-multilingual-v3`) | About 5 to 10 s | 0.5B, 23+ languages, emotion control | **PerTh watermark on every output** |
| Fun-CosyVoice 3 / CosyVoice 2 (Alibaba) | Apache 2.0, yes | No mlx-audio entry found | 3 to 10 s + transcript | Good Chinese and English; PyTorch only | None |
| Orpheus 3B | Apache 2.0 (Llama 3.2 base, check its licence) | Community ports | Fine-tune for best clone | Expressive English | None |
| Zonos v0.1 (Zyphra) | Apache 2.0, yes | No | 10 to 30 s | Good, CUDA-centric | None |
| Kyutai Pocket TTS / TTS 1.6B | CC-BY 4.0 (gated click-through) | mlx-audio has a related model (Phonon) | Wav file or exported embedding | Small, fast, English (+French) | None |
| Higgs Audio v2/v3, MOSS-TTS, OmniVoice, LongCat-AudioDiT | Varies, check each | Listed in mlx-audio | Short clip | Newer, larger; evaluate later | None |
| F5-TTS / E2 | **CC-BY-NC 4.0, not commercial** | Ports exist | 5 to 15 s | Good | None |
| Spark-TTS 0.5B | **CC-BY-NC-SA 4.0, not commercial** | mlx-audio | Short | Good | None |
| XTTS v2 (Coqui) | **Coqui Public Model Licence, non-commercial**; company closed, no way to buy a licence | No | 6 s | Dated | None |
| Fish Audio S2 Pro / OpenAudio | **Fish Audio Research Licence, non-commercial** | mlx-audio (Drama 3) | Short | Very good | None |
| IndexTTS-2 | No licence tag on HF card (treat as unclear) | No | Short | Good | None |

Speed: competitors' Mac figures for cloning models are about 1.5 to 2x real time
(`docs/research/m4max-generation-times.md`); Qwen3-TTS 0.6B should be the fastest of the three picks. Not yet measured by us.

**Recommendation.**
1. Ship **Qwen3-TTS 0.6B Base** (fast) and **1.7B Base** (quality) through mlx-audio on Mac, `qwen-tts` / transformers on CUDA.
2. Add an **imperceptible watermark** to every cloned output with Resemble's **PerTh** (MIT, `resemble-perth`)
   or Meta's **AudioSeal** (code MIT; check the weights licence). PerTh is simpler and audio-only. This changes D-34
   ("no hidden watermark yet") for cloned voices only.
3. Second engine later: VoxCPM2 (48 kHz, Apache). Chatterbox if we want a model with the watermark inside it.
4. Never ship F5, Spark, XTTS or Fish weights as defaults (non-commercial). If a user adds them, the Models page shows the licence.

**Fit with our voice worker** (`backend_service/voice_worker.py`, venv from `voice_env.py`, mlx-audio pinned `>=0.6.2,<0.7`):
- `_load_tts("mlx-audio", repo)` already loads any mlx-audio model; Qwen3-TTS loads the same way.
- `synthesize()` is Kokoro-shaped (`voice` prefix and `lang_code`). Add a branch: when the request has
  `refAudioPath` (+ `refText`), call `model.generate(text, ref_audio=..., ref_text=...)` and do not pass `voice`
  (mlx-audio says a speaker name and `ref_audio` together conflict).
- `refText`: transcribe the reference with our STT lane (`transcribe()` in the same worker) and let the user correct it.
- Store the reference, its transcript and the **consent record** together in `voice_library.py`; the watermark and
  metadata tag go on in the parent after the WAV comes back, so every engine gets them.
- Memory: 1.7B bf16 is a few GB; no new gate needed beyond the voice worker's unload.
- Sources: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base , https://github.com/Blaizzy/mlx-audio ,
  https://huggingface.co/openbmb/VoxCPM2 , https://huggingface.co/ResembleAI/chatterbox ,
  https://github.com/resemble-ai/perth , https://github.com/facebookresearch/audioseal ,
  https://huggingface.co/SWivid/F5-TTS , https://huggingface.co/coqui/XTTS-v2 , https://huggingface.co/fishaudio/s2-pro

## Part 2: law

Two roles: **users** (who make and publish audio) and **us** (who write and distribute the software, free, from the UK).
Everything runs locally; we receive no audio. That fact does a lot of work below.

### UK

- **No general "image right" or voice right.** Protection comes from other law:
  - **Passing off** (false endorsement): using a known person's voice to suggest they endorse something
    (*Irvine v Talksport* [2002] EWHC 367 (Ch); *Fenty v Arcadia* [2015] EWCA Civ 3). User liability.
  - **Copyright and performers' rights** (CDPA 1988): a voice is not copyright, but a recording is, and a
    performance is protected by Part II. Cloning from a commercial recording or film can copy a protected
    work or performance. User liability; our terms should say "only use recordings you have rights to".
- **Data protection (UK GDPR / DPA 2018).** A voice recording of an identifiable person is personal data.
  The ICO says biometric data is *special category* only when used to uniquely identify someone; synthesis is
  not identification, but a speaker-match check in a consent flow is. A purely personal user is outside UK GDPR
  (household exemption); a business user is a controller and needs a lawful basis (consent is the safe one).
  **We are not a controller** while no data reaches us; keep it that way (no telemetry with audio, no cloud).
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/guidance-on-biometric-data/key-data-protection-concepts
- **Criminal law aimed at users:** Fraud Act 2006 s2 (false representation, the classic "CEO voice" scam);
  harassment and communications offences; sexually explicit deepfakes (creating one is an offence since
  6 Feb 2026, SOA 2003 ss66E-66F, via Data (Use and Access) Act 2025 s138).
- **Criminal law that could reach a tool maker:** Fraud Act s7 (supplying an article "knowing that it is
  designed or adapted for use" in fraud, or intending it to be) and Serious Crime Act 2007 ss44-46
  (encouraging or assisting). Both need knowledge, design or intent. A general voice tool with consent
  gating and honest marketing is far from that line. **Never market impersonation.**
- **Crime and Policing Act 2026** creates an offence of making or supplying a tool "for use as" an intimate
  image generator (up to 3 years). It is about images, not voice, but it touches our **image and video**
  models: a lawyer should confirm a general-purpose app with a content filter is outside it.
  https://www.kingsleynapley.co.uk/our-insights/articles/deepfakes-to-deletion-orders-tackling-technology-enabled-sexual-offending-in-the-crime-and-policing-act-2026/
- **Online Safety Act 2023:** applies to user-to-user and search services. A local app with no sharing is not one.
  Re-check if we ever add a shared gallery or voice marketplace.
- **Civil accessory liability:** we would be liable for a user's tort only with a "common design"
  (*Fish & Fish v Sea Shepherd* [2015] UKSC 10), i.e. we would have to help and share the purpose.

### EU (applies to us because we offer the app to EU users, even free)

- **GDPR:** same picture as the UK. Art. 9 biometric rules bite on *identification* uses. Household exemption Art. 2(2)(c).
- **AI Act Art. 50.** Free and open-source systems are *not* exempt from Art. 50 (Art. 2(12) carve-out excludes it).
  - **Art. 50(2), providers (us):** outputs of a system that generates synthetic audio, images or video must be
    "marked in a machine-readable format and detectable as artificially generated", as far as technically feasible.
  - **Art. 50(4), deployers (users acting professionally):** must disclose deepfakes, including audio that
    resembles a real person. Lighter duty for evidently artistic or satirical work. Purely personal,
    non-professional use is outside "deployer" (Art. 3(4)).
  - **Dates:** applies from **2 Aug 2026**; systems already on the market before then have until **2 Dec 2026**
    for marking (Digital Omnibus, Reg. (EU) 2026/1744); an interoperability step is reported for **2 Feb 2027** (unverified).
  - **Code of Practice on marking and labelling** (final, 10 June 2026, voluntary but the regulators' yardstick):
    providers mark with **at least two layers, signed metadata and an imperceptible watermark**, offer detection,
    and test robustness. For audio-only deepfakes, deployers put **an audible disclaimer at the start**.
  - Fines for Art. 50 breaches: up to EUR 15M or 3 % of turnover (lower of the two for SMEs, Art. 99).
  - https://artificialintelligenceact.eu/transparency-rules-article-50/ ,
    https://www.lewissilkin.com/insights/2026/07/24/the-eus-new-ai-labelling-rules-what-every-organisation-needs-to-know-102ne35 ,
    https://www.paulweiss.com/insights/client-memos/eu-finalises-transparency-rules-for-ai-generated-content ,
    https://usercentrics.com/knowledge-hub/eu-ai-act-high-risk-delay-article-50-transparency-consent/
- **New Product Liability Directive (EU) 2024/2853** treats software as a product from Dec 2026; free open-source
  software supplied outside a commercial activity is excluded. Relevant if the app ever becomes paid.
- **Denmark** bill L 49 (face and voice imitation rights, takedown) introduced Oct 2026, expected 2027, not in force.
  https://www.euronews.com/2026/10/08/denmark-introduces-a-bill-protecting-peoples-faces-and-voices-from-deepfakes

### US

- **Tennessee ELVIS Act (in force 1 July 2024):** voice is a protected property right; liability also for
  making available a tool whose **"primary purpose or function"** is producing someone's voice without
  authorisation. A general TTS app with consent gating is not that; "clone any star" marketing would push toward it.
  https://www.manatt.com/insights/newsletters/client-alert/tennessees-elvis-act-expands-publicity-rights-for
- **NO FAKES Act of 2026** (S.4591 / H.R.8915): federal digital-replica right with a notice-and-takedown scheme and
  liability for services "primarily designed" to make unlawful replicas. Passed Senate Judiciary June 2026, on the
  Senate calendar, **not law** as of Nov 2026. https://www.congress.gov/bill/119th-congress/senate-bill/4591
- **California:** AB 2602 (contract terms for digital replicas of living performers) and AB 1836 (replicas of
  deceased personalities need estate consent), both from 1 Jan 2025; plus Civil Code 3344 right of publicity.
  New York has similar contract and post-mortem rules. User and studio liability, not tool liability.
  https://www.manatt.com/insights/newsletters/client-alert/california-enacts-a-host-of-new-ai-and-digital-rep
- **FTC:** the 2024 impersonation rule covers impersonating government and businesses. The proposed
  "means and instrumentalities" liability for tool makers was **not adopted** (2026 Federal Register notice);
  whether the individuals ban was finalised is unverified. FTC Act s5 still punishes deceptive claims by us.
  https://public-inspection.federalregister.gov/2026-20143.pdf
- **FCC (Feb 2024):** AI-generated voices in robocalls count as "artificial" voices under the TCPA. User liability.
- **Illinois BIPA** covers voiceprints collected by private entities; we collect none (local only). Keep it that way.

### Platform rules users will meet (verify current text before quoting in the UI)

YouTube, TikTok and Meta require labels on realistic synthetic or altered media; music services ban
unauthorised voice impersonation. Our metadata tag helps the auto-labelling; the watermark helps detection.

### What we are and are not liable for (summary)

- **Not liable** for a user's lawful-looking tool use that they turn to fraud, defamation or passing off, provided
  we did not design, market or knowingly help it (UK accessory and s7 tests; US "primary purpose" tests).
- **Directly responsible** for: our own Art. 50(2) marking duty (EU), honest marketing, licences of the models we
  ship, and not collecting voice data ourselves.

## Part 3: recommended design for ChaosEngineAI

1. **Two paths when making a voice:** "My own voice" or "Someone who agreed". Public figures, minors and
   deceased people: not allowed (terms + UI copy; we cannot detect them, so this is a promise, not a filter).
2. **Recorded consent:** the speaker reads, on the microphone, a sentence with a random code:
   "I, [name], agree that my voice may be copied by AI on this computer for [purpose]. Code 4721."
   Our STT checks the words and the code. Optional local speaker-match against the sample (on device only;
   it is biometric processing, so it runs only after the speaker has agreed in that same recording).
   Upload-only samples still need this live consent recording from the same person.
3. **Consent record** stored with the voice: name, date, purpose, consent WAV, sample hash, app version.
   "Delete this voice" removes sample, record and embeddings (speaker can withdraw).
4. **Every cloned output:** metadata tag (existing, cannot be switched off for cloned voices per D-34),
   **PerTh or AudioSeal watermark**, and an optional "spoken AI disclaimer at the start" switch for professional users (Art. 50(4) helper).
5. **Remote agents (/mcp):** cloning off by default; consent is a required argument tied to a stored record (D-28).
6. **Local audit log** (JSONL in the app data folder, never uploaded): voice id, time, output hash, text length.
7. **No celebrity presets, no "import from URL/YouTube", no shared voice packs** of third parties.
8. **Terms (src/features/terms):** extend the existing "Only clone a voice with the speaker's consent" line with:
   rights to the source recording, no fraud / impersonation / sexual content, user responsible as controller,
   the labelling we apply and that removing it may break the law, no warranty.
9. **Marketing and docs:** describe it as "make a voice from your own recording or a consenting speaker". Never "clone anyone".

## For the lawyer (open questions)

1. Are we a "provider" under AI Act Art. 50(2) for the whole app, or is each model's maker? (We assume we are.)
2. Does metadata + PerTh/AudioSeal meet "as far as technically feasible" for an open-source app users can modify?
3. UK Fraud Act s7 / Serious Crime Act exposure with the design above; wording for terms and marketing.
4. Crime and Policing Act 2026 "intimate image generator" offence versus our image and video features.
5. Whether the on-device speaker-match step makes the *user* process special category data, and what our UI must say.
6. ELVIS Act and NO FAKES (if passed) safe-harbour conditions for an open-source distributor.
7. Whether to keep a UK/EU-only stance or add US-specific terms.
