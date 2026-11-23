"""Speech-to-text on an NVIDIA GPU inside the voice worker: the ``transformers-asr`` backend.

Runs in the ASR worker (``voice_client.get_asr_worker``): the voice venv's Python and PyTorch (the speech-to-text GPU
add-on, ``voice_env_asr``) with the add-on's own Transformers first on ``PYTHONPATH``, because the speech GPU
add-on pins an older Transformers for Chatterbox. Transformers 5.24 carries every model here natively, so one loader
serves five families:

* ``qwen3-asr``: Qwen3-ASR 1.7B / 0.6B (the ``-hf`` conversions), an audio LLM with language detection.
* ``parakeet-tdt``: Parakeet TDT 0.6B v3 (``ParakeetForTDT``), 25 European languages, no language setting.
* ``nemotron-asr``: Nemotron 3.5 ASR streaming 0.6B (``Nemotron3_5AsrForRNNT``), a language prompt or ``auto``.
* ``granite-speech``: Granite Speech 5.0 2B, an audio LLM steered by a written prompt.
* ``nemotron-transcribe``: Nemotron 3.5 Transcribe 30B-A4B, an audio LLM (needs a large GPU).

Every model loads from the local snapshot the backend downloaded (``local_files_only``). Long recordings are cut
into windows at the quietest moment near each window's end; each window becomes one segment.

Word timings (the request's ``words`` flag): Parakeet TDT and Nemotron ASR give them natively (the frame at which each
token was emitted, ``processor.decode(..., durations=...)``); the audio LLMs (Qwen3-ASR, Granite, Nemotron Transcribe)
have none, so their transcript is aligned to the audio by Qwen3-ForcedAligner-0.6B (Apache-2.0,
``Qwen3ASRForTokenClassification``), which the backend names in ``alignerSource`` once it is downloaded. Nemotron ASR's
own timings run late by its streaming latency, so it prefers the aligner too and keeps its own when that is absent.
Imports nothing from ``backend_service``.
"""

from __future__ import annotations

from typing import Any, Callable

SAMPLE_RATE = 16_000
# Audio LLMs see at most this much audio per request (Qwen3-ASR's feature extractor works in 30 s windows); the
# transducers (TDT, RNN-T) take longer windows, as parakeet-mlx does on the Mac.
_WINDOW_S = {"parakeet-tdt": 120.0, "nemotron-asr": 120.0}
_LLM_WINDOW_S = 30.0
_LLM_FAMILIES = ("qwen3-asr", "granite-speech", "nemotron-transcribe")
# Granite Speech is steered by a prompt; the model card's own wording.
_GRANITE_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."

# Qwen3-ASR answers with the language's English name; the app speaks ISO codes.
_LANGUAGE_CODES = {
    "english": "en", "chinese": "zh", "cantonese": "yue", "japanese": "ja", "korean": "ko", "german": "de",
    "french": "fr", "spanish": "es", "portuguese": "pt", "italian": "it", "russian": "ru", "arabic": "ar",
    "dutch": "nl", "polish": "pl", "turkish": "tr", "vietnamese": "vi", "thai": "th", "indonesian": "id",
    "hindi": "hi", "swedish": "sv", "danish": "da", "finnish": "fi", "czech": "cs", "greek": "el",
    "hungarian": "hu", "romanian": "ro", "ukrainian": "uk", "malay": "ms", "filipino": "fil", "persian": "fa",
    "macedonian": "mk",
}

# Qwen3-ForcedAligner-0.6B speaks eleven languages; Japanese and Korean need the nagisa / soynlp packages to split
# words, which the add-on does not install, so those two are left out (no word timings, with a note).
_ALIGNER_LANGUAGES = {
    "en": "English", "zh": "Chinese", "yue": "Cantonese", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "es": "Spanish",
}
_CJK_ALIGNER = ("zh", "yue")  # no spaces between words: the aligner's own units are characters

_model: dict[str, Any] = {"key": None, "model": None, "processor": None, "device": None}
_aligner: dict[str, Any] = {"source": None, "model": None, "processor": None}


def unload() -> None:
    _model.update(key=None, model=None, processor=None, device=None)
    _aligner.update(source=None, model=None, processor=None)
    try:
        import gc

        gc.collect()
        import torch  # type: ignore[import-not-found]

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch absent or no GPU
        pass


def _auto_class(family: str) -> Any:
    import transformers  # type: ignore[import-not-found]

    names = {
        "qwen3-asr": "AutoModelForMultimodalLM",
        "nemotron-transcribe": "AutoModelForMultimodalLM",
        "parakeet-tdt": "AutoModelForTDT",
        "nemotron-asr": "AutoModelForRNNT",
        "granite-speech": "AutoModelForSpeechSeq2Seq",
    }
    if family not in names:
        raise ValueError(f"Unknown speech-to-text model family {family!r}.")
    try:
        return getattr(transformers, names[family])
    except AttributeError as exc:  # an older Transformers on the path
        raise RuntimeError(
            f"Transformers {transformers.__version__} cannot run this model. Untick and tick \"More speech-to-text "
            "models on the NVIDIA GPU\" under Voice runtime > Add-ons to update it."
        ) from exc


def load(family: str, source: str) -> tuple[Any, Any, str]:
    """(model, processor, device) for a local snapshot, cached until another model is asked for."""
    key = (family, source)
    if _model["key"] == key:
        return _model["model"], _model["processor"], _model["device"]
    unload()  # the old model leaves GPU memory before the next one is mapped in
    import torch  # type: ignore[import-not-found]
    from transformers import AutoProcessor  # type: ignore[import-not-found]

    cuda = torch.cuda.is_available()
    device = "cuda:0" if cuda else "cpu"
    if family in _LLM_FAMILIES:
        # The audio LLMs ship bf16 weights; fp16 on a card without bf16, float32 on the CPU.
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
    else:
        dtype = "auto"  # the transducers ship float32 (2.5 GB); half precision is not checked against their decoders
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    model = _auto_class(family).from_pretrained(source, dtype=dtype, device_map=device, local_files_only=True)
    model.eval()
    _model.update(key=key, model=model, processor=processor, device=device)
    return model, processor, device


def _word(start: float, end: float, text: str) -> dict[str, Any]:
    return {"start": round(float(start), 2), "end": round(float(end), 2), "text": text.strip()}


def _load_aligner(source: str, device: str) -> tuple[Any, Any]:
    """The forced aligner from its local snapshot, cached beside the transcription model."""
    if _aligner["source"] == source:
        return _aligner["model"], _aligner["processor"]
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForTokenClassification, AutoProcessor  # type: ignore[import-not-found]

    cuda = device != "cpu"
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(source, dtype=dtype, device_map=device, local_files_only=True)
    model.eval()
    _aligner.update(source=source, model=model, processor=processor)
    return model, processor


def _align_window(source: str, device: str, audio: Any, text: str, language: str | None) -> list[dict[str, Any]]:
    """Word timings (seconds from the start of ``audio``) of one window's transcript. Each word is a whitespace-separated
    piece of ``text`` as written (punctuation kept); the aligner works on the same pieces without punctuation."""
    import torch  # type: ignore[import-not-found]

    model, processor = _load_aligner(source, device)
    name = _ALIGNER_LANGUAGES.get(language or "")
    inputs, word_lists = processor.prepare_forced_aligner_inputs(audio=audio, transcript=text, language=name)
    inputs = inputs.to(device, model.dtype)
    with torch.inference_mode():
        logits = model(**inputs).logits
    aligned = processor.decode_forced_alignment(
        logits=logits, input_ids=inputs["input_ids"], word_lists=word_lists,
        timestamp_token_id=model.config.timestamp_token_id,
    )[0]
    pieces = text.split()
    if language not in _CJK_ALIGNER and pieces:
        counts = [len(processor.split_words_for_alignment(piece, name)) for piece in pieces]
        if sum(counts) == len(aligned):
            words: list[dict[str, Any]] = []
            at = 0
            for piece, count in zip(pieces, counts):
                if count == 0 and words:  # a lone dash or other mark: part of the word before it
                    words[-1]["text"] += " " + piece
                elif count:
                    words.append(_word(aligned[at]["start_time"], aligned[at + count - 1]["end_time"], piece))
                at += count
            return words
    return [_word(item["start_time"], item["end_time"], item["text"]) for item in aligned]


def _aligned_words(source: str, device: str, audio: Any, text: str, language: str | None,
                   offset: float) -> tuple[list[dict[str, Any]] | None, str | None]:
    """``(words, None)`` for one window (seconds in the file), or ``(None, reason)``: the language is not one the aligner
    speaks, or the alignment failed (the transcript is still returned without word timings)."""
    if language and language.split("-")[0].lower() not in _ALIGNER_LANGUAGES:
        return None, "language"
    try:
        words = _align_window(source, device, audio, text, language.split("-")[0].lower() if language else None)
    except Exception as exc:  # noqa: BLE001 - a missing word time must not lose the transcript
        import sys

        print(f"asr: forced alignment failed: {exc}", file=sys.stderr, flush=True)
        return None, "failed"
    return [{**w, "start": round(w["start"] + offset, 2), "end": round(w["end"] + offset, 2)} for w in words], None


def _token_words(tokens: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    """Words from a transducer's tokens (``{"token", "start", "end"}`` in seconds): a token that starts with a space
    starts a word, the others continue it. Punctuation does not move a word's end."""
    words: list[list[Any]] = []
    for token in tokens:
        piece = str(token["token"])
        if not piece:
            continue
        if not words or piece[:1].isspace():
            words.append([token["start"], token["end"], piece])
        else:
            words[-1][2] += piece
            if any(ch.isalnum() for ch in piece):
                words[-1][1] = token["end"]
    return [_word(start + offset, end + offset, text) for start, end, text in words if text.strip()]


def _transducer_window_words(family: str, model: Any, processor: Any, device: str, audio: Any, language: str | None,
                             offset: float) -> tuple[str, list[dict[str, Any]]]:
    """Text and native word timings (seconds in the file, from ``offset``) of one window for a transducer."""
    import torch  # type: ignore[import-not-found]

    kwargs: dict[str, Any] = {"sampling_rate": SAMPLE_RATE}
    if family == "nemotron-asr":
        kwargs["language"] = _nemotron_language(processor, language)
    with torch.inference_mode():
        inputs = processor(audio, **kwargs).to(device, dtype=model.dtype)
        output = model.generate(**inputs, return_dict_in_generate=True)
        text, stamps = processor.decode(output.sequences, durations=output.durations, skip_special_tokens=True)
    text = text[0] if isinstance(text, list) else str(text)
    return text.strip(), _token_words(stamps[0], offset)


def windows(samples: Any, seconds: float) -> list[tuple[int, int]]:
    """Cut points for ``samples`` in windows of at most ``seconds``: each cut sits at the quietest 100 ms in the last
    quarter of its window, so a cut seldom splits a word. Returns ``[(start, end), ...]`` sample indices."""
    import numpy as np

    total = len(samples)
    size = int(seconds * SAMPLE_RATE)
    if total <= size:
        return [(0, total)] if total else []
    hop = SAMPLE_RATE // 10
    spans: list[tuple[int, int]] = []
    start = 0
    while total - start > size:
        search_from = start + size * 3 // 4
        search_to = start + size
        best, best_energy = search_to, None
        for frame in range(search_from, search_to - hop + 1, hop):
            energy = float(np.mean(np.square(samples[frame:frame + hop])))
            if best_energy is None or energy < best_energy:
                best, best_energy = frame + hop // 2, energy
        spans.append((start, best))
        start = best
    spans.append((start, total))
    return spans


def _request_windows(samples: Any, seconds: float, spans: Any) -> list[tuple[int, int]]:
    """The windows to transcribe: the whole recording, or (speaker labels) each given ``[start, end]`` span in
    seconds, every span cut into windows of its own so a segment never mixes two speakers."""
    if not spans:
        return windows(samples, seconds)
    out: list[tuple[int, int]] = []
    for start_s, end_s in spans:
        start = max(0, int(float(start_s) * SAMPLE_RATE))
        end = min(len(samples), int(float(end_s) * SAMPLE_RATE))
        if end - start >= SAMPLE_RATE // 10:  # under 100 ms holds no word
            out.extend((start + a, start + b) for a, b in windows(samples[start:end], seconds))
    return out


def _qwen_language(language: str | None) -> str | None:
    if not language:
        return None
    base = language.split("-")[0].lower()
    if base not in set(_LANGUAGE_CODES.values()):
        raise ValueError(f"Unsupported language '{language}' for Qwen3-ASR.")
    return base


def _nemotron_language(processor: Any, language: str | None) -> str:
    """Nemotron's prompt key for an ISO code (``de`` -> ``de-DE``); ``auto`` when none is set."""
    if not language:
        return "auto"
    prompts = getattr(processor, "prompt_dictionary", None) or {}
    if language in prompts:
        return language
    base = language.split("-")[0].lower()
    match = next((key for key in prompts if str(key).split("-")[0].lower() == base), None)
    if match is None:
        raise ValueError(f"Unsupported language '{language}' for Nemotron 3.5 ASR.")
    return str(match)


def _run_window(family: str, model: Any, processor: Any, device: str, audio: Any, language: str | None) -> tuple[str, str | None]:
    """Text and (when the model says) the ISO language of one window."""
    import torch  # type: ignore[import-not-found]

    with torch.inference_mode():
        if family == "qwen3-asr":
            inputs = processor.apply_transcription_request(audio=audio, language=_qwen_language(language))
            inputs = inputs.to(device, model.dtype)
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            parsed = processor.decode(output[:, inputs["input_ids"].shape[1]:], return_format="parsed")[0]
            name = str(parsed.get("language") or "").strip().lower()
            return str(parsed.get("transcription") or "").strip(), _LANGUAGE_CODES.get(name, language)
        if family == "nemotron-transcribe":
            inputs = processor.apply_transcription_request(audio).to(device, model.dtype)
            output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            text = processor.decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return (text[0] if isinstance(text, list) else str(text)).strip(), language
        if family == "granite-speech":
            tokenizer = processor.tokenizer
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": _GRANITE_PROMPT}], tokenize=False, add_generation_prompt=True
            )
            inputs = processor(prompt, torch.from_numpy(audio).unsqueeze(0), device=device, return_tensors="pt").to(device)
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(model.dtype)
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
            new_tokens = output[:, inputs["input_ids"].shape[-1]:]
            return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip(), language
        kwargs: dict[str, Any] = {"sampling_rate": SAMPLE_RATE}
        if family == "nemotron-asr":
            kwargs["language"] = _nemotron_language(processor, language)
        inputs = processor(audio, **kwargs)
        inputs = inputs.to(device, dtype=model.dtype)
        output = model.generate(**inputs, return_dict_in_generate=True)
        text = processor.decode(output.sequences, skip_special_tokens=True)
        return (text[0] if isinstance(text, list) else str(text)).strip(), language


def _log_peak(family: str, seconds: float) -> None:
    """The process's own GPU peak since the model loaded, to the worker log (whole-card readings are useless while other
    programs share the GPU)."""
    try:
        import sys

        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            gb = 1024 ** 3
            print(f"asr: {family} {seconds:.1f}s audio, peak allocated {torch.cuda.max_memory_allocated() / gb:.2f} GiB, "
                  f"peak reserved {torch.cuda.max_memory_reserved() / gb:.2f} GiB", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a diagnostic only
        pass


def transcribe(request: dict[str, Any], decode_audio: Callable[..., Any]) -> dict[str, Any]:
    """Text and one segment per window. ``language`` is an ISO code or empty (detect, where the model can)."""
    family = str(request.get("family") or "")
    if request.get("task") == "translate":
        raise ValueError("Translation to English needs a Whisper model.")
    language = request.get("language") or None
    if family in ("parakeet-tdt", "granite-speech"):
        language = None  # these models take no language setting; they detect it themselves
    model, processor, device = load(family, str(request["source"]))
    audio = decode_audio(str(request["path"]), SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    seconds = _WINDOW_S.get(family, _LLM_WINDOW_S)
    segments: list[dict[str, Any]] = []
    detected: str | None = None
    want_words = bool(request.get("words"))
    aligner = request.get("alignerSource")
    timings: set[str] = set()
    unavailable: str | None = None
    # Parakeet's own timings are close to the audio (about 90 ms on the RTX 4090 run); Nemotron ASR's are late by its
    # streaming latency (320 ms), so it takes the aligner's when that is downloaded and keeps its own otherwise.
    own_timings = family == "parakeet-tdt" or (family == "nemotron-asr" and not aligner)
    for start, end in _request_windows(audio, seconds, request.get("spans")):
        chunk = audio[start:end]
        offset = start / SAMPLE_RATE
        words: list[dict[str, Any]] | None = None
        if want_words and own_timings:
            text, words = _transducer_window_words(family, model, processor, device, chunk, language, offset)
            found = language
            if words:
                timings.add("native")
        else:
            text, found = _run_window(family, model, processor, device, chunk, language)
            if text and want_words:
                if aligner:
                    words, unavailable = _aligned_words(aligner, device, chunk, text, found or language, offset)
                    if words:
                        timings.add("aligner")
                elif family in _LLM_FAMILIES:
                    unavailable = "aligner-missing"
                if not words and family == "nemotron-asr":  # the aligner could not place them: the model's own times
                    words = _transducer_window_words(family, model, processor, device, chunk, language, offset)[1]
                    if words:
                        timings.add("native")
        detected = detected or found
        if text:
            segment: dict[str, Any] = {"start": round(offset, 2), "end": round(end / SAMPLE_RATE, 2), "text": text}
            if words:
                segment["words"] = words
            segments.append(segment)
    _log_peak(family, duration)
    cjk = (detected or language or "") in ("zh", "ja", "yue")
    if not cjk:  # Nemotron ASR joins its sentences with two spaces
        for segment in segments:
            segment["text"] = " ".join(segment["text"].split())
    text = ("" if cjk else " ").join(s["text"] for s in segments)
    reply = {"text": text.strip(), "language": detected or language, "segments": segments, "durationS": round(duration, 2)}
    if want_words:
        # ``wordTimings``: where the words came from ("native" or "aligner"), None when there are none;
        # ``wordsUnavailable``: why not ("aligner-missing", "language", "failed").
        reply["wordTimings"] = ("aligner" if "aligner" in timings else "native") if timings else None
        reply["wordsUnavailable"] = None if timings else unavailable
    return reply
