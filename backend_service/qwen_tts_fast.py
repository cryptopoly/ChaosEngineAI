"""Faster Qwen3-TTS generation on a GPU, inside the Qwen worker (imports nothing from ``backend_service``).

Each talker step of ``qwen-tts`` 0.1.1 asks its 5-layer code predictor for 15 more codes through a full
``transformers`` ``generate()`` call: argument validation, logits-processor setup, stopping criteria and cache set-up
run 12.5 times per second of speech, around 15 tiny forward passes each. On a fast GPU that host work, not the
GPU, sets the speed. ``fast_code_predictor`` swaps that call, on this one model instance, for a plain loop that does
the same thing in the same order (same forward, same temperature then top-k, same ``torch.multinomial`` draw), so a
fixed seed gives the same codes. ``cached_clone_prompt`` keeps the reference clip's speaker embedding and codes per
clip and transcript, so cloning the same voice again skips encoding it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

PATCH_FLAG = "_chaosengine_fast_predictor"


def sample_next(logits: Any, do_sample: bool, top_k: int | None, top_p: float | None, temperature: float | None) -> Any:
    """The next token as ``transformers`` 4.57 ``_sample`` picks it: float32 logits, temperature, top-k, top-p,
    softmax and one multinomial draw (argmax when not sampling)."""
    import torch  # type: ignore[import-not-found]

    scores = logits.to(dtype=torch.float32)
    if not do_sample:
        return torch.argmax(scores, dim=-1)
    if temperature is not None and float(temperature) != 1.0:
        scores = scores / float(temperature)
    if top_k is not None and int(top_k) != 0:
        k = min(int(top_k), scores.size(-1))
        kth = torch.topk(scores, k)[0][..., -1, None]
        scores = scores.masked_fill(scores < kth, -float("inf"))
    if top_p is not None and float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative <= (1 - float(top_p))
        remove[..., -1:] = 0
        scores = scores.masked_fill(remove.scatter(1, sorted_indices, remove), -float("inf"))
    probs = torch.nn.functional.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def make_fast_generate(predictor: Any, original: Callable[..., Any],
                       notes: list[str] | None = None) -> Callable[..., Any]:
    """A drop-in for ``predictor.generate`` covering the one call the talker makes (embeddings in, a fixed number of
    new codes, sampling settings); any other call goes to ``original``. If the loop fails, it notes why in ``notes``
    and that call and every later one run through ``original``."""
    state = {"failed": False}

    def generate(*args: Any, inputs_embeds: Any = None, max_new_tokens: int | None = None, do_sample: Any = None,
                 top_p: Any = None, top_k: Any = None, temperature: Any = None, **kwargs: Any) -> Any:
        extra = set(kwargs) - {"output_hidden_states", "return_dict_in_generate"}
        if state["failed"] or args or inputs_embeds is None or not max_new_tokens or extra:
            return original(*args, inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens, do_sample=do_sample,
                            top_p=top_p, top_k=top_k, temperature=temperature, **kwargs)
        try:
            return SimpleNamespace(sequences=_loop(predictor, inputs_embeds, int(max_new_tokens),
                                                   lambda logits: sample_next(logits, bool(do_sample), top_k, top_p,
                                                                              temperature)))
        except Exception as exc:  # noqa: BLE001 - any failure means the library's own path
            state["failed"] = True
            if notes is not None:
                notes.append(f"Fast Qwen3-TTS code predictor is off, using the library's generate: {exc}")
            return original(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens, do_sample=do_sample,
                            top_p=top_p, top_k=top_k, temperature=temperature, **kwargs)

    generate._chaosengine_fast = True  # type: ignore[attr-defined]
    return generate


def _loop(predictor: Any, inputs_embeds: Any, steps: int, sample: Callable[[Any], Any]) -> Any:
    """Prefill the embeddings, then one forward pass per code, reusing one KV cache (what ``generate`` does)."""
    import torch  # type: ignore[import-not-found]
    from transformers.cache_utils import DynamicCache  # type: ignore[import-not-found]

    with torch.inference_mode():
        cache = DynamicCache()
        length = int(inputs_embeds.shape[1])
        device = inputs_embeds.device
        out = predictor(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(length, device=device))
        tokens = []
        for i in range(steps):
            token = sample(out.logits[:, -1, :])
            tokens.append(token)
            if i + 1 == steps:
                break
            out = predictor(input_ids=token[:, None], past_key_values=cache, use_cache=True,
                            cache_position=torch.tensor([length + i], device=device),
                            generation_steps=out.generation_steps)
        return torch.stack(tokens, dim=1)


def fast_code_predictor(model: Any, notes: list[str] | None = None) -> bool:
    """Install the plain loop on the talker's code predictor of a loaded ``Qwen3TTSModel``; False when the model
    does not have the layout this was written against (then nothing changes)."""
    predictor = getattr(getattr(getattr(model, "model", None), "talker", None), "code_predictor", None)
    if predictor is None or not callable(getattr(predictor, "generate", None)):
        return False
    if getattr(predictor, PATCH_FLAG, False):
        return True
    predictor.generate = make_fast_generate(predictor, predictor.generate, notes=notes)
    setattr(predictor, PATCH_FLAG, True)
    return True


_prompts: dict[tuple[Any, ...], Any] = {}


def cached_clone_prompt(model: Any, ref_audio: str, ref_text: str | None, x_vector_only: bool,
                        stat: Callable[[str], Any] | None = None) -> Any:
    """``model.create_voice_clone_prompt`` once per reference clip (path and modification time), transcript and
    mode; the reference's speaker embedding and codes are reused for later clones of the same voice."""
    import os

    mtime = (stat or os.stat)(ref_audio).st_mtime
    key = (id(model), ref_audio, mtime, ref_text or "", bool(x_vector_only))
    if key not in _prompts:
        if len(_prompts) >= 8:
            _prompts.clear()
        _prompts[key] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text,
                                                        x_vector_only_mode=bool(x_vector_only))
    return _prompts[key]


def clear_prompts() -> None:
    _prompts.clear()
