"""The structured licence a text, image or video catalogue row carries (D0135), in the shape the voice and music
rows already use (``voice_models.py``, ``music_models.py``) so one chip component shows all of them.

Per licence: ``id`` (stable key), ``name`` (the licence's own name, never translated), ``commercial`` (``"yes"``,
``"limited"`` or ``"no"``), ``gated`` (the row's repo asks for a Hugging Face accept before download), ``url`` (the
licence text where the repo's notices give one, the canonical text for the common open licences, otherwise the
upstream model card the licence is read from) and ``summary`` (one plain sentence).

Source: the "Weights you download yourself" tables of ``THIRD_PARTY_NOTICES.md`` (checked 2026-11-24 against each
upstream model card's licence field and, for every restricted licence, the licence text). A community conversion
(GGUF, MLX, 4-bit, diffusers) carries the licence of the upstream weights, even when its own card lists another one
or none, so a conversion row gets the upstream's object. ``gated`` is True only for a row whose own repo is the
upstream gated one (or a row whose note says it is gated); a conversion whose gating nobody checked says False.
A row with no licence here has none we could read from the repo: it is left out, never guessed.
"""

from __future__ import annotations

from typing import Any

COMMERCIAL_VALUES = ("yes", "limited", "no")


def licence(
    id: str, name: str, commercial: str, url: str, summary: str, *, gated: bool = False,
) -> dict[str, Any]:
    return {"id": id, "name": name, "commercial": commercial, "gated": gated, "url": url, "summary": summary}


def as_gated(base: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``base`` for a row whose own repo needs a Hugging Face accept."""
    return {**base, "gated": True}


def as_open(base: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``base`` for a conversion whose own repo is not gated although the upstream's is."""
    return {**base, "gated": False}


def with_url(base: dict[str, Any], url: str, *, gated: bool | None = None) -> dict[str, Any]:
    """A copy of ``base`` pointing at another page (the model card of this row's own family)."""
    return {**base, "url": url, **({} if gated is None else {"gated": gated})}


_OPEN_SUMMARY = "Free to use, change and share, commercially too; keep the licence and notices when you share it."

APACHE_2_0 = licence(
    "apache-2.0", "Apache 2.0", "yes", "https://www.apache.org/licenses/LICENSE-2.0", _OPEN_SUMMARY)
MIT = licence("mit", "MIT", "yes", "https://opensource.org/license/mit", _OPEN_SUMMARY)
BSD_3_CLAUSE = licence(
    "bsd-3-clause", "BSD-3-Clause", "yes", "https://opensource.org/license/bsd-3-clause",
    "Free to use, change and share, commercially too; keep the licence and notices when you share it.")
# Gemma 4 is Apache-2.0; Google's own licence page is the link the notices give.
GEMMA_4 = with_url(APACHE_2_0, "https://ai.google.dev/gemma/docs/gemma_4_license")
OPENMDW_1_1 = licence(
    "openmdw-1.1", "OpenMDW-1.1", "yes", "https://openmdw.ai/license/1-1/",
    "Free to use, change and share, commercially too; keep the licence and notices when you share it.")
NVIDIA_NEMOTRON_OPEN = licence(
    "nvidia-nemotron-open-model", "NVIDIA Nemotron Open Model License", "yes",
    "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/",
    "Commercial use allowed; when you share the weights keep the licence and the NVIDIA line, and suing over the "
    "model ends your licence.")

QWEN_COMMUNITY = licence(
    "qwen-community-1.0", "Qwen Community License 1.0", "limited", "https://huggingface.co/Qwen/Qwen3.8-Flash-Next",
    "Show the model name in a product above 100 million monthly users or US$20 million monthly revenue; a "
    "model-as-a-service or AI work assistant business needs a separate licence from Qwen.")
QWEN_RESEARCH = licence(
    "qwen-research", "Qwen Research Licence", "no", "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
    "Research and evaluation only; no commercial use.")
LLAMA_3_3 = licence(
    "llama-3.3-community", "Llama 3.3 Community License", "limited",
    "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
    "Acceptable use policy; show \"Built with Llama\" when you share it; companies above 700 million monthly users "
    "need a licence from Meta.")
LLAMA_4 = licence(
    "llama-4-community", "Llama 4 Community License", "limited",
    "https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "As Llama 3.3, and the multimodal model's rights are not granted to people or companies based in the European "
    "Union (end users of a product built on it are not affected).")
MINIMAX_M2_7 = licence(
    "minimax-m2.7-non-commercial", "MiniMax M2.7 Non-Commercial License", "no",
    "https://huggingface.co/MiniMaxAI/MiniMax-M2.7",
    "Non-commercial use only; commercial use needs MiniMax's prior written authorisation.")
MINIMAX_M3 = licence(
    "minimax-community", "MiniMax Community License", "limited", "https://huggingface.co/MiniMaxAI/MiniMax-M3",
    "Commercial use needs the \"Built with MiniMax\" line and a one-time notice; above US$20 million yearly revenue "
    "you need MiniMax's written authorisation.")
KIMI_MODIFIED_MIT = licence(
    "modified-mit", "Modified MIT", "yes", "https://huggingface.co/moonshotai/Kimi-K2.6",
    "MIT with one extra rule: a product above 100 million monthly users or US$20 million monthly revenue must show "
    "\"Kimi K2.6\" in its interface.")
LFM_OPEN = licence(
    "lfm-open-1.0", "LFM Open License v1.0", "limited", "https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF",
    "Free for non-commercial and research use; commercial use is limited for organisations at or above US$10 "
    "million annual revenue.")

SD_3_5_COMMUNITY = licence(
    "stability-community", "Stability AI Community Licence", "limited",
    "https://stability.ai/community-license-agreement",
    "Free while you and your affiliates earn under US$1 million a year, then you need a licence from Stability AI; "
    "commercial use means registering with Stability AI.", gated=True)
OPENRAIL_PP_M = licence(
    "creativeml-openrail++-m", "CreativeML OpenRAIL++-M", "yes",
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
    "Commercial use allowed, with use restrictions that pass on to anyone you share the model with.")
FLUX_1_DEV_NC = licence(
    "flux-1-dev-non-commercial", "FLUX.1 [dev] Non-Commercial License", "no",
    "https://huggingface.co/black-forest-labs/FLUX.1-dev", "Non-commercial use only.", gated=True)
FLUX_2_NC = licence(
    "flux-non-commercial", "FLUX Non-Commercial License", "no", "https://huggingface.co/black-forest-labs/FLUX.2-dev",
    "Non-commercial use only.", gated=True)
FLUX_KOMMUNITY = licence(
    "flux-kommunity-1.0", "FLUX Kommunity licence v1.0", "limited",
    "https://huggingface.co/black-forest-labs/flux-3-image-klein",
    "Free for non-commercial use; commercial use of the outputs only below US$5 million annual revenue.", gated=True)

LTXV_RAIL_M = licence(
    "ltxv-rail-m", "LTXV RAIL-M", "yes", "https://huggingface.co/Lightricks/LTX-Video",
    "Commercial use allowed, with use restrictions that pass on to anyone you share the model with.")
LTX_2_COMMUNITY = licence(
    "ltx-2-community", "LTX-2 Community License", "limited", "https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2",
    "Free below US$10 million annual revenue (affiliates counted together), a paid licence above it; use "
    "restrictions apply.")
LTX_2_X_COMMUNITY = licence(
    "ltx-2.x-community", "LTX-2.x Community License", "limited",
    "https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x",
    "Organisations at or above US$10 million annual revenue need a paid licence for anything but non-commercial "
    "use; use restrictions apply.", gated=True)
HUNYUAN_COMMUNITY = licence(
    "tencent-hunyuan-community", "Tencent Hunyuan Community License", "limited",
    "https://huggingface.co/tencent/HunyuanVideo",
    "Does not apply in the European Union, the United Kingdom or South Korea; acceptable use policy.")
COGVIDEOX = licence(
    "cogvideox", "CogVideoX License", "limited", "https://huggingface.co/THUDM/CogVideoX-5b",
    "Free for academic research; commercial use needs registering for a free basic commercial licence, up to 1 "
    "million visits a month.")
