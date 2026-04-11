from __future__ import annotations

import base64
import json
import ipaddress
import os
import platform
import socket
import subprocess
import sys
import time
import tomllib
import uuid
import urllib.request
import urllib.parse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
import re
import threading
from threading import RLock
from typing import Any

import signal
import asyncio
import psutil
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend_service.inference import RuntimeController, get_backend_capabilities
from backend_service.image_runtime import (
    ImageGenerationConfig,
    ImageRuntimeManager,
    validate_local_diffusers_snapshot,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
APP_STARTED_AT = time.time()
HF_SNAPSHOT_DOWNLOAD_HELPER = (
    "import sys\n"
    "from huggingface_hub import snapshot_download\n"
    "snapshot_download(repo_id=sys.argv[1], resume_download=True)\n"
)
DEFAULT_PORT = int(os.getenv("CHAOSENGINE_PORT", "8876"))
DEFAULT_HOST = os.getenv("CHAOSENGINE_HOST", "127.0.0.1")
MODEL_FAMILIES: list[dict[str, Any]] = [
    {
        "id": "gemma-4",
        "name": "Gemma 4",
        "provider": "Google",
        "headline": "New multimodal Gemma family for reasoning, coding, and on-device workflows.",
        "summary": "Strong general-purpose family with compact edge variants and a larger A4B option for workstation-class use.",
        "description": (
            "Gemma 4 covers compact edge deployment through larger multimodal reasoning models. "
            "It is a good fit when you want a modern generalist with long context and strong coding support."
        ),
        "updatedLabel": "Updated 3 days ago",
        "popularityLabel": "323k downloads",
        "likesLabel": "58 likes",
        "badges": ["Staff pick", "Vision", "Reasoning"],
        "capabilities": ["reasoning", "vision", "coding", "multilingual"],
        "defaultVariantId": "google/gemma-4-26B-A4B-it",
        "variants": [
            {
                "id": "google/gemma-4-E4B-it",
                "name": "Gemma 4 E4B Instruct",
                "repo": "google/gemma-4-E4B-it",
                "link": "https://huggingface.co/google/gemma-4-E4B-it",
                "paramsB": 4.5,
                "sizeGb": 5.4,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["vision", "reasoning", "tool-use"],
                "note": "Compact Gemma 4 variant for laptops and small desktops.",
                "contextWindow": "256K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "google/gemma-4-12B-it",
                "name": "Gemma 4 12B Instruct",
                "repo": "google/gemma-4-12B-it",
                "link": "https://huggingface.co/google/gemma-4-12B-it",
                "paramsB": 12.0,
                "sizeGb": 14.8,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["vision", "reasoning", "coding"],
                "note": "Higher quality Gemma 4 option for larger local rigs.",
                "contextWindow": "256K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "google/gemma-4-26B-A4B-it",
                "name": "Gemma 4 26B A4B Instruct",
                "repo": "google/gemma-4-26B-A4B-it",
                "link": "https://huggingface.co/google/gemma-4-26B-A4B-it",
                "paramsB": 26.0,
                "sizeGb": 28.1,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["vision", "reasoning", "coding", "agents"],
                "note": "Large multimodal Gemma 4 variant for workstation-class setups.",
                "contextWindow": "256K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "mlx-community/gemma-4-26b-a4b-5bit",
                "name": "Gemma 4 26B A4B MLX 5-bit",
                "repo": "mlx-community/gemma-4-26b-a4b-5bit",
                "link": "https://huggingface.co/mlx-community/gemma-4-26b-a4b-5bit",
                "paramsB": 26.0,
                "sizeGb": 18.5,
                "format": "MLX",
                "quantization": "5-bit",
                "capabilities": ["vision", "reasoning", "coding", "agents"],
                "note": "Community MLX conversion for Apple Silicon when you want a ready-to-run Gemma 4 variant.",
                "contextWindow": "256K",
                "launchMode": "direct",
                "backend": "mlx",
            },
        ],
        "readme": [
            "Gemma 4 is built for reasoning, coding, and multimodal understanding.",
            "Small variants target local deployment; larger variants suit workstation-class hardware.",
            "Official transformer repos still route through conversion, while community MLX variants can run directly when available.",
        ],
    },
    {
        "id": "qwen-3-5",
        "name": "Qwen 3.5",
        "provider": "Qwen",
        "headline": "Hybrid reasoning family with long context and strong agent support.",
        "summary": "Useful when you want modern reasoning and coding performance with a wide spread of model sizes.",
        "description": (
            "Qwen 3.5 spans compact and MoE-heavy variants with long context support. "
            "It fits well for agent-style workflows, tool use, and coding-heavy chats."
        ),
        "updatedLabel": "Updated this month",
        "popularityLabel": "Featured family",
        "likesLabel": "Qwen official",
        "badges": ["Reasoning", "Coding", "Long context"],
        "capabilities": ["reasoning", "coding", "tool-use", "vision"],
        "defaultVariantId": "Qwen/Qwen3.5-9B",
        "variants": [
            {
                "id": "Qwen/Qwen3.5-4B",
                "name": "Qwen3.5 4B",
                "repo": "Qwen/Qwen3.5-4B",
                "link": "https://huggingface.co/Qwen/Qwen3.5-4B",
                "paramsB": 4.0,
                "sizeGb": 5.1,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["reasoning", "coding", "tool-use"],
                "note": "Smaller Qwen 3.5 variant with strong utility for everyday local work.",
                "contextWindow": "262K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "Qwen/Qwen3.5-9B",
                "name": "Qwen3.5 9B",
                "repo": "Qwen/Qwen3.5-9B",
                "link": "https://huggingface.co/Qwen/Qwen3.5-9B",
                "paramsB": 9.0,
                "sizeGb": 10.9,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["reasoning", "coding", "vision", "video", "tool-use"],
                "note": "Balanced Qwen 3.5 option for serious local chat, code, and agent tasks.",
                "contextWindow": "262K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "mlx-community/Qwen3.5-9B-4bit",
                "name": "Qwen3.5 9B MLX 4-bit",
                "repo": "mlx-community/Qwen3.5-9B-4bit",
                "link": "https://huggingface.co/mlx-community/Qwen3.5-9B-4bit",
                "paramsB": 9.0,
                "sizeGb": 6.0,
                "format": "MLX",
                "quantization": "4-bit",
                "capabilities": ["reasoning", "coding", "vision", "video", "tool-use"],
                "note": "Community MLX conversion for Apple Silicon with a much quicker local launch path.",
                "contextWindow": "262K",
                "launchMode": "direct",
                "backend": "mlx",
            },
            {
                "id": "lmstudio-community/Qwen3.5-9B-GGUF",
                "name": "Qwen3.5 9B GGUF",
                "repo": "lmstudio-community/Qwen3.5-9B-GGUF",
                "link": "https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF",
                "paramsB": 9.0,
                "sizeGb": 5.8,
                "format": "GGUF",
                "quantization": "Q4_K_M",
                "capabilities": ["reasoning", "coding", "tool-use"],
                "note": "Community GGUF pack with ready-made quantizations for quick llama.cpp runs.",
                "contextWindow": "262K",
                "launchMode": "direct",
                "backend": "llama.cpp",
            },
            {
                "id": "Qwen/Qwen3.5-35B-A3B-FP8",
                "name": "Qwen3.5 35B A3B",
                "repo": "Qwen/Qwen3.5-35B-A3B-FP8",
                "link": "https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8",
                "paramsB": 35.0,
                "sizeGb": 22.8,
                "format": "Transformers",
                "quantization": "FP8",
                "capabilities": ["reasoning", "coding", "vision", "agents"],
                "note": "Sparse MoE-style Qwen 3.5 variant tuned for high-end local or server workflows.",
                "contextWindow": "262K",
                "launchMode": "convert",
                "backend": "mlx",
            },
        ],
        "readme": [
            "Qwen 3.5 is a reasoning-first family with long context and strong coding support.",
            "Thinking mode is model-native, so it works well for agent-style threads and tool-rich prompts.",
            "The catalog now mixes official transformer repos with community GGUF and MLX variants so you can go straight to a runnable format when one exists.",
        ],
    },
    {
        "id": "nemotron-3-nano",
        "name": "Nemotron 3 Nano",
        "provider": "NVIDIA",
        "headline": "Compact reasoning-first family aimed at efficient local deployment.",
        "summary": "Good fit when you want a modern small model that still feels capable in chat and tool use.",
        "description": (
            "Nemotron 3 Nano focuses on local efficiency and reasoning utility. "
            "The GGUF variant is ready to use directly in the local llama.cpp path, while the BF16 weights can feed conversion workflows."
        ),
        "updatedLabel": "Updated 2 weeks ago",
        "popularityLabel": "New release",
        "likesLabel": "NVIDIA official",
        "badges": ["Popular", "Reasoning", "Local-first"],
        "capabilities": ["reasoning", "tool-use", "chat"],
        "defaultVariantId": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        "variants": [
            {
                "id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
                "name": "Nemotron 3 Nano 4B GGUF",
                "repo": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
                "link": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
                "paramsB": 4.0,
                "sizeGb": 3.1,
                "format": "GGUF",
                "quantization": "Q4_K_M",
                "capabilities": ["reasoning", "chat", "tool-use"],
                "note": "Direct-load GGUF path for fast local trials and thread switching.",
                "contextWindow": "128K",
                "launchMode": "direct",
                "backend": "llama.cpp",
            },
            {
                "id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                "name": "Nemotron 3 Nano 4B BF16",
                "repo": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                "link": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                "paramsB": 4.0,
                "sizeGb": 8.2,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["reasoning", "chat", "tool-use"],
                "note": "Official BF16 weights if you want to convert into an MLX-friendly local artifact.",
                "contextWindow": "128K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                "name": "Nemotron 3 Nano 30B A3B FP8",
                "repo": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                "link": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                "paramsB": 30.0,
                "sizeGb": 35.0,
                "format": "Transformers",
                "quantization": "FP8",
                "capabilities": ["reasoning", "thinking", "tool-use", "coding", "agents"],
                "note": "Larger official Nemotron 3 Nano checkpoint with configurable reasoning traces and very long context.",
                "contextWindow": "1M",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF",
                "name": "Nemotron 3 Nano 30B A3B GGUF",
                "repo": "lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF",
                "link": "https://huggingface.co/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF",
                "paramsB": 30.0,
                "sizeGb": 24.5,
                "format": "GGUF",
                "quantization": "Q4_K_M",
                "capabilities": ["reasoning", "thinking", "tool-use", "coding", "agents"],
                "note": "Community GGUF conversion for the larger Nemotron 3 Nano release when you want a ready llama.cpp path.",
                "contextWindow": "1M",
                "launchMode": "direct",
                "backend": "llama.cpp",
            },
        ],
        "readme": [
            "Nemotron 3 Nano is built for compact local deployments without falling back to obviously weak outputs.",
            "The family now includes both the smaller 4B checkpoints and the newer 30B A3B release.",
            "Official checkpoints remain available for conversion, while community GGUF packs give you a direct runtime path.",
        ],
    },
    {
        "id": "devstral-small",
        "name": "Devstral Small",
        "provider": "Mistral AI",
        "headline": "Agentic coding model family tuned for software engineering tasks.",
        "summary": "Best when the conversation is code-heavy and you want a thread-specific model for repo work.",
        "description": (
            "Devstral focuses on software engineering workflows and tool-using assistants. "
            "The GGUF release is directly usable, and the BF16 release can feed conversion-oriented setups."
        ),
        "updatedLabel": "Updated 3 weeks ago",
        "popularityLabel": "Trending with coding users",
        "likesLabel": "Mistral official",
        "badges": ["Coding", "Agents", "Tool use"],
        "capabilities": ["coding", "agents", "tool-use"],
        "defaultVariantId": "mistralai/Devstral-Small-2507_gguf",
        "variants": [
            {
                "id": "mistralai/Devstral-Small-2507_gguf",
                "name": "Devstral Small 2507 GGUF",
                "repo": "mistralai/Devstral-Small-2507_gguf",
                "link": "https://huggingface.co/mistralai/Devstral-Small-2507_gguf",
                "paramsB": 24.0,
                "sizeGb": 15.7,
                "format": "GGUF",
                "quantization": "Q4_K_M",
                "capabilities": ["coding", "agents", "tool-use"],
                "note": "Direct GGUF path for local coding threads and repo-centric workflows.",
                "contextWindow": "128K",
                "launchMode": "direct",
                "backend": "llama.cpp",
            },
            {
                "id": "mistralai/Devstral-Small-2507",
                "name": "Devstral Small 2507 BF16",
                "repo": "mistralai/Devstral-Small-2507",
                "link": "https://huggingface.co/mistralai/Devstral-Small-2507",
                "paramsB": 24.0,
                "sizeGb": 29.4,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["coding", "agents", "tool-use"],
                "note": "Official BF16 weights for conversion-oriented local setups.",
                "contextWindow": "128K",
                "launchMode": "convert",
                "backend": "mlx",
            },
        ],
        "readme": [
            "Devstral is tuned for software engineering and tool-using workflows.",
            "Use the GGUF variant when you want the quickest path into local code threads.",
            "Use the BF16 repo when you want to convert into an MLX artifact instead of running GGUF.",
        ],
    },
    {
        "id": "qwen3-coder",
        "name": "Qwen3 Coder",
        "provider": "Qwen",
        "headline": "Code-specialised Qwen3 with strong agentic and tool-use support.",
        "summary": "Purpose-built coding model with function calling, long context, and thinking modes.",
        "description": (
            "Qwen3 Coder is optimised for software engineering workflows with agentic tool use, "
            "large context windows, and configurable thinking depth. Great for local coding assistants."
        ),
        "updatedLabel": "Updated 1 week ago",
        "popularityLabel": "185k downloads",
        "likesLabel": "42 likes",
        "badges": ["Coding", "Agents", "Tool-use"],
        "capabilities": ["coding", "agents", "tool-use", "reasoning", "thinking"],
        "defaultVariantId": "Qwen/Qwen3-Coder-Next-8B-Instruct",
        "variants": [
            {
                "id": "Qwen/Qwen3-Coder-Next-8B-Instruct",
                "name": "Qwen3 Coder Next 8B",
                "repo": "Qwen/Qwen3-Coder-Next-8B-Instruct",
                "link": "https://huggingface.co/Qwen/Qwen3-Coder-Next-8B-Instruct",
                "paramsB": 8.0,
                "sizeGb": 9.6,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["coding", "agents", "tool-use", "thinking"],
                "note": "Compact coding model with agentic capabilities. Ideal for local dev assistants.",
                "contextWindow": "256K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "Qwen/Qwen3-Coder-Next-32B-Instruct",
                "name": "Qwen3 Coder Next 32B",
                "repo": "Qwen/Qwen3-Coder-Next-32B-Instruct",
                "link": "https://huggingface.co/Qwen/Qwen3-Coder-Next-32B-Instruct",
                "paramsB": 32.0,
                "sizeGb": 38.4,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["coding", "agents", "tool-use", "reasoning", "thinking"],
                "note": "Full-size coding model for workstation-class Apple Silicon.",
                "contextWindow": "256K",
                "launchMode": "convert",
                "backend": "mlx",
            },
        ],
        "readme": [
            "Qwen3 Coder is purpose-built for software engineering with function calling and agentic workflows.",
            "Use the 8B variant for laptops; the 32B for workstations with 64GB+ RAM.",
        ],
    },
    {
        "id": "qwen-2-5",
        "name": "Qwen 2.5",
        "provider": "Qwen",
        "headline": "Versatile general-purpose family with chat, coding, and multilingual variants.",
        "summary": "Mature, well-tested family spanning 0.5B to 72B with strong community support.",
        "description": (
            "Qwen 2.5 is a proven workhorse family that covers everything from tiny edge models "
            "to large multilingual reasoners. Excellent GGUF availability in the community."
        ),
        "updatedLabel": "Updated 2 weeks ago",
        "popularityLabel": "890k downloads",
        "likesLabel": "156 likes",
        "badges": ["Popular", "Coding", "Multilingual"],
        "capabilities": ["chat", "coding", "multilingual", "reasoning", "tool-use"],
        "defaultVariantId": "Qwen/Qwen2.5-7B-Instruct",
        "variants": [
            {
                "id": "Qwen/Qwen2.5-3B-Instruct",
                "name": "Qwen 2.5 3B Instruct",
                "repo": "Qwen/Qwen2.5-3B-Instruct",
                "link": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
                "paramsB": 3.0,
                "sizeGb": 3.6,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["chat", "coding", "multilingual"],
                "note": "Ultra-compact variant for quick prototyping and edge deployments.",
                "contextWindow": "128K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "name": "Qwen 2.5 7B Instruct",
                "repo": "Qwen/Qwen2.5-7B-Instruct",
                "link": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
                "paramsB": 7.0,
                "sizeGb": 8.4,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["chat", "coding", "reasoning", "multilingual"],
                "note": "Well-rounded 7B for general chat and coding on 16GB+ machines.",
                "contextWindow": "128K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "name": "Qwen 2.5 Coder 7B",
                "repo": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "link": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
                "paramsB": 7.0,
                "sizeGb": 8.4,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["coding", "tool-use", "agents"],
                "note": "Code-focused 7B with function calling and agentic support.",
                "contextWindow": "128K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "Qwen/Qwen2.5-32B-Instruct",
                "name": "Qwen 2.5 32B Instruct",
                "repo": "Qwen/Qwen2.5-32B-Instruct",
                "link": "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct",
                "paramsB": 32.0,
                "sizeGb": 38.4,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["chat", "coding", "reasoning", "multilingual", "tool-use"],
                "note": "Large Qwen 2.5 for workstation setups needing quality and breadth.",
                "contextWindow": "128K",
                "launchMode": "convert",
                "backend": "mlx",
            },
        ],
        "readme": [
            "Qwen 2.5 is a mature, well-supported family with extensive community GGUF availability.",
            "The Coder variant is specialised for software engineering and function calling.",
        ],
    },
    {
        "id": "llama-3-3",
        "name": "Llama 3.3",
        "provider": "Meta",
        "headline": "Meta's flagship open-weight model with strong reasoning and multilingual support.",
        "summary": "High-quality 70B distilled into efficient variants. Industry standard for local inference.",
        "description": (
            "Llama 3.3 brings Meta's strongest open-weight reasoning to local hardware. "
            "The 70B is the gold standard; community quantisations make it accessible on Apple Silicon."
        ),
        "updatedLabel": "Updated 3 weeks ago",
        "popularityLabel": "1.2M downloads",
        "likesLabel": "312 likes",
        "badges": ["Popular", "Reasoning", "Multilingual"],
        "capabilities": ["chat", "reasoning", "coding", "multilingual", "tool-use"],
        "defaultVariantId": "meta-llama/Llama-3.3-70B-Instruct",
        "variants": [
            {
                "id": "meta-llama/Llama-3.3-70B-Instruct",
                "name": "Llama 3.3 70B Instruct",
                "repo": "meta-llama/Llama-3.3-70B-Instruct",
                "link": "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
                "paramsB": 70.0,
                "sizeGb": 84.0,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["chat", "reasoning", "coding", "multilingual", "tool-use"],
                "note": "Full 70B for 128GB+ machines. Highest quality open-weight reasoning.",
                "contextWindow": "128K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "mlx-community/Llama-3.3-70B-Instruct-4bit",
                "name": "Llama 3.3 70B MLX 4-bit",
                "repo": "mlx-community/Llama-3.3-70B-Instruct-4bit",
                "link": "https://huggingface.co/mlx-community/Llama-3.3-70B-Instruct-4bit",
                "paramsB": 70.0,
                "sizeGb": 40.0,
                "format": "MLX",
                "quantization": "4-bit",
                "capabilities": ["chat", "reasoning", "coding", "multilingual", "tool-use"],
                "note": "4-bit quantisation for 64GB machines. Good quality/size tradeoff.",
                "contextWindow": "128K",
                "launchMode": "direct",
                "backend": "mlx",
            },
        ],
        "readme": [
            "Llama 3.3 is Meta's strongest open-weight model, widely used as a benchmark.",
            "The 4-bit MLX variant fits on 64GB Apple Silicon with room for context.",
        ],
    },
    {
        "id": "phi-4",
        "name": "Phi 4",
        "provider": "Microsoft",
        "headline": "Compact reasoning models with strong STEM and coding performance.",
        "summary": "Small-but-capable family that punches above its weight on reasoning benchmarks.",
        "description": (
            "Phi 4 delivers impressive reasoning and STEM performance in compact sizes. "
            "Great for resource-constrained setups that need quality reasoning."
        ),
        "updatedLabel": "Updated 2 weeks ago",
        "popularityLabel": "420k downloads",
        "likesLabel": "89 likes",
        "badges": ["Reasoning", "Compact", "STEM"],
        "capabilities": ["reasoning", "coding", "chat", "multilingual"],
        "defaultVariantId": "microsoft/phi-4",
        "variants": [
            {
                "id": "microsoft/phi-4",
                "name": "Phi 4 14B",
                "repo": "microsoft/phi-4",
                "link": "https://huggingface.co/microsoft/phi-4",
                "paramsB": 14.0,
                "sizeGb": 16.8,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["reasoning", "coding", "chat"],
                "note": "14B model that rivals much larger models on STEM and reasoning tasks.",
                "contextWindow": "16K",
                "launchMode": "convert",
                "backend": "mlx",
            },
            {
                "id": "mlx-community/phi-4-4bit",
                "name": "Phi 4 14B MLX 4-bit",
                "repo": "mlx-community/phi-4-4bit",
                "link": "https://huggingface.co/mlx-community/phi-4-4bit",
                "paramsB": 14.0,
                "sizeGb": 8.5,
                "format": "MLX",
                "quantization": "4-bit",
                "capabilities": ["reasoning", "coding", "chat"],
                "note": "Compact 4-bit variant that runs well on 16GB machines.",
                "contextWindow": "16K",
                "launchMode": "direct",
                "backend": "mlx",
            },
        ],
        "readme": [
            "Phi 4 punches above its weight on reasoning and STEM benchmarks.",
            "The 4-bit MLX variant is one of the best options for 16GB MacBooks.",
        ],
    },
    {
        "id": "mistral-nemo",
        "name": "Mistral Nemo",
        "provider": "Mistral AI",
        "headline": "12B general-purpose model with strong multilingual and tool-use support.",
        "summary": "Compact Mistral model designed for efficient local deployment with broad capabilities.",
        "description": (
            "Mistral Nemo is a 12B model co-developed with NVIDIA, optimised for local inference "
            "with strong multilingual support and function calling."
        ),
        "updatedLabel": "Updated 1 month ago",
        "popularityLabel": "650k downloads",
        "likesLabel": "128 likes",
        "badges": ["Multilingual", "Tool-use"],
        "capabilities": ["chat", "multilingual", "tool-use", "coding", "reasoning"],
        "defaultVariantId": "mistralai/Mistral-Nemo-Instruct-2407",
        "variants": [
            {
                "id": "mistralai/Mistral-Nemo-Instruct-2407",
                "name": "Mistral Nemo 12B Instruct",
                "repo": "mistralai/Mistral-Nemo-Instruct-2407",
                "link": "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
                "paramsB": 12.0,
                "sizeGb": 14.4,
                "format": "Transformers",
                "quantization": "BF16",
                "capabilities": ["chat", "multilingual", "tool-use", "coding"],
                "note": "12B all-rounder with excellent multilingual and function calling support.",
                "contextWindow": "128K",
                "launchMode": "convert",
                "backend": "mlx",
            },
        ],
        "readme": [
            "Mistral Nemo is a compact all-rounder with strong multilingual and tool-use support.",
            "Co-developed with NVIDIA for efficient local deployment.",
        ],
    },
]


def _flatten_catalog() -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        for variant in family["variants"]:
            flattened.append(
                {
                    **variant,
                    "familyId": family["id"],
                    "familyName": family["name"],
                    "provider": family["provider"],
                }
            )
    return flattened


CATALOG: list[dict[str, Any]] = _flatten_catalog()

IMAGE_MODEL_FAMILIES: list[dict[str, Any]] = [
    {
        "id": "flux-fast",
        "name": "FLUX.1 Schnell",
        "provider": "Black Forest Labs",
        "headline": "Fast prompt-to-image path for quick concepting and moodboards.",
        "summary": "Best starter option for rapid local ideation when you care more about speed and iteration than maximum fidelity.",
        "updatedLabel": "Curated starter pick",
        "badges": ["Fast", "General", "Photoreal"],
        "defaultVariantId": "black-forest-labs/FLUX.1-schnell",
        "variants": [
            {
                "id": "black-forest-labs/FLUX.1-schnell",
                "familyId": "flux-fast",
                "name": "FLUX.1 Schnell",
                "provider": "Black Forest Labs",
                "repo": "black-forest-labs/FLUX.1-schnell",
                "link": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
                "runtime": "Stub diffusion pipeline",
                "styleTags": ["photoreal", "general", "fast"],
                "taskSupport": ["txt2img"],
                "sizeGb": 23.7,
                "recommendedResolution": "1024x1024",
                "note": "Fastest concepting option in the curated image catalog.",
                "estimatedGenerationSeconds": 4.2,
            }
        ],
    },
    {
        "id": "flux-dev",
        "name": "FLUX.1 Dev",
        "provider": "Black Forest Labs",
        "headline": "High-fidelity guidance-distilled model for detailed, prompt-faithful generation.",
        "summary": "Balanced option for the eventual production runtime when you want stronger final image quality than the fast path.",
        "updatedLabel": "Curated quality pick",
        "badges": ["Balanced", "General", "Detailed"],
        "defaultVariantId": "black-forest-labs/FLUX.1-dev",
        "variants": [
            {
                "id": "black-forest-labs/FLUX.1-dev",
                "familyId": "flux-dev",
                "name": "FLUX.1 Dev",
                "provider": "Black Forest Labs",
                "repo": "black-forest-labs/FLUX.1-dev",
                "link": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
                "runtime": "Stub diffusion pipeline",
                "styleTags": ["general", "detailed", "balanced"],
                "taskSupport": ["txt2img"],
                "sizeGb": 23.8,
                "recommendedResolution": "1024x1024",
                "note": "Quality-oriented generalist for the curated image lineup.",
                "estimatedGenerationSeconds": 7.4,
            }
        ],
    },
    {
        "id": "sd35-medium",
        "name": "Stable Diffusion 3.5 Medium",
        "provider": "Stability AI",
        "headline": "Latest Stability model optimised for consumer hardware with MMDiT architecture.",
        "summary": "Modern architecture with strong prompt adherence and detail at a manageable model size for local use.",
        "updatedLabel": "Curated balanced pick",
        "badges": ["MMDiT", "Balanced", "Modern"],
        "defaultVariantId": "stabilityai/stable-diffusion-3.5-medium",
        "variants": [
            {
                "id": "stabilityai/stable-diffusion-3.5-medium",
                "familyId": "sd35-medium",
                "name": "Stable Diffusion 3.5 Medium",
                "provider": "Stability AI",
                "repo": "stabilityai/stable-diffusion-3.5-medium",
                "link": "https://huggingface.co/stabilityai/stable-diffusion-3.5-medium",
                "runtime": "Stub diffusion pipeline",
                "styleTags": ["general", "detailed", "modern"],
                "taskSupport": ["txt2img"],
                "sizeGb": 11.9,
                "recommendedResolution": "1024x1024",
                "note": "Latest Stability offering targeting a good quality-to-resource balance.",
                "estimatedGenerationSeconds": 5.8,
            }
        ],
    },
    {
        "id": "sd35-turbo",
        "name": "Stable Diffusion 3.5 Large Turbo",
        "provider": "Stability AI",
        "headline": "Distilled large model for fast, high-quality generation in fewer inference steps.",
        "summary": "Best pick when you want near-instant drafts at higher fidelity than older turbo models.",
        "updatedLabel": "Curated speed pick",
        "badges": ["Turbo", "Fast", "High quality"],
        "defaultVariantId": "stabilityai/stable-diffusion-3.5-large-turbo",
        "variants": [
            {
                "id": "stabilityai/stable-diffusion-3.5-large-turbo",
                "familyId": "sd35-turbo",
                "name": "Stable Diffusion 3.5 Large Turbo",
                "provider": "Stability AI",
                "repo": "stabilityai/stable-diffusion-3.5-large-turbo",
                "link": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo",
                "runtime": "Stub diffusion pipeline",
                "styleTags": ["fast", "quality", "general"],
                "taskSupport": ["txt2img"],
                "sizeGb": 16.5,
                "recommendedResolution": "1024x1024",
                "note": "Fastest high-fidelity model in the curated set.",
                "estimatedGenerationSeconds": 3.1,
            }
        ],
    },
    {
        "id": "sdxl-balanced",
        "name": "SDXL Balanced",
        "provider": "Stability AI",
        "headline": "General-purpose SDXL model for reliable, high-quality everyday image generation.",
        "summary": "Established baseline with wide ecosystem support, LoRA compatibility, and proven quality.",
        "updatedLabel": "Curated classic",
        "badges": ["Balanced", "General", "Detailed"],
        "defaultVariantId": "stabilityai/stable-diffusion-xl-base-1.0",
        "variants": [
            {
                "id": "stabilityai/stable-diffusion-xl-base-1.0",
                "familyId": "sdxl-balanced",
                "name": "Stable Diffusion XL Base 1.0",
                "provider": "Stability AI",
                "repo": "stabilityai/stable-diffusion-xl-base-1.0",
                "link": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
                "runtime": "Stub diffusion pipeline",
                "styleTags": ["general", "detailed", "balanced"],
                "taskSupport": ["txt2img"],
                "sizeGb": 13.1,
                "recommendedResolution": "1024x1024",
                "note": "Widely adopted SDXL baseline with strong community and LoRA ecosystem.",
                "estimatedGenerationSeconds": 7.4,
            }
        ],
    },
]

LATEST_IMAGE_TRACKED_SEEDS: list[dict[str, Any]] = [
    {
        "repo": "Qwen/Qwen-Image",
        "name": "Qwen-Image",
        "provider": "Qwen",
        "styleTags": ["general", "detailed", "qwenimage"],
        "taskSupport": ["txt2img"],
        "sizeGb": 57.7,
        "recommendedResolution": "1024x1024",
        "note": "Tracked diffusers-native Qwen image generation family.",
        "gated": False,
        "pipelineTag": "text-to-image",
        "updatedLabel": "Tracked latest",
    },
    {
        "repo": "Qwen/Qwen-Image-Edit",
        "name": "Qwen-Image-Edit",
        "provider": "Qwen",
        "styleTags": ["edit", "qwenimage", "general"],
        "taskSupport": ["img2img"],
        "sizeGb": 57.7,
        "recommendedResolution": "1024x1024",
        "note": "Tracked Qwen edit lane so Image Discover can surface newer editing-capable models too.",
        "gated": False,
        "pipelineTag": "image-to-image",
        "updatedLabel": "Tracked latest",
    },
    {
        "repo": "HiDream-ai/HiDream-I1-Full",
        "name": "HiDream-I1 Full",
        "provider": "HiDream AI",
        "styleTags": ["hidream", "detailed", "quality"],
        "taskSupport": ["txt2img"],
        "sizeGb": 47.2,
        "recommendedResolution": "1024x1024",
        "note": "Tracked larger open-image generation lane from the HiDream family.",
        "gated": False,
        "pipelineTag": "text-to-image",
        "updatedLabel": "Tracked latest",
    },
    {
        "repo": "zai-org/GLM-Image",
        "name": "GLM-Image",
        "provider": "Z.ai",
        "styleTags": ["general", "edit", "detailed"],
        "taskSupport": ["txt2img", "img2img"],
        "sizeGb": 35.8,
        "recommendedResolution": "1024x1024",
        "note": "Tracked unified generation-and-editing lane from the GLM image family.",
        "gated": False,
        "pipelineTag": "text-to-image",
        "updatedLabel": "Tracked latest",
    },
    {
        "repo": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        "name": "Sana Sprint 0.6B",
        "provider": "Efficient-Large-Model",
        "styleTags": ["sana", "fast", "small"],
        "taskSupport": ["txt2img"],
        "sizeGb": 7.7,
        "recommendedResolution": "1024x1024",
        "note": "Tracked smaller Sana Sprint lane for faster local image generation.",
        "gated": False,
        "pipelineTag": "text-to-image",
        "updatedLabel": "Tracked latest",
    },
    {
        "repo": "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        "name": "Sana Sprint 1.6B",
        "provider": "Efficient-Large-Model",
        "styleTags": ["sana", "fast", "detailed"],
        "taskSupport": ["txt2img"],
        "sizeGb": 9.74,
        "recommendedResolution": "1024x1024",
        "note": "Tracked larger Sana Sprint lane with a better quality-to-speed balance.",
        "gated": False,
        "pipelineTag": "text-to-image",
        "updatedLabel": "Tracked latest",
    },
]


def _stable_image_hash(value: str) -> int:
    acc = 0
    for index, char in enumerate(value):
        acc = (acc + ord(char) * (index + 17)) % 0xFFFFFF
    return acc


def _placeholder_image_data_url(prompt: str, model_name: str, width: int, height: int, seed: int) -> str:
    hash_value = _stable_image_hash(f"{model_name}:{prompt}:{seed}")
    hue_a = hash_value % 360
    hue_b = (hash_value * 7) % 360
    accent_x = 90 + (hash_value % 240)
    accent_y = 80 + ((hash_value >> 3) % 200)
    safe_prompt = html_escape((prompt.strip() or "Generated image preview")[:72])
    safe_model_name = html_escape(model_name)
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="hsl({hue_a} 72% 58%)" />
          <stop offset="100%" stop-color="hsl({hue_b} 68% 46%)" />
        </linearGradient>
      </defs>
      <rect width="{width}" height="{height}" rx="28" fill="url(#bg)" />
      <circle cx="{accent_x}" cy="{accent_y}" r="{max(42, round(width * 0.12))}" fill="rgba(255,255,255,0.18)" />
      <circle cx="{width - accent_x}" cy="{height - accent_y}" r="{max(36, round(width * 0.09))}" fill="rgba(8,12,20,0.18)" />
      <rect x="28" y="{height - 136}" width="{max(240, width - 56)}" height="108" rx="24" fill="rgba(11,15,22,0.38)" stroke="rgba(255,255,255,0.14)" />
      <text x="52" y="{height - 90}" fill="white" font-size="28" font-family="SF Pro Display, Inter, sans-serif" font-weight="700">{safe_model_name}</text>
      <text x="52" y="{height - 52}" fill="rgba(255,255,255,0.88)" font-size="19" font-family="SF Pro Text, Inter, sans-serif">{safe_prompt}</text>
    </svg>
    """.strip()
    return f"data:image/svg+xml;charset=utf-8,{urllib.parse.quote(svg)}"


def _image_model_payloads(library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repo_metadata: dict[str, dict[str, Any]] = {}
    repos = sorted({
        str(variant.get("repo") or "")
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "")
    })
    if repos:
        with ThreadPoolExecutor(max_workers=min(4, len(repos))) as executor:
            future_map = {
                executor.submit(_image_repo_live_metadata, repo): repo
                for repo in repos
            }
            try:
                for future in as_completed(future_map, timeout=8):
                    repo = future_map[future]
                    try:
                        repo_metadata[repo] = future.result(timeout=2)
                    except Exception:
                        repo_metadata[repo] = {
                            "metadataWarning": "Live Hugging Face metadata is temporarily unavailable. Showing curated defaults.",
                        }
            except TimeoutError:
                pass  # Return whatever we have so far; missing repos get curated defaults

    families: list[dict[str, Any]] = []
    for family in IMAGE_MODEL_FAMILIES:
        variants = [
            {
                **variant,
                **repo_metadata.get(str(variant.get("repo") or ""), {}),
                "source": "curated",
                "familyName": family.get("name"),
                "availableLocally": _image_variant_available_locally(variant, library),
                "hasLocalData": _hf_repo_snapshot_dir(str(variant.get("repo") or "")) is not None,
            }
            for variant in family["variants"]
        ]
        families.append(
            {
                **family,
                "updatedLabel": _best_image_family_updated_label(family, variants),
                "variants": variants,
            }
        )
    return families


def _find_image_variant(model_id: str) -> dict[str, Any] | None:
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == model_id:
                return variant
    return None


def _find_image_variant_by_repo(repo: str) -> dict[str, Any] | None:
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["repo"] == repo:
                return variant
    return None


def _resolve_app_version() -> str:
    pyproject_path = WORKSPACE_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return "0.0.0"
    try:
        with pyproject_path.open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except Exception:
        return "0.0.0"


app_version = _resolve_app_version()

def _load_data_location(bootstrap_path: Path, bootstrap_dir: Path) -> Path:
    """Read the bootstrap pointer file. Falls back to ``bootstrap_dir``."""
    if not bootstrap_path.exists():
        return bootstrap_dir
    try:
        payload = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return bootstrap_dir
    raw = payload.get("dataDirectory") if isinstance(payload, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return bootstrap_dir
    try:
        return Path(os.path.expanduser(raw)).resolve()
    except (OSError, RuntimeError):
        return bootstrap_dir


def _save_data_location(target: Path) -> None:
    """Write the bootstrap pointer atomically. ``target`` must be resolved."""
    bootstrap_dir = Path.home() / ".chaosengine"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_path = bootstrap_dir / ".location.json"
    tmp = bootstrap_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"dataDirectory": str(target)}, indent=2),
        encoding="utf-8",
    )
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(str(tmp), str(bootstrap_path))


def _migrate_data_directory(old: Path, new: Path) -> dict[str, Any]:
    """Copy known data files/dirs from ``old`` to ``new``.

    Idempotent: skips entries that already exist at the destination. Never
    deletes anything from ``old``. Validates writability via a probe file.
    Raises ``RuntimeError`` if ``new`` is not writable.
    """
    import shutil

    old = Path(os.path.expanduser(str(old))).resolve()
    new = Path(os.path.expanduser(str(new))).resolve()
    try:
        new.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot create data directory {new}: {exc}") from exc

    probe = new / ".chaosengine-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"Data directory {new} is not writable: {exc}") from exc

    summary: dict[str, Any] = {
        "copied": [],
        "skipped": [],
        "from": str(old),
        "to": str(new),
    }
    if old == new:
        return summary

    for name in ("settings.json", "benchmark-history.json", "chat-sessions.json"):
        src = old / name
        dst = new / name
        if not src.exists():
            continue
        if dst.exists():
            summary["skipped"].append(name)
            continue
        try:
            shutil.copy2(src, dst)
            summary["copied"].append(name)
        except OSError as exc:
            raise RuntimeError(f"Failed to copy {name}: {exc}") from exc

    docs_src = old / "documents"
    docs_dst = new / "documents"
    if docs_src.exists() and docs_src.is_dir():
        if docs_dst.exists():
            summary["skipped"].append("documents/")
        else:
            try:
                shutil.copytree(str(docs_src), str(docs_dst))
                summary["copied"].append("documents/")
            except OSError as exc:
                raise RuntimeError(f"Failed to copy documents/: {exc}") from exc

    return summary


class DataLocation:
    """Resolves where ChaosEngineAI persists user data.

    The bootstrap pointer at ``~/.chaosengine/.location.json`` may redirect
    the actual data directory to a user-chosen path (e.g. a Dropbox folder).
    Missing or unreadable pointer means data lives at the bootstrap dir for
    backwards compatibility with older installs.
    """

    def __init__(self) -> None:
        self.bootstrap_dir: Path = Path.home() / ".chaosengine"
        self.bootstrap_path: Path = self.bootstrap_dir / ".location.json"
        self.bootstrap_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir: Path = _load_data_location(self.bootstrap_path, self.bootstrap_dir)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.data_dir = self.bootstrap_dir
            self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def settings_path(self) -> Path:
        return self.data_dir / "settings.json"

    @property
    def benchmarks_path(self) -> Path:
        return self.data_dir / "benchmark-history.json"

    @property
    def chat_sessions_path(self) -> Path:
        return self.data_dir / "chat-sessions.json"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def image_outputs_dir(self) -> Path:
        return self.images_dir / "outputs"


DATA_LOCATION = DataLocation()
# Backwards-compat aliases captured at import time. We never mutate
# DATA_LOCATION at runtime — a directory change is persisted to the bootstrap
# pointer and picked up cleanly on the next process restart, so all four of
# these reflect a single consistent view of the active data directory.
SETTINGS_DIR = DATA_LOCATION.data_dir
SETTINGS_PATH = DATA_LOCATION.settings_path
BENCHMARKS_PATH = DATA_LOCATION.benchmarks_path
CHAT_SESSIONS_PATH = DATA_LOCATION.chat_sessions_path
DOCUMENTS_DIR = DATA_LOCATION.documents_dir
IMAGE_OUTPUTS_DIR = DATA_LOCATION.image_outputs_dir
MAX_CHAT_SESSIONS = 200
MAX_DOC_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB per file
MAX_SESSION_DOCS_BYTES = 200 * 1024 * 1024  # 200 MB per session
DOC_ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml", ".toml",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".html", ".css", ".sh",
}
CHUNK_SIZE_CHARS = 2000  # Approximately 500 tokens
CHUNK_OVERLAP_CHARS = 400
MAX_BENCHMARK_RUNS = 48
DEFAULT_MODEL_DIRECTORIES: list[dict[str, Any]] = [
    {
        "id": "hf-cache",
        "label": "Hugging Face cache",
        "path": "~/.cache/huggingface/hub",
        "enabled": True,
        "source": "default",
    },
    {
        "id": "mlx-cache",
        "label": "MLX cache",
        "path": "~/.cache/mlx",
        "enabled": True,
        "source": "default",
    },
    {
        "id": "home-models",
        "label": "Models",
        "path": "~/Models",
        "enabled": True,
        "source": "default",
    },
]
DEFAULT_LAUNCH_PREFERENCES = {
    "contextTokens": 8192,
    "maxTokens": 4096,
    "temperature": 0.7,
    "cacheStrategy": "native",
    "cacheBits": 0,
    "fp16Layers": 0,
    "fusedAttention": False,
    "fitModelInMemory": True,
}


def _normalize_slug(value: str, fallback: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or fallback


def _default_settings() -> dict[str, Any]:
    return {
        "modelDirectories": [dict(entry) for entry in DEFAULT_MODEL_DIRECTORIES],
        "preferredServerPort": DEFAULT_PORT,
        "allowRemoteConnections": False,
        "autoStartServer": False,
        "launchPreferences": dict(DEFAULT_LAUNCH_PREFERENCES),
        "remoteProviders": [],
        "huggingFaceToken": "",
        "dataDirectory": str(DATA_LOCATION.data_dir),
    }


def _normalize_model_directory_entry(entry: dict[str, Any], index: int) -> dict[str, Any]:
    raw_path = str(entry.get("path") or "").strip()
    label = str(entry.get("label") or "").strip()
    if not label:
        label = Path(os.path.expanduser(raw_path or f"directory-{index + 1}")).name or f"Directory {index + 1}"
    directory_id = _normalize_slug(str(entry.get("id") or label), f"directory-{index + 1}")
    return {
        "id": directory_id,
        "label": label,
        "path": raw_path,
        "enabled": bool(entry.get("enabled", True)),
        "source": "default" if str(entry.get("source") or "user") == "default" else "user",
    }


def _normalize_model_directories(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not str(entry.get("path") or "").strip():
            continue
        item = _normalize_model_directory_entry(entry, index)
        base_id = item["id"]
        suffix = 2
        while item["id"] in seen_ids:
            item["id"] = f"{base_id}-{suffix}"
            suffix += 1
        seen_ids.add(item["id"])
        normalized.append(item)
    return normalized


def _normalize_launch_preferences(payload: dict[str, Any] | None) -> dict[str, Any]:
    defaults = dict(DEFAULT_LAUNCH_PREFERENCES)
    if not isinstance(payload, dict):
        return defaults

    result = dict(defaults)

    # Migrate legacy TurboQuant fields to the new cache strategy model.
    if "useTurboQuant" in payload and "cacheStrategy" not in payload:
        result["cacheStrategy"] = "native"
    if "turboQuantBits" in payload and "cacheBits" not in payload:
        payload["cacheBits"] = payload["turboQuantBits"]

    integer_fields = {
        "contextTokens": (256, 262144),
        "maxTokens": (1, 32768),
        "cacheBits": (0, 8),
        "fp16Layers": (0, 16),
    }
    for key, (minimum, maximum) in integer_fields.items():
        if key in payload:
            try:
                result[key] = max(minimum, min(maximum, int(payload[key])))
            except (TypeError, ValueError):
                pass

    if result.get("maxTokens", 0) < defaults["maxTokens"]:
        result["maxTokens"] = defaults["maxTokens"]

    if "temperature" in payload:
        try:
            result["temperature"] = max(0.0, min(2.0, float(payload["temperature"])))
        except (TypeError, ValueError):
            pass

    for key in ("fusedAttention", "fitModelInMemory"):
        if key in payload:
            result[key] = bool(payload[key])

    if "cacheStrategy" in payload:
        val = str(payload["cacheStrategy"]).strip()
        result["cacheStrategy"] = val if val else "native"

    return result


def _load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    settings = _default_settings()
    if not path.exists():
        return settings

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings

    model_directories = payload.get("modelDirectories")
    if isinstance(model_directories, list):
        normalized = _normalize_model_directories(model_directories)
        settings["modelDirectories"] = normalized

    try:
        preferred_port = int(payload.get("preferredServerPort", DEFAULT_PORT))
        settings["preferredServerPort"] = max(1024, min(65535, preferred_port))
    except (TypeError, ValueError):
        settings["preferredServerPort"] = DEFAULT_PORT

    settings["allowRemoteConnections"] = bool(payload.get("allowRemoteConnections", False))
    settings["autoStartServer"] = bool(payload.get("autoStartServer", False))

    settings["launchPreferences"] = _normalize_launch_preferences(payload.get("launchPreferences"))

    hf_token = payload.get("huggingFaceToken")
    if isinstance(hf_token, str):
        settings["huggingFaceToken"] = hf_token
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    return settings


def _save_settings(settings: dict[str, Any], path: Path = SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(str(tmp), str(path))


class LoadModelRequest(BaseModel):
    modelRef: str = Field(min_length=1)
    modelName: str | None = None
    source: str = "catalog"
    backend: str = "auto"
    path: str | None = None
    cacheStrategy: str = "native"
    cacheBits: int = Field(default=0, ge=0, le=8)
    fp16Layers: int = Field(default=0, ge=0, le=16)
    fusedAttention: bool = False
    fitModelInMemory: bool = True
    contextTokens: int = Field(default=8192, ge=256, le=262144)


class ModelDirectoryRequest(BaseModel):
    id: str | None = None
    label: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1, max_length=4096)
    enabled: bool = True
    source: str = "user"


class LaunchPreferencesRequest(BaseModel):
    contextTokens: int = Field(default=8192, ge=256, le=262144)
    maxTokens: int = Field(default=4096, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    cacheStrategy: str = "native"
    cacheBits: int = Field(default=0, ge=0, le=8)
    fp16Layers: int = Field(default=0, ge=0, le=16)
    fusedAttention: bool = False
    fitModelInMemory: bool = True


class CreateSessionRequest(BaseModel):
    title: str | None = None


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    model: str | None = None
    modelRef: str | None = None
    modelSource: str | None = None
    modelPath: str | None = None
    modelBackend: str | None = None
    pinned: bool | None = None


class GenerateRequest(BaseModel):
    sessionId: str | None = None
    title: str | None = None
    prompt: str = Field(min_length=1)
    images: list[str] | None = None  # base64-encoded images
    modelRef: str | None = None
    modelName: str | None = None
    source: str = "catalog"
    path: str | None = None
    backend: str = "auto"
    systemPrompt: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    maxTokens: int = Field(default=4096, ge=1, le=32768)
    cacheStrategy: str | None = None
    cacheBits: int | None = Field(default=None, ge=0, le=8)
    fp16Layers: int | None = Field(default=None, ge=0, le=16)
    fusedAttention: bool | None = None
    fitModelInMemory: bool | None = None
    contextTokens: int | None = Field(default=None, ge=256, le=262144)


class RemoteProviderRequest(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    apiBase: str = Field(min_length=8, max_length=512)
    apiKey: str = Field(default="", max_length=512)
    model: str = Field(min_length=1, max_length=200)


class UpdateSettingsRequest(BaseModel):
    modelDirectories: list[ModelDirectoryRequest] | None = None
    preferredServerPort: int | None = Field(default=None, ge=1024, le=65535)
    allowRemoteConnections: bool | None = None
    autoStartServer: bool | None = None
    launchPreferences: LaunchPreferencesRequest | None = None
    remoteProviders: list[RemoteProviderRequest] | None = None
    huggingFaceToken: str | None = Field(default=None, max_length=512)
    dataDirectory: str | None = Field(default=None, max_length=4096)


class OpenAIMessage(BaseModel):
    role: str
    content: Any
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None


class ConvertModelRequest(BaseModel):
    modelRef: str | None = None
    path: str | None = None
    hfRepo: str | None = None
    outputPath: str | None = None
    quantize: bool = True
    qBits: int = Field(default=4, ge=2, le=8)
    qGroupSize: int = Field(default=64, ge=32, le=128)
    dtype: str = Field(default="float16", min_length=3, max_length=16)


class BenchmarkRunRequest(BaseModel):
    mode: str = "throughput"  # "throughput" | "perplexity" | "task_accuracy"
    modelRef: str | None = None
    modelName: str | None = None
    source: str = "catalog"
    backend: str = "auto"
    path: str | None = None
    label: str | None = None
    prompt: str | None = None
    cacheStrategy: str = "native"
    cacheBits: int = Field(default=0, ge=0, le=8)
    fp16Layers: int = Field(default=0, ge=0, le=16)
    fusedAttention: bool = False
    fitModelInMemory: bool = True
    contextTokens: int = Field(default=8192, ge=256, le=262144)
    maxTokens: int = Field(default=512, ge=32, le=32768)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # Perplexity mode
    perplexityDataset: str = "wikitext-2"
    perplexityNumSamples: int = Field(default=64, ge=8, le=1024)
    perplexitySeqLength: int = Field(default=512, ge=128, le=4096)
    perplexityBatchSize: int = Field(default=4, ge=1, le=32)
    # Task accuracy mode
    taskName: str = "mmlu"
    taskLimit: int = Field(default=100, ge=10, le=5000)
    taskNumShots: int = Field(default=5, ge=0, le=10)


class RevealPathRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class DeleteModelRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


_HF_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-]+/[a-zA-Z0-9_.\-]+$")


class DownloadModelRequest(BaseModel):
    repo: str = Field(min_length=3, max_length=256)


class ImageGenerationRequest(BaseModel):
    modelId: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=4000)
    negativePrompt: str | None = Field(default=None, max_length=4000)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    steps: int = Field(default=24, ge=1, le=100)
    guidance: float = Field(default=5.5, ge=1.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, le=2147483647)
    batchSize: int = Field(default=1, ge=1, le=4)
    qualityPreset: str | None = Field(default=None, max_length=32)


class ImageRuntimePreloadRequest(BaseModel):
    modelId: str = Field(min_length=1, max_length=256)


class ImageRuntimeUnloadRequest(BaseModel):
    modelId: str | None = Field(default=None, min_length=1, max_length=256)


def _bytes_to_gb(value: int | float) -> float:
    size_gb = float(value) / (1024 ** 3)
    if 0 < size_gb < 0.01:
        return 0.01
    return round(size_gb, 2)


def _image_output_directory(created_at: str | None = None) -> Path:
    day_label = (created_at or datetime.utcnow().isoformat())[:10]
    output_dir = IMAGE_OUTPUTS_DIR / day_label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _preview_data_url_from_image_path(image_path: str | None) -> str:
    if not image_path:
        return ""
    path = Path(image_path)
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    try:
        if suffix == ".svg":
            return f"data:image/svg+xml;charset=utf-8,{urllib.parse.quote(path.read_text(encoding='utf-8'))}"
        mime_type = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except OSError:
        return ""


def _hydrate_image_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "")
    model_name = str(payload.get("modelName") or payload.get("modelId") or "Image model")
    width = int(payload.get("width") or 1024)
    height = int(payload.get("height") or 1024)
    seed = int(payload.get("seed") or 0)
    image_path = str(payload.get("imagePath") or "")
    metadata_path = str(payload.get("metadataPath") or "")
    preview_url = str(payload.get("previewUrl") or "").strip()
    if not preview_url:
        preview_url = _preview_data_url_from_image_path(image_path) or _placeholder_image_data_url(prompt, model_name, width, height, seed)
    return {
        "artifactId": str(payload.get("artifactId") or ""),
        "modelId": str(payload.get("modelId") or ""),
        "modelName": model_name,
        "prompt": prompt,
        "negativePrompt": str(payload.get("negativePrompt") or ""),
        "width": width,
        "height": height,
        "steps": int(payload.get("steps") or 24),
        "guidance": float(payload.get("guidance") or 5.5),
        "seed": seed,
        "createdAt": str(payload.get("createdAt") or datetime.utcnow().replace(microsecond=0).isoformat() + "Z"),
        "durationSeconds": float(payload.get("durationSeconds") or 0.0),
        "previewUrl": preview_url,
        "imagePath": image_path or None,
        "metadataPath": metadata_path or None,
        "runtimeLabel": str(payload.get("runtimeLabel") or ""),
        "runtimeNote": str(payload.get("runtimeNote") or "") or None,
    }


def _save_image_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    created_at = str(artifact.get("createdAt") or datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
    output_dir = _image_output_directory(created_at)
    artifact_id = str(artifact["artifactId"])
    extension = str(artifact.get("imageExtension") or "").lstrip(".")
    preview_url = str(artifact.get("previewUrl") or "")
    if not extension:
        extension = "svg" if preview_url.startswith("data:image/svg+xml") else "png"
    image_path = output_dir / f"{artifact_id}.{extension}"
    metadata_path = output_dir / f"{artifact_id}.json"
    image_bytes = artifact.get("imageBytes")
    if isinstance(image_bytes, str):
        image_bytes = base64.b64decode(image_bytes.encode("ascii"))

    if isinstance(image_bytes, (bytes, bytearray)):
        image_path.write_bytes(bytes(image_bytes))
    elif preview_url.startswith("data:image/svg+xml"):
        image_path.write_text(
            urllib.parse.unquote(preview_url.split(",", 1)[1]),
            encoding="utf-8",
        )
    elif ";base64," in preview_url:
        encoded = preview_url.split(";base64,", 1)[1]
        image_path.write_bytes(base64.b64decode(encoded.encode("ascii")))
    else:
        image_path.write_text("", encoding="utf-8")

    persisted = {
        **artifact,
        "imagePath": str(image_path),
        "metadataPath": str(metadata_path),
    }
    metadata_payload = {
        key: value
        for key, value in persisted.items()
        if key not in {"imageBytes", "imageMimeType", "imageExtension", "previewUrl"}
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return _hydrate_image_artifact(persisted)


def _load_image_outputs() -> list[dict[str, Any]]:
    if not IMAGE_OUTPUTS_DIR.exists():
        return []
    outputs: list[dict[str, Any]] = []
    for metadata_path in IMAGE_OUTPUTS_DIR.rglob("*.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        outputs.append(_hydrate_image_artifact({**payload, "metadataPath": str(metadata_path)}))
    outputs.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return outputs


def _find_image_output(artifact_id: str) -> dict[str, Any] | None:
    for output in _load_image_outputs():
        if output.get("artifactId") == artifact_id:
            return output
    return None


def _delete_image_output(artifact_id: str) -> bool:
    found = False
    for metadata_path in IMAGE_OUTPUTS_DIR.rglob(f"{artifact_id}.json") if IMAGE_OUTPUTS_DIR.exists() else []:
        found = True
        image_path = metadata_path.with_suffix(".svg")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("imagePath"):
                image_path = Path(str(payload["imagePath"]))
        except (OSError, json.JSONDecodeError):
            pass
        try:
            metadata_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            pass
    return found


def _generate_image_artifacts(
    request: ImageGenerationRequest,
    variant: dict[str, Any],
    runtime_manager: ImageRuntimeManager | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime_manager = runtime_manager or ImageRuntimeManager()
    rendered_images, runtime_status = runtime_manager.generate(
        ImageGenerationConfig(
            modelId=request.modelId,
            modelName=str(variant["name"]),
            repo=str(variant["repo"]),
            prompt=request.prompt,
            negativePrompt=request.negativePrompt or "",
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance=request.guidance,
            batchSize=request.batchSize,
            seed=request.seed,
            qualityPreset=request.qualityPreset,
        )
    )
    created_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    artifacts: list[dict[str, Any]] = []
    for rendered in rendered_images:
        artifact = {
            "artifactId": f"img-{uuid.uuid4().hex[:12]}",
            "modelId": request.modelId,
            "modelName": variant["name"],
            "prompt": request.prompt,
            "negativePrompt": request.negativePrompt or "",
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance": request.guidance,
            "seed": rendered.seed,
            "createdAt": created_at,
            "durationSeconds": rendered.durationSeconds,
            "imageBytes": rendered.bytes,
            "imageMimeType": rendered.mimeType,
            "imageExtension": rendered.extension,
            "runtimeLabel": rendered.runtimeLabel,
            "runtimeNote": rendered.runtimeNote,
        }
        artifacts.append(_save_image_artifact(artifact))
    return artifacts, runtime_status


def _safe_run(command: list[str], timeout: float = 1.5) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _apple_hardware_summary(total_memory_gb: float) -> str | None:
    if platform.system() != "Darwin":
        return None
    payload = _safe_run(["system_profiler", "SPHardwareDataType", "-json"], timeout=2.5)
    if not payload:
        return None
    try:
        hardware = json.loads(payload)["SPHardwareDataType"][0]
    except Exception:
        return None

    chip = hardware.get("chip_type") or hardware.get("cpu_type")
    model = hardware.get("machine_model") or hardware.get("machine_name")
    parts = [part for part in [chip, model] if part]
    if not parts:
        return None
    return f"{' / '.join(parts)} / {total_memory_gb:.0f} GB unified memory"


def _generic_hardware_summary(total_memory_gb: float) -> str:
    system_name = platform.system()
    machine = platform.machine()
    processor = platform.processor() or machine
    return f"{processor} / {system_name} / {total_memory_gb:.0f} GB memory"


def _hf_repo_from_link(link: str | None) -> str | None:
    if not link or "huggingface.co/" not in link:
        return None
    repo = link.split("huggingface.co/", 1)[1].strip("/")
    if not repo:
        return None
    return repo.split("/tree/", 1)[0].split("/blob/", 1)[0].strip("/")


def _get_cache_strategies() -> list[dict[str, Any]]:
    from compression import registry
    return registry.available()


def _runtime_label(capabilities: dict[str, Any] | None = None) -> str:
    native = capabilities or get_backend_capabilities().to_dict()
    on_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if on_apple_silicon and native.get("mlxUsable"):
        return "MLX + ChaosEngine"
    if native.get("ggufAvailable"):
        return "llama.cpp + GGUF sidecar"
    return "Python sidecar"


def _detect_gpu_utilization() -> float | None:
    return None


def _get_compressed_memory_gb() -> float:
    """Parse macOS vm_stat for compressed memory (no sudo)."""
    if platform.system() != "Darwin":
        return 0.0
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2,
        )
        page_size = 16384  # Apple Silicon default
        pages_compressed = 0
        for line in result.stdout.split("\n"):
            if "page size of" in line:
                # "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                try:
                    page_size = int(line.split("page size of")[1].split("bytes")[0].strip())
                except (ValueError, IndexError):
                    pass
            elif "Pages occupied by compressor" in line:
                try:
                    pages_compressed = int(line.split(":")[1].strip().rstrip("."))
                except (ValueError, IndexError):
                    pass
        return round((pages_compressed * page_size) / (1024 ** 3), 1)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 0.0


def _get_battery_info() -> dict[str, Any] | None:
    """Parse pmset -g batt for battery state (no sudo). Returns None on desktops."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=2,
        )
        output = result.stdout
        # First line: "Now drawing from 'AC Power'" or "'Battery Power'"
        power_source = "AC"
        if "Battery Power" in output:
            power_source = "Battery"
        # Subsequent line: " -InternalBattery-0 ... 85%; discharging; ..."
        if "InternalBattery" not in output:
            return None  # No battery (desktop Mac)
        percent = 100
        charging = False
        for line in output.split("\n"):
            if "InternalBattery" in line:
                # Extract "85%"
                if "%" in line:
                    try:
                        parts = line.split("%")[0].split()
                        percent = int(parts[-1])
                    except (ValueError, IndexError):
                        pass
                if "charging" in line.lower() and "discharging" not in line.lower():
                    charging = True
                elif "charged" in line.lower():
                    charging = False
                break
        return {
            "percent": percent,
            "powerSource": power_source,
            "charging": charging,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _get_disk_usage_for_models(settings: dict[str, Any]) -> dict[str, float] | None:
    """Return disk usage of the first enabled model directory."""
    dirs = settings.get("modelDirectories") or []
    for entry in dirs:
        if not entry.get("enabled", True):
            continue
        raw_path = str(entry.get("path") or "").strip()
        if not raw_path:
            continue
        try:
            expanded = Path(os.path.expanduser(raw_path))
            if not expanded.exists():
                continue
            usage = psutil.disk_usage(str(expanded))
            return {
                "totalGb": _bytes_to_gb(usage.total),
                "usedGb": _bytes_to_gb(usage.used),
                "freeGb": _bytes_to_gb(usage.free),
                "path": str(expanded),
            }
        except (OSError, PermissionError):
            continue
    # Fall back to home directory
    try:
        usage = psutil.disk_usage(str(Path.home()))
        return {
            "totalGb": _bytes_to_gb(usage.total),
            "usedGb": _bytes_to_gb(usage.used),
            "freeGb": _bytes_to_gb(usage.free),
            "path": str(Path.home()),
        }
    except OSError:
        return None


def _get_top_memory_map() -> dict[int, float]:
    """Use macOS `top` to get real memory (including GPU/compressed) per PID.

    psutil's RSS misses Metal GPU memory used by MLX models. macOS `top`
    reports the full footprint that matches Activity Monitor.
    """
    try:
        result = subprocess.run(
            ["top", "-l", "1", "-stats", "pid,mem", "-o", "mem", "-n", "120"],
            capture_output=True, text=True, timeout=10,
        )
        mem_map: dict[int, float] = {}
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            mem_str = parts[1].strip().rstrip("+-")
            try:
                if mem_str.endswith("G"):
                    mem_map[pid] = float(mem_str[:-1])
                elif mem_str.endswith("M"):
                    mem_map[pid] = float(mem_str[:-1]) / 1024
                elif mem_str.endswith("K"):
                    mem_map[pid] = float(mem_str[:-1]) / (1024 * 1024)
                else:
                    mem_map[pid] = float(mem_str) / (1024 ** 3)
            except ValueError:
                continue
        return mem_map
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}


def _list_llm_processes(limit: int = 12) -> list[dict[str, Any]]:
    # Process-name keywords that indicate an LLM-related process.
    # Intentionally excludes "chaosengine" and "python" — they are too
    # broad and match unrelated processes whose cmdline or cwd happens
    # to contain a project path like /Users/dan/ChaosEngineAI.
    name_keywords = ("mlx", "llama-server", "llama-cli", "ollama", "openclaw", "chaosengine")
    # For python processes, only match real model workers / runtimes.
    # The API sidecar itself is intentionally excluded so Dashboard does not
    # fill up with idle backend processes that are not serving a model.
    python_module_markers = ("backend_service.mlx_worker", "mlx_worker", "chaosengine", "llama")
    own_markers = ("chaosengine", "backend_service.mlx_worker", "chaosengine-embedded")
    # Only match LM Studio via the ACTUAL binary path — otherwise any
    # llama-server loading a model from /AI_Models/lmstudio-community/...
    # gets mis-labelled as LM Studio because of the model path substring.
    lmstudio_binary_markers = (
        "/applications/lm studio.app/",
        "/.lmstudio/",
        "/lmstudio.app/contents/",
    )
    ollama_markers = ("ollama",)
    # Get real memory from top (includes GPU/Metal memory on macOS)
    top_mem = _get_top_memory_map() if platform.system() == "Darwin" else {}

    matches: list[dict[str, Any]] = []
    try:
        for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent", "ppid"]):
            try:
                name = (process.info.get("name") or "").lower()
                cmdline_parts = process.info.get("cmdline") or []
                cmdline = " ".join(cmdline_parts).lower()
                haystack = f"{name} {cmdline}"

                # Check if this is an LLM process by name
                is_llm = any(keyword in name for keyword in name_keywords)
                # For python processes, check if the module is LLM-related
                if not is_llm and "python" in name:
                    is_llm = any(m in cmdline for m in python_module_markers)
                pid = process.info["pid"]
                ppid = process.info.get("ppid")

                if not is_llm:
                    continue

                # Determine owner. The LM Studio check ONLY looks at the
                # binary path (cmdline[0]) — matching a substring anywhere
                # in the full cmdline mis-labels any llama-server loading a
                # model from a directory like lmstudio-community/ .
                binary_path = (cmdline_parts[0] if cmdline_parts else "").lower()
                if any(m in binary_path for m in lmstudio_binary_markers):
                    owner = "LM Studio"
                elif any(m in haystack for m in ollama_markers):
                    owner = "Ollama"
                elif any(m in haystack for m in own_markers):
                    owner = "ChaosEngineAI"
                else:
                    owner = "System"

                # Use top's memory (includes GPU) if available, fall back to RSS
                rss_gb = _bytes_to_gb(process.info["memory_info"].rss if process.info.get("memory_info") else 0)
                mem_gb = round(top_mem.get(pid, rss_gb), 1)

                # Detect process kind from cmdline for better model mapping
                kind = "other"
                if "mlx_worker" in cmdline:
                    kind = "mlx_worker"
                elif "llama-server" in name or "llama-server" in cmdline:
                    kind = "llama_server"
                elif "backend_service.app" in cmdline:
                    kind = "backend"

                matches.append(
                    {
                        "pid": pid,
                        "name": name or "process",
                        "owner": owner,
                        "memoryGb": mem_gb,
                        "cpuPercent": round(float(process.info.get("cpu_percent") or 0.0), 1),
                        "kind": kind,
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError):
                continue
    except (psutil.Error, PermissionError, OSError):
        return []

    matches.sort(key=lambda item: (item["memoryGb"], item["cpuPercent"]), reverse=True)
    return matches[:limit]


def _build_system_snapshot() -> dict[str, Any]:
    native = get_backend_capabilities().to_dict()
    memory = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory()
        swap_used = swap.used
        swap_total = swap.total
    except OSError:
        swap_used = 0
        swap_total = 0
    total_memory_gb = _bytes_to_gb(memory.total)
    available_memory_gb = _bytes_to_gb(memory.available)
    used_memory_gb = _bytes_to_gb(memory.used)
    swap_used_gb = _bytes_to_gb(swap_used)
    swap_total_gb = _bytes_to_gb(swap_total)
    spare_headroom_gb = round(max(0.0, available_memory_gb - 6.0), 1)
    hardware_summary = _apple_hardware_summary(total_memory_gb) or _generic_hardware_summary(total_memory_gb)

    compressed_memory_gb = _get_compressed_memory_gb()
    battery = _get_battery_info()
    # Memory pressure: used + compressed + swap as a fraction of total
    pressure_numerator = used_memory_gb + compressed_memory_gb + swap_used_gb
    memory_pressure_percent = (
        round(min(100.0, (pressure_numerator / total_memory_gb) * 100), 1)
        if total_memory_gb > 0 else 0.0
    )

    return {
        "platform": platform.system(),
        "arch": platform.machine(),
        "hardwareSummary": hardware_summary,
        "backendLabel": _runtime_label(native),
        "appVersion": app_version,
        "availableCacheStrategies": _get_cache_strategies(),
        "vllmAvailable": native.get("vllmAvailable", False),
        "vllmVersion": native.get("vllmVersion"),
        "mlxAvailable": native["mlxAvailable"],
        "mlxLmAvailable": native["mlxLmAvailable"],
        "mlxUsable": native["mlxUsable"],
        "ggufAvailable": native["ggufAvailable"],
        "converterAvailable": native["converterAvailable"],
        "nativePython": native["pythonExecutable"],
        "llamaServerPath": native["llamaServerPath"],
        "llamaCliPath": native["llamaCliPath"],
        "nativeRuntimeMessage": native["mlxMessage"],
        "totalMemoryGb": total_memory_gb,
        "availableMemoryGb": available_memory_gb,
        "usedMemoryGb": used_memory_gb,
        "swapUsedGb": swap_used_gb,
        "swapTotalGb": swap_total_gb,
        "compressedMemoryGb": compressed_memory_gb,
        "memoryPressurePercent": memory_pressure_percent,
        "cpuUtilizationPercent": round(psutil.cpu_percent(interval=None), 1),
        "gpuUtilizationPercent": _detect_gpu_utilization(),
        "spareHeadroomGb": spare_headroom_gb,
        "battery": battery,
        "runningLlmProcesses": _list_llm_processes(),
        "uptimeMinutes": round((time.time() - APP_STARTED_AT) / 60, 1),
    }


def _best_fit_recommendation(system_stats: dict[str, Any]) -> dict[str, Any]:
    memory_gb = system_stats["totalMemoryGb"]
    is_macos_mlx = (
        system_stats["platform"] == "Darwin"
        and system_stats["arch"] == "arm64"
        and bool(system_stats.get("mlxUsable", False))
    )

    if memory_gb >= 64:
        model_size = "70B"
        cache_label = "Native f16"
        headroom_percent = 68
    elif memory_gb >= 48:
        model_size = "70B"
        cache_label = "Native f16"
        headroom_percent = 65
    elif memory_gb >= 36:
        model_size = "32B"
        cache_label = "Native f16"
        headroom_percent = 54
    elif memory_gb >= 24:
        model_size = "14B"
        cache_label = "Native f16"
        headroom_percent = 49
    else:
        model_size = "7B"
        cache_label = "Native f16"
        headroom_percent = 43

    if is_macos_mlx:
        title = f"Recommended target: {model_size} class @ {cache_label}"
        detail = (
            f"This forecast is relative to a recommended {model_size} class local target on "
            f"{system_stats['hardwareSummary']}, not a currently selected chat model."
        )
    else:
        title = f"Recommended target: {model_size} GGUF"
        detail = (
            "Cross-platform mode will prefer llama.cpp GGUF for broad hardware support."
        )

    return {
        "title": title,
        "detail": detail,
        "targetModel": model_size,
        "cacheLabel": cache_label,
        "headroomPercent": headroom_percent,
    }


def _path_size_bytes(path: Path, *, seen: set[tuple[int, int]] | None = None) -> int:
    visited = seen if seen is not None else set()
    try:
        stat_result = path.stat()
    except OSError:
        return 0

    file_id = (stat_result.st_dev, stat_result.st_ino)
    if file_id in visited:
        return 0
    visited.add(file_id)

    if path.is_file():
        return int(stat_result.st_size)

    total = 0
    try:
        children = list(path.iterdir())
    except OSError:
        return 0

    for child in children:
        total += _path_size_bytes(child, seen=visited)
    return total


def _du_size_gb(path: Path) -> float:
    if path.is_file():
        return _bytes_to_gb(_path_size_bytes(path))

    payload = _safe_run(["du", "-sk", str(path)], timeout=4.0)
    if payload:
        try:
            size_kb = int(payload.split()[0])
            size_gb = round(size_kb / (1024 ** 2), 1)
            if size_gb > 0:
                return size_gb
        except (ValueError, IndexError):
            pass

    fallback_bytes = _path_size_bytes(path)
    return _bytes_to_gb(fallback_bytes) if fallback_bytes > 0 else 0.0


def _relative_depth(path: Path, root: Path) -> int:
    try:
        return len(path.relative_to(root).parts)
    except ValueError:
        return 0


def _candidate_model_dirs(path: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(candidate: Path) -> None:
        try:
            if not candidate.is_dir():
                return
        except OSError:
            return
        key = str(candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    if path.is_dir():
        _add(path)
        snapshots = path / "snapshots"
        try:
            if snapshots.is_dir():
                for snap in sorted(snapshots.iterdir()):
                    _add(snap)
        except OSError:
            pass
    else:
        _add(path.parent)
    return candidates


def _read_model_config(path: Path) -> dict[str, Any] | None:
    for directory in _candidate_model_dirs(path):
        candidate = directory / "config.json"
        try:
            if candidate.exists():
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            continue
    return None


def _model_has_files(path: Path, pattern: str) -> bool:
    try:
        return any(path.rglob(pattern))
    except OSError:
        return False


def _main_gguf_file(path: Path) -> Path | None:
    try:
        candidates = [
            candidate for candidate in path.rglob("*.gguf")
            if "mmproj" not in candidate.name.lower()
        ]
    except OSError:
        return None
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda candidate: candidate.stat().st_size)
    except OSError:
        return candidates[0]


def _quantization_label_from_text(text: str) -> str | None:
    lowered = text.lower()
    match = re.search(r"\b(q\d(?:_[a-z0-9]+)*)\b", lowered)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(\d+)[-_ ]?bit\b", lowered)
    if match:
        return f"{int(match.group(1))}-bit"
    if "bf16" in lowered or "bfloat16" in lowered:
        return "BF16"
    if "fp16" in lowered or "float16" in lowered:
        return "FP16"
    if "fp8" in lowered or "float8" in lowered:
        return "FP8"
    if "fp32" in lowered or "float32" in lowered:
        return "FP32"
    return None


def _mlx_quantization_bits(config: dict[str, Any] | None) -> int | None:
    if not isinstance(config, dict):
        return None
    for key in ("quantization", "quantization_config"):
        payload = config.get(key)
        if isinstance(payload, dict):
            bits = payload.get("bits")
            if isinstance(bits, (int, float)) and bits > 0:
                try:
                    return int(bits)
                except (TypeError, ValueError):
                    return None
    return None


def _dtype_quantization_label(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    candidates: list[Any] = [config.get("torch_dtype"), config.get("dtype")]
    for nested_key in ("text_config", "llm_config"):
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            candidates.extend([nested.get("torch_dtype"), nested.get("dtype")])
    for value in candidates:
        if not value:
            continue
        label = _quantization_label_from_text(str(value))
        if label:
            return label
    return None


def _detect_storage_format(path: Path, *, name_hint: str = "") -> str:
    lowered_hint = f"{name_hint} {path}".lower()
    if path.is_file() and path.suffix.lower() == ".gguf":
        return "GGUF"
    if _model_has_files(path, "*.gguf"):
        return "GGUF"

    config = _read_model_config(path)
    has_safetensors = _model_has_files(path, "*.safetensors")
    has_pytorch_bin = _model_has_files(path, "pytorch_model*.bin")
    looks_like_mlx = "mlx-community" in lowered_hint or bool(re.search(r"(^|[^a-z])mlx([^a-z]|$)", lowered_hint))

    if _mlx_quantization_bits(config) is not None and (config is not None or has_safetensors or has_pytorch_bin):
        return "MLX"
    if looks_like_mlx and (config is not None or has_safetensors or has_pytorch_bin):
        return "MLX"
    if has_safetensors or has_pytorch_bin:
        return "Transformers"
    if config is not None:
        return "MLX" if looks_like_mlx else "Transformers"
    return "unknown"


def _detect_model_quantization(path: Path, fmt: str, *, name_hint: str = "") -> str | None:
    text_hint = f"{name_hint} {path}"
    fmt_upper = (fmt or "").upper()
    if fmt_upper == "GGUF":
        main_file = _main_gguf_file(path if path.is_dir() else path.parent)
        if main_file is not None:
            label = _quantization_label_from_text(main_file.name)
            if label:
                return label
        return _quantization_label_from_text(text_hint)

    config = _read_model_config(path)
    bits = _mlx_quantization_bits(config)
    if bits is not None:
        return f"{bits}-bit"
    dtype_label = _dtype_quantization_label(config)
    if dtype_label:
        return dtype_label
    return _quantization_label_from_text(text_hint)


def _detect_directory_model(path: Path) -> tuple[str, str, str] | None:
    source_kind = "HF cache" if path.name.startswith("models--") else "Directory"
    name = path.name.replace("models--", "").replace("--", "/") if source_kind == "HF cache" else path.name
    if source_kind == "HF cache":
        detected_format = _detect_storage_format(path, name_hint=name)
        return (name, detected_format, source_kind) if detected_format != "unknown" else (name, "Transformers", source_kind)
    if any(path.glob("*.gguf")):
        return (name, "GGUF", source_kind)
    if (path / "config.json").exists() or (path / "tokenizer.json").exists():
        return (name, _detect_storage_format(path, name_hint=name), source_kind)
    return None


def _list_weight_files(raw_path: str) -> dict[str, Any]:
    """Inspect a model path and list its weight files.

    Used by the frontend picker to let users choose a specific .gguf when a
    directory contains multiple weights. Mirrors ``_resolve_gguf_path`` logic
    for GGUF directories.
    """
    target = Path(os.path.expanduser(raw_path or "")).expanduser()
    if not target.exists():
        return {
            "path": str(target),
            "format": "unknown",
            "files": [],
            "broken": True,
            "brokenReason": "Path does not exist",
        }

    def _gb(p: Path) -> float:
        try:
            return round(p.stat().st_size / (1024 ** 3), 2)
        except OSError:
            return 0.0

    # Single file
    if target.is_file():
        suffix = target.suffix.lower()
        if suffix == ".gguf":
            fmt = "GGUF"
        elif suffix == ".safetensors":
            fmt = "Transformers"
        else:
            fmt = "unknown"
        return {
            "path": str(target),
            "format": fmt,
            "files": [
                {
                    "name": target.name,
                    "path": str(target),
                    "sizeGb": _gb(target),
                    "role": "main",
                }
            ],
            "broken": False,
            "brokenReason": None,
        }

    # Directory
    ggufs = sorted(target.rglob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
    if ggufs:
        files = []
        for f in ggufs:
            is_mmproj = "mmproj" in f.name.lower()
            files.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "sizeGb": _gb(f),
                    "role": "mmproj" if is_mmproj else "main",
                }
            )
        return {
            "path": str(target),
            "format": "GGUF",
            "files": files,
            "broken": False,
            "brokenReason": None,
        }

    safetensors = sorted(target.glob("*.safetensors"))
    if safetensors:
        files = [
            {
                "name": f.name,
                "path": str(f),
                "sizeGb": _gb(f),
                "role": "main",
            }
            for f in safetensors
        ]
        has_mlx = any(f.name == "model.safetensors" for f in safetensors) or (target / "model.safetensors").exists()
        fmt = "MLX" if has_mlx and not (target / "model.safetensors.index.json").exists() else "Transformers"
        return {
            "path": str(target),
            "format": fmt,
            "files": files,
            "broken": False,
            "brokenReason": None,
        }

    # No weights found
    return {
        "path": str(target),
        "format": "unknown",
        "files": [],
        "broken": True,
        "brokenReason": "No .gguf or .safetensors weights found",
    }


def _detect_broken_library_item(child: Path, file_format: str, source_kind: str | None = None) -> tuple[bool, str | None]:
    """Return (broken, reason) for a discovered library item.

    Only directory-style entries can be broken; individual .gguf/.safetensors
    files are assumed healthy if they exist on disk.
    """
    try:
        if not child.is_dir():
            return (False, None)
    except OSError:
        return (False, None)

    fmt = (file_format or "").lower()
    source = (source_kind or "").lower()
    try:
        # HF cache entries are polymorphic: the same layout
        # (models--owner--name/snapshots/<rev>/...) can hold GGUF-only
        # mirrors, Transformers safetensors, MLX, or any combination.
        # Only flag broken if NONE of the expected weight formats are
        # present anywhere inside. Looking at file extensions instead of
        # the format label avoids the false-positive where an HF-cache
        # Transformers repo gets mislabelled as "GGUF broken" just
        # because the format slot says "HF cache".
        if source == "hf cache":
            has_gguf = any(child.rglob("*.gguf"))
            has_safetensors = any(child.rglob("*.safetensors"))
            has_pytorch_bin = any(child.rglob("pytorch_model*.bin"))
            if not (has_gguf or has_safetensors or has_pytorch_bin):
                return (True, "No .gguf, .safetensors, or pytorch weights found in HF cache entry")
            return (False, None)
        if fmt == "gguf" or "gguf" in fmt:
            if not any(child.rglob("*.gguf")):
                return (True, "No .gguf weights present")
            return (False, None)
        if fmt == "mlx":
            if not any(child.glob("*.safetensors")) and not (child / "model.safetensors").exists():
                return (True, "MLX directory missing model.safetensors")
            return (False, None)
        if fmt == "transformers":
            has_safetensors = any(child.glob("*.safetensors"))
            has_pytorch_bin = any(child.glob("pytorch_model*.bin"))
            if not has_safetensors and not has_pytorch_bin:
                return (True, "Transformers directory has no safetensors or pytorch weights")
            return (False, None)
    except OSError:
        return (False, None)
    return (False, None)


def _iter_discovered_models(root: Path, *, max_depth: int = 8) -> list[tuple[Path, str, str, str]]:
    discovered: list[tuple[Path, str, str, str]] = []
    # `.locks` is the Hugging Face hub lockfile directory. It mirrors the
    # `models--owner--name/` naming scheme, which would otherwise cause
    # the detector to produce phantom "broken" HF cache duplicates (lock
    # dirs contain no weights).
    skip_names = {"blobs", "refs", ".locks", ".cache", ".git", "__pycache__", ".venv", "node_modules"}

    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        depth = _relative_depth(current, root)
        if depth > max_depth:
            dirnames[:] = []
            continue

        # Prune by explicit skip list AND any dotfile/dot-directory so we
        # never wander into HF's `.locks`, `.cache`, etc.
        dirnames[:] = [
            name for name in dirnames
            if name not in skip_names and not name.startswith(".")
        ]

        if current != root:
            detected = _detect_directory_model(current)
            if detected is not None:
                discovered.append((current, detected[0], detected[1], detected[2]))
                dirnames[:] = []
                continue

        for filename in filenames:
            child = current / filename
            suffix = child.suffix.lower()
            if suffix not in {".gguf", ".safetensors"}:
                continue
            if suffix == ".safetensors" and (current / "config.json").exists():
                continue
            discovered.append((child, child.stem, suffix.replace(".", "").upper(), "File"))

    return discovered


def _discover_local_models(model_directories: list[dict[str, Any]], limit: int = 500) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for directory in model_directories:
        if not directory.get("enabled", True):
            continue
        raw_path = str(directory.get("path") or "").strip()
        if not raw_path:
            continue

        root = Path(os.path.expanduser(raw_path)).expanduser()
        if not root.exists():
            continue

        directory_label = str(directory.get("label") or root.name or "Model directory")
        directory_id = str(directory.get("id") or _normalize_slug(directory_label, "directory"))
        try:
            discovered = _iter_discovered_models(root)
        except OSError:
            continue

        for child, name, file_format, source_kind in discovered:
            if len(items) >= limit:
                return items
            try:
                if not child.exists():
                    continue
                path_key = str(child.resolve())
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                max_context = _detect_model_max_context(child, file_format)
                broken, broken_reason = _detect_broken_library_item(child, file_format, source_kind)
                quantization = _detect_model_quantization(child, file_format, name_hint=name)
                backend = "llama.cpp" if file_format == "GGUF" else "mlx"
                items.append(
                    {
                        "name": name,
                        "path": path_key,
                        "format": file_format,
                        "sourceKind": source_kind,
                        "quantization": quantization,
                        "backend": backend,
                        "sizeGb": _du_size_gb(child),
                        "lastModified": time.strftime("%Y-%m-%d %H:%M", time.localtime(child.stat().st_mtime)),
                        "actions": ["Run Chat", "Run Server", "Cache Preview", "Delete"],
                        "directoryId": directory_id,
                        "directoryLabel": directory_label,
                        "directoryPath": str(root),
                        "maxContext": max_context,
                        "broken": broken,
                        "brokenReason": broken_reason,
                    }
                    )
            except OSError:
                continue

    return items


def _reveal_path_in_file_manager(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{resolved} does not exist.")

    system_name = platform.system()
    if system_name == "Darwin":
        command = ["open", "-R", str(resolved)]
    elif system_name == "Windows":
        if resolved.is_file():
            command = ["explorer", f"/select,{resolved}"]
        else:
            command = ["explorer", str(resolved)]
    else:
        command = ["xdg-open", str(resolved.parent if resolved.is_file() else resolved)]

    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _estimate_runtime_memory_gb(params_b: float, quantization: str) -> float:
    lowered = quantization.lower()
    if "q4" in lowered or "4-bit" in lowered:
        quant_factor = 0.72
    elif "fp8" in lowered or "8" in lowered:
        quant_factor = 0.82
    else:
        quant_factor = 1.0
    return round(max(1.2, params_b * quant_factor + 1.6), 1)


def _variant_available_locally(variant: dict[str, Any], library: list[dict[str, Any]]) -> bool:
    candidates = {
        str(variant.get("repo") or "").lower(),
        str(variant.get("name") or "").lower(),
        str(variant.get("id") or "").lower(),
    }
    compact_candidates = {candidate.split("/")[-1] for candidate in candidates if candidate}
    for item in library:
        name = str(item.get("name") or "").lower()
        if name in candidates or any(candidate and candidate in name for candidate in candidates):
            return True
        if any(candidate and candidate in name for candidate in compact_candidates):
            return True
    return False


def _model_family_payloads(system_stats: dict[str, Any], library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        variants: list[dict[str, Any]] = []
        for variant in family["variants"]:
            runtime_memory = _estimate_runtime_memory_gb(variant["paramsB"], variant["quantization"])
            variants.append(
                {
                    **variant,
                    "familyId": family["id"],
                    "estimatedMemoryGb": runtime_memory,
                    "estimatedCompressedMemoryGb": round(max(1.0, runtime_memory * 0.68), 1),
                    "availableLocally": _variant_available_locally(variant, library),
                    "maxContext": _parse_context_label(variant.get("contextWindow")),
                }
            )

        payloads.append(
            {
                **family,
                "variants": variants,
            }
        )

    return payloads


def _search_huggingface_hub(query: str, library: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for models matching the query."""
    try:
        params = urllib.parse.urlencode({
            "search": query,
            "limit": str(limit),
            "sort": "downloads",
            "direction": "-1",
            "filter": "text-generation",
        })
        url = f"https://huggingface.co/api/models?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for model in data:
        model_id = str(model.get("id") or "")
        if not model_id:
            continue
        tags = model.get("tags") or []
        tag_set = {t.lower() for t in tags}

        # Determine format
        is_gguf = "gguf" in tag_set
        is_mlx = any("mlx" in t for t in tag_set)
        fmt = "GGUF" if is_gguf else "MLX" if is_mlx else "Transformers"
        launch_mode = "direct" if is_gguf else "convert"
        backend = "llama.cpp" if is_gguf else "mlx"

        # Check local availability
        name_lower = model_id.lower()
        available_locally = any(
            name_lower in str(item.get("name", "")).lower()
            or name_lower in str(item.get("path", "")).lower()
            for item in library
        )

        # Extract author/org
        parts = model_id.split("/", 1)
        provider = parts[0] if len(parts) > 1 else "Community"

        downloads = model.get("downloads") or 0
        likes = model.get("likes") or 0

        results.append({
            "id": model_id,
            "repo": model_id,
            "name": parts[-1] if parts else model_id,
            "provider": provider,
            "link": f"https://huggingface.co/{model_id}",
            "format": fmt,
            "tags": tags,
            "downloads": downloads,
            "likes": likes,
            "downloadsLabel": f"{downloads:,} downloads",
            "likesLabel": f"{likes:,} likes",
            "availableLocally": available_locally,
            "launchMode": launch_mode,
            "backend": backend,
        })

    return results


_HUB_FILE_CACHE: dict[str, dict[str, Any]] = {}
_IMAGE_DISCOVER_METADATA_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_IMAGE_DISCOVER_METADATA_TTL_SECONDS = 6 * 60 * 60
_LATEST_IMAGE_MODELS_CACHE: tuple[float, list[dict[str, Any]]] | None = None
_LATEST_IMAGE_MODELS_TTL_SECONDS = 3 * 60 * 60


def _classify_hub_file(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith((".gguf", ".safetensors", ".bin", ".pt", ".pth")):
        if "mmproj" in lowered:
            return "vision_projector"
        return "weight"
    if lowered in {"config.json", "generation_config.json"}:
        return "config"
    if "tokenizer" in lowered or lowered in {"vocab.json", "merges.txt", "special_tokens_map.json"}:
        return "tokenizer"
    if lowered.startswith("readme") or lowered.endswith(".md"):
        return "readme"
    if lowered.endswith((".jinja", ".chat_template")):
        return "template"
    return "other"


def _hub_repo_file_payload(
    repo_id: str,
    files: list[dict[str, Any]],
    *,
    total_bytes: int | None = None,
    license_value: str | None = None,
    tags: list[str] | None = None,
    pipeline_tag: str | None = None,
    last_modified: str | None = None,
    warning: str | None = None,
) -> dict[str, Any]:
    files.sort(key=lambda entry: (-int(entry.get("sizeBytes") or 0), str(entry.get("path") or "")))
    effective_total = total_bytes if total_bytes is not None else sum(int(entry.get("sizeBytes") or 0) for entry in files)
    return {
        "repo": repo_id,
        "files": files,
        "totalSizeBytes": int(effective_total),
        "totalSizeGb": _bytes_to_gb(effective_total),
        "license": license_value,
        "tags": tags or [],
        "pipelineTag": pipeline_tag,
        "lastModified": last_modified,
        "warning": warning,
    }


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_hf_updated_label(value: str | None) -> str | None:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    now = datetime.now(timezone.utc)
    month_label = parsed.strftime("%b")
    if parsed.year == now.year:
        return f"Updated {month_label} {parsed.day}"
    return f"Updated {month_label} {parsed.day}, {parsed.year}"


def _hf_number_label(value: int, noun: str) -> str:
    return f"{value:,} {noun}"


def _image_repo_live_metadata(repo_id: str) -> dict[str, Any]:
    now = time.time()
    cached = _IMAGE_DISCOVER_METADATA_CACHE.get(repo_id)
    if cached is not None:
        cached_at, payload = cached
        if (now - cached_at) < _IMAGE_DISCOVER_METADATA_TTL_SECONDS:
            return payload

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    payload: dict[str, Any]
    try:
        encoded_repo = urllib.parse.quote(repo_id, safe="/")
        url = f"https://huggingface.co/api/models/{encoded_repo}?blobs=true"
        req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())

        total_bytes = 0
        weight_bytes = 0
        for sibling in data.get("siblings") or []:
            if not isinstance(sibling, dict):
                continue
            path = str(sibling.get("rfilename") or "")
            if not path:
                continue
            lfs = sibling.get("lfs") if isinstance(sibling.get("lfs"), dict) else {}
            size_bytes = sibling.get("size") or lfs.get("size") or 0
            try:
                size_int = int(size_bytes)
            except (TypeError, ValueError):
                size_int = 0
            total_bytes += size_int
            if _classify_hub_file(path) == "weight":
                weight_bytes += size_int

        card = data.get("cardData") or {}
        license_value = str(card.get("license") or "").strip() or None if isinstance(card, dict) else None
        downloads = int(data.get("downloads") or 0)
        likes = int(data.get("likes") or 0)
        last_modified = str(data.get("lastModified") or "").strip() or None
        payload = {
            "downloads": downloads,
            "likes": likes,
            "downloadsLabel": _hf_number_label(downloads, "downloads") if downloads > 0 else None,
            "likesLabel": _hf_number_label(likes, "likes") if likes > 0 else None,
            "lastModified": last_modified,
            "updatedLabel": _format_hf_updated_label(last_modified),
            "license": license_value,
            "gated": bool(data.get("gated")),
            "pipelineTag": str(data.get("pipeline_tag") or "").strip() or None,
            "repoSizeBytes": total_bytes or None,
            "repoSizeGb": _bytes_to_gb(total_bytes) if total_bytes > 0 else None,
            "coreWeightsBytes": weight_bytes or None,
            "coreWeightsGb": _bytes_to_gb(weight_bytes) if weight_bytes > 0 else None,
            "metadataWarning": None,
        }
    except urllib.error.HTTPError as exc:
        status = getattr(exc, "code", None)
        payload = {
            "metadataWarning": (
                f"Live Hugging Face metadata is temporarily unavailable (HTTP {status}). Showing curated defaults."
                if status is not None
                else "Live Hugging Face metadata is temporarily unavailable. Showing curated defaults."
            ),
        }
    except (OSError, json.JSONDecodeError):
        payload = {
            "metadataWarning": "Live Hugging Face metadata is temporarily unavailable. Showing curated defaults.",
        }

    _IMAGE_DISCOVER_METADATA_CACHE[repo_id] = (now, payload)
    return payload


def _best_image_family_updated_label(family: dict[str, Any], variants: list[dict[str, Any]]) -> str:
    best_dt: datetime | None = None
    best_label: str | None = None
    for variant in variants:
        last_modified = _parse_iso_datetime(str(variant.get("lastModified") or "") or None)
        if last_modified is None:
            continue
        if best_dt is None or last_modified > best_dt:
            best_dt = last_modified
            best_label = str(variant.get("updatedLabel") or "") or None
    return best_label or str(family.get("updatedLabel") or "Curated")


def _image_task_support_from_metadata(pipeline_tag: str | None, tags: list[str]) -> list[str]:
    pipeline = str(pipeline_tag or "").lower()
    lowered_tags = {str(tag).lower() for tag in tags}
    tasks: list[str] = []
    if (
        pipeline == "text-to-image"
        or "text-to-image" in lowered_tags
        or "image-generation" in lowered_tags
    ):
        tasks.append("txt2img")
    if (
        pipeline == "image-to-image"
        or "image-to-image" in lowered_tags
        or "image-edit" in lowered_tags
        or "editing" in lowered_tags
    ):
        tasks.append("img2img")
    if pipeline == "inpainting" or "inpainting" in lowered_tags or "inpaint" in lowered_tags:
        tasks.append("inpaint")
    return tasks or ["txt2img"]


def _image_recommended_resolution(repo_id: str, pipeline_tag: str | None, tags: list[str]) -> str:
    lowered = repo_id.lower()
    lowered_tags = {str(tag).lower() for tag in tags}
    if "2048" in lowered or "2k" in lowered_tags or "hunyuanimage-2.1" in lowered:
        return "2048x2048"
    if "768" in lowered:
        return "768x768"
    if "512" in lowered:
        return "512x512"
    if "1024" in lowered or "sdxl" in lowered or "flux" in lowered or "sana" in lowered:
        return "1024x1024"
    if str(pipeline_tag or "").lower() == "text-to-image":
        return "1024x1024"
    return "Unknown"


def _image_discover_style_tags(tags: list[str]) -> list[str]:
    preferred = {
        "photoreal",
        "illustration",
        "anime",
        "general",
        "fast",
        "detailed",
        "turbo",
        "distilled",
        "edit",
        "inpaint",
        "flux",
        "sana",
        "qwenimage",
        "hidream",
    }
    seen: list[str] = []
    for tag in tags:
        lowered = str(tag).lower()
        if lowered in preferred and lowered not in seen:
            seen.append(lowered)
    return seen[:4]


def _tracked_latest_seed_payloads(library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for seed in LATEST_IMAGE_TRACKED_SEEDS:
        repo_id = str(seed.get("repo") or "")
        if not repo_id:
            continue
        payloads.append(
            {
                "id": repo_id,
                "familyId": "latest",
                "familyName": "Latest Releases",
                "name": seed.get("name") or repo_id.split("/", 1)[-1],
                "provider": seed.get("provider") or (repo_id.split("/", 1)[0] if "/" in repo_id else "Community"),
                "repo": repo_id,
                "link": f"https://huggingface.co/{repo_id}",
                "runtime": "Tracked diffusers candidate",
                "styleTags": list(seed.get("styleTags") or []),
                "taskSupport": list(seed.get("taskSupport") or ["txt2img"]),
                "sizeGb": float(seed.get("sizeGb") or 0.0),
                "recommendedResolution": str(seed.get("recommendedResolution") or "Unknown"),
                "note": str(
                    seed.get("note")
                    or "Tracked latest image repo surfaced by ChaosEngineAI when the live latest lane is sparse."
                ),
                "availableLocally": _image_repo_runtime_ready(repo_id),
                "estimatedGenerationSeconds": None,
                "downloads": None,
                "likes": None,
                "downloadsLabel": None,
                "likesLabel": None,
                "lastModified": None,
                "updatedLabel": str(seed.get("updatedLabel") or "Tracked latest"),
                "license": seed.get("license"),
                "gated": seed.get("gated"),
                "pipelineTag": seed.get("pipelineTag"),
                "repoSizeBytes": None,
                "repoSizeGb": None,
                "coreWeightsBytes": None,
                "coreWeightsGb": None,
                "metadataWarning": "Showing ChaosEngineAI tracked latest defaults until live Hugging Face metadata is available.",
                "source": "latest",
            }
        )
    return payloads


def _image_download_repo_ids() -> set[str]:
    repos = {
        str(variant.get("repo") or "")
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "")
    }
    repos.update(
        str(seed.get("repo") or "")
        for seed in LATEST_IMAGE_TRACKED_SEEDS
        if str(seed.get("repo") or "")
    )
    cached_entries = _LATEST_IMAGE_MODELS_CACHE
    if cached_entries is not None:
        repos.update(
            str(entry.get("repo") or "")
            for entry in cached_entries[1]
            if str(entry.get("repo") or "")
        )
    return repos


def _is_latest_image_candidate(model: dict[str, Any], curated_repos: set[str]) -> bool:
    model_id = str(model.get("id") or "")
    if not model_id or model_id in curated_repos:
        return False
    lowered = model_id.lower()
    excluded_fragments = (
        "-lora",
        "controlnet",
        "ip-adapter",
        "tensorrt",
        "_amdgpu",
        "onnx",
        "instruct-pix2pix",
    )
    if any(fragment in lowered for fragment in excluded_fragments):
        return False

    tags = {str(tag).lower() for tag in (model.get("tags") or [])}
    pipeline_tag = str(model.get("pipeline_tag") or "").lower()
    allowed_orgs = {
        "black-forest-labs",
        "stabilityai",
        "qwen",
        "hidream-ai",
        "zai-org",
        "efficient-large-model",
        "hunyuanvideo-community",
        "tencent-hunyuan",
        "thudm",
    }
    provider = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
    if provider and provider not in allowed_orgs:
        return False

    if "diffusers" not in tags:
        return False
    image_pipelines = {"text-to-image", "image-to-image", "inpainting"}
    if pipeline_tag in image_pipelines:
        return True
    if {"text-to-image", "image-generation", "image-to-image", "inpainting", "inpaint"} & tags:
        return True
    return False


def _latest_image_model_payloads(library: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    global _LATEST_IMAGE_MODELS_CACHE

    curated_repos = {
        str(variant.get("repo") or "")
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "")
    }

    now = time.time()
    cached_entries = _LATEST_IMAGE_MODELS_CACHE
    if cached_entries is not None and (now - cached_entries[0]) < _LATEST_IMAGE_MODELS_TTL_SECONDS:
        latest = cached_entries[1]
        return [
            {
                **entry,
                "availableLocally": _image_repo_runtime_ready(str(entry.get("repo") or "")),
            }
            for entry in latest
        ]

    try:
        params = urllib.parse.urlencode({
            "filter": "diffusers",
            "sort": "modified",
            "direction": "-1",
            "limit": "48",
            "full": "true",
        })
        url = f"https://huggingface.co/api/models?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        if cached_entries is not None:
            latest = cached_entries[1]
            return [
                {
                    **entry,
                    "availableLocally": _image_repo_runtime_ready(str(entry.get("repo") or "")),
                }
                for entry in latest
            ]
        return _tracked_latest_seed_payloads(library)[:limit]

    candidates: list[dict[str, Any]] = []
    for model in data:
        if not isinstance(model, dict) or not _is_latest_image_candidate(model, curated_repos):
            continue
        model_id = str(model.get("id") or "")
        provider = model_id.split("/", 1)[0] if "/" in model_id else "Community"
        tags = [str(tag) for tag in (model.get("tags") or [])]
        pipeline_tag = str(model.get("pipeline_tag") or "").strip() or None
        metadata = _image_repo_live_metadata(model_id)
        candidates.append({
            "id": model_id,
            "familyId": "latest",
            "familyName": "Latest Releases",
            "name": model_id.split("/", 1)[-1],
            "provider": provider,
            "repo": model_id,
            "link": f"https://huggingface.co/{model_id}",
            "runtime": "Diffusers candidate",
            "styleTags": _image_discover_style_tags(tags),
            "taskSupport": _image_task_support_from_metadata(pipeline_tag, tags),
            "sizeGb": float(metadata.get("coreWeightsGb") or metadata.get("repoSizeGb") or 0.0),
            "recommendedResolution": _image_recommended_resolution(model_id, pipeline_tag, tags),
            "note": (
                "Latest official diffusers-compatible image model tracked by ChaosEngineAI. "
                "Review details on Hugging Face before treating it as a fully curated Studio default."
            ),
            "availableLocally": _image_repo_runtime_ready(model_id),
            "estimatedGenerationSeconds": None,
            "downloads": metadata.get("downloads"),
            "likes": metadata.get("likes"),
            "downloadsLabel": metadata.get("downloadsLabel"),
            "likesLabel": metadata.get("likesLabel"),
            "lastModified": metadata.get("lastModified"),
            "updatedLabel": metadata.get("updatedLabel"),
            "license": metadata.get("license"),
            "gated": bool(metadata.get("gated")) if metadata.get("gated") is not None else None,
            "pipelineTag": metadata.get("pipelineTag") or pipeline_tag,
            "repoSizeBytes": metadata.get("repoSizeBytes"),
            "repoSizeGb": metadata.get("repoSizeGb"),
            "coreWeightsBytes": metadata.get("coreWeightsBytes"),
            "coreWeightsGb": metadata.get("coreWeightsGb"),
            "metadataWarning": metadata.get("metadataWarning"),
            "source": "latest",
        })

    candidates.sort(
        key=lambda entry: (
            _parse_iso_datetime(str(entry.get("lastModified") or "") or None) or datetime.min.replace(tzinfo=timezone.utc),
            int(entry.get("downloads") or 0),
            int(entry.get("likes") or 0),
        ),
        reverse=True,
    )
    seen_repos = {str(entry.get("repo") or "") for entry in candidates}
    for fallback in _tracked_latest_seed_payloads(library):
        repo_id = str(fallback.get("repo") or "")
        if repo_id in seen_repos:
            continue
        candidates.append(fallback)
        seen_repos.add(repo_id)

    latest = candidates[:limit]
    _LATEST_IMAGE_MODELS_CACHE = (now, latest)
    return latest


def _known_repo_size_gb(repo_id: str) -> float | None:
    cached = _HUB_FILE_CACHE.get(repo_id)
    if cached is not None:
        cached_total = cached.get("totalSizeGb")
        if isinstance(cached_total, (int, float)) and cached_total > 0:
            return float(cached_total)

    for family in MODEL_FAMILIES:
        for variant in family["variants"]:
            if str(variant.get("repo") or "") != repo_id:
                continue
            try:
                size_gb = float(variant.get("sizeGb") or 0)
            except (TypeError, ValueError):
                size_gb = 0.0
            if size_gb > 0:
                return size_gb

    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if str(variant.get("repo") or "") != repo_id:
                continue
            try:
                size_gb = float(variant.get("sizeGb") or 0)
            except (TypeError, ValueError):
                size_gb = 0.0
            if size_gb > 0:
                return size_gb

    return None


def _hf_hub_cache_root() -> Path:
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(os.path.expanduser(explicit)).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(os.path.expanduser(hf_home)).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_cache_dir(repo_id: str) -> Path:
    return _hf_hub_cache_root() / f"models--{repo_id.replace('/', '--')}"


def _hf_repo_downloaded_bytes(repo_id: str) -> int:
    cache_dir = _hf_repo_cache_dir(repo_id)
    try:
        if not cache_dir.exists():
            return 0
    except OSError:
        return 0
    try:
        return _path_size_bytes(cache_dir)
    except OSError:
        return 0


def _hf_repo_snapshot_dir(repo_id: str) -> Path | None:
    cache_dir = _hf_repo_cache_dir(repo_id)
    snapshots_dir = cache_dir / "snapshots"
    ref_path = cache_dir / "refs" / "main"
    try:
        if ref_path.exists():
            revision = ref_path.read_text(encoding="utf-8").strip()
            if revision:
                candidate = snapshots_dir / revision
                if candidate.exists():
                    return candidate
    except OSError:
        pass

    try:
        snapshots = sorted(
            [candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir()],
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    return snapshots[0] if snapshots else None


def _is_image_repo(repo_id: str) -> bool:
    return any(
        str(variant.get("repo") or "") == repo_id
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
    )


def _image_repo_runtime_ready(repo_id: str) -> bool:
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return False
    return validate_local_diffusers_snapshot(snapshot_dir, repo_id) is None


def _image_variant_available_locally(variant: dict[str, Any], library: list[dict[str, Any]]) -> bool:
    repo = str(variant.get("repo") or "")
    if repo and _image_repo_runtime_ready(repo):
        return True

    candidates = {
        str(variant.get("repo") or "").lower(),
        str(variant.get("name") or "").lower(),
        str(variant.get("id") or "").lower(),
    }
    compact_candidates = {candidate.split("/")[-1] for candidate in candidates if candidate}
    for item in library:
        name = str(item.get("name") or "").lower()
        if not (
            name in candidates
            or any(candidate and candidate in name for candidate in candidates)
            or any(candidate and candidate in name for candidate in compact_candidates)
        ):
            continue
        item_path = Path(str(item.get("path") or "")).expanduser()
        for directory in _candidate_model_dirs(item_path):
            if validate_local_diffusers_snapshot(directory) is None:
                return True
    return False


def _image_download_validation_error(repo_id: str) -> str | None:
    if not _is_image_repo(repo_id):
        return None
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return (
            f"Download did not produce a local snapshot for {repo_id}. "
            "Retry the download and make sure the backend can access Hugging Face."
        )
    return validate_local_diffusers_snapshot(snapshot_dir, repo_id)


def _friendly_image_download_error(repo_id: str, error: str) -> str:
    if not _is_image_repo(repo_id):
        return error
    lowered = error.lower()
    if (
        "cannot access gated repo" in lowered
        or "gated repo" in lowered
        or "authorized list" in lowered
        or ("access to model" in lowered and "restricted" in lowered)
    ):
        return (
            f"{repo_id} is gated on Hugging Face. Your account or token is not approved for this model yet. "
            f"Open https://huggingface.co/{repo_id}, request or accept access, add a read-enabled HF_TOKEN in Settings, then retry."
        )
    return error


def _hub_repo_files(repo_id: str) -> dict[str, Any]:
    """Return file list + metadata for a Hugging Face repo.

    Uses the public REST API so transient failures on the heavier tree
    endpoint do not make Discover look broken. Honours HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN for gated repos and degrades to a non-fatal
    warning on transient upstream 5xx errors.
    """
    cached = _HUB_FILE_CACHE.get(repo_id)
    if cached is not None:
        return cached

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        encoded_repo = urllib.parse.quote(repo_id, safe="/")
        url = f"https://huggingface.co/api/models/{encoded_repo}?blobs=true"
        req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        status = getattr(exc, "code", None)
        if status in (401, 403):
            raise RuntimeError(
                f"Hugging Face refused access to {repo_id} (HTTP {status}). "
                f"Set HF_TOKEN in Settings."
            ) from exc
        if status == 404:
            raise RuntimeError(f"Hugging Face repository not found: {repo_id}") from exc
        if status is not None and status >= 500:
            return _hub_repo_file_payload(
                repo_id,
                [],
                warning=(
                    f"Hugging Face file preview is temporarily unavailable (HTTP {status}). "
                    "You can still download this repo."
                ),
            )
        raise RuntimeError(f"Hugging Face request failed: {exc}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        return _hub_repo_file_payload(
            repo_id,
            [],
            warning=(
                "Hugging Face file preview is temporarily unavailable right now. "
                "You can still download this repo."
            ),
        )

    all_files: list[dict[str, Any]] = []
    display_files: list[dict[str, Any]] = []
    total_bytes = 0
    for sibling in data.get("siblings") or []:
        path = str(sibling.get("rfilename") or "")
        if not path:
            continue
        lfs = sibling.get("lfs") if isinstance(sibling.get("lfs"), dict) else {}
        size_bytes = sibling.get("size") or lfs.get("size") or 0
        try:
            size_int = int(size_bytes)
        except (TypeError, ValueError):
            size_int = 0
        record = {
            "path": path,
            "sizeBytes": size_int,
            "sizeGb": _bytes_to_gb(size_int),
            "kind": _classify_hub_file(path),
        }
        total_bytes += size_int
        all_files.append(record)
        if "/" not in path:
            display_files.append(record)

    if not display_files:
        display_files = all_files[:40]

    card = data.get("cardData") or {}
    license_value = card.get("license") if isinstance(card, dict) else None
    payload = _hub_repo_file_payload(
        repo_id,
        display_files,
        total_bytes=total_bytes,
        license_value=license_value,
        tags=list(data.get("tags") or []),
        pipeline_tag=data.get("pipeline_tag"),
        last_modified=data.get("lastModified"),
    )
    _HUB_FILE_CACHE[repo_id] = payload
    return payload


def _default_chat_variant() -> dict[str, Any]:
    direct_variants = sorted(
        [entry for entry in CATALOG if entry.get("launchMode") == "direct"],
        key=lambda entry: (float(entry.get("paramsB") or 0), float(entry.get("sizeGb") or 0)),
    )
    return direct_variants[0] if direct_variants else CATALOG[0]


def _seed_chat_sessions() -> list[dict[str, Any]]:
    default_variant = _default_chat_variant()
    return [
        {
            "id": "ui-direction",
            "title": "Compact desktop layout",
            "updatedAt": "Today 17:18",
            "model": "Devstral Small 2507 GGUF",
            "modelRef": "mistralai/Devstral-Small-2507_gguf",
            "modelSource": "catalog",
            "modelPath": None,
            "modelBackend": "llama.cpp",
            "pinned": True,
            "cacheLabel": "Native 3-bit 4+4",
            "messages": [
                {
                    "role": "user",
                    "text": "Make the desktop UI feel tighter and more like a serious desktop tool.",
                },
                {
                    "role": "assistant",
                    "text": "I would reduce padding, tighten nav density, simplify the dashboard, and make threads and models feel more task-oriented than card-oriented.",
                    "metrics": None,
                },
            ],
        },
        {
            "id": "model-shortlist",
            "title": "Try newer local models",
            "updatedAt": "Today 15:42",
            "model": default_variant["name"],
            "modelRef": default_variant["id"],
            "modelSource": "catalog",
            "modelPath": None,
            "modelBackend": default_variant.get("backend", "auto"),
            "pinned": False,
            "cacheLabel": "Native 3-bit 4+4",
            "messages": [
                {
                    "role": "user",
                    "text": "Which local-first models feel fresher than the old Qwen2.5 shortlist?",
                },
                {
                    "role": "assistant",
                    "text": "Gemma 4, Qwen 3.5, Nemotron 3 Nano, and Devstral are stronger directions to browse first depending on whether you care most about vision, reasoning, or coding.",
                    "metrics": None,
                },
            ],
        },
    ]


def _context_label(value: int | None) -> str:
    if value is None:
        return "Unknown"
    if value >= 1_000_000:
        return f"{round(value / 1_000_000, 1)}M"
    if value >= 1_000:
        return f"{round(value / 1_000)}K"
    return str(value)


_CONTEXT_LABEL_RE = re.compile(r"^\s*([\d.]+)\s*([KMkm]?)\s*$")


def _parse_context_label(label: str | None) -> int | None:
    """Parse a human-readable context window label (e.g. '128K', '1M', '262K')
    into an integer token count. Returns None on failure."""
    if label is None:
        return None
    match = _CONTEXT_LABEL_RE.match(str(label))
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    unit = (match.group(2) or "").upper()
    if unit == "K":
        # Labels like "256K" / "262K" / "128K" / "32K" -> power-of-two style values
        # Use 1024 multiplier so "128K" -> 131072 which matches model configs.
        return int(round(value * 1024))
    if unit == "M":
        return int(round(value * 1_000_000))
    return int(round(value))


# GGUF value type codes (see https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
_GGUF_UINT8 = 0
_GGUF_INT8 = 1
_GGUF_UINT16 = 2
_GGUF_INT16 = 3
_GGUF_UINT32 = 4
_GGUF_INT32 = 5
_GGUF_FLOAT32 = 6
_GGUF_BOOL = 7
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_UINT64 = 10
_GGUF_INT64 = 11
_GGUF_FLOAT64 = 12

_GGUF_SCALAR_SIZES = {
    _GGUF_UINT8: 1,
    _GGUF_INT8: 1,
    _GGUF_UINT16: 2,
    _GGUF_INT16: 2,
    _GGUF_UINT32: 4,
    _GGUF_INT32: 4,
    _GGUF_FLOAT32: 4,
    _GGUF_BOOL: 1,
    _GGUF_UINT64: 8,
    _GGUF_INT64: 8,
    _GGUF_FLOAT64: 8,
}


def _read_gguf_context_length(path: Path) -> int | None:
    """Cheaply read the GGUF header and return the architecture's context_length.

    We parse just enough of the metadata KV section to find any key ending in
    ".context_length". We never read tensor data. Returns None on any failure.
    """
    import struct

    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                return None
            header = fh.read(4 + 8 + 8)  # version u32, tensor_count u64, kv_count u64
            if len(header) < 20:
                return None
            version, _tensor_count, kv_count = struct.unpack("<IQQ", header)
            if version < 2 or kv_count > 10_000:
                return None

            def _read_exact(n: int) -> bytes:
                buf = fh.read(n)
                if len(buf) != n:
                    raise EOFError
                return buf

            def _read_string() -> str:
                (length,) = struct.unpack("<Q", _read_exact(8))
                if length > 1 << 20:
                    raise ValueError("gguf string too long")
                return _read_exact(length).decode("utf-8", errors="replace")

            def _skip_value(vtype: int) -> None:
                if vtype in _GGUF_SCALAR_SIZES:
                    _read_exact(_GGUF_SCALAR_SIZES[vtype])
                    return
                if vtype == _GGUF_STRING:
                    _read_string()
                    return
                if vtype == _GGUF_ARRAY:
                    (inner_type,) = struct.unpack("<I", _read_exact(4))
                    (count,) = struct.unpack("<Q", _read_exact(8))
                    if count > 1 << 24:
                        raise ValueError("gguf array too long")
                    if inner_type in _GGUF_SCALAR_SIZES:
                        _read_exact(_GGUF_SCALAR_SIZES[inner_type] * count)
                        return
                    for _ in range(count):
                        _skip_value(inner_type)
                    return
                raise ValueError(f"unknown gguf type {vtype}")

            def _read_value(vtype: int) -> Any:
                if vtype == _GGUF_UINT32:
                    return struct.unpack("<I", _read_exact(4))[0]
                if vtype == _GGUF_INT32:
                    return struct.unpack("<i", _read_exact(4))[0]
                if vtype == _GGUF_UINT64:
                    return struct.unpack("<Q", _read_exact(8))[0]
                if vtype == _GGUF_INT64:
                    return struct.unpack("<q", _read_exact(8))[0]
                if vtype == _GGUF_UINT16:
                    return struct.unpack("<H", _read_exact(2))[0]
                if vtype == _GGUF_INT16:
                    return struct.unpack("<h", _read_exact(2))[0]
                _skip_value(vtype)
                return None

            best: int | None = None
            for _ in range(kv_count):
                key = _read_string()
                (vtype,) = struct.unpack("<I", _read_exact(4))
                if key.endswith(".context_length") or key == "context_length":
                    value = _read_value(vtype)
                    if isinstance(value, int) and value > 0:
                        if best is None or value > best:
                            best = value
                else:
                    _skip_value(vtype)
            return best
    except (OSError, ValueError, EOFError, struct.error):
        return None


def _read_config_max_context(config_path: Path) -> int | None:
    """Read an HF/MLX config.json and extract the model's max context length."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None

    candidates = (
        "max_position_embeddings",
        "max_sequence_length",
        "max_seq_len",
        "n_positions",
        "seq_length",
        "model_max_length",
    )
    base: int | None = None
    for key in candidates:
        value = config.get(key)
        if isinstance(value, (int, float)) and value > 0:
            base = int(value)
            break

    if base is None:
        # Some models nest under text_config / llm_config
        for nested_key in ("text_config", "llm_config"):
            nested = config.get(nested_key)
            if isinstance(nested, dict):
                for key in candidates:
                    value = nested.get(key)
                    if isinstance(value, (int, float)) and value > 0:
                        base = int(value)
                        break
            if base is not None:
                break

    if base is None:
        return None

    rope_scaling = config.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        factor = rope_scaling.get("factor")
        if isinstance(factor, (int, float)) and factor > 1:
            try:
                base = int(base * float(factor))
            except (TypeError, ValueError):
                pass
    return base


def _detect_model_max_context(path: Path, fmt: str) -> int | None:
    """Return the detected max context length for a discovered model, or None.

    Never raises — returns None on any parse failure.
    """
    try:
        fmt_upper = (fmt or "").upper()
        if fmt_upper == "GGUF" or path.suffix.lower() == ".gguf":
            if path.is_file():
                return _read_gguf_context_length(path)
            main_file = _main_gguf_file(path)
            if main_file is not None:
                return _read_gguf_context_length(main_file)
            return None
        # Directory-based (MLX / Transformers / HF cache)
        search_dir = path if path.is_dir() else path.parent
        config_path = search_dir / "config.json"
        if config_path.exists():
            return _read_config_max_context(config_path)
        # HF cache layout: models--org--name/snapshots/<rev>/config.json
        snapshots = search_dir / "snapshots"
        if snapshots.is_dir():
            for snap in snapshots.iterdir():
                candidate = snap / "config.json"
                if candidate.exists():
                    result = _read_config_max_context(candidate)
                    if result is not None:
                        return result
    except Exception:
        return None
    return None


def _benchmark_label(model_name: str, *, cache_strategy: str, bits: int, fp16_layers: int, context_tokens: int) -> str:
    from compression import registry as _strategy_registry
    strat = _strategy_registry.get(cache_strategy) or _strategy_registry.default()
    cache_label = strat.label(bits, fp16_layers)
    return f"{model_name} / {cache_label} / {_context_label(context_tokens)} ctx"


def _seed_benchmark_runs() -> list[dict[str, Any]]:
    return [
        {
            "id": "baseline",
            "label": "Nemotron 3 Nano 4B GGUF / Native f16 / 8K ctx",
            "model": "Nemotron 3 Nano 4B GGUF",
            "modelRef": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            "backend": "llama.cpp",
            "engineLabel": "llama.cpp + GGUF",
            "source": "catalog",
            "measuredAt": "2026-04-05 12:12:08",
            "bits": 16,
            "fp16Layers": 0,
            "cacheStrategy": "native",
            "cacheLabel": "Native f16 cache",
            "cacheGb": 14.0,
            "baselineCacheGb": 14.0,
            "compression": 1.0,
            "tokS": 52.1,
            "quality": 100,
            "responseSeconds": 4.2,
            "loadSeconds": 5.8,
            "totalSeconds": 10.0,
            "promptTokens": 78,
            "completionTokens": 219,
            "totalTokens": 297,
            "contextTokens": 8192,
            "maxTokens": 256,
            "notes": "Seed baseline run for comparison.",
        },
        {
            "id": "native-34",
            "label": "Nemotron 3 Nano 4B GGUF / Native 3-bit 4+4 / 8K ctx",
            "model": "Nemotron 3 Nano 4B GGUF",
            "modelRef": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            "backend": "llama.cpp",
            "engineLabel": "llama.cpp + GGUF",
            "source": "catalog",
            "measuredAt": "2026-04-05 12:18:44",
            "bits": 3,
            "fp16Layers": 4,
            "cacheStrategy": "native",
            "cacheLabel": "Native 3-bit 4+4",
            "cacheGb": 5.9,
            "baselineCacheGb": 14.0,
            "compression": 2.4,
            "tokS": 30.7,
            "quality": 98,
            "responseSeconds": 7.1,
            "loadSeconds": 6.0,
            "totalSeconds": 13.1,
            "promptTokens": 78,
            "completionTokens": 218,
            "totalTokens": 296,
            "contextTokens": 8192,
            "maxTokens": 256,
            "notes": "Seed adaptive cache strategy run.",
        },
        {
            "id": "native-36",
            "label": "Nemotron 3 Nano 4B GGUF / Native 3-bit 6+6 / 8K ctx",
            "model": "Nemotron 3 Nano 4B GGUF",
            "modelRef": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            "backend": "llama.cpp",
            "engineLabel": "llama.cpp + GGUF",
            "source": "catalog",
            "measuredAt": "2026-04-05 12:25:19",
            "bits": 3,
            "fp16Layers": 6,
            "cacheStrategy": "native",
            "cacheLabel": "Native 3-bit 6+6",
            "cacheGb": 7.5,
            "baselineCacheGb": 14.0,
            "compression": 1.9,
            "tokS": 33.0,
            "quality": 98,
            "responseSeconds": 6.7,
            "loadSeconds": 6.1,
            "totalSeconds": 12.8,
            "promptTokens": 78,
            "completionTokens": 220,
            "totalTokens": 298,
            "contextTokens": 8192,
            "maxTokens": 256,
            "notes": "Seed higher-FP16 edge run.",
        },
        {
            "id": "native-44",
            "label": "Nemotron 3 Nano 4B GGUF / Native 4-bit 4+4 / 8K ctx",
            "model": "Nemotron 3 Nano 4B GGUF",
            "modelRef": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            "backend": "llama.cpp",
            "engineLabel": "llama.cpp + GGUF",
            "source": "catalog",
            "measuredAt": "2026-04-05 12:32:57",
            "bits": 4,
            "fp16Layers": 4,
            "cacheStrategy": "native",
            "cacheLabel": "Native 4-bit 4+4",
            "cacheGb": 7.1,
            "baselineCacheGb": 14.0,
            "compression": 2.0,
            "tokS": 35.8,
            "quality": 99,
            "responseSeconds": 6.0,
            "loadSeconds": 6.0,
            "totalSeconds": 12.0,
            "promptTokens": 78,
            "completionTokens": 215,
            "totalTokens": 293,
            "contextTokens": 8192,
            "maxTokens": 256,
            "notes": "Seed higher-quality cache strategy run.",
        },
    ]


def _load_benchmark_runs(path: Path = BENCHMARKS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return _seed_benchmark_runs()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _seed_benchmark_runs()

    if not isinstance(payload, list):
        return _seed_benchmark_runs()

    valid_runs = [item for item in payload if isinstance(item, dict) and item.get("id") and item.get("label")]
    return valid_runs[:MAX_BENCHMARK_RUNS] or _seed_benchmark_runs()


def _save_benchmark_runs(runs: list[dict[str, Any]], path: Path = BENCHMARKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runs[:MAX_BENCHMARK_RUNS], indent=2), encoding="utf-8")


def _load_chat_sessions(path: Path = CHAT_SESSIONS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []

    valid = [s for s in payload if isinstance(s, dict) and s.get("id") and s.get("title")]
    return valid[:MAX_CHAT_SESSIONS]


def _save_chat_sessions(sessions: list[dict[str, Any]], path: Path = CHAT_SESSIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sessions[:MAX_CHAT_SESSIONS], indent=2, default=str), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(str(tmp), str(path))


def _sanitize_filename(name: str) -> str:
    """Strip path traversal and dangerous characters from a filename."""
    name = os.path.basename(name).strip()
    name = re.sub(r"[^\w\-. ]", "_", name)
    return name[:200] or "file"


def _extract_text_from_file(path: Path) -> str:
    """Extract plain text from a supported document file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n\n".join(parts)
        except Exception as exc:
            raise RuntimeError(f"Could not read PDF: {exc}") from exc
    # Plain text files (including code)
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"Could not read file: {exc}") from exc


def _chunk_text(text: str, *, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Sliding-window chunker. Returns list of overlapping text chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _retrieve_relevant_chunks(prompt: str, chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Score chunks by keyword overlap with the prompt and return top K."""
    if not chunks:
        return []
    prompt_terms = set(re.findall(r"\w+", prompt.lower()))
    if not prompt_terms:
        return chunks[:top_k]

    scored = []
    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_terms = set(re.findall(r"\w+", text.lower()))
        if not chunk_terms:
            continue
        overlap = len(prompt_terms & chunk_terms)
        if overlap > 0:
            scored.append((overlap, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def _local_ipv4_addresses() -> list[str]:
    discovered: set[str] = set()
    for interface_addresses in psutil.net_if_addrs().values():
        for address in interface_addresses:
            if address.family != socket.AF_INET:
                continue
            value = str(address.address or "").strip()
            if not value or value.startswith("127.") or value.startswith("169.254."):
                continue
            try:
                ip = ipaddress.ip_address(value)
            except ValueError:
                continue
            if ip.is_loopback or ip.is_link_local:
                continue
            discovered.add(value)
    return sorted(discovered)


def _estimate_baseline_tok_s(system_stats: dict[str, Any]) -> float:
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 8
    baseline = 15.0 + cpu_count * 1.1 + system_stats["totalMemoryGb"] * 0.5
    return round(baseline, 1)


def compute_cache_preview(
    *,
    bits: int = 3,
    fp16_layers: int = 4,
    num_layers: int = 32,
    num_heads: int = 32,
    hidden_size: int = 4096,
    context_tokens: int = 8192,
    params_b: float = 7.0,
    system_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bits = max(1, min(bits, 4))
    num_layers = max(1, num_layers)
    num_heads = max(1, num_heads)
    hidden_size = max(num_heads, hidden_size)
    context_tokens = max(256, context_tokens)
    fp16_layer_count = min(num_layers, fp16_layers * 2)
    compressed_layer_count = max(0, num_layers - fp16_layer_count)

    bytes_per_fp16_layer = context_tokens * hidden_size * 2 * 2
    baseline_bytes = num_layers * bytes_per_fp16_layer

    quantized_vector_bytes = hidden_size * (bits / 8.0)
    norm_overhead_bytes = num_heads * 4
    quantized_layer_bytes = context_tokens * 2 * (quantized_vector_bytes + norm_overhead_bytes)
    optimized_bytes = fp16_layer_count * bytes_per_fp16_layer + compressed_layer_count * quantized_layer_bytes

    compression_ratio = baseline_bytes / optimized_bytes if optimized_bytes else 1.0

    speed_map = {1: 0.45, 2: 0.53, 3: 0.59, 4: 0.68}
    speed_ratio = speed_map[bits] + min(fp16_layers, 8) * 0.012
    speed_ratio = min(speed_ratio, 0.92)

    preview_system = system_stats or _build_system_snapshot()
    baseline_tok_s = _estimate_baseline_tok_s(preview_system)
    model_scale = min(1.55, max(0.2, (7.0 / max(params_b, 1.0)) ** 0.38))
    baseline_tok_s *= model_scale
    estimated_tok_s = round(baseline_tok_s * speed_ratio, 1)

    quality_percent = min(99.5, 92.0 + bits * 1.4 + min(fp16_layers, 8) * 0.35)
    # Estimate on-disk size: BF16 models are ~2 bytes/param, 4-bit quant ~0.5 bytes/param.
    # Use a middle estimate since we don't know the on-disk quantization here.
    disk_size_gb = round(params_b * 1.2, 1)

    summary = (
        f"{bits}-bit cache ({fp16_layers}+{fp16_layers}) lowers cache use to "
        f"{_bytes_to_gb(optimized_bytes):.1f} GB from {_bytes_to_gb(baseline_bytes):.1f} GB, "
        f"about {compression_ratio:.1f}x smaller, with an estimated {estimated_tok_s:.1f} tok/s on this machine."
    )

    return {
        "bits": bits,
        "fp16Layers": fp16_layers,
        "numLayers": num_layers,
        "numHeads": num_heads,
        "hiddenSize": hidden_size,
        "contextTokens": context_tokens,
        "paramsB": params_b,
        "baselineCacheGb": _bytes_to_gb(baseline_bytes),
        "optimizedCacheGb": _bytes_to_gb(optimized_bytes),
        "compressionRatio": round(compression_ratio, 1),
        "estimatedTokS": estimated_tok_s,
        "speedRatio": round(speed_ratio, 2),
        "qualityPercent": round(quality_percent, 1),
        "diskSizeGb": disk_size_gb,
        "summary": summary,
    }


class ChaosEngineState:
    def __init__(
        self,
        *,
        system_snapshot_provider=_build_system_snapshot,
        library_provider=None,
        server_port: int = DEFAULT_PORT,
        settings_path: Path = SETTINGS_PATH,
        benchmarks_path: Path = BENCHMARKS_PATH,
    ) -> None:
        self._lock = RLock()
        self._system_snapshot_provider = system_snapshot_provider
        self._library_provider = library_provider
        self.server_port = server_port
        self._settings_path = settings_path
        self._benchmarks_path = benchmarks_path
        self.settings = _load_settings(self._settings_path)
        self._library_cache: tuple[float, list[dict[str, Any]]] | None = None
        self.runtime = RuntimeController()
        self.image_runtime = ImageRuntimeManager()
        self._chat_sessions_path = CHAT_SESSIONS_PATH
        loaded_sessions = _load_chat_sessions(self._chat_sessions_path)
        self.chat_sessions = loaded_sessions if loaded_sessions else _seed_chat_sessions()
        self.benchmark_runs = _load_benchmark_runs(self._benchmarks_path)
        self.logs: deque[dict[str, Any]] = deque(maxlen=120)
        self._log_subscribers: list = []
        self.activity: deque[dict[str, Any]] = deque(maxlen=60)
        self.requests_served = 0
        self.active_requests = 0
        self._loading_state: dict[str, Any] | None = None
        self._downloads: dict[str, dict[str, Any]] = {}
        self._download_cancel: dict[str, bool] = {}
        self._download_processes: dict[str, subprocess.Popen[str]] = {}
        self._download_tokens: dict[str, str] = {}
        self._bootstrap()

    def _launch_preferences(self) -> dict[str, Any]:
        return dict(self.settings["launchPreferences"])

    def _library(self, *, force: bool = False) -> list[dict[str, Any]]:
        if self._library_provider is not None:
            return self._library_provider()
        if not force and self._library_cache is not None:
            cached_at, cached_items = self._library_cache
            if (time.time() - cached_at) < 30.0:
                return cached_items
        library = _discover_local_models(self.settings["modelDirectories"])
        self._library_cache = (time.time(), library)
        return library

    def _settings_payload(self, library: list[dict[str, Any]]) -> dict[str, Any]:
        model_counts: dict[str, int] = {}
        for item in library:
            directory_id = item.get("directoryId")
            if not directory_id:
                continue
            model_counts[directory_id] = model_counts.get(directory_id, 0) + 1

        directories: list[dict[str, Any]] = []
        for directory in self.settings["modelDirectories"]:
            expanded = Path(os.path.expanduser(str(directory.get("path") or ""))).expanduser()
            directories.append(
                {
                    **directory,
                    "exists": expanded.exists(),
                    "modelCount": model_counts.get(directory["id"], 0),
                }
            )

        # Mask API keys when returning to the frontend
        remote_providers = self.settings.get("remoteProviders") or []
        masked_providers = []
        for p in remote_providers:
            api_key = p.get("apiKey", "")
            masked_providers.append({
                "id": p.get("id"),
                "label": p.get("label"),
                "apiBase": p.get("apiBase"),
                "model": p.get("model"),
                "hasApiKey": bool(api_key),
                "apiKeyMasked": ("•" * 8 + api_key[-4:]) if len(api_key) > 4 else "",
            })

        hf_token_value = str(self.settings.get("huggingFaceToken") or "")
        if len(hf_token_value) > 4:
            hf_token_masked = "•" * 8 + hf_token_value[-4:]
        else:
            hf_token_masked = ""

        return {
            "modelDirectories": directories,
            "preferredServerPort": self.settings["preferredServerPort"],
            "allowRemoteConnections": bool(self.settings.get("allowRemoteConnections", False)),
            "autoStartServer": bool(self.settings.get("autoStartServer", False)),
            "launchPreferences": self._launch_preferences(),
            "remoteProviders": masked_providers,
            "huggingFaceToken": hf_token_masked,
            "hasHuggingFaceToken": bool(hf_token_value),
            "dataDirectory": str(DATA_LOCATION.data_dir),
        }

    def _bootstrap(self) -> None:
        system = self._system_snapshot_provider()
        library = self._library(force=True)
        recommendation = _best_fit_recommendation(system)
        self.add_log("chaosengine", "info", f"Workspace booted in {system['backendLabel']} mode.")
        self.add_log("chaosengine", "info", f"ChaosEngine v{app_version} detected.")
        self.add_log("library", "info", f"Discovered {len(library)} local model entries.")
        self.add_activity("Hardware profile refreshed", recommendation["title"])
        self.add_activity("Library scan completed", f"{len(library)} local entries found across configured model directories.")
        self.add_activity(
            "Backend readiness",
            " / ".join(
                [
                    f"MLX installed: {'yes' if system['mlxAvailable'] else 'no'}",
                    f"mlx-lm installed: {'yes' if system['mlxLmAvailable'] else 'no'}",
                    f"MLX usable: {'yes' if system.get('mlxUsable') else 'no'}",
                    f"GGUF runtime: {'yes' if system.get('ggufAvailable') else 'no'}",
                ]
            ),
        )

    @staticmethod
    def _time_label() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _relative_label() -> str:
        return time.strftime("%H:%M")

    def add_log(self, source: str, level: str, message: str) -> None:
        import queue as _queue_mod
        entry = {
            "ts": self._time_label(),
            "source": source,
            "level": level,
            "message": message,
        }
        self.logs.appendleft(entry)
        for q in self._log_subscribers:
            try:
                q.put_nowait(entry)
            except _queue_mod.Full:
                pass

    def subscribe_logs(self):
        import queue as _queue_mod
        q = _queue_mod.Queue(maxsize=200)
        self._log_subscribers.append(q)
        return q

    def unsubscribe_logs(self, q) -> None:
        try:
            self._log_subscribers.remove(q)
        except ValueError:
            pass

    def add_activity(self, title: str, detail: str) -> None:
        self.activity.appendleft(
            {
                "time": "Now",
                "title": title,
                "detail": detail,
            }
        )

    def _cache_strategy_label(self, bits: int, fp16_layers: int) -> str:
        if bits and bits < 16:
            return f"Native {bits}-bit {fp16_layers}+{fp16_layers}"
        return "Native f16 cache"

    @staticmethod
    def _native_cache_label() -> str:
        return "Native f16 cache"

    def _cache_label(self, *, cache_strategy: str, bits: int, fp16_layers: int) -> str:
        _ = cache_strategy  # reserved for future strategy dispatch
        return self._cache_strategy_label(bits, fp16_layers)

    def _assistant_metrics_payload(self, result: Any) -> dict[str, Any]:
        loaded = self.runtime.loaded_model
        return {
            **result.to_metrics(),
            "model": loaded.name if loaded else None,
            "modelRef": loaded.ref if loaded else None,
            "backend": loaded.backend if loaded else None,
            "engineLabel": self.runtime.engine.engine_label,
            "cacheLabel": self._cache_label(
                cache_strategy=str(loaded.cacheStrategy) if loaded else "native",
                bits=int(loaded.cacheBits) if loaded else 0,
                fp16_layers=int(loaded.fp16Layers) if loaded else 0,
            ),
            "contextTokens": loaded.contextTokens if loaded else None,
            "generatedAt": self._time_label(),
        }

    def _should_reload_for_profile(
        self,
        *,
        model_ref: str | None,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        cache_strategy: str,
        fit_model_in_memory: bool,
        context_tokens: int,
    ) -> bool:
        if model_ref and (
            self.runtime.loaded_model is None
            or model_ref not in {self.runtime.loaded_model.ref, self.runtime.loaded_model.runtimeTarget}
        ):
            return True

        if self.runtime.loaded_model is None:
            return True

        loaded_model = self.runtime.loaded_model
        return any(
            [
                loaded_model.cacheBits != cache_bits,
                loaded_model.fp16Layers != fp16_layers,
                loaded_model.fusedAttention != fused_attention,
                loaded_model.cacheStrategy != cache_strategy,
                loaded_model.fitModelInMemory != fit_model_in_memory,
                loaded_model.contextTokens != context_tokens,
            ]
        )

    def _append_benchmark_run(self, run: dict[str, Any]) -> None:
        self.benchmark_runs = [run, *[item for item in self.benchmark_runs if item["id"] != run["id"]]][:MAX_BENCHMARK_RUNS]
        _save_benchmark_runs(self.benchmark_runs, self._benchmarks_path)

    def _find_catalog_entry(self, model_ref: str) -> dict[str, Any] | None:
        for entry in CATALOG:
            if (
                entry["id"] == model_ref
                or entry["name"] == model_ref
                or entry["repo"] == model_ref
                or entry["link"] == model_ref
            ):
                return entry
        return None

    def _find_library_entry(self, path: str | None, model_ref: str | None) -> dict[str, Any] | None:
        if path is None and model_ref is None:
            return None
        for entry in self._library():
            if path and entry["path"] == path:
                return entry
            if model_ref and entry["name"] == model_ref:
                return entry
        return None

    def _resolve_model_target(
        self,
        *,
        model_ref: str | None,
        path: str | None,
        backend: str,
    ) -> tuple[str | None, str]:
        resolved_backend = backend
        runtime_target = path
        explicit_gguf_path = bool(path and path.lower().endswith(".gguf"))
        catalog_entry = self._find_catalog_entry(model_ref) if model_ref else None
        library_entry = self._find_library_entry(path, model_ref)

        if explicit_gguf_path:
            runtime_target = path
            if backend == "auto":
                resolved_backend = "llama.cpp"
            return runtime_target, resolved_backend

        if catalog_entry is not None:
            runtime_target = _hf_repo_from_link(catalog_entry.get("link")) or runtime_target or model_ref
            if backend == "auto":
                resolved_backend = "llama.cpp" if catalog_entry.get("format") == "GGUF" else "mlx"
        elif library_entry is not None:
            lib_format = library_entry.get("format", "")
            lib_name = library_entry.get("name", "")
            lib_path = library_entry.get("path", "")
            lib_source_kind = library_entry.get("sourceKind", "")
            is_gguf = lib_format == "GGUF" or "gguf" in lib_name.lower() or "gguf" in lib_path.lower()
            if backend == "auto":
                resolved_backend = "llama.cpp" if is_gguf else "mlx"
            if lib_source_kind == "HF cache":
                runtime_target = library_entry["path"] if is_gguf else library_entry["name"]
            else:
                runtime_target = runtime_target or library_entry["path"]
        elif path and path.lower().endswith(".gguf") and backend == "auto":
            resolved_backend = "llama.cpp"

        # Last-resort GGUF detection: if the runtime_target / model_ref / path
        # contains "gguf" anywhere (e.g. an HF repo named "...-GGUF" or an HF
        # cache directory `models--owner--name-GGUF/snapshots/...`), force the
        # llama.cpp backend. Without this, GGUF-only repos fall through to the
        # MLX path and explode on a missing config.json.
        if resolved_backend in {"auto", "mlx"}:
            haystack = " ".join(
                str(value).lower()
                for value in (runtime_target, model_ref, path)
                if value
            )
            if "gguf" in haystack:
                resolved_backend = "llama.cpp"

        return runtime_target or model_ref, resolved_backend

    def _default_session_model(self) -> dict[str, Any]:
        model_info = self.runtime.loaded_model
        launch_preferences = self._launch_preferences()
        if model_info is not None:
            return {
                "model": model_info.name,
                "modelRef": model_info.ref,
                "modelSource": model_info.source,
                "modelPath": model_info.path,
                "modelBackend": model_info.backend,
                "cacheLabel": self._cache_strategy_label(model_info.cacheBits, model_info.fp16Layers),
            }

        default_variant = _default_chat_variant()
        return {
            "model": default_variant["name"],
            "modelRef": default_variant["id"],
            "modelSource": "catalog",
            "modelPath": None,
            "modelBackend": default_variant.get("backend", "auto"),
            "cacheLabel": self._cache_strategy_label(
                launch_preferences["cacheBits"],
                launch_preferences["fp16Layers"],
            ),
        }

    def _promote_session(self, session: dict[str, Any]) -> None:
        self.chat_sessions = [session, *[item for item in self.chat_sessions if item["id"] != session["id"]]]

    def _persist_sessions(self) -> None:
        try:
            _save_chat_sessions(self.chat_sessions, self._chat_sessions_path)
        except OSError:
            pass  # Non-critical — don't crash if disk is full

    def _ensure_session(self, session_id: str | None = None, title: str | None = None) -> dict[str, Any]:
        if session_id:
            for session in self.chat_sessions:
                if session["id"] == session_id:
                    return session

        model_defaults = self._default_session_model()
        session = {
            "id": session_id or f"session-{uuid.uuid4().hex[:8]}",
            "title": title or "New chat",
            "updatedAt": self._time_label(),
            "pinned": False,
            **model_defaults,
            "messages": [],
        }
        self.chat_sessions.insert(0, session)
        self.add_activity("Chat session created", session["title"])
        self._persist_sessions()
        return session

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        with self._lock:
            session = self._ensure_session(title=title)
            return session

    def update_session(self, session_id: str, request: UpdateSessionRequest) -> dict[str, Any]:
        with self._lock:
            session = self._ensure_session(session_id=session_id)
            if request.title is not None and request.title.strip():
                session["title"] = request.title.strip()
            if request.model is not None:
                session["model"] = request.model
            if request.modelRef is not None:
                session["modelRef"] = request.modelRef
            if request.modelSource is not None:
                session["modelSource"] = request.modelSource
            if request.modelPath is not None:
                session["modelPath"] = request.modelPath
            if request.modelBackend is not None:
                session["modelBackend"] = request.modelBackend
            if request.pinned is not None:
                session["pinned"] = request.pinned
            session["updatedAt"] = self._time_label()
            self._promote_session(session)
            self.add_activity("Thread updated", session["title"])
            self._persist_sessions()
            return session

    def update_settings(self, request: UpdateSettingsRequest) -> dict[str, Any]:
        """Returns ``{"settings": ..., "restartRequired"?: bool, "migrationSummary"?: dict}``."""
        with self._lock:
            next_settings = _default_settings()
            next_settings["modelDirectories"] = [dict(entry) for entry in self.settings["modelDirectories"]]
            next_settings["preferredServerPort"] = self.settings["preferredServerPort"]
            next_settings["allowRemoteConnections"] = bool(self.settings.get("allowRemoteConnections", False))
            next_settings["launchPreferences"] = self._launch_preferences()
            next_settings["remoteProviders"] = list(self.settings.get("remoteProviders") or [])
            next_settings["huggingFaceToken"] = str(self.settings.get("huggingFaceToken") or "")

            if request.modelDirectories is not None:
                next_settings["modelDirectories"] = _normalize_model_directories(
                    [entry.model_dump() for entry in request.modelDirectories]
                )
            if request.preferredServerPort is not None:
                next_settings["preferredServerPort"] = request.preferredServerPort
            if request.allowRemoteConnections is not None:
                next_settings["allowRemoteConnections"] = request.allowRemoteConnections
            if request.autoStartServer is not None:
                next_settings["autoStartServer"] = request.autoStartServer
            if request.launchPreferences is not None:
                next_settings["launchPreferences"] = _normalize_launch_preferences(request.launchPreferences.model_dump())
            if request.remoteProviders is not None:
                # Validate URLs (HTTPS or localhost only) and preserve existing keys when not provided
                existing_by_id = {p.get("id"): p for p in (self.settings.get("remoteProviders") or [])}
                normalized = []
                for provider in request.remoteProviders:
                    api_base = provider.apiBase.strip()
                    if not (api_base.startswith("https://") or api_base.startswith("http://127.0.0.1") or api_base.startswith("http://localhost")):
                        raise HTTPException(status_code=400, detail=f"Provider {provider.id} must use HTTPS (or localhost).")
                    api_key = provider.apiKey
                    # If empty key submitted, preserve the existing one
                    if not api_key and provider.id in existing_by_id:
                        api_key = existing_by_id[provider.id].get("apiKey", "")
                    normalized.append({
                        "id": provider.id,
                        "label": provider.label,
                        "apiBase": api_base,
                        "apiKey": api_key,
                        "model": provider.model,
                    })
                next_settings["remoteProviders"] = normalized

            if request.huggingFaceToken is not None:
                # Empty string clears the token; otherwise update and propagate to env
                token_value = request.huggingFaceToken.strip()
                next_settings["huggingFaceToken"] = token_value
                if token_value:
                    os.environ["HF_TOKEN"] = token_value
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = token_value
                else:
                    os.environ.pop("HF_TOKEN", None)
                    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

            # Handle data directory change BEFORE persisting settings: we need
            # to migrate files to the new location, write the bootstrap pointer,
            # and signal restart_required. Settings keep saving to the OLD path
            # until the next process restart picks up the new bootstrap value.
            # That's intentional — it avoids partial-state bugs where some
            # writers see the new dir and others see the old. The next migration
            # (or restart) sweeps any leftover writes forward.
            data_migration: dict[str, Any] | None = None
            restart_required_for_data_dir = False
            if request.dataDirectory is not None:
                raw_dir = request.dataDirectory.strip()
                if raw_dir:
                    if not (raw_dir.startswith("/") or raw_dir.startswith("~")):
                        raise HTTPException(
                            status_code=400,
                            detail="dataDirectory must be an absolute path or start with ~.",
                        )
                    new_dir = Path(os.path.expanduser(raw_dir)).resolve()
                    if new_dir != DATA_LOCATION.data_dir:
                        try:
                            data_migration = _migrate_data_directory(
                                DATA_LOCATION.data_dir, new_dir
                            )
                            _save_data_location(new_dir)
                            restart_required_for_data_dir = True
                        except RuntimeError as exc:
                            raise HTTPException(status_code=400, detail=str(exc)) from exc

            _save_settings(next_settings, self._settings_path)
            self.settings = next_settings
            self._library_cache = None
            library = self._library(force=True)

            self.add_log(
                "settings",
                "info",
                f"Saved settings with {len(self.settings['modelDirectories'])} model directories, preferred API port {self.settings['preferredServerPort']}, and remote access {'enabled' if self.settings['allowRemoteConnections'] else 'disabled'}.",
            )
            if self.settings["preferredServerPort"] != self.server_port:
                self.add_log(
                    "server",
                    "info",
                    f"Preferred API port changed to {self.settings['preferredServerPort']}. Restart the API service to apply it.",
                )
            if bool(self.settings.get("allowRemoteConnections", False)) != (DEFAULT_HOST != "127.0.0.1"):
                self.add_log(
                    "server",
                    "info",
                    "Remote connection setting changed. Restart the API service to apply the new bind mode.",
                )
            self.add_activity(
                "Settings updated",
                f"{len(library)} models discovered across {len(self.settings['modelDirectories'])} configured directories.",
            )
            payload = self._settings_payload(library)
            response: dict[str, Any] = {"settings": payload}
            if restart_required_for_data_dir:
                response["restartRequired"] = True
            if data_migration is not None:
                response["migrationSummary"] = data_migration
            return response

    def _conversion_details(
        self,
        *,
        request: ConvertModelRequest,
        conversion: dict[str, Any],
    ) -> dict[str, Any]:
        library_entry = self._find_library_entry(request.path, request.modelRef)
        catalog_entry = self._find_catalog_entry(request.modelRef or conversion.get("hfRepo") or "")
        params_b = float(catalog_entry.get("paramsB")) if catalog_entry and catalog_entry.get("paramsB") is not None else None
        launch_preferences = self._launch_preferences()

        preview = (
            compute_cache_preview(
                bits=launch_preferences["cacheBits"],
                fp16_layers=launch_preferences["fp16Layers"],
                context_tokens=launch_preferences["contextTokens"],
                params_b=params_b,
                system_stats=self._system_snapshot_provider(),
            )
            if params_b is not None
            else None
        )

        gguf_metadata = conversion.get("ggufMetadata") or {}
        context_length = gguf_metadata.get("contextLength")
        context_window = (
            _context_label(int(context_length))
            if context_length
            else (catalog_entry.get("contextWindow") if catalog_entry is not None else None)
        )

        return {
            **conversion,
            "sourceFormat": library_entry.get("format") if library_entry is not None else (catalog_entry.get("format") if catalog_entry is not None else None),
            "sourceSizeGb": conversion.get("sourceSizeGb") or (library_entry.get("sizeGb") if library_entry is not None else None),
            "paramsB": params_b,
            "contextWindow": context_window,
            "architecture": gguf_metadata.get("architecture") or gguf_metadata.get("name"),
            "estimatedTokS": preview["estimatedTokS"] if preview is not None else None,
            "baselineCacheGb": preview["baselineCacheGb"] if preview is not None else None,
            "optimizedCacheGb": preview["optimizedCacheGb"] if preview is not None else None,
            "compressionRatio": preview["compressionRatio"] if preview is not None else None,
            "qualityPercent": preview["qualityPercent"] if preview is not None else None,
        }

    def run_benchmark(self, request: BenchmarkRunRequest) -> dict[str, Any]:
        with self._lock:
            default_variant = _default_chat_variant()
            effective_model_ref = (
                request.modelRef
                or (self.runtime.loaded_model.ref if self.runtime.loaded_model is not None else None)
                or default_variant["id"]
            )
            catalog_entry = self._find_catalog_entry(effective_model_ref)
            library_entry = self._find_library_entry(request.path, effective_model_ref)
            model_name = request.modelName
            if model_name is None and library_entry is not None:
                model_name = str(library_entry.get("name") or "")
            if model_name is None and catalog_entry is not None:
                model_name = str(catalog_entry.get("name") or "")
            if model_name is None:
                model_name = str(effective_model_ref or default_variant["name"])

            if library_entry is not None and library_entry.get("broken"):
                reason = library_entry.get("brokenReason") or "incomplete or corrupt"
                raise RuntimeError(
                    f"Cannot benchmark '{library_entry.get('name') or effective_model_ref}': {reason}."
                )
            effective_source = request.source or ("library" if library_entry is not None else "catalog")
            effective_path = request.path if request.path is not None else (library_entry.get("path") if library_entry is not None else None)
            effective_backend = request.backend or (
                "llama.cpp"
                if (library_entry and library_entry.get("format") == "GGUF") or (catalog_entry and catalog_entry.get("format") == "GGUF")
                else "mlx"
            )

        load_seconds = 0.0
        if self._should_reload_for_profile(
            model_ref=effective_model_ref,
            cache_bits=request.cacheBits,
            fp16_layers=request.fp16Layers,
            fused_attention=request.fusedAttention,
            cache_strategy=request.cacheStrategy,
            fit_model_in_memory=request.fitModelInMemory,
            context_tokens=request.contextTokens,
        ):
            load_started = time.perf_counter()
            self.load_model(
                LoadModelRequest(
                    modelRef=str(effective_model_ref),
                    modelName=model_name,
                    source=effective_source,
                    backend=effective_backend,
                    path=effective_path,
                    cacheStrategy=request.cacheStrategy,
                    cacheBits=request.cacheBits,
                    fp16Layers=request.fp16Layers,
                    fusedAttention=request.fusedAttention,
                    fitModelInMemory=request.fitModelInMemory,
                    contextTokens=request.contextTokens,
                )
            )
            load_seconds = round(time.perf_counter() - load_started, 2)

        # Common fields for all benchmark modes
        with self._lock:
            params_b = float(catalog_entry.get("paramsB")) if catalog_entry and catalog_entry.get("paramsB") is not None else 7.0
            preview = compute_cache_preview(
                bits=request.cacheBits if request.cacheBits else 4,
                fp16_layers=request.fp16Layers,
                context_tokens=request.contextTokens,
                params_b=params_b,
                system_stats=self._system_snapshot_provider(),
            )
            use_compressed = request.cacheBits > 0
            cache_gb = preview["optimizedCacheGb"] if use_compressed else preview["baselineCacheGb"]
            baseline_cache_gb = preview["baselineCacheGb"]
            compression = round(baseline_cache_gb / cache_gb, 1) if use_compressed and cache_gb else 1.0
            quality = int(round(preview["qualityPercent"])) if use_compressed else 100
            cache_label = self._cache_label(
                cache_strategy=request.cacheStrategy,
                bits=request.cacheBits,
                fp16_layers=request.fp16Layers,
            )

        base_run: dict[str, Any] = {
            "id": f"bench-{uuid.uuid4().hex[:8]}",
            "mode": request.mode,
            "model": model_name,
            "modelRef": effective_model_ref,
            "backend": self.runtime.loaded_model.backend if self.runtime.loaded_model else effective_backend,
            "engineLabel": self.runtime.engine.engine_label,
            "source": effective_source,
            "measuredAt": self._time_label(),
            "bits": request.cacheBits if request.cacheBits > 0 else 16,
            "fp16Layers": request.fp16Layers,
            "cacheStrategy": request.cacheStrategy,
            "cacheLabel": cache_label,
            "cacheGb": cache_gb,
            "baselineCacheGb": baseline_cache_gb,
            "compression": compression,
            "contextTokens": request.contextTokens,
            "maxTokens": request.maxTokens,
            "loadSeconds": load_seconds,
        }

        if request.mode == "perplexity":
            eval_result = self.runtime.engine.eval_perplexity(
                dataset=request.perplexityDataset,
                num_samples=request.perplexityNumSamples,
                seq_length=request.perplexitySeqLength,
                batch_size=request.perplexityBatchSize,
            )
            run = {
                **base_run,
                "label": request.label or f"{model_name} / Perplexity / {request.perplexityDataset}",
                "perplexity": eval_result["perplexity"],
                "perplexityStdError": eval_result["standardError"],
                "perplexityDataset": eval_result["dataset"],
                "perplexityNumSamples": eval_result["numSamples"],
                "evalTokensPerSecond": eval_result["evalTokensPerSecond"],
                "evalSeconds": eval_result["evalSeconds"],
                "quality": quality,
                "tokS": eval_result["evalTokensPerSecond"],
                "responseSeconds": eval_result["evalSeconds"],
                "totalSeconds": round(load_seconds + eval_result["evalSeconds"], 2),
                "promptTokens": 0,
                "completionTokens": 0,
                "totalTokens": 0,
                "notes": f"Perplexity: {eval_result['perplexity']:.2f} ± {eval_result['standardError']:.2f} on {eval_result['dataset']} ({eval_result['numSamples']} samples)",
            }
        elif request.mode == "task_accuracy":
            eval_result = self.runtime.engine.eval_task_accuracy(
                task_name=request.taskName,
                limit=request.taskLimit,
                num_shots=request.taskNumShots,
            )
            accuracy_pct = round(eval_result["accuracy"] * 100, 1)
            run = {
                **base_run,
                "label": request.label or f"{model_name} / {request.taskName.upper()} / {eval_result['correct']}/{eval_result['total']}",
                "taskName": eval_result["taskName"],
                "taskAccuracy": eval_result["accuracy"],
                "taskCorrect": eval_result["correct"],
                "taskTotal": eval_result["total"],
                "taskNumShots": eval_result["numShots"],
                "evalSeconds": eval_result["evalSeconds"],
                "quality": quality,
                "tokS": 0,
                "responseSeconds": eval_result["evalSeconds"],
                "totalSeconds": round(load_seconds + eval_result["evalSeconds"], 2),
                "promptTokens": 0,
                "completionTokens": 0,
                "totalTokens": 0,
                "notes": f"{request.taskName.upper()}: {accuracy_pct}% ({eval_result['correct']}/{eval_result['total']}) {eval_result['numShots']}-shot",
            }
        else:
            # Throughput mode (default)
            prompt = request.prompt or (
                "Summarize the practical trade-offs of this runtime profile for a local desktop user in six short bullets."
            )
            result = self.runtime.generate(
                prompt=prompt,
                history=[],
                system_prompt="Return a concise but complete answer so ChaosEngineAI can benchmark response speed consistently.",
                max_tokens=request.maxTokens,
                temperature=request.temperature,
            )
            run = {
                **base_run,
                "label": request.label
                or _benchmark_label(
                    model_name,
                    cache_strategy=request.cacheStrategy,
                    bits=request.cacheBits,
                    fp16_layers=request.fp16Layers,
                    context_tokens=request.contextTokens,
                ),
                "tokS": round(result.tokS, 1),
                "quality": quality,
                "responseSeconds": round(result.responseSeconds, 2),
                "totalSeconds": round(load_seconds + result.responseSeconds, 2),
                "promptTokens": result.promptTokens,
                "completionTokens": result.completionTokens,
                "totalTokens": result.totalTokens,
                "notes": result.runtimeNote,
            }

        with self._lock:
            self._append_benchmark_run(run)
            mode_label = {"perplexity": "Perplexity", "task_accuracy": "Task accuracy"}.get(request.mode, "Throughput")
            self.add_log("benchmark", "info", f"{mode_label} benchmark completed for {model_name}: {run.get('notes', '')}")
            self.add_activity("Benchmark completed", run["label"])
            return {
                "result": run,
                "benchmarks": self.benchmark_runs,
                "runtime": self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served),
            }

    def load_model(self, request: LoadModelRequest) -> dict[str, Any]:
        # Resolve metadata under the lock, then release it so the slow
        # model load doesn't block workspace polling and health checks.
        with self._lock:
            catalog_entry = self._find_catalog_entry(request.modelRef)
            library_entry = self._find_library_entry(request.path, request.modelRef)
            # Clamp requested context tokens to the model's detected hard limit, if known.
            detected_max: int | None = None
            if library_entry is not None:
                detected_max = library_entry.get("maxContext")
            if detected_max is None and catalog_entry is not None:
                detected_max = _parse_context_label(catalog_entry.get("contextWindow"))
            if detected_max is not None and request.contextTokens > detected_max:
                self.add_log(
                    "runtime",
                    "warning",
                    f"Requested context {request.contextTokens} exceeds model max {detected_max}; clamping.",
                )
                try:
                    request.contextTokens = int(detected_max)
                except Exception:
                    pass
            model_name = request.modelName
            if model_name is None and catalog_entry is not None:
                model_name = catalog_entry["name"]
            if model_name is None and library_entry is not None:
                model_name = library_entry["name"]
            runtime_target, resolved_backend = self._resolve_model_target(
                model_ref=request.modelRef,
                path=request.path,
                backend=request.backend,
            )
            display_name = model_name or request.modelRef
            self._loading_state = {
                "modelName": display_name,
                "stage": "loading",
                "startedAt": time.time(),
                "progress": None,
                "progressPercent": None,
                "progressPhase": None,
                "progressMessage": None,
                "recentLogLines": [],
            }
            self.add_log("runtime", "info", f"Loading {display_name}...")

        def _on_load_progress(prog: dict[str, Any]) -> None:
            try:
                with self._lock:
                    if self._loading_state is None:
                        return
                    percent = prog.get("percent")
                    phase = prog.get("phase")
                    message = prog.get("message")
                    self._loading_state["progressPercent"] = percent
                    self._loading_state["progressPhase"] = phase
                    self._loading_state["progressMessage"] = message
                    self._loading_state["progress"] = percent
                    if message or phase:
                        line = f"[{phase}] {message}" if phase and message else str(message or phase)
                        tail = list(self._loading_state.get("recentLogLines") or [])
                        tail.append(line)
                        if len(tail) > 5:
                            tail = tail[-5:]
                        self._loading_state["recentLogLines"] = tail
            except Exception:
                pass

        # Actual model load — potentially slow (download + init). Run
        # WITHOUT the state lock so other endpoints remain responsive.
        try:
            loaded = self.runtime.load_model(
                model_ref=request.modelRef,
                model_name=model_name,
                source=request.source,
                backend=resolved_backend,
                path=request.path,
                runtime_target=runtime_target,
                cache_strategy=request.cacheStrategy,
                cache_bits=request.cacheBits,
                fp16_layers=request.fp16Layers,
                fused_attention=request.fusedAttention,
                fit_model_in_memory=request.fitModelInMemory,
                context_tokens=request.contextTokens,
                progress_callback=_on_load_progress,
            )
        except Exception:
            with self._lock:
                self._loading_state = None
            raise

        with self._lock:
            self._loading_state = None
            loaded_cache_label = self._cache_strategy_label(loaded.cacheBits, loaded.fp16Layers)
            self.add_log("runtime", "info", f"Model loaded: {loaded.name} via {loaded.engine}.")
            self.add_activity("Model loaded", f"{loaded.name} / {loaded_cache_label}")
            return self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served)

    def unload_model(self, ref: str | None = None) -> dict[str, Any]:
        with self._lock:
            if ref:
                # Try warm-pool unload first; if that ref is actually the active
                # model, fall through to a full unload.
                if self.runtime.loaded_model and ref in {
                    self.runtime.loaded_model.ref,
                    self.runtime.loaded_model.runtimeTarget,
                    self.runtime.loaded_model.path,
                    self.runtime.loaded_model.name,
                }:
                    name = self.runtime.loaded_model.name
                    self.runtime.unload_model()
                    self.add_log("runtime", "info", f"Model unloaded: {name}.")
                    self.add_activity("Model unloaded", name)
                else:
                    unloaded = self.runtime.unload_warm_model_by_ref(ref)
                    if unloaded:
                        self.add_log("runtime", "info", f"Warm model unloaded: {ref}.")
                        self.add_activity("Warm model unloaded", ref)
                    else:
                        self.add_log("runtime", "info", f"Unload no-op: {ref} not found.")
            else:
                name = self.runtime.loaded_model.name if self.runtime.loaded_model else "No model"
                self.runtime.unload_model()
                self.add_log("runtime", "info", f"Model unloaded: {name}.")
                self.add_activity("Model unloaded", name)
            return self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served)

    def convert_model(self, request: ConvertModelRequest) -> dict[str, Any]:
        with self._lock:
            runtime_target, _ = self._resolve_model_target(
                model_ref=request.modelRef,
                path=request.path,
                backend="auto",
            )
            conversion = self.runtime.convert_model(
                source_ref=runtime_target if request.path is None else request.modelRef,
                source_path=request.path,
                output_path=request.outputPath,
                hf_repo=request.hfRepo,
                quantize=request.quantize,
                q_bits=request.qBits,
                q_group_size=request.qGroupSize,
                dtype=request.dtype,
            )
            conversion = self._conversion_details(request=request, conversion=conversion)
            self.add_log(
                "conversion",
                "info",
                f"Converted {conversion['sourceLabel']} to MLX at {conversion['outputPath']}.",
            )
            self.add_activity("Model converted", f"{conversion['sourceLabel']} -> {Path(conversion['outputPath']).name}")
            return {
                "conversion": conversion,
                "library": self._library(force=True),
                "runtime": self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served),
            }

    def reveal_model_path(self, path: str) -> dict[str, Any]:
        with self._lock:
            target = Path(path).expanduser()
            _reveal_path_in_file_manager(target)
            resolved = str(target.resolve())
            self.add_log("library", "info", f"Revealed model path: {resolved}.")
            return {"revealed": resolved}

    def delete_model_path(self, path: str) -> dict[str, Any]:
        """Delete a local model file or directory on disk.

        Safety rails:
        - Path must resolve inside one of the configured model directories.
        - Cannot target a configured model directory itself (only children).
        - If the runtime is currently serving from this path, unload first.
        - Never follows symlinks outside the allowed roots.
        """
        with self._lock:
            target = Path(path).expanduser()
            try:
                resolved = target.resolve(strict=True)
            except (OSError, RuntimeError):
                raise HTTPException(status_code=404, detail=f"Path not found: {path}")

            # Whitelist: must be a strict child of a configured model dir.
            allowed = False
            for directory in self.settings.get("modelDirectories", []):
                if not directory.get("enabled", True):
                    continue
                root_raw = str(directory.get("path") or "").strip()
                if not root_raw:
                    continue
                try:
                    root = Path(os.path.expanduser(root_raw)).resolve()
                except (OSError, RuntimeError):
                    continue
                if resolved == root:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Refusing to delete a configured model directory. "
                            "Only files/subdirectories inside it may be removed."
                        ),
                    )
                try:
                    resolved.relative_to(root)
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Refusing to delete {resolved}: not inside any "
                        f"configured model directory."
                    ),
                )

            # Unload if the runtime is currently using this path.
            try:
                loaded = getattr(self.runtime, "loaded_model", None)
                if loaded and getattr(loaded, "path", None):
                    loaded_resolved = Path(str(loaded.path)).expanduser().resolve()
                    if loaded_resolved == resolved or loaded_resolved.is_relative_to(resolved):
                        self.runtime.unload_model()
            except (OSError, RuntimeError, AttributeError):
                pass

            # Perform the delete.
            try:
                if resolved.is_dir() and not resolved.is_symlink():
                    import shutil as _shutil
                    _shutil.rmtree(resolved)
                else:
                    resolved.unlink()
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete {resolved}: {exc}",
                )

            self.add_log("library", "info", f"Deleted model at {resolved}.")
            return {
                "deleted": str(resolved),
                "library": self._library(force=True),
            }

    def _session_docs_dir(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^\w\-]", "_", session_id)
        return DOCUMENTS_DIR / safe_id

    def list_documents(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            session = self._ensure_session(session_id)
            return list(session.get("documents", []))

    def upload_document(self, session_id: str, original_name: str, raw_bytes: bytes) -> dict[str, Any]:
        if len(raw_bytes) > MAX_DOC_SIZE_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_DOC_SIZE_BYTES // (1024*1024)}MB limit.")
        sanitized = _sanitize_filename(original_name)
        ext = Path(sanitized).suffix.lower()
        if ext not in DOC_ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type not supported: {ext}")

        with self._lock:
            session = self._ensure_session(session_id)
            existing = session.get("documents") or []
            current_total = sum(d.get("sizeBytes", 0) for d in existing)
            if current_total + len(raw_bytes) > MAX_SESSION_DOCS_BYTES:
                raise HTTPException(status_code=413, detail="Session document quota exceeded (200MB).")

            doc_id = f"doc-{uuid.uuid4().hex[:12]}"
            session_dir = self._session_docs_dir(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            try:
                session_dir.chmod(0o700)
            except OSError:
                pass

            doc_path = session_dir / f"{doc_id}{ext}"
            doc_path.write_bytes(raw_bytes)
            try:
                doc_path.chmod(0o600)
            except OSError:
                pass

        # Extract text and chunk OUTSIDE the lock (can be slow for large PDFs)
        try:
            text = _extract_text_from_file(doc_path)
        except RuntimeError as exc:
            doc_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        chunks = _chunk_text(text)
        chunks_path = session_dir / f"{doc_id}.chunks.json"
        chunks_path.write_text(
            json.dumps([{"index": i, "text": c} for i, c in enumerate(chunks)], indent=2),
            encoding="utf-8",
        )

        with self._lock:
            session = self._ensure_session(session_id)
            doc_meta = {
                "id": doc_id,
                "filename": doc_path.name,
                "originalName": sanitized,
                "sizeBytes": len(raw_bytes),
                "chunkCount": len(chunks),
                "uploadedAt": self._time_label(),
            }
            session.setdefault("documents", []).append(doc_meta)
            session["updatedAt"] = self._time_label()
            self.add_log("chat", "info", f"Document uploaded to session {session_id}: {sanitized} ({len(chunks)} chunks)")
            self._persist_sessions()
            return doc_meta

    def delete_document(self, session_id: str, doc_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._ensure_session(session_id)
            docs = session.get("documents") or []
            target = next((d for d in docs if d.get("id") == doc_id), None)
            if not target:
                raise HTTPException(status_code=404, detail="Document not found.")
            session["documents"] = [d for d in docs if d.get("id") != doc_id]
            session["updatedAt"] = self._time_label()
            session_dir = self._session_docs_dir(session_id)
            for f in session_dir.glob(f"{doc_id}*"):
                try:
                    f.unlink()
                except OSError:
                    pass
            self.add_log("chat", "info", f"Document removed: {target.get('originalName')}")
            self._persist_sessions()
            return {"deleted": doc_id}

    def delete_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            target = next((s for s in self.chat_sessions if s.get("id") == session_id), None)
            if not target:
                raise HTTPException(status_code=404, detail="Session not found.")
            self.chat_sessions = [s for s in self.chat_sessions if s.get("id") != session_id]
            self.add_log("chat", "info", f"Session deleted: {target.get('title', session_id)}")
            self._persist_sessions()
            return {"deleted": session_id}

    def _retrieve_session_context(self, session_id: str, prompt: str, top_k: int = 5) -> str:
        """Load all chunks from session documents and return the most relevant joined as context."""
        session_dir = self._session_docs_dir(session_id)
        if not session_dir.exists():
            return ""
        all_chunks: list[dict[str, Any]] = []
        for chunk_file in session_dir.glob("*.chunks.json"):
            try:
                doc_chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
                doc_name = chunk_file.stem.replace(".chunks", "")
                for c in doc_chunks:
                    all_chunks.append({"text": c.get("text", ""), "source": doc_name})
            except (OSError, json.JSONDecodeError):
                continue
        relevant = _retrieve_relevant_chunks(prompt, all_chunks, top_k=top_k)
        if not relevant:
            return ""
        return "\n\n---\n\n".join(c["text"] for c in relevant)

    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        with self._lock:
            session = self._ensure_session(request.sessionId, request.title)
            launch_preferences = self._launch_preferences()
            effective_model_ref = request.modelRef or session.get("modelRef")
            effective_model_name = request.modelName or session.get("model")
            effective_source = request.source or session.get("modelSource") or "catalog"
            effective_path = request.path if request.path is not None else session.get("modelPath")
            effective_backend = request.backend or session.get("modelBackend") or "auto"
            desired_cache_strategy = (
                request.cacheStrategy if request.cacheStrategy is not None else launch_preferences["cacheStrategy"]
            )
            desired_cache_bits = (
                request.cacheBits if request.cacheBits is not None else launch_preferences["cacheBits"]
            )
            desired_fp16_layers = (
                request.fp16Layers if request.fp16Layers is not None else launch_preferences["fp16Layers"]
            )
            desired_fused_attention = (
                launch_preferences["fusedAttention"] if request.fusedAttention is None else request.fusedAttention
            )
            desired_fit_model = (
                launch_preferences["fitModelInMemory"]
                if request.fitModelInMemory is None
                else request.fitModelInMemory
            )
            desired_context_tokens = (
                request.contextTokens if request.contextTokens is not None else launch_preferences["contextTokens"]
            )

            should_reload_model = self._should_reload_for_profile(
                model_ref=effective_model_ref,
                cache_bits=desired_cache_bits,
                fp16_layers=desired_fp16_layers,
                fused_attention=desired_fused_attention,
                cache_strategy=desired_cache_strategy,
                fit_model_in_memory=desired_fit_model,
                context_tokens=desired_context_tokens,
            )

            if effective_model_ref and should_reload_model:
                self.load_model(
                    LoadModelRequest(
                        modelRef=effective_model_ref,
                        modelName=effective_model_name,
                        source=effective_source,
                        backend=effective_backend,
                        path=effective_path,
                        cacheStrategy=desired_cache_strategy,
                        cacheBits=desired_cache_bits,
                        fp16Layers=desired_fp16_layers,
                        fusedAttention=desired_fused_attention,
                        fitModelInMemory=desired_fit_model,
                        contextTokens=desired_context_tokens,
                    )
                )

            if self.runtime.loaded_model is None:
                raise HTTPException(status_code=409, detail="Load a model before sending prompts.")

            history = [{"role": message["role"], "text": message["text"]} for message in session["messages"]]
            session["messages"].append({"role": "user", "text": request.prompt, "metrics": None})
            session["updatedAt"] = self._time_label()
            session["model"] = self.runtime.loaded_model.name
            session["modelRef"] = self.runtime.loaded_model.ref
            session["modelSource"] = self.runtime.loaded_model.source
            session["modelPath"] = self.runtime.loaded_model.path
            session["modelBackend"] = self.runtime.loaded_model.backend
            session["cacheLabel"] = self._cache_strategy_label(
                self.runtime.loaded_model.cacheBits,
                self.runtime.loaded_model.fp16Layers,
            )
            if session["title"] == "New chat":
                session["title"] = request.title or " ".join(request.prompt.strip().split()[:4]) or "New chat"
            model_tag = self.runtime.loaded_model.name if self.runtime.loaded_model else "unknown"
            msg_count = len(history) + 1
            self.add_log("chat", "info", f"[{model_tag}] Running chat completion on conversation with {msg_count} messages.")
            self.add_log("chat", "info", f"[{model_tag}] Generating response...")
            self.active_requests += 1
            # Build effective system prompt with RAG context if session has docs
            effective_system_prompt = request.systemPrompt
            doc_context = self._retrieve_session_context(session["id"], request.prompt)
            if doc_context:
                rag_preamble = (
                    "You have access to the following document context retrieved from the user's uploaded files. "
                    "Use it to answer their questions when relevant.\n\n--- DOCUMENT CONTEXT ---\n"
                    + doc_context
                    + "\n--- END CONTEXT ---"
                )
                effective_system_prompt = (rag_preamble + "\n\n" + (request.systemPrompt or "")).strip()
                self.add_log("chat", "info", f"[{model_tag}] Injected {len(doc_context)} chars of document context.")

        gen_start = time.perf_counter()
        try:
            # Run generation WITHOUT holding the lock so workspace polling
            # and health checks remain responsive during inference.
            result = self.runtime.generate(
                prompt=request.prompt,
                history=history,
                system_prompt=effective_system_prompt,
                max_tokens=request.maxTokens,
                temperature=request.temperature,
                images=request.images,
            )
        except RuntimeError as exc:
            with self._lock:
                self.active_requests = max(0, self.active_requests - 1)
                self.add_log("chat", "error", f"[{model_tag}] Generation failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        gen_elapsed = round(time.perf_counter() - gen_start, 2)
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.requests_served += 1
            assistant_message = {
                "role": "assistant",
                "text": result.text,
                "metrics": self._assistant_metrics_payload(result),
            }
            session["messages"].append(assistant_message)
            session["updatedAt"] = self._time_label()
            self._promote_session(session)
            self.add_log(
                "chat", "info",
                f"[{model_tag}] Finished response — {result.completionTokens} tokens in {gen_elapsed}s "
                f"({result.tokS} tok/s, {result.promptTokens} prompt tokens).",
            )
            self.add_activity("Chat completion", session["title"])
            self._persist_sessions()

            return {
                "session": session,
                "assistant": assistant_message,
                "runtime": self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served),
            }

    def generate_stream(self, request: GenerateRequest):
        """SSE streaming version of generate(). Returns a StreamingResponse."""
        # Reuse the same session/model preparation as generate()
        with self._lock:
            session = self._ensure_session(request.sessionId, request.title)
            launch_preferences = self._launch_preferences()
            effective_model_ref = request.modelRef or session.get("modelRef")
            effective_model_name = request.modelName or session.get("model")
            effective_source = request.source or session.get("modelSource") or "catalog"
            effective_path = request.path if request.path is not None else session.get("modelPath")
            effective_backend = request.backend or session.get("modelBackend") or "auto"
            desired_cache_strategy = request.cacheStrategy if request.cacheStrategy is not None else launch_preferences["cacheStrategy"]
            desired_cache_bits = request.cacheBits if request.cacheBits is not None else launch_preferences["cacheBits"]
            desired_fp16_layers = request.fp16Layers if request.fp16Layers is not None else launch_preferences["fp16Layers"]
            desired_fused_attention = launch_preferences["fusedAttention"] if request.fusedAttention is None else request.fusedAttention
            desired_fit_model = launch_preferences["fitModelInMemory"] if request.fitModelInMemory is None else request.fitModelInMemory
            desired_context_tokens = request.contextTokens if request.contextTokens is not None else launch_preferences["contextTokens"]

            should_reload = self._should_reload_for_profile(
                model_ref=effective_model_ref, cache_bits=desired_cache_bits,
                fp16_layers=desired_fp16_layers, fused_attention=desired_fused_attention,
                cache_strategy=desired_cache_strategy, fit_model_in_memory=desired_fit_model,
                context_tokens=desired_context_tokens,
            )
            if effective_model_ref and should_reload:
                self.load_model(LoadModelRequest(
                    modelRef=effective_model_ref, modelName=effective_model_name,
                    source=effective_source, backend=effective_backend, path=effective_path,
                    cacheStrategy=desired_cache_strategy, cacheBits=desired_cache_bits,
                    fp16Layers=desired_fp16_layers,
                    fusedAttention=desired_fused_attention,
                    fitModelInMemory=desired_fit_model, contextTokens=desired_context_tokens,
                ))

            if self.runtime.loaded_model is None:
                raise HTTPException(status_code=409, detail="Load a model before sending prompts.")

            history = [{"role": m["role"], "text": m["text"]} for m in session["messages"]]
            session["messages"].append({"role": "user", "text": request.prompt, "metrics": None})
            session["updatedAt"] = self._time_label()
            session["model"] = self.runtime.loaded_model.name
            session["modelRef"] = self.runtime.loaded_model.ref
            session["modelSource"] = self.runtime.loaded_model.source
            session["modelPath"] = self.runtime.loaded_model.path
            session["modelBackend"] = self.runtime.loaded_model.backend
            session["cacheLabel"] = self._cache_strategy_label(
                self.runtime.loaded_model.cacheBits, self.runtime.loaded_model.fp16Layers,
            )
            if session["title"] == "New chat":
                session["title"] = request.title or " ".join(request.prompt.strip().split()[:4]) or "New chat"
            model_tag = self.runtime.loaded_model.name
            self.add_log("chat", "info", f"[{model_tag}] Streaming response...")
            self.active_requests += 1
            # Build effective system prompt with RAG context if session has docs
            effective_system_prompt = request.systemPrompt
            doc_context = self._retrieve_session_context(session["id"], request.prompt)
            if doc_context:
                rag_preamble = (
                    "You have access to the following document context retrieved from the user's uploaded files. "
                    "Use it to answer their questions when relevant.\n\n--- DOCUMENT CONTEXT ---\n"
                    + doc_context
                    + "\n--- END CONTEXT ---"
                )
                effective_system_prompt = (rag_preamble + "\n\n" + (request.systemPrompt or "")).strip()
                self.add_log("chat", "info", f"[{model_tag}] Injected {len(doc_context)} chars of document context.")

        chaosengine = self
        gen_start = time.perf_counter()

        def _sse_stream():
            full_text = ""
            final_chunk = None
            try:
                for chunk in chaosengine.runtime.stream_generate(
                    prompt=request.prompt, history=history,
                    system_prompt=effective_system_prompt,
                    max_tokens=request.maxTokens, temperature=request.temperature,
                    images=request.images,
                ):
                    if chunk.text:
                        full_text += chunk.text
                        yield f"data: {json.dumps({'token': chunk.text})}\n\n"
                    if chunk.done:
                        final_chunk = chunk
            except RuntimeError as exc:
                with chaosengine._lock:
                    chaosengine.active_requests = max(0, chaosengine.active_requests - 1)
                    chaosengine.add_log("chat", "error", f"[{model_tag}] Streaming failed: {exc}")
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return

            gen_elapsed = round(time.perf_counter() - gen_start, 2)
            with chaosengine._lock:
                chaosengine.active_requests = max(0, chaosengine.active_requests - 1)
                chaosengine.requests_served += 1

                # Build a GenerationResult-like object for metrics
                tok_s = final_chunk.tok_s if final_chunk else 0
                prompt_tokens = final_chunk.prompt_tokens if final_chunk else 0
                completion_tokens = final_chunk.completion_tokens if final_chunk else 0
                # Fallback: compute tok/s from elapsed time if engine didn't provide one
                if (not tok_s or tok_s == 0) and completion_tokens > 0 and gen_elapsed > 0:
                    tok_s = round(completion_tokens / gen_elapsed, 1)

                assistant_message = {
                    "role": "assistant",
                    "text": full_text,
                    "metrics": {
                        "finishReason": final_chunk.finish_reason if final_chunk else "stop",
                        "promptTokens": prompt_tokens,
                        "completionTokens": completion_tokens,
                        "totalTokens": prompt_tokens + completion_tokens,
                        "tokS": tok_s,
                        "responseSeconds": gen_elapsed,
                        "runtimeNote": final_chunk.runtime_note if final_chunk else None,
                        "model": chaosengine.runtime.loaded_model.name if chaosengine.runtime.loaded_model else None,
                        "modelRef": chaosengine.runtime.loaded_model.ref if chaosengine.runtime.loaded_model else None,
                        "backend": chaosengine.runtime.loaded_model.backend if chaosengine.runtime.loaded_model else None,
                        "engineLabel": chaosengine.runtime.engine.engine_label,
                        "cacheLabel": chaosengine._cache_strategy_label(
                            chaosengine.runtime.loaded_model.cacheBits,
                            chaosengine.runtime.loaded_model.fp16Layers,
                        ) if chaosengine.runtime.loaded_model else None,
                        "contextTokens": chaosengine.runtime.loaded_model.contextTokens if chaosengine.runtime.loaded_model else None,
                        "generatedAt": chaosengine._time_label(),
                    },
                }
                session["messages"].append(assistant_message)
                session["updatedAt"] = chaosengine._time_label()
                chaosengine._promote_session(session)
                chaosengine.add_log(
                    "chat", "info",
                    f"[{model_tag}] Finished streaming — {completion_tokens} tokens in {gen_elapsed}s ({tok_s} tok/s).",
                )
                chaosengine._persist_sessions()

                done_payload = {
                    "done": True,
                    "session": session,
                    "assistant": assistant_message,
                    "runtime": chaosengine.runtime.status(
                        active_requests=chaosengine.active_requests,
                        requests_served=chaosengine.requests_served,
                    ),
                }
            yield f"data: {json.dumps(done_payload)}\n\n"

        return StreamingResponse(
            _sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    def start_download(self, repo: str) -> dict[str, Any]:
        if not _HF_REPO_PATTERN.match(repo):
            raise HTTPException(status_code=400, detail="Invalid repo format. Expected 'owner/model-name'.")
        if repo in self._downloads and self._downloads[repo].get("state") == "downloading":
            return self._downloads[repo]

        total_gb = _known_repo_size_gb(repo)
        downloaded_gb = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
        initial_progress = 0.0
        if isinstance(total_gb, (int, float)) and total_gb > 0 and downloaded_gb > 0:
            initial_progress = min(0.99, downloaded_gb / float(total_gb))
        elif downloaded_gb > 0:
            initial_progress = 0.01
        download_token = uuid.uuid4().hex
        self._downloads[repo] = {
            "repo": repo,
            "state": "downloading",
            "progress": initial_progress,
            "downloadedGb": downloaded_gb,
            "totalGb": total_gb,
            "error": None,
        }
        self._download_cancel[repo] = False
        self._download_tokens[repo] = download_token
        self.add_log("library", "info", f"{'Resuming' if downloaded_gb > 0 else 'Starting'} download: {repo}")

        def _download_worker():
            stop_progress = threading.Event()
            process: subprocess.Popen[str] | None = None

            def _progress_worker() -> None:
                while not stop_progress.wait(1.0):
                    downloaded_bytes = _hf_repo_downloaded_bytes(repo)
                    downloaded_gb = _bytes_to_gb(downloaded_bytes)
                    with self._lock:
                        current = self._downloads.get(repo)
                        if (
                            current is None
                            or current.get("state") != "downloading"
                            or self._download_tokens.get(repo) != download_token
                        ):
                            return
                        current["downloadedGb"] = downloaded_gb
                        total = current.get("totalGb")
                        if isinstance(total, (int, float)) and total > 0:
                            current["progress"] = min(0.99, downloaded_gb / float(total))
                        elif downloaded_gb > 0:
                            current["progress"] = max(float(current.get("progress") or 0.0), 0.01)

            monitor = threading.Thread(target=_progress_worker, daemon=True)
            monitor.start()
            try:
                with self._lock:
                    if self._download_tokens.get(repo) != download_token:
                        return
                env = os.environ.copy()
                env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                env.setdefault("PYTHONUNBUFFERED", "1")
                process = subprocess.Popen(
                    [sys.executable, "-c", HF_SNAPSHOT_DOWNLOAD_HELPER, repo],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                with self._lock:
                    if self._download_tokens.get(repo) == download_token:
                        self._download_processes[repo] = process

                while True:
                    with self._lock:
                        cancel_requested = self._download_cancel.get(repo, False)
                        token_matches = self._download_tokens.get(repo) == download_token
                    if not token_matches:
                        return
                    if cancel_requested:
                        if process.poll() is None:
                            try:
                                process.terminate()
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)

                stderr_output = ""
                if process.stderr is not None:
                    stderr_output = process.stderr.read().strip()
                returncode = process.returncode if process.returncode is not None else process.wait()

                with self._lock:
                    if self._download_tokens.get(repo) != download_token:
                        return
                    cancelled = self._download_cancel.get(repo, False)
                if cancelled:
                    downloaded_gb = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
                    with self._lock:
                        current = self._downloads.get(repo)
                        if current is None or self._download_tokens.get(repo) != download_token:
                            return
                        current["state"] = "cancelled"
                        current["error"] = None
                        current["downloadedGb"] = downloaded_gb
                        total = current.get("totalGb")
                        if isinstance(total, (int, float)) and total > 0:
                            current["progress"] = min(0.99, downloaded_gb / float(total))
                        elif downloaded_gb > 0:
                            current["progress"] = max(float(current.get("progress") or 0.0), 0.01)
                    return

                if returncode != 0:
                    raise RuntimeError(stderr_output or f"snapshot_download exited with status {returncode}")

                validation_error = _image_download_validation_error(repo)
                if validation_error:
                    with self._lock:
                        if self._download_tokens.get(repo) != download_token:
                            return
                        self._downloads[repo]["state"] = "failed"
                        self._downloads[repo]["error"] = validation_error
                        self.add_log("library", "error", validation_error)
                    return
                downloaded_gb = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
                with self._lock:
                    if self._download_tokens.get(repo) != download_token:
                        return
                    self._downloads[repo]["state"] = "completed"
                    self._downloads[repo]["progress"] = 1.0
                    self._downloads[repo]["downloadedGb"] = downloaded_gb
                    if downloaded_gb > 0:
                        current_total = self._downloads[repo].get("totalGb")
                        if not isinstance(current_total, (int, float)) or current_total <= 0:
                            self._downloads[repo]["totalGb"] = downloaded_gb
                        else:
                            self._downloads[repo]["totalGb"] = max(float(current_total), downloaded_gb)
                    self._library_cache = None  # Invalidate so next scan picks up new model
                    self.add_log("library", "info", f"Download completed: {repo}")
            except Exception as exc:
                with self._lock:
                    if self._download_tokens.get(repo) != download_token:
                        return
                    self._downloads[repo]["state"] = "failed"
                    friendly_error = _friendly_image_download_error(repo, str(exc))
                    self._downloads[repo]["error"] = friendly_error
                    self.add_log("library", "error", f"Download failed for {repo}: {friendly_error}")
            finally:
                stop_progress.set()
                monitor.join(timeout=1.0)
                with self._lock:
                    if process is not None and self._download_processes.get(repo) is process:
                        self._download_processes.pop(repo, None)
                    if self._download_tokens.get(repo) == download_token and self._downloads.get(repo, {}).get("state") != "downloading":
                        self._download_tokens.pop(repo, None)
                        self._download_cancel.pop(repo, None)

        t = threading.Thread(target=_download_worker, daemon=True)
        t.start()
        return self._downloads[repo]

    def download_status(self) -> list[dict[str, Any]]:
        return list(self._downloads.values())

    def cancel_download(self, repo: str) -> dict[str, Any]:
        with self._lock:
            current = self._downloads.get(repo)
            if current is None:
                return {"repo": repo, "state": "not_found"}
            if current.get("state") == "completed":
                return current
            self._download_cancel[repo] = True
            process = self._download_processes.get(repo)

        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                pass

        downloaded_gb = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
        with self._lock:
            current = self._downloads.get(repo)
            if current is None:
                return {"repo": repo, "state": "not_found"}
            current["state"] = "cancelled"
            current["error"] = None
            current["downloadedGb"] = downloaded_gb
            total = current.get("totalGb")
            if isinstance(total, (int, float)) and total > 0:
                current["progress"] = min(0.99, downloaded_gb / float(total))
            elif downloaded_gb > 0:
                current["progress"] = max(float(current.get("progress") or 0.0), 0.01)
            self.add_log("library", "info", f"Download paused: {repo}")
            return current
        return {"repo": repo, "state": "not_found"}

    def server_status(self) -> dict[str, Any]:
        runtime = self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served)
        loaded = runtime["loadedModel"]
        recent_server_logs = [
            entry["message"] for entry in list(self.logs) if entry["source"] in {"runtime", "chat", "server"}
        ][:3]
        status = "running" if runtime["serverReady"] else "idle"
        remote_enabled = DEFAULT_HOST != "127.0.0.1"
        localhost_url = f"http://127.0.0.1:{self.server_port}/v1"
        lan_urls = [f"http://{address}:{self.server_port}/v1" for address in _local_ipv4_addresses()] if remote_enabled else []
        base_url = localhost_url
        preferred_port = self.settings["preferredServerPort"]
        port_note = (
            f"Preferred API port is {preferred_port}. Restart the API service to apply it."
            if preferred_port != self.server_port
            else (
                "Remote access is enabled for local-network clients. Allow incoming connections in your firewall if prompted."
                if remote_enabled
                else "Third-party tools on this machine can target the displayed localhost URL."
            )
        )
        loading = None
        if self._loading_state is not None:
            elapsed = time.time() - self._loading_state["startedAt"]
            loading = {
                "modelName": self._loading_state["modelName"],
                "stage": self._loading_state["stage"],
                "elapsedSeconds": round(elapsed, 1),
                "progress": self._loading_state.get("progress"),
                "progressPercent": self._loading_state.get("progressPercent"),
                "progressPhase": self._loading_state.get("progressPhase"),
                "progressMessage": self._loading_state.get("progressMessage"),
                "recentLogLines": list(self._loading_state.get("recentLogLines") or []),
            }

        return {
            "status": status,
            "baseUrl": base_url,
            "localhostUrl": localhost_url,
            "lanUrls": lan_urls,
            "bindHost": DEFAULT_HOST,
            "remoteAccessActive": remote_enabled,
            "port": self.server_port,
            "activeConnections": runtime["activeRequests"],
            "concurrentRequests": runtime["activeRequests"],
            "requestsServed": runtime["requestsServed"],
            "loadedModelName": loaded["name"] if loaded else None,
            "loading": loading,
            "logTail": recent_server_logs or [
                "Load a model to make the OpenAI-compatible local API ready for external tools.",
                "Ports and concurrency are configurable in Settings.",
                port_note,
            ],
        }

    def workspace(self) -> dict[str, Any]:
        system_stats = self._system_snapshot_provider()
        # Enrich running LLM processes with the currently loaded model name
        # (and warm-pool model names) so the dashboard shows what's actually loaded
        try:
            loaded_name = self.runtime.loaded_model.name if self.runtime.loaded_model else None
            loaded_engine = self.runtime.engine.engine_name if self.runtime.engine else None
            warm_entries = [
                (engine.engine_name, info.name)
                for engine, info in self.runtime._warm_pool.values()
            ]
            procs = system_stats.get("runningLlmProcesses") or []

            # Group processes by kind — sort each group by memory (already sorted by _list_llm_processes)
            mlx_workers = [p for p in procs if p.get("kind") == "mlx_worker"]
            llama_servers = [p for p in procs if p.get("kind") == "llama_server"]

            # Assign loaded model to the correct process type
            assigned_loaded = False
            if loaded_name and loaded_engine == "mlx" and mlx_workers:
                mlx_workers[0]["modelName"] = loaded_name
                mlx_workers[0]["modelStatus"] = "active"
                assigned_loaded = True
            elif loaded_name and loaded_engine == "llama.cpp" and llama_servers:
                llama_servers[0]["modelName"] = loaded_name
                llama_servers[0]["modelStatus"] = "active"
                assigned_loaded = True

            # Fallback: biggest ChaosEngineAI Python process gets the loaded model
            if loaded_name and not assigned_loaded:
                for proc in procs:
                    if proc.get("owner") == "ChaosEngineAI" and not proc.get("modelName"):
                        proc["modelName"] = loaded_name
                        proc["modelStatus"] = "active"
                        break

            # Assign warm pool entries to remaining matching processes.
            # Defensively exclude the active model name — a warm entry with
            # the same name as `loaded_name` would falsely show the same
            # model as both ACTIVE and WARM in the UI.
            warm_mlx = [
                name for engine, name in warm_entries
                if engine == "mlx" and name != loaded_name
            ]
            warm_llama = [
                name for engine, name in warm_entries
                if engine == "llama.cpp" and name != loaded_name
            ]
            for proc in mlx_workers[1:]:
                if warm_mlx and not proc.get("modelName"):
                    proc["modelName"] = warm_mlx.pop(0)
                    proc["modelStatus"] = "warm"
            for proc in llama_servers[1:]:
                if warm_llama and not proc.get("modelName"):
                    proc["modelName"] = warm_llama.pop(0)
                    proc["modelStatus"] = "warm"
        except Exception:
            pass

        # Add disk usage for the first enabled model directory
        try:
            disk_info = _get_disk_usage_for_models(self.settings)
            if disk_info:
                system_stats["diskFreeGb"] = disk_info["freeGb"]
                system_stats["diskTotalGb"] = disk_info["totalGb"]
                system_stats["diskUsedGb"] = disk_info["usedGb"]
                system_stats["diskPath"] = disk_info.get("path")
        except Exception:
            pass
        library = self._library()
        recommendation = _best_fit_recommendation(system_stats)
        launch_preferences = self._launch_preferences()
        return {
            "system": system_stats,
            "recommendation": recommendation,
            "featuredModels": _model_family_payloads(system_stats, library),
            "library": library,
            "settings": self._settings_payload(library),
            "chatSessions": self.chat_sessions,
            "runtime": self.runtime.status(
                active_requests=self.active_requests,
                requests_served=self.requests_served,
            ),
            "server": self.server_status(),
            "benchmarks": self.benchmark_runs,
            "logs": [entry for entry in self.logs if entry.get("level") != "debug"],
            "activity": list(self.activity),
            "preview": compute_cache_preview(
                bits=launch_preferences["cacheBits"],
                fp16_layers=launch_preferences["fp16Layers"],
                context_tokens=launch_preferences["contextTokens"],
                system_stats=system_stats,
            ),
            "quickActions": [
                "Online Models",
                "New Thread",
                "Start Server",
                "Convert to MLX",
                "Run Benchmark",
                "Open Logs",
            ],
        }

    def openai_models(self) -> dict[str, Any]:
        runtime = self.runtime.status(active_requests=self.active_requests, requests_served=self.requests_served)
        loaded = runtime["loadedModel"]
        if loaded is None:
            return {"object": "list", "data": []}
        created = int(time.time())
        # Expose both the canonical ref and the runtimeTarget so that
        # third-party tools (Goose, Continue, etc.) can match by either name.
        seen: set[str] = set()
        data: list[dict[str, Any]] = []
        for model_id in (loaded["ref"], loaded.get("runtimeTarget")):
            if model_id and model_id not in seen:
                seen.add(model_id)
                data.append({
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "chaosengine",
                })
        return {"object": "list", "data": data}

    def openai_chat_completion(self, request: OpenAIChatCompletionRequest) -> dict[str, Any] | StreamingResponse:
        if not request.messages:
            raise HTTPException(status_code=400, detail="At least one message is required.")

        last_user = None
        last_user_images: list[str] = []
        history: list[dict[str, Any]] = []
        system_prompt = None
        for message in request.messages:
            # Extract text and images from multimodal content arrays
            if isinstance(message.content, list):
                text_parts = []
                for part in message.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(str(part.get("text", "")))
                        elif part.get("type") == "image_url":
                            url = (part.get("image_url") or {}).get("url", "")
                            if url.startswith("data:") and ";base64," in url:
                                last_user_images.append(url.split(";base64,", 1)[1])
                content = " ".join(text_parts) if text_parts else ""
            else:
                content = str(message.content) if message.content is not None else ""

            if message.role == "system" and system_prompt is None:
                system_prompt = content
            elif message.role == "user":
                last_user = content
                history.append({"role": "user", "text": content})
            elif message.role == "assistant":
                # Handle assistant messages with tool_calls
                if message.tool_calls:
                    history.append({"role": "assistant", "text": content, "tool_calls": message.tool_calls})
                else:
                    history.append({"role": "assistant", "text": content})
            elif message.role == "tool":
                history.append({"role": "tool", "text": content, "tool_call_id": message.tool_call_id})

        if last_user is None:
            raise HTTPException(status_code=400, detail="A user message is required.")

        msg_count = len(request.messages)

        with self._lock:
            launch_preferences = self._launch_preferences()
            if self.runtime.loaded_model is None and request.model:
                # No model loaded — try to auto-load the requested one
                self.add_log("server", "info", f"[{request.model}] Auto-loading model for /v1/chat/completions...")
                self.load_model(
                    LoadModelRequest(
                        modelRef=request.model,
                        modelName=request.model,
                        source="openai",
                        backend="auto",
                        cacheStrategy=launch_preferences["cacheStrategy"],
                        cacheBits=launch_preferences["cacheBits"],
                        fp16Layers=launch_preferences["fp16Layers"],
                        fusedAttention=launch_preferences["fusedAttention"],
                        fitModelInMemory=launch_preferences["fitModelInMemory"],
                        contextTokens=launch_preferences["contextTokens"],
                    )
                )
            # Multi-model routing: if request.model matches a warm-pool entry,
            # serve from that engine without disturbing the active model. If it
            # doesn't match anything, fall back to the active engine.
            if self.runtime.loaded_model is None:
                raise HTTPException(status_code=409, detail="Load a model before calling /v1/chat/completions.")

            try:
                target_engine, target_info = self.runtime.get_engine_for_request(request.model)
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            self.active_requests += 1
            model_ref = target_info.ref
            model_tag = target_info.name
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            self.add_log("server", "info", f"[{model_tag}] Running chat completion on conversation with {msg_count} messages.")

        if request.stream:
            chaosengine = self

            def _stream_chunks():
                stream_start = time.perf_counter()
                with chaosengine._lock:
                    chaosengine.add_log("server", "info", f"[{model_tag}] Streaming response...")
                token_count = 0
                prompt_tokens = 0
                try:
                    first = True
                    for chunk in chaosengine.runtime.stream_generate(
                        prompt=last_user,
                        history=history[:-1],
                        system_prompt=system_prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        images=last_user_images or None,
                        tools=request.tools,
                        engine=target_engine,
                    ):
                        if chunk.text:
                            token_count += 1
                            delta = {"content": chunk.text}
                            if first:
                                delta["role"] = "assistant"
                                first = False
                            sse_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_ref,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(sse_chunk)}\n\n"
                        if chunk.done:
                            if hasattr(chunk, "prompt_tokens") and chunk.prompt_tokens:
                                prompt_tokens = chunk.prompt_tokens
                            if hasattr(chunk, "completion_tokens") and chunk.completion_tokens:
                                token_count = chunk.completion_tokens
                            done_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_ref,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": chunk.finish_reason or "stop",
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(done_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except RuntimeError as exc:
                    with chaosengine._lock:
                        chaosengine.add_log("server", "error", f"[{model_tag}] Streaming failed: {exc}")
                finally:
                    elapsed = round(time.perf_counter() - stream_start, 2)
                    tok_s = round(token_count / elapsed, 1) if elapsed > 0 else 0
                    with chaosengine._lock:
                        chaosengine.active_requests = max(0, chaosengine.active_requests - 1)
                        chaosengine.requests_served += 1
                        chaosengine.add_log(
                            "server", "info",
                            f"[{model_tag}] Finished streaming response — {token_count} tokens in {elapsed}s "
                            f"({tok_s} tok/s{f', {prompt_tokens} prompt tokens' if prompt_tokens else ''}).",
                        )

            return StreamingResponse(
                _stream_chunks(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        with self._lock:
            self.add_log("server", "info", f"[{model_tag}] Generating response...")
        gen_start = time.perf_counter()
        try:
            # Run generation WITHOUT holding the lock so workspace polling
            # and health checks remain responsive during inference.
            result = self.runtime.generate(
                prompt=last_user,
                history=history[:-1],
                system_prompt=system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                images=last_user_images or None,
                tools=request.tools,
                engine=target_engine,
            )
        except RuntimeError as exc:
            with self._lock:
                self.active_requests = max(0, self.active_requests - 1)
                self.add_log("server", "error", f"[{model_tag}] Generation failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        gen_elapsed = round(time.perf_counter() - gen_start, 2)
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.requests_served += 1
            self.add_log(
                "server", "info",
                f"[{model_tag}] Finished response — {result.completionTokens} tokens in {gen_elapsed}s "
                f"({result.tokS} tok/s, {result.promptTokens} prompt tokens).",
            )

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_ref,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": result.finishReason,
                        "message": {
                            "role": "assistant",
                            "content": result.text,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": result.promptTokens,
                    "completion_tokens": result.completionTokens,
                    "total_tokens": result.totalTokens,
                },
            }


def _build_benchmarks() -> list[dict[str, Any]]:
    return [
        {
            "id": "baseline",
            "label": "FP16 baseline",
            "bits": 16,
            "fp16Layers": 32,
            "cacheGb": 14.0,
            "compression": 1.0,
            "tokS": 52.1,
            "quality": 100,
        },
        {
            "id": "native-34",
            "label": "Native 3-bit 4+4",
            "bits": 3,
            "fp16Layers": 4,
            "cacheGb": 5.9,
            "compression": 2.4,
            "tokS": 30.7,
            "quality": 98,
        },
        {
            "id": "native-36",
            "label": "Native 3-bit 6+6",
            "bits": 3,
            "fp16Layers": 6,
            "cacheGb": 7.5,
            "compression": 1.9,
            "tokS": 33.0,
            "quality": 98,
        },
        {
            "id": "native-44",
            "label": "Native 4-bit 4+4",
            "bits": 4,
            "fp16Layers": 4,
            "cacheGb": 7.1,
            "compression": 2.0,
            "tokS": 35.8,
            "quality": 99,
        },
    ]


def create_app(state: ChaosEngineState | None = None) -> FastAPI:
    app = FastAPI(title="ChaosEngineAI Sidecar", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.chaosengine = state or ChaosEngineState(server_port=DEFAULT_PORT)

    # Shutdown hook: kill any running llama-server / MLX worker children
    # on backend exit. Runs on clean shutdown (uvicorn SIGTERM), Ctrl-C,
    # and normal Python exit. Prevents orphan llama-server processes from
    # surviving across dev sessions.
    import atexit as _atexit
    import signal as _signal

    def _shutdown_children(*_args: Any) -> None:
        try:
            runtime = getattr(app.state.chaosengine, "runtime", None)
            if runtime is None:
                return
            engine = getattr(runtime, "engine", None)
            # LlamaCppEngine exposes _cleanup_process; MLXWorkerEngine has
            # a worker with close(); remote engines have nothing to do.
            if engine is not None:
                cleanup = getattr(engine, "_cleanup_process", None)
                if callable(cleanup):
                    try:
                        cleanup()
                    except Exception:
                        pass
                worker = getattr(engine, "worker", None)
                if worker is not None and hasattr(worker, "close"):
                    try:
                        worker.close()
                    except Exception:
                        pass
            # Also close any warm-pool engines.
            warm_pool = getattr(runtime, "_warm_pool", None)
            if isinstance(warm_pool, dict):
                for key, entry in list(warm_pool.items()):
                    try:
                        warm_engine = entry[0] if isinstance(entry, tuple) else entry
                        cleanup = getattr(warm_engine, "_cleanup_process", None)
                        if callable(cleanup):
                            cleanup()
                        worker = getattr(warm_engine, "worker", None)
                        if worker is not None and hasattr(worker, "close"):
                            worker.close()
                    except Exception:
                        pass
        except Exception:
            pass

    _atexit.register(_shutdown_children)
    # Also catch SIGTERM explicitly (uvicorn's normal shutdown signal).
    try:
        _signal.signal(_signal.SIGTERM, lambda *a: _shutdown_children())
    except (ValueError, OSError):
        pass  # not in main thread or signal not available

    @app.middleware("http")
    async def log_requests(request, call_next):
        path = request.url.path
        # Skip noisy internal polling endpoints that flood the log
        _quiet_paths = {
            "/api/server/logs/stream",
            "/api/health",
            "/api/workspace",
            "/api/runtime",
            "/api/cache/preview",
        }
        skip = path in _quiet_paths
        if not skip:
            # Don't log routine HTTP requests — they flood the log.
            # Meaningful operations (model load, chat, etc.) log themselves.
            pass
        response = await call_next(request)
        if not skip and response.status_code >= 400:
            app.state.chaosengine.add_log(
                "server", "warn",
                f"{request.method} {path} -> {response.status_code}",
            )
        return response

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        runtime_status = app.state.chaosengine.runtime.status(
            active_requests=app.state.chaosengine.active_requests,
            requests_served=app.state.chaosengine.requests_served,
        )
        return {
            "status": "ok",
            "workspaceRoot": str(WORKSPACE_ROOT),
            "runtime": _runtime_label(),
            "appVersion": app_version,
            "engine": runtime_status["engine"],
            "loadedModel": runtime_status["loadedModel"],
            "nativeBackends": runtime_status["nativeBackends"],
        }

    @app.get("/api/workspace")
    def workspace() -> dict[str, Any]:
        return app.state.chaosengine.workspace()

    @app.get("/api/runtime")
    def runtime_status() -> dict[str, Any]:
        return app.state.chaosengine.runtime.status(
            active_requests=app.state.chaosengine.active_requests,
            requests_served=app.state.chaosengine.requests_served,
        )

    @app.get("/api/settings")
    def settings() -> dict[str, Any]:
        library = app.state.chaosengine._library()
        return {"settings": app.state.chaosengine._settings_payload(library)}

    @app.patch("/api/settings")
    def update_settings(request: UpdateSettingsRequest) -> dict[str, Any]:
        return app.state.chaosengine.update_settings(request)

    @app.post("/api/models/load")
    def load_model(request: LoadModelRequest) -> dict[str, Any]:
        try:
            runtime = app.state.chaosengine.load_model(request)
            return {"runtime": runtime}
        except HTTPException:
            raise
        except Exception as exc:
            detail = str(exc) or "Unknown error during model loading."
            app.state.chaosengine.add_log("runtime", "error", f"Load failed for {request.modelRef}: {detail}")
            raise HTTPException(status_code=500, detail=detail) from exc

    @app.post("/api/models/unload")
    async def unload_model(http_request: Request) -> dict[str, Any]:
        ref: str | None = None
        try:
            body = await http_request.body()
            if body:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    ref = payload.get("ref")
        except Exception:
            ref = None
        runtime = app.state.chaosengine.unload_model(ref=ref)
        return {"runtime": runtime}

    @app.post("/api/models/convert")
    def convert_model(request: ConvertModelRequest) -> dict[str, Any]:
        try:
            return app.state.chaosengine.convert_model(request)
        except RuntimeError as exc:
            detail = str(exc)
            app.state.chaosengine.add_log("conversion", "error", f"Conversion failed: {detail}")
            raise HTTPException(status_code=400, detail=detail) from exc

    @app.post("/api/benchmarks/run")
    def run_benchmark(request: BenchmarkRunRequest) -> dict[str, Any]:
        try:
            return app.state.chaosengine.run_benchmark(request)
        except RuntimeError as exc:
            detail = str(exc)
            app.state.chaosengine.add_log("benchmark", "error", f"Benchmark failed: {detail}")
            raise HTTPException(status_code=400, detail=detail) from exc

    @app.post("/api/models/reveal")
    def reveal_model_path(request: RevealPathRequest) -> dict[str, Any]:
        return app.state.chaosengine.reveal_model_path(request.path)

    @app.post("/api/models/delete")
    def delete_model_path(request: DeleteModelRequest) -> dict[str, Any]:
        return app.state.chaosengine.delete_model_path(request.path)

    @app.get("/api/models/list-weights")
    def list_weights(path: str) -> dict[str, Any]:
        return _list_weight_files(path)

    @app.get("/api/chat/sessions/{session_id}/documents")
    def list_session_documents(session_id: str) -> dict[str, Any]:
        return {"documents": app.state.chaosengine.list_documents(session_id)}

    @app.post("/api/chat/sessions/{session_id}/documents")
    async def upload_session_document(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        raw = await file.read()
        return {"document": app.state.chaosengine.upload_document(session_id, file.filename or "document", raw)}

    @app.delete("/api/chat/sessions/{session_id}/documents/{doc_id}")
    def delete_session_document(session_id: str, doc_id: str) -> dict[str, Any]:
        return app.state.chaosengine.delete_document(session_id, doc_id)

    @app.delete("/api/chat/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        return app.state.chaosengine.delete_session(session_id)

    @app.post("/api/models/download")
    def download_model(request: DownloadModelRequest) -> dict[str, Any]:
        return {"download": app.state.chaosengine.start_download(request.repo)}

    @app.get("/api/models/download/status")
    def download_status() -> dict[str, Any]:
        return {"downloads": app.state.chaosengine.download_status()}

    @app.post("/api/models/download/cancel")
    def cancel_download(request: DownloadModelRequest) -> dict[str, Any]:
        return {"download": app.state.chaosengine.cancel_download(request.repo)}

    @app.get("/api/images/catalog")
    def image_catalog() -> dict[str, Any]:
        library = app.state.chaosengine._library()
        return {
            "families": _image_model_payloads(library),
            "latest": _latest_image_model_payloads(library),
        }

    @app.get("/api/images/runtime")
    def image_runtime_status() -> dict[str, Any]:
        return {"runtime": app.state.chaosengine.image_runtime.capabilities()}

    @app.post("/api/images/preload")
    def preload_image_model(request: ImageRuntimePreloadRequest) -> dict[str, Any]:
        variant = _find_image_variant(request.modelId)
        if variant is None:
            raise HTTPException(status_code=404, detail=f"Unknown image model '{request.modelId}'.")
        library = app.state.chaosengine._library()
        if not _image_variant_available_locally(variant, library):
            validation_error = _image_download_validation_error(variant["repo"])
            detail = validation_error or f"{variant['name']} is not installed locally yet."
            raise HTTPException(status_code=409, detail=detail)
        try:
            runtime = app.state.chaosengine.image_runtime.preload(variant["repo"])
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        app.state.chaosengine.add_log("images", "info", f"Preloaded image model {variant['name']}.")
        app.state.chaosengine.add_activity("Image model loaded", variant["name"])
        return {"runtime": runtime}

    @app.post("/api/images/unload")
    def unload_image_model(request: ImageRuntimeUnloadRequest | None = None) -> dict[str, Any]:
        requested_repo: str | None = None
        requested_name: str | None = None
        if request and request.modelId:
            variant = _find_image_variant(request.modelId)
            if variant is None:
                raise HTTPException(status_code=404, detail=f"Unknown image model '{request.modelId}'.")
            requested_repo = variant["repo"]
            requested_name = variant["name"]
        current_runtime = app.state.chaosengine.image_runtime.capabilities()
        current_repo = str(current_runtime.get("loadedModelRepo") or "") or None
        try:
            runtime = app.state.chaosengine.image_runtime.unload(requested_repo)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        unloaded_repo = requested_repo or current_repo
        if unloaded_repo and (requested_repo is None or requested_repo == current_repo):
            unloaded_variant = _find_image_variant_by_repo(unloaded_repo)
            unloaded_name = unloaded_variant["name"] if unloaded_variant else requested_name or unloaded_repo
            app.state.chaosengine.add_log("images", "info", f"Unloaded image model {unloaded_name}.")
            app.state.chaosengine.add_activity("Image model unloaded", unloaded_name)
        return {"runtime": runtime}

    @app.get("/api/images/library")
    def image_library() -> dict[str, Any]:
        library = app.state.chaosengine._library()
        installed_models: list[dict[str, Any]] = []
        for family in _image_model_payloads(library):
            for variant in family["variants"]:
                if variant.get("availableLocally"):
                    installed_models.append(
                        {
                            **variant,
                            "familyName": family["name"],
                        }
                    )
        return {"models": installed_models}

    @app.post("/api/images/download")
    def download_image_model(request: DownloadModelRequest) -> dict[str, Any]:
        return {"download": app.state.chaosengine.start_download(request.repo)}

    @app.get("/api/images/download/status")
    def image_download_status() -> dict[str, Any]:
        image_repos = _image_download_repo_ids()
        downloads = [
            item
            for item in app.state.chaosengine.download_status()
            if str(item.get("repo") or "") in image_repos
        ]
        return {"downloads": downloads}

    @app.post("/api/images/download/cancel")
    def cancel_image_download(request: DownloadModelRequest) -> dict[str, Any]:
        return {"download": app.state.chaosengine.cancel_download(request.repo)}

    @app.post("/api/images/generate")
    def generate_image(request: ImageGenerationRequest) -> dict[str, Any]:
        variant = _find_image_variant(request.modelId)
        if variant is None:
            raise HTTPException(status_code=404, detail=f"Unknown image model '{request.modelId}'.")
        artifacts, runtime = _generate_image_artifacts(request, variant, app.state.chaosengine.image_runtime)
        app.state.chaosengine.add_log(
            "images",
            "info",
            f"Generated {len(artifacts)} image(s) with {variant['name']} via {runtime.get('activeEngine', 'unknown')}.",
        )
        app.state.chaosengine.add_activity(
            "Image generated",
            f"{variant['name']} · {request.width}x{request.height}",
        )
        return {"artifacts": artifacts, "outputs": _load_image_outputs(), "runtime": runtime}

    @app.get("/api/images/outputs")
    def image_outputs() -> dict[str, Any]:
        return {"outputs": _load_image_outputs()}

    @app.get("/api/images/outputs/{artifact_id}")
    def image_output_detail(artifact_id: str) -> dict[str, Any]:
        output = _find_image_output(artifact_id)
        if output is None:
            raise HTTPException(status_code=404, detail=f"Image output '{artifact_id}' not found.")
        return {"artifact": output}

    @app.delete("/api/images/outputs/{artifact_id}")
    def delete_image_output(artifact_id: str) -> dict[str, Any]:
        deleted = _delete_image_output(artifact_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Image output '{artifact_id}' not found.")
        app.state.chaosengine.add_log("images", "info", f"Deleted image output {artifact_id}.")
        return {"deleted": artifact_id, "outputs": _load_image_outputs()}

    @app.get("/api/models/search")
    def search_models(query: str = Query("", alias="q", min_length=0, max_length=120)) -> dict[str, Any]:
        system_stats = app.state.chaosengine._system_snapshot_provider()
        library = app.state.chaosengine._library()
        catalog = _model_family_payloads(system_stats, library)
        haystack = query.strip().lower()
        if not haystack:
            results = catalog
        else:
            results = [
                family
                for family in catalog
                if haystack in family["name"].lower()
                or haystack in family["provider"].lower()
                or any(haystack in capability for capability in family["capabilities"])
                or any(
                    haystack in variant["name"].lower()
                    or haystack in variant["format"].lower()
                    or haystack in variant["quantization"].lower()
                    or haystack in variant["repo"].lower()
                    for variant in family["variants"]
                )
            ]

        # Also search HuggingFace Hub when there's a query
        hub_results: list[dict[str, Any]] = []
        if haystack and len(haystack) >= 2:
            hub_results = _search_huggingface_hub(haystack, library)

        return {"query": query, "results": results, "hubResults": hub_results}

    @app.get("/api/models/hub-search")
    def hub_search(query: str = Query("", alias="q", min_length=2, max_length=120)) -> dict[str, Any]:
        library = app.state.chaosengine._library()
        results = _search_huggingface_hub(query.strip().lower(), library)
        return {"query": query, "results": results}

    @app.get("/api/models/hub-files")
    def hub_files(repo: str = Query(min_length=3, max_length=200)) -> dict[str, Any]:
        if "/" not in repo:
            raise HTTPException(status_code=400, detail="Repo must be in `owner/name` format.")
        try:
            return _hub_repo_files(repo)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/server/logs/stream")
    async def stream_server_logs():
        import queue as _queue_mod
        state = app.state.chaosengine
        q = state.subscribe_logs()

        async def event_stream():
            try:
                # Send recent logs first (skip debug noise)
                for entry in reversed(list(state.logs)[-50:]):
                    if entry.get("level") == "debug":
                        continue
                    yield f"data: {json.dumps(entry)}\n\n"
                # Then stream new entries by polling the thread-safe queue
                while True:
                    try:
                        entry = q.get(block=False)
                        yield f"data: {json.dumps(entry)}\n\n"
                    except _queue_mod.Empty:
                        yield ": keepalive\n\n"
                        await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                pass
            finally:
                state.unsubscribe_logs(q)

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })

    @app.post("/api/chat/sessions")
    def create_session(request: CreateSessionRequest) -> dict[str, Any]:
        session = app.state.chaosengine.create_session(title=request.title)
        return {"session": session}

    @app.patch("/api/chat/sessions/{session_id}")
    def update_session(session_id: str, request: UpdateSessionRequest) -> dict[str, Any]:
        session = app.state.chaosengine.update_session(session_id, request)
        return {"session": session}

    @app.post("/api/chat/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        return app.state.chaosengine.generate(request)

    @app.post("/api/chat/generate/stream")
    def generate_stream(request: GenerateRequest):
        return app.state.chaosengine.generate_stream(request)

    @app.get("/api/cache/preview")
    def cache_preview(
        bits: int = Query(3, ge=1, le=4),
        fp16_layers: int = Query(4, ge=0, le=16),
        num_layers: int = Query(32, ge=1, le=160),
        num_heads: int = Query(32, ge=1, le=256),
        hidden_size: int = Query(4096, ge=256, le=32768),
        context_tokens: int = Query(8192, ge=256, le=262144),
        params_b: float = Query(7.0, ge=0.5, le=1000.0),
    ) -> dict[str, Any]:
        system_stats = _build_system_snapshot()
        return compute_cache_preview(
            bits=bits,
            fp16_layers=fp16_layers,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            context_tokens=context_tokens,
            params_b=params_b,
            system_stats=system_stats,
        )

    @app.get("/api/server/status")
    def server_status() -> dict[str, Any]:
        return app.state.chaosengine.server_status()

    @app.post("/api/server/shutdown")
    def shutdown_server() -> dict[str, Any]:
        app.state.chaosengine.add_log("server", "info", "Shutdown requested via API.")
        # Schedule a graceful shutdown after responding
        import threading
        def _delayed_shutdown():
            time.sleep(0.5)
            sig = signal.SIGTERM if hasattr(signal, "SIGTERM") else signal.SIGINT
            os.kill(os.getpid(), sig)
        threading.Thread(target=_delayed_shutdown, daemon=True).start()
        return {"status": "shutting_down"}

    @app.get("/v1/models")
    def list_openai_models() -> dict[str, Any]:
        return app.state.chaosengine.openai_models()

    @app.post("/v1/chat/completions")
    def openai_chat_completion(request: OpenAIChatCompletionRequest):
        return app.state.chaosengine.openai_chat_completion(request)

    return app


app = create_app()


def _watch_parent_and_exit():
    """Exit if our parent process dies (e.g. Tauri shell killed via Ctrl+C).

    This prevents orphaned backend + MLX worker processes from holding
    GPU memory after the desktop app shuts down.
    """
    import threading
    initial_ppid = os.getppid()
    if initial_ppid <= 1:
        return  # Already orphaned or running standalone

    def _watcher():
        while True:
            time.sleep(2)
            current_ppid = os.getppid()
            if current_ppid != initial_ppid or current_ppid == 1:
                # Parent died — kill ourselves and any subprocess children
                try:
                    if hasattr(os, "killpg"):
                        # Unix: kill our entire process group (includes MLX worker children)
                        os.killpg(os.getpgrp(), signal.SIGTERM)
                    else:
                        # Windows: terminate our own process
                        os.kill(os.getpid(), signal.SIGTERM)
                except Exception:
                    pass
                os._exit(0)

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()


def main() -> None:
    import uvicorn

    # Watch for parent death so we don't orphan ourselves
    _watch_parent_and_exit()

    uvicorn.run(
        "backend_service.app:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
