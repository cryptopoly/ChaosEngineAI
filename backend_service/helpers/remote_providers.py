"""Remote provider engine implementations for Anthropic, Groq, and Azure."""
from __future__ import annotations
import json
import urllib.request
import urllib.error
from typing import Any
from dataclasses import dataclass

@dataclass
class RemoteGenerationResult:
    text: str
    promptTokens: int = 0
    completionTokens: int = 0
    totalTokens: int = 0
    tokS: float = 0.0
    finishReason: str = "stop"
    runtimeNote: str | None = None
    tool_calls: list | None = None
    responseSeconds: float = 0.0

class AnthropicProvider:
    """Call Anthropic Messages API via urllib (no SDK dependency)."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"

    def generate(self, *, prompt: str, history: list, system_prompt: str | None,
                 max_tokens: int, temperature: float, **kwargs) -> RemoteGenerationResult:
        messages = []
        for msg in history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("text", "")})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_prompt:
            body["system"] = system_prompt

        data = json.dumps(body).encode()
        req = urllib.request.Request(self.base_url, data=data, headers={
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        })

        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())

        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = result.get("usage", {})
        return RemoteGenerationResult(
            text=text,
            promptTokens=usage.get("input_tokens", 0),
            completionTokens=usage.get("output_tokens", 0),
            totalTokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            finishReason=result.get("stop_reason", "stop"),
            runtimeNote=f"Anthropic {self.model}",
        )

class GroqProvider:
    """Call Groq API (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, *, prompt: str, history: list, system_prompt: str | None,
                 max_tokens: int, temperature: float, **kwargs) -> RemoteGenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("text", "")})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        data = json.dumps(body).encode()
        req = urllib.request.Request(self.base_url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())

        choice = result.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = result.get("usage", {})

        return RemoteGenerationResult(
            text=text,
            promptTokens=usage.get("prompt_tokens", 0),
            completionTokens=usage.get("completion_tokens", 0),
            totalTokens=usage.get("total_tokens", 0),
            finishReason=choice.get("finish_reason", "stop"),
            runtimeNote=f"Groq {self.model}",
        )
