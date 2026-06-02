"""Ollama-compatible API shim (#3).

A large slice of the local-AI tool ecosystem (Open WebUI, Continue.dev,
Raycast, n8n, Obsidian plugins, …) ships an "Ollama" connection preset
that speaks Ollama's *native* HTTP shape, not OpenAI's ``/v1``. This
module serves the native endpoints on the same backend so those tools
work against ChaosEngineAI with zero code on their side — point the
app's Ollama host at our base URL.

Implementation strategy: translate each Ollama request into the existing
``OpenAIChatCompletionRequest`` / ``OpenAIEmbeddingsRequest`` and reuse
``state.openai_chat_completion`` / ``state.openai_embeddings`` so all of
the auto-load, engine-resolution, sampler, tool, and JSON-schema logic is
inherited unchanged. The only Ollama-specific work is wire-format
translation:

* OpenAI streams **SSE** (``data: {json}\\n\\n`` … ``data: [DONE]``).
* Ollama streams **NDJSON** (one JSON object per line, terminated by an
  object with ``"done": true``).

Because we *produce* the SSE in ``state/openai_compat.py``, parsing it
back is deterministic.

Auth: these routes sit under ``/api`` and inherit the same bearer-token
middleware as ``/v1`` (the Server tab's "Require API token" toggle gates
both), so no per-route auth handling is needed here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from backend_service.models import (
    OpenAIChatCompletionRequest,
    OpenAIEmbeddingsRequest,
    OpenAIMessage,
)

router = APIRouter()


# --------------------------------------------------------------------------
# Request bodies (only the fields we consume; Ollama clients send more).
# --------------------------------------------------------------------------


class OllamaChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = []
    stream: bool = True  # Ollama defaults to streaming
    options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    format: Any = None  # "json" or a JSON-schema object
    keep_alive: Any = None


class OllamaGenerateRequest(BaseModel):
    model: str | None = None
    prompt: str = ""
    system: str | None = None
    stream: bool = True
    options: dict[str, Any] | None = None
    format: Any = None
    keep_alive: Any = None


class OllamaEmbeddingsRequest(BaseModel):
    # Legacy /api/embeddings — single prompt, returns {"embedding": [...]}.
    model: str | None = None
    prompt: str = ""


class OllamaEmbedRequest(BaseModel):
    # New /api/embed — single string or list, returns {"embeddings": [[...]]}.
    model: str | None = None
    input: str | list[str] = ""


class OllamaShowRequest(BaseModel):
    model: str | None = None
    name: str | None = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_to_response_format(fmt: Any) -> dict[str, Any] | None:
    """Map Ollama's ``format`` onto OpenAI's ``response_format`` envelope.

    ``"json"`` → a permissive object schema; a dict → used verbatim as the
    JSON schema. Both light up the existing constrained-decode path.
    """
    if fmt == "json":
        return {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
    if isinstance(fmt, dict) and fmt:
        return {"type": "json_schema", "json_schema": {"schema": fmt}}
    return None


def _build_openai_request(
    *,
    model: str | None,
    messages: list[dict[str, Any]],
    stream: bool,
    options: dict[str, Any] | None,
    tools: list[dict[str, Any]] | None,
    fmt: Any,
) -> OpenAIChatCompletionRequest:
    """Translate an Ollama chat body into an OpenAIChatCompletionRequest.

    Only options that are present are forwarded; everything else falls
    through to the request model's defaults so we never override a runtime
    default with a guess.
    """
    opts = options or {}
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [OpenAIMessage(role=str(m.get("role", "user")), content=m.get("content", "")) for m in messages],
        "stream": stream,
    }
    if "temperature" in opts and opts["temperature"] is not None:
        kwargs["temperature"] = float(opts["temperature"])
    if "num_predict" in opts and opts["num_predict"] is not None:
        # Ollama's -1/-2 mean "unbounded"; the OpenAI model requires a
        # positive int, so only forward sensible positive caps.
        try:
            n = int(opts["num_predict"])
            if n > 0:
                kwargs["max_tokens"] = n
        except (TypeError, ValueError):
            pass
    if opts.get("top_p") is not None:
        kwargs["top_p"] = float(opts["top_p"])
    if opts.get("top_k") is not None:
        kwargs["top_k"] = int(opts["top_k"])
    if opts.get("seed") is not None:
        kwargs["seed"] = int(opts["seed"])
    if opts.get("stop") is not None:
        kwargs["stop"] = opts["stop"]
    if tools:
        kwargs["tools"] = tools
    rf = _format_to_response_format(fmt)
    if rf is not None:
        kwargs["response_format"] = rf
    return OpenAIChatCompletionRequest(**kwargs)


async def _iter_sse_events(body_iterator) -> Any:
    """Yield decoded SSE payload strings from an OpenAI StreamingResponse.

    Buffers across chunks and splits on the ``\\n\\n`` event boundary so
    we're robust to whatever chunking / bytes-vs-str the underlying
    response uses. Yields the part after ``data: `` for each event.
    """
    buffer = ""
    async for raw in body_iterator:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        buffer += raw
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    yield line[len("data:"):].strip()


def _ollama_stream(openai_response: StreamingResponse, *, model: str, mode: str) -> StreamingResponse:
    """Wrap an OpenAI SSE StreamingResponse as Ollama NDJSON.

    ``mode`` is ``"chat"`` (emit ``message.content`` deltas) or
    ``"generate"`` (emit ``response`` deltas).
    """

    async def ndjson():
        finish_reason = "stop"
        try:
            async for payload in _iter_sse_events(openai_response.body_iterator):
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                if content:
                    if mode == "chat":
                        line = {
                            "model": model,
                            "created_at": _now_rfc3339(),
                            "message": {"role": "assistant", "content": content},
                            "done": False,
                        }
                    else:
                        line = {
                            "model": model,
                            "created_at": _now_rfc3339(),
                            "response": content,
                            "done": False,
                        }
                    yield json.dumps(line) + "\n"
        finally:
            if mode == "chat":
                final = {
                    "model": model,
                    "created_at": _now_rfc3339(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": finish_reason,
                }
            else:
                final = {
                    "model": model,
                    "created_at": _now_rfc3339(),
                    "response": "",
                    "done": True,
                    "done_reason": finish_reason,
                    "context": [],
                }
            yield json.dumps(final) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@router.get("/api/version")
def ollama_version() -> dict[str, Any]:
    from backend_service.helpers.system_hardware import _resolve_app_version  # noqa: PLC0415

    return {"version": _resolve_app_version()}


@router.get("/api/tags")
def ollama_tags(request: Request) -> dict[str, Any]:
    """List available models in Ollama's ``/api/tags`` shape."""
    state = request.app.state.chaosengine
    models = state.openai_models().get("data", [])
    now = _now_rfc3339()
    return {
        "models": [
            {
                "name": m["id"],
                "model": m["id"],
                "modified_at": now,
                "size": 0,
                "digest": "",
                "details": {
                    "family": "",
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }
            for m in models
        ]
    }


@router.post("/api/show")
def ollama_show(request: Request, body: OllamaShowRequest) -> dict[str, Any]:
    """Minimal ``/api/show`` — enough fields for clients that probe before chatting."""
    name = body.model or body.name or ""
    return {
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": {"family": "", "parameter_size": "", "quantization_level": ""},
        "model_info": {},
        "modified_at": _now_rfc3339(),
        "model": name,
    }


@router.post("/api/chat")
def ollama_chat(request: Request, body: OllamaChatRequest):
    state = request.app.state.chaosengine
    oai_req = _build_openai_request(
        model=body.model,
        messages=body.messages,
        stream=body.stream,
        options=body.options,
        tools=body.tools,
        fmt=body.format,
    )
    result = state.openai_chat_completion(oai_req)
    model_label = body.model or "chaosengine"
    if isinstance(result, StreamingResponse):
        return _ollama_stream(result, model=model_label, mode="chat")
    # Non-streaming dict → single Ollama chat object.
    choice = (result.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = result.get("usage") or {}
    return {
        "model": result.get("model", model_label),
        "created_at": _now_rfc3339(),
        "message": {"role": "assistant", "content": msg.get("content", "")},
        "done": True,
        "done_reason": choice.get("finish_reason", "stop"),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


@router.post("/api/generate")
def ollama_generate(request: Request, body: OllamaGenerateRequest):
    state = request.app.state.chaosengine
    messages: list[dict[str, Any]] = []
    if body.system:
        messages.append({"role": "system", "content": body.system})
    messages.append({"role": "user", "content": body.prompt})
    oai_req = _build_openai_request(
        model=body.model,
        messages=messages,
        stream=body.stream,
        options=body.options,
        tools=None,
        fmt=body.format,
    )
    result = state.openai_chat_completion(oai_req)
    model_label = body.model or "chaosengine"
    if isinstance(result, StreamingResponse):
        return _ollama_stream(result, model=model_label, mode="generate")
    choice = (result.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = result.get("usage") or {}
    return {
        "model": result.get("model", model_label),
        "created_at": _now_rfc3339(),
        "response": msg.get("content", ""),
        "done": True,
        "done_reason": choice.get("finish_reason", "stop"),
        "context": [],
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


@router.post("/api/embeddings")
def ollama_embeddings(request: Request, body: OllamaEmbeddingsRequest) -> dict[str, Any]:
    """Legacy single-prompt embeddings → ``{"embedding": [...]}``."""
    state = request.app.state.chaosengine
    result = state.openai_embeddings(OpenAIEmbeddingsRequest(model=body.model, input=body.prompt))
    data = result.get("data") or []
    if not data:
        raise HTTPException(status_code=500, detail="embedding produced no vector")
    return {"embedding": data[0]["embedding"]}


@router.post("/api/embed")
def ollama_embed(request: Request, body: OllamaEmbedRequest) -> dict[str, Any]:
    """New batch embeddings → ``{"model", "embeddings": [[...], ...]}``."""
    state = request.app.state.chaosengine
    result = state.openai_embeddings(OpenAIEmbeddingsRequest(model=body.model, input=body.input))
    data = result.get("data") or []
    return {
        "model": body.model or "chaosengine-embed",
        "embeddings": [row["embedding"] for row in data],
    }
