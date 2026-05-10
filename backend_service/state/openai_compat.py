"""OpenAI-compatible API surface for ``ChaosEngineState``.

Three endpoints lifted out of ``state/__init__.py``:

* ``openai_models`` — ``/v1/models`` shape; lists the loaded model +
  any ``runtimeTarget`` alias so external scripts can probe before
  calling completions.
* ``openai_embeddings`` — ``/v1/embeddings`` (Phase 2.13) routed
  through the bundled GGUF embedding client.
* ``openai_chat_completion`` — ``/v1/chat/completions`` with both
  streaming and non-streaming branches. Auto-loads the requested
  model when none is loaded; maps OpenAI's sampler + response_format
  envelopes into the runtime sampler dict + JSON-schema constrained
  decode path.

All three take the ``ChaosEngineState`` instance as their first
argument. The class methods become thin wrappers.

Extracted as part of the v0.8.0 Phase 1a-6 refactor.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from backend_service.models import (
    LoadModelRequest,
    OpenAIChatCompletionRequest,
    OpenAIEmbeddingsRequest,
)


if TYPE_CHECKING:
    from backend_service.state import ChaosEngineState


def openai_models(state: ChaosEngineState) -> dict[str, Any]:
    runtime = state.runtime.status(
        active_requests=state.active_requests,
        requests_served=state.requests_served,
    )
    loaded = runtime["loadedModel"]
    if loaded is None:
        return {"object": "list", "data": []}
    created = int(time.time())
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


def openai_embeddings(
    state: ChaosEngineState, request: OpenAIEmbeddingsRequest
) -> dict[str, Any]:
    """Phase 2.13: OpenAI-compatible embeddings endpoint.

    Routes through the bundled GGUF embedding model (Phase 2.6).
    Returns a 503 when no embedding client is available; returns
    the OpenAI-shaped response shape on success so external
    scripts can drop us in for OpenAI without code changes.
    """
    from backend_service.app import DOCUMENTS_DIR
    from backend_service.rag import resolve_embedding_client
    from backend_service.rag.embedding_client import EmbeddingClientUnavailable

    client = resolve_embedding_client(DOCUMENTS_DIR.parent)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No embedding model is configured. Set CHAOSENGINE_EMBEDDING_MODEL "
                "or drop a *.gguf into <dataDir>/embeddings/."
            ),
        )

    if isinstance(request.input, str):
        inputs = [request.input]
    else:
        inputs = list(request.input)

    if not inputs:
        raise HTTPException(
            status_code=400,
            detail="`input` must be a non-empty string or list of strings.",
        )

    try:
        vectors = client.embed_batch(inputs)
    except EmbeddingClientUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Truncate per OpenAI's `dimensions` parameter when set. We don't
    # re-normalise after truncation; the bundled model is already
    # L2-normalised end-to-end, so cosine similarity stays well-defined.
    if request.dimensions is not None:
        vectors = [vec[: request.dimensions] for vec in vectors]

    prompt_tokens = sum(max(1, len(text.split())) for text in inputs)
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": vec,
                "index": idx,
            }
            for idx, vec in enumerate(vectors)
        ],
        "model": request.model or "chaosengine-embed",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }


def openai_chat_completion(
    state: ChaosEngineState, request: OpenAIChatCompletionRequest
) -> dict[str, Any] | StreamingResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")

    last_user = None
    last_user_images: list[str] = []
    history: list[dict[str, Any]] = []
    system_prompt = None
    for message in request.messages:
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
            if message.tool_calls:
                history.append({"role": "assistant", "text": content, "tool_calls": message.tool_calls})
            else:
                history.append({"role": "assistant", "text": content})
        elif message.role == "tool":
            history.append({"role": "tool", "text": content, "tool_call_id": message.tool_call_id})

    if last_user is None:
        raise HTTPException(status_code=400, detail="A user message is required.")

    msg_count = len(request.messages)

    with state._lock:
        launch_preferences = state._launch_preferences()
        if state.runtime.loaded_model is None and request.model:
            state.add_log(
                "server", "info",
                f"[{request.model}] Auto-loading model for /v1/chat/completions...",
            )
            state.load_model(
                LoadModelRequest(
                    modelRef=request.model,
                    modelName=request.model,
                    canonicalRepo=state._resolve_canonical_repo(
                        model_ref=request.model,
                        path=None,
                        canonical_repo=None,
                    ),
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
        if state.runtime.loaded_model is None:
            raise HTTPException(
                status_code=409,
                detail="Load a model before calling /v1/chat/completions.",
            )

        try:
            target_engine, target_info = state.runtime.get_engine_for_request(request.model)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        state.active_requests += 1
        model_ref = target_info.ref
        model_tag = target_info.name
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        state.add_log(
            "server", "info",
            f"[{model_tag}] Running chat completion on conversation with {msg_count} messages.",
        )

    # Phase 2.13: build a sampler dict from OpenAI-shaped fields. The
    # runtime accepts the same llama-server key names so we map field
    # → key here once and pass the dict to both stream + non-stream
    # paths. None values drop out so they don't override server
    # defaults.
    oai_samplers: dict[str, Any] = {}
    if request.top_p is not None:
        oai_samplers["top_p"] = request.top_p
    if request.top_k is not None:
        oai_samplers["top_k"] = request.top_k
    if request.frequency_penalty is not None:
        oai_samplers["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        oai_samplers["presence_penalty"] = request.presence_penalty
    if request.seed is not None:
        oai_samplers["seed"] = request.seed
    if request.stop is not None:
        oai_samplers["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]

    # Phase 2.13: pull a JSON schema out of OpenAI's response_format
    # envelope so the constrained-decode path lights up. Anything
    # other than `json_schema` → no constraint (json_object would
    # require a different code path llama-server already handles
    # via response_format= but we don't surface that here).
    oai_json_schema: dict[str, Any] | None = None
    if isinstance(request.response_format, dict):
        rf_type = request.response_format.get("type")
        if rf_type == "json_schema":
            schema_envelope = request.response_format.get("json_schema") or {}
            schema_obj = schema_envelope.get("schema")
            if isinstance(schema_obj, dict):
                oai_json_schema = schema_obj

    if request.stream:
        def _stream_chunks():
            stream_start = time.perf_counter()
            with state._lock:
                state.add_log("server", "info", f"[{model_tag}] Streaming response...")
            token_count = 0
            prompt_tokens = 0
            try:
                first = True
                for chunk in state.runtime.stream_generate(
                    prompt=last_user,
                    history=history[:-1],
                    system_prompt=system_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    images=last_user_images or None,
                    tools=request.tools,
                    engine=target_engine,
                    samplers=oai_samplers or None,
                    json_schema=oai_json_schema,
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
                with state._lock:
                    state.add_log("server", "error", f"[{model_tag}] Streaming failed: {exc}")
            finally:
                elapsed = round(time.perf_counter() - stream_start, 2)
                tok_s = round(token_count / elapsed, 1) if elapsed > 0 else 0
                with state._lock:
                    state.active_requests = max(0, state.active_requests - 1)
                    state.requests_served += 1
                    state.add_log(
                        "server", "info",
                        f"[{model_tag}] Finished streaming response -- {token_count} tokens in {elapsed}s "
                        f"({tok_s} tok/s{f', {prompt_tokens} prompt tokens' if prompt_tokens else ''}).",
                    )

        return StreamingResponse(
            _stream_chunks(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    with state._lock:
        state.add_log("server", "info", f"[{model_tag}] Generating response...")
    gen_start = time.perf_counter()
    try:
        result = state.runtime.generate(
            prompt=last_user,
            history=history[:-1],
            system_prompt=system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            images=last_user_images or None,
            tools=request.tools,
            engine=target_engine,
            samplers=oai_samplers or None,
            json_schema=oai_json_schema,
        )
    except RuntimeError as exc:
        with state._lock:
            state.active_requests = max(0, state.active_requests - 1)
            state.add_log("server", "error", f"[{model_tag}] Generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    gen_elapsed = round(time.perf_counter() - gen_start, 2)
    with state._lock:
        state.active_requests = max(0, state.active_requests - 1)
        state.requests_served += 1
        state.add_log(
            "server", "info",
            f"[{model_tag}] Finished response -- {result.completionTokens} tokens in {gen_elapsed}s "
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
