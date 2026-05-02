from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LoadModelRequest(BaseModel):
    modelRef: str = Field(min_length=1)
    modelName: str | None = None
    canonicalRepo: str | None = None
    source: str = "catalog"
    backend: str = "auto"
    path: str | None = None
    cacheStrategy: str = "native"
    cacheBits: int = Field(default=0, ge=0, le=8)
    fp16Layers: int = Field(default=0, ge=0, le=16)
    fusedAttention: bool = False
    fitModelInMemory: bool = True
    contextTokens: int = Field(default=8192, ge=256, le=2097152)
    speculativeDecoding: bool = False
    treeBudget: int = Field(default=0, ge=0, le=64)
    adapterPath: str | None = None


class ModelDirectoryRequest(BaseModel):
    id: str | None = None
    label: str = Field(min_length=1, max_length=80)
    path: str = Field(min_length=1, max_length=4096)
    enabled: bool = True
    source: str = "user"


class LaunchPreferencesRequest(BaseModel):
    contextTokens: int = Field(default=8192, ge=256, le=2097152)
    maxTokens: int = Field(default=4096, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    cacheStrategy: str = "native"
    cacheBits: int = Field(default=0, ge=0, le=8)
    fp16Layers: int = Field(default=0, ge=0, le=16)
    fusedAttention: bool = False
    fitModelInMemory: bool = True
    speculativeDecoding: bool = False
    treeBudget: int = Field(default=0, ge=0, le=64)


class CreateSessionRequest(BaseModel):
    title: str | None = None


class AddVariantRequest(BaseModel):
    """Phase 2.5: generate a sibling variant of an assistant message.

    The frontend calls this after the user picks an alternate model
    from the assistant-message hover action. The chosen model must
    already be the loaded runtime (call /api/models/load first if
    needed). Backend runs a non-streaming generation using messages
    truncated to the prior user prompt, then attaches the result as
    a new entry on `messages[messageIndex].variants`.
    """

    messageIndex: int = Field(ge=0)
    modelRef: str = Field(min_length=1)
    modelName: str = Field(min_length=1)
    canonicalRepo: str | None = None
    source: str = "catalog"
    path: str | None = None
    backend: str = "auto"
    maxTokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ForkSessionRequest(BaseModel):
    """Phase 2.4: fork a thread at a specific assistant message.

    `forkAtMessageIndex` is the 0-based index of the last message to
    include in the fork — typically the assistant turn the user
    wants to branch from. The fork keeps every message up to and
    including this index, then becomes a fresh thread for divergent
    continuation.
    """

    forkAtMessageIndex: int = Field(ge=0)
    title: str | None = Field(default=None, max_length=200)


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    model: str | None = None
    modelRef: str | None = None
    canonicalRepo: str | None = None
    modelSource: str | None = None
    modelPath: str | None = None
    modelBackend: str | None = None
    thinkingMode: Literal["off", "auto"] | None = None
    reasoningEffort: Literal["low", "medium", "high"] | None = None
    pinned: bool | None = None
    cacheStrategy: str | None = None
    cacheBits: int | None = None
    fp16Layers: int | None = None
    fusedAttention: bool | None = None
    fitModelInMemory: bool | None = None
    contextTokens: int | None = None
    speculativeDecoding: bool | None = None
    treeBudget: int | None = None
    dflashDraftModel: str | None = None
    messages: list[dict[str, Any]] | None = None
    # Phase 3.7: assign / unassign a session to a workspace.
    # Pass empty string to clear; None leaves the value untouched.
    workspaceId: str | None = None


class GenerateRequest(BaseModel):
    sessionId: str | None = None
    title: str | None = None
    prompt: str = Field(min_length=1)
    images: list[str] | None = None  # base64-encoded images
    modelRef: str | None = None
    modelName: str | None = None
    canonicalRepo: str | None = None
    source: str = "catalog"
    path: str | None = None
    backend: str = "auto"
    thinkingMode: Literal["off", "auto"] | None = None
    # Phase 1.12: reasoning effort hint forwarded to OpenAI-compat
    # `reasoning_effort` chat-completion parameter on backends that respect it
    # (recent llama-server builds + several reasoning models). Backends that
    # ignore it remain unaffected. Null means no override.
    reasoningEffort: Literal["low", "medium", "high"] | None = None
    systemPrompt: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    maxTokens: int = Field(default=4096, ge=1, le=32768)
    # Optional per-message sampler overrides. None means "let backend default
    # apply" (llama.cpp / mlx-lm defaults). Phase 2.2 closes the Phase 1.10
    # deferral and exposes the full sampler chain end-to-end. Each backend
    # forwards what it supports and silently ignores the rest:
    #   - llama-server: all of these (native /v1/chat/completions params)
    #   - mlx-lm: temperature, topP, topK, minP, repeatPenalty, seed
    # DRY / XTC are intentionally deferred — DRY ships in llama-server but
    # is sensitive to context-length growth; XTC is too new to expose
    # broadly. Free-form GBNF grammars are skipped in favour of the safer
    # JSON-schema response format which covers most practical use cases.
    topP: float | None = Field(default=None, ge=0.0, le=1.0)
    topK: int | None = Field(default=None, ge=0, le=200)
    minP: float | None = Field(default=None, ge=0.0, le=1.0)
    repeatPenalty: float | None = Field(default=None, ge=0.0, le=2.0)
    # Mirostat: mode 0 = off, 1 = mirostat v1, 2 = mirostat v2. tau is the
    # target entropy; eta the learning rate. Pass None to use llama-server
    # defaults; pass mode=0 to explicitly disable on a model whose template
    # leaves it on.
    mirostatMode: Literal[0, 1, 2] | None = None
    mirostatTau: float | None = Field(default=None, ge=0.0, le=10.0)
    mirostatEta: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    # Constrained decoding: when set, llama-server enforces a JSON schema
    # via its `response_format: {type: "json_schema", json_schema: {...}}`
    # parameter. The shape mirrors the OpenAI structured-outputs spec.
    jsonSchema: dict[str, Any] | None = None
    cacheStrategy: str | None = None
    cacheBits: int | None = Field(default=None, ge=0, le=8)
    fp16Layers: int | None = Field(default=None, ge=0, le=16)
    fusedAttention: bool | None = None
    fitModelInMemory: bool | None = None
    contextTokens: int | None = Field(default=None, ge=256, le=2097152)
    speculativeDecoding: bool | None = None
    treeBudget: int | None = Field(default=None, ge=0, le=64)
    # Agent tool-use
    enableTools: bool = False
    availableTools: list[str] | None = None  # None = all registered tools
    # Phase 2.12: when True, the modelRef / canonicalRepo / source / etc.
    # in this request are treated as a one-turn override — the model
    # loads (or stays) for this turn, but the session's stored
    # `modelRef` / `model` / `canonicalRepo` / `modelSource` /
    # `modelPath` / `modelBackend` fields are NOT updated. The session
    # default sticks so the next plain message goes back to the
    # original model. Default False preserves the existing behaviour
    # where sending with a different model permanently switches the
    # thread.
    oneTurnOverride: bool = False


class RemoteProviderRequest(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    apiBase: str = Field(min_length=8, max_length=512)
    apiKey: str = Field(default="", max_length=512)
    model: str = Field(min_length=1, max_length=200)
    providerType: str = "openai"


class McpServerConfigRequest(BaseModel):
    """Phase 2.10: one MCP server entry for the settings payload.

    Maps onto `backend_service.mcp.McpServerConfig`. The shape mirrors
    the standard mcp-clients config blob (`command`, `args`, `env`) so
    config files copied from other MCP-aware tools work with minimal
    edits. `id` is a short opaque key surfaced on tool provenance
    badges.
    """

    id: str = Field(min_length=1, max_length=64)
    command: str = Field(min_length=1, max_length=512)
    args: list[str] | None = None
    env: dict[str, str] | None = None
    enabled: bool = True


class UpdateSettingsRequest(BaseModel):
    modelDirectories: list[ModelDirectoryRequest] | None = None
    preferredServerPort: int | None = Field(default=None, ge=1024, le=65535)
    allowRemoteConnections: bool | None = None
    requireApiAuth: bool | None = None
    autoStartServer: bool | None = None
    launchPreferences: LaunchPreferencesRequest | None = None
    remoteProviders: list[RemoteProviderRequest] | None = None
    # Phase 2.10: list of MCP servers to spawn at startup. Each entry's
    # `tools/list` output is merged into the agent tool registry with
    # `provenance: mcp:<id>` tags. None = leave existing list alone;
    # empty list = remove all configured servers.
    mcpServers: list[McpServerConfigRequest] | None = None
    huggingFaceToken: str | None = Field(default=None, max_length=512)
    dataDirectory: str | None = Field(default=None, max_length=4096)
    # Per-modality output overrides. Empty string clears the override and
    # restores the default (data-dir/images/outputs or data-dir/videos/outputs).
    imageOutputsDirectory: str | None = Field(default=None, max_length=4096)
    videoOutputsDirectory: str | None = Field(default=None, max_length=4096)
    # HF_HOME override — redirects every snapshot_download to a different
    # drive. Applied by the Tauri shell at backend spawn; requires restart
    # to take effect. Empty string clears the override.
    hfCachePath: str | None = Field(default=None, max_length=4096)


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
    # Phase 2.13: standard OpenAI sampler parameters. llama-server
    # supports them natively; mlx-lm consumes top_p / top_k / seed and
    # silently ignores the rest. Pass None to use the runtime default.
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0, le=200)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    stop: list[str] | str | None = None
    response_format: dict[str, Any] | None = None


class OpenAIEmbeddingsRequest(BaseModel):
    """Phase 2.13: OpenAI-shaped embeddings input.

    `input` accepts a single string or a list of strings, mirroring
    the OpenAI spec. The `model` field is informational — we use the
    bundled embedding GGUF regardless.
    """
    model: str | None = None
    input: str | list[str]
    encoding_format: Literal["float"] | None = "float"
    dimensions: int | None = Field(default=None, ge=8, le=8192)
    user: str | None = None


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
    contextTokens: int = Field(default=8192, ge=256, le=2097152)
    speculativeDecoding: bool = False
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


class DownloadModelRequest(BaseModel):
    repo: str = Field(min_length=3, max_length=256)
    modelId: str | None = Field(default=None, min_length=1, max_length=256)


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
    draftMode: bool = Field(default=False)
    sampler: str | None = Field(default=None, max_length=32)


class ImageRuntimePreloadRequest(BaseModel):
    modelId: str = Field(min_length=1, max_length=256)


class ImageRuntimeUnloadRequest(BaseModel):
    modelId: str | None = Field(default=None, min_length=1, max_length=256)


class VideoRuntimePreloadRequest(BaseModel):
    modelId: str = Field(min_length=1, max_length=256)


class VideoRuntimeUnloadRequest(BaseModel):
    modelId: str | None = Field(default=None, min_length=1, max_length=256)


class VideoGenerationRequest(BaseModel):
    """Shape accepted by POST /api/video/generate.

    Defaults are intentionally conservative — num_frames and steps in particular
    dominate generation time on consumer hardware, so we err on the side of a
    short, fast clip and let the user dial up quality from the Studio UI.
    """
    modelId: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=4000)
    negativePrompt: str | None = Field(default=None, max_length=4000)
    width: int = Field(default=768, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    numFrames: int = Field(default=97, ge=8, le=257)
    fps: int = Field(default=24, ge=1, le=60)
    steps: int = Field(default=50, ge=1, le=100)
    guidance: float = Field(default=3.0, ge=1.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, le=2147483647)
    # Smoothness post-processing. 1 = generated frames only; 2 or 4
    # inserts blended intermediates to raise the effective fps.
    interpolationFactor: int = Field(default=1, ge=1, le=4)
    # Diffusers scheduler override. ``"auto"`` (default) lets the runtime
    # pick the upstream-recommended scheduler for the chosen model
    # (Uni-PC for Wan, Euler for LTX, etc.). Explicit values short-circuit
    # the auto-pick. Recognised ids: ``unipc``, ``euler``, ``dpm++``,
    # ``ddim``. Anything else logs a warning and keeps the pipeline
    # default. See ``_VIDEO_PIPELINE_DEFAULTS`` in video_runtime for the
    # auto-pick table.
    scheduler: Literal["auto", "unipc", "euler", "flow-euler", "dpm++", "ddim"] | None = Field(default="auto")
    # bitsandbytes NF4 4-bit quantization for the video DiT transformer.
    # CUDA-only; ignored on MPS / CPU. Wan 2.1 14B drops from ~28 GB bf16
    # to ~7 GB on the RTX 4090 with negligible quality loss.
    useNf4: bool = Field(default=False)
    # LTX-Video two-stage spatial upscale via LTXLatentUpsamplePipeline.
    # Frame budget grows ~1.5×; runtimeNote surfaces the substitution.
    enableLtxRefiner: bool = Field(default=False)
    # Phase E1: append model-specific structural hints to short prompts
    # (< 25 words). Diffusion video models train against detailed prompts;
    # short user prompts under-condition the model and drift. Default on
    # so the typical short-prompt user gets quality uplift; explicit
    # ``false`` opts out for users who've crafted a long custom prompt
    # and want it sent verbatim.
    enhancePrompt: bool = Field(default=True)
    # Phase E2: CFG decay schedule. Flow-match video models (LTX-Video,
    # Wan, HunyuanVideo) benefit from higher CFG early in sampling
    # (locks semantic structure) and lower CFG late (preserves fine
    # detail, reduces oversaturation). When True, the engine decays
    # ``guidance_scale`` linearly from the user's setting at step 0
    # to 1.0 at the final step. Default-on for flow-match pipelines.
    cfgDecay: bool = Field(default=True)
