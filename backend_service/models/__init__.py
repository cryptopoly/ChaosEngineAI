from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
    contextTokens: int = Field(default=8192, ge=256, le=2097152)
    speculativeDecoding: bool = False
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


class CreateSessionRequest(BaseModel):
    title: str | None = None


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    model: str | None = None
    modelRef: str | None = None
    modelSource: str | None = None
    modelPath: str | None = None
    modelBackend: str | None = None
    thinkingMode: Literal["off", "auto"] | None = None
    pinned: bool | None = None
    messages: list[dict[str, Any]] | None = None


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
    thinkingMode: Literal["off", "auto"] | None = None
    systemPrompt: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    maxTokens: int = Field(default=4096, ge=1, le=32768)
    cacheStrategy: str | None = None
    cacheBits: int | None = Field(default=None, ge=0, le=8)
    fp16Layers: int | None = Field(default=None, ge=0, le=16)
    fusedAttention: bool | None = None
    fitModelInMemory: bool | None = None
    contextTokens: int | None = Field(default=None, ge=256, le=2097152)
    speculativeDecoding: bool | None = None
    # Agent tool-use
    enableTools: bool = False
    availableTools: list[str] | None = None  # None = all registered tools


class RemoteProviderRequest(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    apiBase: str = Field(min_length=8, max_length=512)
    apiKey: str = Field(default="", max_length=512)
    model: str = Field(min_length=1, max_length=200)
    providerType: str = "openai"


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
