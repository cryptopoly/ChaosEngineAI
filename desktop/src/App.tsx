import { useDeferredValue, useEffect, useRef, useState, type KeyboardEvent } from "react";
import Markdown from "react-markdown";
import {
  cancelImageDownload,
  checkBackend,
  convertModel,
  createSession,
  deleteImageOutput,
  downloadImageModel,
  generateChat,
  generateImage,
  generateChatStream,
  getDownloadStatus,
  getImageCatalog,
  getImageDownloadStatus,
  getImageOutputs,
  getImageRuntime,
  preloadImageModel,
  downloadModel,
  cancelDownload,
  uploadSessionDocument,
  deleteSessionDocument,
  getTauriBackendInfo,
  getCachePreview,
  getWorkspace,
  loadModel,
  resolveApiBase,
  deleteModelPath,
  revealModelPath,
  restartManagedBackend,
  runBenchmark,
  listHubFiles,
  searchModels,
  shutdownServer,
  stopManagedBackend,
  unloadImageModel,
  unloadModel,
  updateSettings,
  updateSession,
} from "./api";
import type { DownloadStatus } from "./api";
import { checkForUpdates } from "./updater";
import { Panel } from "./components/Panel";
import { PerformancePreview } from "./components/PerformancePreview";
import { LiveProgress, type LiveProgressPhase } from "./components/LiveProgress";
import { ProgressRow } from "./components/ProgressRow";
import { StatCard } from "./components/StatCard";
import { ModelPicker } from "./components/ModelPicker";
import { RuntimeControls } from "./components/RuntimeControls";
import { SliderField } from "./components/SliderField";
import { mockWorkspace } from "./mockData";
import type {
  AppSettings,
  BenchmarkRunPayload,
  BenchmarkResult,
  ChatSession,
  ConversionResult,
  HubFileListResponse,
  HubModel,
  ImageModelFamily,
  ImageModelVariant,
  ImageOutputArtifact,
  ImageQualityPreset,
  ImageRuntimeStatus,
  LaunchPreferences,
  LibraryItem,
  ModelFamily,
  ModelDirectorySetting,
  ModelLoadingState,
  ModelVariant,
  PreviewMetrics,
  RuntimeStatus,
  TabId,
  TauriBackendInfo,
  WorkspaceData,
} from "./types";

function renderModelLoadingProgress(loading: ModelLoadingState) {
  const rawPct =
    typeof loading.progressPercent === "number" && !Number.isNaN(loading.progressPercent)
      ? loading.progressPercent
      : null;
  const pct = rawPct !== null ? Math.max(0, Math.min(100, rawPct)) : null;
  const phase = loading.progressPhase ?? null;
  const message = loading.progressMessage ?? null;
  const recentLogs = loading.recentLogLines ?? [];
  const hasProgress = pct !== null;
  return (
    <div className="model-loading-progress">
      <div className="loading-progress-bar">
        {hasProgress ? (
          <div
            className="loading-progress-bar-fill"
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div className="loading-progress-bar-fill loading-progress-bar-indeterminate" />
        )}
      </div>
      {hasProgress ? (
        <p className="loading-progress-label">
          {Math.round(pct as number)}%{phase ? ` - ${phase}` : ""}
          {message ? ` - ${message}` : ""}
          {" "}
          <span className="loading-progress-elapsed">({loading.elapsedSeconds}s)</span>
        </p>
      ) : (
        <p className="loading-progress-label">
          Loading {loading.modelName}... {loading.elapsedSeconds}s elapsed
        </p>
      )}
      {recentLogs.length > 0 ? (
        <div className="loading-recent-logs">
          {recentLogs.slice(-5).map((line, idx) => (
            <div key={idx} className="loading-recent-log-line">
              {line}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

const tabs: Array<{ id: TabId; label: string; caption: string }> = [
  { id: "dashboard", label: "Dashboard", caption: "System overview" },
  { id: "online-models", label: "Discover", caption: "Browse and download AI models" },
  { id: "my-models", label: "My Models", caption: "Models on this machine" },
  { id: "image-discover", label: "Image Discover", caption: "Curated local image models" },
  { id: "image-models", label: "Image Models", caption: "Installed image generators" },
  { id: "image-studio", label: "Image Studio", caption: "Prompt, generate, and iterate" },
  { id: "image-gallery", label: "Image Gallery", caption: "Saved outputs and filters" },
  { id: "server", label: "Server", caption: "OpenAI-compatible local API" },
  { id: "chat", label: "Chat", caption: "Local AI chat" },
  { id: "benchmarks", label: "Benchmarks", caption: "Run a new benchmark" },
  { id: "benchmark-history", label: "History", caption: "Compare saved runs" },
  { id: "conversion", label: "Conversion", caption: "Convert models to MLX format" },
  { id: "logs", label: "Logs", caption: "Runtime events" },
  { id: "settings", label: "Settings", caption: "Directories and defaults" },
];

const CAPABILITY_META: Record<string, { shortLabel: string; title: string; icon: string; color: string }> = {
  agents: { shortLabel: "Agents", title: "Agent workflows", icon: "\uD83E\uDD16", color: "#c084fc" },
  chat: { shortLabel: "Chat", title: "General chat", icon: "\uD83D\uDCAC", color: "#8fb4ff" },
  coding: { shortLabel: "Code", title: "Coding support", icon: "\uD83D\uDCBB", color: "#34d399" },
  multilingual: { shortLabel: "Multi", title: "Multilingual support", icon: "\uD83C\uDF10", color: "#fbbf24" },
  reasoning: { shortLabel: "Reason", title: "Reasoning-focused", icon: "\uD83E\uDDE0", color: "#f472b6" },
  thinking: { shortLabel: "Think", title: "Thinking / deliberate reasoning", icon: "\uD83D\uDCA1", color: "#facc15" },
  "tool-use": { shortLabel: "Tools", title: "Tool use / function calling", icon: "\uD83D\uDD27", color: "#fb923c" },
  video: { shortLabel: "Video", title: "Video understanding", icon: "\uD83C\uDFA5", color: "#f87171" },
  vision: { shortLabel: "Vision", title: "Image / vision support", icon: "\uD83D\uDC41\uFE0F", color: "#22d3ee" },
};

const IMAGE_RATIO_PRESETS = [
  { id: "square", label: "Square", hint: "1024 x 1024", width: 1024, height: 1024 },
  { id: "portrait", label: "Portrait", hint: "832 x 1216", width: 832, height: 1216 },
  { id: "landscape", label: "Landscape", hint: "1216 x 832", width: 1216, height: 832 },
  { id: "wide", label: "Wide", hint: "1344 x 768", width: 1344, height: 768 },
] as const;

const IMAGE_QUALITY_PRESETS: Array<{
  id: ImageQualityPreset;
  label: string;
  hint: string;
  steps: number;
  guidance: number;
}> = [
  { id: "fast", label: "Fast", hint: "Quick drafts", steps: 12, guidance: 4.5 },
  { id: "balanced", label: "Balanced", hint: "Best default", steps: 24, guidance: 6 },
  { id: "quality", label: "High Quality", hint: "Slower final pass", steps: 36, guidance: 7 },
];

function number(value: number, digits = 1) {
  return value.toFixed(digits);
}

function flattenImageVariants(families: ImageModelFamily[]): ImageModelVariant[] {
  return families.flatMap((family) => family.variants);
}

function defaultImageVariantForFamily(family?: ImageModelFamily | null): ImageModelVariant | null {
  if (!family) return null;
  return family.variants.find((variant) => variant.id === family.defaultVariantId) ?? family.variants[0] ?? null;
}

function findImageVariantById(families: ImageModelFamily[], variantId: string): ImageModelVariant | null {
  for (const family of families) {
    const match = family.variants.find((variant) => variant.id === variantId);
    if (match) return match;
  }
  return null;
}

function findImageVariantByRepo(families: ImageModelFamily[], repo: string | null | undefined): ImageModelVariant | null {
  if (!repo) return null;
  for (const family of families) {
    const match = family.variants.find((variant) => variant.repo === repo);
    if (match) return match;
  }
  return null;
}

function formatImageTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

type ImageGalleryRuntimeFilter = "all" | "diffusers" | "placeholder" | "warning";
type ImageGalleryOrientationFilter = "all" | "square" | "portrait" | "landscape";
type ImageGallerySort = "newest" | "oldest";
type ImageDiscoverSourceFilter = "all" | "curated" | "latest";
type ImageDiscoverTaskFilter = "all" | "txt2img" | "img2img" | "inpaint";
type ImageDiscoverAccessFilter = "all" | "open" | "gated";

function imageRuntimeKind(label?: string | null) {
  const lowered = (label ?? "").toLowerCase();
  if (lowered.includes("placeholder")) return "placeholder";
  if (lowered.includes("diffusers")) return "diffusers";
  return "other";
}

function imageOrientation(width: number, height: number): Exclude<ImageGalleryOrientationFilter, "all"> {
  if (width === height) return "square";
  return width > height ? "landscape" : "portrait";
}

function imageArtifactTimestamp(artifact: ImageOutputArtifact) {
  const timestamp = Date.parse(artifact.createdAt);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

function serverOriginFromBase(baseUrl: string) {
  return baseUrl.replace(/\/v1\/?$/, "");
}

function serverBaseFromOrigin(origin: string) {
  return `${origin.replace(/\/$/, "")}/v1`;
}

function upsertSession(sessions: ChatSession[], nextSession: ChatSession): ChatSession[] {
  return [nextSession, ...sessions.filter((session) => session.id !== nextSession.id)];
}

function sessionPreview(session: ChatSession) {
  return session.messages[session.messages.length - 1]?.text ?? "No messages yet";
}

function sortSessions(sessions: ChatSession[]) {
  return [...sessions].sort((left, right) => {
    if (Boolean(left.pinned) !== Boolean(right.pinned)) {
      return left.pinned ? -1 : 1;
    }
    return 0;
  });
}

function parseContextK(ctx: string | undefined | null): number {
  if (!ctx) return 0;
  const upper = ctx.toUpperCase();
  if (upper.endsWith("M")) return parseFloat(ctx) * 1000;
  if (upper.endsWith("K")) return parseFloat(ctx);
  return parseFloat(ctx) / 1024;
}

function syncRuntime(current: WorkspaceData, runtime: RuntimeStatus): WorkspaceData {
  return {
    ...current,
    runtime,
    server: {
      ...current.server,
      status: runtime.serverReady ? "running" : "idle",
      activeConnections: runtime.activeRequests,
      concurrentRequests: runtime.activeRequests,
      requestsServed: runtime.requestsServed,
      loadedModelName: runtime.loadedModel?.name ?? null,
      loading: runtime.loadedModel ? null : current.server.loading,
    },
  };
}

function syncStoppedBackend(current: WorkspaceData, runtimeInfo: TauriBackendInfo | null): WorkspaceData {
  const origin = runtimeInfo?.apiBase ?? serverOriginFromBase(current.server.localhostUrl ?? current.server.baseUrl);
  const localhostUrl = serverBaseFromOrigin(origin);
  return {
    ...current,
    runtime: {
      ...current.runtime,
      state: "idle",
      loadedModel: null,
      supportsGeneration: false,
      serverReady: false,
      activeRequests: 0,
      runtimeNote: "The local API service is stopped.",
    },
    server: {
      ...current.server,
      status: "idle",
      baseUrl: localhostUrl,
      localhostUrl,
      activeConnections: 0,
      concurrentRequests: 0,
      loadedModelName: null,
      loading: null,
      remoteAccessActive: false,
      logTail: [
        "API service stopped.",
        "Use Restart to bring the local API back online.",
        ...(runtimeInfo?.startupError ? [runtimeInfo.startupError] : []),
      ].slice(0, 3),
    },
  };
}

function flattenVariants(families: ModelFamily[]): ModelVariant[] {
  return families.flatMap((family) => family.variants);
}

function normalizeCapability(capability: string): string {
  return capability.trim().toLowerCase().replace(/\s+/g, "-");
}

function capabilityMeta(capability: string) {
  const normalized = normalizeCapability(capability);
  return (
    CAPABILITY_META[normalized] ?? {
      shortLabel: normalized.slice(0, 4).toUpperCase(),
      title: capability,
    }
  );
}

function sizeLabel(sizeGb: number) {
  return sizeGb > 0 ? `${number(sizeGb)} GB` : "Unknown";
}

function formatImageLicenseLabel(value?: string | null) {
  if (!value) return null;
  return value
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function imagePrimarySizeLabel(variant: ImageModelVariant) {
  if (typeof variant.coreWeightsGb === "number" && variant.coreWeightsGb > 0) {
    return `${sizeLabel(variant.coreWeightsGb)} weights`;
  }
  if (typeof variant.repoSizeGb === "number" && variant.repoSizeGb > 0) {
    return `${sizeLabel(variant.repoSizeGb)} download`;
  }
  return sizeLabel(variant.sizeGb);
}

function imageSecondarySizeLabel(variant: ImageModelVariant) {
  if (
    typeof variant.repoSizeGb === "number" &&
    variant.repoSizeGb > 0 &&
    typeof variant.coreWeightsGb === "number" &&
    variant.coreWeightsGb > 0 &&
    Math.abs(variant.repoSizeGb - variant.coreWeightsGb) > 0.2
  ) {
    return `${sizeLabel(variant.repoSizeGb)} full repo`;
  }
  return null;
}

function imageVariantMatchesDiscoverFilters(
  variant: ImageModelVariant,
  taskFilter: ImageDiscoverTaskFilter,
  accessFilter: ImageDiscoverAccessFilter,
) {
  if (taskFilter !== "all" && !variant.taskSupport.includes(taskFilter)) {
    return false;
  }
  if (accessFilter === "open" && variant.gated === true) {
    return false;
  }
  if (accessFilter === "gated" && variant.gated !== true) {
    return false;
  }
  return true;
}

function imageDiscoverVariantHaystack(variant: ImageModelVariant) {
  return [
    variant.name,
    variant.familyName ?? "",
    variant.provider,
    variant.repo,
    variant.runtime,
    variant.recommendedResolution,
    variant.note,
    variant.updatedLabel ?? "",
    variant.license ?? "",
    variant.pipelineTag ?? "",
    variant.downloadsLabel ?? "",
    variant.likesLabel ?? "",
    String(variant.sizeGb),
    String(variant.repoSizeGb ?? ""),
    String(variant.coreWeightsGb ?? ""),
    variant.gated ? "gated access" : "open access",
    variant.taskSupport.join(" "),
    variant.styleTags.join(" "),
  ]
    .join(" ")
    .toLowerCase();
}

function imageDiscoverFamilyHaystack(family: ImageModelFamily) {
  return [
    family.name,
    family.provider,
    family.headline,
    family.summary,
    family.updatedLabel,
    family.badges.join(" "),
    ...family.variants.map((variant) => imageDiscoverVariantHaystack(variant)),
  ]
    .join(" ")
    .toLowerCase();
}

function imageDiscoverVariantMatchesQuery(variant: ImageModelVariant, query: string) {
  if (!query) return true;
  return imageDiscoverVariantHaystack(variant).includes(query);
}

function imageDiscoverFamilyMatchesQuery(family: ImageModelFamily, query: string) {
  if (!query) return true;
  return imageDiscoverFamilyHaystack(family).includes(query);
}

function buildDownloadStatusMap(statuses: DownloadStatus[]): Record<string, DownloadStatus> {
  return Object.fromEntries(statuses.map((status) => [status.repo, status]));
}

function pendingDownloadStatus(repo: string, existing?: DownloadStatus | null): DownloadStatus {
  return {
    repo,
    state: "downloading",
    progress: Math.max(0, existing?.progress ?? 0),
    downloadedGb: Math.max(0, existing?.downloadedGb ?? 0),
    totalGb: typeof existing?.totalGb === "number" ? existing.totalGb : null,
    error: null,
  };
}

function failedDownloadStatus(repo: string, error: string): DownloadStatus {
  return {
    repo,
    state: "failed",
    progress: 0,
    downloadedGb: 0,
    totalGb: null,
    error,
  };
}

function downloadProgressLabel(download?: DownloadStatus | null) {
  if (!download) return "Preparing download...";
  const prefix = download.state === "cancelled" ? "Paused" : download.state === "downloading" ? "Downloading" : "";
  if (!prefix) return "";
  const totalGb = typeof download.totalGb === "number" && download.totalGb > 0 ? download.totalGb : null;
  const downloadedGb = Math.max(0, download.downloadedGb ?? 0);
  if (totalGb !== null) {
    const pct = Math.max(downloadedGb > 0 ? 1 : 0, Math.round((download.progress ?? 0) * 100));
    return `${prefix} ${pct}%`;
  }
  if (downloadedGb > 0) {
    return download.state === "cancelled" ? `${prefix} at ${number(downloadedGb)} GB` : `${prefix} ${number(downloadedGb)} GB`;
  }
  return download.state === "cancelled" ? "Paused" : "Preparing download...";
}

function imageRuntimeErrorStatus(error: unknown): ImageRuntimeStatus {
  return {
    activeEngine: "unavailable",
    realGenerationAvailable: false,
    message: error instanceof Error ? error.message : "Image runtime unavailable.",
    missingDependencies: [],
  };
}

function isGatedImageAccessError(message: string | null | undefined) {
  if (!message) return false;
  const lowered = message.toLowerCase();
  return (
    lowered.includes("cannot access gated repo")
    || lowered.includes("gated repo")
    || lowered.includes("authorized list")
    || (lowered.includes("access to model") && lowered.includes("restricted"))
  );
}

function formatImageAccessError(
  message: string | null | undefined,
  variant?: Pick<ImageModelVariant, "name" | "link"> | null,
) {
  if (!message) return "";
  if (!isGatedImageAccessError(message)) {
    return message;
  }
  return `${variant?.name ?? "This model"} is gated on Hugging Face. Your account or token is not approved for it yet. Open Hugging Face, request or accept access, add a read-enabled HF token in Settings, then retry.`;
}

// Estimate transformer architecture from parameter count (Llama-style scaling).
// Used to recompute the cache preview when a model is selected without explicit arch metadata.
function estimateArchFromParams(paramsB: number): { numLayers: number; hiddenSize: number; numHeads: number } {
  if (paramsB <= 1.5) return { numLayers: 22, hiddenSize: 2048, numHeads: 32 };
  if (paramsB <= 4) return { numLayers: 26, hiddenSize: 3072, numHeads: 24 };
  if (paramsB <= 9) return { numLayers: 32, hiddenSize: 4096, numHeads: 32 };
  if (paramsB <= 16) return { numLayers: 40, hiddenSize: 5120, numHeads: 40 };
  if (paramsB <= 35) return { numLayers: 60, hiddenSize: 6656, numHeads: 52 };
  if (paramsB <= 50) return { numLayers: 64, hiddenSize: 7168, numHeads: 56 };
  return { numLayers: 80, hiddenSize: 8192, numHeads: 64 };
}

// Estimate parameter count from on-disk size + bits-per-weight (when paramsB metadata is missing).
function estimateParamsBFromDisk(diskGb: number, bitsPerWeight: number): number {
  if (!diskGb || !bitsPerWeight) return 0;
  return (diskGb * 8) / bitsPerWeight;
}

// Detect bits-per-weight from a model name/format string (4bit, q4, bf16, etc).
function detectBitsPerWeight(haystack: string): number {
  const text = haystack.toLowerCase();
  const match = text.match(/(\d)[\s-]?bit|q(\d)/);
  if (match) {
    const bits = Number(match[1] ?? match[2]);
    if (bits >= 2 && bits <= 8) return bits + 0.5;
  }
  if (/bf16|fp16|float16|f16/.test(text)) return 16;
  if (/fp32|float32|f32/.test(text)) return 32;
  return 16;
}

function tokenSet(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length >= 3 && !["gguf", "mlx", "bf16", "fp8", "instruct", "community"].includes(token));
}

function normalizeQuantizationLabel(value: string | null | undefined): string {
  return (value ?? "").toLowerCase().replace(/[\s-]+/g, "");
}

function inferQuantizationLabel(text: string): string | null {
  const lowered = text.toLowerCase();
  const qMatch = lowered.match(/\b(q\d(?:_[a-z0-9]+)*)\b/);
  if (qMatch) return qMatch[1].toUpperCase();
  const bitMatch = lowered.match(/\b(\d+)[-_ ]?bit\b/);
  if (bitMatch) return `${bitMatch[1]}-bit`;
  if (/(^|[^a-z])bf16([^a-z]|$)|bfloat16/.test(lowered)) return "BF16";
  if (/fp16|float16|(^|[^a-z])f16([^a-z]|$)/.test(lowered)) return "FP16";
  if (/fp8|float8/.test(lowered)) return "FP8";
  if (/fp32|float32/.test(lowered)) return "FP32";
  return null;
}

function libraryItemSourceKind(item: LibraryItem): string {
  if (item.sourceKind) return item.sourceKind;
  if (item.path.includes("/models--")) return "HF cache";
  return /\.(gguf|safetensors)$/i.test(item.path) ? "File" : "Directory";
}

function libraryItemFormat(item: LibraryItem, matchedVariant?: ModelVariant | null): string {
  const explicit = (item.format ?? "").trim();
  if (explicit && explicit.toLowerCase() !== "hf cache" && explicit.toLowerCase() !== "unknown") {
    return explicit;
  }
  const haystack = `${item.name} ${item.path}`.toLowerCase();
  if (item.backend === "llama.cpp" || haystack.includes("gguf")) return "GGUF";
  if (explicit.toLowerCase() === "mlx" || /(^|[^a-z])mlx([^a-z]|$)|mlx-community/.test(haystack)) return "MLX";
  if (matchedVariant?.format) return matchedVariant.format;
  return explicit || "Unknown";
}

function libraryItemQuantization(item: LibraryItem, matchedVariant?: ModelVariant | null): string | null {
  return item.quantization ?? matchedVariant?.quantization ?? inferQuantizationLabel(`${item.name} ${item.path}`);
}

function libraryItemBackend(item: LibraryItem, matchedVariant?: ModelVariant | null): string {
  if (item.backend) return item.backend;
  const format = libraryItemFormat(item, matchedVariant).toLowerCase();
  if (format === "gguf") return "llama.cpp";
  if (matchedVariant?.backend) return matchedVariant.backend;
  return "mlx";
}

function libraryVariantMatchScore(item: LibraryItem, variant: ModelVariant): number {
  const haystack = `${item.name} ${item.path}`.toLowerCase();
  let score = 0;
  const exactCandidates = [variant.id, variant.repo, variant.name, variant.link]
    .map((value) => value.toLowerCase())
    .filter(Boolean);
  if (exactCandidates.some((candidate) => haystack.includes(candidate))) {
    score += 80;
  }

  const compactCandidates = exactCandidates
    .flatMap((candidate) => [candidate.split("/").pop() ?? "", candidate.replace(/\//g, "-")])
    .filter(Boolean);
  if (compactCandidates.some((candidate) => haystack.includes(candidate))) {
    score += 40;
  }

  const hits = tokenSet(`${variant.repo} ${variant.name}`).filter((token) => haystack.includes(token));
  score += hits.length * 6;

  const itemFormat = libraryItemFormat(item).toLowerCase();
  if (itemFormat && variant.format) {
    if (variant.format.toLowerCase() === itemFormat) score += 14;
    else if ((item.format ?? "").toLowerCase() !== "hf cache") score -= 6;
  }

  const itemQuant = normalizeQuantizationLabel(libraryItemQuantization(item));
  const variantQuant = normalizeQuantizationLabel(variant.quantization);
  if (itemQuant && variantQuant) {
    score += itemQuant === variantQuant ? 18 : -8;
  }

  if (haystack.includes("gguf")) {
    score += variant.format === "GGUF" ? 8 : -4;
  }
  if (haystack.includes("mlx")) {
    score += variant.format === "MLX" ? 8 : -4;
  }

  return score;
}

function libraryItemMatchesVariant(item: LibraryItem, variant: ModelVariant): boolean {
  return libraryVariantMatchScore(item, variant) >= 12;
}

function findLibraryItemForVariant(library: LibraryItem[], variant: ModelVariant): LibraryItem | null {
  let best: { item: LibraryItem; score: number } | null = null;
  for (const item of library) {
    const score = libraryVariantMatchScore(item, variant);
    if (!best || score > best.score) {
      best = { item, score };
    }
  }
  return best && best.score >= 12 ? best.item : null;
}

function findCatalogVariantForLibraryItem(families: ModelFamily[], item: LibraryItem): ModelVariant | null {
  let best: { variant: ModelVariant; score: number } | null = null;
  for (const variant of flattenVariants(families)) {
    const score = libraryVariantMatchScore(item, variant);
    if (!best || score > best.score) {
      best = { variant, score };
    }
  }
  return best && best.score >= 12 ? best.variant : null;
}

function defaultVariantForFamily(family: ModelFamily | null | undefined): ModelVariant | null {
  if (!family) {
    return null;
  }
  return family.variants.find((variant) => variant.id === family.defaultVariantId) ?? family.variants[0] ?? null;
}

function findVariantById(families: ModelFamily[], variantId: string | null | undefined): ModelVariant | null {
  if (!variantId) {
    return null;
  }
  for (const family of families) {
    const variant = family.variants.find((item) => item.id === variantId);
    if (variant) {
      return variant;
    }
  }
  return null;
}

function firstDirectVariant(families: ModelFamily[]): ModelVariant | null {
  const directVariants = flattenVariants(families)
    .filter((variant) => variant.launchMode === "direct")
    .sort((left, right) => left.paramsB - right.paramsB || left.sizeGb - right.sizeGb);
  return directVariants[0] ?? flattenVariants(families)[0] ?? null;
}

function findVariantForReference(
  families: ModelFamily[],
  modelRef: string | null | undefined,
  modelName?: string | null,
): ModelVariant | null {
  if (!modelRef && !modelName) {
    return null;
  }
  const loweredRef = modelRef?.toLowerCase();
  const loweredName = modelName?.toLowerCase();
  for (const variant of flattenVariants(families)) {
    if (
      loweredRef &&
      [variant.id, variant.repo, variant.name, variant.link].some((candidate) => candidate.toLowerCase() === loweredRef)
    ) {
      return variant;
    }
    if (loweredName && variant.name.toLowerCase() === loweredName) {
      return variant;
    }
  }
  return null;
}

function titleFromPrompt(prompt: string) {
  return prompt.trim().split(/\s+/).slice(0, 4).join(" ") || "New chat";
}

function compareOptionalNumber(left: number | null | undefined, right: number | null | undefined, dir: 1 | -1) {
  const leftKnown = typeof left === "number" && Number.isFinite(left);
  const rightKnown = typeof right === "number" && Number.isFinite(right);
  if (leftKnown && rightKnown) return dir * ((left as number) - (right as number));
  if (leftKnown && !rightKnown) return -1;
  if (!leftKnown && rightKnown) return 1;
  return 0;
}

function signedDelta(value: number, digits = 1, suffix = "") {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(digits)}${suffix}`;
}

function handleActionKeyDown(
  event: KeyboardEvent<HTMLElement>,
  action: () => void,
) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

const BENCHMARK_PROMPTS: Array<{ id: string; label: string; prompt: string }> = [
  {
    id: "balanced",
    label: "Balanced summary",
    prompt: "Summarize the trade-offs of this local runtime profile for a desktop user in six concise bullets.",
  },
  {
    id: "reasoning",
    label: "Reasoning check",
    prompt: "Explain how you would choose between cache efficiency, context length, and answer quality for local inference.",
  },
  {
    id: "coding",
    label: "Coding reply",
    prompt: "Review a small backend service architecture and list practical improvements for reliability, testing, and latency.",
  },
];

interface ChatModelOption {
  key: string;
  label: string;
  detail: string;
  group: string;
  model: string;
  modelRef: string;
  source: string;
  path?: string | null;
  backend: string;
  paramsB?: number;
  sizeGb?: number;
  contextWindow?: string;
  format?: string;
  quantization?: string;
  maxContext?: number | null;
}

function settingsDraftFromWorkspace(settings: AppSettings) {
  return {
    modelDirectories: settings?.modelDirectories ?? [],
    preferredServerPort: settings?.preferredServerPort ?? 8876,
    allowRemoteConnections: settings?.allowRemoteConnections ?? false,
    autoStartServer: settings?.autoStartServer ?? false,
    remoteProviders: settings?.remoteProviders ?? [],
    huggingFaceToken: "",
    hasHuggingFaceToken: settings?.hasHuggingFaceToken ?? false,
    huggingFaceTokenMasked: settings?.huggingFaceToken ?? "",
    dataDirectory: settings?.dataDirectory ?? "",
  };
}

interface DataDirRestartPrompt {
  migration: {
    copied: string[];
    skipped: string[];
    from: string;
    to: string;
  } | null;
}

export default function App() {
  const [workspace, setWorkspace] = useState<WorkspaceData>(mockWorkspace);
  const [loading, setLoading] = useState(true);
  const [backendOnline, setBackendOnline] = useState(false);
  const [tauriBackend, setTauriBackend] = useState<TauriBackendInfo | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("dashboard");
  const [searchInput, setSearchInput] = useState("");
  const deferredSearch = useDeferredValue(searchInput);
  const [searchResults, setSearchResults] = useState<ModelFamily[]>(mockWorkspace.featuredModels);
  const [hubResults, setHubResults] = useState<HubModel[]>([]);
  const [expandedHubId, setExpandedHubId] = useState<string | null>(null);
  const [hubFileCache, setHubFileCache] = useState<Record<string, HubFileListResponse>>({});
  const [hubFileLoading, setHubFileLoading] = useState<Record<string, boolean>>({});
  const [hubFileError, setHubFileError] = useState<Record<string, string>>({});
  const [detailFamilyId, setDetailFamilyId] = useState<string | null>(null);
  const [selectedFamilyId, setSelectedFamilyId] = useState(mockWorkspace.featuredModels[0]?.id ?? "");
  const [selectedVariantId, setSelectedVariantId] = useState(
    defaultVariantForFamily(mockWorkspace.featuredModels[0])?.id ?? "",
  );
  const [librarySearchInput, setLibrarySearchInput] = useState("");
  const [selectedLibraryPath, setSelectedLibraryPath] = useState(mockWorkspace.library[0]?.path ?? "");
  const [imageCatalog, setImageCatalog] = useState<ImageModelFamily[]>([]);
  const [latestImageDiscoverResults, setLatestImageDiscoverResults] = useState<ImageModelVariant[]>([]);
  const [imageDiscoverSourceFilter, setImageDiscoverSourceFilter] = useState<ImageDiscoverSourceFilter>("all");
  const [imageDiscoverTaskFilter, setImageDiscoverTaskFilter] = useState<ImageDiscoverTaskFilter>("all");
  const [imageDiscoverAccessFilter, setImageDiscoverAccessFilter] = useState<ImageDiscoverAccessFilter>("all");
  const [imageDiscoverSearchInput, setImageDiscoverSearchInput] = useState("");
  const deferredImageDiscoverSearch = useDeferredValue(imageDiscoverSearchInput);
  const [selectedImageModelId, setSelectedImageModelId] = useState("");
  const [imagePrompt, setImagePrompt] = useState("");
  const [imageNegativePrompt, setImageNegativePrompt] = useState("");
  const [imageQualityPreset, setImageQualityPreset] = useState<ImageQualityPreset>("balanced");
  const [imageRatioId, setImageRatioId] = useState<(typeof IMAGE_RATIO_PRESETS)[number]["id"]>("square");
  const [imageWidth, setImageWidth] = useState(1024);
  const [imageHeight, setImageHeight] = useState(1024);
  const [imageSteps, setImageSteps] = useState(24);
  const [imageGuidance, setImageGuidance] = useState(6);
  const [imageBatchSize, setImageBatchSize] = useState(1);
  const [imageUseRandomSeed, setImageUseRandomSeed] = useState(true);
  const [imageSeedInput, setImageSeedInput] = useState("");
  const [imageOutputs, setImageOutputs] = useState<ImageOutputArtifact[]>([]);
  const [imageGallerySearchInput, setImageGallerySearchInput] = useState("");
  const deferredImageGallerySearch = useDeferredValue(imageGallerySearchInput);
  const [imageGalleryModelFilter, setImageGalleryModelFilter] = useState<string>("all");
  const [imageGalleryRuntimeFilter, setImageGalleryRuntimeFilter] = useState<ImageGalleryRuntimeFilter>("all");
  const [imageGalleryOrientationFilter, setImageGalleryOrientationFilter] = useState<ImageGalleryOrientationFilter>("all");
  const [imageGallerySort, setImageGallerySort] = useState<ImageGallerySort>("newest");
  const [imageRuntimeStatus, setImageRuntimeStatus] = useState<ImageRuntimeStatus>({
    activeEngine: "placeholder",
    realGenerationAvailable: false,
    message: "Image Studio is currently using the placeholder engine on this machine.",
    missingDependencies: [],
  });
  const [imageBusyLabel, setImageBusyLabel] = useState<string | null>(null);
  const [showImageGenerationModal, setShowImageGenerationModal] = useState(false);
  const [imageGenerationStartedAt, setImageGenerationStartedAt] = useState<number | null>(null);
  const [imageGenerationError, setImageGenerationError] = useState<string | null>(null);
  const [imageGenerationArtifacts, setImageGenerationArtifacts] = useState<ImageOutputArtifact[]>([]);
  const [selectedImageGenerationArtifactId, setSelectedImageGenerationArtifactId] = useState<string | null>(null);
  const [imageGenerationRunInfo, setImageGenerationRunInfo] = useState<{
    modelName: string;
    prompt: string;
    batchSize: number;
    steps: number;
    needsPipelineLoad: boolean;
  } | null>(null);
  const [activeImageDownloads, setActiveImageDownloads] = useState<Record<string, DownloadStatus>>({});
  const [preview, setPreview] = useState<PreviewMetrics>(mockWorkspace.preview);
  const [previewControls, setPreviewControls] = useState({
    bits: mockWorkspace.settings.launchPreferences.cacheBits,
    fp16Layers: mockWorkspace.settings.launchPreferences.fp16Layers,
    numLayers: mockWorkspace.preview.numLayers,
    numHeads: mockWorkspace.preview.numHeads,
    hiddenSize: mockWorkspace.preview.hiddenSize,
    contextTokens: mockWorkspace.settings.launchPreferences.contextTokens,
    paramsB: mockWorkspace.preview.paramsB,
  });
  const [activeChatId, setActiveChatId] = useState(mockWorkspace.chatSessions[0]?.id ?? "");
  const [threadTitleDraft, setThreadTitleDraft] = useState(mockWorkspace.chatSessions[0]?.title ?? "");
  const [draftMessage, setDraftMessage] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [launchSettings, setLaunchSettings] = useState<LaunchPreferences>(mockWorkspace.settings.launchPreferences);
  const [settingsDraft, setSettingsDraft] = useState(settingsDraftFromWorkspace(mockWorkspace.settings));
  const [dataDirRestartPrompt, setDataDirRestartPrompt] = useState<DataDirRestartPrompt | null>(null);
  const [serverModelKey, setServerModelKey] = useState("");
  const [newDirectoryLabel, setNewDirectoryLabel] = useState("");
  const [newDirectoryPath, setNewDirectoryPath] = useState("");
  const [conversionDraft, setConversionDraft] = useState({
    modelRef: "",
    path: "",
    hfRepo: "",
    outputPath: "",
    quantize: true,
    qBits: 4,
    qGroupSize: 64,
    dtype: "float16",
  });
  const [lastConversion, setLastConversion] = useState<ConversionResult | null>(null);
  const [benchmarkPromptId, setBenchmarkPromptId] = useState(BENCHMARK_PROMPTS[0]?.id ?? "balanced");
  const [benchmarkDraft, setBenchmarkDraft] = useState<BenchmarkRunPayload>({
    cacheBits: mockWorkspace.settings.launchPreferences.cacheBits,
    fp16Layers: mockWorkspace.settings.launchPreferences.fp16Layers,
    fusedAttention: mockWorkspace.settings.launchPreferences.fusedAttention,
    cacheStrategy: mockWorkspace.settings.launchPreferences.cacheStrategy,
    fitModelInMemory: mockWorkspace.settings.launchPreferences.fitModelInMemory,
    contextTokens: mockWorkspace.settings.launchPreferences.contextTokens,
    maxTokens: 4096,
    temperature: 0.2,
  });
  const [benchmarkModelKey, setBenchmarkModelKey] = useState("");
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState(mockWorkspace.benchmarks[0]?.id ?? "");
  const [compareBenchmarkId, setCompareBenchmarkId] = useState(mockWorkspace.benchmarks[1]?.id ?? "");
  const [logQuery, setLogQuery] = useState("");
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [conversionStartedAt, setConversionStartedAt] = useState<number | null>(null);
  const [benchmarkStartedAt, setBenchmarkStartedAt] = useState<number | null>(null);
  const busy = busyAction !== null;
  const imageBusy = imageBusyLabel !== null;
  const [chatBusySessionId, setChatBusySessionId] = useState<string | null>(null);
  const [pendingImages, setPendingImages] = useState<string[]>([]);
  const [error, setErrorState] = useState<string | null>(null);
  const errorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  function setError(msg: string | null) {
    if (errorTimerRef.current) clearTimeout(errorTimerRef.current);
    setErrorState(msg);
    // Keep llama.cpp rebuild errors sticky — the user needs to click the
    // rebuild button. All other errors auto-dismiss after 8s.
    const isSticky = !!msg && msg.includes("update-llama-cpp.sh");
    if (msg && !isSticky) {
      errorTimerRef.current = setTimeout(() => setErrorState(null), 8000);
    }
  }
  const [rebuildingLlama, setRebuildingLlama] = useState(false);
  const [rebuildOutput, setRebuildOutput] = useState<string | null>(null);
  async function handleRebuildLlamaCpp() {
    setRebuildingLlama(true);
    setRebuildOutput(null);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const result = await invoke<{ ok: boolean; exitCode: number | null; output: string }>(
        "rebuild_llama_cpp",
      );
      setRebuildOutput(result.output);
      if (result.ok) {
        setError(null);
        // Restart the backend sidecar so it picks up the new binary.
        try {
          await invoke("restart_backend_sidecar");
        } catch {
          /* best-effort restart */
        }
      } else {
        setError(`Rebuild failed (exit ${result.exitCode ?? "?"}). See output below.`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Rebuild failed to start.");
    } finally {
      setRebuildingLlama(false);
    }
  }
  const [pendingLaunch, setPendingLaunch] = useState<{
    action: "chat" | "server" | "thread";
    preselectedKey?: string;
  } | null>(null);
  const [launchModelSearch, setLaunchModelSearch] = useState("");
  const [expandedFamilyId, setExpandedFamilyId] = useState<string | null>(null);
  const [expandedVariantId, setExpandedVariantId] = useState<string | null>(null);
  const [expandedLibraryPath, setExpandedLibraryPath] = useState<string | null>(null);
  const [librarySortKey, setLibrarySortKey] = useState<"name" | "format" | "backend" | "size" | "ram" | "compressed" | "modified" | "context">("modified");
  const [librarySortDir, setLibrarySortDir] = useState<"asc" | "desc">("desc");
  const [showRemoteTest, setShowRemoteTest] = useState(false);
  const [testModelId, setTestModelId] = useState<string | null>(null);
  const [serverLogEntries, setServerLogEntries] = useState<Array<{ ts: string; level: string; message: string }>>([]);
  const [discoverCapFilter, setDiscoverCapFilter] = useState<string | null>(null);
  const [libraryCapFilter, setLibraryCapFilter] = useState<string | null>(null);
  const [discoverFormatFilter, setDiscoverFormatFilter] = useState<string | null>(null);
  const [libraryFormatFilter, setLibraryFormatFilter] = useState<string | null>(null);
  const [libraryBackendFilter, setLibraryBackendFilter] = useState<string | null>(null);
  const [showConversionPicker, setShowConversionPicker] = useState(false);
  const [showBenchmarkPicker, setShowBenchmarkPicker] = useState(false);
  const [showConversionModal, setShowConversionModal] = useState(false);
  const [conversionError, setConversionError] = useState<string | null>(null);
  const [showBenchmarkModal, setShowBenchmarkModal] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
  const [activeDownloads, setActiveDownloads] = useState<Record<string, DownloadStatus>>({});

  async function refreshWorkspace(preferredChatId?: string) {
    const [online, payload] = await Promise.all([checkBackend(), getWorkspace()]);
    setBackendOnline(online);
    setWorkspace(payload);
    // Use functional setState so a slow in-flight refresh cannot clobber a
    // more recent user thread selection (otherwise background polls race
    // against clicks and cause focus flip-flop).
    setActiveChatId((current) => {
      if (current && payload.chatSessions.some((session) => session.id === current)) {
        return current;
      }
      if (preferredChatId && payload.chatSessions.some((session) => session.id === preferredChatId)) {
        return preferredChatId;
      }
      return payload.chatSessions[0]?.id ?? "";
    });
  }

  async function refreshImageData() {
    const [catalog, outputs, statuses, runtime] = await Promise.allSettled([
      getImageCatalog(),
      getImageOutputs(),
      getImageDownloadStatus(),
      getImageRuntime(),
    ]);
    const failures = [catalog, outputs, statuses, runtime].filter(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );

    if (catalog.status === "fulfilled") {
      setImageCatalog(catalog.value.families);
      setLatestImageDiscoverResults(catalog.value.latest ?? []);
    }
    if (outputs.status === "fulfilled") {
      setImageOutputs(outputs.value);
    }
    if (statuses.status === "fulfilled") {
      setActiveImageDownloads(buildDownloadStatusMap(statuses.value));
    }
    if (runtime.status === "fulfilled") {
      setImageRuntimeStatus(runtime.value);
    } else if (failures.length > 0) {
      setImageRuntimeStatus(imageRuntimeErrorStatus(failures[0].reason));
    }

    if (failures.length > 0) {
      const firstError = failures[0].reason;
      setError(firstError instanceof Error ? firstError.message : "Could not load image runtime data.");
    }
  }

  useEffect(() => {
    // Background check on startup. Silent so users on the latest version
    // see nothing; if an update is available the updater module prompts.
    const timer = setTimeout(() => {
      void checkForUpdates({ silent: true });
    }, 4000);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    let cancelled = false;

    // Retry the initial workspace fetch with backoff. The Python sidecar
    // takes a moment to spawn after the Tauri shell starts; without retry,
    // a single failed attempt leaves us stuck on the 4-family mock data
    // until the user manually triggers a search.
    async function loadInitial(): Promise<void> {
      const delays = [0, 400, 800, 1500, 2500, 4000, 6000];
      for (const delay of delays) {
        if (cancelled) return;
        if (delay) await new Promise((r) => setTimeout(r, delay));
        try {
          const [online, payload, runtimeInfo] = await Promise.all([
            checkBackend(),
            getWorkspace(),
            getTauriBackendInfo(),
          ]);
          if (cancelled) return;
          setBackendOnline(online);
          setTauriBackend(runtimeInfo);
          setWorkspace(payload);
          setSearchResults(payload.featuredModels);
          setPreview(payload.preview);
          setLaunchSettings(payload.settings.launchPreferences);
          setSettingsDraft(settingsDraftFromWorkspace(payload.settings));
          setPreviewControls({
            bits: payload.settings.launchPreferences.cacheBits,
            fp16Layers: payload.settings.launchPreferences.fp16Layers,
            numLayers: payload.preview.numLayers,
            numHeads: payload.preview.numHeads,
            hiddenSize: payload.preview.hiddenSize,
            contextTokens: payload.settings.launchPreferences.contextTokens,
            paramsB: payload.preview.paramsB,
          });
          setActiveChatId(payload.chatSessions[0]?.id ?? "");
          setThreadTitleDraft(payload.chatSessions[0]?.title ?? "");
          await refreshImageData();
          setLoading(false);
          return;
        } catch {
          // backend not ready yet — keep retrying
          if (!cancelled) setBackendOnline(false);
        }
      }
      if (!cancelled) setLoading(false);
    }

    void loadInitial();

    return () => {
      cancelled = true;
    };
  }, []);

  // Poll for backend availability and refresh workspace periodically.
  // Skip polling while an operation is in progress — the handler will
  // refresh workspace when it completes, and marking the backend as
  // offline mid-operation causes cascading UI glitches.
  // Poll faster (2s) when a model is loading to show real-time progress.
  const isModelLoading = workspace.server.loading !== null;
  useEffect(() => {
    const pollInterval = !backendOnline ? 2500 : isModelLoading ? 2000 : 10000;
    const interval = window.setInterval(() => {
      // Skip polling during any chat generation or global busy state
      // — otherwise refreshWorkspace overwrites the streaming message
      if ((busy || chatBusySessionId !== null) && !isModelLoading) return;
      void (async () => {
        const online = await checkBackend();
        setBackendOnline(online);
        if (online) {
          void refreshWorkspace(activeChatId || undefined);
        }
      })();
    }, pollInterval);

    return () => {
      window.clearInterval(interval);
    };
  }, [activeChatId, backendOnline, busy, isModelLoading, chatBusySessionId]);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const { families, hubModels } = await searchModels(deferredSearch);
        if (!cancelled) {
          setSearchResults(families);
          setHubResults(hubModels);
        }
      } catch (searchError) {
        if (!cancelled) {
          setHubResults([]);
          setError(searchError instanceof Error ? searchError.message : "Could not search models.");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [deferredSearch]);

  useEffect(() => {
    if (!searchResults.length) {
      setSelectedFamilyId("");
      setSelectedVariantId("");
      return;
    }
    const familyValid = searchResults.some((family) => family.id === selectedFamilyId);
    const nextFamilyId = familyValid ? selectedFamilyId : searchResults[0].id;
    if (nextFamilyId !== selectedFamilyId) {
      setSelectedFamilyId(nextFamilyId);
    }
    const family = searchResults.find((f) => f.id === nextFamilyId) ?? searchResults[0];
    if (family && !family.variants.some((v) => v.id === selectedVariantId)) {
      const dv = defaultVariantForFamily(family);
      if (dv) setSelectedVariantId(dv.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchResults]);

  useEffect(() => {
    if (!imageCatalog.length) {
      setSelectedImageModelId("");
      return;
    }
    const variants = flattenImageVariants(imageCatalog);
    if (variants.some((variant) => variant.id === selectedImageModelId)) {
      return;
    }
    const preferred =
      variants.find((variant) => variant.availableLocally) ??
      defaultImageVariantForFamily(imageCatalog[0]);
    setSelectedImageModelId(preferred?.id ?? "");
  }, [imageCatalog, selectedImageModelId]);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      void (async () => {
        const nextPreview = await getCachePreview(previewControls);
        setPreview(nextPreview);
      })();
    }, 220);

    return () => {
      window.clearTimeout(timeout);
    };
  }, [previewControls]);

  useEffect(() => {
    setPreviewControls((current) => {
      if (current.bits === launchSettings.cacheBits && current.fp16Layers === launchSettings.fp16Layers && current.contextTokens === launchSettings.contextTokens) {
        return current;
      }
      return { ...current, bits: launchSettings.cacheBits, fp16Layers: launchSettings.fp16Layers, contextTokens: launchSettings.contextTokens };
    });
  }, [launchSettings.contextTokens, launchSettings.fp16Layers, launchSettings.cacheBits]);

  // Benchmark page: sync benchmarkDraft sliders -> previewControls so PerformancePreview reflects them live
  useEffect(() => {
    if (activeTab !== "benchmarks") return;
    setPreviewControls((current) => {
      if (
        current.bits === benchmarkDraft.cacheBits &&
        current.fp16Layers === benchmarkDraft.fp16Layers &&
        current.contextTokens === benchmarkDraft.contextTokens
      ) {
        return current;
      }
      return {
        ...current,
        bits: benchmarkDraft.cacheBits,
        fp16Layers: benchmarkDraft.fp16Layers,
        contextTokens: benchmarkDraft.contextTokens,
      };
    });
  }, [activeTab, benchmarkDraft.cacheBits, benchmarkDraft.fp16Layers, benchmarkDraft.contextTokens]);


  useEffect(() => {
    setBenchmarkDraft((current) => {
      if (
        current.cacheBits === launchSettings.cacheBits &&
        current.fp16Layers === launchSettings.fp16Layers &&
        current.fusedAttention === launchSettings.fusedAttention &&
        current.cacheStrategy === launchSettings.cacheStrategy &&
        current.fitModelInMemory === launchSettings.fitModelInMemory &&
        current.contextTokens === launchSettings.contextTokens
      ) return current;
      return {
      ...current,
      cacheBits: launchSettings.cacheBits,
      fp16Layers: launchSettings.fp16Layers,
      fusedAttention: launchSettings.fusedAttention,
      cacheStrategy: launchSettings.cacheStrategy,
      fitModelInMemory: launchSettings.fitModelInMemory,
      contextTokens: launchSettings.contextTokens,
    };});
  }, [
    launchSettings.contextTokens,
    launchSettings.fitModelInMemory,
    launchSettings.fp16Layers,
    launchSettings.fusedAttention,
    launchSettings.cacheBits,
    launchSettings.cacheStrategy,
  ]);

  useEffect(() => {
    if (!workspace.chatSessions.some((session) => session.id === activeChatId)) {
      setActiveChatId(workspace.chatSessions[0]?.id ?? "");
    }
  }, [workspace.chatSessions, activeChatId]);

  useEffect(() => {
    const nextFilteredLibrary = workspace.library
      .filter((item) => {
        const haystack = `${item.name} ${item.path} ${item.format} ${item.directoryLabel ?? ""}`.toLowerCase();
        return haystack.includes(librarySearchInput.trim().toLowerCase());
      })
      .sort((left, right) => {
        if (left.lastModified !== right.lastModified) {
          return right.lastModified.localeCompare(left.lastModified);
        }
        return right.sizeGb - left.sizeGb;
      });

    if (!nextFilteredLibrary.length) {
      setSelectedLibraryPath("");
      return;
    }
    setSelectedLibraryPath((current) =>
      nextFilteredLibrary.some((item) => item.path === current) ? current : nextFilteredLibrary[0].path,
    );
  }, [workspace.library, librarySearchInput]);

  useEffect(() => {
    const nextActiveChat = workspace.chatSessions.find((session) => session.id === activeChatId) ?? workspace.chatSessions[0];
    setThreadTitleDraft(nextActiveChat?.title ?? "");
    // Only re-sync the rename draft when the active thread actually changes,
    // not on every workspace refresh (which would clobber mid-typing edits).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeChatId]);

  useEffect(() => {
    if (!workspace.settings?.modelDirectories) return;
    setSettingsDraft((current) => {
      const dirs = (current.modelDirectories ?? []);
      let changed = false;
      const updated = dirs.map((directory) => {
        const latest = workspace.settings.modelDirectories.find((item) => item.id === directory.id);
        if (latest && (directory.exists !== latest.exists || directory.modelCount !== latest.modelCount)) {
          changed = true;
          return { ...directory, exists: latest.exists, modelCount: latest.modelCount };
        }
        return directory;
      });
      return changed ? { ...current, modelDirectories: updated } : current;
    });
  }, [workspace.settings?.modelDirectories]);

  useEffect(() => {
    if (!workspace.benchmarks.length) {
      setSelectedBenchmarkId("");
      setCompareBenchmarkId("");
      return;
    }

    setSelectedBenchmarkId((current) =>
      workspace.benchmarks.some((benchmark) => benchmark.id === current) ? current : workspace.benchmarks[0]?.id ?? "",
    );
    setCompareBenchmarkId((current) =>
      workspace.benchmarks.some((benchmark) => benchmark.id === current)
        ? current
        : workspace.benchmarks[1]?.id ?? workspace.benchmarks[0]?.id ?? "",
    );
  }, [workspace.benchmarks]);

  const sortedChatSessions = sortSessions(workspace.chatSessions);
  const activeChat = workspace.chatSessions.find((session) => session.id === activeChatId) ?? sortedChatSessions[0];
  const activeFamilies = searchResults;
  const selectedFamily = activeFamilies.find((family) => family.id === selectedFamilyId) ?? activeFamilies[0] ?? null;
  const selectedVariant = findVariantById(activeFamilies, selectedVariantId) ?? defaultVariantForFamily(selectedFamily);
  const imageVariants = flattenImageVariants(imageCatalog);
  const selectedImageVariant = findImageVariantById(imageCatalog, selectedImageModelId) ?? imageVariants[0] ?? null;
  const selectedImageFamily = imageCatalog.find((family) =>
    family.variants.some((variant) => variant.id === selectedImageVariant?.id),
  ) ?? null;
  const loadedImageVariant = findImageVariantByRepo(imageCatalog, imageRuntimeStatus.loadedModelRepo);
  const selectedImageLoaded =
    !!selectedImageVariant &&
    !!imageRuntimeStatus.loadedModelRepo &&
    imageRuntimeStatus.loadedModelRepo === selectedImageVariant.repo;
  const selectedImageWillLoadOnGenerate =
    !!selectedImageVariant &&
    selectedImageVariant.availableLocally &&
    imageRuntimeStatus.realGenerationAvailable &&
    !selectedImageLoaded;
  const imageRuntimeLoadedDifferentModel =
    !!selectedImageVariant &&
    !!loadedImageVariant &&
    loadedImageVariant.repo !== selectedImageVariant.repo;
  const selectedImageGenerationArtifact =
    imageGenerationArtifacts.find((artifact) => artifact.artifactId === selectedImageGenerationArtifactId) ??
    imageGenerationArtifacts[0] ??
    null;
  const installedImageVariants = imageVariants.filter((variant) => variant.availableLocally);
  const imageDiscoverSearchQuery = deferredImageDiscoverSearch.trim().toLowerCase();
  const filteredImageDiscoverFamilies = imageCatalog
    .map((family) => ({
      ...family,
      variants: family.variants.filter((variant) => {
        if (!imageVariantMatchesDiscoverFilters(variant, imageDiscoverTaskFilter, imageDiscoverAccessFilter)) {
          return false;
        }
        return (
          imageDiscoverFamilyMatchesQuery(family, imageDiscoverSearchQuery) ||
          imageDiscoverVariantMatchesQuery(variant, imageDiscoverSearchQuery)
        );
      }),
    }))
    .filter((family) => family.variants.length > 0);
  const filteredLatestImageDiscoverResults = latestImageDiscoverResults.filter((variant) =>
    imageVariantMatchesDiscoverFilters(variant, imageDiscoverTaskFilter, imageDiscoverAccessFilter) &&
    imageDiscoverVariantMatchesQuery(variant, imageDiscoverSearchQuery),
  );
  const showCuratedImageDiscoverSection = imageDiscoverSourceFilter !== "latest";
  const showLatestImageDiscoverSection = imageDiscoverSourceFilter !== "curated";
  const imageDiscoverHasActiveFilters =
    imageDiscoverSourceFilter !== "all" ||
    imageDiscoverTaskFilter !== "all" ||
    imageDiscoverAccessFilter !== "all" ||
    imageDiscoverSearchQuery.length > 0;
  const imageOutputsNewestFirst = [...imageOutputs].sort((left, right) => imageArtifactTimestamp(right) - imageArtifactTimestamp(left));
  const recentImageOutputs = imageOutputsNewestFirst.slice(0, 2);
  const imageGalleryModelOptions = Array.from(
    new Map(
      imageOutputsNewestFirst.map((artifact) => [
        artifact.modelId,
        { id: artifact.modelId, name: artifact.modelName },
      ]),
    ).values(),
  ).sort((left, right) => left.name.localeCompare(right.name));
  const imageGalleryHasActiveFilters =
    imageGalleryModelFilter !== "all" ||
    imageGalleryRuntimeFilter !== "all" ||
    imageGalleryOrientationFilter !== "all" ||
    imageGallerySort !== "newest" ||
    deferredImageGallerySearch.trim().length > 0;
  const filteredImageOutputs = [...imageOutputs].filter((artifact) => {
    if (imageGalleryModelFilter !== "all" && artifact.modelId !== imageGalleryModelFilter) {
      return false;
    }
    if (imageGalleryRuntimeFilter === "diffusers" && imageRuntimeKind(artifact.runtimeLabel) !== "diffusers") {
      return false;
    }
    if (imageGalleryRuntimeFilter === "placeholder" && imageRuntimeKind(artifact.runtimeLabel) !== "placeholder") {
      return false;
    }
    if (imageGalleryRuntimeFilter === "warning" && !artifact.runtimeNote) {
      return false;
    }
    if (
      imageGalleryOrientationFilter !== "all" &&
      imageOrientation(artifact.width, artifact.height) !== imageGalleryOrientationFilter
    ) {
      return false;
    }
    const query = deferredImageGallerySearch.trim().toLowerCase();
    if (!query) {
      return true;
    }
    const haystack = `${artifact.modelName} ${artifact.prompt} ${artifact.runtimeLabel ?? ""} ${artifact.runtimeNote ?? ""}`.toLowerCase();
    return haystack.includes(query);
  }).sort((left, right) => (
    imageGallerySort === "oldest"
      ? imageArtifactTimestamp(left) - imageArtifactTimestamp(right)
      : imageArtifactTimestamp(right) - imageArtifactTimestamp(left)
  ));
  const imageGalleryRealCount = imageOutputs.filter((artifact) => imageRuntimeKind(artifact.runtimeLabel) === "diffusers").length;
  const imageGalleryPlaceholderCount = imageOutputs.filter((artifact) => imageRuntimeKind(artifact.runtimeLabel) === "placeholder").length;
  const imageGalleryWarningCount = imageOutputs.filter((artifact) => Boolean(artifact.runtimeNote)).length;
  const imageGalleryModelCount = imageGalleryModelOptions.length;
  const allFeaturedVariants = flattenVariants(workspace.featuredModels);
  const libraryRows = workspace.library.map((item) => {
    const matchedVariant = findCatalogVariantForLibraryItem(workspace.featuredModels, item);
    return {
      item,
      matchedVariant,
      displayFormat: libraryItemFormat(item, matchedVariant),
      displayQuantization: libraryItemQuantization(item, matchedVariant),
      displayBackend: libraryItemBackend(item, matchedVariant),
      sourceKind: libraryItemSourceKind(item),
    };
  });
  const filteredLibraryRows = libraryRows
    .filter(({ item, displayFormat, displayQuantization, displayBackend, sourceKind }) => {
      const haystack = `${item.name} ${item.path} ${displayFormat} ${displayQuantization ?? ""} ${displayBackend} ${sourceKind} ${item.directoryLabel ?? ""}`.toLowerCase();
      return haystack.includes(librarySearchInput.trim().toLowerCase());
    })
    .sort((left, right) => {
      const dir = librarySortDir === "asc" ? 1 : -1;
      switch (librarySortKey) {
        case "name":
          return dir * left.item.name.localeCompare(right.item.name);
        case "format":
          return dir * left.displayFormat.localeCompare(right.displayFormat);
        case "backend":
          return dir * left.displayBackend.localeCompare(right.displayBackend);
        case "size":
          return dir * (left.item.sizeGb - right.item.sizeGb);
        case "ram":
          return compareOptionalNumber(left.matchedVariant?.estimatedMemoryGb, right.matchedVariant?.estimatedMemoryGb, dir);
        case "compressed":
          return compareOptionalNumber(
            left.matchedVariant?.estimatedCompressedMemoryGb,
            right.matchedVariant?.estimatedCompressedMemoryGb,
            dir,
          );
        case "context": {
          const lc = parseContextK(left.matchedVariant?.contextWindow);
          const rc = parseContextK(right.matchedVariant?.contextWindow);
          return dir * (lc - rc);
        }
        case "modified":
        default:
          return dir * left.item.lastModified.localeCompare(right.item.lastModified);
      }
    });
  const selectedLibraryRow = filteredLibraryRows.find(({ item }) => item.path === selectedLibraryPath) ?? filteredLibraryRows[0] ?? null;
  const selectedLibraryItem = selectedLibraryRow?.item ?? null;
  const selectedLibraryVariant = selectedLibraryRow?.matchedVariant ?? null;
  const defaultChatVariant = firstDirectVariant(workspace.featuredModels);
  const benchmarkMaxTokS = Math.max(1, ...workspace.benchmarks.map((item) => item.tokS));
  const benchmarkMaxCacheGb = Math.max(1, ...workspace.benchmarks.map((item) => item.baselineCacheGb || item.cacheGb));
  const nativeBackends = workspace.runtime.nativeBackends;
  const filteredLogs = workspace.logs.filter((entry) => {
    const haystack = `${entry.ts} ${entry.source} ${entry.level} ${entry.message}`.toLowerCase();
    return haystack.includes(logQuery.toLowerCase());
  });
  const previewSavings = Math.max(0, preview.baselineCacheGb - preview.optimizedCacheGb);
  const conversionReady = Boolean(nativeBackends?.converterAvailable ?? workspace.system.mlxLmAvailable);
  const currentCacheLabel = launchSettings.cacheStrategy === "native"
    ? "Native f16"
    : `${launchSettings.cacheStrategy} ${launchSettings.cacheBits}-bit ${launchSettings.fp16Layers}+${launchSettings.fp16Layers}`;
  const launchCacheLabel = currentCacheLabel;
  const loadedModelCacheLabel = workspace.runtime.loadedModel
    ? workspace.runtime.loadedModel.cacheStrategy === "native"
      ? "Native f16"
      : `${workspace.runtime.loadedModel.cacheStrategy} ${workspace.runtime.loadedModel.cacheBits}-bit ${workspace.runtime.loadedModel.fp16Layers}+${workspace.runtime.loadedModel.fp16Layers}`
    : launchCacheLabel;
  const enabledDirectoryCount = (workspace.settings?.modelDirectories ?? []).filter((directory) => directory.enabled).length;
  const libraryTotalSizeGb = workspace.library.reduce((sum, item) => sum + item.sizeGb, 0);
  const localVariantCount = allFeaturedVariants.filter((variant) => variant.availableLocally).length;
  const fileRevealLabel =
    workspace.system.platform === "Darwin"
      ? "Show in Finder"
      : workspace.system.platform === "Windows"
        ? "Show in Explorer"
        : "Show in Files";

  const catalogChatOptions: ChatModelOption[] = allFeaturedVariants
    .filter((variant) => variant.launchMode === "direct")
    .map((variant) => ({
      key: `catalog:${variant.id}`,
      label: variant.name,
      detail: `${variant.format} / ${variant.quantization}`,
      group: "Catalog",
      model: variant.name,
      modelRef: variant.id,
      source: "catalog",
      backend: variant.backend,
      paramsB: variant.paramsB,
      sizeGb: variant.sizeGb,
      contextWindow: variant.contextWindow,
      format: variant.format,
      quantization: variant.quantization,
      maxContext: variant.maxContext ?? null,
    }));

  const libraryChatOptions: ChatModelOption[] = workspace.library.map((item) => {
    const matched = findCatalogVariantForLibraryItem(workspace.featuredModels, item);
    const displayFormat = libraryItemFormat(item, matched);
    const displayQuantization = libraryItemQuantization(item, matched);
    return {
      key: `library:${item.path}`,
      label: item.name,
      detail: `${displayFormat} / ${number(item.sizeGb)} GB`,
      group: "Local library",
      model: item.name,
      modelRef: item.name,
      source: "library",
      path: item.path,
      backend: libraryItemBackend(item, matched),
      paramsB: matched?.paramsB,
      sizeGb: item.sizeGb,
      contextWindow: matched?.contextWindow,
      format: displayFormat,
      quantization: displayQuantization ?? undefined,
      maxContext: item.maxContext ?? matched?.maxContext ?? null,
    };
  });

  const threadModelOptions = [...catalogChatOptions, ...libraryChatOptions];

  // Re-anchor the preview to the model selected in the Launch modal
  // (otherwise cache/tok-s/quality stay locked to whatever was loaded at startup).
  useEffect(() => {
    if (!pendingLaunch) return;
    const key = pendingLaunch.preselectedKey;
    if (!key) return;
    const options = libraryChatOptions.length > 0 ? libraryChatOptions : threadModelOptions;
    const opt = options.find((o) => o.key === key);
    if (!opt) return;
    let paramsB = opt.paramsB ?? 0;
    if (!paramsB && opt.sizeGb) {
      const bpw = detectBitsPerWeight(`${opt.label} ${opt.format ?? ""} ${opt.quantization ?? ""}`);
      paramsB = estimateParamsBFromDisk(opt.sizeGb, bpw);
    }
    if (!paramsB) return;
    const arch = estimateArchFromParams(paramsB);
    setPreviewControls((current) => {
      if (
        current.paramsB === paramsB &&
        current.numLayers === arch.numLayers &&
        current.numHeads === arch.numHeads &&
        current.hiddenSize === arch.hiddenSize
      ) {
        return current;
      }
      return { ...current, paramsB, ...arch };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingLaunch?.preselectedKey, libraryChatOptions, threadModelOptions]);

  const activeThreadOption =
    threadModelOptions.find((option) => option.modelRef === activeChat?.modelRef && option.path === (activeChat?.modelPath ?? undefined)) ??
    threadModelOptions.find((option) => option.modelRef === activeChat?.modelRef) ??
    null;
  const serverOptionKeySignature = threadModelOptions.map((option) => option.key).join("|");
  const loadedModelOption =
    threadModelOptions.find(
      (option) =>
        option.modelRef === workspace.runtime.loadedModel?.ref &&
        option.path === (workspace.runtime.loadedModel?.path ?? undefined),
    ) ??
    threadModelOptions.find((option) => option.modelRef === workspace.runtime.loadedModel?.ref) ??
    null;
  const selectedServerOptionBase =
    threadModelOptions.find((option) => option.key === serverModelKey) ?? loadedModelOption ?? activeThreadOption ?? null;
  const selectedServerOption = (() => {
    if (!selectedServerOptionBase || selectedServerOptionBase.source !== "catalog") {
      return selectedServerOptionBase;
    }
    const variant = findVariantForReference(
      workspace.featuredModels,
      selectedServerOptionBase.modelRef,
      selectedServerOptionBase.model,
    );
    if (!variant) {
      return selectedServerOptionBase;
    }
    const localItem = findLibraryItemForVariant(workspace.library, variant);
    if (!localItem) {
      return selectedServerOptionBase;
    }
    return libraryChatOptions.find((option) => option.path === localItem.path) ?? selectedServerOptionBase;
  })();
  const convertibleLibrary = workspace.library.filter((item) => libraryItemFormat(item) !== "MLX");
  const conversionSource = convertibleLibrary.find((item) => item.path === conversionDraft.path) ?? null;
  const conversionVariant =
    (conversionSource ? findCatalogVariantForLibraryItem(workspace.featuredModels, conversionSource) : null) ??
    findVariantForReference(workspace.featuredModels, conversionDraft.modelRef, conversionDraft.modelRef) ??
    null;
  const benchmarkOption =
    threadModelOptions.find((option) => option.key === benchmarkModelKey) ??
    threadModelOptions.find((option) => option.modelRef === benchmarkDraft.modelRef && option.path === benchmarkDraft.path) ??
    activeThreadOption ??
    loadedModelOption ??
    threadModelOptions[0] ??
    null;
  // Re-anchor previewControls.paramsB / arch when the active page's selected model changes
  // (Benchmark page: benchmarkOption; Conversion page: conversionSource).
  useEffect(() => {
    let label = "", sizeGb = 0, paramsB = 0;
    if (activeTab === "benchmarks" && benchmarkOption) {
      label = `${benchmarkOption.label} ${benchmarkOption.format ?? ""} ${benchmarkOption.quantization ?? ""}`;
      sizeGb = benchmarkOption.sizeGb ?? 0;
      paramsB = benchmarkOption.paramsB ?? 0;
    } else if (activeTab === "conversion" && conversionSource) {
      label = `${conversionSource.name} ${conversionSource.format ?? ""}`;
      sizeGb = conversionSource.sizeGb ?? 0;
      paramsB = conversionVariant?.paramsB ?? 0;
    } else {
      return;
    }
    if (!paramsB && sizeGb) {
      paramsB = estimateParamsBFromDisk(sizeGb, detectBitsPerWeight(label));
    }
    if (!paramsB) return;
    const arch = estimateArchFromParams(paramsB);
    setPreviewControls((current) => {
      if (
        current.paramsB === paramsB &&
        current.numLayers === arch.numLayers &&
        current.numHeads === arch.numHeads &&
        current.hiddenSize === arch.hiddenSize
      ) {
        return current;
      }
      return { ...current, paramsB, ...arch };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, benchmarkOption?.key, conversionSource?.path, conversionVariant?.id]);

  const selectedBenchmark = workspace.benchmarks.find((item) => item.id === selectedBenchmarkId) ?? workspace.benchmarks[0] ?? null;
  const compareBenchmark =
    workspace.benchmarks.find((item) => item.id === compareBenchmarkId && item.id !== selectedBenchmark?.id) ??
    workspace.benchmarks.find((item) => item.id !== selectedBenchmark?.id) ??
    null;
  const benchmarkSpeedDelta = selectedBenchmark && compareBenchmark ? selectedBenchmark.tokS - compareBenchmark.tokS : 0;
  const benchmarkCacheDelta =
    selectedBenchmark && compareBenchmark ? selectedBenchmark.cacheGb - compareBenchmark.cacheGb : 0;
  const benchmarkLatencyDelta =
    selectedBenchmark && compareBenchmark ? selectedBenchmark.responseSeconds - compareBenchmark.responseSeconds : 0;
  const previewVariant =
    activeTab === "server"
      ? findVariantForReference(
          workspace.featuredModels,
          selectedServerOption?.modelRef ?? workspace.runtime.loadedModel?.ref,
          selectedServerOption?.model ?? workspace.runtime.loadedModel?.name,
        ) ??
        findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ??
        defaultChatVariant
      : activeTab === "my-models"
        ? selectedLibraryVariant ??
          findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ??
          defaultChatVariant
      : activeTab === "chat"
        ? findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ??
          findVariantForReference(
            workspace.featuredModels,
            selectedServerOption?.modelRef ?? workspace.runtime.loadedModel?.ref,
            selectedServerOption?.model ?? workspace.runtime.loadedModel?.name,
          ) ??
          defaultChatVariant
        : selectedVariant ??
          findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ??
          defaultChatVariant;
  const remoteAccessRequested = settingsDraft.allowRemoteConnections;
  const remoteAccessActive = workspace.server.remoteAccessActive ?? false;
  const serverLanUrls = workspace.server.lanUrls ?? [];
  const localServerUrl = workspace.server.localhostUrl ?? workspace.server.baseUrl;
  const localServerOrigin = serverOriginFromBase(localServerUrl);
  const primaryLanUrl = serverLanUrls[0] ?? null;
  const primaryLanOrigin = primaryLanUrl ? serverOriginFromBase(primaryLanUrl) : null;
  const bindingSettingsDirty =
    settingsDraft.preferredServerPort !== (workspace.settings?.preferredServerPort ?? 8876) ||
    settingsDraft.allowRemoteConnections !== (workspace.settings?.allowRemoteConnections ?? false);
  const preferredPortUnavailable = workspace.server.port !== (workspace.settings?.preferredServerPort ?? 8876);
  const localHealthCurl = `curl -sS ${localServerOrigin}/api/health`;
  const localModelsCurl = `curl -sS ${localServerUrl}/models`;
  const remoteHealthCurl = primaryLanOrigin ? `curl -sS ${primaryLanOrigin}/api/health` : null;
  const remoteModelsCurl = primaryLanUrl ? `curl -sS ${primaryLanUrl}/models` : null;

  useEffect(() => {
    if (!previewVariant) {
      return;
    }
    setPreviewControls((current) =>
      current.paramsB === previewVariant.paramsB ? current : { ...current, paramsB: previewVariant.paramsB },
    );
  }, [previewVariant?.paramsB]);

  useEffect(() => {
    if (!threadModelOptions.length) {
      setServerModelKey("");
      return;
    }
    setServerModelKey((current) => {
      if (threadModelOptions.some((option) => option.key === current)) {
        return current;
      }
      return loadedModelOption?.key ?? activeThreadOption?.key ?? threadModelOptions[0].key;
    });
  }, [activeThreadOption?.key, loadedModelOption?.key, serverOptionKeySignature]);

  useEffect(() => {
    if (!threadModelOptions.length) {
      setBenchmarkModelKey("");
      return;
    }
    setBenchmarkModelKey((current) => {
      if (threadModelOptions.some((option) => option.key === current)) {
        return current;
      }
      // Prefer a healthy local library model as the default benchmark target.
      const firstHealthy = workspace.library.find((item) => !item.broken);
      if (firstHealthy) return `library:${firstHealthy.path}`;
      if (workspace.library.length > 0) return `library:${workspace.library[0].path}`;
      return activeThreadOption?.key ?? loadedModelOption?.key ?? threadModelOptions[0].key;
    });
  }, [activeThreadOption?.key, loadedModelOption?.key, serverOptionKeySignature, workspace.library]);

  useEffect(() => {
    if (!benchmarkOption) {
      return;
    }
    setBenchmarkDraft((current) => {
      if (current.modelRef === benchmarkOption.modelRef && current.source === benchmarkOption.source && current.backend === benchmarkOption.backend) {
        return current;
      }
      return {
        ...current,
        modelRef: benchmarkOption.modelRef,
        modelName: benchmarkOption.model,
        source: benchmarkOption.source,
        backend: benchmarkOption.backend,
        path: benchmarkOption.path ?? undefined,
      };
    });
  }, [benchmarkOption?.key]);

  const serverLogRef = useRef<HTMLDivElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll the chat thread whenever:
  //   - the message count changes (new turn)
  //   - the last message's text length changes (streaming tokens)
  //   - the user switches BACK to the chat tab (remount lands at the top
  //     because of unmount/remount; we snap to the latest turn)
  //   - the active chat changes (opening a different thread)
  const lastMessageLength = activeChat?.messages[activeChat.messages.length - 1]?.text?.length ?? 0;
  useEffect(() => {
    if (activeTab !== "chat") return;
    // Use rAF so we run after the DOM has laid out the new messages.
    const handle = requestAnimationFrame(() => {
      if (chatScrollRef.current) {
        chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
      }
    });
    return () => cancelAnimationFrame(handle);
  }, [activeTab, activeChat?.id, activeChat?.messages.length, lastMessageLength]);

  useEffect(() => {
    if (activeTab !== "server" || !backendOnline) return;

    let eventSource: EventSource | null = null;
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect(base: string) {
      if (cancelled) return;
      eventSource = new EventSource(`${base}/api/server/logs/stream`);
      eventSource.onmessage = (event) => {
        try {
          const entry = JSON.parse(event.data);
          if (entry.level === "debug") return;
          setServerLogEntries((prev) => {
            const next = [...prev, entry];
            return next.length > 100 ? next.slice(-100) : next;
          });
          // scroll handled by dedicated useEffect below
        } catch {
          // ignore malformed data
        }
      };
      eventSource.onerror = () => {
        eventSource?.close();
        // Reconnect after 3 seconds
        if (!cancelled) {
          reconnectTimer = setTimeout(() => connect(base), 3000);
        }
      };
    }

    void (async () => {
      const base = await resolveApiBase();
      connect(base);
    })();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      eventSource?.close();
    };
  }, [activeTab, backendOnline]);

  // Poll download status when any download is active
  const hasActiveDownloads = Object.values(activeDownloads).some((d) => d.state === "downloading");
  useEffect(() => {
    if (!hasActiveDownloads || !backendOnline) return;
    const interval = window.setInterval(() => {
      void (async () => {
        try {
          const statuses = await getDownloadStatus();
          setActiveDownloads(buildDownloadStatusMap(statuses));
          // Refresh workspace when a download completes
          if (statuses.some((s) => s.state === "completed")) {
            void refreshWorkspace(activeChatId || undefined);
          }
        } catch { /* ignore */ }
      })();
    }, 2000);
    return () => window.clearInterval(interval);
  }, [hasActiveDownloads, backendOnline, activeChatId]);

  const hasActiveImageDownloads = Object.values(activeImageDownloads).some((download) => download.state === "downloading");
  useEffect(() => {
    if (!hasActiveImageDownloads || !backendOnline) return;
    const interval = window.setInterval(() => {
      void (async () => {
        try {
          const statuses = await getImageDownloadStatus();
          setActiveImageDownloads(buildDownloadStatusMap(statuses));
          if (statuses.some((status) => status.state === "completed")) {
            void refreshImageData();
          }
        } catch {
          // keep the last known state until the next poll
        }
      })();
    }, 2000);
    return () => window.clearInterval(interval);
  }, [hasActiveImageDownloads, backendOnline]);

  async function handleDownloadModel(repo: string) {
    try {
      setActiveDownloads((prev) => ({ ...prev, [repo]: pendingDownloadStatus(repo, prev[repo]) }));
      const download = await downloadModel(repo);
      setActiveDownloads((prev) => ({ ...prev, [repo]: download }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
      setActiveDownloads((prev) => ({ ...prev, [repo]: failedDownloadStatus(repo, String(err)) }));
    }
  }

  async function handleCancelModelDownload(repo: string) {
    try {
      const download = await cancelDownload(repo);
      setActiveDownloads((prev) => ({ ...prev, [repo]: download }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not pause download");
    }
  }

  async function handleImageDownload(repo: string) {
    try {
      setActiveImageDownloads((prev) => ({ ...prev, [repo]: pendingDownloadStatus(repo, prev[repo]) }));
      const download = await downloadImageModel(repo);
      setActiveImageDownloads((prev) => ({ ...prev, [repo]: download }));
      void refreshImageData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Image download failed");
      setActiveImageDownloads((prev) => ({ ...prev, [repo]: failedDownloadStatus(repo, String(err)) }));
    }
  }

  async function handleCancelImageDownload(repo: string) {
    try {
      const download = await cancelImageDownload(repo);
      setActiveImageDownloads((prev) => ({ ...prev, [repo]: download }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not pause image download");
    }
  }

  // Auto-scroll server log to bottom only when already near the bottom
  const [serverLogAtBottom, setServerLogAtBottom] = useState(true);

  function handleServerLogScroll() {
    const el = serverLogRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    setServerLogAtBottom(atBottom);
  }

  function scrollServerLogToBottom() {
    if (serverLogRef.current) {
      serverLogRef.current.scrollTop = serverLogRef.current.scrollHeight;
      setServerLogAtBottom(true);
    }
  }

  useEffect(() => {
    if (serverLogAtBottom && serverLogRef.current) {
      serverLogRef.current.scrollTop = serverLogRef.current.scrollHeight;
    }
  }, [serverLogEntries, serverLogAtBottom]);

  function updateConversionDraft<K extends keyof typeof conversionDraft>(key: K, value: (typeof conversionDraft)[K]) {
    setConversionDraft((current) => ({
      ...current,
      [key]: value,
    }));
  }

  function updateLaunchSetting<K extends keyof LaunchPreferences>(key: K, value: LaunchPreferences[K]) {
    setLaunchSettings((current) => ({
      ...current,
      [key]: value,
    }));
  }

  function applyImageRatioPreset(presetId: (typeof IMAGE_RATIO_PRESETS)[number]["id"]) {
    const preset = IMAGE_RATIO_PRESETS.find((item) => item.id === presetId);
    if (!preset) return;
    setImageRatioId(presetId);
    setImageWidth(preset.width);
    setImageHeight(preset.height);
  }

  function applyImageQuality(presetId: ImageQualityPreset) {
    const preset = IMAGE_QUALITY_PRESETS.find((item) => item.id === presetId);
    if (!preset) return;
    setImageQualityPreset(presetId);
    setImageSteps(preset.steps);
    setImageGuidance(preset.guidance);
  }

  function openImageStudio(modelId?: string) {
    if (modelId) {
      setSelectedImageModelId(modelId);
    }
    setActiveTab("image-studio");
    setError(null);
  }

  function openImageGallery(modelId?: string) {
    if (modelId) {
      setImageGalleryModelFilter(modelId);
    }
    setActiveTab("image-gallery");
    setError(null);
  }

  function resetImageGalleryFilters() {
    setImageGallerySearchInput("");
    setImageGalleryModelFilter("all");
    setImageGalleryRuntimeFilter("all");
    setImageGalleryOrientationFilter("all");
    setImageGallerySort("newest");
  }

  function hydrateImageFormFromArtifact(artifact: ImageOutputArtifact, randomizeSeed = false) {
    setSelectedImageModelId(artifact.modelId);
    setImagePrompt(artifact.prompt);
    setImageNegativePrompt(artifact.negativePrompt ?? "");
    setImageWidth(artifact.width);
    setImageHeight(artifact.height);
    setImageSteps(artifact.steps);
    setImageGuidance(artifact.guidance);
    setImageBatchSize(1);
    const ratioPreset = IMAGE_RATIO_PRESETS.find(
      (preset) => preset.width === artifact.width && preset.height === artifact.height,
    );
    if (ratioPreset) {
      setImageRatioId(ratioPreset.id);
    }
    const qualityPreset = IMAGE_QUALITY_PRESETS.find(
      (preset) => preset.steps === artifact.steps && preset.guidance === artifact.guidance,
    );
    if (qualityPreset) {
      setImageQualityPreset(qualityPreset.id);
    }
    setImageUseRandomSeed(randomizeSeed);
    setImageSeedInput(randomizeSeed ? "" : String(artifact.seed));
    openImageStudio(artifact.modelId);
  }

  async function submitImageGeneration(overrides?: {
    modelId?: string;
    prompt?: string;
    negativePrompt?: string;
    width?: number;
    height?: number;
    steps?: number;
    guidance?: number;
    batchSize?: number;
    qualityPreset?: ImageQualityPreset;
    seed?: number | null;
  }) {
    const modelId = overrides?.modelId ?? selectedImageModelId;
    const prompt = (overrides?.prompt ?? imagePrompt).trim();
    if (!modelId) {
      setError("Choose an image model first.");
      return;
    }
    if (!prompt) {
      setError("Write a prompt before generating.");
      return;
    }
    const seed =
      overrides && "seed" in overrides
        ? overrides.seed ?? null
        : imageUseRandomSeed
          ? null
          : (() => {
              const parsed = Number.parseInt(imageSeedInput, 10);
              return Number.isFinite(parsed) ? parsed : null;
            })();

    const selectedVariantForRequest =
      findImageVariantById(imageCatalog, modelId) ??
      (selectedImageVariant?.id === modelId ? selectedImageVariant : null);
    const requestNeedsPipelineLoad =
      !!selectedVariantForRequest &&
      selectedVariantForRequest.availableLocally &&
      imageRuntimeStatus.realGenerationAvailable &&
      imageRuntimeStatus.loadedModelRepo !== selectedVariantForRequest.repo;
    setShowImageGenerationModal(true);
    setImageGenerationStartedAt(Date.now());
    setImageGenerationError(null);
    setImageGenerationArtifacts([]);
    setSelectedImageGenerationArtifactId(null);
    setImageGenerationRunInfo({
      modelName: selectedVariantForRequest?.name ?? "Image model",
      prompt,
      batchSize: overrides?.batchSize ?? imageBatchSize,
      steps: overrides?.steps ?? imageSteps,
      needsPipelineLoad: requestNeedsPipelineLoad,
    });
    setImageBusyLabel(
      requestNeedsPipelineLoad
        ? `Loading ${selectedVariantForRequest?.name ?? "image model"} into memory...`
        : "Generating image...",
    );
    try {
      const response = await generateImage({
        modelId,
        prompt,
        negativePrompt: overrides?.negativePrompt ?? imageNegativePrompt,
        width: overrides?.width ?? imageWidth,
        height: overrides?.height ?? imageHeight,
        steps: overrides?.steps ?? imageSteps,
        guidance: overrides?.guidance ?? imageGuidance,
        batchSize: overrides?.batchSize ?? imageBatchSize,
        qualityPreset: overrides?.qualityPreset ?? imageQualityPreset,
        seed,
      });
      setImageOutputs(response.outputs);
      if (response.runtime) {
        setImageRuntimeStatus(response.runtime);
      }
      setImageGenerationArtifacts(response.artifacts);
      setSelectedImageGenerationArtifactId(response.artifacts[0]?.artifactId ?? null);
      if (seed !== null && !imageUseRandomSeed && !overrides) {
        setImageSeedInput(String(seed));
      }
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Image generation failed.";
      setError(message);
      setImageGenerationError(message);
    } finally {
      setImageBusyLabel(null);
      setImageGenerationStartedAt(null);
    }
  }

  async function handlePreloadImageModel(variant?: ImageModelVariant | null) {
    if (!variant) {
      setError("Choose an installed image model first.");
      return;
    }
    setImageBusyLabel(`Loading ${variant.name} into memory...`);
    try {
      const runtime = await preloadImageModel(variant.id);
      setImageRuntimeStatus(runtime);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not preload the image model.");
    } finally {
      setImageBusyLabel(null);
    }
  }

  async function handleUnloadImageModel(variant?: ImageModelVariant | null) {
    setImageBusyLabel(`Unloading ${(variant?.name ?? loadedImageVariant?.name ?? "image model")} from memory...`);
    try {
      const runtime = await unloadImageModel(variant?.id);
      setImageRuntimeStatus(runtime);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not unload the image model.");
    } finally {
      setImageBusyLabel(null);
    }
  }

  async function handleDeleteImageArtifact(artifactId: string) {
    try {
      const response = await deleteImageOutput(artifactId);
      setImageOutputs(response.outputs);
      const nextArtifacts = imageGenerationArtifacts.filter((artifact) => artifact.artifactId !== artifactId);
      setImageGenerationArtifacts(nextArtifacts);
      setSelectedImageGenerationArtifactId((current) => {
        if (current && nextArtifacts.some((artifact) => artifact.artifactId === current)) {
          return current;
        }
        return nextArtifacts[0]?.artifactId ?? null;
      });
      if (showImageGenerationModal && nextArtifacts.length === 0 && !imageBusy) {
        setShowImageGenerationModal(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete image output.");
    }
  }

  async function handleVaryImageSeed(artifact: ImageOutputArtifact) {
    const matchedQualityPreset =
      IMAGE_QUALITY_PRESETS.find(
        (preset) => preset.steps === artifact.steps && preset.guidance === artifact.guidance,
      )?.id ?? imageQualityPreset;
    hydrateImageFormFromArtifact(artifact, true);
    await submitImageGeneration({
      modelId: artifact.modelId,
      prompt: artifact.prompt,
      negativePrompt: artifact.negativePrompt ?? "",
      width: artifact.width,
      height: artifact.height,
      steps: artifact.steps,
      guidance: artifact.guidance,
      batchSize: 1,
      qualityPreset: matchedQualityPreset,
      seed: Math.floor(Math.random() * 2147483647),
    });
  }

  function handleUseSameImageSettings(artifact: ImageOutputArtifact, closeModal = false) {
    hydrateImageFormFromArtifact(artifact);
    if (closeModal) {
      setShowImageGenerationModal(false);
    }
  }

  function renderImageOutputCard(artifact: ImageOutputArtifact) {
    const artifactVariant = findImageVariantById(imageCatalog, artifact.modelId);
    const friendlyRuntimeNote = formatImageAccessError(artifact.runtimeNote, artifactVariant);
    const artifactNeedsGatedAccess = isGatedImageAccessError(artifact.runtimeNote);
    return (
      <article key={artifact.artifactId} className="image-output-card">
        <img src={artifact.previewUrl} alt={artifact.prompt} className="image-output-preview" />
        <div className="image-output-card-body">
          <div className="image-output-card-head">
            <strong>{artifact.modelName}</strong>
            <span className="badge muted">{formatImageTimestamp(artifact.createdAt)}</span>
          </div>
          {artifact.runtimeLabel ? (
            <div className="chip-row">
              <span className="badge subtle">{artifact.runtimeLabel}</span>
              <span className="badge muted">{imageOrientation(artifact.width, artifact.height)}</span>
            </div>
          ) : null}
          <p className="image-output-prompt">{artifact.prompt}</p>
          {artifact.runtimeNote ? <p className="muted-text">{friendlyRuntimeNote}</p> : null}
          {artifactNeedsGatedAccess && artifactVariant ? (
            <div className="button-row">
              <button className="secondary-button" type="button" onClick={() => void handleOpenExternalUrl(artifactVariant.link)}>
                Hugging Face
              </button>
              <button className="secondary-button" type="button" onClick={() => setActiveTab("settings")}>
                Settings
              </button>
            </div>
          ) : null}
          <div className="image-output-meta">
            <span>{artifact.width} x {artifact.height}</span>
            <span>{artifact.steps} steps</span>
            <span>CFG {artifact.guidance}</span>
            <span>Seed {artifact.seed}</span>
            <span>{number(artifact.durationSeconds)}s</span>
          </div>
          <div className="button-row">
            <button className="secondary-button" type="button" onClick={() => handleUseSameImageSettings(artifact)}>
              Use Same Settings
            </button>
            <button className="secondary-button" type="button" onClick={() => void handleVaryImageSeed(artifact)} disabled={imageBusy}>
              Vary Seed
            </button>
          </div>
          <div className="button-row">
            <button
              className="secondary-button"
              type="button"
              onClick={() => void handleOpenExternalUrl(artifact.imagePath ?? artifact.previewUrl)}
            >
              Open
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={() => artifact.imagePath ? void handleRevealPath(artifact.imagePath) : void handleOpenExternalUrl(artifact.previewUrl)}
            >
              Reveal
            </button>
            <button className="secondary-button danger-button" type="button" onClick={() => void handleDeleteImageArtifact(artifact.artifactId)}>
              Delete
            </button>
          </div>
        </div>
      </article>
    );
  }

  function prepareCatalogConversion(model: ModelVariant) {
    const matchingItem = findLibraryItemForVariant(convertibleLibrary, model);
    if (matchingItem) {
      prepareLibraryConversion(matchingItem);
      return;
    }
    setActiveTab("conversion");
    setLastConversion(null);
  }

  function prepareLibraryConversion(item: LibraryItem, resolvedPath?: string) {
    const isGguf = libraryItemFormat(item).toUpperCase() === "GGUF";
    setConversionDraft({
      modelRef: item.name,
      path: resolvedPath ?? item.path,
      hfRepo: isGguf ? "" : item.name,
      outputPath: "",
      quantize: true,
      qBits: 4,
      qGroupSize: 64,
      dtype: "float16",
    });
    setLastConversion(null);
    setActiveTab("conversion");
  }

  function loadPayloadFromVariant(variant: ModelVariant, nextTab?: TabId) {
    const localItem = findLibraryItemForVariant(workspace.library, variant);
    if (localItem) {
      return {
        modelRef: localItem.name,
        modelName: localItem.name,
        source: "library",
        backend: libraryItemBackend(localItem),
        path: localItem.path,
        nextTab,
      };
    }
    return {
      modelRef: variant.id,
      modelName: variant.name,
      source: "catalog",
      backend: variant.backend,
      nextTab,
    };
  }

  function sessionModelPayload(session?: ChatSession | null) {
    if (session?.modelRef) {
      return {
        modelRef: session.modelRef,
        modelName: session.model,
        source: session.modelSource ?? "catalog",
        path: session.modelPath ?? undefined,
        backend: session.modelBackend ?? "auto",
      };
    }
    if (workspace.runtime.loadedModel) {
      return {
        modelRef: workspace.runtime.loadedModel.ref,
        modelName: workspace.runtime.loadedModel.name,
        source: workspace.runtime.loadedModel.source,
        path: workspace.runtime.loadedModel.path ?? undefined,
        backend: workspace.runtime.loadedModel.backend,
      };
    }
    if (defaultChatVariant) {
      return {
        modelRef: defaultChatVariant.id,
        modelName: defaultChatVariant.name,
        source: "catalog",
        backend: defaultChatVariant.backend,
      };
    }
    return null;
  }

  function mergeSessionMetadata(session: ChatSession, patch: Partial<ChatSession>): ChatSession {
    return { ...session, ...patch };
  }

  async function persistSessionChanges(sessionId: string, patch: Partial<ChatSession>) {
    setWorkspace((current) => ({
      ...current,
      chatSessions: current.chatSessions.map((session) =>
        session.id === sessionId ? mergeSessionMetadata(session, patch) : session,
      ),
    }));

    if (!backendOnline) {
      return;
    }

    try {
      const session = await updateSession(sessionId, {
        title: patch.title,
        model: patch.model,
        modelRef: patch.modelRef,
        modelSource: patch.modelSource,
        modelPath: patch.modelPath,
        modelBackend: patch.modelBackend,
        pinned: patch.pinned,
      });
      setWorkspace((current) => ({
        ...current,
        chatSessions: upsertSession(current.chatSessions, session),
      }));
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to update thread.");
    }
  }

  async function handleConvertModel() {
    const modelRef = conversionDraft.modelRef.trim();
    const path = conversionDraft.path.trim();
    const hfRepo = conversionDraft.hfRepo.trim();
    const outputPath = conversionDraft.outputPath.trim();

    if (!modelRef && !path) {
      setError("Enter a model reference or a local path before starting conversion.");
      return;
    }

    setBusyAction("Converting model...");
    setConversionStartedAt(Date.now());
    setConversionError(null);
    setShowConversionModal(true);

    try {
      const response = await convertModel({
        modelRef: modelRef || undefined,
        path: path || undefined,
        hfRepo: hfRepo || undefined,
        outputPath: outputPath || undefined,
        quantize: conversionDraft.quantize,
        qBits: conversionDraft.qBits,
        qGroupSize: conversionDraft.qGroupSize,
        dtype: conversionDraft.dtype,
      });
      setLastConversion(response.conversion);
      setWorkspace((current) =>
        syncRuntime(
          {
            ...current,
            library: response.library,
          },
          response.runtime,
        ),
      );
      await refreshWorkspace(activeChatId || undefined);
    } catch (actionError) {
      const message = actionError instanceof Error ? actionError.message : "Failed to convert model.";
      setError(message);
      setConversionError(message);
    } finally {
      setBusyAction(null);
      setConversionStartedAt(null);
    }
  }

  async function handleLoadModel(payload: {
    modelRef: string;
    modelName?: string;
    source?: string;
    backend?: string;
    path?: string;
    nextTab?: TabId;
  }) {
    setError(null);
    setBusyAction("Loading model...");
    try {
      const loadPayload = {
        modelRef: payload.modelRef,
        modelName: payload.modelName,
        source: payload.source ?? "catalog",
        backend: payload.backend ?? "auto",
        path: payload.path,
        cacheBits: launchSettings.cacheBits,
        fp16Layers: launchSettings.fp16Layers,
        fusedAttention: launchSettings.fusedAttention,
        cacheStrategy: launchSettings.cacheStrategy,
        fitModelInMemory: launchSettings.fitModelInMemory,
        contextTokens: launchSettings.contextTokens,
      };

      let loadSucceeded = false;
      let loadErrorMessage: string | null = null;
      try {
        const runtime = await loadModel(loadPayload);
        setWorkspace((current) => syncRuntime(current, runtime));
        loadSucceeded = true;
      } catch (loadErr) {
        // Capture the actual error detail from the backend
        loadErrorMessage = loadErr instanceof Error ? loadErr.message : String(loadErr);
        // The load request may have timed out while the backend was still
        // loading. Poll the runtime status for up to 15 minutes — if the
        // backend is still actively loading, keep waiting. If loading
        // finished (null) but the model isn't loaded, it's a real failure.
        // 450 attempts × 2s = 15 min, matching the HTTP request ceiling.
        for (let attempt = 0; attempt < 450; attempt++) {
          await new Promise((r) => setTimeout(r, 2000));
          try {
            const ws = await getWorkspace();
            if (ws.runtime.loadedModel?.ref === payload.modelRef || ws.runtime.loadedModel?.runtimeTarget === payload.path) {
              setWorkspace(ws);
              loadSucceeded = true;
              loadErrorMessage = null;
              break;
            }
            // Backend is still actively loading — keep waiting + surface progress
            if (ws.server.loading) {
              setWorkspace(ws);
              continue;
            }
            // Backend finished loading but model didn't match — real failure
            break;
          } catch {
            // backend unreachable, keep trying
          }
        }
      }

      if (loadSucceeded) {
        await refreshWorkspace(activeChatId || undefined);
        if (payload.nextTab) {
          setActiveTab(payload.nextTab);
        }
      } else {
        const detail = loadErrorMessage || "The model could not be loaded. Check the server logs for details.";
        setError(`Failed to load ${payload.modelName ?? payload.modelRef}: ${detail}`);
      }
    } catch (actionError) {
      const detail = actionError instanceof Error ? actionError.message : "Unknown error";
      console.error("[handleLoadModel] Load failed for", payload.modelRef, "—", detail, actionError);
      setError(`Failed to load ${payload.modelName ?? payload.modelRef}: ${detail}`);
    } finally {
      setBusyAction(null);
    }
  }

  async function handleLoadLibraryItem(item: LibraryItem, nextTab: TabId) {
    await handleLoadModel({
      modelRef: item.name,
      modelName: item.name,
      source: "library",
      backend: libraryItemBackend(item),
      path: item.path,
      nextTab,
    });
  }

  function openModelSelector(action: "chat" | "server" | "thread", preselectedKey?: string) {
    setLaunchModelSearch("");
    let normalizedKey = preselectedKey;
    if (normalizedKey?.startsWith("catalog:")) {
      const modelRef = normalizedKey.slice("catalog:".length);
      const variant = findVariantForReference(workspace.featuredModels, modelRef, undefined);
      const localItem = variant ? findLibraryItemForVariant(workspace.library, variant) : null;
      if (localItem) {
        normalizedKey = `library:${localItem.path}`;
      }
    }
    setPendingLaunch({ action, preselectedKey: normalizedKey });
  }

  async function confirmLaunch(selectedKey: string) {
    if (!pendingLaunch) return;
    const option = threadModelOptions.find((o) => o.key === selectedKey);
    if (!option) return;
    const { action } = pendingLaunch;
    setPendingLaunch(null);
    setError(null);

    // Switch to target tab immediately
    if (action === "chat" || action === "thread") {
      setActiveTab("chat");
    } else if (action === "server") {
      setActiveTab("server");
    }

    if (action === "thread") {
      const variant = findVariantForReference(workspace.featuredModels, option.modelRef, option.model);
      if (variant) {
        await handleStartThreadWithVariant(variant);
      } else {
        await handleLoadModel({
          modelRef: option.modelRef,
          modelName: option.model,
          source: option.source,
          backend: option.backend,
          path: option.path ?? undefined,
        });
      }
    } else if (action === "chat") {
      if (activeChat) {
        await persistSessionChanges(activeChat.id, {
          model: option.model,
          modelRef: option.modelRef,
          modelSource: option.source,
          modelPath: option.path ?? null,
          modelBackend: option.backend,
          updatedAt: new Date().toLocaleString(),
        });
      }
      await handleLoadModel({
        modelRef: option.modelRef,
        modelName: option.model,
        source: option.source,
        backend: option.backend,
        path: option.path ?? undefined,
      });
    } else if (action === "server") {
      await handleLoadModel({
        modelRef: option.modelRef,
        modelName: option.model,
        source: option.source,
        backend: option.backend,
        path: option.path ?? undefined,
      });
    }
  }

  async function handleRevealPath(path: string) {
    try {
      if (backendOnline) {
        await revealModelPath(path);
        return;
      }
    } catch {
      // Backend reveal failed, try Tauri fallback
    }
    // Tauri fallback: use invoke to open path
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      // Use shell open via Tauri plugin or just open the parent directory
      const parentDir = path.replace(/\/[^/]+$/, "");
      await tauriInvoke("plugin:shell|open", { path: parentDir });
    } catch {
      // Last resort: just set error
      setError("Could not open file location. Try navigating manually to: " + path);
    }
  }

  async function handleOpenExternalUrl(url: string) {
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      await tauriInvoke("plugin:shell|open", { path: url });
      return;
    } catch {
      // Fall through to browser open for non-Tauri runs.
    }
    try {
      const opened = window.open(url, "_blank", "noopener,noreferrer");
      if (opened) {
        return;
      }
    } catch {
      // Ignore and show the manual fallback below.
    }
    setError(`Could not open link. Try opening this URL manually: ${url}`);
  }

  async function handleDeleteModel(item: LibraryItem) {
    const confirmed = window.confirm(
      `Delete "${item.name}"?\n\n` +
      `This will permanently remove the files at:\n${item.path}\n\n` +
      `This action cannot be undone.`,
    );
    if (!confirmed) return;
    setBusyAction("Deleting model...");
    try {
      const result = await deleteModelPath(item.path);
      setWorkspace((current) => ({ ...current, library: result.library }));
      await refreshWorkspace(activeChatId || undefined);
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to delete model.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleUnloadModel(ref?: string) {
    setBusyAction("Unloading model...");
    try {
      const runtime = await unloadModel(ref);
      setWorkspace((current) => syncRuntime(current, runtime));
      await refreshWorkspace(activeChatId || undefined);
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to unload model.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleUnloadWarmModel(ref: string) {
    await handleUnloadModel(ref);
  }

  async function handleCreateSession() {
    const fallbackModel = sessionModelPayload(activeChat);
    if (!backendOnline) {
      const localSession: ChatSession = {
        id: `local-${Date.now()}`,
        title: "New chat",
        updatedAt: new Date().toLocaleString(),
        pinned: false,
        model: fallbackModel?.modelName ?? "Choose a model",
        modelRef: fallbackModel?.modelRef ?? null,
        modelSource: fallbackModel?.source ?? "catalog",
        modelPath: fallbackModel?.path ?? null,
        modelBackend: fallbackModel?.backend ?? "auto",
        cacheLabel:
          workspace.runtime.loadedModel ? loadedModelCacheLabel : launchCacheLabel,
        messages: [],
      };
      setWorkspace((current) => ({ ...current, chatSessions: [localSession, ...current.chatSessions] }));
      setActiveChatId(localSession.id);
      setThreadTitleDraft(localSession.title);
      return;
    }

    try {
      const session = await createSession("New chat");
      setWorkspace((current) => ({ ...current, chatSessions: upsertSession(current.chatSessions, session) }));
      setActiveChatId(session.id);
      setThreadTitleDraft(session.title);
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to create session.");
    }
  }

  async function handleRenameActiveThread() {
    if (!activeChat) {
      return;
    }
    const nextTitle = threadTitleDraft.trim();
    if (!nextTitle || nextTitle === activeChat.title) {
      setThreadTitleDraft(activeChat.title);
      return;
    }
    await persistSessionChanges(activeChat.id, { title: nextTitle, updatedAt: new Date().toLocaleString() });
  }

  async function handleToggleThreadPin(session: ChatSession) {
    await persistSessionChanges(session.id, {
      pinned: !session.pinned,
      updatedAt: new Date().toLocaleString(),
    });
  }

  async function handleSelectThreadModel(nextKey: string) {
    const nextOption = threadModelOptions.find((option) => option.key === nextKey);
    if (!activeChat || !nextOption) {
      return;
    }
    await persistSessionChanges(activeChat.id, {
      model: nextOption.model,
      modelRef: nextOption.modelRef,
      modelSource: nextOption.source,
      modelPath: nextOption.path ?? null,
      modelBackend: nextOption.backend,
      updatedAt: new Date().toLocaleString(),
    });
  }

  async function handleLoadActiveThreadModel() {
    const modelSelection = sessionModelPayload(activeChat);
    if (!modelSelection) {
      setError("Choose a model for this thread before loading it.");
      return;
    }
    await handleLoadModel({
      modelRef: modelSelection.modelRef,
      modelName: modelSelection.modelName,
      source: modelSelection.source,
      backend: modelSelection.backend,
      path: modelSelection.path,
      nextTab: "chat",
    });
  }

  function handleSelectServerModel(nextKey: string) {
    setServerModelKey(nextKey);
  }

  async function handleLoadServerModel() {
    if (!selectedServerOption) {
      setError("Choose a model for the server before loading it.");
      return;
    }
    await handleLoadModel({
      modelRef: selectedServerOption.modelRef,
      modelName: selectedServerOption.model,
      source: selectedServerOption.source,
      backend: selectedServerOption.backend,
      path: selectedServerOption.path ?? undefined,
      nextTab: "server",
    });
  }

  function updateBenchmarkDraft<K extends keyof BenchmarkRunPayload>(key: K, value: BenchmarkRunPayload[K]) {
    setBenchmarkDraft((current) => ({
      ...current,
      [key]: value,
    }));
  }

  async function handleRunBenchmark() {
    const promptPreset = BENCHMARK_PROMPTS.find((item) => item.id === benchmarkPromptId) ?? BENCHMARK_PROMPTS[0];
    if (!benchmarkOption) {
      setError("Choose a model before running a benchmark.");
      return;
    }

    setBusyAction("Running benchmark...");
    setBenchmarkStartedAt(Date.now());
    setShowBenchmarkModal(true);
    setBenchmarkError(null);

    try {
      const response = await runBenchmark({
        ...benchmarkDraft,
        modelRef: benchmarkOption.modelRef,
        modelName: benchmarkOption.model,
        source: benchmarkOption.source,
        backend: benchmarkOption.backend,
        path: benchmarkOption.path ?? undefined,
        prompt: promptPreset.prompt,
        label: `${benchmarkOption.model} / ${benchmarkDraft.cacheStrategy === "native" ? "Native f16" : `${benchmarkDraft.cacheStrategy} ${benchmarkDraft.cacheBits}-bit ${benchmarkDraft.fp16Layers}+${benchmarkDraft.fp16Layers}`} / ${Math.round(benchmarkDraft.contextTokens / 1024)}K ctx`,
      });
      setWorkspace((current) =>
        syncRuntime(
          {
            ...current,
            benchmarks: response.benchmarks,
          },
          response.runtime,
        ),
      );
      setSelectedBenchmarkId(response.result.id);
      setCompareBenchmarkId((current) => (current === response.result.id ? selectedBenchmark?.id ?? current : current));
    } catch (actionError) {
      const message = actionError instanceof Error ? actionError.message : "Failed to run benchmark.";
      setError(message);
      setBenchmarkError(message);
    } finally {
      setBusyAction(null);
      setBenchmarkStartedAt(null);
    }
  }

  function handleAddDirectory() {
    const path = newDirectoryPath.trim();
    if (!path) {
      setError("Enter a directory path before adding it.");
      return;
    }
    const fallbackLabel = path.split("/").filter(Boolean).pop() || "Custom directory";
    const nextDirectory: ModelDirectorySetting = {
      id: `user-${Date.now()}`,
      label: newDirectoryLabel.trim() || fallbackLabel,
      path,
      enabled: true,
      source: "user",
    };
    setSettingsDraft((current) => ({
      ...current,
      modelDirectories: [...current.modelDirectories, nextDirectory],
    }));
    setNewDirectoryLabel("");
    setNewDirectoryPath("");
  }

  function handleToggleDirectory(directoryId: string) {
    setSettingsDraft((current) => ({
      ...current,
      modelDirectories: current.modelDirectories.map((directory) =>
        directory.id === directoryId ? { ...directory, enabled: !directory.enabled } : directory,
      ),
    }));
  }

  function handleRemoveDirectory(directoryId: string) {
    setSettingsDraft((current) => ({
      ...current,
      modelDirectories: current.modelDirectories.filter((directory) => directory.id !== directoryId),
    }));
  }

  async function pickDirectory(currentPath?: string): Promise<string | null> {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      let defaultPath = currentPath?.trim() || undefined;
      if (!defaultPath) {
        try {
          const { homeDir } = await import("@tauri-apps/api/path");
          defaultPath = await homeDir();
        } catch {
          /* ignore — picker will open at its default */
        }
      }
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Select model directory",
        defaultPath,
      });
      return typeof selected === "string" && selected ? selected : null;
    } catch (err) {
      console.error("Folder picker failed", err);
      return null;
    }
  }

  function handleUpdateDirectoryPath(directoryId: string, nextPath: string) {
    setSettingsDraft((current) => ({
      ...current,
      modelDirectories: current.modelDirectories.map((directory) =>
        directory.id === directoryId ? { ...directory, path: nextPath } : directory,
      ),
    }));
  }

  async function handleSaveSettings() {
    setBusyAction("Saving settings...");
    try {
      const prevSettings = workspace.settings ?? { preferredServerPort: 8876, allowRemoteConnections: false };
      const response = await updateSettings({
        modelDirectories: settingsDraft.modelDirectories,
        preferredServerPort: settingsDraft.preferredServerPort,
        allowRemoteConnections: settingsDraft.allowRemoteConnections,
        autoStartServer: settingsDraft.autoStartServer,
        launchPreferences: launchSettings,
        remoteProviders: (settingsDraft.remoteProviders ?? []).map((p) => ({
          id: p.id,
          label: p.label,
          apiBase: p.apiBase,
          apiKey: p.apiKey ?? "",
          model: p.model,
        })),
        // Only send token when user typed a new one; empty means "no change"
        ...(settingsDraft.huggingFaceToken
          ? { huggingFaceToken: settingsDraft.huggingFaceToken }
          : {}),
        // Send dataDirectory only when changed; empty string resets to default.
        ...(settingsDraft.dataDirectory !== (workspace.settings?.dataDirectory ?? "")
          ? { dataDirectory: settingsDraft.dataDirectory }
          : {}),
      });
      const settings = response.settings;
      setSettingsDraft(settingsDraftFromWorkspace(settings));
      setLaunchSettings(settings.launchPreferences);
      await refreshWorkspace(activeChatId || undefined);
      if (response.restartRequired) {
        setDataDirRestartPrompt({ migration: response.migrationSummary ?? null });
      }
      const restartRequired =
        settings.preferredServerPort !== prevSettings.preferredServerPort ||
        settings.allowRemoteConnections !== prevSettings.allowRemoteConnections;
      if (restartRequired && tauriBackend?.managedByTauri) {
        setBusyAction("Restarting server to apply changes...");
        const runtimeInfo = await restartManagedBackend();
        if (runtimeInfo) {
          setTauriBackend(runtimeInfo);
          if (runtimeInfo.started) {
            await refreshWorkspace(activeChatId || undefined);
            await refreshImageData();
            setError(null);
            setTauriBackend((await getTauriBackendInfo(true)) ?? runtimeInfo);
            setBackendOnline(true);
          }
        }
      }
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to save settings.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStopServer() {
    setBusyAction("Stopping server...");

    try {
      if (tauriBackend?.managedByTauri) {
        const runtimeInfo = await stopManagedBackend();
        if (!runtimeInfo) {
          throw new Error("The desktop sidecar could not be stopped.");
        }
        setTauriBackend(runtimeInfo);
        setBackendOnline(false);
        setWorkspace((current) => syncStoppedBackend(current, runtimeInfo));
      } else {
        // Standalone mode — tell the API server to shut itself down
        try {
          await shutdownServer();
        } catch {
          // Expected: server shuts down before response completes
        }
        setBackendOnline(false);
        setWorkspace((current) => syncStoppedBackend(current, null));
      }
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to stop the API service.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRestartServer() {
    setBusyAction("Restarting server...");

    try {
      if (tauriBackend?.managedByTauri) {
        const runtimeInfo = await restartManagedBackend();
        if (!runtimeInfo) {
          throw new Error("The desktop sidecar could not be restarted.");
        }
        setTauriBackend(runtimeInfo);
        if (!runtimeInfo.started) {
          throw new Error(runtimeInfo.startupError ?? "The API service did not come back online.");
        }
        await refreshWorkspace(activeChatId || undefined);
        await refreshImageData();
        setError(null);
        setTauriBackend((await getTauriBackendInfo(true)) ?? runtimeInfo);
        setBackendOnline(true);
      } else {
        // Standalone mode — shut down, then poll until back online
        try {
          await shutdownServer();
        } catch {
          // Expected: server shuts down before response completes
        }
        setBackendOnline(false);
        // Poll for the server to come back (user must restart manually)
        let came_back = false;
        for (let i = 0; i < 15; i++) {
          await new Promise((r) => setTimeout(r, 2000));
          const online = await checkBackend();
          if (online) {
            came_back = true;
            break;
          }
        }
        if (came_back) {
          await refreshWorkspace(activeChatId || undefined);
          await refreshImageData();
          setError(null);
          setBackendOnline(true);
        } else {
          setError("Server was stopped. Please restart it manually, then it will reconnect.");
        }
      }
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to restart the API service.");
    } finally {
      setBusyAction(null);
    }
  }

  function threadPatchFromVariant(variant: ModelVariant): Pick<
    ChatSession,
    "model" | "modelRef" | "modelSource" | "modelPath" | "modelBackend" | "cacheLabel" | "updatedAt"
  > {
    const localItem = findLibraryItemForVariant(workspace.library, variant);
    return {
      model: localItem?.name ?? variant.name,
      modelRef: localItem?.name ?? variant.id,
      modelSource: localItem ? "library" : "catalog",
      modelPath: localItem?.path ?? null,
      modelBackend: localItem ? libraryItemBackend(localItem, variant) : variant.backend,
      cacheLabel: launchCacheLabel,
      updatedAt: new Date().toLocaleString(),
    };
  }

  async function handleApplyVariantToActiveThread(variant: ModelVariant) {
    if (!activeChat) {
      return;
    }
    await persistSessionChanges(activeChat.id, threadPatchFromVariant(variant));
    setActiveTab("chat");
  }

  async function handleStartThreadWithVariant(variant: ModelVariant) {

    if (!backendOnline) {
      const localSession: ChatSession = {
        id: `local-${Date.now()}`,
        title: "New chat",
        pinned: false,
        messages: [],
        ...threadPatchFromVariant(variant),
      };
      setWorkspace((current) => ({ ...current, chatSessions: upsertSession(current.chatSessions, localSession) }));
      setActiveChatId(localSession.id);
      setThreadTitleDraft(localSession.title);
      setActiveTab("chat");
      return;
    }

    try {
      const session = await createSession("New chat");
      const updated = await updateSession(session.id, threadPatchFromVariant(variant));
      setWorkspace((current) => ({
        ...current,
        chatSessions: upsertSession(current.chatSessions, updated),
      }));
      setActiveChatId(updated.id);
      setThreadTitleDraft(updated.title);
      setActiveTab("chat");
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to start a new thread.");
    }
  }

  const DOC_EXTENSIONS = new Set([
    "pdf", "txt", "md", "rst", "csv", "json", "yaml", "yml", "toml",
    "py", "js", "ts", "tsx", "jsx", "rs", "go", "java", "c", "cpp",
    "h", "hpp", "rb", "php", "swift", "kt", "html", "css", "sh",
  ]);

  async function handleChatFileDrop(files: FileList) {
    if (!files?.length || !activeChat) return;
    for (const file of Array.from(files)) {
      if (file.type.startsWith("image/")) {
        // Image -> attach to next message
        if (file.size > 10 * 1024 * 1024) {
          setError("Image must be under 10MB");
          continue;
        }
        await new Promise<void>((resolve) => {
          const reader = new FileReader();
          reader.onload = () => {
            const b64 = (reader.result as string).split(",")[1];
            if (b64) setPendingImages((prev) => [...prev, b64]);
            resolve();
          };
          reader.readAsDataURL(file);
        });
      } else {
        // Document -> upload for RAG
        const ext = (file.name.split(".").pop() ?? "").toLowerCase();
        if (!DOC_EXTENSIONS.has(ext)) {
          setError(`Unsupported file type: .${ext}`);
          continue;
        }
        try {
          await uploadSessionDocument(activeChat.id, file);
        } catch (err) {
          setError(err instanceof Error ? err.message : "Upload failed");
        }
      }
    }
    await refreshWorkspace(activeChat.id);
  }

  const originalWindowSizeRef = useRef<{ width: number; height: number } | null>(null);
  const [openDetailsCount, setOpenDetailsCount] = useState(0);

  async function handleDetailsToggle(opened: boolean) {
    try {
      const { isTauri } = await import("@tauri-apps/api/core");
      if (!isTauri()) return;
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const win = getCurrentWindow();

      setOpenDetailsCount((prev) => {
        const next = opened ? prev + 1 : Math.max(0, prev - 1);

        void (async () => {
          if (next > 0 && prev === 0) {
            // First details opened — remember current size and grow
            const size = await win.innerSize();
            originalWindowSizeRef.current = { width: size.width, height: size.height };
            await win.setSize(new (await import("@tauri-apps/api/window")).LogicalSize(
              Math.min(1800, Math.round(size.width * 1.15)),
              Math.min(1100, Math.round(size.height * 1.1)),
            ));
          } else if (next === 0 && prev > 0 && originalWindowSizeRef.current) {
            // All details closed — restore original size
            const { LogicalSize } = await import("@tauri-apps/api/window");
            await win.setSize(new LogicalSize(
              originalWindowSizeRef.current.width,
              originalWindowSizeRef.current.height,
            ));
            originalWindowSizeRef.current = null;
          }
        })();

        return next;
      });
    } catch {
      // Not running in Tauri or window API unavailable
    }
  }

  function handleCopyMessage(text: string) {
    void navigator.clipboard.writeText(text).catch(() => {});
  }

  function handleDeleteMessage(index: number) {
    if (!activeChat) return;
    setWorkspace((current) => ({
      ...current,
      chatSessions: current.chatSessions.map((s) =>
        s.id === activeChat.id
          ? { ...s, messages: s.messages.filter((_, i) => i !== index) }
          : s,
      ),
    }));
  }

  async function handleRetryMessage(index: number) {
    if (!activeChat) return;
    const messages = activeChat.messages;
    const target = messages[index];
    if (!target || target.role !== "assistant") return;

    // Find the preceding user message
    let userIdx = index - 1;
    while (userIdx >= 0 && messages[userIdx].role !== "user") userIdx--;
    if (userIdx < 0) return;
    const userText = messages[userIdx].text;

    // Trim everything from the user message forward (including this assistant)
    setWorkspace((current) => ({
      ...current,
      chatSessions: current.chatSessions.map((s) =>
        s.id === activeChat.id ? { ...s, messages: s.messages.slice(0, userIdx) } : s,
      ),
    }));
    setDraftMessage(userText);
    await new Promise((r) => setTimeout(r, 50));
    await sendMessage();
  }

  async function sendMessage() {
    const trimmed = draftMessage.trim();
    if (!trimmed) {
      return;
    }

    // Show user message immediately
    setDraftMessage("");
    if (activeChat) {
      setWorkspace((current) => ({
        ...current,
        chatSessions: current.chatSessions.map((s) =>
          s.id === activeChat.id
            ? { ...s, messages: [...s.messages, { role: "user" as const, text: trimmed, metrics: null }], updatedAt: new Date().toLocaleString() }
            : s,
        ),
      }));
    }

    const sendingSessionId = activeChat?.id ?? null;
    setChatBusySessionId(sendingSessionId);
    setError(null);

    const threadModel = sessionModelPayload(activeChat);

    // Re-check backend before falling back to offline mode
    const isOnline = backendOnline || await checkBackend();
    if (isOnline && !backendOnline) setBackendOnline(true);

    if (!isOnline) {
      const fallbackSession = activeChat ?? {
        id: `local-${Date.now()}`,
        title: titleFromPrompt(trimmed),
        updatedAt: new Date().toLocaleString(),
        model: threadModel?.modelName ?? "Choose a model",
        modelRef: threadModel?.modelRef ?? null,
        modelSource: threadModel?.source ?? "catalog",
        modelPath: threadModel?.path ?? null,
        modelBackend: threadModel?.backend ?? "auto",
        cacheLabel: launchCacheLabel,
        messages: [],
      };
      const updatedSession: ChatSession = {
        ...fallbackSession,
        updatedAt: new Date().toLocaleString(),
        messages: [
          ...fallbackSession.messages,
          { role: "user", text: trimmed, metrics: null },
          {
            role: "assistant",
            text: "The desktop shell is offline, so this local fallback kept the chat moving. Start the FastAPI sidecar to route prompts through the runtime endpoints.",
            metrics: null,
          },
        ],
      };
      setWorkspace((current) => ({ ...current, chatSessions: upsertSession(current.chatSessions, updatedSession) }));
      setActiveChatId(updatedSession.id);
      setDraftMessage("");
      setChatBusySessionId(null);
      return;
    }

    try {
      let sessionId = activeChat?.id;
      const shouldLoadThreadModel =
        threadModel &&
        (!workspace.runtime.loadedModel ||
          ![workspace.runtime.loadedModel.ref, workspace.runtime.loadedModel.runtimeTarget].includes(threadModel.modelRef));

      if (shouldLoadThreadModel && threadModel) {
        await handleLoadModel({
          modelRef: threadModel.modelRef,
          modelName: threadModel.modelName,
          source: threadModel.source,
          backend: threadModel.backend,
          path: threadModel.path,
        });
      } else if (!workspace.runtime.loadedModel && defaultChatVariant) {
          await handleLoadModel({
            modelRef: defaultChatVariant.id,
            modelName: defaultChatVariant.name,
            source: "catalog",
            backend: defaultChatVariant.backend,
          });
      }
      if (!sessionId) {
        const session = await createSession(activeChat?.title ?? "New chat");
        setWorkspace((current) => ({ ...current, chatSessions: upsertSession(current.chatSessions, session) }));
        setActiveChatId(session.id);
        sessionId = session.id;
      }

      const imagesToSend = pendingImages.length > 0 ? [...pendingImages] : undefined;
      setPendingImages([]);

      const streamPayload = {
        sessionId,
        title: threadTitleDraft.trim() || activeChat?.title,
        prompt: trimmed,
        images: imagesToSend,
        modelRef: threadModel?.modelRef,
        modelName: threadModel?.modelName,
        source: threadModel?.source,
        path: threadModel?.path,
        backend: threadModel?.backend,
        temperature: launchSettings.temperature,
        maxTokens: launchSettings.maxTokens,
        systemPrompt: systemPrompt || undefined,
        cacheBits: launchSettings.cacheBits,
        fp16Layers: launchSettings.fp16Layers,
        fusedAttention: launchSettings.fusedAttention,
        cacheStrategy: launchSettings.cacheStrategy,
        fitModelInMemory: launchSettings.fitModelInMemory,
        contextTokens: launchSettings.contextTokens,
      };

      // Add a placeholder assistant message for streaming tokens into
      const streamingChatId = sessionId ?? sendingSessionId;
      if (streamingChatId) {
        setWorkspace((current) => ({
          ...current,
          chatSessions: current.chatSessions.map((s) =>
            s.id === streamingChatId
              ? { ...s, messages: [...s.messages, { role: "assistant" as const, text: "", metrics: null }] }
              : s,
          ),
        }));
      }

      await generateChatStream(streamPayload, {
        onToken: (token) => {
          // Append token to the last assistant message
          if (streamingChatId) {
            setWorkspace((current) => ({
              ...current,
              chatSessions: current.chatSessions.map((s) => {
                if (s.id !== streamingChatId) return s;
                const msgs = [...s.messages];
                const last = msgs[msgs.length - 1];
                if (last?.role === "assistant") {
                  msgs[msgs.length - 1] = { ...last, text: last.text + token };
                }
                return { ...s, messages: msgs };
              }),
            }));
          }
        },
        onDone: (response) => {
          setWorkspace((current) =>
            syncRuntime(
              { ...current, chatSessions: upsertSession(current.chatSessions, response.session) },
              response.runtime,
            ),
          );
          setActiveChatId(response.session.id);
        },
        onError: (errMsg) => {
          setError(`Chat error: ${errMsg}`);
        },
      });
      setDraftMessage("");
    } catch (actionError) {
      const detail = actionError instanceof Error ? actionError.message : "Unknown error";
      if (detail.includes("Load a model") || detail.includes("409")) {
        setError("No model is loaded. Select a model and try again.");
      } else {
        setError(`Chat error: ${detail}`);
      }
      // Rollback the optimistic user message on failure
      if (sendingSessionId) {
        setWorkspace((current) => ({
          ...current,
          chatSessions: current.chatSessions.map((s) => {
            if (s.id !== sendingSessionId) return s;
            const last = s.messages[s.messages.length - 1];
            if (last?.role === "user" && last.text === trimmed) {
              return { ...s, messages: s.messages.slice(0, -1) };
            }
            return s;
          }),
        }));
      }
    } finally {
      setChatBusySessionId(null);
    }
  }

  function renderCapabilityIcons(capabilities: string[], max = 5) {
    return (
      <div className="capability-strip">
        {capabilities.slice(0, max).map((capability) => {
          const meta = capabilityMeta(capability);
          const fullMeta = CAPABILITY_META[capability];
          return (
            <span
              className="capability-icon"
              key={capability}
              title={meta.title}
              style={fullMeta ? { borderColor: `${fullMeta.color}40`, color: fullMeta.color } : undefined}
            >
              {fullMeta?.icon ?? ""} {meta.shortLabel}
            </span>
          );
        })}
      </div>
    );
  }

  function renderLaunchModal() {
    if (!pendingLaunch) return null;
    const actionLabel = pendingLaunch.action === "thread" ? "Start Chat" : pendingLaunch.action === "chat" ? "Load for Chat" : "Load for Server";
    // Prefer local library, but if the caller preselected a catalog key
    // (e.g. the Discover Chat/Server buttons pass `catalog:${variant.id}`)
    // or the library is empty, fall back to the combined set so the
    // preselection actually resolves.
    const preselectedIsCatalog = pendingLaunch.preselectedKey?.startsWith("catalog:") ?? false;
    const localOptions =
      preselectedIsCatalog || libraryChatOptions.length === 0
        ? threadModelOptions
        : libraryChatOptions;
    const searchLower = launchModelSearch.toLowerCase();
    const filteredOptions = localOptions.filter(
      (o) => !searchLower || o.label.toLowerCase().includes(searchLower) || o.detail.toLowerCase().includes(searchLower),
    );
    const selectedLaunchKey = pendingLaunch.preselectedKey ?? localOptions[0]?.key ?? "";
    const setSelectedLaunchKey = (key: string) => setPendingLaunch((prev) => prev ? { ...prev, preselectedKey: key } : null);
    const selectedOption = localOptions.find((o) => o.key === selectedLaunchKey);
    const hasPreselection = Boolean(pendingLaunch.preselectedKey);
    const showList = !hasPreselection || launchModelSearch.length > 0;

    return (
      <div className="modal-overlay" onClick={() => setPendingLaunch(null)}>
        <div className="modal-content modal-wide" onClick={(event) => event.stopPropagation()}>
          <div className="modal-header">
            <h3>Select Model</h3>
          </div>
          <div className="modal-body">
            {selectedOption ? (
              <div className="model-selected-card">
                <div className="model-selected-info">
                  <strong>{selectedOption.label}</strong>
                  <div className="model-selected-meta">
                    {selectedOption.paramsB ? <span className="badge muted">{number(selectedOption.paramsB)}B</span> : null}
                    <span className="badge muted">{selectedOption.format ?? selectedOption.detail}</span>
                    {selectedOption.quantization ? <span className="badge muted">{selectedOption.quantization}</span> : null}
                    {selectedOption.sizeGb ? <span className="badge muted">{sizeLabel(selectedOption.sizeGb)}</span> : null}
                    {selectedOption.contextWindow ? <span className="badge muted">{selectedOption.contextWindow}</span> : null}
                    <span className={`badge ${selectedOption.source === "library" ? "success" : "accent"}`}>{selectedOption.group}</span>
                  </div>
                </div>
                <button className="secondary-button" type="button" onClick={() => { setLaunchModelSearch(""); setPendingLaunch((prev) => prev ? { ...prev, preselectedKey: undefined } : null); }}>
                  Change
                </button>
              </div>
            ) : null}

            {showList || !selectedOption ? (
              <>
                <input
                  className="text-input"
                  type="search"
                  placeholder="Search models..."
                  value={launchModelSearch}
                  onChange={(e) => setLaunchModelSearch(e.target.value)}
                  autoFocus={!hasPreselection}
                />
                <div className="model-select-list">
                  {filteredOptions.map((option) => (
                    <button
                      key={option.key}
                      className={`model-select-item${option.key === selectedLaunchKey ? " active" : ""}`}
                      type="button"
                      onClick={() => { setSelectedLaunchKey(option.key); setLaunchModelSearch(""); }}
                    >
                      <div className="model-select-item-info">
                        <strong>{option.label}</strong>
                        <div className="model-select-item-meta">
                          {option.paramsB ? <span>{number(option.paramsB)}B</span> : null}
                          <span>{option.format ?? option.detail}</span>
                          {option.quantization ? <span>{option.quantization}</span> : null}
                          {option.sizeGb ? <span>{sizeLabel(option.sizeGb)}</span> : null}
                          {option.contextWindow ? <span>{option.contextWindow}</span> : null}
                          {option.maxContext ? <span>{`${option.maxContext >= 1_000_000 ? (option.maxContext / 1_048_576).toFixed(1) + "M" : Math.round(option.maxContext / 1024) + "K"} detected`}</span> : null}
                        </div>
                      </div>
                      <span className={`badge ${option.source === "library" ? "success" : "accent"}`}>{option.group}</span>
                    </button>
                  ))}
                  {filteredOptions.length === 0 ? <p className="model-select-empty">No models match your search.</p> : null}
                </div>
              </>
            ) : null}

            <div className="model-select-settings">
              <span className="eyebrow">Launch settings</span>
              <RuntimeControls
                settings={launchSettings}
                onChange={updateLaunchSetting}
                maxContext={selectedOption?.maxContext}
                diskSizeGb={selectedOption?.sizeGb}
                preview={preview}
                availableMemoryGb={workspace.system.availableMemoryGb}
                totalMemoryGb={workspace.system.totalMemoryGb}
                availableCacheStrategies={workspace.system.availableCacheStrategies}
                compact
              />
            </div>
          </div>
          <div className="modal-footer">
            <button
              className="primary-button"
              type="button"
              disabled={!selectedLaunchKey}
              onClick={() => void confirmLaunch(selectedLaunchKey)}
            >
              {actionLabel}
            </button>
            <button className="secondary-button" type="button" onClick={() => setPendingLaunch(null)}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  function renderDashboard() {
    const warmModels = workspace.runtime.warmModels ?? [];
    const activeReq = workspace.runtime.activeRequests ?? 0;
    const servedReq = workspace.runtime.requestsServed ?? 0;
    const memPressure = workspace.system.memoryPressurePercent ?? 0;
    const compressedGb = workspace.system.compressedMemoryGb ?? 0;
    const swapGb = workspace.system.swapUsedGb ?? 0;
    const diskFree = workspace.system.diskFreeGb;
    const diskTotal = workspace.system.diskTotalGb;
    const battery = workspace.system.battery;

    return (
      <div className="content-grid">
        <Panel
          title="Live System Stats"
          subtitle="Refreshed from the Python sidecar so the desktop shell can make fit recommendations."
          className="span-2"
        >
          <div className="stat-grid">
            <StatCard
              label="Runtime engine"
              value={workspace.runtime.engineLabel}
              hint={workspace.runtime.loadedModel ? workspace.runtime.loadedModel.name : "No model loaded"}
            />
            <StatCard
              label="Inference activity"
              value={`${activeReq} active`}
              hint={`${servedReq} total served`}
            />
            <StatCard
              label="Warm pool"
              value={`${warmModels.length} model${warmModels.length === 1 ? "" : "s"}`}
              hint={warmModels.length > 0 ? warmModels.map((w) => w.name).join(" · ") : "No warm models"}
            />
            {diskFree !== undefined && diskTotal ? (
              <StatCard
                label="Model disk"
                value={`${number(diskFree, 2)} GB free`}
                hint={`${number(diskTotal, 2)} GB total`}
              />
            ) : (
              <StatCard
                label="Spare headroom"
                value={`${number(workspace.system.spareHeadroomGb, 2)} GB`}
                hint={`${number(workspace.recommendation.headroomPercent, 0)}% working headroom`}
              />
            )}
          </div>
          <div className="panel-grid">
            <div className="stack">
              <ProgressRow
                label="Memory in use"
                value={workspace.system.usedMemoryGb}
                max={workspace.system.totalMemoryGb}
                valueLabel={`${number(workspace.system.usedMemoryGb, 2)} GB / ${number(workspace.system.totalMemoryGb, 2)} GB`}
              />
              <ProgressRow
                label="Memory pressure"
                value={memPressure}
                valueLabel={`${number(memPressure, 0)}%${compressedGb > 0 ? ` · ${number(compressedGb, 2)} GB compressed` : ""}`}
              />
              {swapGb > 0.01 ? (
                <ProgressRow
                  label="Swap usage"
                  value={swapGb}
                  max={Math.max(workspace.system.swapTotalGb ?? swapGb, swapGb, 0.01)}
                  valueLabel={`${number(swapGb, 2)} GB${workspace.system.swapTotalGb ? ` / ${number(workspace.system.swapTotalGb, 2)} GB` : ""}`}
                />
              ) : null}
              <ProgressRow
                label="CPU load"
                value={workspace.system.cpuUtilizationPercent}
                valueLabel={`${number(workspace.system.cpuUtilizationPercent, 0)}%`}
              />
              <ProgressRow
                label={`Headroom for ${workspace.recommendation.targetModel}`}
                value={workspace.recommendation.headroomPercent}
                valueLabel={`${workspace.recommendation.headroomPercent}%`}
              />
              {battery ? (
                <div className={`battery-card${battery.powerSource === "Battery" && battery.percent < 20 ? " battery-card--low" : ""}`}>
                  <div className="battery-card-header">
                    <span className="eyebrow">Power</span>
                    <span className={`badge ${battery.powerSource === "AC" ? "success" : battery.percent < 20 ? "warning" : "muted"}`}>
                      {battery.powerSource === "AC" ? (battery.charging ? "Charging" : "AC Power") : "On Battery"}
                    </span>
                  </div>
                  <div className="battery-card-bar">
                    <div
                      className="battery-card-fill"
                      style={{ width: `${battery.percent}%` }}
                    />
                  </div>
                  <div className="battery-card-footer">
                    <strong>{battery.percent}%</strong>
                    {battery.powerSource === "Battery" ? (
                      <small>Unplugged — inference may throttle on thermal pressure</small>
                    ) : null}
                  </div>
                </div>
              ) : null}
            </div>
            <div className="data-table compact-table">
              <div className="table-row table-head">
                <span>Process</span>
                <span>Owner</span>
                <span>Memory</span>
                <span>CPU</span>
              </div>
              <div className="data-table-body">
                {workspace.system.runningLlmProcesses.length ? (
                  workspace.system.runningLlmProcesses.map((process) => (
                    <div className="table-row" key={process.pid}>
                      <div className="process-name-cell">
                        <div className="process-name-line">
                          <strong>{process.name}</strong>
                          {process.modelStatus ? (
                            <span className={`badge ${process.modelStatus === "active" ? "success" : "muted"} process-status-badge`}>
                              {process.modelStatus === "active" ? "ACTIVE" : "WARM"}
                            </span>
                          ) : null}
                        </div>
                        {process.modelName ? <small className="process-model-name">{process.modelName}</small> : null}
                      </div>
                      <span><span className={`badge ${process.owner === "ChaosEngineAI" ? "accent" : "muted"}`}>{process.owner ?? "System"}</span></span>
                      <span>{number(process.memoryGb, 2)} GB</span>
                      <span>{number(process.cpuPercent, 0)}%</span>
                    </div>
                  ))
                ) : (
                  <div className="empty-state small-empty">
                    <p>No active local LLM processes were detected.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </Panel>

        <Panel title="Hardware Fit" subtitle="Guidance relative to the recommended target profile for this machine.">
          <div className="callout">
            <span className="badge accent">Recommended target</span>
            <h3>{workspace.recommendation.title}</h3>
            <p>{workspace.recommendation.detail}</p>
          </div>
          <ProgressRow
            label={`Headroom for ${workspace.recommendation.targetModel}`}
            value={workspace.recommendation.headroomPercent}
            valueLabel={`${workspace.recommendation.headroomPercent}%`}
          />
          <div className="callout quiet">
            <h3>Current runtime</h3>
            <p>
              {workspace.runtime.loadedModel
                ? `${workspace.runtime.loadedModel.name} loaded via ${workspace.runtime.engineLabel}.`
                : "No model is loaded yet. Pick a thread model in Chat or browse a newer family in Online Models."}
            </p>
          </div>
        </Panel>

        <Panel title="Activity Feed" subtitle="Operational events that should stay visible without digging into logs.">
          <div className="list scrollable-list">
            {workspace.activity.map((item, idx) => (
              <div className="list-row" key={`${idx}-${item.title}`}>
                <div>
                  <strong>{item.title}</strong>
                  <p>{item.detail}</p>
                </div>
                <span className="badge muted">{item.time}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    );
  }

  function renderCapabilityFilterBar(
    active: string | null,
    setActive: (cap: string | null) => void,
    capabilities: string[],
  ) {
    // Show all known capabilities that appear in the data, in a consistent order
    const capOrder = Object.keys(CAPABILITY_META);
    const present = new Set(capabilities);
    const uniqueCaps = capOrder.filter((c) => present.has(c));
    return (
      <div className="cap-filter-bar">
        <button
          className={`cap-filter-btn${active === null ? " cap-filter-btn--active" : ""}`}
          type="button"
          onClick={() => setActive(null)}
        >
          All
        </button>
        {uniqueCaps.map((cap) => {
          const meta = CAPABILITY_META[cap];
          return (
            <button
              key={cap}
              className={`cap-filter-btn${active === cap ? " cap-filter-btn--active" : ""}`}
              type="button"
              onClick={() => setActive(active === cap ? null : cap)}
              title={meta?.title ?? cap}
              style={active === cap && meta ? { borderColor: meta.color, color: meta.color, background: `${meta.color}15` } : undefined}
            >
              {meta?.icon ?? ""} {meta?.shortLabel ?? cap}
            </button>
          );
        })}
      </div>
    );
  }

  function renderFormatFilterBar(
    active: string | null,
    setActive: (fmt: string | null) => void,
    formats: string[],
    allLabel = "All formats",
  ) {
    const uniqueFormats = [...new Set(formats)].sort();
    if (uniqueFormats.length < 2) return null;
    return (
      <div className="cap-filter-bar">
        <button
          className={`cap-filter-btn${active === null ? " cap-filter-btn--active" : ""}`}
          type="button"
          onClick={() => setActive(null)}
        >
          {allLabel}
        </button>
        {uniqueFormats.map((fmt) => (
          <button
            key={fmt}
            className={`cap-filter-btn${active === fmt ? " cap-filter-btn--active" : ""}`}
            type="button"
            onClick={() => setActive(active === fmt ? null : fmt)}
          >
            {fmt}
          </button>
        ))}
      </div>
    );
  }

  async function toggleHubExpand(repo: string) {
    const next = expandedHubId === repo ? null : repo;
    setExpandedHubId(next);
    if (next && !hubFileCache[repo] && !hubFileLoading[repo]) {
      setHubFileLoading((current) => ({ ...current, [repo]: true }));
      setHubFileError((current) => {
        const { [repo]: _omit, ...rest } = current;
        return rest;
      });
      try {
        const payload = await listHubFiles(repo);
        setHubFileCache((current) => ({ ...current, [repo]: payload }));
      } catch (err) {
        setHubFileError((current) => ({
          ...current,
          [repo]: err instanceof Error ? err.message : "Could not load file list.",
        }));
      } finally {
        setHubFileLoading((current) => {
          const { [repo]: _omit, ...rest } = current;
          return rest;
        });
      }
    }
  }

  function renderOnlineModels() {
    const allDiscoverCaps = searchResults.flatMap((f) => f.capabilities);
    const allDiscoverFormats = searchResults.flatMap((f) => f.variants.map((v) => v.format));
    let filteredResults = searchResults;
    if (discoverCapFilter) {
      filteredResults = filteredResults.filter((f) => f.capabilities.includes(discoverCapFilter) || f.variants.some((v) => v.capabilities.includes(discoverCapFilter!)));
    }
    if (discoverFormatFilter) {
      filteredResults = filteredResults.filter((f) => f.variants.some((v) => v.format === discoverFormatFilter));
    }

    return (
      <div className="content-grid discover-page">
        <Panel
          title="Discover Models"
          subtitle={`${searchResults.length} model families / ${localVariantCount} downloaded locally`}
          className="span-2 discover-panel"
          actions={
            <input
              className="text-input discover-search"
              type="search"
              placeholder="Search by name, provider, or capability..."
              value={searchInput}
              onChange={(event) => setSearchInput(event.target.value)}
            />
          }
        >
          {renderCapabilityFilterBar(discoverCapFilter, setDiscoverCapFilter, allDiscoverCaps)}
          {renderFormatFilterBar(discoverFormatFilter, setDiscoverFormatFilter, allDiscoverFormats)}
          {filteredResults.length > 0 || hubResults.length > 0 ? (
            <div className="discover-sections">
              {filteredResults.length > 0 ? (
                <section className="discover-section" aria-label="Curated model families">
                  <div className="discover-list">
                    {filteredResults.map((family) => {
                      const isExpanded = expandedFamilyId === family.id;
                      const localCount = family.variants.filter((v) => v.availableLocally).length;
                      const paramRange = family.variants.length > 1
                        ? `${number(Math.min(...family.variants.map((v) => v.paramsB)))}B - ${number(Math.max(...family.variants.map((v) => v.paramsB)))}B`
                        : `${number(family.variants[0]?.paramsB ?? 0)}B`;
                      const formats = [...new Set(family.variants.map((v) => v.format))];
                      return (
                        <div key={family.id} className={`discover-card${isExpanded ? " expanded" : ""}`}>
                          <div
                            className="discover-card-header discover-card-header--interactive"
                            role="button"
                            tabIndex={0}
                            onClick={() => { setExpandedFamilyId(isExpanded ? null : family.id); setExpandedVariantId(null); }}
                            onKeyDown={(event) => handleActionKeyDown(event, () => {
                              setExpandedFamilyId(isExpanded ? null : family.id);
                              setExpandedVariantId(null);
                            })}
                          >
                            <div className="discover-card-info">
                              <div className="discover-card-title">
                                <strong>{family.name}</strong>
                                <span className="badge muted">{family.provider}</span>
                                <span className="badge muted">{paramRange}</span>
                                {formats.map((f) => <span key={f} className="badge muted">{f}</span>)}
                                {localCount > 0 ? <span className="badge success">{localCount} downloaded</span> : null}
                              </div>
                              <p>{family.headline}</p>
                              <div className="discover-card-meta">
                                {renderCapabilityIcons(family.capabilities, 8)}
                                <small>{family.variants.length} variants</small>
                                <small>{family.updatedLabel}</small>
                              </div>
                            </div>
                            <div className="discover-card-head-actions">
                              <button
                                className="secondary-button"
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setDetailFamilyId(family.id);
                                }}
                                title="Show full details in a focused view"
                              >
                                Details
                              </button>
                              <span className="discover-chevron">{isExpanded ? "\u25B2" : "\u25BC"}</span>
                            </div>
                          </div>

                          {isExpanded ? (
                            <div className="discover-card-body">
                              <p className="discover-summary">{family.summary}</p>
                              <div className="discover-variant-table">
                                <div className="discover-variant-head">
                                  <span>Variant</span>
                                  <span>Format</span>
                                  <span>Backend</span>
                                  <span>Params</span>
                                  <span>Size</span>
                                  <span>RAM</span>
                                  <span>Compressed</span>
                                  <span>Context</span>
                                  <span></span>
                                </div>
                                {family.variants.map((variant) => {
                                  const matchedLocal = findLibraryItemForVariant(workspace.library, variant);
                                  const isVariantExpanded = expandedVariantId === variant.id;
                                  const downloadState = activeDownloads[variant.repo];
                                  const isDownloading = downloadState?.state === "downloading";
                                  const isDownloadPaused = downloadState?.state === "cancelled";
                                  const isDownloadComplete = downloadState?.state === "completed";
                                  return (
                                    <div key={variant.id}>
                                      <div
                                        className={`discover-variant-row${isVariantExpanded ? " expanded" : ""}${variant.availableLocally || isDownloadComplete ? " downloaded" : ""}`}
                                        onClick={() => setExpandedVariantId(isVariantExpanded ? null : variant.id)}
                                        role="button"
                                        tabIndex={0}
                                      >
                                        <div className="discover-variant-name">
                                          <strong>{variant.name}</strong>
                                          {renderCapabilityIcons(variant.capabilities, 4)}
                                        </div>
                                        <span>{variant.format} / {variant.quantization}</span>
                                        <span>{variant.backend}</span>
                                        <span>{number(variant.paramsB)}B</span>
                                        <span>{sizeLabel(variant.sizeGb)}</span>
                                        <span>{variant.estimatedMemoryGb ? `~${number(variant.estimatedMemoryGb)}GB` : "?"}</span>
                                        <span>{variant.estimatedCompressedMemoryGb ? `~${number(variant.estimatedCompressedMemoryGb)}GB` : "?"}</span>
                                        <span>{variant.contextWindow}</span>
                                        <div className="discover-variant-actions" onClick={(e) => e.stopPropagation()}>
                                          {variant.availableLocally ? (
                                            <>
                                              {variant.launchMode === "convert" ? (
                                                <button className="primary-button action-convert" type="button" onClick={() => prepareCatalogConversion(variant)}>CONVERT</button>
                                              ) : null}
                                              <button className="primary-button action-chat" type="button" onClick={() => openModelSelector("thread", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`)}>CHAT</button>
                                              <button className="primary-button action-server" type="button" onClick={() => openModelSelector("server", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`)}>SERVER</button>
                                            </>
                                          ) : isDownloading ? (
                                            <>
                                              <span className="badge accent">{downloadProgressLabel(downloadState).toUpperCase()}</span>
                                              <button className="secondary-button" type="button" onClick={() => void handleCancelModelDownload(variant.repo)}>PAUSE</button>
                                            </>
                                          ) : isDownloadPaused ? (
                                            <>
                                              <span className="badge warning">{downloadProgressLabel(downloadState).toUpperCase()}</span>
                                              <button className="secondary-button" type="button" onClick={() => void handleDownloadModel(variant.repo)}>RESUME</button>
                                            </>
                                          ) : isDownloadComplete ? (
                                            <span className="badge success">DOWNLOAD COMPLETE</span>
                                          ) : (
                                            <button className="secondary-button" type="button" onClick={() => void handleDownloadModel(variant.repo)}>DOWNLOAD</button>
                                          )}
                                        </div>
                                      </div>
                                      {isVariantExpanded ? (
                                        <div className="variant-detail-expand">
                                          <div className="variant-detail-left">
                                            <p>{variant.note}</p>
                                            {matchedLocal ? <p className="mono-text variant-local-path">{matchedLocal.path}</p> : null}
                                            <a
                                              className="text-link"
                                              href={variant.link}
                                              target="_blank"
                                              rel="noreferrer"
                                              onClick={(event) => {
                                                event.preventDefault();
                                                void handleOpenExternalUrl(variant.link);
                                              }}
                                            >
                                              Open model card on HuggingFace
                                            </a>
                                          </div>
                                        </div>
                                      ) : null}
                                    </div>
                                  );
                                })}
                              </div>
                              {family.readme.length > 0 ? (
                                <div className="discover-readme">
                                  {family.readme.slice(0, 2).map((line, i) => <p key={i}>{line}</p>)}
                                </div>
                              ) : null}
                            </div>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </section>
              ) : null}

              {hubResults.length > 0 ? (
                <section className="discover-section discover-section--hub" aria-label="Hugging Face Hub results">
                  <div className="hub-section-header">
                    <span className="eyebrow">HuggingFace Hub</span>
                    <p>{hubResults.length} results from huggingface.co</p>
                  </div>
                  <div className="discover-list">
                    {hubResults
                      .filter((model) => {
                        if (discoverFormatFilter && model.format !== discoverFormatFilter) return false;
                        return true;
                      })
                      .map((model) => {
                        const isExpanded = expandedHubId === model.id;
                        const fileData = hubFileCache[model.id];
                        const loading = !!hubFileLoading[model.id];
                        const errorMsg = hubFileError[model.id];
                        const downloadState = activeDownloads[model.repo];
                        const isDownloading = downloadState?.state === "downloading";
                        const isDownloadPaused = downloadState?.state === "cancelled";
                        const isDownloadComplete = downloadState?.state === "completed";
                        return (
                          <div key={model.id} className={`discover-card${isExpanded ? " expanded" : ""}`}>
                            <div
                              className="discover-card-header discover-card-header--interactive"
                              role="button"
                              tabIndex={0}
                              onClick={() => void toggleHubExpand(model.id)}
                              onKeyDown={(event) => handleActionKeyDown(event, () => {
                                void toggleHubExpand(model.id);
                              })}
                            >
                              <div className="discover-card-info">
                                <div className="discover-card-title">
                                  <strong>{model.name}</strong>
                                  <span className="badge muted">{model.provider}</span>
                                  <span className={`badge ${model.format === "GGUF" ? "accent" : "muted"}`}>{model.format}</span>
                                  {model.availableLocally ? <span className="badge success">Downloaded</span> : null}
                                  {!model.availableLocally && isDownloadComplete ? <span className="badge success">Download complete</span> : null}
                                </div>
                                <div className="discover-card-meta">
                                  <small>{model.downloadsLabel}</small>
                                  <small>{model.likesLabel}</small>
                                </div>
                              </div>
                              <span className="discover-chevron">{isExpanded ? "\u25B2" : "\u25BC"}</span>
                            </div>
                            {isExpanded ? (
                              <div className="discover-card-body">
                                {loading ? (
                                  <p className="muted-text">Loading file list from Hugging Face...</p>
                                ) : errorMsg ? (
                                  <div className="callout error">
                                    <p>{errorMsg}</p>
                                  </div>
                                ) : fileData ? (
                                  <>
                                    {fileData.warning ? (
                                      <div className="callout quiet">
                                        <div className="chip-row">
                                          <span className="badge warning">Preview unavailable</span>
                                        </div>
                                        <p>{fileData.warning}</p>
                                      </div>
                                    ) : null}
                                    <div className="hub-detail-meta">
                                      {fileData.license ? <span className="badge muted">License: {fileData.license}</span> : null}
                                      {fileData.pipelineTag ? <span className="badge muted">{fileData.pipelineTag}</span> : null}
                                      {fileData.totalSizeGb ? <span className="badge muted">{number(fileData.totalSizeGb)} GB total</span> : null}
                                      {fileData.lastModified ? <span className="badge muted">Updated {fileData.lastModified.slice(0, 10)}</span> : null}
                                    </div>
                                    {fileData.tags.length > 0 ? (
                                      <div className="hub-detail-tags">
                                        {fileData.tags.slice(0, 12).map((tag) => (
                                          <span key={tag} className="badge muted hub-tag">{tag}</span>
                                        ))}
                                        {fileData.tags.length > 12 ? <small className="muted-text">+{fileData.tags.length - 12} more</small> : null}
                                      </div>
                                    ) : null}
                                    {fileData.files.length === 0 ? (
                                      <p className="muted-text">File preview is not available for this repo right now.</p>
                                    ) : (() => {
                                      const weights = fileData.files.filter((f) => f.kind === "weight" || f.kind === "vision_projector");
                                      const tokenizer = fileData.files.filter((f) => f.kind === "tokenizer" || f.kind === "config" || f.kind === "template");
                                      const other = fileData.files.filter((f) => !weights.includes(f) && !tokenizer.includes(f));
                                      return (
                                        <div className="hub-file-groups">
                                          {weights.length > 0 ? (
                                            <div className="hub-file-group">
                                              <span className="eyebrow">Weights ({weights.length})</span>
                                              <ul className="hub-file-list">
                                                {weights.map((f) => (
                                                  <li key={f.path}>
                                                    <code>{f.path}</code>
                                                    <span className="muted-text">{f.sizeGb ? `${number(f.sizeGb)} GB` : ""}</span>
                                                    {f.kind === "vision_projector" ? <span className="badge muted">vision</span> : null}
                                                  </li>
                                                ))}
                                              </ul>
                                            </div>
                                          ) : null}
                                          {tokenizer.length > 0 ? (
                                            <div className="hub-file-group">
                                              <span className="eyebrow">Config &amp; tokenizer</span>
                                              <ul className="hub-file-list">
                                                {tokenizer.map((f) => (
                                                  <li key={f.path}><code>{f.path}</code></li>
                                                ))}
                                              </ul>
                                            </div>
                                          ) : null}
                                          {other.length > 0 ? (
                                            <details className="hub-file-extras">
                                              <summary>+{other.length} other files</summary>
                                              <ul className="hub-file-list">
                                                {other.map((f) => (
                                                  <li key={f.path}><code>{f.path}</code></li>
                                                ))}
                                              </ul>
                                            </details>
                                          ) : null}
                                        </div>
                                      );
                                    })()}
                                  </>
                                ) : null}
                                <div className="button-row">
                                  {model.availableLocally ? (
                                    <>
                                      <button className="primary-button action-chat" type="button" onClick={() => openModelSelector("thread")}>Chat</button>
                                      <button className="primary-button action-server" type="button" onClick={() => openModelSelector("server")}>Server</button>
                                    </>
                                  ) : isDownloading ? (
                                    <>
                                      <span className="badge accent">{downloadProgressLabel(downloadState)}</span>
                                      <button className="secondary-button" type="button" onClick={() => void handleCancelModelDownload(model.repo)}>
                                        Pause
                                      </button>
                                    </>
                                  ) : isDownloadPaused ? (
                                    <>
                                      <span className="badge warning">{downloadProgressLabel(downloadState)}</span>
                                      <button
                                        className="secondary-button"
                                        type="button"
                                        onClick={() => void handleDownloadModel(model.repo)}
                                      >
                                        Resume
                                      </button>
                                    </>
                                  ) : isDownloadComplete ? (
                                    <span className="badge success">Download complete</span>
                                  ) : (
                                    <button
                                      className="primary-button"
                                      type="button"
                                      onClick={() => void handleDownloadModel(model.repo)}
                                    >
                                      Download
                                    </button>
                                  )}
                                  <a
                                    className="text-link"
                                    href={model.link}
                                    target="_blank"
                                    rel="noreferrer"
                                    onClick={(event) => {
                                      event.preventDefault();
                                      void handleOpenExternalUrl(model.link);
                                    }}
                                  >
                                    Open on HuggingFace ↗
                                  </a>
                                </div>
                              </div>
                            ) : null}
                          </div>
                        );
                      })}
                  </div>
                </section>
              ) : null}
            </div>
          ) : null}

          {filteredResults.length === 0 && hubResults.length === 0 ? (
            <div className="empty-state">
              <p>{discoverCapFilter ? `No models match the "${CAPABILITY_META[discoverCapFilter]?.shortLabel ?? discoverCapFilter}" filter.` : searchInput ? `No models match "${searchInput}". Try a different search term.` : "Type to search for models."}</p>
            </div>
          ) : null}
        </Panel>
      </div>
    );
  }

  function toggleLibrarySort(key: "name" | "format" | "backend" | "size" | "ram" | "compressed" | "modified" | "context") {
    if (librarySortKey === key) {
      setLibrarySortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setLibrarySortKey(key);
      setLibrarySortDir(key === "name" ? "asc" : "desc");
    }
  }

  function sortIndicator(key: string) {
    if (librarySortKey !== key) return "";
    return librarySortDir === "asc" ? " \u25B2" : " \u25BC";
  }

  function renderMyModels() {
    const allLibraryCaps = filteredLibraryRows.flatMap(({ matchedVariant }) => matchedVariant?.capabilities ?? []);
    let capFilteredLibrary = libraryCapFilter
      ? filteredLibraryRows.filter(({ matchedVariant }) => {
          return matchedVariant?.capabilities?.includes(libraryCapFilter!) ?? false;
        })
      : filteredLibraryRows;
    if (libraryFormatFilter) {
      capFilteredLibrary = capFilteredLibrary.filter(({ displayFormat }) => displayFormat === libraryFormatFilter);
    }
    if (libraryBackendFilter) {
      capFilteredLibrary = capFilteredLibrary.filter(({ displayBackend }) => displayBackend === libraryBackendFilter);
    }
    const allLibraryFormats = filteredLibraryRows.map(({ displayFormat }) => displayFormat);
    const allLibraryBackends = filteredLibraryRows.map(({ displayBackend }) => displayBackend);

    return (
      <div className="content-grid">
        <Panel
          title="My Models"
          subtitle={`${filteredLibraryRows.length} models / ${sizeLabel(libraryTotalSizeGb)} on disk / ${enabledDirectoryCount} directories`}
          className="span-2"
          actions={
            <input
              className="text-input discover-search"
              type="search"
              placeholder="Filter by name, path, format, quant, or backend..."
              value={librarySearchInput}
              onChange={(event) => setLibrarySearchInput(event.target.value)}
            />
          }
        >
          {renderCapabilityFilterBar(libraryCapFilter, setLibraryCapFilter, allLibraryCaps)}
          {renderFormatFilterBar(libraryFormatFilter, setLibraryFormatFilter, allLibraryFormats)}
          {renderFormatFilterBar(libraryBackendFilter, setLibraryBackendFilter, allLibraryBackends, "All backends")}
          {capFilteredLibrary.length ? (
            <div className="library-full-table">
              <div className="library-head">
                <button className="sort-header" type="button" onClick={() => toggleLibrarySort("name")}>Model{sortIndicator("name")}</button>
                <button className="sort-header" type="button" onClick={() => toggleLibrarySort("format")}>Format{sortIndicator("format")}</button>
                <span className="sort-header">Quant</span>
                <button className="sort-header" type="button" onClick={() => toggleLibrarySort("backend")}>Backend{sortIndicator("backend")}</button>
                <button className="sort-header" type="button" onClick={() => toggleLibrarySort("size")}>Size{sortIndicator("size")}</button>
                <button className="sort-header" type="button" onClick={() => toggleLibrarySort("ram")}>RAM{sortIndicator("ram")}</button>
                <button className="sort-header" type="button" onClick={() => toggleLibrarySort("compressed")}>Compressed{sortIndicator("compressed")}</button>
                <button className="sort-header" type="button" onClick={() => toggleLibrarySort("context")}>Context{sortIndicator("context")}</button>
                <span className="sort-header"></span>
              </div>
              <div className="library-rows">
                {capFilteredLibrary.map(({ item, matchedVariant, displayFormat, displayQuantization, displayBackend, sourceKind }) => {
                  const isExpanded = expandedLibraryPath === item.path;
                  return (
                    <div key={item.path} className={`library-item-wrap${isExpanded ? " expanded" : ""}`}>
                      <div
                        className="library-item-row"
                        role="button"
                        tabIndex={0}
                        onClick={() => setExpandedLibraryPath(isExpanded ? null : item.path)}
                      >
                        <div className="library-item-name">
                          <strong>{item.name}</strong>
                          <div className="library-item-meta-row">
                            <span className="badge muted">{sourceKind}</span>
                          </div>
                          {matchedVariant ? renderCapabilityIcons(matchedVariant.capabilities, 5) : null}
                          {item.broken ? (
                            <span className="broken-tag">
                              <span className="badge warning">BROKEN</span>
                              <small className="broken-reason">{item.brokenReason ?? "Incomplete or broken"}</small>
                            </span>
                          ) : null}
                        </div>
                        <span>{displayFormat}</span>
                        <span>{displayQuantization ?? "-"}</span>
                        <span>{displayBackend}</span>
                        <span>{sizeLabel(item.sizeGb)}</span>
                        <span>{matchedVariant?.estimatedMemoryGb ? `~${number(matchedVariant.estimatedMemoryGb)}GB` : "?"}</span>
                        <span>{matchedVariant?.estimatedCompressedMemoryGb ? `~${number(matchedVariant.estimatedCompressedMemoryGb)}GB` : "?"}</span>
                        <span>{matchedVariant?.contextWindow ?? ""}</span>
                        <div className="library-row-actions" onClick={(e) => e.stopPropagation()}>
                          {displayFormat !== "MLX" ? (
                            <button className="primary-button action-convert" type="button" onClick={() => prepareLibraryConversion(item)}>CONVERT</button>
                          ) : null}
                          <button className="primary-button action-chat" type="button" onClick={() => openModelSelector("chat", `library:${item.path}`)}>CHAT</button>
                          <button className="primary-button action-server" type="button" onClick={() => openModelSelector("server", `library:${item.path}`)}>SERVER</button>
                          <button className="secondary-button icon-button" type="button" title={fileRevealLabel} onClick={() => void handleRevealPath(item.path)}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                              <polyline points="15 3 21 3 21 9" />
                              <line x1="10" y1="14" x2="21" y2="3" />
                            </svg>
                          </button>
                          <button className="secondary-button icon-button danger-button" type="button" title="Delete model" onClick={() => void handleDeleteModel(item)}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <polyline points="3 6 5 6 21 6" />
                              <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                              <path d="M10 11v6" />
                              <path d="M14 11v6" />
                              <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                            </svg>
                          </button>
                        </div>
                      </div>
                      {isExpanded ? (
                        <div className="library-item-detail">
                          <div className="library-detail-left">
                            <p className="mono-text library-path">{item.path}</p>
                            {matchedVariant?.note ? <p className="variant-note">{matchedVariant.note}</p> : null}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <p>No models found. Add directories in Settings to scan for local models.</p>
            </div>
          )}
        </Panel>
      </div>
    );
  }

  function renderImageDiscover() {
    return (
      <div className="image-discover-stack">
        <Panel
          title="Image Discover"
          subtitle={`${imageVariants.length} curated starter models / ${latestImageDiscoverResults.length} tracked latest models / live Hugging Face metadata`}
        >
          <div className="image-hero">
            <div>
              <span className="eyebrow">Curated</span>
              <h3>Discover curated local image models, then scout newer live releases without leaving ChaosEngineAI.</h3>
              <p className="muted-text">
                The curated lane stays optimized for ChaosEngineAI workflows, while the latest lane keeps an eye on newer
                diffusers-compatible image releases with live metadata and access signals.
              </p>
            </div>
            <div className="image-hero-actions">
              <button className="secondary-button" type="button" onClick={() => setActiveTab("image-models")}>
                Installed Models
              </button>
              <button className="primary-button" type="button" onClick={() => openImageStudio(selectedImageVariant?.id)}>
                Open Studio
              </button>
            </div>
          </div>
          <div className="callout image-callout image-runtime-callout">
            <p>{imageRuntimeStatus.message}</p>
            <div className="chip-row">
              <span className={`badge ${imageRuntimeStatus.realGenerationAvailable ? "success" : "warning"}`}>
                {imageRuntimeStatus.realGenerationAvailable
                  ? "Real engine ready"
                  : imageRuntimeStatus.activeEngine === "unavailable"
                    ? "Runtime unavailable"
                    : "Fallback active"}
              </span>
              <span className="badge muted">Engine: {imageRuntimeStatus.activeEngine}</span>
              {imageRuntimeStatus.device ? <span className="badge muted">Device: {imageRuntimeStatus.device}</span> : null}
              {(imageRuntimeStatus.missingDependencies ?? []).slice(0, 4).map((dependency) => (
                <span key={dependency} className="badge subtle">{dependency}</span>
              ))}
            </div>
            {!imageRuntimeStatus.realGenerationAvailable ? (
              <div className="image-runtime-actions">
                {imageRuntimeStatus.pythonExecutable ?? tauriBackend?.pythonExecutable ? (
                  <span className="mono-text muted-text">
                    Backend Python: {imageRuntimeStatus.pythonExecutable ?? tauriBackend?.pythonExecutable}
                  </span>
                ) : null}
                {tauriBackend?.managedByTauri ? (
                  <button className="secondary-button" type="button" onClick={() => void handleRestartServer()} disabled={busy}>
                    {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend"}
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>

          <div className="image-discover-filter-row">
            <label className="image-discover-search">
              Search
              <input
                className="text-input"
                type="search"
                value={imageDiscoverSearchInput}
                onChange={(event) => setImageDiscoverSearchInput(event.target.value)}
                placeholder="Search FLUX, SDXL, provider, task, tags, license..."
              />
            </label>
            <label>
              Show
              <select
                className="text-input"
                value={imageDiscoverSourceFilter}
                onChange={(event) => setImageDiscoverSourceFilter(event.target.value as ImageDiscoverSourceFilter)}
              >
                <option value="all">Curated + Latest</option>
                <option value="curated">Curated only</option>
                <option value="latest">Latest only</option>
              </select>
            </label>
            <label>
              Task
              <select
                className="text-input"
                value={imageDiscoverTaskFilter}
                onChange={(event) => setImageDiscoverTaskFilter(event.target.value as ImageDiscoverTaskFilter)}
              >
                <option value="all">All tasks</option>
                <option value="txt2img">Text to image</option>
                <option value="img2img">Image to image</option>
                <option value="inpaint">Inpaint</option>
              </select>
            </label>
            <label>
              Access
              <select
                className="text-input"
                value={imageDiscoverAccessFilter}
                onChange={(event) => setImageDiscoverAccessFilter(event.target.value as ImageDiscoverAccessFilter)}
              >
                <option value="all">Open + gated</option>
                <option value="open">Open only</option>
                <option value="gated">Gated only</option>
              </select>
            </label>
            <div className="image-discover-filter-actions">
              <button
                className="secondary-button"
                type="button"
                onClick={() => {
                  setImageDiscoverSearchInput("");
                  setImageDiscoverSourceFilter("all");
                  setImageDiscoverTaskFilter("all");
                  setImageDiscoverAccessFilter("all");
                }}
                disabled={!imageDiscoverHasActiveFilters}
              >
                Clear Filters
              </button>
            </div>
          </div>

          <div className="image-discover-results-summary">
            <span>
              Showing {filteredImageDiscoverFamilies.length} curated{" "}
              {filteredImageDiscoverFamilies.length === 1 ? "family" : "families"} and {filteredLatestImageDiscoverResults.length} latest release
              {filteredLatestImageDiscoverResults.length === 1 ? "" : "s"}.
            </span>
            {imageDiscoverSearchQuery ? (
              <span className="badge subtle">Search: {imageDiscoverSearchInput.trim()}</span>
            ) : null}
            {imageDiscoverSourceFilter !== "all" ? (
              <span className="badge muted">
                {imageDiscoverSourceFilter === "curated" ? "Curated only" : "Latest only"}
              </span>
            ) : null}
            {imageDiscoverTaskFilter !== "all" ? (
              <span className="badge muted">Task: {imageDiscoverTaskFilter}</span>
            ) : null}
            {imageDiscoverAccessFilter !== "all" ? (
              <span className="badge muted">
                Access: {imageDiscoverAccessFilter === "open" ? "Open only" : "Gated only"}
              </span>
            ) : null}
          </div>
        </Panel>

        {showCuratedImageDiscoverSection ? (
          <Panel
            title="Curated Picks"
            subtitle={filteredImageDiscoverFamilies.length > 0
              ? `${filteredImageDiscoverFamilies.length} curated families ready for local-first workflows`
              : "No curated image models match the current filters"}
            className="image-discover-section-panel"
          >
            {filteredImageDiscoverFamilies.length === 0 ? (
              <div className="empty-state image-empty-state">
                <p>Try broadening the filters or search terms to bring curated image models back into view.</p>
              </div>
            ) : (
              <div className="image-discover-grid image-discover-grid--curated">
                {filteredImageDiscoverFamilies.map((family) => {
                  const variant = defaultImageVariantForFamily(family);
                  if (!variant) return null;
                  const downloadState = activeImageDownloads[variant.repo];
                  const isDownloadPaused = downloadState?.state === "cancelled";
                  const isDownloadComplete = downloadState?.state === "completed";
                  const isDownloadFailed = downloadState?.state === "failed";
                  const friendlyDownloadError = formatImageAccessError(downloadState?.error, variant);
                  const needsGatedAccess = isGatedImageAccessError(downloadState?.error);
                  return (
                    <article key={family.id} className="image-family-card">
                      <div className="image-family-card-head">
                        <div>
                          <div className="image-family-title-row">
                            <h3>{family.name}</h3>
                            <span className="badge muted">{family.provider}</span>
                            {variant.availableLocally ? <span className="badge success">Installed</span> : null}
                            {!variant.availableLocally && isDownloadComplete ? <span className="badge success">Downloaded</span> : null}
                            {isDownloadPaused ? <span className="badge warning">Paused</span> : null}
                            {isDownloadFailed ? <span className="badge warning">Download Failed</span> : null}
                          </div>
                          <p>{family.headline}</p>
                        </div>
                        <span className="badge accent">{family.updatedLabel}</span>
                      </div>

                      <div className="image-family-meta">
                        <span>{imagePrimarySizeLabel(variant)}</span>
                        {imageSecondarySizeLabel(variant) ? <span>{imageSecondarySizeLabel(variant)}</span> : null}
                        <span>{variant.recommendedResolution}</span>
                        <span>{variant.estimatedGenerationSeconds ? `~${number(variant.estimatedGenerationSeconds)}s` : "Stub timing"}</span>
                      </div>

                      <div className="image-family-meta">
                        {variant.downloadsLabel ? <span>{variant.downloadsLabel}</span> : null}
                        {variant.likesLabel ? <span>{variant.likesLabel}</span> : null}
                        {variant.license ? <span>{formatImageLicenseLabel(variant.license)}</span> : null}
                        {typeof variant.gated === "boolean" ? <span>{variant.gated ? "Gated access" : "Open access"}</span> : null}
                      </div>

                      <div className="chip-row">
                        {family.badges.map((badge) => (
                          <span key={badge} className="badge muted">{badge}</span>
                        ))}
                        {variant.styleTags.map((tag) => (
                          <span key={tag} className="badge subtle">{tag}</span>
                        ))}
                        {variant.pipelineTag ? <span className="badge subtle">{variant.pipelineTag}</span> : null}
                      </div>

                      <p className="muted-text">{family.summary}</p>
                      <p className="muted-text">{variant.note}</p>
                      {isDownloadFailed && downloadState?.error ? (
                        <div className="callout error image-callout">
                          <p>{friendlyDownloadError}</p>
                          {needsGatedAccess ? (
                            <div className="button-row">
                              <button className="secondary-button" type="button" onClick={() => void handleOpenExternalUrl(variant.link)}>
                                Hugging Face
                              </button>
                              <button className="secondary-button" type="button" onClick={() => setActiveTab("settings")}>
                                Settings
                              </button>
                            </div>
                          ) : null}
                          {friendlyDownloadError !== downloadState.error ? (
                            <details className="debug-details">
                              <summary>Technical details</summary>
                              <p className="mono-text">{downloadState.error}</p>
                            </details>
                          ) : null}
                        </div>
                      ) : null}

                      <div className="button-row">
                        {variant.availableLocally ? (
                          <button className="primary-button" type="button" onClick={() => openImageStudio(variant.id)}>
                            Generate
                          </button>
                        ) : downloadState?.state === "downloading" ? (
                          <>
                            <span className="badge accent">{downloadProgressLabel(downloadState)}</span>
                            <button className="secondary-button" type="button" onClick={() => void handleCancelImageDownload(variant.repo)}>
                              Pause
                            </button>
                          </>
                        ) : isDownloadPaused ? (
                          <>
                            <span className="badge warning">{downloadProgressLabel(downloadState)}</span>
                            <button className="secondary-button" type="button" onClick={() => void handleImageDownload(variant.repo)}>
                              Resume
                            </button>
                          </>
                        ) : isDownloadComplete ? (
                          <span className="badge success">Download complete</span>
                        ) : (
                          <button className="secondary-button" type="button" onClick={() => void handleImageDownload(variant.repo)}>
                            {isDownloadFailed ? "Retry Download" : "Download"}
                          </button>
                        )}
                        <button className="secondary-button" type="button" onClick={() => openImageStudio(variant.id)}>
                          Use In Studio
                        </button>
                        <button className="secondary-button" type="button" onClick={() => void handleOpenExternalUrl(variant.link)}>
                          Hugging Face
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </Panel>
        ) : null}

        {showLatestImageDiscoverSection ? (
          <Panel
            title="Latest Releases"
            subtitle={filteredLatestImageDiscoverResults.length > 0
              ? `${filteredLatestImageDiscoverResults.length} live diffusers-compatible image releases`
              : "No latest tracked image releases match the current filters"}
            className="image-discover-section-panel image-discover-section-panel--latest"
          >
            <p className="muted-text image-discover-section-note">
              Latest Releases is a live scouting lane. These cards help you track newer official image repos before they are
              fully curated into ChaosEngineAI Studio defaults.
            </p>
            {filteredLatestImageDiscoverResults.length === 0 ? (
              <div className="empty-state image-empty-state">
                <p>Try broadening the filters or search terms to see the latest tracked image releases again.</p>
              </div>
            ) : (
              <div className="image-discover-grid image-discover-grid--latest">
                {filteredLatestImageDiscoverResults.map((variant) => renderLatestImageDiscoverCard(variant))}
              </div>
            )}
          </Panel>
        ) : null}
      </div>
    );
  }

  function renderImageModels() {
    return (
      <div className="content-grid image-page-grid">
        <Panel
          title="Installed Image Models"
          subtitle={installedImageVariants.length > 0
            ? `${installedImageVariants.length} curated models ready for Image Studio`
            : "No curated image models detected locally yet"}
          className="span-2"
          actions={
            <button className="secondary-button" type="button" onClick={() => setActiveTab("image-discover")}>
              Browse Catalog
            </button>
          }
        >
          {installedImageVariants.length === 0 ? (
            <div className="empty-state image-empty-state">
              <p>Download one of the curated image models from Image Discover to build out the local image library.</p>
            </div>
          ) : (
            <div className="image-library-grid">
              {installedImageVariants.map((variant) => {
                const family = imageCatalog.find((item) => item.variants.some((candidate) => candidate.id === variant.id));
                return (
                  <article key={variant.id} className="image-library-card">
                    <div className="image-library-card-head">
                      <div>
                        <h3>{variant.name}</h3>
                        <p>{family?.name ?? "Curated image model"}</p>
                      </div>
                      <span className="badge success">Installed</span>
                    </div>
                    <div className="image-library-stats">
                      <span>{sizeLabel(variant.sizeGb)}</span>
                      <span>{variant.runtime}</span>
                      <span>{variant.recommendedResolution}</span>
                    </div>
                    <div className="chip-row">
                      {variant.styleTags.map((tag) => (
                        <span key={tag} className="badge subtle">{tag}</span>
                      ))}
                    </div>
                    <p className="muted-text">{variant.note}</p>
                    <div className="button-row">
                      <button className="primary-button" type="button" onClick={() => openImageStudio(variant.id)}>
                        Generate
                      </button>
                      <button className="secondary-button" type="button" onClick={() => void handleOpenExternalUrl(variant.link)}>
                        Model Card
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </Panel>
      </div>
    );
  }

  function renderLatestImageDiscoverCard(variant: ImageModelVariant) {
    const downloadState = activeImageDownloads[variant.repo];
    const isDownloadPaused = downloadState?.state === "cancelled";
    const isDownloadComplete = downloadState?.state === "completed";
    const isDownloadFailed = downloadState?.state === "failed";
    const friendlyDownloadError = formatImageAccessError(downloadState?.error, variant);
    const needsGatedAccess = isGatedImageAccessError(downloadState?.error);
    return (
      <article key={variant.id} className="image-family-card image-family-card--latest">
        <div className="image-family-card-head">
          <div>
            <div className="image-family-title-row">
              <h3>{variant.name}</h3>
              <span className="badge muted">{variant.provider}</span>
              <span className="badge accent">Latest</span>
              {variant.availableLocally ? <span className="badge success">Installed</span> : null}
              {!variant.availableLocally && isDownloadComplete ? <span className="badge success">Downloaded</span> : null}
              {isDownloadPaused ? <span className="badge warning">Paused</span> : null}
              {isDownloadFailed ? <span className="badge warning">Download Failed</span> : null}
            </div>
            <p>{variant.note}</p>
          </div>
          <span className="badge muted">{variant.updatedLabel ?? "Recently updated"}</span>
        </div>

        <div className="image-family-meta">
          <span>{imagePrimarySizeLabel(variant)}</span>
          {imageSecondarySizeLabel(variant) ? <span>{imageSecondarySizeLabel(variant)}</span> : null}
          <span>{variant.recommendedResolution}</span>
          {variant.pipelineTag ? <span>{variant.pipelineTag}</span> : null}
        </div>

        <div className="image-family-meta">
          {variant.downloadsLabel ? <span>{variant.downloadsLabel}</span> : null}
          {variant.likesLabel ? <span>{variant.likesLabel}</span> : null}
          {variant.license ? <span>{formatImageLicenseLabel(variant.license)}</span> : null}
          {typeof variant.gated === "boolean" ? <span>{variant.gated ? "Gated access" : "Open access"}</span> : null}
        </div>

        <div className="chip-row">
          {variant.taskSupport.map((task) => (
            <span key={task} className="badge muted">{task}</span>
          ))}
          {variant.styleTags.map((tag) => (
            <span key={tag} className="badge subtle">{tag}</span>
          ))}
        </div>

        <p className="muted-text">
          Latest Releases is a live scouting lane for newer official image repos. These models are visible here before they
          become fully curated ChaosEngineAI Studio defaults.
        </p>

        {isDownloadFailed && downloadState?.error ? (
          <div className="callout error image-callout">
            <p>{friendlyDownloadError}</p>
            {needsGatedAccess ? (
              <div className="button-row">
                <button className="secondary-button" type="button" onClick={() => void handleOpenExternalUrl(variant.link)}>
                  Hugging Face
                </button>
                <button className="secondary-button" type="button" onClick={() => setActiveTab("settings")}>
                  Settings
                </button>
              </div>
            ) : null}
            {friendlyDownloadError !== downloadState.error ? (
              <details className="debug-details">
                <summary>Technical details</summary>
                <p className="mono-text">{downloadState.error}</p>
              </details>
            ) : null}
          </div>
        ) : null}

        <div className="button-row">
          {variant.availableLocally ? (
            <span className="badge success">Downloaded locally</span>
          ) : downloadState?.state === "downloading" ? (
            <>
              <span className="badge accent">{downloadProgressLabel(downloadState)}</span>
              <button className="secondary-button" type="button" onClick={() => void handleCancelImageDownload(variant.repo)}>
                Pause
              </button>
            </>
          ) : isDownloadPaused ? (
            <>
              <span className="badge warning">{downloadProgressLabel(downloadState)}</span>
              <button className="secondary-button" type="button" onClick={() => void handleImageDownload(variant.repo)}>
                Resume
              </button>
            </>
          ) : isDownloadComplete ? (
            <span className="badge success">Download complete</span>
          ) : (
            <button className="secondary-button" type="button" onClick={() => void handleImageDownload(variant.repo)}>
              {isDownloadFailed ? "Retry Download" : "Download"}
            </button>
          )}
          <button className="secondary-button" type="button" onClick={() => void handleOpenExternalUrl(variant.link)}>
            Open on Hugging Face
          </button>
        </div>
      </article>
    );
  }

  function renderImageStudio() {
    const selectedRatioPreset =
      IMAGE_RATIO_PRESETS.find((preset) => preset.width === imageWidth && preset.height === imageHeight) ??
      IMAGE_RATIO_PRESETS.find((preset) => preset.id === imageRatioId) ??
      IMAGE_RATIO_PRESETS[0];
    const selectedQuality =
      IMAGE_QUALITY_PRESETS.find((preset) => preset.id === imageQualityPreset) ?? IMAGE_QUALITY_PRESETS[1];
    const selectedImageDownload = selectedImageVariant ? activeImageDownloads[selectedImageVariant.repo] : undefined;
    const selectedImageDownloadPaused = selectedImageDownload?.state === "cancelled";
    const selectedImageDownloadComplete = selectedImageDownload?.state === "completed";
    const selectedImageDownloadFailed = selectedImageDownload?.state === "failed";
    const selectedImageFriendlyDownloadError = formatImageAccessError(selectedImageDownload?.error, selectedImageVariant);
    const selectedImageNeedsGatedAccess = isGatedImageAccessError(selectedImageDownload?.error);

    return (
      <div className="content-grid image-page-grid">
        <Panel
          title="Image Studio"
          subtitle={selectedImageVariant
            ? `${selectedImageVariant.name} / ${selectedImageVariant.runtime} / ${imageOutputs.length} saved outputs`
            : "Choose a model, prompt it, and iterate on saved outputs"}
          className="span-2"
          actions={
            <div className="button-row">
              <button className="secondary-button" type="button" onClick={() => setActiveTab("image-discover")}>
                Discover
              </button>
              <button className="secondary-button" type="button" onClick={() => setActiveTab("image-models")}>
                Installed
              </button>
              <button className="secondary-button" type="button" onClick={() => openImageGallery()}>
                Gallery
              </button>
            </div>
          }
        >
          <div className="image-studio-hero">
            <div>
              <span className="eyebrow">Current Runtime</span>
              <h3>{selectedImageVariant?.name ?? "Select an image model"}</h3>
              <p className="muted-text">
                This screen is the MVP vertical slice from the image plan: local model selection, prompt controls, saved
                outputs, and quick iteration actions. When the optional image runtime packages are installed, this view can
                generate locally; otherwise it falls back to placeholder outputs while keeping the same contract stable.
              </p>
            </div>
            {selectedImageVariant ? (
              <div className="image-studio-hero-stats">
                <span className="badge muted">{selectedImageFamily?.name ?? selectedImageVariant.provider}</span>
                <span className="badge muted">{selectedImageVariant.recommendedResolution}</span>
                <span className="badge muted">{sizeLabel(selectedImageVariant.sizeGb)}</span>
                {selectedImageVariant.availableLocally ? <span className="badge success">Installed</span> : null}
                {selectedImageLoaded ? <span className="badge success">Loaded In Memory</span> : null}
                {selectedImageWillLoadOnGenerate ? <span className="badge subtle">Loads On First Generate</span> : null}
                {imageBusy && selectedImageWillLoadOnGenerate ? <span className="badge accent">Loading Into Memory</span> : null}
                {!selectedImageVariant.availableLocally && selectedImageDownloadComplete ? <span className="badge success">Downloaded</span> : null}
              </div>
            ) : null}
          </div>
          <div className="callout image-callout image-runtime-callout">
            <p>{imageRuntimeStatus.message}</p>
            <div className="chip-row">
              <span className={`badge ${imageRuntimeStatus.realGenerationAvailable ? "success" : "warning"}`}>
                {imageRuntimeStatus.realGenerationAvailable
                  ? "Real local generation available"
                  : imageRuntimeStatus.activeEngine === "unavailable"
                    ? "Runtime unavailable"
                    : "Using placeholder outputs"}
              </span>
              <span className="badge muted">Engine: {imageRuntimeStatus.activeEngine}</span>
              {imageRuntimeStatus.device ? <span className="badge muted">Device: {imageRuntimeStatus.device}</span> : null}
              {(imageRuntimeStatus.missingDependencies ?? []).slice(0, 4).map((dependency) => (
                <span key={dependency} className="badge subtle">{dependency}</span>
              ))}
            </div>
            {selectedImageVariant && imageRuntimeStatus.realGenerationAvailable ? (
              <div className="image-runtime-summary">
                <p className="muted-text">
                  {selectedImageLoaded
                    ? `${selectedImageVariant.name} is loaded in memory and ready to generate immediately.`
                    : imageRuntimeLoadedDifferentModel && loadedImageVariant
                      ? `${loadedImageVariant.name} is currently loaded in memory. Generating with ${selectedImageVariant.name} will swap the pipeline and take a little longer.`
                      : selectedImageWillLoadOnGenerate
                        ? `${selectedImageVariant.name} is installed locally but not loaded yet. The first generate will take longer while the diffusion pipeline warms up.`
                        : !selectedImageVariant.availableLocally
                          ? `${selectedImageVariant.name} is not installed locally yet, so Studio cannot keep it loaded in memory on this machine.`
                          : "This model will load on demand when you generate."}
                </p>
                {imageBusy && selectedImageWillLoadOnGenerate ? (
                  <p className="busy-indicator"><span className="busy-dot" />First run may take longer while the model loads into memory.</p>
                ) : null}
                {(selectedImageVariant.availableLocally || loadedImageVariant) ? (
                  <div className="button-row image-runtime-control-row">
                    {selectedImageVariant.availableLocally && !selectedImageLoaded ? (
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => void handlePreloadImageModel(selectedImageVariant)}
                        disabled={imageBusy || busy || !backendOnline}
                      >
                        Preload Model
                      </button>
                    ) : null}
                    {selectedImageLoaded ? (
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => void handleUnloadImageModel(selectedImageVariant)}
                        disabled={imageBusy || busy || !backendOnline}
                      >
                        Unload Model
                      </button>
                    ) : null}
                    {!selectedImageLoaded && loadedImageVariant ? (
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => void handleUnloadImageModel()}
                        disabled={imageBusy || busy || !backendOnline}
                      >
                        Unload {loadedImageVariant.name}
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : null}
            {!imageRuntimeStatus.realGenerationAvailable ? (
              <div className="image-runtime-actions">
                <p className="muted-text">
                  If you installed the image packages recently, restart the backend so Image Studio can re-probe the
                  Python environment instead of quietly staying on placeholders.
                </p>
                <div className="button-row">
                  {imageRuntimeStatus.pythonExecutable ?? tauriBackend?.pythonExecutable ? (
                    <span className="mono-text muted-text">
                      Backend Python: {imageRuntimeStatus.pythonExecutable ?? tauriBackend?.pythonExecutable}
                    </span>
                  ) : null}
                  {tauriBackend?.managedByTauri ? (
                    <button className="secondary-button" type="button" onClick={() => void handleRestartServer()} disabled={busy}>
                      {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend"}
                    </button>
                  ) : null}
                </div>
              </div>
            ) : null}
          </div>
        </Panel>

        <Panel
          title="Prompt"
          subtitle="Choose a curated model, set the aspect ratio and quality, then generate into the local gallery."
          className="image-studio-form-panel"
          actions={
            <button
              className="primary-button"
              type="button"
              onClick={() => void submitImageGeneration()}
              disabled={imageBusy || !selectedImageVariant}
            >
              {imageBusy ? "Generating..." : "Generate"}
            </button>
          }
        >
          <div className="image-form-stack">
            <label>
              Model
              <select
                className="text-input"
                value={selectedImageModelId}
                onChange={(event) => setSelectedImageModelId(event.target.value)}
              >
                {imageCatalog.map((family) => (
                  <optgroup key={family.id} label={family.name}>
                    {family.variants.map((variant) => (
                      <option key={variant.id} value={variant.id}>
                        {variant.name}{variant.availableLocally ? " - installed" : ""}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </label>

            {!selectedImageVariant?.availableLocally && selectedImageVariant ? (
              <div className="callout image-callout">
                <p>
                  {selectedImageDownloadFailed
                    ? `${selectedImageVariant.name} did not finish downloading correctly. ChaosEngineAI only found a partial local snapshot, so it cannot load the real image pipeline yet.`
                    : selectedImageDownloadPaused
                    ? `${selectedImageVariant.name} is partially downloaded. Resume when you're ready and ChaosEngineAI will continue from the files already on disk.`
                    : selectedImageDownloadComplete
                    ? `${selectedImageVariant.name} finished downloading. The installed-model scan will refresh automatically.`
                    : `${selectedImageVariant.name} is not installed locally yet, so the real runtime cannot use it on this machine yet. You can still browse the studio flow now, then download the curated weights before generating for real.`}
                </p>
                {selectedImageDownloadFailed && selectedImageDownload?.error ? (
                  <>
                    <p className="muted-text">{selectedImageFriendlyDownloadError}</p>
                    {selectedImageNeedsGatedAccess ? (
                      <div className="button-row">
                        <button className="secondary-button" type="button" onClick={() => void handleOpenExternalUrl(selectedImageVariant.link)}>
                          Hugging Face
                        </button>
                        <button className="secondary-button" type="button" onClick={() => setActiveTab("settings")}>
                          Settings
                        </button>
                      </div>
                    ) : null}
                    {selectedImageFriendlyDownloadError !== selectedImageDownload.error ? (
                      <details className="debug-details">
                        <summary>Technical details</summary>
                        <p className="mono-text">{selectedImageDownload.error}</p>
                      </details>
                    ) : null}
                  </>
                ) : null}
                <div className="button-row">
                  {selectedImageDownload?.state === "downloading" ? (
                    <>
                      <span className="badge accent">{downloadProgressLabel(selectedImageDownload)}</span>
                      <button className="secondary-button" type="button" onClick={() => void handleCancelImageDownload(selectedImageVariant.repo)}>
                        Pause
                      </button>
                    </>
                  ) : selectedImageDownloadPaused ? (
                    <>
                      <span className="badge warning">{downloadProgressLabel(selectedImageDownload)}</span>
                      <button className="secondary-button" type="button" onClick={() => void handleImageDownload(selectedImageVariant.repo)}>
                        Resume
                      </button>
                    </>
                  ) : selectedImageDownloadComplete ? (
                    <span className="badge success">Download complete</span>
                  ) : (
                    <button className="secondary-button" type="button" onClick={() => void handleImageDownload(selectedImageVariant.repo)}>
                      {selectedImageDownloadFailed ? "Retry Download" : "Download Model"}
                    </button>
                  )}
                  <button className="secondary-button" type="button" onClick={() => void handleOpenExternalUrl(selectedImageVariant.link)}>
                    Hugging Face
                  </button>
                </div>
              </div>
            ) : null}

            <label>
              Prompt
              <textarea
                className="text-input prompt-area"
                rows={5}
                placeholder="Moody cinematic alleyway after rain, neon reflections, 35mm photo, shallow depth of field"
                value={imagePrompt}
                onChange={(event) => setImagePrompt(event.target.value)}
              />
            </label>

            <label>
              Negative prompt
              <textarea
                className="text-input prompt-area prompt-area--secondary"
                rows={3}
                placeholder="blurry, deformed hands, extra limbs, overexposed"
                value={imageNegativePrompt}
                onChange={(event) => setImageNegativePrompt(event.target.value)}
              />
            </label>

            <div className="control-stack">
              <span className="eyebrow">Aspect Ratio</span>
              <div className="image-pill-row">
                {IMAGE_RATIO_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    className={selectedRatioPreset.id === preset.id ? "pill-button active" : "pill-button"}
                    type="button"
                    onClick={() => applyImageRatioPreset(preset.id)}
                  >
                    <strong>{preset.label}</strong>
                    <span>{preset.hint}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="control-stack">
              <span className="eyebrow">Quality Preset</span>
              <div className="image-pill-row">
                {IMAGE_QUALITY_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    className={selectedQuality.id === preset.id ? "pill-button active" : "pill-button"}
                    type="button"
                    onClick={() => applyImageQuality(preset.id)}
                  >
                    <strong>{preset.label}</strong>
                    <span>{preset.hint}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="field-grid image-field-grid">
              <label>
                Width
                <input
                  className="text-input"
                  type="number"
                  min={256}
                  max={2048}
                  step={64}
                  value={imageWidth}
                  onChange={(event) => setImageWidth(Number(event.target.value) || 1024)}
                />
              </label>
              <label>
                Height
                <input
                  className="text-input"
                  type="number"
                  min={256}
                  max={2048}
                  step={64}
                  value={imageHeight}
                  onChange={(event) => setImageHeight(Number(event.target.value) || 1024)}
                />
              </label>
              <label>
                Steps
                <input
                  className="text-input"
                  type="number"
                  min={1}
                  max={100}
                  value={imageSteps}
                  onChange={(event) => setImageSteps(Number(event.target.value) || 24)}
                />
              </label>
              <label>
                Guidance
                <input
                  className="text-input"
                  type="number"
                  min={1}
                  max={20}
                  step={0.5}
                  value={imageGuidance}
                  onChange={(event) => setImageGuidance(Number(event.target.value) || 6)}
                />
              </label>
              <label>
                Images
                <input
                  className="text-input"
                  type="number"
                  min={1}
                  max={4}
                  value={imageBatchSize}
                  onChange={(event) => setImageBatchSize(Math.max(1, Math.min(4, Number(event.target.value) || 1)))}
                />
              </label>
              <label className="checkbox-card">
                <span className="checkbox-card-label">Random seed</span>
                <input
                  type="checkbox"
                  checked={imageUseRandomSeed}
                  onChange={(event) => setImageUseRandomSeed(event.target.checked)}
                />
              </label>
            </div>

            {!imageUseRandomSeed ? (
              <label>
                Seed
                <input
                  className="text-input"
                  type="number"
                  min={0}
                  max={2147483647}
                  value={imageSeedInput}
                  onChange={(event) => setImageSeedInput(event.target.value)}
                />
              </label>
            ) : null}

            {imageBusyLabel ? (
              <p className="busy-indicator"><span className="busy-dot" />{imageBusyLabel}</p>
            ) : null}
          </div>
        </Panel>

        <Panel
          title="Recent Outputs"
          subtitle={imageOutputs.length > 0 ? `${recentImageOutputs.length} newest of ${imageOutputs.length} saved generations` : "Generated images will appear here"}
          className="image-gallery-panel"
          actions={
            <button className="secondary-button" type="button" onClick={() => openImageGallery()}>
              Open Gallery
            </button>
          }
        >
          {imageOutputs.length === 0 ? (
            <div className="empty-state image-empty-state">
              <p>Generate a prompt to create the first saved image artifact for this branch.</p>
            </div>
          ) : (
            <div className="image-gallery-grid">
              {recentImageOutputs.map((artifact) => renderImageOutputCard(artifact))}
            </div>
          )}
          {imageOutputs.length > recentImageOutputs.length ? (
            <p className="muted-text image-gallery-footnote">
              Showing the newest {recentImageOutputs.length} saved images here. Open Image Gallery to browse everything,
              filter by model, and manage older runs.
            </p>
          ) : null}
        </Panel>
      </div>
    );
  }

  function renderImageGallery() {
    return (
      <div className="content-grid image-page-grid">
        <Panel
          title="Image Gallery"
          subtitle={imageOutputs.length > 0
            ? `${filteredImageOutputs.length} of ${imageOutputs.length} saved outputs`
            : "Saved generations, filters, and quick reuse actions"}
          className="span-2"
          actions={
            <div className="button-row">
              <button className="secondary-button" type="button" onClick={() => openImageStudio()}>
                Studio
              </button>
              <button className="secondary-button" type="button" onClick={() => setActiveTab("image-models")}>
                Installed
              </button>
            </div>
          }
        >
          <div className="image-studio-hero">
            <div>
              <span className="eyebrow">Saved Outputs</span>
              <h3>Browse, filter, and reuse generated images</h3>
              <p className="muted-text">
                Keep Image Studio focused on prompting and generation, then use Image Gallery to search old outputs,
                compare models, and jump back into Studio with the same settings.
              </p>
            </div>
            <div className="image-studio-hero-stats">
              <span className="badge muted">{imageOutputs.length} saved</span>
              <span className="badge muted">{imageGalleryModelCount} models used</span>
              {imageGalleryRealCount > 0 ? <span className="badge success">{imageGalleryRealCount} real engine</span> : null}
              {imageGalleryPlaceholderCount > 0 ? <span className="badge warning">{imageGalleryPlaceholderCount} placeholder</span> : null}
              {imageGalleryWarningCount > 0 ? <span className="badge subtle">{imageGalleryWarningCount} with notes</span> : null}
            </div>
          </div>

          <div className="image-gallery-toolbar">
            <label className="image-gallery-search">
              Search
              <input
                className="text-input"
                type="search"
                placeholder="Prompt, model, runtime note"
                value={imageGallerySearchInput}
                onChange={(event) => setImageGallerySearchInput(event.target.value)}
              />
            </label>
            <label>
              Model
              <select
                className="text-input"
                value={imageGalleryModelFilter}
                onChange={(event) => setImageGalleryModelFilter(event.target.value)}
              >
                <option value="all">All models</option>
                {imageGalleryModelOptions.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Runtime
              <select
                className="text-input"
                value={imageGalleryRuntimeFilter}
                onChange={(event) => setImageGalleryRuntimeFilter(event.target.value as ImageGalleryRuntimeFilter)}
              >
                <option value="all">All runtimes</option>
                <option value="diffusers">Real engine</option>
                <option value="placeholder">Placeholder</option>
                <option value="warning">With notes</option>
              </select>
            </label>
            <label>
              Frame
              <select
                className="text-input"
                value={imageGalleryOrientationFilter}
                onChange={(event) => setImageGalleryOrientationFilter(event.target.value as ImageGalleryOrientationFilter)}
              >
                <option value="all">All frames</option>
                <option value="square">Square</option>
                <option value="portrait">Portrait</option>
                <option value="landscape">Landscape</option>
              </select>
            </label>
            <label>
              Sort
              <select
                className="text-input"
                value={imageGallerySort}
                onChange={(event) => setImageGallerySort(event.target.value as ImageGallerySort)}
              >
                <option value="newest">Newest first</option>
                <option value="oldest">Oldest first</option>
              </select>
            </label>
          </div>

          {imageGalleryHasActiveFilters ? (
            <div className="button-row image-gallery-toolbar-actions">
              <span className="muted-text">
                Showing {filteredImageOutputs.length} matching output{filteredImageOutputs.length === 1 ? "" : "s"}.
              </span>
              <button className="secondary-button" type="button" onClick={resetImageGalleryFilters}>
                Clear Filters
              </button>
            </div>
          ) : null}
        </Panel>

        <Panel
          title="Saved Outputs"
          subtitle={filteredImageOutputs.length > 0
            ? `${filteredImageOutputs.length} image${filteredImageOutputs.length === 1 ? "" : "s"} ready to browse`
            : imageOutputs.length > 0
              ? "No saved outputs match the current filters"
              : "Generate in Image Studio to start building the gallery"}
          className="span-2 image-gallery-page-panel"
        >
          {filteredImageOutputs.length === 0 ? (
            <div className="empty-state image-empty-state">
              <div className="image-empty-state-copy">
                <p>
                  {imageOutputs.length === 0
                    ? "Generate a prompt in Image Studio to create the first saved image artifact for this branch."
                    : "No saved images match the current filters yet. Try broadening the search or clearing one of the filters."}
                </p>
                <div className="button-row">
                  {imageOutputs.length === 0 ? (
                    <button className="secondary-button" type="button" onClick={() => openImageStudio()}>
                      Open Studio
                    </button>
                  ) : (
                    <button className="secondary-button" type="button" onClick={resetImageGalleryFilters}>
                      Clear Filters
                    </button>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="image-gallery-grid">
              {filteredImageOutputs.map((artifact) => renderImageOutputCard(artifact))}
            </div>
          )}
        </Panel>
      </div>
    );
  }

  function renderImageGenerationModal() {
    if (!showImageGenerationModal) {
      return null;
    }

    const activeArtifact = selectedImageGenerationArtifact;
    const runInfo = imageGenerationRunInfo;
    const activeArtifactVariant = activeArtifact ? findImageVariantById(imageCatalog, activeArtifact.modelId) : null;
    const activeArtifactRuntimeNote = formatImageAccessError(activeArtifact?.runtimeNote, activeArtifactVariant);
    const activeArtifactNeedsGatedAccess = isGatedImageAccessError(activeArtifact?.runtimeNote);
    const imagePhases: LiveProgressPhase[] = [
      ...(runInfo?.needsPipelineLoad
        ? [{ id: "load", label: "Loading model into memory", estimatedSeconds: 12 }]
        : []),
      { id: "prompt", label: "Encoding prompt", estimatedSeconds: 3 },
      {
        id: "diffuse",
        label: `Diffusing ${runInfo?.batchSize ?? 1} image${(runInfo?.batchSize ?? 1) > 1 ? "s" : ""}`,
        estimatedSeconds: Math.max(10, Math.round(((runInfo?.steps ?? imageSteps) * 0.9) + ((runInfo?.batchSize ?? 1) * 4))),
      },
      { id: "decode", label: "Decoding pixels", estimatedSeconds: 4 },
      { id: "save", label: "Saving to output gallery", estimatedSeconds: 2 },
    ];

    return (
      <div className="modal-overlay image-result-modal">
        <div className="modal-content" onClick={(event) => event.stopPropagation()}>
          <div className="modal-header">
            <h3>
              {imageBusy
                ? "Generating image"
                : imageGenerationError
                  ? "Image generation failed"
                  : imageGenerationArtifacts.length > 1
                    ? "Images ready"
                    : "Image ready"}
            </h3>
            {!imageBusy && !imageGenerationError && activeArtifact ? (
              <p>
                {activeArtifact.modelName} · {formatImageTimestamp(activeArtifact.createdAt)}
              </p>
            ) : null}
          </div>
          <div className="modal-body">
            {imageBusy && imageGenerationStartedAt ? (
              <LiveProgress
                title="Generating image"
                subtitle={runInfo?.modelName ?? selectedImageVariant?.name ?? undefined}
                startedAt={imageGenerationStartedAt}
                accent="image"
                phases={imagePhases}
              />
            ) : imageGenerationError ? (
              <div className="callout error">
                <h3>Image generation failed</h3>
                <p>{imageGenerationError}</p>
                <p className="muted-text">
                  Adjust the prompt or runtime settings, then try again. The gallery keeps any earlier successful outputs.
                </p>
              </div>
            ) : activeArtifact ? (
              <div className="image-generation-result">
                <div className="image-generation-preview-shell">
                  <img
                    src={activeArtifact.previewUrl}
                    alt={activeArtifact.prompt}
                    className="image-generation-preview"
                  />
                  {imageGenerationArtifacts.length > 1 ? (
                    <div className="image-generation-thumb-strip">
                      {imageGenerationArtifacts.map((artifact) => (
                        <button
                          key={artifact.artifactId}
                          className={`image-generation-thumb${artifact.artifactId === activeArtifact.artifactId ? " active" : ""}`}
                          type="button"
                          onClick={() => setSelectedImageGenerationArtifactId(artifact.artifactId)}
                        >
                          <img src={artifact.previewUrl} alt={artifact.prompt} />
                        </button>
                      ))}
                    </div>
                  ) : null}
                </div>
                <div className="image-generation-info">
                  <div className="chip-row">
                    <span className="badge success">Saved To Gallery</span>
                    {activeArtifact.runtimeLabel ? <span className="badge subtle">{activeArtifact.runtimeLabel}</span> : null}
                  </div>
                  <div>
                    <h3>{activeArtifact.modelName}</h3>
                    <p className="image-output-prompt">{activeArtifact.prompt}</p>
                    {activeArtifact.runtimeNote ? <p className="muted-text">{activeArtifactRuntimeNote}</p> : null}
                  </div>
                  {activeArtifactNeedsGatedAccess && activeArtifactVariant ? (
                    <div className="button-row">
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => void handleOpenExternalUrl(activeArtifactVariant.link)}
                      >
                        Hugging Face
                      </button>
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => {
                          setShowImageGenerationModal(false);
                          setActiveTab("settings");
                        }}
                      >
                        Settings
                      </button>
                    </div>
                  ) : null}
                  <div className="image-output-meta">
                    <span>{activeArtifact.width} x {activeArtifact.height}</span>
                    <span>{activeArtifact.steps} steps</span>
                    <span>CFG {activeArtifact.guidance}</span>
                    <span>Seed {activeArtifact.seed}</span>
                    <span>{number(activeArtifact.durationSeconds)}s</span>
                  </div>
                  {imageGenerationArtifacts.length > 1 ? (
                    <p className="muted-text">
                      Generated {imageGenerationArtifacts.length} images in this run. Click a thumbnail to inspect a different result.
                    </p>
                  ) : null}
                  <div className="button-row">
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => handleUseSameImageSettings(activeArtifact, true)}
                    >
                      Use Same Settings
                    </button>
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => void handleVaryImageSeed(activeArtifact)}
                      disabled={imageBusy}
                    >
                      Vary Seed
                    </button>
                  </div>
                  <div className="button-row">
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => void handleOpenExternalUrl(activeArtifact.imagePath ?? activeArtifact.previewUrl)}
                    >
                      Open
                    </button>
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => activeArtifact.imagePath ? void handleRevealPath(activeArtifact.imagePath) : void handleOpenExternalUrl(activeArtifact.previewUrl)}
                    >
                      Reveal
                    </button>
                    <button
                      className="secondary-button danger-button"
                      type="button"
                      onClick={() => void handleDeleteImageArtifact(activeArtifact.artifactId)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
          {!imageBusy ? (
            <div className="modal-footer">
              <button
                className="primary-button"
                type="button"
                onClick={() => setShowImageGenerationModal(false)}
              >
                {imageGenerationError ? "Close" : "Done"}
              </button>
            </div>
          ) : null}
        </div>
      </div>
    );
  }

  function renderConversion() {
    const beforeSize = conversionSource?.sizeGb ?? lastConversion?.sourceSizeGb ?? null;
    const estimatedContext = lastConversion?.contextWindow ?? conversionVariant?.contextWindow ?? "Varies";

    // --- Live projection from the conversion draft (responds immediately to slider changes) ---
    // Detect the source's effective bits-per-weight from name/quantization metadata
    const detectSourceBpw = (): number => {
      const haystack = `${conversionSource?.name ?? ""} ${conversionSource?.format ?? ""} ${conversionVariant?.quantization ?? ""} ${conversionVariant?.format ?? ""}`.toLowerCase();
      const match = haystack.match(/(\d)[\s-]?bit|q(\d)/);
      if (match) {
        const bits = Number(match[1] ?? match[2]);
        if (bits >= 2 && bits <= 8) return bits + 0.5; // +0.5 for group overhead
      }
      if (/bf16|fp16|float16|f16/.test(haystack)) return 16;
      if (/fp32|float32|f32/.test(haystack)) return 32;
      return 16; // safe default — assume bf16
    };
    const sourceBpw = detectSourceBpw();
    const isReQuantizing = sourceBpw < 12; // source is already quantized

    const dtypeBytes = conversionDraft.dtype === "float32" ? 4 : 2;
    // Group quantization adds ~16 bits scale/zero per group, amortized per weight
    const groupOverheadBitsPerWeight = conversionDraft.quantize ? 16 / Math.max(8, conversionDraft.qGroupSize) : 0;
    const effectiveBitsPerWeight = conversionDraft.quantize
      ? conversionDraft.qBits + groupOverheadBitsPerWeight
      : dtypeBytes * 8;

    // Project disk-after by scaling source disk size by the bits ratio (much more accurate than paramsB-based math)
    const projectedDiskGb = beforeSize ? beforeSize * (effectiveBitsPerWeight / sourceBpw) : null;
    const afterSize = lastConversion?.outputSizeGb ?? projectedDiskGb;

    // Quality model: anchored to MLX-LM published recovery numbers (vs FP16 reference)
    const qualityByBits: Record<number, number> = { 2: 78, 3: 90, 4: 96.5, 5: 98.5, 6: 99.3, 8: 99.85 };
    const baseQuality = conversionDraft.quantize ? (qualityByBits[conversionDraft.qBits] ?? 95) : 100;
    // Smaller groups = higher quality (up to +1.5); larger groups = up to -1.5
    const groupQualityShift = conversionDraft.quantize ? Math.max(-1.5, Math.min(1.5, ((64 - conversionDraft.qGroupSize) / 32) * 0.75)) : 0;
    // Re-quantization penalty: requantizing an already-quantized source loses additional quality
    let reQuantPenalty = 0;
    if (isReQuantizing && conversionDraft.quantize) {
      if (conversionDraft.qBits >= sourceBpw - 0.5) {
        reQuantPenalty = 0.5; // round-trip noise
      } else {
        // Going lower than source — losses compound
        const drop = sourceBpw - conversionDraft.qBits;
        reQuantPenalty = Math.min(20, drop * 6);
      }
    }
    const projectedQualityPercent = Math.min(100, Math.max(0, baseQuality + groupQualityShift - reQuantPenalty));

    // Speed projection: memory-bandwidth bound, scales inversely with effective bytes/weight vs source
    const speedupVsSource = sourceBpw / effectiveBitsPerWeight;
    const baseTokS = preview.estimatedTokS > 0 ? preview.estimatedTokS : 35;
    const projectedTokS = baseTokS * speedupVsSource;
    const estimatedTokS = lastConversion?.estimatedTokS ?? projectedTokS;

    const cacheBefore = lastConversion?.baselineCacheGb ?? preview.baselineCacheGb;
    const cacheAfter = lastConversion?.optimizedCacheGb ?? preview.optimizedCacheGb;

    const conversionCompression =
      beforeSize && afterSize && afterSize > 0
        ? `${number(beforeSize / afterSize)}x smaller on disk`
        : projectedDiskGb && beforeSize
          ? `≈ ${number(beforeSize / projectedDiskGb)}x projected`
          : "Pick a source and bits to project disk footprint";

    return (
      <div className="content-grid">
        <Panel
          title="MLX Conversion"
          subtitle="Prepare a local source, compare before and after stats, then convert into an MLX-ready output."
          className="span-2"
          actions={
            <span className={`badge ${conversionReady ? "success" : "warning"}`}>
              {conversionReady ? "Converter ready" : "Converter unavailable"}
            </span>
          }
        >
          <div className="conversion-layout">
            <div className="conversion-builder">
              {convertibleLibrary.length ? (
                <>
                  <div className="conversion-source-picker">
                    <span className="eyebrow">Source model</span>
                    {conversionSource ? (
                      <div className="model-selected-card">
                        <div className="model-selected-info">
                          <strong>{conversionSource.name}</strong>
                          <div className="model-selected-meta">
                            <span className="badge muted">{conversionSource.format}</span>
                            <span className="badge muted">{sizeLabel(conversionSource.sizeGb)}</span>
                            {conversionSource.directoryLabel ? <span className="badge muted">{conversionSource.directoryLabel}</span> : null}
                          </div>
                        </div>
                        <button className="secondary-button" type="button" onClick={() => setShowConversionPicker(true)}>
                          Change
                        </button>
                      </div>
                    ) : (
                      <button className="secondary-button" type="button" onClick={() => setShowConversionPicker(true)} style={{ width: "100%" }}>
                        Select a model to convert...
                      </button>
                    )}
                  </div>

                  <div className="field-grid">
                    <label>
                      Output path
                      <div className="input-with-button">
                        <input
                          className="text-input"
                          type="text"
                          placeholder="Leave blank to use ~/Models/<name>-mlx"
                          value={conversionDraft.outputPath}
                          onChange={(event) => updateConversionDraft("outputPath", event.target.value)}
                        />
                        <button
                          className="secondary-button"
                          type="button"
                          onClick={() => void handlePickConversionOutputDir()}
                          title="Choose output folder"
                        >
                          Browse...
                        </button>
                      </div>
                    </label>
                    {conversionSource?.format?.toUpperCase() === "GGUF" ? (
                      <label>
                        Base HF repo (required for GGUF)
                        <input
                          className="text-input"
                          type="text"
                          placeholder="e.g. Qwen/Qwen2.5-7B-Instruct"
                          value={conversionDraft.hfRepo}
                          onChange={(event) => updateConversionDraft("hfRepo", event.target.value)}
                        />
                      </label>
                    ) : null}
                  </div>

                  <SliderField
                    label="Quantization bits"
                    value={conversionDraft.qBits}
                    min={2} max={8} step={1}
                    ticks={[{ value: 2, label: "2" }, { value: 3, label: "3" }, { value: 4, label: "4" }, { value: 5, label: "5" }, { value: 6, label: "6" }, { value: 7, label: "7" }, { value: 8, label: "8" }]}
                    formatValue={(v) => `${v}-bit`}
                    onChange={(v) => updateConversionDraft("qBits", v)}
                  />

                  <SliderField
                    label="Group size"
                    value={conversionDraft.qGroupSize}
                    min={32} max={128} step={32}
                    ticks={[{ value: 32, label: "32" }, { value: 64, label: "64" }, { value: 96, label: "96" }, { value: 128, label: "128" }]}
                    formatValue={(v) => `${v} weights/group`}
                    onChange={(v) => updateConversionDraft("qGroupSize", v)}
                  />

                  <div className="field-grid">
                    <label>
                      Dtype
                      <select
                        className="text-input"
                        value={conversionDraft.dtype}
                        onChange={(event) => updateConversionDraft("dtype", event.target.value)}
                      >
                        <option value="float16">float16</option>
                        <option value="bfloat16">bfloat16</option>
                        <option value="float32">float32</option>
                      </select>
                    </label>
                    <label className="check-row">
                      <input
                        type="checkbox"
                        checked={conversionDraft.quantize}
                        onChange={(event) => updateConversionDraft("quantize", event.target.checked)}
                      />
                      Quantize converted weights
                    </label>
                    <div className={`callout ${isReQuantizing ? "warning" : "quiet"} compact-callout`}>
                      <h3>{isReQuantizing ? "Re-quantizing an already quantized source" : "Backend note"}</h3>
                      <p>
                        {isReQuantizing
                          ? `Source is already ~${Math.round(sourceBpw)}-bit. Going lower compounds quality loss — for best results convert from the original FP16/BF16 weights.`
                          : conversionReady
                            ? "mlx-lm conversion is available in the active backend."
                            : nativeBackends?.mlxMessage ?? "Start the native sidecar to enable conversion."}
                      </p>
                    </div>
                  </div>

                  <div className="button-row">
                    <button
                      className="primary-button"
                      type="button"
                      onClick={() => void handleConvertModel()}
                      disabled={!conversionReady || !conversionDraft.path || busy}
                    >
                      {busy ? "Converting..." : "Convert to MLX"}
                    </button>
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() =>
                        setConversionDraft({
                          modelRef: "",
                          path: "",
                          hfRepo: "",
                          outputPath: "",
                          quantize: true,
                          qBits: 4,
                          qGroupSize: 64,
                          dtype: "float16",
                        })
                      }
                    >
                      Clear
                    </button>
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  <p>Add model directories in Settings first, then conversion sources found there will appear here.</p>
                </div>
              )}
            </div>

            <div className="conversion-visuals">
              <div className="stat-grid compact-grid">
                <StatCard label="Params" value={conversionVariant ? `${number(conversionVariant.paramsB)}B` : (lastConversion?.paramsB ? `${number(lastConversion.paramsB)}B` : "Unknown")} hint={estimatedContext} />
                <StatCard
                  label="Disk before"
                  value={beforeSize ? sizeLabel(beforeSize) : "Unknown"}
                  hint={conversionSource?.format ?? lastConversion?.sourceFormat ?? "Source"}
                />
                <StatCard
                  label="Disk after"
                  value={afterSize ? sizeLabel(afterSize) : "Pending"}
                  hint={conversionCompression}
                />
                <StatCard
                  label="Est. tok/s"
                  value={`${number(estimatedTokS)} tok/s`}
                  hint={`Using ${launchCacheLabel}`}
                />
              </div>

              <div className="conversion-compare">
                <div className="conversion-card">
                  <span className="eyebrow">Before</span>
                  <h3>{conversionSource?.name ?? lastConversion?.sourceLabel ?? "Choose a source"}</h3>
                  <p>{conversionSource?.path ?? lastConversion?.sourcePath ?? "Select a local GGUF or HF-cache source to inspect its current footprint."}</p>
                  <div className="metric-list">
                    <div className="metric-row">
                      <span>Format</span>
                      <strong>{conversionSource?.format ?? lastConversion?.sourceFormat ?? "Unknown"}{isReQuantizing ? ` · ~${Math.round(sourceBpw)}-bit` : ""}</strong>
                    </div>
                    <div className="metric-row">
                      <span>On-disk size</span>
                      <strong>{beforeSize ? sizeLabel(beforeSize) : "Unknown"}</strong>
                    </div>
                    <div className="metric-row">
                      <span>Context</span>
                      <strong>{estimatedContext}</strong>
                    </div>
                    <div className="metric-row">
                      <span>Cache footprint</span>
                      <strong>{number(cacheBefore)} GB</strong>
                    </div>
                  </div>
                </div>

                <div className="conversion-arrow" aria-hidden="true">
                  <span>MLX</span>
                </div>

                <div className="conversion-card accent-card">
                  <span className="eyebrow">After</span>
                  <h3>{lastConversion ? "MLX-ready output" : "Target preview"}</h3>
                  <p>{lastConversion?.outputPath ?? "Converted output will appear here together with derived stats and metadata."}</p>
                  <div className="metric-list">
                    <div className="metric-row">
                      <span>Target profile</span>
                      <strong>{conversionDraft.quantize ? `${conversionDraft.qBits}-bit g${conversionDraft.qGroupSize}` : "Unquantized"} / {conversionDraft.dtype}</strong>
                    </div>
                    <div className="metric-row">
                      <span>On-disk size</span>
                      <strong>{afterSize ? sizeLabel(afterSize) : "Pending"}</strong>
                    </div>
                    <div className="metric-row">
                      <span>Cache footprint</span>
                      <strong>{number(cacheAfter)} GB</strong>
                    </div>
                    <div className="metric-row">
                      <span>Quality estimate</span>
                      <strong>{number(lastConversion?.qualityPercent ?? projectedQualityPercent, 1)}%</strong>
                    </div>
                  </div>
                </div>
              </div>

              <PerformancePreview
                preview={preview}
                availableMemoryGb={workspace.system.availableMemoryGb}
                totalMemoryGb={workspace.system.totalMemoryGb}
              />

              {lastConversion && !busy ? (
                <div className="callout">
                  <span className="badge success">Last conversion</span>
                  <h3>{lastConversion.sourceLabel}</h3>
                  <p>{lastConversion.outputPath}</p>
                  <div className="field-grid detail-grid">
                    <div>
                      <span className="eyebrow">Base repo</span>
                      <p>{lastConversion.hfRepo}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Architecture</span>
                      <p>{lastConversion.architecture ?? "Unknown"}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Context</span>
                      <p>{lastConversion.contextWindow ?? estimatedContext}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Compression</span>
                      <p>{lastConversion.compressionRatio ? `${number(lastConversion.compressionRatio)}x cache reduction` : conversionCompression}</p>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </Panel>

        <ModelPicker
          open={showConversionPicker}
          title="Select Source Model"
          library={workspace.library}
          filter={(item) => libraryItemFormat(item) !== "MLX"}
          selectedPath={conversionDraft.path || null}
          onSelect={(item, resolvedPath) => {
            prepareLibraryConversion(item, resolvedPath);
          }}
          onClose={() => setShowConversionPicker(false)}
        />

        {showConversionModal ? (
          <div className="modal-overlay conversion-result-modal">
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>
                  {busyAction === "Converting model..."
                    ? "Converting model"
                    : conversionError
                      ? "Conversion failed"
                      : "Conversion complete"}
                </h3>
              </div>
              <div className="modal-body">
                {busyAction === "Converting model..." && conversionStartedAt ? (
                  <LiveProgress
                    title="Converting model"
                    subtitle={conversionSource?.name ?? conversionDraft.modelRef ?? undefined}
                    startedAt={conversionStartedAt}
                    accent="convert"
                    phases={[
                      { id: "resolve", label: "Resolving source", estimatedSeconds: 3 },
                      { id: "download", label: "Fetching weights", estimatedSeconds: 60 },
                      { id: "load", label: "Loading into MLX", estimatedSeconds: 15 },
                      { id: "quantize", label: `Quantizing to ${conversionDraft.qBits}-bit g${conversionDraft.qGroupSize}`, estimatedSeconds: 45 },
                      { id: "shard", label: "Sharding & writing safetensors", estimatedSeconds: 10 },
                      { id: "verify", label: "Verifying output", estimatedSeconds: 5 },
                    ] as LiveProgressPhase[]}
                  />
                ) : conversionError ? (
                  <div className="callout error">
                    <h3>Conversion failed</h3>
                    <p>{conversionError}</p>
                    <details className="debug-details">
                      <summary>Debug details</summary>
                      <dl className="debug-grid">
                        <dt>Model ref</dt>
                        <dd><code>{conversionDraft.modelRef || "—"}</code></dd>
                        <dt>Source path</dt>
                        <dd><code>{conversionDraft.path || "—"}</code></dd>
                        <dt>HF repo override</dt>
                        <dd><code>{conversionDraft.hfRepo || "—"}</code></dd>
                        <dt>Output path</dt>
                        <dd>
                          <code>{conversionDraft.outputPath || "(default)"}</code>
                          {conversionDraft.outputPath && !conversionDraft.outputPath.startsWith("/") && !conversionDraft.outputPath.startsWith("~") ? (
                            <small className="muted-text"> → resolved under <code>~/Models/</code></small>
                          ) : null}
                        </dd>
                        <dt>Quantize</dt>
                        <dd>{conversionDraft.quantize ? `yes · q${conversionDraft.qBits} g${conversionDraft.qGroupSize}` : "no"}</dd>
                        <dt>Dtype</dt>
                        <dd>{conversionDraft.dtype}</dd>
                      </dl>
                      <p className="muted-text debug-hint">
                        Backend log: <code>~/Library/.../chaosengine-backend-8876.log</code>. Run <code>tail -100 $(ls -t $TMPDIR/chaosengine-backend-*.log | head -1)</code> in Terminal for full stderr.
                      </p>
                    </details>
                  </div>
                ) : lastConversion ? (
                  <div className="callout">
                    <span className="badge success">✓ Conversion complete</span>
                    <h3>{lastConversion.sourceLabel}</h3>
                    <div className="conversion-output-row">
                      <p className="mono-text">{lastConversion.outputPath}</p>
                      <button
                        className="secondary-button icon-button"
                        type="button"
                        title={fileRevealLabel}
                        onClick={() => void handleRevealPath(lastConversion.outputPath)}
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                          <polyline points="15 3 21 3 21 9" />
                          <line x1="10" y1="14" x2="21" y2="3" />
                        </svg>
                      </button>
                    </div>
                    <div className="field-grid detail-grid">
                      <div>
                        <span className="eyebrow">Base repo</span>
                        <p>{lastConversion.hfRepo}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Architecture</span>
                        <p>{lastConversion.architecture ?? "Unknown"}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Context</span>
                        <p>{lastConversion.contextWindow ?? "Varies"}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Compression</span>
                        <p>{lastConversion.compressionRatio ? `${number(lastConversion.compressionRatio)}x cache reduction` : "—"}</p>
                      </div>
                      <div>
                        <span className="eyebrow">Quality</span>
                        <p>{number(lastConversion.qualityPercent ?? 0, 1)}%</p>
                      </div>
                      <div>
                        <span className="eyebrow">Disk before → after</span>
                        <p>
                          {lastConversion.sourceSizeGb ? sizeLabel(lastConversion.sourceSizeGb) : "—"}
                          {" → "}
                          {lastConversion.outputSizeGb ? sizeLabel(lastConversion.outputSizeGb) : "—"}
                        </p>
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
              {busyAction !== "Converting model..." ? (
                <div className="modal-footer">
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => setShowConversionModal(false)}
                  >
                    {conversionError ? "Close" : "OK"}
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
    );
  }

  function renderChat() {
    const pinnedThreads = sortedChatSessions.filter((session) => session.pinned);
    const recentThreads = sortedChatSessions.filter((session) => !session.pinned);

    return (
      <div className="chat-layout-2col">
        <Panel
          title="Chats"
          subtitle=""
          className="chat-column"
          actions={
            <button className="secondary-button" type="button" onClick={() => void handleCreateSession()}>
              New thread
            </button>
          }
        >
          <div className="thread-list-panel">
            <div className="session-list">
              {sortedChatSessions.map((session) => (
                <div className="session-row" key={session.id}>
                  <button
                    className={session.id === activeChat?.id ? "session-button active" : "session-button"}
                    type="button"
                    onClick={() => setActiveChatId(session.id)}
                  >
                    <div className="session-title-row">
                      <strong>{session.title}</strong>
                      <span
                        className={`pin-icon${session.pinned ? " pinned" : ""}`}
                        role="button"
                        tabIndex={0}
                        title={session.pinned ? "Unpin" : "Pin"}
                        onClick={(e) => { e.stopPropagation(); void handleToggleThreadPin(session); }}
                        onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); void handleToggleThreadPin(session); } }}
                      >
                        {"\uD83D\uDCCC"}
                      </span>
                    </div>
                    <div className="session-meta-row">
                      <small>{session.updatedAt}</small>
                      {session.modelRef && workspace.runtime.warmModels?.some((w) => w.ref === session.modelRef) ? (
                        <span
                          className="badge success session-warm-badge"
                          title="Model is already loaded — this chat will respond instantly with no reload time."
                        >
                          {"\u26A1"} ready
                        </span>
                      ) : null}
                    </div>
                  </button>
                </div>
              ))}
            </div>
          </div>
        </Panel>

        <Panel title="Active Thread" subtitle="Response metadata is collapsed by default, but available per agent turn." className="chat-thread">
          <div className="thread-toolbar">
            <label className="thread-title-field">
              Thread name
              <input
                className="text-input"
                type="text"
                value={threadTitleDraft}
                onChange={(event) => setThreadTitleDraft(event.target.value)}
                onBlur={() => void handleRenameActiveThread()}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void handleRenameActiveThread();
                  }
                }}
              />
            </label>
            <div className="thread-toolbar-actions">
              <button className="secondary-button" type="button" onClick={() => openModelSelector("chat", activeThreadOption?.key)}>
                {activeChat?.model ?? "Select Model"}
              </button>
              {activeChat?.modelRef === workspace.runtime.loadedModel?.ref ? (
                <span className="badge success">Ready</span>
              ) : workspace.server.loading ? (
                <div className="badge accent chat-loading-pill">
                  <span className="busy-dot" />
                  Loading {workspace.server.loading.modelName}... {workspace.server.loading.elapsedSeconds}s
                </div>
              ) : busyAction === "Loading model..." ? (
                <div className="badge accent chat-loading-pill">
                  <span className="busy-dot" />
                  Loading model...
                </div>
              ) : activeChat?.modelRef ? (
                <button
                  className="primary-button action-convert"
                  type="button"
                  disabled={busy}
                  title="Load this chat's model"
                  onClick={() => {
                    if (!activeChat?.modelRef) return;
                    void handleLoadModel({
                      modelRef: activeChat.modelRef,
                      modelName: activeChat.model,
                      source: activeChat.modelSource ?? "library",
                      backend: activeChat.modelBackend ?? "auto",
                      path: activeChat.modelPath ?? undefined,
                    });
                  }}
                >
                  {busy ? "Loading..." : "Load model"}
                </button>
              ) : null}
            </div>
            {activeChat?.documents && activeChat.documents.length > 0 ? (
              <div className="session-documents">
                {activeChat.documents.map((doc) => (
                  <span key={doc.id} className="session-document-chip" title={`${doc.chunkCount} chunks · ${(doc.sizeBytes / 1024).toFixed(0)} KB`}>
                    {"\uD83D\uDCC4"} {doc.originalName}
                    <button
                      type="button"
                      className="session-document-remove"
                      onClick={async () => {
                        if (!activeChat) return;
                        try {
                          await deleteSessionDocument(activeChat.id, doc.id);
                          await refreshWorkspace(activeChat.id);
                        } catch (err) {
                          setError(err instanceof Error ? err.message : "Delete failed");
                        }
                      }}
                    >
                      &times;
                    </button>
                  </span>
                ))}
              </div>
            ) : null}
          </div>

          <div
            className="message-list message-scroll"
            ref={chatScrollRef}
            onDragOver={(event) => {
              event.preventDefault();
              event.currentTarget.classList.add("drag-over");
            }}
            onDragLeave={(event) => {
              event.currentTarget.classList.remove("drag-over");
            }}
            onDrop={(event) => {
              event.preventDefault();
              event.currentTarget.classList.remove("drag-over");
              if (event.dataTransfer?.files) {
                void handleChatFileDrop(event.dataTransfer.files);
              }
            }}
          >
            {activeChat?.messages.length ? (
              activeChat.messages.map((message, index) => {
                const isStreamingMessage = chatBusySessionId === activeChat?.id && index === activeChat.messages.length - 1 && !message.metrics;
                return (
                <div className={`message-bubble ${message.role}`} key={`${message.role}-${index}`}>
                  <div className="message-header">
                    <span className="eyebrow">{message.role === "assistant" ? "Agent" : "User"}</span>
                    {!isStreamingMessage ? (
                      <div className="message-actions">
                        <button
                          type="button"
                          className="message-action-btn"
                          title="Copy message"
                          onClick={() => handleCopyMessage(message.text)}
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                          </svg>
                        </button>
                        {message.role === "assistant" ? (
                          <button
                            type="button"
                            className="message-action-btn"
                            title="Retry response"
                            onClick={() => void handleRetryMessage(index)}
                          >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <polyline points="23 4 23 10 17 10" />
                              <polyline points="1 20 1 14 7 14" />
                              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                            </svg>
                          </button>
                        ) : null}
                        <button
                          type="button"
                          className="message-action-btn message-action-delete"
                          title="Delete message"
                          onClick={() => handleDeleteMessage(index)}
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                            <line x1="10" y1="11" x2="10" y2="17" />
                            <line x1="14" y1="11" x2="14" y2="17" />
                          </svg>
                        </button>
                      </div>
                    ) : null}
                  </div>
                  {message.role === "assistant" ? (
                    <div className={`markdown-content${isStreamingMessage ? " streaming-cursor" : ""}`}>
                      <Markdown>{message.text || "\u200B"}</Markdown>
                    </div>
                  ) : (
                    <p>{message.text}</p>
                  )}
                  {message.metrics ? (
                    <details className="message-details" onToggle={(event) => void handleDetailsToggle(event.currentTarget.open)}>
                      <summary>
                        <span>Model details</span>
                        <small className="message-meta">
                          {(message.metrics.model ?? activeChat.model) || "Unknown"} | {number(message.metrics.tokS)} tok/s | {number(message.metrics.responseSeconds ?? 0)} s
                        </small>
                      </summary>
                      <div className="message-detail-grid">
                        <div>
                          <span className="eyebrow">Model</span>
                          <p>{message.metrics.model ?? activeChat.model}</p>
                        </div>
                        <div>
                          <span className="eyebrow">Runtime</span>
                          <p>{message.metrics.engineLabel ?? workspace.runtime.engineLabel}</p>
                        </div>
                        <div>
                          <span className="eyebrow">Cache</span>
                          <p>{message.metrics.cacheLabel ?? activeChat.cacheLabel}</p>
                        </div>
                        <div>
                          <span className="eyebrow">Backend</span>
                          <p>{message.metrics.backend ?? activeChat.modelBackend ?? "Auto"}</p>
                        </div>
                        <div>
                          <span className="eyebrow">Tokens</span>
                          <p>{message.metrics.totalTokens} total</p>
                        </div>
                        <div>
                          <span className="eyebrow">Response time</span>
                          <p>{number(message.metrics.responseSeconds ?? 0)} s</p>
                        </div>
                        <div>
                          <span className="eyebrow">Decode speed</span>
                          <p>{number(message.metrics.tokS)} tok/s</p>
                        </div>
                        <div>
                          <span className="eyebrow">Context</span>
                          <p>{message.metrics.contextTokens?.toLocaleString() ?? launchSettings.contextTokens.toLocaleString()}</p>
                        </div>
                      </div>
                    </details>
                  ) : null}
                </div>
                );
              })
            ) : (
              <div className="empty-state">
                <p>Send a message to start the conversation.</p>
              </div>
            )}
            {chatBusySessionId === activeChat?.id && workspace.server.loading ? (
              <div className="message-bubble assistant">
                <span className="eyebrow">Agent</span>
                <div className="model-loading-chat">
                  {renderModelLoadingProgress(workspace.server.loading)}
                </div>
              </div>
            ) : null}
          </div>
          <div className="composer">
            {pendingImages.length > 0 ? (
              <div className="composer-image-previews">
                {pendingImages.map((img, i) => (
                  <div key={i} className="composer-image-thumb">
                    <img src={`data:image/png;base64,${img}`} alt={`Attachment ${i + 1}`} />
                    <button
                      className="composer-image-remove"
                      type="button"
                      onClick={() => setPendingImages((prev) => prev.filter((_, j) => j !== i))}
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
            ) : null}
            <textarea
              className="text-area"
              placeholder="Type a message... (Enter to send, Shift+Enter for new line)"
              rows={3}
              value={draftMessage}
              onChange={(event) => setDraftMessage(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
              onDrop={(event) => {
                const files = event.dataTransfer?.files;
                if (!files?.length) return;
                event.preventDefault();
                void handleChatFileDrop(files);
              }}
              onDragOver={(event) => event.preventDefault()}
            />
            <div className="button-row">
              <label className="secondary-button composer-attach-btn" title="Attach image">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  hidden
                  onChange={(event) => {
                    const files = event.target.files;
                    if (!files) return;
                    for (const file of Array.from(files)) {
                      if (file.size > 10 * 1024 * 1024) { setError("Image must be under 10MB"); continue; }
                      const reader = new FileReader();
                      reader.onload = () => {
                        const b64 = (reader.result as string).split(",")[1];
                        if (b64) setPendingImages((prev) => [...prev, b64]);
                      };
                      reader.readAsDataURL(file);
                    }
                    event.target.value = "";
                  }}
                />
                {"\uD83D\uDCCE"}
              </label>
              <button className="primary-button" type="button" onClick={() => void sendMessage()} disabled={chatBusySessionId !== null}>
                Send
              </button>
              <button className="secondary-button" type="button" onClick={() => { setDraftMessage(""); setPendingImages([]); }}>
                Clear
              </button>
            </div>
          </div>
        </Panel>

      </div>
    );
  }

  function copyText(text: string) {
    void navigator.clipboard.writeText(text);
  }

  function renderServer() {
    return (
      <div className="content-grid">
        <Panel
          title="Server"
          subtitle="OpenAI-compatible local API"
          className="span-2"
        >
          <div className="server-layout">
            <div className="server-main-col">
              <div className="server-status-strip">
                <div className="server-status-copy">
                  <div className="server-status-heading">
                    <span className={`badge ${workspace.server.status === "running" ? "success" : "warning"}`}>
                      {workspace.server.status.toUpperCase()}
                    </span>
                    <h3>{localServerUrl}</h3>
                  </div>
                  {remoteAccessActive && primaryLanUrl && (
                    <p className="mono-text muted-text">{primaryLanUrl}</p>
                  )}
                  {busyAction ? (
                    <p className="busy-indicator"><span className="busy-dot" />{busyAction}</p>
                  ) : null}
                </div>
                <div className="button-row server-actions">
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => openModelSelector("server", selectedServerOption?.key)}
                    disabled={busy || !backendOnline}
                  >
                    Load Model
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => void handleRestartServer()}
                    disabled={busy || !backendOnline}
                  >
                    Restart
                  </button>
                  <button
                    className="secondary-button danger-button"
                    type="button"
                    onClick={() => void handleStopServer()}
                    disabled={busy || !backendOnline}
                  >
                    Stop
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => { setTestModelId(null); setShowRemoteTest(true); }}
                  >
                    Test
                  </button>
                </div>
              </div>

              {(() => {
                const warmModels = workspace.runtime.warmModels ?? [];
                const loadingRef = workspace.server.loading?.modelRef ?? null;
                const loadingName = workspace.server.loading?.modelName ?? null;
                if (warmModels.length === 0) {
                  return null;
                }
                return (
                  <div className="warm-pool-list">
                    {warmModels.map((w) => {
                      const isLoading = loadingRef === w.ref;
                      const badgeClass = w.active ? "success" : isLoading ? "accent" : "muted";
                      const badgeLabel = w.active ? "ACTIVE" : isLoading ? "LOADING" : "WARM";
                      const endpoint = `${localServerUrl}  ${w.ref}`;
                      return (
                        <div key={w.ref} className={`warm-pool-row${w.active ? " active" : ""}${isLoading ? " loading" : ""}`}>
                          <div className="row-meta">
                            <div className="row-meta-head">
                              <span className={`badge ${badgeClass}`}>{badgeLabel}</span>
                              <h4>{w.name}</h4>
                              <small className="row-engine">{w.engine}</small>
                            </div>
                            <div className="row-endpoint">
                              <p className="mono-text">{localServerUrl}</p>
                              <p className="mono-text muted-text">model id: {w.ref}</p>
                              <button
                                className="secondary-button"
                                type="button"
                                onClick={() => copyText(endpoint)}
                              >
                                Copy
                              </button>
                            </div>
                            {isLoading && workspace.server.loading ? (
                              renderModelLoadingProgress(workspace.server.loading)
                            ) : null}
                          </div>
                          <div className="row-actions button-row">
                            <button
                              className="primary-button"
                              type="button"
                              disabled={w.active || busy || !backendOnline}
                              onClick={() => void handleLoadModel({ modelRef: w.ref, modelName: w.name, source: "warm-pool" })}
                            >
                              Activate
                            </button>
                            <button
                              className="secondary-button"
                              type="button"
                              disabled={busy || !backendOnline}
                              onClick={() => void handleUnloadWarmModel(w.ref)}
                            >
                              Unload
                            </button>
                            <button
                              className="secondary-button"
                              type="button"
                              onClick={() => { setTestModelId(w.ref); setShowRemoteTest(true); }}
                            >
                              Test
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                );
              })()}

              <div className="stat-grid server-stat-grid">
                <StatCard label="Port" value={String(workspace.server.port)} hint={preferredPortUnavailable ? "Preferred port is busy" : "Active"} />
                <StatCard
                  label="Active"
                  value={(workspace.runtime.warmModels ?? []).find((m) => m.active)?.name ?? "None"}
                  hint={workspace.runtime.engineLabel}
                />
                <StatCard
                  label="Warm pool"
                  value={String((workspace.runtime.warmModels ?? []).length)}
                  hint={`${(workspace.runtime.warmModels ?? []).filter((m) => m.warm).length} warm`}
                />
                <StatCard
                  label="LAN"
                  value={remoteAccessActive ? "Enabled" : "Local only"}
                  hint={
                    remoteAccessRequested && !remoteAccessActive
                      ? "Restart to enable LAN"
                      : remoteAccessActive
                        ? primaryLanOrigin ?? "0.0.0.0"
                        : "Localhost"
                  }
                />
                <StatCard label="Requests" value={String(workspace.server.requestsServed)} hint={`${workspace.server.activeConnections} active`} />
              </div>

              <div className="server-compact-settings">
                <div className="server-settings-row">
                  <label>
                    Port
                    <input
                      className="text-input"
                      type="number"
                      min="1024"
                      max="65535"
                      disabled={busy}
                      value={settingsDraft.preferredServerPort}
                      onChange={(event) => setSettingsDraft((c) => ({ ...c, preferredServerPort: Number(event.target.value) }))}
                    />
                  </label>
                  <label className="check-row">
                    <input
                      type="checkbox"
                      checked={settingsDraft.allowRemoteConnections}
                      disabled={busy}
                      onChange={(event) => setSettingsDraft((c) => ({ ...c, allowRemoteConnections: event.target.checked }))}
                    />
                    LAN access
                  </label>
                  <label className="check-row">
                    <input
                      type="checkbox"
                      checked={settingsDraft.autoStartServer}
                      disabled={busy}
                      onChange={(event) => setSettingsDraft((c) => ({ ...c, autoStartServer: event.target.checked }))}
                    />
                    Auto-start
                  </label>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => void handleSaveSettings()}
                    disabled={busy || !backendOnline}
                  >
                    Save
                  </button>
                </div>
              </div>

              <div className="server-log-panel">
                <span className="eyebrow">Server Log</span>
                <div className="server-log-scroll" ref={serverLogRef} onScroll={handleServerLogScroll}>
                  {serverLogEntries.length > 0 ? (
                    serverLogEntries.map((entry, i) => (
                      <div className="server-log-line" key={`${entry.ts}-${i}`}>
                        <small className="server-log-ts">{entry.ts}</small>
                        <span className={`log-level ${entry.level}`}>{entry.level}</span>
                        <span>{entry.message}</span>
                      </div>
                    ))
                  ) : (
                    <div className="server-log-line">
                      <span className="server-log-placeholder">Waiting for log events...</span>
                    </div>
                  )}
                </div>
                {!serverLogAtBottom && serverLogEntries.length > 0 ? (
                  <button
                    className="server-log-jump"
                    type="button"
                    onClick={scrollServerLogToBottom}
                  >
                    Latest
                  </button>
                ) : null}
              </div>
            </div>
          </div>
        </Panel>

        {showRemoteTest ? (
          <div className="modal-overlay" onClick={() => setShowRemoteTest(false)}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>API Test Commands</h3>
                <p>Copy these commands to test the server from a terminal.{testModelId ? ` Pre-filled for ${testModelId}.` : ""}</p>
              </div>
              <div className="modal-body">
                <div className="server-command-list">
                  <div className="server-command">
                    <div className="server-command-header">
                      <strong>Health check</strong>
                      <button className="secondary-button" type="button" onClick={() => copyText(localHealthCurl)}>Copy</button>
                    </div>
                    <p className="mono-text">{localHealthCurl}</p>
                  </div>
                  <div className="server-command">
                    <div className="server-command-header">
                      <strong>List models</strong>
                      <button className="secondary-button" type="button" onClick={() => copyText(localModelsCurl)}>Copy</button>
                    </div>
                    <p className="mono-text">{localModelsCurl}</p>
                  </div>
                  {testModelId ? (() => {
                    const cmd = `curl -sS ${localServerUrl}/chat/completions -H 'Content-Type: application/json' -d '{"model":"${testModelId}","messages":[{"role":"user","content":"Hello"}]}'`;
                    return (
                      <div className="server-command">
                        <div className="server-command-header">
                          <strong>Chat completion ({testModelId})</strong>
                          <button className="secondary-button" type="button" onClick={() => copyText(cmd)}>Copy</button>
                        </div>
                        <p className="mono-text">{cmd}</p>
                      </div>
                    );
                  })() : null}
                  {remoteAccessActive && remoteHealthCurl ? (
                    <>
                      <div className="server-command">
                        <div className="server-command-header">
                          <strong>Remote health</strong>
                          <button className="secondary-button" type="button" onClick={() => copyText(remoteHealthCurl)}>Copy</button>
                        </div>
                        <p className="mono-text">{remoteHealthCurl}</p>
                      </div>
                      {remoteModelsCurl ? (
                        <div className="server-command">
                          <div className="server-command-header">
                            <strong>Remote models</strong>
                            <button className="secondary-button" type="button" onClick={() => copyText(remoteModelsCurl)}>Copy</button>
                          </div>
                          <p className="mono-text">{remoteModelsCurl}</p>
                        </div>
                      ) : null}
                    </>
                  ) : null}
                </div>
              </div>
              <div className="modal-footer">
                <button className="secondary-button" type="button" onClick={() => setShowRemoteTest(false)}>Close</button>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    );
  }

  function renderBenchmarkGauge(value: number, max: number, label: string, subtitle?: string) {
    // SVG circular gauge: 180x180, ring with stroke-dasharray
    const size = 180;
    const strokeWidth = 14;
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const pct = Math.min(1, Math.max(0, value / max));
    const arcLength = circumference * 0.75; // 270° arc
    const dashOffset = arcLength * (1 - pct);
    const color = pct > 0.5 ? "#8fcf9f" : pct > 0.25 ? "var(--accent)" : "#e4be75";

    return (
      <div className="benchmark-gauge">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          <g transform={`rotate(135 ${size / 2} ${size / 2})`}>
            {/* Track */}
            <circle
              cx={size / 2} cy={size / 2} r={radius}
              fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={strokeWidth}
              strokeDasharray={`${arcLength} ${circumference}`}
              strokeLinecap="round"
            />
            {/* Fill */}
            <circle
              cx={size / 2} cy={size / 2} r={radius}
              fill="none" stroke={color} strokeWidth={strokeWidth}
              strokeDasharray={`${arcLength} ${circumference}`}
              strokeDashoffset={dashOffset}
              strokeLinecap="round"
              style={{ transition: "stroke-dashoffset 0.6s ease" }}
            />
          </g>
          <text x={size / 2} y={size / 2 - 4} textAnchor="middle" className="benchmark-gauge-value-text">
            {value.toFixed(1)}
          </text>
          <text x={size / 2} y={size / 2 + 22} textAnchor="middle" className="benchmark-gauge-label-text">
            {label}
          </text>
          {subtitle ? (
            <text x={size / 2} y={size / 2 + 44} textAnchor="middle" className="benchmark-gauge-subtitle-text">
              {subtitle}
            </text>
          ) : null}
        </svg>
      </div>
    );
  }

  function renderBenchmarkScatter(runs: BenchmarkResult[], selectedId: string | null, compareId: string | null, onSelect: (id: string) => void) {
    if (runs.length === 0) {
      return <div className="empty-state"><p>Run some benchmarks to populate the scatter plot.</p></div>;
    }
    const padding = { top: 20, right: 20, bottom: 44, left: 50 };
    const width = 560;
    const height = 320;
    const innerW = width - padding.left - padding.right;
    const innerH = height - padding.top - padding.bottom;

    const maxCache = Math.max(1, ...runs.map((r) => r.cacheGb));
    const maxTokS = Math.max(1, ...runs.map((r) => r.tokS));
    const maxQuality = 100;

    const xScale = (v: number) => (v / maxCache) * innerW;
    const yScale = (v: number) => innerH - (v / maxTokS) * innerH;
    const rScale = (q: number) => 4 + (q / maxQuality) * 6;

    const bitColor = (bits: number, strategy: string) => {
      if (strategy === "native" || bits >= 16) return "#8fb4ff";
      if (bits === 1) return "#f87171";
      if (bits === 2) return "#fb923c";
      if (bits === 3) return "#facc15";
      return "#8fcf9f";
    };

    const gridYTicks = [0, 0.25, 0.5, 0.75, 1.0];

    return (
      <div className="benchmark-scatter-wrap">
        <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="benchmark-scatter">
          {/* Grid lines */}
          {gridYTicks.map((t) => (
            <g key={t}>
              <line
                x1={padding.left} x2={padding.left + innerW}
                y1={padding.top + innerH * (1 - t)} y2={padding.top + innerH * (1 - t)}
                stroke="var(--border)" strokeWidth="1" strokeDasharray="2 4"
              />
              <text
                x={padding.left - 8} y={padding.top + innerH * (1 - t) + 4}
                textAnchor="end" className="scatter-axis-label"
              >
                {(maxTokS * t).toFixed(0)}
              </text>
            </g>
          ))}
          {gridYTicks.map((t) => (
            <text
              key={`x-${t}`}
              x={padding.left + innerW * t} y={padding.top + innerH + 18}
              textAnchor="middle" className="scatter-axis-label"
            >
              {(maxCache * t).toFixed(1)}
            </text>
          ))}
          {/* Axis labels */}
          <text x={padding.left + innerW / 2} y={height - 6} textAnchor="middle" className="scatter-axis-title">
            Cache (GB) →
          </text>
          <text
            x={14} y={padding.top + innerH / 2}
            textAnchor="middle" className="scatter-axis-title"
            transform={`rotate(-90 14 ${padding.top + innerH / 2})`}
          >
            ↑ Tokens/sec
          </text>
          {/* Points */}
          {runs.map((run) => {
            const cx = padding.left + xScale(run.cacheGb);
            const cy = padding.top + yScale(run.tokS);
            const r = rScale(run.quality);
            const isSelected = run.id === selectedId;
            const isCompare = run.id === compareId;
            return (
              <g key={run.id}>
                {isSelected ? (
                  <circle cx={cx} cy={cy} r={r + 5} fill="none" stroke="var(--accent)" strokeWidth="2" />
                ) : null}
                {isCompare ? (
                  <circle cx={cx} cy={cy} r={r + 5} fill="none" stroke="#8fcf9f" strokeWidth="2" strokeDasharray="3 3" />
                ) : null}
                <circle
                  cx={cx} cy={cy} r={r}
                  fill={bitColor(run.bits, run.cacheStrategy)}
                  opacity="0.8"
                  onClick={() => onSelect(run.id)}
                  style={{ cursor: "pointer" }}
                >
                  <title>{`${run.label}\n${run.tokS.toFixed(1)} tok/s · ${run.cacheGb.toFixed(1)} GB · ${run.quality}% quality`}</title>
                </circle>
              </g>
            );
          })}
        </svg>
        <div className="scatter-legend">
          <span className="scatter-legend-item"><span className="scatter-dot" style={{ background: "#f87171" }} />1-bit</span>
          <span className="scatter-legend-item"><span className="scatter-dot" style={{ background: "#fb923c" }} />2-bit</span>
          <span className="scatter-legend-item"><span className="scatter-dot" style={{ background: "#facc15" }} />3-bit</span>
          <span className="scatter-legend-item"><span className="scatter-dot" style={{ background: "#8fcf9f" }} />4-bit</span>
          <span className="scatter-legend-item"><span className="scatter-dot" style={{ background: "#8fb4ff" }} />Native f16</span>
          <span className="scatter-legend-item scatter-legend-spacer">Dot size = quality %</span>
        </div>
      </div>
    );
  }

  function renderBenchmarkRun() {
    const latestRun = workspace.benchmarks[0] ?? null;
    const fastestRun = [...workspace.benchmarks].sort((left, right) => right.tokS - left.tokS)[0] ?? null;
    const selectedPrompt = BENCHMARK_PROMPTS.find((p) => p.id === benchmarkPromptId) ?? BENCHMARK_PROMPTS[0];
    // Find the most recent run for the currently selected model, if any
    const prevForModel = benchmarkOption
      ? workspace.benchmarks.find((b) => b.modelRef === benchmarkOption.modelRef && b.id !== latestRun?.id) ?? null
      : null;
    const speedDeltaVsPrev = latestRun && prevForModel ? latestRun.tokS - prevForModel.tokS : null;

    return (
      <div className="content-grid">
        <Panel
          title="Run Benchmark"
          subtitle="Launch a consistent benchmark run and see how this profile performs."
          className="span-2 benchmark-run-page-panel"
        >
          <div className="benchmark-run-page">
            <div className="benchmark-run-config scrollable-panel-content">
              <div className="field-grid">
                <label>
                  Benchmark model
                  <div className="model-selected-card">
                    <div className="model-selected-info">
                      <strong>{benchmarkOption?.label ?? benchmarkDraft.modelName ?? "Select a model"}</strong>
                      <div className="model-selected-meta">
                        {benchmarkOption?.format ? <span className="badge muted">{benchmarkOption.format}</span> : null}
                        {benchmarkOption?.sizeGb ? <span className="badge muted">{sizeLabel(benchmarkOption.sizeGb)}</span> : null}
                      </div>
                    </div>
                    <button className="secondary-button" type="button" onClick={() => setShowBenchmarkPicker(true)}>
                      Change
                    </button>
                  </div>
                </label>
                <label>
                  Benchmark mode
                  <select
                    className="text-input"
                    value={benchmarkDraft.mode ?? "throughput"}
                    onChange={(event) => updateBenchmarkDraft("mode", event.target.value as any)}
                  >
                    <option value="throughput">Throughput (tok/s)</option>
                    <option value="perplexity">Perplexity (quality)</option>
                    <option value="task_accuracy">Task Accuracy (MMLU / HellaSwag)</option>
                  </select>
                </label>
                {(!benchmarkDraft.mode || benchmarkDraft.mode === "throughput") ? (
                  <label>
                    Prompt preset
                    <select
                      className="text-input"
                      value={benchmarkPromptId}
                      onChange={(event) => setBenchmarkPromptId(event.target.value)}
                    >
                      {BENCHMARK_PROMPTS.map((preset) => (
                        <option key={preset.id} value={preset.id}>
                          {preset.label}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                {benchmarkDraft.mode === "perplexity" ? (
                  <>
                    <label>
                      Dataset
                      <select
                        className="text-input"
                        value={benchmarkDraft.perplexityDataset ?? "wikitext-2"}
                        onChange={(event) => updateBenchmarkDraft("perplexityDataset", event.target.value as any)}
                      >
                        <option value="wikitext-2">WikiText-2</option>
                      </select>
                    </label>
                    <label>
                      Samples
                      <input
                        className="text-input"
                        type="number"
                        min="8"
                        max="1024"
                        step="8"
                        value={benchmarkDraft.perplexityNumSamples ?? 64}
                        onChange={(event) => updateBenchmarkDraft("perplexityNumSamples", Number(event.target.value) as any)}
                      />
                    </label>
                  </>
                ) : null}
                {benchmarkDraft.mode === "task_accuracy" ? (
                  <>
                    <label>
                      Task
                      <select
                        className="text-input"
                        value={benchmarkDraft.taskName ?? "mmlu"}
                        onChange={(event) => updateBenchmarkDraft("taskName", event.target.value as any)}
                      >
                        <option value="mmlu">MMLU (multiple choice)</option>
                        <option value="hellaswag">HellaSwag (sentence completion)</option>
                      </select>
                    </label>
                    <label>
                      Questions
                      <input
                        className="text-input"
                        type="number"
                        min="10"
                        max="5000"
                        step="10"
                        value={benchmarkDraft.taskLimit ?? 100}
                        onChange={(event) => updateBenchmarkDraft("taskLimit", Number(event.target.value) as any)}
                      />
                    </label>
                  </>
                ) : null}
              </div>

              {selectedPrompt && (!benchmarkDraft.mode || benchmarkDraft.mode === "throughput") ? (
                <div className="callout quiet benchmark-prompt-preview">
                  <span className="eyebrow">Prompt</span>
                  <p>{selectedPrompt.prompt ?? selectedPrompt.label}</p>
                </div>
              ) : null}
              {benchmarkDraft.mode === "perplexity" ? (
                <div className="callout quiet benchmark-prompt-preview">
                  <span className="eyebrow">Perplexity</span>
                  <p>Measures how well the model predicts text. Lower is better. Compares real quality loss across quantization levels.</p>
                </div>
              ) : null}
              {benchmarkDraft.mode === "task_accuracy" ? (
                <div className="callout quiet benchmark-prompt-preview">
                  <span className="eyebrow">Task Accuracy</span>
                  <p>Runs multiple-choice questions and scores the model's answers. Higher accuracy is better.</p>
                </div>
              ) : null}

              <RuntimeControls
                settings={benchmarkDraft}
                onChange={updateBenchmarkDraft}
                maxContext={benchmarkOption?.maxContext}
                diskSizeGb={benchmarkOption?.sizeGb}
                preview={preview}
                availableMemoryGb={workspace.system.availableMemoryGb}
                totalMemoryGb={workspace.system.totalMemoryGb}
                availableCacheStrategies={workspace.system.availableCacheStrategies}
                showTemperature={false}
                showPreview={false}
              />

              <div className="button-row">
                <button className="primary-button benchmark-run-btn" type="button" onClick={() => void handleRunBenchmark()} disabled={busy}>
                  {busy ? "Running..." : "▶ Run benchmark"}
                </button>
              </div>

              <div className="callout quiet">
                <h3>Benchmarking approach</h3>
                <p>
                  ChaosEngineAI loads the chosen runtime profile if needed, runs a consistent prompt, captures decode speed and response time, then stores the result so you can compare later runs side by side.
                </p>
              </div>
            </div>

            <div className="benchmark-run-preview scrollable-panel-content">
              <PerformancePreview
                preview={preview}
                availableMemoryGb={workspace.system.availableMemoryGb}
                totalMemoryGb={workspace.system.totalMemoryGb}
              />

              {latestRun ? (
                <div className="benchmark-last-run-card">
                  <div className="benchmark-last-run-header">
                    <span className="eyebrow">Latest run</span>
                    <span className="badge muted">{latestRun.measuredAt}</span>
                  </div>
                  <h3>{latestRun.model}</h3>
                  <p className="muted-text">{latestRun.cacheLabel} · {latestRun.engineLabel}</p>

                  {latestRun.mode === "perplexity" ? (
                    renderBenchmarkGauge(latestRun.perplexity ?? 0, 50, "perplexity", "lower is better")
                  ) : latestRun.mode === "task_accuracy" ? (
                    renderBenchmarkGauge((latestRun.taskAccuracy ?? 0) * 100, 100, "% accuracy")
                  ) : (
                    renderBenchmarkGauge(latestRun.tokS, 40, "tok/s")
                  )}

                  <div className="stat-grid compact-grid benchmark-last-run-stats">
                    {latestRun.mode === "perplexity" ? (
                      <>
                        <StatCard label="Perplexity" value={`${number(latestRun.perplexity ?? 0)}`} hint={`± ${number(latestRun.perplexityStdError ?? 0)} SE`} />
                        <StatCard label="Eval speed" value={`${number(latestRun.evalTokensPerSecond ?? 0)} tok/s`} hint={`${number(latestRun.evalSeconds ?? 0)} s total`} />
                        <StatCard label="Dataset" value={latestRun.perplexityDataset ?? "wikitext-2"} hint={`${latestRun.perplexityNumSamples ?? 0} samples`} />
                        <StatCard label="Cache" value={`${number(latestRun.cacheGb)} GB`} hint={`${number(latestRun.compression)}x compression`} />
                      </>
                    ) : latestRun.mode === "task_accuracy" ? (
                      <>
                        <StatCard label="Accuracy" value={`${((latestRun.taskAccuracy ?? 0) * 100).toFixed(1)}%`} hint={`${latestRun.taskCorrect}/${latestRun.taskTotal} correct`} />
                        <StatCard label="Task" value={(latestRun.taskName ?? "mmlu").toUpperCase()} hint={`${latestRun.taskNumShots ?? 5}-shot`} />
                        <StatCard label="Eval time" value={`${number(latestRun.evalSeconds ?? 0)} s`} hint={`${number(latestRun.loadSeconds)} s load`} />
                        <StatCard label="Cache" value={`${number(latestRun.cacheGb)} GB`} hint={`${number(latestRun.compression)}x compression`} />
                      </>
                    ) : (
                      <>
                        <StatCard label="Response time" value={`${number(latestRun.responseSeconds)} s`} hint={`${number(latestRun.loadSeconds)} s load`} />
                        <StatCard label="Cache footprint" value={`${number(latestRun.cacheGb)} GB`} hint={`${number(latestRun.compression)}x compression`} />
                        <StatCard label="Quality" value={`${latestRun.quality}%`} hint={`${latestRun.completionTokens} tokens generated`} />
                        <StatCard label="Context" value={`${latestRun.contextTokens.toLocaleString()}`} hint={`${latestRun.maxTokens} max`} />
                      </>
                    )}
                  </div>

                  {speedDeltaVsPrev !== null && prevForModel ? (
                    <div className="callout quiet benchmark-delta-note">
                      <p>
                        {signedDelta(speedDeltaVsPrev)} tok/s vs your previous {prevForModel.cacheLabel} run
                        {fastestRun && fastestRun.id !== latestRun.id ? ` · ${signedDelta(latestRun.tokS - fastestRun.tokS)} tok/s vs fastest (${fastestRun.cacheLabel})` : ""}
                      </p>
                    </div>
                  ) : null}

                  <div className="button-row">
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => {
                        setSelectedBenchmarkId(latestRun.id);
                        if (prevForModel) setCompareBenchmarkId(prevForModel.id);
                        setActiveTab("benchmark-history");
                      }}
                    >
                      View in History
                    </button>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <p>No benchmark runs yet. Configure a profile on the left and click Run benchmark.</p>
                </div>
              )}
            </div>
          </div>
        </Panel>

        {showBenchmarkModal ? (
          <div className="modal-overlay benchmark-result-modal">
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>
                  {busyAction === "Running benchmark..."
                    ? "Running benchmark"
                    : benchmarkError
                      ? "Benchmark failed"
                      : "Benchmark complete"}
                </h3>
              </div>
              <div className="modal-body">
                {busyAction === "Running benchmark..." && benchmarkStartedAt ? (
                  <LiveProgress
                    title="Running benchmark"
                    subtitle={benchmarkOption?.model ?? undefined}
                    startedAt={benchmarkStartedAt}
                    accent="benchmark"
                    phases={[
                      { id: "load", label: "Loading model into memory", estimatedSeconds: 12 },
                      { id: "warm", label: "Warming up KV cache", estimatedSeconds: 4 },
                      { id: "prompt", label: "Processing prompt", estimatedSeconds: 3 },
                      { id: "generate", label: `Generating ${benchmarkDraft.maxTokens} tokens`, estimatedSeconds: Math.max(8, benchmarkDraft.maxTokens / 25) },
                      { id: "measure", label: "Measuring stats", estimatedSeconds: 2 },
                    ] as LiveProgressPhase[]}
                  />
                ) : benchmarkError ? (
                  <div className="callout error">
                    <h3>Benchmark failed</h3>
                    <p>{benchmarkError}</p>
                    <details className="debug-details">
                      <summary>Debug details</summary>
                      <dl className="debug-grid">
                        <dt>Model</dt>
                        <dd><code>{benchmarkDraft.modelRef || "—"}</code></dd>
                        <dt>Source</dt>
                        <dd><code>{benchmarkDraft.source || "—"}</code></dd>
                        <dt>Backend</dt>
                        <dd>{benchmarkDraft.backend || "auto"}</dd>
                        <dt>Path</dt>
                        <dd><code>{benchmarkDraft.path || "—"}</code></dd>
                        <dt>Profile</dt>
                        <dd>{benchmarkDraft.cacheStrategy} {benchmarkDraft.cacheBits}-bit · fp16×{benchmarkDraft.fp16Layers} · ctx {benchmarkDraft.contextTokens} · {benchmarkDraft.maxTokens} tok</dd>
                        <dt>Prompt preset</dt>
                        <dd>{benchmarkPromptId}</dd>
                      </dl>
                      <p className="muted-text debug-hint">
                        Run <code>tail -100 $(ls -t $TMPDIR/chaosengine-backend-*.log | head -1)</code> in Terminal for full stderr.
                      </p>
                    </details>
                  </div>
                ) : latestRun ? (
                  <div className="benchmark-last-run-card">
                    <div className="benchmark-last-run-header">
                      <span className="eyebrow">Latest run</span>
                      <span className="badge muted">{latestRun.measuredAt}</span>
                    </div>
                    <h3>{latestRun.model}</h3>
                    <p className="muted-text">{latestRun.cacheLabel} · {latestRun.engineLabel}</p>

                    {renderBenchmarkGauge(latestRun.tokS, 40, "tok/s")}

                    <div className="stat-grid compact-grid benchmark-last-run-stats">
                      <StatCard
                        label="Response time"
                        value={`${number(latestRun.responseSeconds)} s`}
                        hint={`${number(latestRun.loadSeconds)} s load`}
                      />
                      <StatCard
                        label="Cache footprint"
                        value={`${number(latestRun.cacheGb)} GB`}
                        hint={`${number(latestRun.compression)}x compression`}
                      />
                      <StatCard
                        label="Quality"
                        value={`${latestRun.quality}%`}
                        hint={`${latestRun.completionTokens} tokens generated`}
                      />
                      <StatCard
                        label="Context"
                        value={`${latestRun.contextTokens.toLocaleString()}`}
                        hint={`${latestRun.maxTokens} max`}
                      />
                    </div>

                    {speedDeltaVsPrev !== null && prevForModel ? (
                      <div className="callout quiet benchmark-delta-note">
                        <p>
                          {signedDelta(speedDeltaVsPrev)} tok/s vs your previous {prevForModel.cacheLabel} run
                          {fastestRun && fastestRun.id !== latestRun.id ? ` · ${signedDelta(latestRun.tokS - fastestRun.tokS)} tok/s vs fastest (${fastestRun.cacheLabel})` : ""}
                        </p>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
              {busyAction !== "Running benchmark..." ? (
                <div className="modal-footer">
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => setShowBenchmarkModal(false)}
                  >
                    {benchmarkError ? "Close" : "OK"}
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
        <ModelPicker
          open={showBenchmarkPicker}
          title="Select Benchmark Model"
          library={workspace.library}
          selectedPath={benchmarkDraft.path ?? null}
          onSelect={(item, resolvedPath) => {
            const libraryKey = `library:${item.path}`;
            setBenchmarkModelKey(libraryKey);
            const backend = libraryItemBackend(item);
            setBenchmarkDraft((current) => ({
              ...current,
              modelRef: item.name,
              modelName: item.name,
              source: "library",
              backend,
              path: resolvedPath ?? item.path,
            }));
          }}
          onClose={() => setShowBenchmarkPicker(false)}
        />
      </div>
    );
  }

  function renderBenchmarkHistory() {
    const latestRun = workspace.benchmarks[0] ?? null;
    const fastestRun = [...workspace.benchmarks].sort((left, right) => right.tokS - left.tokS)[0] ?? null;
    const mostEfficientRun = [...workspace.benchmarks].sort(
      (left, right) => left.cacheGb - right.cacheGb || right.quality - left.quality,
    )[0] ?? null;
    const bestQualityRun = [...workspace.benchmarks].sort((a, b) => b.quality - a.quality)[0] ?? null;
    const bestValueRun = [...workspace.benchmarks].sort(
      (a, b) => (b.tokS / Math.max(b.cacheGb, 0.1)) - (a.tokS / Math.max(a.cacheGb, 0.1)),
    )[0] ?? null;

    const pctDelta = (a: number, b: number) => {
      if (b === 0) return 0;
      return ((a - b) / b) * 100;
    };

    return (
      <div className="content-grid">
        <Panel
          title="Benchmark History"
          subtitle="Click any run to inspect; double-click to set as the comparison baseline."
          className="span-2 benchmark-history-page-panel"
          actions={
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                if (!selectedBenchmark || !compareBenchmark) return;
                const a = selectedBenchmark.id;
                const b = compareBenchmark.id;
                setSelectedBenchmarkId(b);
                setCompareBenchmarkId(a);
              }}
              disabled={!selectedBenchmark || !compareBenchmark}
            >
              ⇄ Swap
            </button>
          }
        >
          <div className="benchmark-history-page">
            <div className="benchmark-summary-row">
              <StatCard label="Total runs" value={String(workspace.benchmarks.length)} hint="Persistent history" />
              <StatCard label="Latest" value={latestRun ? `${number(latestRun.tokS)} tok/s` : "None"} hint={latestRun?.cacheLabel ?? "No runs"} />
              <StatCard label="Fastest" value={fastestRun ? `${number(fastestRun.tokS)} tok/s` : "None"} hint={fastestRun?.label ?? "No runs"} />
              <StatCard label="Leanest cache" value={mostEfficientRun ? `${number(mostEfficientRun.cacheGb)} GB` : "None"} hint={mostEfficientRun?.label ?? "No runs"} />
              <StatCard label="Best quality" value={bestQualityRun ? `${bestQualityRun.quality}%` : "None"} hint={bestQualityRun?.cacheLabel ?? "No runs"} />
              <StatCard label="Best value" value={bestValueRun ? `${number(bestValueRun.tokS / Math.max(bestValueRun.cacheGb, 0.1))} tok/s/GB` : "None"} hint={bestValueRun?.cacheLabel ?? "No runs"} />
            </div>

            {selectedBenchmark ? (
              <div className="benchmark-compare-row">
                <div className="benchmark-compare-card benchmark-compare-card--selected">
                  <div className="benchmark-compare-header">
                    <span className="eyebrow">Selected</span>
                    <span className="badge accent">{selectedBenchmark.cacheLabel}</span>
                  </div>
                  <h3>{selectedBenchmark.model}</h3>
                  <div className="benchmark-compare-headline">
                    <strong>{number(selectedBenchmark.tokS)}</strong>
                    <span>tok/s</span>
                  </div>
                  <div className="benchmark-compare-mini-stats">
                    <div><span className="eyebrow">Response</span><p>{number(selectedBenchmark.responseSeconds)} s</p></div>
                    <div><span className="eyebrow">Cache</span><p>{number(selectedBenchmark.cacheGb)} GB</p></div>
                    <div><span className="eyebrow">Quality</span><p>{selectedBenchmark.quality}%</p></div>
                    <div><span className="eyebrow">Compression</span><p>{number(selectedBenchmark.compression)}x</p></div>
                  </div>
                </div>

                <div className="benchmark-compare-divider">
                  {compareBenchmark ? (
                    <>
                      <div className="compare-delta">
                        <span className="eyebrow">Speed</span>
                        <span className={`delta-badge-large ${benchmarkSpeedDelta > 0 ? "delta-badge--positive" : benchmarkSpeedDelta < 0 ? "delta-badge--negative" : ""}`}>
                          {signedDelta(benchmarkSpeedDelta)} tok/s
                        </span>
                        <small>{signedDelta(pctDelta(selectedBenchmark.tokS, compareBenchmark.tokS), 0)}%</small>
                      </div>
                      <div className="compare-delta">
                        <span className="eyebrow">Cache</span>
                        <span className={`delta-badge-large ${benchmarkCacheDelta < 0 ? "delta-badge--positive" : benchmarkCacheDelta > 0 ? "delta-badge--negative" : ""}`}>
                          {signedDelta(benchmarkCacheDelta)} GB
                        </span>
                        <small>{signedDelta(pctDelta(selectedBenchmark.cacheGb, compareBenchmark.cacheGb), 0)}%</small>
                      </div>
                      <div className="compare-delta">
                        <span className="eyebrow">Latency</span>
                        <span className={`delta-badge-large ${benchmarkLatencyDelta < 0 ? "delta-badge--positive" : benchmarkLatencyDelta > 0 ? "delta-badge--negative" : ""}`}>
                          {signedDelta(benchmarkLatencyDelta)} s
                        </span>
                        <small>{signedDelta(pctDelta(selectedBenchmark.responseSeconds, compareBenchmark.responseSeconds), 0)}%</small>
                      </div>
                      <div className="compare-delta">
                        <span className="eyebrow">Quality</span>
                        <span className={`delta-badge-large ${selectedBenchmark.quality - compareBenchmark.quality > 0 ? "delta-badge--positive" : selectedBenchmark.quality - compareBenchmark.quality < 0 ? "delta-badge--negative" : ""}`}>
                          {signedDelta(selectedBenchmark.quality - compareBenchmark.quality)}%
                        </span>
                      </div>
                    </>
                  ) : (
                    <p className="muted-text">Pick a run below<br />to compare.</p>
                  )}
                </div>

                {compareBenchmark ? (
                  <div className="benchmark-compare-card">
                    <div className="benchmark-compare-header">
                      <span className="eyebrow">Compare against</span>
                      <span className="badge muted">{compareBenchmark.cacheLabel}</span>
                    </div>
                    <h3>{compareBenchmark.model}</h3>
                    <div className="benchmark-compare-headline">
                      <strong>{number(compareBenchmark.tokS)}</strong>
                      <span>tok/s</span>
                    </div>
                    <div className="benchmark-compare-mini-stats">
                      <div><span className="eyebrow">Response</span><p>{number(compareBenchmark.responseSeconds)} s</p></div>
                      <div><span className="eyebrow">Cache</span><p>{number(compareBenchmark.cacheGb)} GB</p></div>
                      <div><span className="eyebrow">Quality</span><p>{compareBenchmark.quality}%</p></div>
                      <div><span className="eyebrow">Compression</span><p>{number(compareBenchmark.compression)}x</p></div>
                    </div>
                  </div>
                ) : (
                  <div className="benchmark-compare-card benchmark-compare-card--empty">
                    <p className="muted-text">Double-click any run in the table to set as comparison baseline.</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-state">
                <p>Run a benchmark to populate comparison stats.</p>
              </div>
            )}

            <div className="benchmark-bottom-row">
              <div className="benchmark-history-table-wrap">
                <div className="benchmark-history-table">
                  <div className="table-row table-head benchmark-history-row">
                    <span>Run</span>
                    <span>Mode</span>
                    <span>Result</span>
                    <span>Time</span>
                    <span>Cache</span>
                  </div>
                  <div className="benchmark-history-list">
                    {workspace.benchmarks.map((result) => {
                      const isSelected = result.id === selectedBenchmark?.id;
                      const isCompare = result.id === compareBenchmark?.id;
                      const mode = result.mode ?? "throughput";
                      const resultValue = mode === "perplexity"
                        ? `${number(result.perplexity ?? 0)} ppl`
                        : mode === "task_accuracy"
                        ? `${((result.taskAccuracy ?? 0) * 100).toFixed(1)}%`
                        : `${number(result.tokS)} tok/s`;
                      const modeLabel = mode === "perplexity" ? "PPL" : mode === "task_accuracy" ? (result.taskName ?? "mmlu").toUpperCase() : "Speed";
                      return (
                        <button
                          key={result.id}
                          type="button"
                          className={`table-row table-button-row benchmark-history-row${isSelected ? " active" : ""}${isCompare ? " compare" : ""}`}
                          onClick={() => setSelectedBenchmarkId(result.id)}
                          onDoubleClick={() => setCompareBenchmarkId(result.id)}
                        >
                          <div>
                            <strong>{result.label}</strong>
                            <small>{result.measuredAt} · {result.model}</small>
                          </div>
                          <span className="badge muted">{modeLabel}</span>
                          <span>{resultValue}</span>
                          <span>{number(result.responseSeconds)} s</span>
                          <span>{number(result.cacheGb)} GB</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div className="benchmark-scatter-panel">
                <div className="benchmark-scatter-header">
                  <span className="eyebrow">Cache vs Speed</span>
                  <small>Click a dot to select</small>
                </div>
                {renderBenchmarkScatter(workspace.benchmarks, selectedBenchmark?.id ?? null, compareBenchmark?.id ?? null, (id) => setSelectedBenchmarkId(id))}
              </div>
            </div>
          </div>
        </Panel>
      </div>
    );
  }

  function renderLogs() {
    return (
      <div className="content-grid">
        <Panel
          title="Logs"
          subtitle="Searchable view over the sidecar, runtime, and server channels."
          className="span-2"
          actions={
            <input
              className="text-input"
              type="search"
              placeholder="Filter logs"
              value={logQuery}
              onChange={(event) => setLogQuery(event.target.value)}
            />
          }
        >
          <div className="log-list">
            {filteredLogs.length ? (
              filteredLogs.map((entry) => (
                <div className="log-line" key={`${entry.ts}-${entry.source}-${entry.message}`}>
                  <span className={`log-level ${entry.level}`}>{entry.level}</span>
                  <div>
                    <strong>
                      {entry.ts} / {entry.source}
                    </strong>
                    <p>{entry.message}</p>
                  </div>
                </div>
              ))
            ) : (
              <div className="empty-state">
                <p>No log lines match the current filter.</p>
              </div>
            )}
          </div>
        </Panel>
      </div>
    );
  }

  async function handlePickDataDirectory() {
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      const picked = await tauriInvoke<string | null>("pick_directory");
      if (picked) {
        setSettingsDraft((current) => ({ ...current, dataDirectory: picked }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open the directory picker.");
    }
  }

  async function handlePickConversionOutputDir() {
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      const picked = await tauriInvoke<string | null>("pick_directory");
      if (picked) {
        // Suggest a sensible filename inside the chosen directory if the
        // user hasn't typed one yet, otherwise just set the directory.
        const suggested = conversionSource?.name
          ? `${picked.replace(/\/$/, "")}/${conversionSource.name.replace(/[^\w.-]/g, "-")}-mlx`
          : picked;
        updateConversionDraft("outputPath", suggested);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open the directory picker.");
    }
  }

  function renderSettings() {
    return (
      <div className="content-grid">
        <Panel
          title="Data Directory"
          subtitle="Where ChaosEngineAI stores chat history, settings, and benchmark runs. Change to a cloud-synced folder (Dropbox, iCloud) to back up or share across machines."
        >
          <div className="control-stack">
            <div className="directory-add-row">
              <input
                className="text-input directory-add-path mono-text"
                type="text"
                readOnly
                value={settingsDraft.dataDirectory || "~/.chaosengine"}
              />
              <button className="secondary-button" type="button" onClick={() => void handlePickDataDirectory()}>
                Browse...
              </button>
              <button
                className="secondary-button"
                type="button"
                onClick={() => setSettingsDraft((current) => ({ ...current, dataDirectory: "" }))}
              >
                Reset to default
              </button>
            </div>
            <p className="help-text">
              Changes take effect after the backend restarts. Existing data will be copied to the new location; the
              old files are left in place.
            </p>
          </div>
        </Panel>
        <Panel
          title="Model Directories"
          subtitle="Add the folders ChaosEngineAI should scan for local models, including custom, Ollama, LM Studio, or shared model paths."
          actions={
            <button className="primary-button" type="button" onClick={() => void handleSaveSettings()}>
              Save settings
            </button>
          }
        >
          <div className="control-stack directory-stack">
            <div className="directory-add-row">
              <input
                className="text-input directory-add-label"
                type="text"
                placeholder="Label (e.g. LM Studio models)"
                value={newDirectoryLabel}
                onChange={(event) => setNewDirectoryLabel(event.target.value)}
              />
              <input
                className="text-input directory-add-path"
                type="text"
                placeholder="/Users/dan/AI_Models"
                value={newDirectoryPath}
                onChange={(event) => setNewDirectoryPath(event.target.value)}
              />
              <button
                className="secondary-button"
                type="button"
                onClick={async () => {
                  const picked = await pickDirectory(newDirectoryPath);
                  if (picked) {
                    setNewDirectoryPath(picked);
                    if (!newDirectoryLabel.trim()) {
                      const leaf = picked.split(/[\\/]/).filter(Boolean).pop();
                      if (leaf) setNewDirectoryLabel(leaf);
                    }
                  }
                }}
              >
                Browse…
              </button>
              <button className="secondary-button" type="button" onClick={handleAddDirectory}>
                Add
              </button>
            </div>
            <div className="list scrollable-list directory-list">
              {settingsDraft.modelDirectories.map((directory) => (
                <div className="list-row directory-row" key={directory.id}>
                  <div className="directory-row-info">
                    <strong>{directory.label}</strong>
                    <p className="mono-text">{directory.path}</p>
                  </div>
                  <div className="directory-row-actions">
                    <span className={`badge ${directory.exists ? "success" : "warning"}`}>
                      {directory.exists ? "Found" : "Missing"}
                    </span>
                    <span className="badge muted">{directory.modelCount ?? 0} models</span>
                    <button
                      className="secondary-button small-button"
                      type="button"
                      onClick={async () => {
                        const picked = await pickDirectory(directory.path);
                        if (picked) handleUpdateDirectoryPath(directory.id, picked);
                      }}
                    >
                      Change…
                    </button>
                    <button className="secondary-button small-button" type="button" onClick={() => handleToggleDirectory(directory.id)}>
                      {directory.enabled ? "Disable" : "Enable"}
                    </button>
                    {directory.source === "user" ? (
                      <button className="secondary-button small-button" type="button" onClick={() => handleRemoveDirectory(directory.id)}>
                        Remove
                      </button>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Panel>

        <Panel
          title="Remote Providers"
          subtitle="Configure cloud OpenAI-compatible APIs as a fallback. Keys are stored locally with 0600 permissions."
          actions={
            <button className="secondary-button" type="button" onClick={() => {
              const id = `remote-${Date.now()}`;
              setSettingsDraft((c) => ({
                ...c,
                remoteProviders: [...(c.remoteProviders ?? []), { id, label: "New Provider", apiBase: "https://api.openai.com/v1", apiKey: "", model: "gpt-4o-mini" }],
              }));
            }}>
              + Add Provider
            </button>
          }
        >
          <div className="control-stack">
            {(settingsDraft.remoteProviders ?? []).length === 0 ? (
              <p className="empty-state">No remote providers configured. Add one to use cloud models as a fallback.</p>
            ) : null}
            {(settingsDraft.remoteProviders ?? []).map((provider, idx) => (
              <div key={provider.id} className="remote-provider-card">
                <div className="field-grid">
                  <label>
                    Label
                    <input
                      className="text-input"
                      type="text"
                      value={provider.label}
                      onChange={(event) => {
                        const next = [...(settingsDraft.remoteProviders ?? [])];
                        next[idx] = { ...next[idx], label: event.target.value };
                        setSettingsDraft((c) => ({ ...c, remoteProviders: next }));
                      }}
                    />
                  </label>
                  <label>
                    Model name
                    <input
                      className="text-input"
                      type="text"
                      value={provider.model}
                      onChange={(event) => {
                        const next = [...(settingsDraft.remoteProviders ?? [])];
                        next[idx] = { ...next[idx], model: event.target.value };
                        setSettingsDraft((c) => ({ ...c, remoteProviders: next }));
                      }}
                    />
                  </label>
                  <label>
                    API base URL (must be HTTPS)
                    <input
                      className="text-input"
                      type="url"
                      placeholder="https://api.openai.com/v1"
                      value={provider.apiBase}
                      onChange={(event) => {
                        const next = [...(settingsDraft.remoteProviders ?? [])];
                        next[idx] = { ...next[idx], apiBase: event.target.value };
                        setSettingsDraft((c) => ({ ...c, remoteProviders: next }));
                      }}
                    />
                  </label>
                  <label>
                    API key
                    <input
                      className="text-input"
                      type="password"
                      placeholder={provider.hasApiKey ? provider.apiKeyMasked || "•••• stored ••••" : "sk-..."}
                      value={provider.apiKey ?? ""}
                      onChange={(event) => {
                        const next = [...(settingsDraft.remoteProviders ?? [])];
                        next[idx] = { ...next[idx], apiKey: event.target.value };
                        setSettingsDraft((c) => ({ ...c, remoteProviders: next }));
                      }}
                    />
                  </label>
                </div>
                <div className="button-row">
                  <button className="secondary-button" type="button" onClick={() => {
                    const next = (settingsDraft.remoteProviders ?? []).filter((_, i) => i !== idx);
                    setSettingsDraft((c) => ({ ...c, remoteProviders: next }));
                  }}>
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel
          title="Hugging Face"
          subtitle="Required for gated models like Mistral, Llama, Gemma. Get one at huggingface.co/settings/tokens"
          actions={
            <button className="primary-button" type="button" onClick={() => void handleSaveSettings()}>
              Save settings
            </button>
          }
        >
          <div className="control-stack">
            <label>
              Hugging Face token
              <input
                className="text-input"
                type="password"
                placeholder={
                  settingsDraft.hasHuggingFaceToken
                    ? settingsDraft.huggingFaceTokenMasked || "•••• stored ••••"
                    : "hf_..."
                }
                value={settingsDraft.huggingFaceToken}
                onChange={(event) =>
                  setSettingsDraft((c) => ({ ...c, huggingFaceToken: event.target.value }))
                }
              />
            </label>
            <p className="muted-text">
              Stored locally. Used by MLX conversion when fetching gated models from Hugging Face.
            </p>
          </div>
        </Panel>

        <Panel
          title="Integrations"
          subtitle="Connect external tools to ChaosEngineAI's OpenAI-compatible API."
          className="settings-integrations-panel"
        >
          <div className="control-stack">
            <p className="muted-text">
              Use these snippets to connect popular tools to ChaosEngineAI as their LLM backend. The server must be running on{" "}
              <span className="mono-text">{workspace.server.localhostUrl ?? `http://127.0.0.1:${workspace.server.port}/v1`}</span>.
            </p>
            {[
              { name: "Continue.dev (VS Code)", config: `{\n  "models": [{\n    "title": "ChaosEngineAI",\n    "provider": "openai",\n    "model": "${workspace.runtime.loadedModel?.name ?? "current-model"}",\n    "apiBase": "${workspace.server.localhostUrl ?? `http://127.0.0.1:${workspace.server.port}/v1`}",\n    "apiKey": "not-needed"\n  }]\n}` },
              { name: "Goose", config: `# In ~/.config/goose/config.yaml\nGOOSE_PROVIDER: openai\nGOOSE_MODEL: ${workspace.runtime.loadedModel?.name ?? "current-model"}\nOPENAI_BASE_URL: ${workspace.server.localhostUrl ?? `http://127.0.0.1:${workspace.server.port}/v1`}\nOPENAI_API_KEY: not-needed` },
              { name: "Cursor", config: `1. Settings → Models → Add Model\n2. OpenAI API Key: not-needed\n3. Override OpenAI Base URL: ${workspace.server.localhostUrl ?? `http://127.0.0.1:${workspace.server.port}/v1`}\n4. Add custom model: ${workspace.runtime.loadedModel?.name ?? "current-model"}` },
              { name: "Claude Code (via OpenAI proxy)", config: `# Set environment variables before running claude\nexport ANTHROPIC_BASE_URL=${workspace.server.localhostUrl ?? `http://127.0.0.1:${workspace.server.port}/v1`}\nexport ANTHROPIC_AUTH_TOKEN=not-needed` },
            ].map((item) => (
              <div key={item.name} className="integration-card">
                <div className="integration-card-header">
                  <strong>{item.name}</strong>
                  <button className="secondary-button" type="button" onClick={() => copyText(item.config)}>
                    Copy
                  </button>
                </div>
                <pre className="integration-snippet">{item.config}</pre>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    );
  }

  let content = renderDashboard();
  if (activeTab === "online-models") {
    content = renderOnlineModels();
  } else if (activeTab === "my-models") {
    content = renderMyModels();
  } else if (activeTab === "image-discover") {
    content = renderImageDiscover();
  } else if (activeTab === "image-models") {
    content = renderImageModels();
  } else if (activeTab === "image-studio") {
    content = renderImageStudio();
  } else if (activeTab === "image-gallery") {
    content = renderImageGallery();
  } else if (activeTab === "conversion") {
    content = renderConversion();
  } else if (activeTab === "chat") {
    content = renderChat();
  } else if (activeTab === "server") {
    content = renderServer();
  } else if (activeTab === "benchmarks") {
    content = renderBenchmarkRun();
  } else if (activeTab === "benchmark-history") {
    content = renderBenchmarkHistory();
  } else if (activeTab === "logs") {
    content = renderLogs();
  } else if (activeTab === "settings") {
    content = renderSettings();
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-title-row">
            <img src="/logo.svg" alt="ChaosEngineAI" className="brand-logo" />
            <h1>ChaosEngineAI</h1>
          </div>
          <span className="brand-kicker">Local AI model runner</span>
        </div>

        <nav className="nav-list" aria-label="Primary">
          {tabs.filter((tab) => {
            // Conversion is MLX-only — hide on non-Darwin
            if (tab.id === "conversion" && workspace.system.platform && workspace.system.platform !== "Darwin") {
              return false;
            }
            return true;
          }).map((tab) => (
            <button
              key={tab.id}
              className={activeTab === tab.id ? "nav-button active" : "nav-button"}
              type="button"
              onClick={() => { setActiveTab(tab.id); setError(null); }}
            >
              <strong>{tab.label}</strong>
              <span>{tab.caption}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className={`badge ${backendOnline ? "success" : "warning"}`}>
            {backendOnline ? "Backend online" : "Offline"}
          </span>
          <p>{workspace.runtime.engineLabel}</p>
          <small>{workspace.runtime.loadedModel?.name ?? "No model loaded"}</small>
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div>
            <span className="eyebrow">Workspace</span>
            <h2>{tabs.find((tab) => tab.id === activeTab)?.label}</h2>
            <p>{tabs.find((tab) => tab.id === activeTab)?.caption}</p>
          </div>
          <div className="header-badges">
            <span className="badge muted">{workspace.system.platform}</span>
            <span className="badge muted">{workspace.system.arch}</span>
            <span className="badge accent">{workspace.runtime.loadedModel ? loadedModelCacheLabel : launchCacheLabel}</span>
          </div>
        </header>

        <div className="workspace-status-stack">
          {error ? (
            <div className="notice-banner error-banner">
              <span>{error}</span>
              {error.includes("update-llama-cpp.sh") ? (
                <button
                  className="primary-button"
                  type="button"
                  disabled={rebuildingLlama}
                  onClick={() => void handleRebuildLlamaCpp()}
                >
                  {rebuildingLlama ? "Rebuilding llama.cpp..." : "Rebuild llama.cpp"}
                </button>
              ) : null}
            </div>
          ) : null}
          {rebuildOutput ? (
            <div className="notice-banner">
              <pre className="mono-text rebuild-output">{rebuildOutput.slice(-2000)}</pre>
              <button
                className="secondary-button"
                type="button"
                onClick={() => setRebuildOutput(null)}
              >
                Dismiss
              </button>
            </div>
          ) : null}
        </div>

        <div className="workspace-content-frame">
          {loading ? <div className="loading-state">Loading workspace state...</div> : content}
        </div>
      </main>
      {renderLaunchModal()}
      {renderImageGenerationModal()}
      {(() => {
        if (!detailFamilyId) return null;
        const family = workspace.featuredModels.find((f) => f.id === detailFamilyId);
        if (!family) return null;
        const localCount = family.variants.filter((v) => v.availableLocally).length;
        return (
          <div className="modal-overlay" onClick={() => setDetailFamilyId(null)}>
            <div className="modal-content modal-wide detail-modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <div className="detail-header-main">
                  <h3>{family.name}</h3>
                  <div className="detail-header-badges">
                    <span className="badge muted">{family.provider}</span>
                    {family.variants.map((v) => v.format).filter((f, i, a) => a.indexOf(f) === i).map((f) => (
                      <span key={f} className="badge muted">{f}</span>
                    ))}
                    {localCount > 0 ? <span className="badge success">{localCount} downloaded</span> : null}
                    <span className="badge muted">{family.updatedLabel}</span>
                  </div>
                  <p className="muted-text">{family.headline}</p>
                </div>
              </div>
              <div className="modal-body detail-modal-body">
                <p className="discover-summary">{family.summary}</p>
                {renderCapabilityIcons(family.capabilities, 12)}
                <div className="detail-variants">
                  <span className="eyebrow">Variants ({family.variants.length})</span>
                  {family.variants.map((variant) => {
                    const matchedLocal = findLibraryItemForVariant(workspace.library, variant);
                    const downloadState = activeDownloads[variant.repo];
                    const isDownloading = downloadState?.state === "downloading";
                    const isDownloadPaused = downloadState?.state === "cancelled";
                    const isDownloadComplete = downloadState?.state === "completed";
                    return (
                      <div key={variant.id} className={`detail-variant${variant.availableLocally || isDownloadComplete ? " downloaded" : ""}`}>
                        <div className="detail-variant-info">
                          <strong>{variant.name}</strong>
                          <div className="detail-variant-meta">
                            <span>{variant.format}</span>
                            <span>{variant.quantization}</span>
                            <span>{number(variant.paramsB)}B params</span>
                            <span>{sizeLabel(variant.sizeGb)}</span>
                            <span>~{number(variant.estimatedMemoryGb ?? 0)} GB RAM</span>
                            <span>{variant.contextWindow} ctx</span>
                          </div>
                          {variant.note ? <p className="detail-variant-note">{variant.note}</p> : null}
                        </div>
                        <div className="detail-variant-actions">
                          {variant.availableLocally ? (
                            <>
                              {variant.launchMode === "convert" ? (
                                <button className="primary-button action-convert" type="button" onClick={() => { prepareCatalogConversion(variant); setDetailFamilyId(null); }}>Convert</button>
                              ) : null}
                              <button className="primary-button action-chat" type="button" onClick={() => { openModelSelector("thread", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`); setDetailFamilyId(null); }}>Chat</button>
                              <button className="primary-button action-server" type="button" onClick={() => { openModelSelector("server", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`); setDetailFamilyId(null); }}>Server</button>
                            </>
                          ) : isDownloading ? (
                            <>
                              <span className="badge accent">{downloadProgressLabel(downloadState)}</span>
                              <button className="secondary-button" type="button" onClick={() => void handleCancelModelDownload(variant.repo)}>Pause</button>
                            </>
                          ) : isDownloadPaused ? (
                            <>
                              <span className="badge warning">{downloadProgressLabel(downloadState)}</span>
                              <button className="secondary-button" type="button" onClick={() => void handleDownloadModel(variant.repo)}>Resume</button>
                            </>
                          ) : isDownloadComplete ? (
                            <span className="badge success">Download complete</span>
                          ) : (
                            <button className="secondary-button" type="button" onClick={() => void handleDownloadModel(variant.repo)}>Download</button>
                          )}
                          <a
                            className="text-link"
                            href={variant.link}
                            target="_blank"
                            rel="noreferrer"
                            onClick={(event) => {
                              event.preventDefault();
                              void handleOpenExternalUrl(variant.link);
                            }}
                          >
                            HF ↗
                          </a>
                        </div>
                      </div>
                    );
                  })}
                </div>
                {family.readme && family.readme.length > 0 ? (
                  <div className="detail-readme">
                    <span className="eyebrow">Overview</span>
                    {family.readme.map((line, i) => <p key={i}>{line}</p>)}
                  </div>
                ) : null}
              </div>
              <div className="modal-footer">
                <button className="primary-button" type="button" onClick={() => setDetailFamilyId(null)}>Close</button>
              </div>
            </div>
          </div>
        );
      })()}
      {dataDirRestartPrompt ? (
        <div className="modal-overlay" onClick={() => setDataDirRestartPrompt(null)}>
          <div className="modal-content" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h3>Restart required</h3>
            </div>
            <div className="modal-body">
              <p>
                The data directory has been changed. ChaosEngineAI needs to restart the backend to start
                reading and writing files in the new location.
              </p>
              {dataDirRestartPrompt.migration ? (
                <div className="control-stack">
                  <p className="mono-text">From: {dataDirRestartPrompt.migration.from}</p>
                  <p className="mono-text">To: {dataDirRestartPrompt.migration.to}</p>
                  {dataDirRestartPrompt.migration.copied.length > 0 ? (
                    <p>Copied: {dataDirRestartPrompt.migration.copied.join(", ")}</p>
                  ) : null}
                  {dataDirRestartPrompt.migration.skipped.length > 0 ? (
                    <p>Skipped (already present at destination): {dataDirRestartPrompt.migration.skipped.join(", ")}</p>
                  ) : null}
                </div>
              ) : null}
            </div>
            <div className="modal-footer">
              <button
                className="secondary-button"
                type="button"
                onClick={() => setDataDirRestartPrompt(null)}
              >
                Not now
              </button>
              <button
                className="primary-button"
                type="button"
                onClick={async () => {
                  setDataDirRestartPrompt(null);
                  setBusyAction("Restarting backend...");
                  try {
                    const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
                    await tauriInvoke("restart_backend_sidecar");
                    await refreshWorkspace(activeChatId || undefined);
                  } catch (err) {
                    setError(err instanceof Error ? err.message : "Failed to restart backend.");
                  } finally {
                    setBusyAction(null);
                  }
                }}
              >
                OK, restart now
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
