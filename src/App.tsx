import { useEffect, useMemo, useRef, useState } from "react";
import {
  checkBackend,
  convertModel,
  deleteSessionDocument,
  installCudaTorch,
  loadModel,
  getWorkspace,
  deleteModelPath,
  revealModelPath,
  resolveApiToken,
  unloadModel,
  updateSession,
} from "./api";
import { LaunchModal } from "./components/LaunchModal";
import { sanitizeSpeculativeSelection } from "./components/runtimeSupport";
import { ImageGenerationModal } from "./components/ImageGenerationModal";
import { VideoGenerationModal } from "./components/VideoGenerationModal";
import { Sidebar } from "./components/Sidebar";
import { StartupProgressPanel } from "./components/StartupProgressPanel";
import { SubtabBar } from "./components/SubtabBar";
import { LogsTab } from "./features/logs/LogsTab";
import { SettingsTab } from "./features/settings/SettingsTab";
import { DashboardTab } from "./features/dashboard/DashboardTab";
import { ServerTab } from "./features/server/ServerTab";
import { ChatTab } from "./features/chat/ChatTab";
import { CompareView } from "./features/chat/CompareView";
import { FineTuningTab } from "./features/finetuning/FineTuningTab";
import { PromptLibraryTab } from "./features/prompts/PromptLibraryTab";
import { PluginsTab } from "./features/plugins/PluginsTab";
import { ConversionTab } from "./features/conversion/ConversionTab";
import { BenchmarkRunTab } from "./features/benchmarks/BenchmarkRunTab";
import { BenchmarkHistoryTab } from "./features/benchmarks/BenchmarkHistoryTab";
import { OnlineModelsTab } from "./features/models/OnlineModelsTab";
import { MyModelsTab } from "./features/models/MyModelsTab";
import { ImageDiscoverTab } from "./features/images/ImageDiscoverTab";
import { ImageModelsTab } from "./features/images/ImageModelsTab";
import { ImageStudioTab } from "./features/images/ImageStudioTab";
import { ImageGalleryTab } from "./features/images/ImageGalleryTab";
import { VideoDiscoverTab } from "./features/video/VideoDiscoverTab";
import { VideoModelsTab } from "./features/video/VideoModelsTab";
import { VideoStudioTab } from "./features/video/VideoStudioTab";
import { VideoGalleryTab } from "./features/video/VideoGalleryTab";
import type {
  ChatSession,
  LibraryItem,
  LoadModelActionResult,
  ModelVariant,
  TabId,
} from "./types";
import type { ChatModelOption } from "./types/chat";
import { tabs, CAPABILITY_META } from "./constants";
import {
  number,
  sizeLabel,
  flattenVariants,
  capabilityMeta,
  firstDirectVariant,
  findVariantById,
  findVariantForReference,
  findLibraryItemForVariant,
  findCatalogVariantForLibraryItem,
  estimateLibraryItemResidentGb,
  estimateLibraryItemCompressedGb,
  libraryItemFormat,
  libraryItemQuantization,
  libraryItemBackend,
  libraryItemSourceKind,
  inferHfRepoFromLocalPath,
  isChatLibraryItem,
  resolveCapabilities,
  downloadProgressLabel,
  syncRuntime,
  settingsDraftFromWorkspace,
  parseContextK,
  estimateArchFromParams,
  estimateParamsBFromDisk,
  detectBitsPerWeight,
  compareOptionalNumber,
  serverOriginFromBase,
} from "./utils";
import {
  useWorkspace,
  useModels,
  useChat,
  useImageState,
  useVideoState,
  useBenchmarks,
  useSettings,
  useSidebarPrefs,
  useGpuStatus,
} from "./hooks";

export default function App() {
  // ── Workspace (core state) ─────────────────────────────────
  const ws = useWorkspace();
  const {
    workspace, setWorkspace,
    loading, loadingElapsedSeconds, backendOnline, setBackendOnline,
    tauriBackend, setTauriBackend,
    error, setError,
    busyAction, setBusyAction, busy,
    rebuildingLlama, rebuildOutput, setRebuildOutput,
    handleRebuildLlamaCpp,
    refreshWorkspace,
  } = ws;

  const [activeTab, setActiveTab] = useState<TabId>("dashboard");
  const [compareMode, setCompareMode] = useState(false);
  const [apiToken, setApiToken] = useState<string | null>(null);
  const sidebarPrefs = useSidebarPrefs();
  const gpuStatus = useGpuStatus(backendOnline);
  const [installingCudaTorch, setInstallingCudaTorch] = useState(false);
  const [cudaTorchResult, setCudaTorchResult] = useState<
    | { ok: true; indexUrl: string | null; pythonVersion: string | null }
    | { ok: false; message: string; pythonVersion: string | null; noWheelForPython: boolean }
    | null
  >(null);

  const handleInstallCudaTorch = async () => {
    if (installingCudaTorch) return;
    setInstallingCudaTorch(true);
    setCudaTorchResult(null);
    try {
      const result = await installCudaTorch();
      if (result.ok) {
        setCudaTorchResult({
          ok: true,
          indexUrl: result.indexUrl,
          pythonVersion: result.pythonVersion,
        });
      } else {
        const last = result.attempts[result.attempts.length - 1];
        const tail = (last?.output ?? result.output ?? "").split("\n").slice(-3).join("\n");
        setCudaTorchResult({
          ok: false,
          message: tail || "pip install failed — see backend logs for details.",
          pythonVersion: result.pythonVersion,
          noWheelForPython: result.noWheelForPython,
        });
      }
    } catch (err) {
      setCudaTorchResult({
        ok: false,
        message: err instanceof Error ? err.message : String(err),
        pythonVersion: null,
        noWheelForPython: false,
      });
    } finally {
      setInstallingCudaTorch(false);
    }
  };

  // ── Settings / Server / Preview ────────────────────────────
  const imgState = useImageState(backendOnline, setError, setActiveTab);
  const videoState = useVideoState(backendOnline, setError, setActiveTab);
  const settings = useSettings(
    workspace, setWorkspace,
    backendOnline, setBackendOnline,
    tauriBackend, setTauriBackend,
    setError, setBusyAction,
    "", // activeChatId placeholder — will be patched below
    async (preferredChatId?: string) => { await refreshWorkspace(preferredChatId); },
    imgState.refreshImageData,
  );
  const {
    settingsDraft, setSettingsDraft,
    launchSettings, setLaunchSettings,
    preview, setPreview,
    previewControls, setPreviewControls,
    dataDirRestartPrompt, setDataDirRestartPrompt,
    newDirectoryLabel, setNewDirectoryLabel,
    newDirectoryPath, setNewDirectoryPath,
    conversionDraft, setConversionDraft,
    lastConversion, setLastConversion,
    systemPrompt, setSystemPrompt,
    serverModelKey, setServerModelKey,
    installingPackage,
    installLogs,
    updateLaunchSetting,
    updateConversionDraft,
    handleAddDirectory,
    handleToggleDirectory,
    handleRemoveDirectory,
    handleUpdateDirectoryPath,
    pickDirectory,
    handlePickDataDirectory,
    handlePickImageOutputsDirectory,
    handlePickVideoOutputsDirectory,
    handleSaveSettings,
    handleStopServer,
    handleRestartServer,
    handleInstallPackage,
  } = settings;

  // ── Models (search, downloads) ─────────────────────────────
  const models = useModels(
    backendOnline,
    "", // activeChatId placeholder — patched below
    workspace.featuredModels,
    setError,
    async (preferredChatId?: string) => { await refreshWorkspace(preferredChatId); },
  );
  const {
    searchInput, setSearchInput,
    searchResults,
    hubResults,
    searchError,
    expandedHubId,
    hubFileCache,
    hubFileLoading,
    hubFileError,
    detailFamilyId, setDetailFamilyId,
    selectedFamilyId,
    selectedVariantId,
    expandedFamilyId, setExpandedFamilyId,
    expandedVariantId, setExpandedVariantId,
    activeDownloads,
    discoverCapFilter, setDiscoverCapFilter,
    discoverFormatFilter, setDiscoverFormatFilter,
    handleDownloadModel,
    handleCancelModelDownload,
    handleDeleteModelDownload,
    toggleHubExpand,
  } = models;

  // ── Derived model values ───────────────────────────────────
  const allFeaturedVariants = flattenVariants(workspace.featuredModels);
  const defaultChatVariant = firstDirectVariant(workspace.featuredModels);
  const activeFamilies = searchResults;
  const selectedFamily = activeFamilies.find((family) => family.id === selectedFamilyId) ?? activeFamilies[0] ?? null;
  const selectedVariant = findVariantById(activeFamilies, selectedVariantId) ?? (selectedFamily ? findVariantById([selectedFamily], selectedVariantId) : null);

  // ── Library state ──────────────────────────────────────────
  const [librarySearchInput, setLibrarySearchInput] = useState("");
  const [selectedLibraryPath, setSelectedLibraryPath] = useState(workspace.library.find(isChatLibraryItem)?.path ?? "");
  const [expandedLibraryPath, setExpandedLibraryPath] = useState<string | null>(null);
  const [librarySortKey, setLibrarySortKey] = useState<"name" | "format" | "backend" | "size" | "ram" | "compressed" | "modified" | "context">("modified");
  const [librarySortDir, setLibrarySortDir] = useState<"asc" | "desc">("desc");
  const [libraryCapFilter, setLibraryCapFilter] = useState<string | null>(null);
  const [libraryFormatFilter, setLibraryFormatFilter] = useState<string | null>(null);
  const [libraryBackendFilter, setLibraryBackendFilter] = useState<string | null>(null);
  const chatLibrary = useMemo(() => workspace.library.filter(isChatLibraryItem), [workspace.library]);

  // Library search sync
  useEffect(() => {
    const nextFilteredLibrary = chatLibrary
      .filter((item) => {
        const haystack = `${item.name} ${item.path} ${item.format} ${item.directoryLabel ?? ""}`.toLowerCase();
        return haystack.includes(librarySearchInput.trim().toLowerCase());
      })
      .sort((left, right) => {
        if (left.lastModified !== right.lastModified) return right.lastModified.localeCompare(left.lastModified);
        return right.sizeGb - left.sizeGb;
      });
    if (!nextFilteredLibrary.length) { setSelectedLibraryPath(""); return; }
    setSelectedLibraryPath((current) =>
      nextFilteredLibrary.some((item) => item.path === current) ? current : nextFilteredLibrary[0].path,
    );
  }, [chatLibrary, librarySearchInput]);

  // Library rows
  const libraryRows = chatLibrary.map((item) => {
    const matchedVariant = findCatalogVariantForLibraryItem(workspace.featuredModels, item);
    return {
      item,
      matchedVariant,
      displayFormat: libraryItemFormat(item, matchedVariant),
      displayQuantization: libraryItemQuantization(item, matchedVariant),
      displayBackend: libraryItemBackend(item, matchedVariant),
      sourceKind: libraryItemSourceKind(item),
      estimatedRamGb: estimateLibraryItemResidentGb(item, matchedVariant),
      estimatedCompressedGb: estimateLibraryItemCompressedGb(item, matchedVariant),
    };
  });
  // Synthetic rows for active downloads that have not yet landed in
  // ``workspace.library`` (e.g. the backend hasn't rescanned, or the partial
  // file isn't on disk yet). Without this a user clicking Download sees no
  // feedback in My Models until the backend refresh happens. We match by
  // any ``repo`` we can already attribute to a library row — via matched
  // catalog variant or an inferred HF cache path — so we don't duplicate
  // rows once the download hits disk.
  const knownDownloadRepos = new Set<string>();
  for (const row of libraryRows) {
    const repo = inferHfRepoFromLocalPath(row.item.path) ?? row.matchedVariant?.repo ?? (row.item.name.includes("/") ? row.item.name : null);
    if (repo) knownDownloadRepos.add(repo);
  }
  const syntheticDownloadRows = Object.values(activeDownloads)
    .filter((download) => download.state === "downloading" || download.state === "cancelled" || download.state === "failed")
    .filter((download) => !knownDownloadRepos.has(download.repo))
    .map((download) => {
      const variant = allFeaturedVariants.find((candidate) => candidate.repo === download.repo) ?? null;
      const now = new Date().toISOString().replace("T", " ").slice(0, 16);
      const downloadedGb = Math.max(0, download.downloadedGb ?? 0);
      const item: LibraryItem = {
        name: variant?.name ?? download.repo,
        path: `download://${download.repo}`,
        format: variant?.format ?? "Downloading",
        sourceKind: "HF cache",
        quantization: variant?.quantization ?? null,
        backend: variant?.backend ?? null,
        modelType: "text",
        sizeGb: downloadedGb,
        lastModified: now,
        actions: [],
        broken: false,
        brokenReason: null,
      };
      return {
        item,
        matchedVariant: variant,
        displayFormat: variant?.format ?? "Downloading",
        displayQuantization: variant?.quantization ?? null,
        displayBackend: variant?.backend ?? "",
        sourceKind: "Download",
        estimatedRamGb: variant?.estimatedMemoryGb ?? null,
        estimatedCompressedGb: variant?.estimatedCompressedMemoryGb ?? null,
      };
    });
  const filteredLibraryRows = [...libraryRows, ...syntheticDownloadRows]
    .filter(({ item, displayFormat, displayQuantization, displayBackend, sourceKind }) => {
      const haystack = `${item.name} ${item.path} ${displayFormat} ${displayQuantization ?? ""} ${displayBackend} ${sourceKind} ${item.directoryLabel ?? ""}`.toLowerCase();
      return haystack.includes(librarySearchInput.trim().toLowerCase());
    })
    .sort((left, right) => {
      const dir = librarySortDir === "asc" ? 1 : -1;
      switch (librarySortKey) {
        case "name": return dir * left.item.name.localeCompare(right.item.name);
        case "format": return dir * left.displayFormat.localeCompare(right.displayFormat);
        case "backend": return dir * left.displayBackend.localeCompare(right.displayBackend);
        case "size": return dir * (left.item.sizeGb - right.item.sizeGb);
        case "ram": return compareOptionalNumber(left.estimatedRamGb, right.estimatedRamGb, dir);
        case "compressed": return compareOptionalNumber(left.estimatedCompressedGb, right.estimatedCompressedGb, dir);
        case "context": { const lc = parseContextK(left.matchedVariant?.contextWindow); const rc = parseContextK(right.matchedVariant?.contextWindow); return dir * (lc - rc); }
        case "modified": default: return dir * left.item.lastModified.localeCompare(right.item.lastModified);
      }
    });
  const selectedLibraryRow = filteredLibraryRows.find(({ item }) => item.path === selectedLibraryPath) ?? filteredLibraryRows[0] ?? null;
  const selectedLibraryItem = selectedLibraryRow?.item ?? null;
  const selectedLibraryVariant = selectedLibraryRow?.matchedVariant ?? null;

  // ── Chat model options ─────────────────────────────────────
  // Only list models present in the local library — catalog-only entries
  // would let the user pick a model that isn't downloaded yet, which then
  // 500s on Load. Discover tab is the place to pull a new model.
  const libraryChatOptions: ChatModelOption[] = chatLibrary
    .filter((item) => !item.broken)
    .map((item) => {
      const matched = findCatalogVariantForLibraryItem(workspace.featuredModels, item);
      const displayFormat = libraryItemFormat(item, matched);
      const displayQuantization = libraryItemQuantization(item, matched);
      const canonicalRepo = matched?.repo ?? inferHfRepoFromLocalPath(item.path);
      return {
        key: `library:${item.path}`,
        label: item.name,
        detail: `${displayFormat} / ${number(item.sizeGb)} GB`,
        group: "Local library",
        model: item.name,
        modelRef: item.name,
        canonicalRepo,
        source: "library",
        path: item.path,
        backend: libraryItemBackend(item, matched),
        paramsB: matched?.paramsB,
        sizeGb: item.sizeGb,
        contextWindow: matched?.contextWindow,
        format: displayFormat,
        quantization: displayQuantization ?? undefined,
        maxContext: item.maxContext ?? matched?.maxContext ?? null,
        // Phase 2.11: resolve typed capabilities so the picker can show
        // capability badges per option without re-deriving in each view.
        capabilities: resolveCapabilities(canonicalRepo ?? item.name, matched?.capabilities ?? null),
      };
    });

  const threadModelOptions = libraryChatOptions;

  // ── Cache labels (needed early by useChat) ──────────────────
  const currentCacheLabel = launchSettings.cacheStrategy === "native"
    ? "Native f16"
    : `${launchSettings.cacheStrategy} ${launchSettings.cacheBits}-bit ${launchSettings.fp16Layers}+${launchSettings.fp16Layers}`;
  const launchCacheLabel = currentCacheLabel;
  const loadedModelCacheLabel = workspace.runtime.loadedModel
    ? workspace.runtime.loadedModel.cacheStrategy === "native"
      ? "Native f16"
      : `${workspace.runtime.loadedModel.cacheStrategy} ${workspace.runtime.loadedModel.cacheBits}-bit ${workspace.runtime.loadedModel.fp16Layers}+${workspace.runtime.loadedModel.fp16Layers}`
    : launchCacheLabel;

  function sanitizeSpeculativeForModel(params: {
    backend: string;
    modelRef: string;
    canonicalRepo?: string | null;
    modelName: string;
    speculativeDecoding: boolean;
    treeBudget: number;
  }) {
    return sanitizeSpeculativeSelection({
      dflashInfo: workspace.system.dflash,
      selectedBackend: params.backend,
      modelRef: params.modelRef,
      canonicalRepo: params.canonicalRepo ?? null,
      modelName: params.modelName,
      speculativeDecoding: params.speculativeDecoding,
      treeBudget: params.treeBudget,
    });
  }

  // ── Load model handler (stays in App — cross-domain) ───────
  async function handleLoadModel(payload: {
    modelRef: string;
    modelName?: string;
    canonicalRepo?: string | null;
    source?: string;
    backend?: string;
    path?: string;
    nextTab?: TabId;
    busyLabel?: string;
    cacheBits?: number;
    fp16Layers?: number;
    fusedAttention?: boolean;
    cacheStrategy?: string;
    fitModelInMemory?: boolean;
    contextTokens?: number;
    speculativeDecoding?: boolean;
    treeBudget?: number;
  }): Promise<LoadModelActionResult> {
    setError(null);
    setBusyAction(payload.busyLabel ?? "Loading model...");
    try {
      const sanitizedSpeculative = sanitizeSpeculativeForModel({
        backend: payload.backend ?? "auto",
        modelRef: payload.modelRef,
        canonicalRepo: payload.canonicalRepo ?? undefined,
        modelName: payload.modelName ?? payload.modelRef,
        speculativeDecoding: payload.speculativeDecoding ?? launchSettings.speculativeDecoding,
        treeBudget: payload.treeBudget ?? launchSettings.treeBudget,
      });
      const loadPayload = {
        modelRef: payload.modelRef,
        modelName: payload.modelName,
        canonicalRepo: payload.canonicalRepo ?? undefined,
        source: payload.source ?? "catalog",
        backend: payload.backend ?? "auto",
        path: payload.path,
        cacheBits: payload.cacheBits ?? launchSettings.cacheBits,
        fp16Layers: payload.fp16Layers ?? launchSettings.fp16Layers,
        fusedAttention: payload.fusedAttention ?? launchSettings.fusedAttention,
        cacheStrategy: payload.cacheStrategy ?? launchSettings.cacheStrategy,
        fitModelInMemory: payload.fitModelInMemory ?? launchSettings.fitModelInMemory,
        contextTokens: payload.contextTokens ?? launchSettings.contextTokens,
        speculativeDecoding: sanitizedSpeculative.speculativeDecoding,
        treeBudget: sanitizedSpeculative.treeBudget,
      };

      let loadSucceeded = false;
      let loadErrorMessage: string | null = null;
      let loadedRuntime: LoadModelActionResult["runtime"];
      try {
        const runtime = await loadModel(loadPayload);
        setWorkspace((current) => syncRuntime(current, runtime));
        loadSucceeded = true;
        loadedRuntime = runtime;
      } catch (loadErr) {
        loadErrorMessage = loadErr instanceof Error ? loadErr.message : String(loadErr);
        // If the error is a definitive failure (server returned 500, backend
        // unavailable, etc.), show it immediately — don't poll for 15 minutes.
        // Only poll when the error looks like a timeout (the load may still be
        // progressing in the background).
        const isTimeout = /timed?\s*out|timeout|abort/i.test(loadErrorMessage);
        if (isTimeout) {
          for (let attempt = 0; attempt < 450; attempt++) {
            await new Promise((r) => setTimeout(r, 2000));
            try {
              const ws = await getWorkspace();
              if (ws.runtime.loadedModel?.ref === payload.modelRef || ws.runtime.loadedModel?.runtimeTarget === payload.path) {
                setWorkspace(ws);
                loadSucceeded = true;
                loadErrorMessage = null;
                loadedRuntime = ws.runtime;
                break;
              }
              if (ws.server.loading) { setWorkspace(ws); continue; }
              break;
            } catch { /* backend unreachable */ }
          }
          if (loadSucceeded) {
            // Let the shared success path below refresh the workspace and tab state.
          }
        }
      }

      if (loadSucceeded) {
        await refreshWorkspace(chat.activeChatId || undefined);
        if (payload.nextTab) setActiveTab(payload.nextTab);
        return { ok: true, runtime: loadedRuntime };
      } else {
        const detail = loadErrorMessage || "The model could not be loaded. Check the server logs for details.";
        setError(`Failed to load ${payload.modelName ?? payload.modelRef}: ${detail}`);
        return { ok: false, error: detail };
      }
    } catch (actionError) {
      const detail = actionError instanceof Error ? actionError.message : "Unknown error";
      console.error("[handleLoadModel] Load failed for", payload.modelRef, "—", detail, actionError);
      setError(`Failed to load ${payload.modelName ?? payload.modelRef}: ${detail}`);
      return { ok: false, error: detail };
    } finally {
      setBusyAction(null);
    }
  }

  // ── Chat ───────────────────────────────────────────────────
  const chat = useChat(
    workspace, setWorkspace,
    backendOnline, setBackendOnline,
    setError,
    launchSettings,
    systemPrompt,
    setActiveTab,
    handleLoadModel,
    defaultChatVariant,
    threadModelOptions,
    currentCacheLabel,
    loadedModelCacheLabel,
    async (preferredChatId?: string) => { await refreshWorkspace(preferredChatId); },
  );
  const {
    activeChatId, setActiveChatId,
    threadTitleDraft, setThreadTitleDraft,
    draftMessage, setDraftMessage,
    chatBusySessionId,
    pendingImages, setPendingImages,
    chatScrollRef,
    sortedChatSessions,
    activeChat,
    activeThinkingMode,
    activeRuntimeProfileMatchesLaunchSettings,
    sessionModelPayload,
    persistSessionChanges,
    handleCreateSession,
    handleRenameActiveThread,
    handleToggleThreadPin,
    handleThinkingModeChange,
    handleDeleteSession,
    sendMessage,
  } = chat;

  // ── Benchmarks ─────────────────────────────────────────────
  const benchmarks = useBenchmarks(
    workspace, setWorkspace,
    launchSettings,
    activeTab,
    setError, setBusyAction,
  );
  const {
    benchmarkDraft, setBenchmarkDraft,
    benchmarkModelKey, setBenchmarkModelKey,
    selectedBenchmarkId, setSelectedBenchmarkId,
    compareBenchmarkId, setCompareBenchmarkId,
    benchmarkModelFilter, setBenchmarkModelFilter,
    benchmarkViewMode, setBenchmarkViewMode,
    benchmarkPromptId, setBenchmarkPromptId,
    benchmarkStartedAt, benchmarkError,
    showBenchmarkPicker, setShowBenchmarkPicker,
    showBenchmarkModal, setShowBenchmarkModal,
    updateBenchmarkDraft,
    selectedBenchmark, compareBenchmark,
  } = benchmarks;

  // ── Remaining cross-domain state ───────────────────────────
  const [logQuery, setLogQuery] = useState("");
  const [showConversionPicker, setShowConversionPicker] = useState(false);
  const [showConversionModal, setShowConversionModal] = useState(false);
  const [conversionError, setConversionError] = useState<string | null>(null);
  const [conversionStartedAt, setConversionStartedAt] = useState<number | null>(null);
  const [showRemoteTest, setShowRemoteTest] = useState(false);
  const [testModelId, setTestModelId] = useState<string | null>(null);

  // Launch modal
  const [pendingLaunch, setPendingLaunch] = useState<{ action: "chat" | "server" | "thread"; preselectedKey?: string } | null>(null);
  const [launchModelSearch, setLaunchModelSearch] = useState("");

  // ── Initial load side-effects (runs after useWorkspace resolves) ──
  // Sync initial payload into all hooks on first successful load
  useEffect(() => {
    if (loading) return;
    setPreview(workspace.preview);
    setLaunchSettings(workspace.settings.launchPreferences);
    setSettingsDraft(settingsDraftFromWorkspace(workspace.settings));
    setPreviewControls({
      bits: workspace.settings.launchPreferences.cacheBits,
      fp16Layers: workspace.settings.launchPreferences.fp16Layers,
      numLayers: workspace.preview.numLayers,
      numHeads: workspace.preview.numHeads,
      numKvHeads: workspace.preview.numKvHeads,
      hiddenSize: workspace.preview.hiddenSize,
      contextTokens: workspace.settings.launchPreferences.contextTokens,
      paramsB: workspace.preview.paramsB,
      strategy: workspace.settings.launchPreferences.cacheStrategy,
    });
    setActiveChatId(workspace.chatSessions[0]?.id ?? "");
    setThreadTitleDraft(workspace.chatSessions[0]?.title ?? "");
    void imgState.refreshImageData();
    void videoState.refreshVideoData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading]);

  // Backend polling
  const isModelLoading = workspace.server.loading !== null;
  useEffect(() => {
    const pollInterval = !backendOnline ? 2500 : isModelLoading ? 2000 : 5000;
    const interval = window.setInterval(() => {
      if ((busy || chatBusySessionId !== null) && !isModelLoading) return;
      void (async () => {
        const online = await checkBackend();
        setBackendOnline(online);
        if (online) void refreshWorkspace(activeChatId || undefined);
      })();
    }, pollInterval);
    return () => window.clearInterval(interval);
  }, [activeChatId, backendOnline, busy, isModelLoading, chatBusySessionId, refreshWorkspace, setBackendOnline]);

  // ── Cross-domain derived state ─────────────────────────────
  const nativeBackends = workspace.runtime.nativeBackends;
  const filteredLogs = workspace.logs.filter((entry) => {
    const haystack = `${entry.ts} ${entry.source} ${entry.level} ${entry.message}`.toLowerCase();
    return haystack.includes(logQuery.toLowerCase());
  });
  const serverLogEntries = workspace.logs.slice(0, 120).reverse();
  const previewSavings = Math.max(0, preview.baselineCacheGb - preview.optimizedCacheGb);
  const conversionReady = Boolean(nativeBackends?.converterAvailable ?? workspace.system.mlxLmAvailable);
  const enabledDirectoryCount = (workspace.settings?.modelDirectories ?? []).filter((directory) => directory.enabled).length;
  const libraryTotalSizeGb = chatLibrary.reduce((sum, item) => sum + item.sizeGb, 0);
  const localVariantCount = allFeaturedVariants.filter((variant) => variant.availableLocally).length;
  const fileRevealLabel =
    workspace.system.platform === "Darwin" ? "Show in Finder" :
    workspace.system.platform === "Windows" ? "Show in Explorer" : "Show in Files";

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
    if (!selectedServerOptionBase || selectedServerOptionBase.source !== "catalog") return selectedServerOptionBase;
    const variant = findVariantForReference(workspace.featuredModels, selectedServerOptionBase.modelRef, selectedServerOptionBase.model);
    if (!variant) return selectedServerOptionBase;
    const localItem = findLibraryItemForVariant(chatLibrary, variant);
    if (!localItem) return selectedServerOptionBase;
    return libraryChatOptions.find((option) => option.path === localItem.path) ?? selectedServerOptionBase;
  })();
  const convertibleLibrary = chatLibrary.filter((item) => libraryItemFormat(item) !== "MLX");
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
  const previewVariant =
    activeTab === "server"
      ? findVariantForReference(workspace.featuredModels, selectedServerOption?.modelRef ?? workspace.runtime.loadedModel?.ref, selectedServerOption?.model ?? workspace.runtime.loadedModel?.name) ??
        findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ?? defaultChatVariant
      : activeTab === "my-models"
        ? selectedLibraryVariant ?? findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ?? defaultChatVariant
      : activeTab === "chat"
        ? findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ??
          findVariantForReference(workspace.featuredModels, selectedServerOption?.modelRef ?? workspace.runtime.loadedModel?.ref, selectedServerOption?.model ?? workspace.runtime.loadedModel?.name) ?? defaultChatVariant
        : selectedVariant ?? findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ?? defaultChatVariant;

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
  const authHeaderFlag = apiToken
    ? ` -H 'Authorization: Bearer ${apiToken}'`
    : " -H 'Authorization: Bearer <chaosengine-api-token>'";
  const localHealthCurl = `curl -sS ${localServerOrigin}/api/health`;
  const localModelsCurl = `curl -sS ${localServerUrl}/models${authHeaderFlag}`;
  const remoteHealthCurl = primaryLanOrigin ? `curl -sS ${primaryLanOrigin}/api/health` : null;
  const remoteModelsCurl = primaryLanUrl ? `curl -sS ${primaryLanUrl}/models${authHeaderFlag}` : null;

  // ── Cross-domain effects ───────────────────────────────────

  useEffect(() => {
    let cancelled = false;
    if (!backendOnline) {
      setApiToken(null);
      return;
    }
    void resolveApiToken()
      .then((token) => {
        if (!cancelled) {
          setApiToken(token);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setApiToken(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [backendOnline, tauriBackend?.apiBase, tauriBackend?.apiToken, workspace.server.port]);

  // Benchmark page: sync benchmarkDraft sliders -> previewControls
  useEffect(() => {
    if (activeTab !== "benchmarks") return;
    setPreviewControls((current) => {
      if (
        current.bits === benchmarkDraft.cacheBits &&
        current.fp16Layers === benchmarkDraft.fp16Layers &&
        current.contextTokens === benchmarkDraft.contextTokens &&
        current.strategy === benchmarkDraft.cacheStrategy
      ) return current;
      return {
        ...current,
        bits: benchmarkDraft.cacheBits,
        fp16Layers: benchmarkDraft.fp16Layers,
        contextTokens: benchmarkDraft.contextTokens,
        strategy: benchmarkDraft.cacheStrategy,
      };
    });
  }, [activeTab, benchmarkDraft.cacheBits, benchmarkDraft.fp16Layers, benchmarkDraft.contextTokens, benchmarkDraft.cacheStrategy, setPreviewControls]);

  // Sync previewVariant -> previewControls.paramsB
  useEffect(() => {
    if (!previewVariant) return;
    setPreviewControls((current) =>
      current.paramsB === previewVariant.paramsB ? current : { ...current, paramsB: previewVariant.paramsB },
    );
  }, [previewVariant?.paramsB, setPreviewControls]);

  // Sync serverModelKey when options change
  useEffect(() => {
    if (!threadModelOptions.length) { setServerModelKey(""); return; }
    setServerModelKey((current) => {
      if (threadModelOptions.some((option) => option.key === current)) return current;
      return loadedModelOption?.key ?? activeThreadOption?.key ?? threadModelOptions[0].key;
    });
  }, [activeThreadOption?.key, loadedModelOption?.key, serverOptionKeySignature, setServerModelKey]);

  // Sync benchmarkModelKey when options change
  useEffect(() => {
    if (!threadModelOptions.length) { setBenchmarkModelKey(""); return; }
    setBenchmarkModelKey((current) => {
      if (threadModelOptions.some((option) => option.key === current)) return current;
      const firstHealthy = chatLibrary.find((item) => !item.broken);
      if (firstHealthy) return `library:${firstHealthy.path}`;
      if (chatLibrary.length > 0) return `library:${chatLibrary[0].path}`;
      return activeThreadOption?.key ?? loadedModelOption?.key ?? threadModelOptions[0].key;
    });
  }, [activeThreadOption?.key, chatLibrary, loadedModelOption?.key, serverOptionKeySignature, setBenchmarkModelKey]);

  // Sync benchmarkDraft model fields
  useEffect(() => {
    if (!benchmarkOption) return;
    setBenchmarkDraft((current) => {
      if (current.modelRef === benchmarkOption.modelRef && current.source === benchmarkOption.source && current.backend === benchmarkOption.backend) return current;
      return { ...current, modelRef: benchmarkOption.modelRef, modelName: benchmarkOption.model, source: benchmarkOption.source, backend: benchmarkOption.backend, path: benchmarkOption.path ?? undefined };
    });
  }, [benchmarkOption?.key, setBenchmarkDraft]);

  // Auto-scroll chat
  const lastMessageLength = activeChat?.messages[activeChat.messages.length - 1]?.text?.length ?? 0;
  useEffect(() => {
    if (activeTab !== "chat") return;
    const handle = requestAnimationFrame(() => {
      const el = chatScrollRef.current;
      if (!el) return;
      // Only auto-scroll if the user is already near the bottom.
      // This lets users scroll up to read earlier content during streaming
      // without being yanked back to the bottom on every token.
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
      if (nearBottom) {
        el.scrollTop = el.scrollHeight;
      }
    });
    return () => cancelAnimationFrame(handle);
  }, [activeTab, activeChat?.id, activeChat?.messages.length, lastMessageLength, chatScrollRef]);

  // Re-anchor previewControls when benchmark/conversion model changes
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
    if (!paramsB && sizeGb) paramsB = estimateParamsBFromDisk(sizeGb, detectBitsPerWeight(label));
    if (!paramsB) return;
    const arch = estimateArchFromParams(paramsB);
    setPreviewControls((current) => {
      if (current.paramsB === paramsB && current.numLayers === arch.numLayers && current.numHeads === arch.numHeads && current.numKvHeads === arch.numKvHeads && current.hiddenSize === arch.hiddenSize) return current;
      return { ...current, paramsB, ...arch };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, benchmarkOption?.key, conversionSource?.path, conversionVariant?.id]);

  // Launch modal: re-anchor preview to selected model
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
      if (current.paramsB === paramsB && current.numLayers === arch.numLayers && current.numHeads === arch.numHeads && current.numKvHeads === arch.numKvHeads && current.hiddenSize === arch.hiddenSize) return current;
      return { ...current, paramsB, ...arch };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingLaunch?.preselectedKey, libraryChatOptions, threadModelOptions]);

  // ── Cross-domain handlers ──────────────────────────────────

  async function handleLoadLibraryItem(item: LibraryItem, nextTab: TabId) {
    const canonicalRepo = inferHfRepoFromLocalPath(item.path);
    await handleLoadModel({
      modelRef: item.name,
      modelName: item.name,
      canonicalRepo,
      source: "library",
      backend: libraryItemBackend(item),
      path: item.path,
      nextTab,
    });
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

  function handleSelectServerModel(nextKey: string) {
    setServerModelKey(nextKey);
  }

  async function handleLoadServerModel() {
    if (!selectedServerOption) { setError("Choose a model for the server before loading it."); return; }
    await handleLoadModel({
      modelRef: selectedServerOption.modelRef,
      modelName: selectedServerOption.model,
      canonicalRepo: selectedServerOption.canonicalRepo,
      source: selectedServerOption.source,
      backend: selectedServerOption.backend,
      path: selectedServerOption.path ?? undefined,
      nextTab: "server",
    });
  }

  async function handleRevealPath(path: string) {
    try {
      if (backendOnline) { await revealModelPath(path); return; }
    } catch { /* fallback below */ }
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      const parentDir = path.replace(/\/[^/]+$/, "");
      await tauriInvoke("plugin:opener|open_path", { path: parentDir });
    } catch {
      setError("Could not open file location. Try navigating manually to: " + path);
    }
  }

  async function handleOpenExternalUrl(url: string) {
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      await tauriInvoke("plugin:opener|open_url", { url });
      return;
    } catch { /* fall through */ }
    try {
      const opened = window.open(url, "_blank", "noopener,noreferrer");
      if (opened) return;
    } catch { /* fall through */ }
    setError(`Could not open link. Try opening this URL manually: ${url}`);
  }

  async function handleDeleteModel(item: LibraryItem) {
    const confirmed = window.confirm(
      `Delete "${item.name}"?\n\nThis will permanently remove the files at:\n${item.path}\n\nThis action cannot be undone.`,
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

  function prepareCatalogConversion(model: ModelVariant) {
    const matchingItem = findLibraryItemForVariant(convertibleLibrary, model);
    if (matchingItem) { prepareLibraryConversion(matchingItem); return; }
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

  async function handleConvertModel() {
    const modelRef = conversionDraft.modelRef.trim();
    const path = conversionDraft.path.trim();
    const hfRepo = conversionDraft.hfRepo.trim();
    const outputPath = conversionDraft.outputPath.trim();
    if (!modelRef && !path) { setError("Enter a model reference or a local path before starting conversion."); return; }
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
      setWorkspace((current) => syncRuntime({ ...current, library: response.library }, response.runtime));
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

  async function handlePickConversionOutputDir() {
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      const picked = await tauriInvoke<string | null>("pick_directory");
      if (picked) {
        const suggested = conversionSource?.name
          ? `${picked.replace(/\/$/, "")}/${conversionSource.name.replace(/[^\w.-]/g, "-")}-mlx`
          : picked;
        updateConversionDraft("outputPath", suggested);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open the directory picker.");
    }
  }

  function loadPayloadFromVariant(variant: ModelVariant, nextTab?: TabId) {
    const localItem = findLibraryItemForVariant(chatLibrary, variant);
    if (localItem) {
      return {
        modelRef: localItem.name,
        modelName: localItem.name,
        canonicalRepo: variant.repo,
        source: "library",
        backend: libraryItemBackend(localItem),
        path: localItem.path,
        nextTab,
      };
    }
    return {
      modelRef: variant.id,
      modelName: variant.name,
      canonicalRepo: variant.repo,
      source: "catalog",
      backend: variant.backend,
      nextTab,
    };
  }

  function threadPatchFromVariant(variant: ModelVariant): Pick<
    ChatSession,
    "model" | "modelRef" | "canonicalRepo" | "modelSource" | "modelPath" | "modelBackend" | "cacheLabel" | "updatedAt"
    | "cacheStrategy" | "cacheBits" | "fp16Layers" | "fusedAttention" | "fitModelInMemory"
    | "contextTokens" | "speculativeDecoding" | "dflashDraftModel" | "treeBudget"
  > {
    const localItem = findLibraryItemForVariant(chatLibrary, variant);
    const modelRef = localItem?.name ?? variant.id;
    const modelName = localItem?.name ?? variant.name;
    const modelBackend = localItem ? libraryItemBackend(localItem, variant) : variant.backend;
    const sanitizedSpeculative = sanitizeSpeculativeForModel({
      backend: modelBackend,
      modelRef,
      canonicalRepo: variant.repo,
      modelName,
      speculativeDecoding: launchSettings.speculativeDecoding,
      treeBudget: launchSettings.treeBudget,
    });
    return {
      model: modelName,
      modelRef,
      canonicalRepo: variant.repo,
      modelSource: localItem ? "library" : "catalog",
      modelPath: localItem?.path ?? null,
      modelBackend,
      cacheLabel: launchCacheLabel,
      cacheStrategy: launchSettings.cacheStrategy,
      cacheBits: launchSettings.cacheBits,
      fp16Layers: launchSettings.fp16Layers,
      fusedAttention: launchSettings.fusedAttention,
      fitModelInMemory: launchSettings.fitModelInMemory,
      contextTokens: launchSettings.contextTokens,
      speculativeDecoding: sanitizedSpeculative.speculativeDecoding,
      dflashDraftModel: null,
      treeBudget: sanitizedSpeculative.treeBudget,
      updatedAt: new Date().toLocaleString(),
    };
  }

  async function handleApplyVariantToActiveThread(variant: ModelVariant) {
    if (!activeChat) return;
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
      setWorkspace((current) => ({ ...current, chatSessions: [...current.chatSessions, localSession] }));
      setActiveChatId(localSession.id);
      setThreadTitleDraft(localSession.title);
      setActiveTab("chat");
      return;
    }
    try {
      const { createSession } = await import("./api");
      const { upsertSession } = await import("./utils");
      const session = await createSession("New chat");
      const updated = await updateSession(session.id, threadPatchFromVariant(variant));
      setWorkspace((current) => ({ ...current, chatSessions: upsertSession(current.chatSessions, updated) }));
      setActiveChatId(updated.id);
      setThreadTitleDraft(updated.title);
      setActiveTab("chat");
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to start a new thread.");
    }
  }

  function openModelSelector(action: "chat" | "server" | "thread", preselectedKey?: string) {
    setLaunchModelSearch("");
    let normalizedKey = preselectedKey;
    if (normalizedKey?.startsWith("catalog:")) {
      const modelRef = normalizedKey.slice("catalog:".length);
      const variant = findVariantForReference(workspace.featuredModels, modelRef, undefined);
      const localItem = variant ? findLibraryItemForVariant(chatLibrary, variant) : null;
      if (localItem) normalizedKey = `library:${localItem.path}`;
    }
    // If no key given, or the key references a model no longer in the options
    // (e.g. deleted/broken), fall back to the currently loaded model.
    if ((!normalizedKey || !threadModelOptions.some((o) => o.key === normalizedKey)) && loadedModelOption) {
      normalizedKey = loadedModelOption.key;
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
    if (action === "chat" || action === "thread") setActiveTab("chat");
    else if (action === "server") setActiveTab("server");
    if (action === "thread") {
      const variant = findVariantForReference(workspace.featuredModels, option.modelRef, option.model);
      if (variant) { await handleStartThreadWithVariant(variant); }
      else {
        await handleLoadModel({
          modelRef: option.modelRef,
          modelName: option.model,
          canonicalRepo: option.canonicalRepo,
          source: option.source,
          backend: option.backend,
          path: option.path ?? undefined,
        });
      }
    } else if (action === "chat") {
      if (activeChat) {
        const sanitizedSpeculative = sanitizeSpeculativeForModel({
          backend: option.backend,
          modelRef: option.modelRef,
          canonicalRepo: option.canonicalRepo ?? null,
          modelName: option.model,
          speculativeDecoding: activeChat.speculativeDecoding ?? launchSettings.speculativeDecoding,
          treeBudget: activeChat.treeBudget ?? launchSettings.treeBudget,
        });
        await persistSessionChanges(activeChat.id, {
          model: option.model,
          modelRef: option.modelRef,
          canonicalRepo: option.canonicalRepo ?? null,
          modelSource: option.source,
          modelPath: option.path ?? null,
          modelBackend: option.backend,
          speculativeDecoding: sanitizedSpeculative.speculativeDecoding,
          dflashDraftModel: null,
          treeBudget: sanitizedSpeculative.treeBudget,
          updatedAt: new Date().toLocaleString(),
        });
      }
      await handleLoadModel({
        modelRef: option.modelRef,
        modelName: option.model,
        canonicalRepo: option.canonicalRepo,
        source: option.source,
        backend: option.backend,
        path: option.path ?? undefined,
      });
    } else if (action === "server") {
      await handleLoadModel({
        modelRef: option.modelRef,
        modelName: option.model,
        canonicalRepo: option.canonicalRepo,
        source: option.source,
        backend: option.backend,
        path: option.path ?? undefined,
      });
    }
  }

  function copyText(text: string) {
    void navigator.clipboard.writeText(text);
  }

  // Window resize for details panels
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
            const size = await win.innerSize();
            originalWindowSizeRef.current = { width: size.width, height: size.height };
            await win.setSize(new (await import("@tauri-apps/api/window")).LogicalSize(
              Math.min(1800, Math.round(size.width * 1.15)),
              Math.min(1100, Math.round(size.height * 1.1)),
            ));
          } else if (next === 0 && prev > 0 && originalWindowSizeRef.current) {
            const { LogicalSize } = await import("@tauri-apps/api/window");
            await win.setSize(new LogicalSize(originalWindowSizeRef.current.width, originalWindowSizeRef.current.height));
            originalWindowSizeRef.current = null;
          }
        })();
        return next;
      });
    } catch { /* Not running in Tauri */ }
  }

  function renderCapabilityIcons(capabilities: string[], max = 5) {
    return (
      <div className="capability-strip">
        {capabilities.slice(0, max).map((capability) => {
          const meta = capabilityMeta(capability);
          const fullMeta = CAPABILITY_META[capability];
          return (
            <span className="capability-icon" key={capability} title={meta.title}
              style={fullMeta ? { borderColor: `${fullMeta.color}40`, color: fullMeta.color } : undefined}>
              {fullMeta?.icon ?? ""} {meta.shortLabel}
            </span>
          );
        })}
      </div>
    );
  }

  // ── Tab content ────────────────────────────────────────────
  let content = <DashboardTab
    system={workspace.system}
    recommendation={workspace.recommendation}
    runtime={workspace.runtime}
    activity={workspace.activity}
    backendOnline={backendOnline}
  />;
  if (activeTab === "online-models") {
    content = (
      <OnlineModelsTab
        searchResults={searchResults}
        searchInput={searchInput}
        onSearchInputChange={setSearchInput}
        searchError={searchError}
        localVariantCount={localVariantCount}
        discoverCapFilter={discoverCapFilter}
        onDiscoverCapFilterChange={setDiscoverCapFilter}
        discoverFormatFilter={discoverFormatFilter}
        onDiscoverFormatFilterChange={setDiscoverFormatFilter}
        expandedFamilyId={expandedFamilyId}
        onExpandedFamilyIdChange={setExpandedFamilyId}
        expandedVariantId={expandedVariantId}
        onExpandedVariantIdChange={setExpandedVariantId}
        onDetailFamilyIdChange={setDetailFamilyId}
        library={chatLibrary}
        activeDownloads={activeDownloads}
        onDownloadModel={(repo) => void handleDownloadModel(repo)}
        onCancelModelDownload={(repo) => void handleCancelModelDownload(repo)}
        onDeleteModelDownload={(repo) => void handleDeleteModelDownload(repo)}
        onPrepareCatalogConversion={prepareCatalogConversion}
        onOpenModelSelector={openModelSelector}
        onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
        hubResults={hubResults}
        expandedHubId={expandedHubId}
        onToggleHubExpand={(repo) => void toggleHubExpand(repo)}
        hubFileCache={hubFileCache}
        hubFileLoading={hubFileLoading}
        hubFileError={hubFileError}
        availableMemoryGb={workspace.system.availableMemoryGb}
      />
    );
  } else if (activeTab === "my-models") {
    content = (
      <MyModelsTab
        filteredLibraryRows={filteredLibraryRows}
        libraryTotalSizeGb={libraryTotalSizeGb}
        enabledDirectoryCount={enabledDirectoryCount}
        librarySearchInput={librarySearchInput}
        onLibrarySearchInputChange={setLibrarySearchInput}
        libraryCapFilter={libraryCapFilter}
        onLibraryCapFilterChange={setLibraryCapFilter}
        libraryFormatFilter={libraryFormatFilter}
        onLibraryFormatFilterChange={setLibraryFormatFilter}
        libraryBackendFilter={libraryBackendFilter}
        onLibraryBackendFilterChange={setLibraryBackendFilter}
        strategyCompat={{
          turboInstalled: !!workspace.system.llamaServerTurboPath,
          turboquantMlxAvailable: workspace.system.availableCacheStrategies?.some((s) => s.id === "turboquant" && s.available) ?? false,
          chaosengineAvailable: workspace.system.availableCacheStrategies?.some((s) => s.id === "chaosengine" && s.available) ?? false,
          dflashSupportedModels: workspace.system.dflash?.supportedModels ?? [],
        }}
        activeDownloads={activeDownloads}
        expandedLibraryPath={expandedLibraryPath}
        onExpandedLibraryPathChange={setExpandedLibraryPath}
        fileRevealLabel={fileRevealLabel}
        onDownloadModel={(repo) => void handleDownloadModel(repo)}
        onCancelModelDownload={(repo) => void handleCancelModelDownload(repo)}
        onDeleteModelDownload={(repo) => void handleDeleteModelDownload(repo)}
        onPrepareLibraryConversion={prepareLibraryConversion}
        onOpenModelSelector={openModelSelector}
        onRevealPath={(path) => void handleRevealPath(path)}
        onDeleteModel={(item) => void handleDeleteModel(item)}
        librarySortKey={librarySortKey}
        librarySortDir={librarySortDir}
        onLibrarySortKeyChange={setLibrarySortKey}
        onLibrarySortDirChange={setLibrarySortDir}
      />
    );
  } else if (activeTab === "image-discover") {
    content = (
      <ImageDiscoverTab
        combinedImageDiscoverResults={imgState.combinedImageDiscoverResults}
        imageDiscoverSearchInput={imgState.imageDiscoverSearchInput}
        onImageDiscoverSearchInputChange={imgState.setImageDiscoverSearchInput}
        imageDiscoverTaskFilter={imgState.imageDiscoverTaskFilter}
        onImageDiscoverTaskFilterChange={imgState.setImageDiscoverTaskFilter}
        imageDiscoverAccessFilter={imgState.imageDiscoverAccessFilter}
        onImageDiscoverAccessFilterChange={imgState.setImageDiscoverAccessFilter}
        imageDiscoverSort={imgState.imageDiscoverSort}
        onImageDiscoverSortChange={imgState.setImageDiscoverSort}
        imageDiscoverHasActiveFilters={imgState.imageDiscoverHasActiveFilters}
        imageDiscoverSearchQuery={imgState.imageDiscoverSearchQuery}
        activeImageDownloads={imgState.activeImageDownloads}
        selectedImageVariant={imgState.selectedImageVariant}
        fileRevealLabel={fileRevealLabel}
        onActiveTabChange={setActiveTab}
        onOpenImageStudio={imgState.openImageStudio}
        onImageDownload={(repo) => void imgState.handleImageDownload(repo)}
        onCancelImageDownload={(repo) => void imgState.handleCancelImageDownload(repo)}
        onDeleteImageDownload={(repo) => void imgState.handleDeleteImageDownload(repo)}
        onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
        onRevealPath={(path) => void handleRevealPath(path)}
      />
    );
  } else if (activeTab === "image-models") {
    content = (
      <ImageModelsTab
        installedImageVariants={imgState.installedImageVariants}
        imageCatalog={imgState.imageCatalog}
        activeImageDownloads={imgState.activeImageDownloads}
        fileRevealLabel={fileRevealLabel}
        onActiveTabChange={setActiveTab}
        onOpenImageStudio={imgState.openImageStudio}
        onImageDownload={(repo) => void imgState.handleImageDownload(repo)}
        onCancelImageDownload={(repo) => void imgState.handleCancelImageDownload(repo)}
        onDeleteImageDownload={(repo) => void imgState.handleDeleteImageDownload(repo)}
        onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
        onRevealPath={(path) => void handleRevealPath(path)}
      />
    );
  } else if (activeTab === "image-studio") {
    content = (
      <ImageStudioTab
        imageCatalog={imgState.imageCatalogWithLatest}
        selectedImageModelId={imgState.selectedImageModelId}
        onSelectedImageModelIdChange={imgState.setSelectedImageModelId}
        selectedImageVariant={imgState.selectedImageVariant}
        selectedImageFamily={imgState.selectedImageFamily}
        selectedImageLoaded={imgState.selectedImageLoaded}
        selectedImageWillLoadOnGenerate={imgState.selectedImageWillLoadOnGenerate}
        imageRuntimeLoadedDifferentModel={imgState.imageRuntimeLoadedDifferentModel}
        loadedImageVariant={imgState.loadedImageVariant}
        imageRuntimeStatus={imgState.imageRuntimeStatus}
        tauriBackend={tauriBackend}
        busy={busy}
        busyAction={busyAction}
        imageBusy={imgState.imageBusy}
        imageBusyLabel={imgState.imageBusyLabel}
        backendOnline={backendOnline}
        activeImageDownloads={imgState.activeImageDownloads}
        imagePrompt={imgState.imagePrompt}
        onImagePromptChange={imgState.setImagePrompt}
        imageNegativePrompt={imgState.imageNegativePrompt}
        onImageNegativePromptChange={imgState.setImageNegativePrompt}
        imageQualityPreset={imgState.imageQualityPreset}
        imageDraftMode={imgState.imageDraftMode}
        onImageDraftModeChange={imgState.setImageDraftMode}
        imageSampler={imgState.imageSampler}
        onImageSamplerChange={imgState.setImageSampler}
        imageRatioId={imgState.imageRatioId}
        imageWidth={imgState.imageWidth}
        onImageWidthChange={imgState.setImageWidth}
        imageHeight={imgState.imageHeight}
        onImageHeightChange={imgState.setImageHeight}
        imageSteps={imgState.imageSteps}
        onImageStepsChange={imgState.setImageSteps}
        imageGuidance={imgState.imageGuidance}
        onImageGuidanceChange={imgState.setImageGuidance}
        imageBatchSize={imgState.imageBatchSize}
        onImageBatchSizeChange={imgState.setImageBatchSize}
        imageUseRandomSeed={imgState.imageUseRandomSeed}
        onImageUseRandomSeedChange={imgState.setImageUseRandomSeed}
        imageSeedInput={imgState.imageSeedInput}
        onImageSeedInputChange={imgState.setImageSeedInput}
        imageOutputs={imgState.imageOutputs}
        recentImageOutputs={imgState.recentImageOutputs}
        onActiveTabChange={setActiveTab}
        onOpenImageStudio={imgState.openImageStudio}
        onOpenImageGallery={imgState.openImageGallery}
        onSubmitImageGeneration={() => void imgState.submitImageGeneration()}
        onApplyImageRatioPreset={imgState.applyImageRatioPreset}
        onApplyImageQuality={imgState.applyImageQuality}
        onPreloadImageModel={(variant) => void imgState.handlePreloadImageModel(variant)}
        onUnloadImageModel={(variant) => void imgState.handleUnloadImageModel(variant)}
        onInstallImageRuntime={() => imgState.handleInstallImageRuntime()}
        gpuBundleJob={imgState.gpuBundleJob}
        onImageDownload={(repo) => void imgState.handleImageDownload(repo)}
        onCancelImageDownload={(repo) => void imgState.handleCancelImageDownload(repo)}
        onDeleteImageDownload={(repo) => void imgState.handleDeleteImageDownload(repo)}
        onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
        onRestartServer={() => void handleRestartServer()}
        onUseSameImageSettings={imgState.handleUseSameImageSettings}
        onVaryImageSeed={(a) => void imgState.handleVaryImageSeed(a)}
        onRevealPath={(path) => void handleRevealPath(path)}
        onDeleteImageArtifact={(id) => void imgState.handleDeleteImageArtifact(id)}
      />
    );
  } else if (activeTab === "image-gallery") {
    content = (
      <ImageGalleryTab
        imageOutputs={imgState.imageOutputs}
        filteredImageOutputs={imgState.filteredImageOutputs}
        imageCatalog={imgState.imageCatalog}
        imageBusy={imgState.imageBusy}
        imageGallerySearchInput={imgState.imageGallerySearchInput}
        onImageGallerySearchInputChange={imgState.setImageGallerySearchInput}
        imageGalleryModelFilter={imgState.imageGalleryModelFilter}
        onImageGalleryModelFilterChange={imgState.setImageGalleryModelFilter}
        imageGalleryRuntimeFilter={imgState.imageGalleryRuntimeFilter}
        onImageGalleryRuntimeFilterChange={imgState.setImageGalleryRuntimeFilter}
        imageGalleryOrientationFilter={imgState.imageGalleryOrientationFilter}
        onImageGalleryOrientationFilterChange={imgState.setImageGalleryOrientationFilter}
        imageGallerySort={imgState.imageGallerySort}
        onImageGallerySortChange={imgState.setImageGallerySort}
        imageGalleryModelOptions={imgState.imageGalleryModelOptions}
        imageGalleryModelCount={imgState.imageGalleryModelCount}
        imageGalleryRealCount={imgState.imageGalleryRealCount}
        imageGalleryPlaceholderCount={imgState.imageGalleryPlaceholderCount}
        imageGalleryWarningCount={imgState.imageGalleryWarningCount}
        imageGalleryHasActiveFilters={imgState.imageGalleryHasActiveFilters}
        onActiveTabChange={setActiveTab}
        onOpenImageStudio={imgState.openImageStudio}
        onResetImageGalleryFilters={imgState.resetImageGalleryFilters}
        onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
        onUseSameImageSettings={imgState.handleUseSameImageSettings}
        onVaryImageSeed={(a) => void imgState.handleVaryImageSeed(a)}
        onRevealPath={(path) => void handleRevealPath(path)}
        onDeleteImageArtifact={(id) => void imgState.handleDeleteImageArtifact(id)}
      />
    );
  } else if (activeTab === "video-discover") {
    content = (
      <VideoDiscoverTab
        combinedVideoDiscoverResults={videoState.combinedVideoDiscoverResults}
        videoDiscoverSearchInput={videoState.videoDiscoverSearchInput}
        onVideoDiscoverSearchInputChange={videoState.setVideoDiscoverSearchInput}
        videoDiscoverTaskFilter={videoState.videoDiscoverTaskFilter}
        onVideoDiscoverTaskFilterChange={videoState.setVideoDiscoverTaskFilter}
        videoDiscoverSort={videoState.videoDiscoverSort}
        onVideoDiscoverSortChange={videoState.setVideoDiscoverSort}
        videoDiscoverHasActiveFilters={videoState.videoDiscoverHasActiveFilters}
        videoDiscoverSearchQuery={videoState.videoDiscoverSearchQuery}
        activeVideoDownloads={videoState.activeVideoDownloads}
        selectedVideoVariant={videoState.selectedVideoVariant}
        fileRevealLabel={fileRevealLabel}
        longLiveStatus={videoState.longLiveStatus}
        installingLongLive={videoState.installingLongLive}
        longLiveJob={videoState.longLiveJob}
        onActiveTabChange={setActiveTab}
        onOpenVideoStudio={videoState.openVideoStudio}
        onVideoDownload={(repo, modelId) => void videoState.handleVideoDownload(repo, modelId)}
        onCancelVideoDownload={(repo) => void videoState.handleCancelVideoDownload(repo)}
        onDeleteVideoDownload={(repo) => void videoState.handleDeleteVideoDownload(repo)}
        onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
        onRevealPath={(path) => void handleRevealPath(path)}
        onRefreshLongLiveStatus={() => void videoState.refreshLongLiveStatus()}
        onInstallLongLive={() => videoState.handleInstallLongLive()}
      />
    );
  } else if (activeTab === "video-models") {
    content = (
      <VideoModelsTab
        installedVideoVariants={videoState.installedVideoVariants}
        videoCatalog={videoState.videoCatalog}
        activeVideoDownloads={videoState.activeVideoDownloads}
        videoRuntimeStatus={videoState.videoRuntimeStatus}
        videoBusy={videoState.videoBusy}
        videoBusyLabel={videoState.videoBusyLabel}
        loadedVideoVariant={videoState.loadedVideoVariant}
        fileRevealLabel={fileRevealLabel}
        onActiveTabChange={setActiveTab}
        onOpenVideoStudio={videoState.openVideoStudio}
        onVideoDownload={(repo, modelId) => void videoState.handleVideoDownload(repo, modelId)}
        onCancelVideoDownload={(repo) => void videoState.handleCancelVideoDownload(repo)}
        onDeleteVideoDownload={(repo) => void videoState.handleDeleteVideoDownload(repo)}
        onPreloadVideoModel={(variant) => void videoState.handlePreloadVideoModel(variant)}
        onUnloadVideoModel={(variant) => void videoState.handleUnloadVideoModel(variant)}
        onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
        onRevealPath={(path) => void handleRevealPath(path)}
      />
    );
  } else if (activeTab === "video-studio") {
    content = (
      <VideoStudioTab
        videoCatalog={videoState.videoCatalogWithLatest}
        selectedVideoModelId={videoState.selectedVideoModelId}
        onSelectedVideoModelIdChange={videoState.setSelectedVideoModelId}
        selectedVideoVariant={videoState.selectedVideoVariant}
        selectedVideoFamily={videoState.selectedVideoFamily}
        selectedVideoLoaded={videoState.selectedVideoLoaded}
        selectedVideoWillLoadOnGenerate={videoState.selectedVideoWillLoadOnGenerate}
        videoRuntimeLoadedDifferentModel={videoState.videoRuntimeLoadedDifferentModel}
        loadedVideoVariant={videoState.loadedVideoVariant}
        videoRuntimeStatus={videoState.videoRuntimeStatus}
        tauriBackend={tauriBackend}
        busy={busy}
        busyAction={busyAction}
        videoBusy={videoState.videoBusy}
        videoBusyLabel={videoState.videoBusyLabel}
        backendOnline={backendOnline}
        activeVideoDownloads={videoState.activeVideoDownloads}
        videoPrompt={videoState.videoPrompt}
        onVideoPromptChange={videoState.setVideoPrompt}
        videoNegativePrompt={videoState.videoNegativePrompt}
        onVideoNegativePromptChange={videoState.setVideoNegativePrompt}
        videoUseRandomSeed={videoState.videoUseRandomSeed}
        onVideoUseRandomSeedChange={videoState.setVideoUseRandomSeed}
        videoSeedInput={videoState.videoSeedInput}
        onVideoSeedInputChange={videoState.setVideoSeedInput}
        videoWidth={videoState.videoWidth}
        onVideoWidthChange={videoState.setVideoWidth}
        videoHeight={videoState.videoHeight}
        onVideoHeightChange={videoState.setVideoHeight}
        videoNumFrames={videoState.videoNumFrames}
        onVideoNumFramesChange={videoState.setVideoNumFrames}
        videoFps={videoState.videoFps}
        onVideoFpsChange={videoState.setVideoFps}
        videoSteps={videoState.videoSteps}
        onVideoStepsChange={videoState.setVideoSteps}
        videoGuidance={videoState.videoGuidance}
        onVideoGuidanceChange={videoState.setVideoGuidance}
        videoUseNf4={videoState.videoUseNf4}
        onVideoUseNf4Change={videoState.setVideoUseNf4}
        videoEnableLtxRefiner={videoState.videoEnableLtxRefiner}
        onVideoEnableLtxRefinerChange={videoState.setVideoEnableLtxRefiner}
        videoEnhancePrompt={videoState.videoEnhancePrompt}
        onVideoEnhancePromptChange={videoState.setVideoEnhancePrompt}
        videoCfgDecay={videoState.videoCfgDecay}
        onVideoCfgDecayChange={videoState.setVideoCfgDecay}
        onActiveTabChange={setActiveTab}
        onPreloadVideoModel={(variant) => void videoState.handlePreloadVideoModel(variant)}
        onUnloadVideoModel={(variant) => void videoState.handleUnloadVideoModel(variant)}
        onVideoDownload={(repo, modelId) => void videoState.handleVideoDownload(repo, modelId)}
        onGenerateVideo={() => void videoState.handleVideoGenerate()}
        onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
        onRestartServer={() => void handleRestartServer()}
        onInstallVideoOutputDeps={(packages) => videoState.handleInstallVideoOutputDeps(packages)}
        onInstallVideoGpuRuntime={() => videoState.handleInstallVideoGpuRuntime()}
        longLiveStatus={videoState.longLiveStatus}
        installingLongLive={videoState.installingLongLive}
        onRefreshLongLiveStatus={() => void videoState.refreshLongLiveStatus()}
        onInstallLongLive={() => videoState.handleInstallLongLive()}
        longLiveJob={videoState.longLiveJob}
        mlxVideoStatus={videoState.mlxVideoStatus}
        installingMlxVideo={videoState.installingMlxVideo}
        onRefreshMlxVideoStatus={() => void videoState.refreshMlxVideoStatus()}
        onInstallMlxVideo={() => videoState.handleInstallMlxVideo()}
        gpuBundleJob={videoState.gpuBundleJob}
      />
    );
  } else if (activeTab === "video-gallery") {
    content = (
      <VideoGalleryTab
        videoOutputs={videoState.videoOutputs}
        videoBusy={videoState.videoBusy}
        onActiveTabChange={setActiveTab}
        onOpenVideoStudio={(modelId) => videoState.openVideoStudio(modelId)}
        onRevealPath={(path) => void handleRevealPath(path)}
        onDeleteVideoArtifact={(id) => void videoState.handleDeleteVideoOutput(id)}
      />
    );
  } else if (activeTab === "conversion") {
    content = (
      <ConversionTab
        conversionSource={conversionSource}
        conversionVariant={conversionVariant}
        conversionDraft={conversionDraft}
        lastConversion={lastConversion}
        conversionReady={conversionReady}
        convertibleLibrary={convertibleLibrary}
        nativeBackends={nativeBackends}
        preview={preview}
        workspace={{ system: workspace.system, library: chatLibrary }}
        launchCacheLabel={launchCacheLabel}
        busy={busy}
        busyAction={busyAction}
        conversionStartedAt={conversionStartedAt}
        conversionError={conversionError}
        showConversionPicker={showConversionPicker}
        showConversionModal={showConversionModal}
        fileRevealLabel={fileRevealLabel}
        onConversionDraftChange={updateConversionDraft}
        onConversionDraftReset={() =>
          setConversionDraft({
            modelRef: "", path: "", hfRepo: "", outputPath: "",
            quantize: true, qBits: 4, qGroupSize: 64, dtype: "float16",
          })
        }
        onConvertModel={() => void handleConvertModel()}
        onPickConversionOutputDir={() => void handlePickConversionOutputDir()}
        onShowConversionPickerChange={setShowConversionPicker}
        onShowConversionModalChange={setShowConversionModal}
        onPrepareLibraryConversion={prepareLibraryConversion}
        onRevealPath={(path) => void handleRevealPath(path)}
      />
    );
  } else if (activeTab === "chat") {
    content = (
      <ChatTab
        sortedChatSessions={sortedChatSessions}
        activeChat={activeChat}
        activeChatId={activeChatId}
        threadTitleDraft={threadTitleDraft}
        draftMessage={draftMessage}
        pendingImages={pendingImages}
        chatBusySessionId={chatBusySessionId}
        busy={busy}
        busyAction={busyAction}
        chatScrollRef={chatScrollRef}
        serverLoading={workspace.server.loading}
        loadedModelRef={workspace.runtime.loadedModel?.ref}
        loadedModelCapabilities={workspace.runtime.loadedModel?.capabilities ?? null}
        loadedModelEngine={workspace.runtime.loadedModel?.engine ?? null}
        engineLabel={workspace.runtime.engineLabel}
        launchSettings={launchSettings}
        warmModels={workspace.runtime.warmModels ?? []}
        activeThreadOptionKey={activeThreadOption?.key}
        onSetActiveChatId={setActiveChatId}
        onThreadTitleDraftChange={setThreadTitleDraft}
        thinkingMode={activeThinkingMode}
        runtimeProfileReady={activeRuntimeProfileMatchesLaunchSettings}
        onThinkingModeChange={handleThinkingModeChange}
        onDraftMessageChange={setDraftMessage}
        onPendingImagesChange={setPendingImages}
        onCreateSession={handleCreateSession}
        onToggleThreadPin={handleToggleThreadPin}
        onDeleteSession={handleDeleteSession}
        onRenameActiveThread={handleRenameActiveThread}
        onOpenModelSelector={openModelSelector}
        onLoadModel={handleLoadModel}
        onChatFileDrop={chat.handleChatFileDrop}
        onDeleteSessionDocument={deleteSessionDocument}
        onRefreshWorkspace={async (id?: string) => { await refreshWorkspace(id); }}
        onCopyMessage={chat.handleCopyMessage}
        onRetryMessage={chat.handleRetryMessage}
        onDeleteMessage={chat.handleDeleteMessage}
        onForkAtMessage={chat.handleForkAtMessage}
        onAddVariant={chat.handleAddVariant}
        onDelveMessage={chat.handleDelveMessage}
        onDetailsToggle={handleDetailsToggle}
        onSendMessage={sendMessage}
        onSetError={setError}
        enableTools={chat.enableTools}
        onToggleTools={chat.setEnableTools}
        onCompareMode={() => setCompareMode(true)}
        onCancelGeneration={chat.cancelGeneration}
        oneTurnOverride={chat.oneTurnOverride}
        onOneTurnOverrideChange={chat.setOneTurnOverride}
        availableCacheStrategies={workspace.system.availableCacheStrategies}
      />
    );
  } else if (activeTab === "server") {
    content = (
      <ServerTab
        serverStatus={workspace.server.status}
        serverPort={workspace.server.port}
        localServerUrl={localServerUrl}
        primaryLanUrl={primaryLanUrl}
        primaryLanOrigin={primaryLanOrigin}
        remoteAccessActive={remoteAccessActive}
        remoteAccessRequested={remoteAccessRequested}
        preferredPortUnavailable={preferredPortUnavailable}
        busyAction={busyAction}
        busy={busy}
        backendOnline={backendOnline}
        warmModels={workspace.runtime.warmModels ?? []}
        serverLoading={workspace.server.loading}
        requestsServed={workspace.server.requestsServed}
        activeConnections={workspace.server.activeConnections}
        engineLabel={workspace.runtime.engineLabel}
        settingsDraft={settingsDraft}
        serverLogEntries={serverLogEntries}
        showRemoteTest={showRemoteTest}
        testModelId={testModelId}
        apiToken={apiToken}
        localHealthCurl={localHealthCurl}
        localModelsCurl={localModelsCurl}
        remoteHealthCurl={remoteHealthCurl}
        remoteModelsCurl={remoteModelsCurl}
        recentOrphanedWorkers={workspace.server.recentOrphanedWorkers ?? []}
        selectedServerOptionKey={selectedServerOption?.key}
        onOpenModelSelector={openModelSelector}
        onRestartServer={handleRestartServer}
        onStopServer={handleStopServer}
        onLoadModel={handleLoadModel}
        onUnloadWarmModel={handleUnloadWarmModel}
        onSaveSettings={handleSaveSettings}
        onSettingsDraftChange={setSettingsDraft}
        onShowRemoteTestChange={setShowRemoteTest}
        onTestModelIdChange={setTestModelId}
      />
    );
  } else if (activeTab === "benchmarks") {
    content = (
      <BenchmarkRunTab
        workspace={{
          benchmarks: workspace.benchmarks,
          library: chatLibrary,
          system: {
            availableMemoryGb: workspace.system.availableMemoryGb,
            totalMemoryGb: workspace.system.totalMemoryGb,
            availableCacheStrategies: workspace.system.availableCacheStrategies,
            llamaServerTurboPath: workspace.system.llamaServerTurboPath,
            dflash: workspace.system.dflash,
          },
        }}
        threadModelOptions={threadModelOptions}
        benchmarkDraft={benchmarkDraft}
        benchmarkOption={benchmarkOption}
        benchmarkPromptId={benchmarkPromptId}
        preview={preview}
        busy={busy}
        busyAction={busyAction}
        benchmarkStartedAt={benchmarkStartedAt}
        benchmarkError={benchmarkError}
        showBenchmarkPicker={showBenchmarkPicker}
        showBenchmarkModal={showBenchmarkModal}
        installingPackage={installingPackage}
        installLogs={installLogs}
        onBenchmarkDraftChange={updateBenchmarkDraft}
        onBenchmarkPromptIdChange={setBenchmarkPromptId}
        onBenchmarkModelKeyChange={setBenchmarkModelKey}
        onBenchmarkDraftUpdate={setBenchmarkDraft}
        onRunBenchmark={() => void benchmarks.handleRunBenchmark(benchmarkOption)}
        onShowBenchmarkPickerChange={setShowBenchmarkPicker}
        onShowBenchmarkModalChange={setShowBenchmarkModal}
        onSelectedBenchmarkIdChange={setSelectedBenchmarkId}
        onCompareBenchmarkIdChange={setCompareBenchmarkId}
        onActiveTabChange={(tab) => setActiveTab(tab as TabId)}
        onInstallPackage={handleInstallPackage}
      />
    );
  } else if (activeTab === "benchmark-history") {
    content = (
      <BenchmarkHistoryTab
        benchmarks={workspace.benchmarks}
        benchmarkModelFilter={benchmarkModelFilter}
        benchmarkViewMode={benchmarkViewMode}
        selectedBenchmarkId={selectedBenchmarkId}
        compareBenchmarkId={compareBenchmarkId}
        onBenchmarkModelFilterChange={setBenchmarkModelFilter}
        onBenchmarkViewModeChange={setBenchmarkViewMode}
        onSelectedBenchmarkIdChange={setSelectedBenchmarkId}
        onCompareBenchmarkIdChange={setCompareBenchmarkId}
      />
    );
  } else if (activeTab === "finetuning") {
    content = <FineTuningTab backendOnline={backendOnline} />;
  } else if (activeTab === "prompt-library") {
    content = <PromptLibraryTab backendOnline={backendOnline} onApplyTemplate={(prompt) => { setSystemPrompt(prompt); setActiveTab("chat"); }} />;
  } else if (activeTab === "plugins") {
    content = <PluginsTab backendOnline={backendOnline} />;
  } else if (activeTab === "logs") {
    content = <LogsTab filteredLogs={filteredLogs} logQuery={logQuery} onLogQueryChange={setLogQuery} />;
  } else if (activeTab === "settings") {
    content = <SettingsTab
      settingsDraft={settingsDraft}
      onSettingsDraftChange={setSettingsDraft}
      newDirectoryLabel={newDirectoryLabel}
      onNewDirectoryLabelChange={setNewDirectoryLabel}
      newDirectoryPath={newDirectoryPath}
      onNewDirectoryPathChange={setNewDirectoryPath}
      onPickDataDirectory={handlePickDataDirectory}
      onPickImageOutputsDirectory={handlePickImageOutputsDirectory}
      onPickVideoOutputsDirectory={handlePickVideoOutputsDirectory}
      onSaveSettings={handleSaveSettings}
      onPickDirectory={pickDirectory}
      onAddDirectory={handleAddDirectory}
      onUpdateDirectoryPath={handleUpdateDirectoryPath}
      onToggleDirectory={handleToggleDirectory}
      onRemoveDirectory={handleRemoveDirectory}
      onCopyText={copyText}
      serverLocalhostUrl={workspace.server.localhostUrl}
      serverPort={workspace.server.port}
      loadedModelName={workspace.runtime.loadedModel?.name}
      apiToken={apiToken}
      sidebarMode={sidebarPrefs.mode}
      onSidebarModeChange={sidebarPrefs.setMode}
      backendOnline={backendOnline}
      onRestartServer={() => void handleRestartServer()}
      busyAction={busyAction}
    />;
  }

  const compareView = compareMode ? (
    <div style={{ display: activeTab === "chat" ? "block" : "none", height: "100%" }}>
      <CompareView
        modelOptions={libraryChatOptions}
        onBack={() => setCompareMode(false)}
        launchSettings={launchSettings}
        availableMemoryGb={workspace.system.availableMemoryGb}
        totalMemoryGb={workspace.system.totalMemoryGb}
        availableCacheStrategies={workspace.system.availableCacheStrategies}
        dflashInfo={workspace.system.dflash}
        turboInstalled={Boolean(workspace.system.llamaServerTurboPath)}
        onInstallPackage={handleInstallPackage}
        installingPackage={installingPackage}
        installLogs={installLogs}
      />
    </div>
  ) : null;

  return (
    <div className="app-shell">
      <Sidebar
        activeTab={activeTab}
        onTabChange={(tabId) => { setActiveTab(tabId); setError(null); }}
        platform={workspace.system.platform}
        appVersion={workspace.system.appVersion}
        backendOnline={backendOnline}
        engineLabel={workspace.runtime.engineLabel}
        loadedModelName={workspace.runtime.loadedModel?.name ?? null}
        mode={sidebarPrefs.mode}
        collapsedGroups={sidebarPrefs.collapsedGroups}
        onToggleGroupCollapsed={sidebarPrefs.toggleGroupCollapsed}
        lastChildByGroup={sidebarPrefs.lastChildByGroup}
        onRememberLastChild={sidebarPrefs.rememberLastChild}
      />


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
          {gpuStatus.showBanner && gpuStatus.status ? (
            <div className="notice-banner warn-banner">
              <span>
                <strong>Running on CPU.</strong>{" "}
                {gpuStatus.status.recommendation ??
                  "An NVIDIA GPU is visible but torch can't reach CUDA — image and video generation will be very slow."}
                {cudaTorchResult?.ok ? (
                  <>
                    {" "}
                    <strong>
                      CUDA torch installed{cudaTorchResult.indexUrl
                        ? ` from ${cudaTorchResult.indexUrl.replace("https://download.pytorch.org/whl/", "")}`
                        : ""}
                      {cudaTorchResult.pythonVersion ? ` into bundled Python ${cudaTorchResult.pythonVersion}` : ""}
                      . Restart the app to use the GPU.
                    </strong>
                  </>
                ) : cudaTorchResult && !cudaTorchResult.ok ? (
                  <>
                    {" "}
                    {cudaTorchResult.noWheelForPython ? (
                      <>
                        <strong>
                          No CUDA torch wheel for Python{cudaTorchResult.pythonVersion
                            ? ` ${cudaTorchResult.pythonVersion}`
                            : " (bundled venv)"}
                          .
                        </strong>{" "}
                        PyTorch currently ships CUDA wheels for Python 3.9–3.13. Reinstall the
                        app with a supported Python on PATH (e.g. 3.12 or 3.13) and try again.
                      </>
                    ) : (
                      <>
                        <strong>Install failed:</strong>{" "}
                        <span className="mono-text">{cudaTorchResult.message}</span>
                      </>
                    )}
                  </>
                ) : installingCudaTorch ? (
                  <>
                    {" "}
                    <em>Downloading ~2.5 GB CUDA wheel into the app's bundled Python — this can take several minutes.</em>
                  </>
                ) : null}
              </span>
              <button
                className="primary-button"
                type="button"
                disabled={installingCudaTorch || cudaTorchResult?.ok === true}
                onClick={() => void handleInstallCudaTorch()}
              >
                {installingCudaTorch
                  ? "Installing CUDA torch..."
                  : cudaTorchResult?.ok
                    ? "Installed"
                    : "Install CUDA torch"}
              </button>
              <button className="secondary-button" type="button" onClick={gpuStatus.dismiss}>
                Dismiss
              </button>
            </div>
          ) : null}
          {error ? (
            <div className="notice-banner error-banner">
              <span>{error}</span>
              {error.includes("update-llama-cpp.sh") ? (
                <button className="primary-button" type="button" disabled={rebuildingLlama} onClick={() => void handleRebuildLlamaCpp()}>
                  {rebuildingLlama ? "Rebuilding llama.cpp..." : "Rebuild llama.cpp"}
                </button>
              ) : null}
            </div>
          ) : null}
          {rebuildOutput ? (
            <div className="notice-banner">
              <pre className="mono-text rebuild-output">{rebuildOutput.slice(-2000)}</pre>
              <button className="secondary-button" type="button" onClick={() => setRebuildOutput(null)}>Dismiss</button>
            </div>
          ) : null}
        </div>

        {sidebarPrefs.mode === "tabs" ? (
          <SubtabBar
            activeTab={activeTab}
            onTabChange={(tabId) => { setActiveTab(tabId); setError(null); }}
            platform={workspace.system.platform}
            onRememberLastChild={sidebarPrefs.rememberLastChild}
          />
        ) : null}

        <div className="workspace-content-frame">
          {loading ? (
            <StartupProgressPanel
              elapsedSeconds={loadingElapsedSeconds}
              backendOnline={backendOnline}
              tauriBackend={tauriBackend}
            />
          ) : (
            <>
              {compareView}
              {(!compareMode || activeTab !== "chat") ? content : null}
            </>
          )}
        </div>
      </main>
      <LaunchModal
        pendingLaunch={pendingLaunch}
        launchModelSearch={launchModelSearch}
        libraryChatOptions={libraryChatOptions}
        threadModelOptions={threadModelOptions}
        launchSettings={launchSettings}
        preview={preview}
        availableMemoryGb={workspace.system.availableMemoryGb}
        totalMemoryGb={workspace.system.totalMemoryGb}
        availableCacheStrategies={workspace.system.availableCacheStrategies}
        dflashInfo={workspace.system.dflash}
        installingPackage={installingPackage}
        installLogs={installLogs}
        turboInstalled={Boolean(workspace.system.llamaServerTurboPath)}
        onPendingLaunchChange={setPendingLaunch}
        onLaunchModelSearchChange={setLaunchModelSearch}
        onLaunchSettingChange={updateLaunchSetting}
        onConfirmLaunch={(key) => void confirmLaunch(key)}
        onInstallPackage={handleInstallPackage}
      />
      <ImageGenerationModal
        showImageGenerationModal={imgState.showImageGenerationModal}
        imageBusy={imgState.imageBusy}
        imageGenerationStartedAt={imgState.imageGenerationStartedAt}
        imageGenerationError={imgState.imageGenerationError}
        imageGenerationCancelled={imgState.imageGenerationCancelled}
        imageGenerationCancelling={imgState.imageGenerationCancelling}
        imageGenerationArtifacts={imgState.imageGenerationArtifacts}
        selectedImageGenerationArtifact={imgState.selectedImageGenerationArtifact}
        imageGenerationRunInfo={imgState.imageGenerationRunInfo}
        imageCatalog={imgState.imageCatalog}
        imageSteps={imgState.imageSteps}
        selectedImageVariant={imgState.selectedImageVariant}
        onShowImageGenerationModalChange={imgState.setShowImageGenerationModal}
        onSelectedImageGenerationArtifactIdChange={imgState.setSelectedImageGenerationArtifactId}
        onActiveTabChange={setActiveTab}
        onUseSameSettings={imgState.handleUseSameImageSettings}
        onVarySeed={(a) => void imgState.handleVaryImageSeed(a)}
        onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
        onRevealPath={(path) => void handleRevealPath(path)}
        onDeleteArtifact={(id) => void imgState.handleDeleteImageArtifact(id)}
        onCancelGeneration={() => void imgState.handleCancelImageGeneration()}
      />
      <VideoGenerationModal
        showVideoGenerationModal={videoState.showVideoGenerationModal}
        videoBusy={videoState.videoBusy}
        videoGenerationStartedAt={videoState.videoGenerationStartedAt}
        videoGenerationError={videoState.videoGenerationError}
        videoGenerationCancelled={videoState.videoGenerationCancelled}
        videoGenerationCancelling={videoState.videoGenerationCancelling}
        videoGenerationArtifact={videoState.videoGenerationArtifact}
        videoGenerationRunInfo={videoState.videoGenerationRunInfo}
        selectedVideoVariant={videoState.selectedVideoVariant}
        onShowVideoGenerationModalChange={videoState.setShowVideoGenerationModal}
        onActiveTabChange={setActiveTab}
        onRevealPath={(path) => void handleRevealPath(path)}
        onDeleteArtifact={(id) => void videoState.handleDeleteVideoOutput(id)}
        onCancelGeneration={() => void videoState.handleCancelVideoGeneration()}
      />
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
                    const matchedLocal = findLibraryItemForVariant(chatLibrary, variant);
                    const downloadState = activeDownloads[variant.repo];
                    const isDownloading = downloadState?.state === "downloading";
                    const isDownloadPaused = downloadState?.state === "cancelled";
                    const isDownloadFailed = downloadState?.state === "failed";
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
                              <button className="secondary-button danger-button" type="button" onClick={() => void handleDeleteModelDownload(variant.repo)}>Cancel</button>
                            </>
                          ) : isDownloadPaused ? (
                            <>
                              <span className="badge warning">{downloadProgressLabel(downloadState)}</span>
                              <button className="secondary-button" type="button" onClick={() => void handleDownloadModel(variant.repo)}>Resume</button>
                              <button className="secondary-button danger-button" type="button" onClick={() => void handleDeleteModelDownload(variant.repo)}>Delete</button>
                            </>
                          ) : isDownloadFailed ? (
                            <>
                              <span className="badge warning">Download failed</span>
                              <button className="secondary-button" type="button" onClick={() => void handleDownloadModel(variant.repo)}>Retry</button>
                              <button className="secondary-button danger-button" type="button" onClick={() => void handleDeleteModelDownload(variant.repo)}>Delete</button>
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
                            onClick={(event) => { event.preventDefault(); void handleOpenExternalUrl(variant.link); }}
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
            <div className="modal-header"><h3>Restart required</h3></div>
            <div className="modal-body">
              <p>The data directory has been changed. ChaosEngineAI needs to restart the backend to start reading and writing files in the new location.</p>
              {dataDirRestartPrompt.migration ? (
                <div className="control-stack">
                  <p className="mono-text">From: {dataDirRestartPrompt.migration.from}</p>
                  <p className="mono-text">To: {dataDirRestartPrompt.migration.to}</p>
                  {dataDirRestartPrompt.migration.copied.length > 0 ? <p>Copied: {dataDirRestartPrompt.migration.copied.join(", ")}</p> : null}
                  {dataDirRestartPrompt.migration.skipped.length > 0 ? <p>Skipped (already present at destination): {dataDirRestartPrompt.migration.skipped.join(", ")}</p> : null}
                </div>
              ) : null}
            </div>
            <div className="modal-footer">
              <button className="secondary-button" type="button" onClick={() => setDataDirRestartPrompt(null)}>Not now</button>
              <button className="primary-button" type="button" onClick={async () => {
                setDataDirRestartPrompt(null);
                setBusyAction("Restarting backend...");
                try {
                  const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
                  await tauriInvoke("restart_backend_sidecar");
                  // Backend may still be starting — retry until healthy.
                  let online = await checkBackend();
                  if (!online) {
                    for (let i = 0; i < 5; i++) {
                      await new Promise((r) => setTimeout(r, 2000));
                      online = await checkBackend();
                      if (online) break;
                    }
                  }
                  if (online) {
                    setBackendOnline(true);
                    await refreshWorkspace(activeChatId || undefined);
                  } else {
                    setError("Backend did not come back online after restart.");
                  }
                } catch (err) {
                  setError(err instanceof Error ? err.message : "Failed to restart backend.");
                } finally {
                  setBusyAction(null);
                }
              }}>OK, restart now</button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
