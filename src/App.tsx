import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useTranslation } from "react-i18next";
import {
  checkBackend,
  convertModel,
  getVoiceRuntime,
  saveVoiceSettings,
  resolveApiToken,
  updateSession,
  updateSettings as updateSettingsApi,
} from "./api";
import { FirstLaunchLocaleBanner } from "./components/FirstLaunchLocaleBanner";
import { changeLocale, normaliseLocale, SUPPORTED_LOCALES, type SupportedLocale } from "./i18n";
import {
  performConvertModel,
  pickConversionOutputDir,
  prepareCatalogConversion as prepareCatalogConversionAction,
  prepareLibraryConversion as prepareLibraryConversionAction,
} from "./features/app/conversionActions";
import { performDeleteModel, performUnloadModel } from "./features/app/modelActions";
import { threadPatchFromVariant } from "./features/app/variantPayloads";
import { } from "./components/LaunchModal";
import { } from "./components/runtimeSupport";
import { } from "./components/ImageGenerationModal";
import { } from "./components/VideoGenerationModal";
import { Sidebar } from "./components/Sidebar";
import { StartupProgressPanel } from "./components/StartupProgressPanel";
import { SubtabBar } from "./components/SubtabBar";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { tabViewFor } from "./app/tabViews";
import { deriveModelSelections } from "./app/modelSelections";
import { WorkspaceHeader } from "./app/WorkspaceHeader";
import { StatusBanners } from "./app/StatusBanners";
import { LaunchModalHost, GenerationModals } from "./app/AppModals";
import { FamilyDetailModal } from "./app/FamilyDetailModal";
import { DataDirRestartModal } from "./app/DataDirRestartModal";
import type { AppTabContext, LoadModelPayload } from "./app/context";
import { buildLibraryChatOptions } from "./app/libraryChatOptions";
import { performLoadModel } from "./app/loadModelAction";
import {
  normalizePreselectedKey,
  performConfirmLaunch,
  sanitizeSpeculativeForSystem,
  type SpeculativeParams,
} from "./app/launchActions";

import type { OpenRequest as VoiceOpenRequest } from "./features/voice/VoiceStudioTab";
import { VoiceProvider, type VoiceContextValue } from "./features/voice/VoiceContext";
import type {
  ChatSession,
  LibraryItem,
  LoadModelActionResult,
  ModelVariant,
  TabId,
  VoiceRuntime,
} from "./types";
import type { ChatModelOption } from "./types/chat";
import { modelsAreaForTab, resolveTabRequest, tabs } from "./constants";
import type { TabRequest } from "./constants";
import type { ModelAvailabilityFilter, ModelsArea } from "./components/models/modelEntry";
import {
  firstDirectVariant,
  findVariantById,
  findVariantForReference,
  findLibraryItemForVariant,
  findCatalogVariantForLibraryItem,
  libraryItemBackend,
  inferHfRepoFromLocalPath,
  isChatLibraryItem,
  settingsDraftFromWorkspace,
  estimateParamsBFromDisk,
  previewModelControls,
  samePreviewModel,
  detectBitsPerWeight,
  isUnsavedEmptySession,
  isAppleSiliconHost,
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
  useUiScale,
  useGpuStatus,
  useCudaTorchInstall,
  useDetailsWindowResize,
  useFileActions,
} from "./hooks";
import { useMtplxInstall } from "./hooks/useMtplxInstall";
import { useTensorfoldInstall } from "./hooks/useTensorfoldInstall";
import type { TensorfoldLaunchControls } from "./components/tensorfoldSupport";
import { newStickyScroll, stickToBottom, updateStickyScroll, type StickyScroll } from "./utils/stickyScroll";

export default function App() {
  // FU-042: i18n hook — used for the workspace header tab label /
  // caption resolution and the first-launch locale banner.  Other
  // tab-level surfaces use their own ``useTranslation`` call to keep
  // this top-level component's dependency surface tight.
  const { t: tCommon } = useTranslation("common");
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

  const [activeTab, setActiveTabState] = useState<TabId>("dashboard");
  // Availability filter of each area's Models page (FU-129). Lives here so
  // it survives tab switches and so a navigation can preset it.
  const [modelsAvailability, setModelsAvailability] = useState<Record<ModelsArea, ModelAvailabilityFilter>>({
    chat: "all",
    image: "all",
    video: "all",
  });
  const setAreaAvailability = useCallback((area: ModelsArea, value: ModelAvailabilityFilter) => {
    setModelsAvailability((previous) => (previous[area] === value ? previous : { ...previous, [area]: value }));
  }, []);
  // The one tab setter every caller gets. Old Discover / My Models ids
  // redirect to the area's Models page with a preset filter
  // (``constants/tabRedirects.ts``).
  const setActiveTab = useCallback((request: TabRequest) => {
    const navigation = resolveTabRequest(request);
    if (!navigation) return;
    const area = modelsAreaForTab(navigation.tab);
    if (area && navigation.modelsPreset) setAreaAvailability(area, navigation.modelsPreset);
    setActiveTabState(navigation.tab);
  }, [setAreaAvailability]);
  const [apiToken, setApiToken] = useState<string | null>(null);
  const [voiceRuntime, setVoiceRuntime] = useState<VoiceRuntime | null>(null);
  const [voiceOpenRequest, setVoiceOpenRequest] = useState<VoiceOpenRequest | null>(null);
  const sidebarPrefs = useSidebarPrefs();
  const uiScalePrefs = useUiScale();
  const gpuStatus = useGpuStatus(backendOnline);
  const { i18n: i18nInstance } = useTranslation();

  // FU-042: hydrate the persisted locale from settings as soon as
  // ``getWorkspace`` resolves.  Boot-time ``initI18n`` only had the
  // navigator default; the saved ``settings.locale`` overrides it here
  // so the UI flips to the user's chosen language on cold start.
  useEffect(() => {
    if (loading) return;
    const persisted = ws.workspace.settings?.locale;
    if (!persisted || persisted === "system") return;
    const normalised = normaliseLocale(persisted) as SupportedLocale;
    if (!SUPPORTED_LOCALES.includes(normalised)) return;
    if (i18nInstance.language === normalised) return;
    void changeLocale(normalised);
  }, [loading, ws.workspace.settings?.locale, i18nInstance]);

  // ── Settings / Server / Preview ────────────────────────────
  const imgState = useImageState(backendOnline, setError, setActiveTab);
  const videoState = useVideoState(backendOnline, setError, setActiveTab);
  const {
    mtplxJob,
    installingMtplx,
    handleInstallMtplx,
  } = useMtplxInstall();
  // TensorFold engine install. A finished install re-reads the workspace so
  // the launch settings see ``system.tensorfold.available`` at once.
  const {
    tensorfoldJob,
    installingTensorfold,
    handleInstallTensorfold,
  } = useTensorfoldInstall(async () => {
    await refreshWorkspace();
  });

  const {
    installingCudaTorch,
    cudaTorchResult,
    cudaTorchRawResult,
    handleInstallCudaTorch,
  } = useCudaTorchInstall({
    onAfterInstall: async () => {
      // Both Studios subscribe to their own runtime probes via
      // useImageState / useVideoState; re-running them refreshes the
      // ``torchInstallWarning`` banner so it self-clears (or updates
      // with a new failure mode) without a backend restart.
      try {
        await imgState.refreshImageData();
      } catch {
        /* refresh is best-effort */
      }
      try {
        await videoState.refreshVideoData();
      } catch {
        /* refresh is best-effort */
      }
    },
  });
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
    conversionDraft, setConversionDraft,
    setLastConversion,
    systemPrompt,
    serverModelKey, setServerModelKey,
    updateConversionDraft,
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
    searchInput,
    searchResults,
    selectedFamilyId,
    selectedVariantId,
  } = models;

  // ── Derived model values ───────────────────────────────────
  const defaultChatVariant = firstDirectVariant(workspace.featuredModels);
  const activeFamilies = searchResults;
  const selectedFamily = activeFamilies.find((family) => family.id === selectedFamilyId) ?? activeFamilies[0] ?? null;
  const selectedVariant = findVariantById(activeFamilies, selectedVariantId) ?? (selectedFamily ? findVariantById([selectedFamily], selectedVariantId) : null);

  // ── Library state ──────────────────────────────────────────
  const chatLibrary = useMemo(() => workspace.library.filter(isChatLibraryItem), [workspace.library]);
  // Model the cache preview follows on the Chat Models page when it does
  // not show only the catalog (R18): the newest local model that matches
  // the page's search, as the old My Models tab did.
  const chatModelsLibraryVariant = useMemo(() => {
    const query = searchInput.trim().toLowerCase();
    const newest = chatLibrary
      .filter((item) => `${item.name} ${item.path} ${item.format} ${item.directoryLabel ?? ""}`.toLowerCase().includes(query))
      .sort((left, right) => {
        if (left.lastModified !== right.lastModified) return right.lastModified.localeCompare(left.lastModified);
        return right.sizeGb - left.sizeGb;
      })[0];
    return newest ? findCatalogVariantForLibraryItem(workspace.featuredModels, newest) : null;
  }, [chatLibrary, searchInput, workspace.featuredModels]);

  // ── Chat model options ─────────────────────────────────────
  const libraryChatOptions: ChatModelOption[] = buildLibraryChatOptions(chatLibrary, workspace.featuredModels, workspace.system);

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

  const tensorfoldControls: TensorfoldLaunchControls = {
    info: workspace.system.tensorfold,
    onInstall: (extras) => void handleInstallTensorfold(extras),
    installing: installingTensorfold,
    job: tensorfoldJob,
  };

  function sanitizeSpeculativeForModel(params: SpeculativeParams) {
    return sanitizeSpeculativeForSystem(workspace.system, params);
  }

  // ── Load model handler (stays in App — cross-domain) ───────
  async function handleLoadModel(payload: LoadModelPayload): Promise<LoadModelActionResult> {
    return performLoadModel(payload, {
      setError,
      setBusyAction,
      setWorkspace,
      refreshWorkspace,
      setActiveTab,
      launchSettings,
      sanitizeSpeculativeForModel,
      activeChatId: chat.activeChatId,
    });
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
    setThreadTitleDraft,
    chatBusySessionId,
    chatScrollRef,
    activeChat,
    sessionModelPayload,
    persistSessionChanges,
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
      modelPath: null,
      modelRepo: null,
      weightsGb: null,
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
  // FU-056 follow-up: derive once, thread to surfaces that gate
  // Apple-Silicon-only affordances (MTPLX in launch settings, MLX-LM
  // install panels, mlx-video install rows). Reads platform/arch from
  // the system probe — falls to ``false`` on early paint before the
  // probe lands, which is the safe default (don't flash MLX UI).
  const isAppleSilicon = isAppleSiliconHost(workspace.system);
  const previewSavings = Math.max(0, preview.baselineCacheGb - preview.optimizedCacheGb);
  const conversionReady = Boolean(nativeBackends?.converterAvailable ?? workspace.system.mlxLmAvailable);
  const enabledDirectoryCount = (workspace.settings?.modelDirectories ?? []).filter((directory) => directory.enabled).length;
  const fileRevealLabel =
    workspace.system.platform === "Darwin" ? tCommon("fileReveal.finder", { defaultValue: "Show in Finder" }) :
    workspace.system.platform === "Windows" ? tCommon("fileReveal.explorer", { defaultValue: "Show in Explorer" }) : tCommon("fileReveal.files", { defaultValue: "Show in Files" });

  const {
    activeThreadOption,
    loadedModelOption,
    selectedServerOption,
    convertibleLibrary,
    conversionSource,
    conversionVariant,
    benchmarkOption,
  } = deriveModelSelections({
    workspace, chatLibrary, threadModelOptions, libraryChatOptions, activeChat,
    serverModelKey, benchmarkModelKey, benchmarkDraft, conversionDraft,
  });
  const serverOptionKeySignature = threadModelOptions.map((option) => option.key).join("|");
  const previewVariant =
    activeTab === "server"
      ? findVariantForReference(workspace.featuredModels, selectedServerOption?.modelRef ?? workspace.runtime.loadedModel?.ref, selectedServerOption?.model ?? workspace.runtime.loadedModel?.name) ??
        findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ?? defaultChatVariant
      : activeTab === "chat-models" && modelsAvailability.chat !== "available"
        ? chatModelsLibraryVariant ?? findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ?? defaultChatVariant
      : activeTab === "chat"
        ? findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ??
          findVariantForReference(workspace.featuredModels, selectedServerOption?.modelRef ?? workspace.runtime.loadedModel?.ref, selectedServerOption?.model ?? workspace.runtime.loadedModel?.name) ?? defaultChatVariant
        : selectedVariant ?? findVariantForReference(workspace.featuredModels, activeChat?.modelRef, activeChat?.model) ?? defaultChatVariant;

  const bindingSettingsDirty =
    settingsDraft.preferredServerPort !== (workspace.settings?.preferredServerPort ?? 8876) ||
    settingsDraft.allowRemoteConnections !== (workspace.settings?.allowRemoteConnections ?? false);

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

  // Voice runtime — one-shot fetch when backend comes online.
  // Not polled aggressively; user can navigate to Voice Studio to refresh.
  const refreshVoiceRuntime = useCallback(() => {
    if (!backendOnline) return;
    void getVoiceRuntime()
      .then((rt) => setVoiceRuntime(rt))
      .catch(() => {});
  }, [backendOnline]);

  useEffect(() => {
    if (!backendOnline) {
      setVoiceRuntime(null);
      return;
    }
    refreshVoiceRuntime();
  }, [backendOnline, refreshVoiceRuntime]);

  const voiceContext = useMemo<VoiceContextValue>(
    () => ({
      runtime: voiceRuntime,
      platform: workspace.system.platform,
      refresh: refreshVoiceRuntime,
      saveSettings: async (patch) => {
        const settings = await saveVoiceSettings(patch);
        setVoiceRuntime((current) => (current ? { ...current, settings } : current));
      },
      openVoiceModels: () => void setActiveTab("voice-models"),
    }),
    [voiceRuntime, workspace.system.platform, refreshVoiceRuntime, setActiveTab],
  );

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

  // Sync previewVariant -> previewControls.paramsB + architecture
  // estimate. Bug surfaced 2026-05-05: this effect previously only
  // pushed paramsB and left numLayers / numHeads / numKvHeads /
  // hiddenSize at 0, which collapsed the Native f16 cache estimate
  // to ~0 bytes (kv_elements = num_layers * num_kv_heads * head_dim *
  // ctx — anything * 0 = 0) and made "Fits Easily" fire on models
  // that absolutely don't fit. Also pushed paramsB=0 cases through.
  const previewVariantPath = previewVariant ? findLibraryItemForVariant(chatLibrary, previewVariant)?.path ?? null : null;
  useEffect(() => {
    // While the launch modal is open it owns the preview (effect below);
    // closing it re-runs this so the tab's own model comes back.
    if (pendingLaunch || !previewVariant?.paramsB) return;
    const next = previewModelControls(
      previewVariant.paramsB, previewVariantPath, previewVariant.repo, previewVariant.sizeGb,
    );
    setPreviewControls((current) => (samePreviewModel(current, next) ? current : { ...current, ...next }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingLaunch === null, previewVariant?.paramsB, previewVariant?.repo, previewVariant?.sizeGb, previewVariantPath, setPreviewControls]);

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

  // Auto-scroll chat: follow the stream until the user scrolls up (see
  // utils/stickyScroll). A new chat or a new message turns the follow on.
  const chatSticky = useRef<StickyScroll>(newStickyScroll());
  const lastMessageLength = activeChat?.messages[activeChat.messages.length - 1]?.text?.length ?? 0;
  useEffect(() => {
    if (activeTab !== "chat") return;
    const el = chatScrollRef.current;
    if (!el) return;
    const onScroll = () => { updateStickyScroll(el, chatSticky.current); };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [activeTab, activeChat?.id, chatScrollRef]);
  useEffect(() => {
    chatSticky.current.stuck = true;
  }, [activeChat?.id, activeChat?.messages.length]);
  useEffect(() => {
    if (activeTab !== "chat") return;
    const handle = requestAnimationFrame(() => {
      const el = chatScrollRef.current;
      if (!el || !chatSticky.current.stuck) return;
      stickToBottom(el, chatSticky.current);
    });
    return () => cancelAnimationFrame(handle);
  }, [activeTab, activeChat?.id, activeChat?.messages.length, lastMessageLength, chatScrollRef]);

  // Re-anchor previewControls when benchmark/conversion model changes
  useEffect(() => {
    let label = "", sizeGb = 0, paramsB = 0;
    let modelPath: string | null = null;
    let modelRepo: string | null = null;
    if (activeTab === "benchmarks" && benchmarkOption) {
      label = `${benchmarkOption.label} ${benchmarkOption.format ?? ""} ${benchmarkOption.quantization ?? ""}`;
      sizeGb = benchmarkOption.sizeGb ?? 0;
      paramsB = benchmarkOption.paramsB ?? 0;
      modelPath = benchmarkOption.path ?? null;
      modelRepo = benchmarkOption.canonicalRepo ?? null;
    } else if (activeTab === "conversion" && conversionSource) {
      label = `${conversionSource.name} ${conversionSource.format ?? ""}`;
      sizeGb = conversionSource.sizeGb ?? 0;
      paramsB = conversionVariant?.paramsB ?? 0;
      modelPath = conversionSource.path;
      modelRepo = conversionVariant?.repo ?? null;
    } else {
      return;
    }
    if (!paramsB && sizeGb) paramsB = estimateParamsBFromDisk(sizeGb, detectBitsPerWeight(label));
    if (!paramsB) return;
    const next = previewModelControls(paramsB, modelPath, modelRepo, sizeGb);
    setPreviewControls((current) => (samePreviewModel(current, next) ? current : { ...current, ...next }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, benchmarkOption?.key, conversionSource?.path, conversionVariant?.id]);

  // Launch modal: re-anchor preview to the model the modal has selected.
  // Mirrors LaunchModal's own option list and default (the first option when
  // nothing is preselected); Discover's Chat / Server buttons preselect a
  // ``catalog:`` key that only exists in the full option list.
  const launchModalOpen = pendingLaunch !== null;
  useEffect(() => {
    if (!pendingLaunch) return;
    const key = pendingLaunch.preselectedKey;
    const preselectedIsCatalog = key?.startsWith("catalog:") ?? false;
    const options = preselectedIsCatalog || libraryChatOptions.length === 0 ? threadModelOptions : libraryChatOptions;
    const opt = key
      ? options.find((o) => o.key === key) ?? threadModelOptions.find((o) => o.key === key)
      : options[0];
    if (!opt) return;
    let paramsB = opt.paramsB ?? 0;
    if (!paramsB && opt.sizeGb) {
      const bpw = detectBitsPerWeight(`${opt.label} ${opt.format ?? ""} ${opt.quantization ?? ""}`);
      paramsB = estimateParamsBFromDisk(opt.sizeGb, bpw);
    }
    if (!paramsB) return;
    const repo = opt.canonicalRepo ?? (opt.modelRef.includes("/") ? opt.modelRef : null);
    const next = previewModelControls(paramsB, opt.path ?? null, repo, opt.sizeGb);
    setPreviewControls((current) => (samePreviewModel(current, next) ? current : { ...current, ...next }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [launchModalOpen, pendingLaunch?.preselectedKey, libraryChatOptions, threadModelOptions]);

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
    return performUnloadModel(ref, {
      setBusyAction,
      setError,
      setWorkspace,
      refreshWorkspace,
      activeChatId,
    });
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

  const { handleRevealPath, handleOpenFilePath, handleOpenExternalUrl } = useFileActions(backendOnline, setError);

  async function handleDeleteModel(item: LibraryItem) {
    return performDeleteModel(item, {
      setBusyAction,
      setError,
      setWorkspace,
      refreshWorkspace,
      activeChatId,
    });
  }

  function prepareCatalogConversion(model: ModelVariant) {
    return prepareCatalogConversionAction(model, {
      convertibleLibrary,
      setConversionDraft,
      setLastConversion,
      setActiveTab,
    });
  }

  function prepareLibraryConversion(item: LibraryItem, resolvedPath?: string) {
    return prepareLibraryConversionAction(item, resolvedPath, {
      setConversionDraft,
      setLastConversion,
      setActiveTab,
    });
  }

  async function handleConvertModel() {
    return performConvertModel({
      conversionDraft,
      setError,
      setBusyAction,
      setConversionStartedAt,
      setConversionError,
      setShowConversionModal,
      setLastConversion,
      setWorkspace,
      refreshWorkspace,
      activeChatId,
    });
  }

  async function handlePickConversionOutputDir() {
    return pickConversionOutputDir({
      conversionSource,
      setError,
      updateConversionDraft,
    });
  }

  function threadPatchFromVariantLocal(variant: ModelVariant) {
    return threadPatchFromVariant(variant, {
      chatLibrary,
      launchSettings,
      launchCacheLabel,
      sanitizeSpeculativeForModel,
    });
  }

  async function handleApplyVariantToActiveThread(variant: ModelVariant) {
    if (!activeChat) return;
    await persistSessionChanges(activeChat.id, threadPatchFromVariantLocal(variant));
    setActiveTab("chat");
  }

  async function handleStartThreadWithVariant(variant: ModelVariant) {
    const localSession: ChatSession = {
      id: `draft-${Date.now()}`,
      title: "New chat",
      pinned: false,
      messages: [],
      ...threadPatchFromVariantLocal(variant),
    };
    setWorkspace((current) => ({
      ...current,
      chatSessions: [
        localSession,
        ...current.chatSessions.filter((session) => !isUnsavedEmptySession(session)),
      ],
    }));
    setActiveChatId(localSession.id);
    setThreadTitleDraft(localSession.title);
    setActiveTab("chat");
  }

  function openModelSelector(action: "chat" | "server" | "thread", preselectedKey?: string) {
    setLaunchModelSearch("");
    const normalizedKey = normalizePreselectedKey(preselectedKey, {
      featuredModels: workspace.featuredModels,
      chatLibrary,
      threadModelOptions,
      loadedModelOption,
    });
    setPendingLaunch({ action, preselectedKey: normalizedKey });
  }

  async function confirmLaunch(selectedKey: string) {
    return performConfirmLaunch(selectedKey, {
      pendingLaunch,
      threadModelOptions,
      launchSettings,
      previewControls,
      featuredModels: workspace.featuredModels,
      activeChat,
      setPendingLaunch,
      setError,
      setActiveTab,
      handleStartThreadWithVariant,
      handleLoadModel,
      persistSessionChanges,
      sanitizeSpeculativeForModel,
    });
  }

  function copyText(text: string) {
    void navigator.clipboard.writeText(text);
  }

  const { handleDetailsToggle } = useDetailsWindowResize();

  // ── Tab context ────────────────────────────────────────────
  const ctx: AppTabContext = {
    workspace, backendOnline, tauriBackend, busy, busyAction, setError, refreshWorkspace, setActiveTab,
    imgState, videoState, models, chat, benchmarks, settings, sidebarPrefs, uiScalePrefs,
    mtplxJob, installingMtplx, handleInstallMtplx, tensorfoldControls,
    installingCudaTorch, cudaTorchRawResult, handleInstallCudaTorch,
    chatLibrary, libraryChatOptions, threadModelOptions, activeThreadOption, selectedServerOption,
    benchmarkOption, convertibleLibrary, conversionSource, conversionVariant,
    modelsAvailability, setAreaAvailability, nativeBackends, isAppleSilicon, enabledDirectoryCount,
    fileRevealLabel, conversionReady, launchCacheLabel,
    handleRevealPath, handleOpenFilePath, handleOpenExternalUrl, handleLoadModel, handleUnloadModel,
    handleUnloadWarmModel, handleDeleteModel, prepareCatalogConversion, prepareLibraryConversion,
    handleConvertModel, handlePickConversionOutputDir, openModelSelector, handleDetailsToggle, copyText,
    conversionStartedAt, conversionError, showConversionPicker, setShowConversionPicker,
    showConversionModal, setShowConversionModal,
    apiToken, showRemoteTest, setShowRemoteTest, testModelId, setTestModelId, logQuery, setLogQuery,
    voiceRuntime, voiceOpenRequest, setVoiceOpenRequest, refreshVoiceRuntime,
  };

  // ── Tab content ────────────────────────────────────────────
  const ActiveView = tabViewFor(activeTab);
  const content = <ActiveView ctx={ctx} />;

  return (
    <div
      className="app-shell"
      style={{ "--ui-scale": uiScalePrefs.uiScale } as CSSProperties}
    >
      <FirstLaunchLocaleBanner
        persistedLocale={workspace.settings?.locale}
        onPersistLocale={(locale) => {
          // Fire-and-forget — we don't await because the banner already
          // mutates the live i18n state via ``changeLocale`` and sets the
          // localStorage dismissal flag.  This PATCH only ensures the
          // choice survives a cold restart; a network failure here is
          // self-healing (the banner just re-appears next launch).
          void updateSettingsApi({ locale });
        }}
      />
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
        keepAllOpen={sidebarPrefs.keepAllOpen}
        onToggleGroupCollapsed={sidebarPrefs.toggleGroupCollapsed}
        lastChildByGroup={sidebarPrefs.lastChildByGroup}
        onRememberLastChild={sidebarPrefs.rememberLastChild}
      />


      <main className="workspace">
        <WorkspaceHeader
          activeTab={activeTab}
          system={workspace.system}
          cacheLabel={workspace.runtime.loadedModel ? loadedModelCacheLabel : launchCacheLabel}
        />

        <StatusBanners
          gpuStatus={gpuStatus}
          cudaTorchResult={cudaTorchResult}
          installingCudaTorch={installingCudaTorch}
          handleInstallCudaTorch={handleInstallCudaTorch}
          error={error}
          rebuildingLlama={rebuildingLlama}
          handleRebuildLlamaCpp={handleRebuildLlamaCpp}
          rebuildOutput={rebuildOutput}
          setRebuildOutput={setRebuildOutput}
        />

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
            // FU-037: per-tab ErrorBoundary so an uncaught render error
            // in one tab no longer blanks the whole workspace. ``key``
            // is the active tab id so switching tabs unmounts the
            // boundary and gives the user a clean recovery path.
            <ErrorBoundary key={activeTab} scope={tabs.find((tab) => tab.id === activeTab)?.label ?? activeTab}>
              <VoiceProvider value={voiceContext}>{content}</VoiceProvider>
            </ErrorBoundary>
          )}
        </div>
      </main>
      <LaunchModalHost
        ctx={ctx}
        pendingLaunch={pendingLaunch}
        launchModelSearch={launchModelSearch}
        setPendingLaunch={setPendingLaunch}
        setLaunchModelSearch={setLaunchModelSearch}
        confirmLaunch={confirmLaunch}
      />
      <GenerationModals ctx={ctx} />
      <FamilyDetailModal ctx={ctx} />
      <DataDirRestartModal
        dataDirRestartPrompt={dataDirRestartPrompt}
        setDataDirRestartPrompt={setDataDirRestartPrompt}
        setBusyAction={setBusyAction}
        setBackendOnline={setBackendOnline}
        setError={setError}
        refreshWorkspace={refreshWorkspace}
        activeChatId={activeChatId}
      />
    </div>
  );
}
