import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { InfoTooltip } from "../../components/InfoTooltip";
import { Ltx2EngineBanner } from "../../components/Ltx2EngineBanner";
import { RecommendedSettingsPanel } from "../../components/RecommendedSettingsPanel";
import { ActionMenu, type ActionMenuItem } from "../../components/ActionMenu";
import { Disclosure } from "../../components/Disclosure";
import { RunEstimateStrip } from "../../components/RunEstimateStrip";
import { DisclosureMemoryContext, useDisclosureMemory } from "../../components/studioDisclosureMemory";
import { runEstimateBasis, summaryLine } from "../../components/studioSummary";
import { NO_ACK, ackKey, dangerBlocked, isAcked, type DangerAck } from "../../components/studioGating";
import { videoMenuItems } from "./videoMenu";
import { capacityLine as buildCapacityLine, formatCapacityGb } from "./videoCapacity";
import { useBusyAdvice } from "../../hooks/useBusyAdvice";
import { DEFAULT_VIDEO_NEGATIVE_PROMPT, DEFAULT_VIDEO_STEPS, recommendedGuidanceForRepo } from "../../hooks/videoStateDefaults";
import {
  DEFAULT_MAX_SIDE,
  clampStudioSide,
  fitStudioSize,
  recommendationRows,
  type RecommendationAppliers,
  type RecommendationCurrent,
} from "../../components/recommendedSettings";
import { VideoStudioRuntimeBanner } from "./VideoStudioRuntimeBanner";
import type { CudaTorchInstallResult, DownloadStatus, GpuBundleJobState, InstallResult, LongLiveJobState } from "../../api";
import type {
  TauriBackendInfo,
  VideoCacheStrategyId,
  VideoModelFamily,
  VideoModelVariant,
  VideoRuntimeStatus,
} from "../../types";
import {
  VIDEO_ASPECTS,
  VIDEO_RESOLUTIONS,
  fitsLimit,
  nearestAspect,
  nearestVideoResolution,
  sizeForShortEdge,
} from "../../utils/sizePresets";
import { InputImagePicker, type InputImage } from "../../components/InputImagePicker";
import type { ImageOutputArtifact } from "../../types/image";
import type { NativeBackendStatus } from "../../types/server";
import type { SystemStats } from "../../types/system";
import {
  hasNvidiaGpu,
  imageOrVideoVariantPlatformGate,
  isVariantCompatibleWithHost,
} from "../../utils/platform";
import {
  IMAGE_CACHE_STRATEGIES,
  VIDEO_CACHE_STRATEGY_DEFAULT_THRESH,
} from "../../constants";
import {
  assessVideoGenerationSafety,
  defaultVideoVariantForFamily,
  downloadProgressLabel,
  number,
  videoDownloadStatusForVariant,
  videoPrimarySizeLabel,
  videoSecondarySizeLabel,
} from "../../utils";
import {
  KNOWN_INSTALLABLE_VIDEO_DEPS,
  MLX_VIDEO_SUPPORTED_REPOS,
  QUALITY_PRESETS,
  type VideoQualityPreset,
  displayNumber,
  isLtx2DistilledRepo,
  onNumericBlur,
  onNumericChange,
} from "./videoStudioConstants";
import { modelsPageRequest, type TabRequest } from "../../constants/tabRedirects";

export interface VideoStudioTabProps {
  videoCatalog: VideoModelFamily[];
  /** GB held by loaded chat models; the memory estimate counts it. */
  residentChatModelGb?: number;
  /** Filter platform-incompatible variants out of the model dropdown
   * (FU-056 hide-unrecoverable-options policy). Threaded from App.tsx →
   * workspace.system. Optional for test harnesses. */
  hostSystem?: Pick<SystemStats, "platform" | "arch" | "gpuVramTotalGb">;
  selectedVideoModelId: string;
  onSelectedVideoModelIdChange: (id: string) => void;
  selectedVideoVariant: VideoModelVariant | null;
  selectedVideoFamily: VideoModelFamily | null;
  selectedVideoLoaded: boolean;
  selectedVideoWillLoadOnGenerate: boolean;
  videoRuntimeLoadedDifferentModel: boolean;
  loadedVideoVariant: VideoModelVariant | null;
  videoRuntimeStatus: VideoRuntimeStatus;
  tauriBackend: TauriBackendInfo | null;
  /** FU-056 Phase 4: capability snapshot for the runtime banner's
   * "Performance boosters" sub-section. Optional — defaults to
   * undefined when the parent's workspace probe hasn't reported yet. */
  nativeBackends?: NativeBackendStatus;
  busy: boolean;
  busyAction: string | null;
  videoBusy: boolean;
  videoBusyLabel: string | null;
  backendOnline: boolean;
  activeVideoDownloads: Record<string, DownloadStatus>;
  videoPrompt: string;
  onVideoPromptChange: (value: string) => void;
  videoNegativePrompt: string;
  videoStartImage: InputImage | null;
  onVideoStartImageChange: (next: InputImage | null) => void;
  videoEndImage: InputImage | null;
  onVideoEndImageChange: (next: InputImage | null) => void;
  imageGallery: ImageOutputArtifact[];
  onVideoNegativePromptChange: (value: string) => void;
  videoUseRandomSeed: boolean;
  onVideoUseRandomSeedChange: (value: boolean) => void;
  videoSeedInput: string;
  onVideoSeedInputChange: (value: string) => void;
  videoWidth: number;
  onVideoWidthChange: (value: number) => void;
  videoHeight: number;
  onVideoHeightChange: (value: number) => void;
  videoNumFrames: number;
  onVideoNumFramesChange: (value: number) => void;
  videoFps: number;
  onVideoFpsChange: (value: number) => void;
  videoSteps: number;
  onVideoStepsChange: (value: number) => void;
  videoGuidance: number;
  onVideoGuidanceChange: (value: number) => void;
  videoUseNf4: boolean;
  onVideoUseNf4Change: (value: boolean) => void;
  videoEnableLtxRefiner: boolean;
  onVideoEnableLtxRefinerChange: (value: boolean) => void;
  videoCfgDecay: boolean;
  onVideoCfgDecayChange: (value: boolean) => void;
  /** FU-018: TAESD/TAEHV preview-decode VAE swap. Off by default. */
  videoPreviewVae: boolean;
  onVideoPreviewVaeChange: (value: boolean) => void;
  /** FU-024: opt-in FP8 layerwise casting (CUDA SM 8.9+). */
  videoFp8LayerwiseCasting: boolean;
  onVideoFp8LayerwiseCastingChange: (value: boolean) => void;
  /** FU-015: diffusion cache strategy id ("none" / "fbcache" / "teacache"). */
  videoCacheStrategy: VideoCacheStrategyId;
  onVideoCacheStrategyChange: (value: VideoCacheStrategyId) => void;
  /** Optional caching threshold; null defers to strategy default. */
  videoCacheRelL1Thresh: number | null;
  onVideoCacheRelL1ThreshChange: (value: number | null) => void;
  videoStgScale: number;
  onVideoStgScaleChange: (value: number) => void;
  videoFastPreview: boolean;
  onVideoFastPreviewChange: (value: boolean) => void;
  onActiveTabChange: (tab: TabRequest) => void;
  onPreloadVideoModel: (variant: VideoModelVariant) => void;
  onUnloadVideoModel: (variant?: VideoModelVariant) => void;
  onVideoDownload: (repo: string, modelId?: string) => void;
  onGenerateVideo: () => void;
  onOpenExternalUrl: (url: string) => void;
  onRestartServer: () => void;
  onInstallVideoOutputDeps: (packages?: readonly string[]) => Promise<InstallResult>;
  onInstallVideoGpuRuntime: () => Promise<InstallResult>;
  /** Trigger /api/setup/install-cuda-torch directly from the GPU
   * acceleration warning banner. Lets the user fix the +cpu wheel
   * without navigating away to Settings > Setup. */
  onInstallCudaTorch?: () => void;
  installingCudaTorch?: boolean;
  /** Raw result from the most recent install attempt; drives the
   * collapsible terminal log under the Install button. */
  cudaTorchResult?: CudaTorchInstallResult | null;
  // LongLive (long-form causal video) surface — separate from the main
  // diffusers runtime because LongLive runs via a torchrun subprocess
  // against an isolated venv at ~/.chaosengine/longlive. Null until the
  // user selects a LongLive variant and the status is probed.
  longLiveStatus: VideoRuntimeStatus | null;
  installingLongLive: boolean;
  onRefreshLongLiveStatus: () => void;
  onInstallLongLive: () => Promise<InstallResult>;
  // Live state of the LongLive install job — drives the InstallLogPanel
  // beside the "Install LongLive" button so the user sees per-phase
  // progress (~9 phases over 10-20 minutes) rather than a static spinner.
  longLiveJob: LongLiveJobState | null;
  // mlx-video (Blaizzy) Apple Silicon engine probe (FU-009). Same
  // separation as LongLive — mlx-video runs as an MLX-native subprocess
  // (Wan2.1/2.2/LTX-2) rather than diffusers, so it gets a dedicated
  // chip + install action. Probe returns device="mps" on Apple Silicon
  // and device=null off-platform; chip hides off-platform.
  mlxVideoStatus: VideoRuntimeStatus | null;
  installingMlxVideo: boolean;
  onRefreshMlxVideoStatus: () => void;
  onInstallMlxVideo: () => Promise<InstallResult>;
  // Live state of the GPU bundle install job — drives the InstallLogPanel
  // under the install button so users see per-step pip output.
  gpuBundleJob: GpuBundleJobState | null;
}

// Pipeline-specific tokenizer / text-encoder packages that diffusers loads
// lazily — surfaced by the runtime probe via missingDependencies. Mirrors
// _VIDEO_MODEL_DEPS in backend_service/video_runtime.py so the Studio knows
// which "missing dep" chips it can offer a one-click install for.

export function VideoStudioTab({
  videoCatalog,
  hostSystem,
  residentChatModelGb,
  selectedVideoModelId,
  onSelectedVideoModelIdChange,
  selectedVideoVariant,
  selectedVideoFamily,
  selectedVideoLoaded,
  selectedVideoWillLoadOnGenerate,
  videoRuntimeLoadedDifferentModel,
  loadedVideoVariant,
  videoRuntimeStatus,
  tauriBackend,
  nativeBackends,
  busy,
  busyAction,
  videoBusy,
  videoBusyLabel,
  backendOnline,
  activeVideoDownloads,
  videoPrompt,
  onVideoPromptChange,
  videoNegativePrompt,
  videoStartImage,
  onVideoStartImageChange,
  videoEndImage,
  onVideoEndImageChange,
  imageGallery,
  onVideoNegativePromptChange,
  videoUseRandomSeed,
  onVideoUseRandomSeedChange,
  videoSeedInput,
  onVideoSeedInputChange,
  videoWidth,
  onVideoWidthChange,
  videoHeight,
  onVideoHeightChange,
  videoNumFrames,
  onVideoNumFramesChange,
  videoFps,
  onVideoFpsChange,
  videoSteps,
  onVideoStepsChange,
  videoGuidance,
  onVideoGuidanceChange,
  videoUseNf4,
  onVideoUseNf4Change,
  videoEnableLtxRefiner,
  onVideoEnableLtxRefinerChange,
  videoCfgDecay,
  onVideoCfgDecayChange,
  videoPreviewVae,
  onVideoPreviewVaeChange,
  videoFp8LayerwiseCasting,
  onVideoFp8LayerwiseCastingChange,
  videoCacheStrategy,
  onVideoCacheStrategyChange,
  videoCacheRelL1Thresh,
  onVideoCacheRelL1ThreshChange,
  videoStgScale,
  onVideoStgScaleChange,
  videoFastPreview,
  onVideoFastPreviewChange,
  onActiveTabChange,
  onPreloadVideoModel,
  onUnloadVideoModel,
  onVideoDownload,
  onGenerateVideo,
  onOpenExternalUrl,
  onRestartServer,
  onInstallVideoOutputDeps,
  onInstallVideoGpuRuntime,
  onInstallCudaTorch,
  installingCudaTorch,
  cudaTorchResult,
  longLiveStatus,
  installingLongLive,
  onRefreshLongLiveStatus,
  onInstallLongLive,
  longLiveJob,
  mlxVideoStatus,
  installingMlxVideo,
  onRefreshMlxVideoStatus,
  onInstallMlxVideo,
  gpuBundleJob,
}: VideoStudioTabProps) {
  const { t } = useTranslation("studio");
  const [installingOutputDeps, setInstallingOutputDeps] = useState(false);
  const [installingGpuRuntime, setInstallingGpuRuntime] = useState(false);
  // Opt-in acknowledgement that unlocks Generate when the safety heuristic
  // says "danger". We keep this behind an explicit checkbox because on
  // Apple Silicon an MPS kernel-panic can hard-reset the whole machine
  // (not just kill the sidecar) — one confirmed crash report from a 64 GB
  // M4 Max running Wan 2.2 A14B. The checkbox resets whenever the chosen
  // model, resolution, or frame count change so it's a per-configuration
  // override, not a permanent bypass.
  const [dangerAck, setDangerAck] = useState<DangerAck>(NO_ACK);
  const dangerKey = ackKey([selectedVideoVariant?.id, videoWidth, videoHeight, videoNumFrames]);
  const dangerOverrideAck = isAcked(dangerAck, dangerKey);
  const missingDependencies = videoRuntimeStatus.missingDependencies ?? [];
  // imageio + imageio-ffmpeg are the two pip packages diffusers video
  // pipelines need to export mp4s. Everything else we surface as a badge;
  // these two get a dedicated install button because they're the thing that
  // actually blocks "Generate video" from producing an output for ANY model.
  const mp4EncoderMissing = missingDependencies.some(
    (dep) => dep === "imageio" || dep === "imageio-ffmpeg",
  );
  // Hide once the runtime probe confirms torch/diffusers actually loaded —
  // the auto-restart at install completion may already have made the new
  // packages live, in which case ``requiresRestart`` from the install job
  // is stale. Without this check the banner stayed forever and clicking
  // Restart Backend appeared to do nothing because the badge never cleared.
  const gpuBundleRestartRequired =
    gpuBundleJob?.phase === "done"
    && gpuBundleJob.requiresRestart
    && !videoRuntimeStatus.realGenerationAvailable;
  // Tokenizer / text-encoder packages individual pipelines need lazily —
  // tiktoken for LTX-Video, sentencepiece for Wan / HunyuanVideo / CogVideoX
  // / Mochi, plus the protobuf + ftfy support libs. We list them out as a
  // single "Install missing video dependencies" button so the user doesn't
  // get a "tiktoken is required" mid-generate error after waiting on a long
  // model preload.
  const missingTokenizerDeps = missingDependencies.filter(
    (dep) => KNOWN_INSTALLABLE_VIDEO_DEPS.has(dep) && dep !== "imageio" && dep !== "imageio-ffmpeg",
  );
  const otherMissingDependencies = missingDependencies.filter(
    (dep) => !KNOWN_INSTALLABLE_VIDEO_DEPS.has(dep),
  );

  async function handleInstallOutputDeps() {
    if (installingOutputDeps) return;
    setInstallingOutputDeps(true);
    try {
      await onInstallVideoOutputDeps();
    } finally {
      setInstallingOutputDeps(false);
    }
  }

  async function handleInstallTokenizerDeps() {
    if (installingOutputDeps) return;
    if (missingTokenizerDeps.length === 0) return;
    setInstallingOutputDeps(true);
    try {
      await onInstallVideoOutputDeps(missingTokenizerDeps);
    } finally {
      setInstallingOutputDeps(false);
    }
  }

  // One-click install for the full GPU runtime (torch + diffusers + video
  // deps). Triggered when the probe reports the engine as unavailable —
  // i.e. torch or diffusers is missing from the persistent extras dir.
  async function handleInstallGpuRuntime() {
    if (installingGpuRuntime) return;
    setInstallingGpuRuntime(true);
    try {
      const result = await onInstallVideoGpuRuntime();
      if (result.ok && result.output.toLowerCase().includes("restart")) {
        onRestartServer();
      }
    } finally {
      setInstallingGpuRuntime(false);
    }
  }
  // Only offer variants the user can actually generate with. We include
  // models that are currently downloading because the user will want to keep
  // them selected while the download finishes. Everything else lives in
  // Discover / My Models.
  const studioFamilies = useMemo(
    () =>
      videoCatalog
        .map((family) => ({
          ...family,
          variants: family.variants.filter((variant) => {
            // FU-056: hide platform-incompatible variants entirely so
            // macOS users don't see CUDA-only entries and Win/Linux
            // users don't see mlx-video / mflux MLX-only entries.
            if (
              !isVariantCompatibleWithHost(
                imageOrVideoVariantPlatformGate(variant),
                hostSystem,
              )
            ) {
              return false;
            }
            if (variant.availableLocally) return true;
            if (variant.hasLocalData) return true;
            const downloadState = videoDownloadStatusForVariant(activeVideoDownloads, variant);
            return downloadState?.state === "downloading" || downloadState?.state === "completed";
          }),
        }))
        .filter((family) => family.variants.length > 0),
    [videoCatalog, activeVideoDownloads, hostSystem],
  );
  const hasAnyInstalled = studioFamilies.length > 0;

  // Ensure a valid model is selected once the catalog loads. "Valid" means
  // the model is present in ``studioFamilies`` — the installed / in-flight
  // subset the dropdown actually renders options for. Merely being in the
  // full ``videoCatalog`` isn't enough: a ``<select>`` whose ``value``
  // doesn't match any ``<option>`` silently shows the first option
  // visually while React state stays stale, which produces the classic
  // "dropdown says Wan 2.2 but every derived field still says LTX"
  // inconsistency after the previously-selected model is deleted.
  useEffect(() => {
    if (selectedVideoModelId) {
      const reachableFromDropdown = studioFamilies.some((family) =>
        family.variants.some((variant) => variant.id === selectedVideoModelId),
      );
      if (reachableFromDropdown) return;
    }
    const installed = studioFamilies[0]?.variants[0];
    if (installed?.id) {
      onSelectedVideoModelIdChange(installed.id);
      return;
    }
    const fallback = defaultVideoVariantForFamily(videoCatalog[0]);
    if (fallback?.id) onSelectedVideoModelIdChange(fallback.id);
  }, [selectedVideoModelId, videoCatalog, studioFamilies, onSelectedVideoModelIdChange]);

  // The danger-override acknowledgement belongs to one configuration (see ``dangerKey``): a user who
  // ticked "generate anyway" for a 720×480 × 33-frame Wan 2.2 run does not keep it when they bump
  // frames to 161, because the new configuration has its own risk profile.

  // Probe LongLive install state whenever the user selects a LongLive
  // variant so the Studio can surface an install callout without the
  // user having to click "generate" to find out the subprocess engine
  // isn't ready yet.
  const isLongLiveVariant =
    selectedVideoVariant?.repo?.startsWith("NVlabs/LongLive") ?? false;
  useEffect(() => {
    if (isLongLiveVariant) onRefreshLongLiveStatus();
  }, [isLongLiveVariant, onRefreshLongLiveStatus]);

  // Same probe-on-select pattern for mlx-video. Backend probe is
  // cheap (find_spec + platform check, no torch import) so refreshing
  // when the user picks a Wan/LTX variant gives the chip up-to-date
  // install state. Off-platform the probe returns ``device=null``
  // so the chip stays hidden — see render gate below.
  const isMlxVideoVariant =
    !!selectedVideoVariant?.repo && MLX_VIDEO_SUPPORTED_REPOS.has(selectedVideoVariant.repo);
  const isLtx2DistilledVariant = isLtx2DistilledRepo(selectedVideoVariant?.repo);
  // ltx-2-mlx runs a fixed distilled sampler: steps, guidance, the negative prompt, CFG decay,
  // the preview VAE and the diffusion cache do not exist for it, so they are not offered.
  const fixedSampler = Boolean(selectedVideoVariant?.runtime?.toLowerCase().includes("ltx-2-mlx"));
  const ltx2DevSibling = selectedVideoFamily?.variants.find(
    (variant) => variant.repo === selectedVideoVariant?.repo.replace(/-distilled$/i, "-dev"),
  ) ?? null;

  // FU-015 / FU-007: TeaCache patches only ship for FLUX, HunyuanVideo,
  // LTX-Video, CogVideoX, Mochi. Wan2.1 / Wan2.2 are deliberately
  // covered by FBCache instead (the diffusers 0.36 model-agnostic
  // hook) — the upstream ali-vilab teacache_generate.py targets the
  // standalone Wan-Video repo signature, not the diffusers
  // WanTransformer3DModel block layout, so a faithful TeaCache port
  // would need calibration table access we don't have. Hide the
  // TeaCache option for Wan repos so users don't pick it expecting a
  // win that won't materialise (the backend would just runtimeNote
  // "TeaCache not applied" and run the stock pipeline).
  //
  // The mlx-video subprocess path (LTX-2 / LTX-2.3) doesn't go
  // through diffusers cache hooks at all — it shells out to a
  // separate process. Hide both cache strategies there to avoid the
  // false promise.
  const selectedRepo = selectedVideoVariant?.repo ?? "";
  const isWanRepo = selectedRepo.startsWith("Wan-AI/");
  const isMlxVideoSubprocessPath =
    !!selectedRepo && MLX_VIDEO_SUPPORTED_REPOS.has(selectedRepo);
  const availableCacheStrategies = useMemo(() => {
    if (isMlxVideoSubprocessPath) {
      // Subprocess path doesn't see the diffusers transformer.
      return IMAGE_CACHE_STRATEGIES.filter((s) => s.id === "none");
    }
    if (isWanRepo) {
      // FBCache covers Wan; TeaCache patches don't.
      return IMAGE_CACHE_STRATEGIES.filter((s) => s.id !== "teacache");
    }
    return IMAGE_CACHE_STRATEGIES;
  }, [isMlxVideoSubprocessPath, isWanRepo]);

  // If the user previously picked TeaCache then switched to a Wan
  // variant (or to LTX-2 mlx-video), reset the strategy to "none"
  // so the dropdown reflects what's actually available. Avoids
  // submitting a stale id that the backend would have to swallow.
  useEffect(() => {
    const allowedIds = new Set(availableCacheStrategies.map((s) => s.id));
    if (!allowedIds.has(videoCacheStrategy)) {
      onVideoCacheStrategyChange("none");
    }
  }, [availableCacheStrategies, videoCacheStrategy, onVideoCacheStrategyChange]);
  // Fast-preview swap target: only the variants that opt-in via the
  // catalog's ``fastPreviewSiblingId`` field surface the toggle. Today
  // that's LTX-2 dev → distilled; any future model family can opt in
  // by setting the field. We look the sibling up so the toggle copy
  // can name the actual model that would render.
  const fastPreviewSibling =
    selectedVideoVariant?.fastPreviewSiblingId && selectedVideoFamily
      ? selectedVideoFamily.variants.find(
          (variant) => variant.id === selectedVideoVariant.fastPreviewSiblingId,
        ) ?? null
      : null;
  const fastPreviewActive = videoFastPreview && !!fastPreviewSibling;
  useEffect(() => {
    if (isMlxVideoVariant) onRefreshMlxVideoStatus();
  }, [isMlxVideoVariant, onRefreshMlxVideoStatus]);

  // Apple Silicon detection from the probe result rather than a
  // separate user-agent sniff — backend reports device="mps" or
  // expectedDevice="mps" only on Darwin arm64.
  const isAppleSiliconHost =
    mlxVideoStatus !== null
    && (mlxVideoStatus.device === "mps" || mlxVideoStatus.expectedDevice === "mps");
  // CUDA-only controls (NF4, FP8) need an NVIDIA GPU: hidden on a Mac and on a PC without one.
  const showCudaControls = hostSystem ? hasNvidiaGpu(hostSystem) : !isAppleSiliconHost;
  // Per-model size limit (``maxSide`` from the catalog; 4096 when the backend sends none).
  const activeVideoAspect = nearestAspect(videoWidth, videoHeight, VIDEO_ASPECTS);
  const activeVideoResolution = nearestVideoResolution(videoWidth, videoHeight);
  const videoMaxSide = selectedVideoVariant?.maxSide && selectedVideoVariant.maxSide > 0
    ? selectedVideoVariant.maxSide
    : DEFAULT_MAX_SIDE;
  const videoSizeLimit = { maxSide: videoMaxSide, maxPixels: selectedVideoVariant?.maxPixels ?? null };
  useEffect(() => {
    const fitted = fitStudioSize(videoWidth, videoHeight, { maxSide: videoMaxSide });
    if (fitted.width !== videoWidth) onVideoWidthChange(fitted.width);
    if (fitted.height !== videoHeight) onVideoHeightChange(fitted.height);
  }, [videoMaxSide, videoWidth, videoHeight, onVideoWidthChange, onVideoHeightChange]);

  // "Recommended for this machine": applied through the same setters as the controls
  // below, and only for settings this variant shows (ltx-2-mlx: size, frames and fps).
  const showVideoDiffusionCache = !fixedSampler && !isMlxVideoSubprocessPath;
  const recommendationAppliers = useMemo<RecommendationAppliers>(() => {
    const allowedCache = new Set(availableCacheStrategies.map((strategy) => strategy.id as string));
    return {
      width: (value) => onVideoWidthChange(clampStudioSide(value, { maxSide: videoMaxSide }, 832)),
      height: (value) => onVideoHeightChange(clampStudioSide(value, { maxSide: videoMaxSide }, 480)),
      numFrames: onVideoNumFramesChange,
      fps: onVideoFpsChange,
      steps: fixedSampler ? undefined : onVideoStepsChange,
      guidance: fixedSampler ? undefined : onVideoGuidanceChange,
      cfgDecay: fixedSampler ? undefined : onVideoCfgDecayChange,
      previewVae: fixedSampler ? undefined : onVideoPreviewVaeChange,
      cacheStrategy: showVideoDiffusionCache
        ? (value) => {
            if (allowedCache.has(value)) onVideoCacheStrategyChange(value as VideoCacheStrategyId);
          }
        : undefined,
      cacheThreshold: showVideoDiffusionCache ? onVideoCacheRelL1ThreshChange : undefined,
      fp8LayerwiseCasting: showCudaControls && !fixedSampler ? onVideoFp8LayerwiseCastingChange : undefined,
    };
  }, [
    availableCacheStrategies, videoMaxSide, fixedSampler, showVideoDiffusionCache, showCudaControls,
    onVideoWidthChange, onVideoHeightChange, onVideoNumFramesChange, onVideoFpsChange,
    onVideoStepsChange, onVideoGuidanceChange, onVideoCfgDecayChange, onVideoPreviewVaeChange,
    onVideoCacheStrategyChange, onVideoCacheRelL1ThreshChange, onVideoFp8LayerwiseCastingChange,
  ]);
  const recommendationCurrent = useMemo<RecommendationCurrent>(() => ({
    width: videoWidth,
    height: videoHeight,
    numFrames: videoNumFrames,
    fps: videoFps,
    steps: videoSteps,
    guidance: videoGuidance,
    cfgDecay: videoCfgDecay,
    previewVae: videoPreviewVae,
    cacheStrategy: videoCacheStrategy,
    cacheThreshold: videoCacheRelL1Thresh ?? VIDEO_CACHE_STRATEGY_DEFAULT_THRESH[videoCacheStrategy],
    fp8LayerwiseCasting: videoFp8LayerwiseCasting,
  }), [
    videoWidth, videoHeight, videoNumFrames, videoFps, videoSteps, videoGuidance, videoCfgDecay,
    videoPreviewVae, videoCacheStrategy, videoCacheRelL1Thresh, videoFp8LayerwiseCasting,
  ]);

  const mlxVideoMissing =
    !fixedSampler
    && isAppleSiliconHost
    && !mlxVideoStatus.realGenerationAvailable
    && (mlxVideoStatus.missingDependencies ?? []).includes("mlx-video");
  const mlxVideoInstalledScaffold =
    !fixedSampler
    && isAppleSiliconHost
    && !mlxVideoStatus.realGenerationAvailable
    && !(mlxVideoStatus.missingDependencies ?? []).includes("mlx-video");

  const downloadState = useMemo(
    () => (selectedVideoVariant ? videoDownloadStatusForVariant(activeVideoDownloads, selectedVideoVariant) : undefined),
    [activeVideoDownloads, selectedVideoVariant],
  );
  const isDownloading = downloadState?.state === "downloading";
  const isDownloaded =
    !!selectedVideoVariant && (selectedVideoVariant.availableLocally || downloadState?.state === "completed");
  const hasPrompt = videoPrompt.trim().length > 0;
  const selectedVideoRuntimeStatus: VideoRuntimeStatus = fixedSampler
    ? {
        ...videoRuntimeStatus,
        activeEngine: "ltx-2-mlx",
        device: "mlx",
        expectedDevice: "mlx",
        realGenerationAvailable: true,
        message: t("videoStudio.ltx2.engineMessage", {
          defaultValue: "This model runs on its own MLX engine (ltx-2-mlx) on the Apple GPU, with sound.",
        }),
      }
    : isMlxVideoVariant && mlxVideoStatus?.realGenerationAvailable
      ? {
          ...mlxVideoStatus,
          deviceMemoryGb: mlxVideoStatus.deviceMemoryGb ?? videoRuntimeStatus.deviceMemoryGb,
        }
      : videoRuntimeStatus;
  const generateButtonLabel =
    videoBusy && videoBusyLabel?.startsWith("Generating")
      ? videoBusyLabel
      : t("videoStudio.generateButton", { defaultValue: "Generate video" });
  // We compute the disable *reason* (not just the boolean) so the user can see
  // inline why a previous failure might have left the button in a stuck state —
  // the hover-only tooltip wasn't enough ("generate stays disabled after a Wan
  // crash" bug report, April 2026). ``null`` means enabled.
  // We defer the danger-safety check until AFTER ``generationSafety`` is
  // computed below — this variable is reassigned a few lines further down
  // to add "danger risk without explicit acknowledgement" to the chain.
  // Keeping the base chain readable here; see ``generateDisabledReason``
  // reassignment after ``generationSafety``.
  let generateDisabledReason: string | null = !selectedVideoVariant
    ? t("videoStudio.disabled.noModel", { defaultValue: "Choose a video model first." })
    : !isDownloaded
      ? t("videoStudio.disabled.notInstalled", {
          defaultValue: "{name} is not installed locally yet.",
          name: selectedVideoVariant.name,
        })
      : gpuBundleRestartRequired
        ? t("videoStudio.disabled.restartRequired", {
            defaultValue: "Restart the backend to activate the newly installed GPU runtime before generating.",
          })
      : !selectedVideoRuntimeStatus.realGenerationAvailable
        ? (selectedVideoRuntimeStatus.message || t("videoStudio.disabled.runtimeNotReady", { defaultValue: "Video runtime is not ready." }))
        : !hasPrompt
          ? t("videoStudio.disabled.noPrompt", { defaultValue: "Write a prompt before generating." })
          : !backendOnline
            ? t("videoStudio.disabled.backendOffline", { defaultValue: "Backend is offline." })
            : videoBusy
              ? (videoBusyLabel ?? t("videoStudio.disabled.busy", { defaultValue: "Busy…" }))
              : null;

  // Safety estimate for the chosen width × height × frames against the active
  // device. We surface this *before* the user hits Generate because on Apple
  // Silicon the failure mode is a hard sidecar crash (MPS assertion → Tauri
  // restart loop), not a graceful error — by the time the user sees "Load
  // failed" in the runtime status, the process has already died. See
  // ``assessVideoGenerationSafety`` for the heuristic and the bug it traces.
  //
  // We pass the selected variant's ``sizeGb`` through as ``baseModelFootprintGb``
  // so the estimate accounts for the dominant cost on MPS — weights + text
  // encoder + VAE sitting in unified memory — rather than estimating only
  // the attention kernel peak. Wan 2.1 T2V 1.3B is the key case: its 16 GB
  // on-disk footprint inflates to ~23 GB resident, which is the actual
  // reason it detonates 64 GB Macs at modest frame counts.
  const generationSafety = useMemo(
    () =>
      assessVideoGenerationSafety({
        width: videoWidth,
        height: videoHeight,
        numFrames: videoNumFrames,
        device: selectedVideoRuntimeStatus.device,
        deviceMemoryGb: selectedVideoRuntimeStatus.deviceMemoryGb,
        baseModelFootprintGb: selectedVideoVariant?.sizeGb,
        runtimeFootprintGb: selectedVideoVariant?.runtimeFootprintGb,
        runtimeFootprintMpsGb: selectedVideoVariant?.runtimeFootprintMpsGb,
        runtimeFootprintCudaGb: selectedVideoVariant?.runtimeFootprintCudaGb,
        runtimeFootprintCpuGb: selectedVideoVariant?.runtimeFootprintCpuGb,
        repo: selectedVideoVariant?.repo,
        useNf4: videoUseNf4 && !selectedVideoVariant?.ggufFile,
        otherResidentGb: residentChatModelGb,
        skipAttentionEstimate: fixedSampler,
      }),
    [
      fixedSampler,
      residentChatModelGb,
      videoWidth,
      videoHeight,
      videoNumFrames,
      selectedVideoRuntimeStatus.device,
      selectedVideoRuntimeStatus.deviceMemoryGb,
      selectedVideoVariant?.sizeGb,
      selectedVideoVariant?.runtimeFootprintGb,
      selectedVideoVariant?.runtimeFootprintMpsGb,
      selectedVideoVariant?.runtimeFootprintCudaGb,
      selectedVideoVariant?.runtimeFootprintCpuGb,
      selectedVideoVariant?.repo,
      selectedVideoVariant?.ggufFile,
      videoUseNf4,
    ],
  );

  // Danger-level runs are gated behind an explicit acknowledgement because
  // the failure mode on Apple Silicon is a hard MPS kernel panic that can
  // reset the whole machine, not just the sidecar. The base-reason chain
  // above covers "can't generate at all" conditions; this layer covers
  // "could generate but we think it will crash your computer". If the user
  // has ticked the override, we allow the generate — same UX pattern as
  // destructive-operation confirmations elsewhere in the app.
  if (generateDisabledReason === null && dangerBlocked(generationSafety.riskLevel, dangerOverrideAck)) {
    generateDisabledReason = t("videoStudio.disabled.dangerLevel", {
      defaultValue:
        "This configuration is likely to crash the backend. Tick \"Run anyway\" in the run estimate after reviewing the warning, or lower resolution/frames/model.",
    });
  }
  const generateTitle = generateDisabledReason ?? t("videoStudio.generateTitle", { defaultValue: "Start generating this clip." });
  const generationDisabled = generateDisabledReason !== null;

  // Format GB with one decimal for small numbers so 2.3 GB / 7.5 GB read
  // clearly, but drop the decimal once we're at 10+ (no user needs "14.0 GB").
  const formatGb = formatCapacityGb;

  // Whether the backend reported the device memory; when it did not, the strip's total is a
  // default and a one-line note under the strip says so.
  const backendReportedMemory =
    selectedVideoRuntimeStatus.deviceMemoryGb != null
    && Number.isFinite(selectedVideoRuntimeStatus.deviceMemoryGb)
    && selectedVideoRuntimeStatus.deviceMemoryGb > 0;
  // The device and capacity facts the old always-on line carried, now the memory item's tooltip.
  const capacityLine = buildCapacityLine(t as unknown as (key: string, options: Record<string, unknown>) => string, {
    device: selectedVideoRuntimeStatus.device,
    activeEngine: selectedVideoRuntimeStatus.activeEngine,
    effectiveDevice: generationSafety.effectiveDevice,
    deviceMemoryGb: generationSafety.deviceMemoryGb,
    memoryReported: backendReportedMemory,
    modelFootprintGb: generationSafety.modelFootprintGb,
    estimatedPeakGb: generationSafety.estimatedPeakGb,
  });

  const disclosureMemory = useDisclosureMemory("video");

  // Run estimate: the basis chip reads the same rows the recommended-settings panel shows.
  const estimateRows = recommendationRows(
    selectedVideoVariant?.availableLocally ? selectedVideoVariant.recommendedSettings : null,
    recommendationAppliers,
    recommendationCurrent,
  );
  const estimateBasis = runEstimateBasis(estimateRows, selectedVideoVariant?.estimatedGenerationSeconds != null);

  // Value summaries on the Options / Expert disclosures: only what differs from the model's defaults.
  const labelValue = (label: string, value: string | number): string =>
    t("studioKit.summary.labelValue", { defaultValue: "{label} {value}", label, value });
  const onOff = (on: boolean): string =>
    on ? t("recommended.on", { defaultValue: "On" }) : t("recommended.off", { defaultValue: "Off" });
  const defaultSteps = selectedVideoVariant?.defaultSteps ?? DEFAULT_VIDEO_STEPS;
  const defaultGuidance = selectedVideoVariant
    ? selectedVideoVariant.cfgOverride ?? recommendedGuidanceForRepo(selectedVideoVariant.repo)
    : null;
  const isLtxVideoRepo = selectedVideoVariant?.repo === "Lightricks/LTX-Video";
  const cacheLabel = t(`videoStudio.cacheStrategies.${videoCacheStrategy}.label`, {
    defaultValue: IMAGE_CACHE_STRATEGIES.find((strategy) => strategy.id === videoCacheStrategy)?.label ?? videoCacheStrategy,
  });
  const expertSummary = summaryLine(
    [
      !fixedSampler && videoCacheStrategy !== "none"
        && labelValue(
          t("videoStudio.diffusionCache.label", { defaultValue: "Diffusion cache" }),
          videoCacheRelL1Thresh != null
            ? t("recommended.cacheValue", { defaultValue: "{name}, threshold {threshold}", name: cacheLabel, threshold: videoCacheRelL1Thresh })
            : cacheLabel,
        ),
      !fixedSampler && !videoCfgDecay && labelValue(t("videoStudio.toggles.cfgDecay.label", { defaultValue: "CFG decay" }), onOff(false)),
      isMlxVideoVariant && !isLtx2DistilledVariant && videoStgScale !== 1
        && labelValue(t("videoStudio.stgScale.label", { defaultValue: "STG scale" }), videoStgScale),
      !fixedSampler && videoPreviewVae && labelValue(t("videoStudio.toggles.previewVae.label", { defaultValue: "Preview VAE" }), onOff(true)),
      showCudaControls && videoFp8LayerwiseCasting
        && labelValue(t("videoStudio.toggles.fp8Layerwise.label", { defaultValue: "FP8 layerwise (CUDA Ada+)" }), onOff(true)),
      showCudaControls && videoUseNf4 && labelValue(t("videoStudio.toggles.nf4.label", { defaultValue: "4-bit (NF4)" }), onOff(true)),
      isLtxVideoRepo && videoEnableLtxRefiner
        && labelValue(t("videoStudio.toggles.ltxRefiner.label", { defaultValue: "LTX two-stage spatial upscale" }), onOff(true)),
      fastPreviewActive && t("videoStudio.fastPreview.label", { defaultValue: "Fast preview" }),
      !fixedSampler && videoSteps !== defaultSteps && labelValue(t("videoStudio.fields.steps", { defaultValue: "Steps" }), videoSteps),
      !fixedSampler && defaultGuidance != null && videoGuidance !== defaultGuidance
        && labelValue(t("videoStudio.fields.guidance", { defaultValue: "Guidance" }), videoGuidance),
    ],
    t("disclosure.defaults", { defaultValue: "Model defaults" }),
  );
  const optionsSummary = summaryLine([
    t("studioKit.summary.size", { defaultValue: "{width} × {height}", width: videoWidth, height: videoHeight }),
    t("studioKit.summary.framesFps", { defaultValue: "{frames} frames at {fps} fps", frames: videoNumFrames, fps: videoFps }),
    videoUseRandomSeed
      ? t("videoStudio.fields.randomSeed", { defaultValue: "Random seed" })
      : labelValue(t("videoStudio.fields.seed", { defaultValue: "Seed" }), videoSeedInput || "?"),
    !fixedSampler && videoNegativePrompt !== DEFAULT_VIDEO_NEGATIVE_PROMPT
      && t("studioKit.summary.negativeEdited", { defaultValue: "Negative prompt edited" }),
    selectedVideoVariant?.acceptsImageInput && videoEndImage != null
      && t("studioKit.summary.endImageSet", { defaultValue: "End image set" }),
  ]);

  // The less-used model actions live in one menu (the header keeps Models).
  const headerMenuItems: ActionMenuItem[] = videoMenuItems(
    t as unknown as (key: string, options: Record<string, unknown>) => string,
    {
      selected: selectedVideoVariant,
      loadedVariant: loadedVideoVariant,
      isDownloaded,
      selectedLoaded: selectedVideoLoaded,
      videoBusy,
      realGenerationAvailable: videoRuntimeStatus.realGenerationAvailable,
    },
    { openLink: onOpenExternalUrl, preload: onPreloadVideoModel, unload: onUnloadVideoModel },
  );

  function handleApplySafeSettings(): void {
    const suggestion = generationSafety.suggestion;
    if (!suggestion) return;
    onVideoWidthChange(suggestion.width);
    onVideoHeightChange(suggestion.height);
    onVideoNumFramesChange(suggestion.numFrames);
  }

  const busyAdvice = useBusyAdvice(selectedVideoVariant ? generationSafety.estimatedPeakGb : null);

  return (
    <DisclosureMemoryContext.Provider value={disclosureMemory}>
    <div className="content-grid image-page-grid">
      <Panel
        title={t("video.title")}
        subtitle={selectedVideoVariant?.name ?? t("videoStudio.subtitle", { defaultValue: "Choose a video model to get started" })}
        className="span-2"
        actions={
          <div className="button-row">
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange(modelsPageRequest("video"))}>
              {t("studioKit.models", { defaultValue: "Models" })}
            </button>
            {headerMenuItems.length > 0 ? (
              <ActionMenu label={t("studioKit.moreActions", { defaultValue: "More actions" })} items={headerMenuItems} />
            ) : null}
          </div>
        }
      >
        <Ltx2EngineBanner modelNeedsLtx2={Boolean(selectedVideoVariant?.runtime?.toLowerCase().includes("ltx-2-mlx"))} />
        <VideoStudioRuntimeBanner
          hostSystem={hostSystem}
          videoRuntimeStatus={selectedVideoRuntimeStatus}
          loadedVideoVariant={loadedVideoVariant}
          busy={busy}
          busyAction={busyAction}
          backendOnline={backendOnline}
          onRestartServer={onRestartServer}
          onInstallCudaTorch={onInstallCudaTorch}
          installingCudaTorch={installingCudaTorch}
          cudaTorchResult={cudaTorchResult}
          gpuBundleRestartRequired={gpuBundleRestartRequired}
          isMlxVideoVariant={isMlxVideoVariant}
          isAppleSiliconHost={isAppleSiliconHost}
          isLongLiveVariant={isLongLiveVariant}
          isWanRepo={isWanRepo}
          selectedRepo={selectedRepo}
          mp4EncoderMissing={mp4EncoderMissing}
          mlxVideoMissing={mlxVideoMissing}
          mlxVideoInstalledScaffold={mlxVideoInstalledScaffold}
          missingTokenizerDeps={missingTokenizerDeps}
          otherMissingDependencies={otherMissingDependencies}
          longLiveStatus={longLiveStatus}
          longLiveJob={longLiveJob}
          installingLongLive={installingLongLive}
          onInstallLongLive={() => void onInstallLongLive()}
          mlxVideoStatus={mlxVideoStatus}
          installingMlxVideo={installingMlxVideo}
          onInstallMlxVideo={() => void onInstallMlxVideo()}
          installingOutputDeps={installingOutputDeps}
          installingGpuRuntime={installingGpuRuntime}
          gpuBundleJob={gpuBundleJob}
          onInstallOutputDeps={() => void handleInstallOutputDeps()}
          onInstallTokenizerDeps={() => void handleInstallTokenizerDeps()}
          onInstallGpuRuntime={() => void handleInstallGpuRuntime()}
          selectedVideoVariant={selectedVideoVariant}
          nativeBackends={nativeBackends}
        />

        <div className="image-studio-grid video-studio-top-grid" style={{ display: "grid", gap: "0.5rem", gridTemplateColumns: "1fr" }}>
          <label>
            {t("video.modelLabel", { defaultValue: "Video Model" })}
            {hasAnyInstalled ? (
              <select
                className="text-input"
                value={selectedVideoModelId}
                onChange={(event) => onSelectedVideoModelIdChange(event.target.value)}
              >
                {studioFamilies.flatMap((family) =>
                  family.variants.map((variant) => {
                    const downloadState = videoDownloadStatusForVariant(activeVideoDownloads, variant);
                    const isDownloadingVariant = downloadState?.state === "downloading";
                    const suffix = variant.availableLocally
                      ? ` (${t("videoStudio.modelOption.installed", { defaultValue: "installed" })})`
                      : isDownloadingVariant
                        ? ` (${downloadProgressLabel(downloadState)})`
                        : ` (${t("videoStudio.modelOption.incomplete", { defaultValue: "incomplete" })})`;
                    return (
                      <option key={variant.id} value={variant.id}>
                        {variant.name} — {family.name}
                        {suffix}
                      </option>
                    );
                  }),
                )}
              </select>
            ) : (
              <div className="callout image-callout">
                <p>{t("videoStudio.noModelsCallout.message", {
                  defaultValue: "No video models installed yet. Open Video > Models to download one.",
                })}</p>
                <div className="button-row">
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => onActiveTabChange(modelsPageRequest("video", "available"))}
                  >
                    {t("videoStudio.noModelsCallout.openDiscover", { defaultValue: "Open Video Models" })}
                  </button>
                </div>
              </div>
            )}
          </label>

          {selectedVideoVariant ? (
            <div className="image-library-stats">
              <span>{videoPrimarySizeLabel(selectedVideoVariant)}</span>
              {videoSecondarySizeLabel(selectedVideoVariant) ? (
                <span>{videoSecondarySizeLabel(selectedVideoVariant)}</span>
              ) : null}
              <span>{selectedVideoVariant.recommendedResolution}</span>
              <span>{t("videoStudio.stats.clipDuration", {
                defaultValue: "{value}s clip",
                value: number(selectedVideoVariant.defaultDurationSeconds),
              })}</span>
              <span className="badge subtle">{selectedVideoFamily?.name ?? selectedVideoVariant.provider}</span>
              {isDownloaded ? (
                <span className="badge success">{t("videoStudio.badges.installed", { defaultValue: "Installed" })}</span>
              ) : isDownloading ? (
                <span className="badge accent">{downloadProgressLabel(downloadState)}</span>
              ) : (
                <span className="badge warning" title={selectedVideoVariant.localStatusReason ?? undefined}>
                  {selectedVideoVariant.hasLocalData
                    ? t("videoStudio.badges.incomplete", { defaultValue: "Incomplete" })
                    : t("videoStudio.badges.notDownloaded", { defaultValue: "Not downloaded" })}
                </span>
              )}
              {selectedVideoLoaded ? <span className="badge accent">{t("videoStudio.badges.inMemory", { defaultValue: "In Memory" })}</span> : null}
              {videoRuntimeLoadedDifferentModel && loadedVideoVariant ? (
                <span className="badge muted">{t("videoStudio.badges.loadedModel", {
                  defaultValue: "Loaded model: {name}",
                  name: loadedVideoVariant.name,
                })}</span>
              ) : null}
            </div>
          ) : null}

          {selectedVideoVariant?.localStatusReason && !isDownloaded && !isDownloading ? (
            <p className="muted-text" style={{ color: "var(--warning, #f2c66d)" }}>
              {selectedVideoVariant.localStatusReason}
            </p>
          ) : null}

          <label>
            <span className="prompt-label-row">
              {t("videoStudio.prompt.label", { defaultValue: "Prompt" })}
            </span>
            <textarea
              className="text-input"
              rows={3}
              value={videoPrompt}
              onChange={(event) => onVideoPromptChange(event.target.value)}
              placeholder={t("videoStudio.prompt.placeholder", {
                defaultValue: "A cinematic drone shot of a misty pine forest at dawn...",
              })}
            />
            {fixedSampler ? (
              <p className="muted-text" role="note">
                {t("videoStudio.prompt.soundHint", {
                  defaultValue:
                    "This model makes sound too, and the sound follows your prompt. Describe it: footsteps crunching in snow, wind in the trees, a low engine hum.",
                })}
              </p>
            ) : null}
            {selectedVideoVariant?.repo === "Lightricks/LTX-Video"
              && hasPrompt
              && videoPrompt.trim().split(/\s+/).length < 25 ? (
              <p className="caution-text" role="note">
                {t("videoStudio.prompt.ltxShortWarning", {
                  defaultValue:
                    "LTX-Video produces best results with detailed prompts (~50-100 words). Short prompts (\"cartoon llama eating straw\") under-condition the model and tend to drift. Lightricks recommends starting with the action, then adding visual details, lighting, and camera direction.",
                })}
              </p>
            ) : null}
          </label>

          {selectedVideoVariant?.acceptsImageInput ? (
            <InputImagePicker
              label={t("inputImage.startLabel")}
              hint={t("inputImage.startHint")}
              images={videoStartImage ? [videoStartImage] : []}
              max={1}
              onChange={(next) => onVideoStartImageChange(next[0] ?? null)}
              gallery={imageGallery}
            />
          ) : null}

          {/*
            Quality preset pills. Jump straight to Draft/Standard/High/Max
            rather than making users learn what frames/steps mean for each
            model. Guidance stays model-aware (set in the hook) — presets
            intentionally don't overwrite it so LTX-at-3 / Hunyuan-at-6
            survive a preset click. Pill shows "active" when current state
            matches the preset exactly (so a user who tweaks a slider sees
            the active ring drop, confirming they're off-preset).

            The Quality preset and Aspect ratio pill groups sit inside a
            ``preset-row-pair`` flex container so they share a single row
            at typical Studio widths and wrap onto two lines on narrow
            workspaces. The label-on-top + pills layout inside each group
            is unchanged.
          */}
          <div className="preset-row-pair">
          {!fixedSampler ? (
            <div className="preset-row">
              <span className="preset-row-label">
                {t("video.qualityPreset", { defaultValue: "Quality preset" })}
                <InfoTooltip text={t("videoStudio.qualityPreset.tooltip", {
                  defaultValue:
                    "Sets the denoising step count. More steps = sharper frames + longer generation time. Frame count (clip length) and guidance stay as set — presets don't touch them.",
                })} />
              </span>
              {(Object.keys(QUALITY_PRESETS) as VideoQualityPreset[]).map((key) => {
                const preset = QUALITY_PRESETS[key];
                const active = videoSteps === preset.steps;
                return (
                  <button
                    key={key}
                    type="button"
                    className={`preset-pill ${active ? "active" : ""}`.trim()}
                    onClick={() => {
                      onVideoStepsChange(preset.steps);
                    }}
                  >
                    <span className="preset-pill-label">{t(`videoStudio.quality.${key}.label`, { defaultValue: preset.label })}</span>
                    <span className="preset-pill-sub">{t(`videoStudio.quality.${key}.sub`, { defaultValue: preset.sub })}</span>
                  </button>
                );
              })}
            </div>
          ) : null}

          {/*
            Aspect-ratio preset pills. Fixed resolutions (not "apply ratio
            to current base") so one click is fully deterministic. Values
            are all divisible by 8 and sit inside every supported model's
            tested envelope — safer than letting users pick arbitrary W×H.
          */}
          <div className="preset-row">
            <span className="preset-row-label">{t("sizePresets.aspect")}</span>
            {VIDEO_ASPECTS.map((aspect) => {
              const shortEdge = nearestVideoResolution(videoWidth, videoHeight)?.shortEdge ?? Math.min(videoWidth, videoHeight);
              const size = sizeForShortEdge(aspect, shortEdge);
              const ok = fitsLimit(size, videoSizeLimit);
              return (
                <button
                  key={aspect.id}
                  type="button"
                  className={`preset-pill ${activeVideoAspect?.id === aspect.id ? "active" : ""}`.trim()}
                  disabled={!ok}
                  title={ok ? `${size.width} x ${size.height}` : t("sizePresets.overLimit")}
                  onClick={() => {
                    onVideoWidthChange(size.width);
                    onVideoHeightChange(size.height);
                  }}
                >
                  <span className="preset-pill-label">{aspect.label}</span>
                </button>
              );
            })}
          </div>
          <div className="preset-row">
            <span className="preset-row-label">
              {t("sizePresets.resolution")}
              <InfoTooltip text={t("videoStudio.aspectRatio.tooltip", {
                defaultValue:
                  "Sets Width and Height from a common name such as HD or 4K, in the aspect ratio chosen above. Sizes past this model's limits are greyed out. A large size also needs a lot of memory: the Studio refuses a run that would not fit.",
              })} />
            </span>
            {VIDEO_RESOLUTIONS.map((resolution) => {
              const aspect = activeVideoAspect ?? VIDEO_ASPECTS[0];
              const size = sizeForShortEdge(aspect, resolution.shortEdge);
              const ok = fitsLimit(size, videoSizeLimit);
              return (
                <button
                  key={resolution.id}
                  type="button"
                  className={`preset-pill ${activeVideoResolution?.id === resolution.id ? "active" : ""}`.trim()}
                  disabled={!ok}
                  title={ok ? `${size.width} x ${size.height}` : t("sizePresets.overLimit")}
                  onClick={() => {
                    onVideoWidthChange(size.width);
                    onVideoHeightChange(size.height);
                  }}
                >
                  <span className="preset-pill-label">{resolution.label}</span>
                  <span className="preset-pill-sub">{resolution.sub} · {size.width}×{size.height}</span>
                </button>
              );
            })}
          </div>
          </div>

          {isLtx2DistilledVariant ? (
            <div className="callout quiet video-model-note" role="note">
              <p>
                <strong>{t("videoStudio.ltx2Distilled.title", {
                  defaultValue: "LTX-2 distilled is the fast sampler.",
                })}</strong>
                {" "}
                {t("videoStudio.ltx2Distilled.body", {
                  defaultValue:
                    "mlx-video runs it as fixed 8+3 denoise passes with CFG disabled, so the Steps and Guidance controls do not improve this variant. Use a dev variant for quality comparisons against the reference defaults.",
                })}
              </p>
              {ltx2DevSibling ? (
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => onSelectedVideoModelIdChange(ltx2DevSibling.id)}
                    disabled={videoBusy}
                  >
                    {t("videoStudio.ltx2Distilled.switchTo", {
                      defaultValue: "Switch to {name}",
                      name: ltx2DevSibling.name,
                    })}
                  </button>
                </div>
              ) : null}
            </div>
          ) : null}

          {selectedVideoVariant?.maxPixels
            && Number.isFinite(videoWidth) && Number.isFinite(videoHeight)
            && videoWidth * videoHeight > selectedVideoVariant.maxPixels ? (
            <p className="muted-text" role="note">
              {t("videoStudio.fields.maxPixelsNote", {
                defaultValue:
                  "{name} renders at most {megapixels} megapixels per frame; this size is reduced to fit, keeping its shape.",
                name: selectedVideoVariant.name,
                megapixels: (selectedVideoVariant.maxPixels / 1_000_000).toFixed(1),
              })}
            </p>
          ) : null}

          {Number.isFinite(videoNumFrames) && Number.isFinite(videoFps) && videoFps > 0 ? (
            <p className="muted-text" aria-live="polite">
              {t("videoStudio.clipLength.line", {
                defaultValue: "Clip length: {seconds}s ({frames} frames ÷ {fps} fps)",
                seconds: (videoNumFrames / videoFps).toFixed(2).replace(/\.?0+$/, ""),
                frames: videoNumFrames,
                fps: videoFps,
              })}
            </p>
          ) : null}

          {generationSafety.riskLevel !== "safe" ? (
            <div
              className={`callout image-callout ${
                generationSafety.riskLevel === "danger" ? "error" : "warning"
              }`}
              role="alert"
            >
              <p>
                <strong>
                  {generationSafety.riskLevel === "danger"
                    ? t("videoStudio.safety.dangerTitle", { defaultValue: "Likely to crash the backend" })
                    : t("videoStudio.safety.warningTitle", { defaultValue: "Heads up — may struggle on this device" })}
                  :
                </strong>{" "}
                {generationSafety.reason}
              </p>
              {generationSafety.suggestion ? (
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={handleApplySafeSettings}
                    disabled={videoBusy}
                    title={t("videoStudio.safety.applyTitle", {
                      defaultValue: "Apply {label}",
                      label: generationSafety.suggestion.label,
                    })}
                  >
                    {t("videoStudio.safety.useSafer", {
                      defaultValue: "Use safer settings ({label})",
                      label: generationSafety.suggestion.label,
                    })}
                  </button>
                </div>
              ) : generationSafety.riskLevel === "danger" ? (
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => onActiveTabChange(modelsPageRequest("video", "available"))}
                    disabled={videoBusy}
                  >
                    {t("videoStudio.safety.browseSmaller", { defaultValue: "Browse smaller models" })}
                  </button>
                </div>
              ) : null}
            </div>
          ) : null}

          {/*
            Pre-flight cost: time, peak memory of total and the danger-level
            acknowledgement sit directly above Generate. Danger-level runs stay
            disabled until the box is ticked; it resets on any change to
            variant / width / height / frames (see the dedicated useEffect that
            clears ``dangerOverrideAck``). Warning-level still generates freely.
          */}
          <div className="studio-shell__estimate">
            <RunEstimateStrip
              seconds={selectedVideoVariant?.estimatedGenerationSeconds ?? null}
              peakGb={selectedVideoVariant ? generationSafety.estimatedPeakGb : null}
              totalGb={selectedVideoVariant ? generationSafety.deviceMemoryGb : null}
              diskMb={null}
              basis={estimateBasis}
              risk={selectedVideoVariant ? generationSafety.riskLevel : "safe"}
              busy={busyAdvice}
              ack={{ checked: dangerOverrideAck, onChange: (checked) => setDangerAck({ key: dangerKey, checked }) }}
              memoryTitle={selectedVideoVariant ? capacityLine : undefined}
              action={
                <>
                  {selectedVideoVariant && !isDownloaded && !isDownloading ? (
                    <button
                      className="secondary-button"
                      type="button"
                      disabled={!backendOnline}
                      onClick={() => selectedVideoVariant && onVideoDownload(selectedVideoVariant.repo, selectedVideoVariant.id)}
                    >
                      {t("videoStudio.actions.downloadModel", { defaultValue: "Download model" })}
                    </button>
                  ) : null}
                  <button
                    className="primary-button"
                    type="button"
                    disabled={generationDisabled}
                    title={generateTitle}
                    onClick={() => onGenerateVideo()}
                  >
                    {generateButtonLabel}
                  </button>
                </>
              }
            />
            {videoBusy && videoBusyLabel ? (
              <p className="busy-indicator" role="status">
                <span className="busy-dot" />
                {videoBusyLabel}
              </p>
            ) : null}
            {selectedVideoWillLoadOnGenerate ? (
              <p className="muted-text">
                {t("videoStudio.willLoadHint", {
                  defaultValue:
                    "The selected model will be loaded into memory on the next generate. First load can take a minute for the larger variants.",
                })}
              </p>
            ) : null}
            {selectedVideoVariant && !backendReportedMemory ? (
              <p className="muted-text">
                {t("videoStudio.memory.defaultFallback", {
                  defaultValue: "~{value} (default — restart backend for real detection)",
                  value: formatGb(generationSafety.deviceMemoryGb),
                })}
              </p>
            ) : null}
          </div>

          {/*
            Make the disable reason visible even when the user isn't hovering
            the button. A failure-recovery flow that left the button stuck
            (real bug, April 2026) was only diagnosable via the tooltip, which
            is easy to miss — this turns the same string into an always-on
            callout so the root cause is obvious at a glance.
          */}
          {generateDisabledReason && !videoBusy ? (
            <p className="muted-text">{t("videoStudio.disabledHint", {
              defaultValue: "Generate disabled: {reason}",
              reason: generateDisabledReason,
            })}</p>
          ) : null}

          <div className="studio-shell__options">
            <Disclosure id="options" title={t("disclosure.options", { defaultValue: "Options" })} summary={optionsSummary}>
              {!fixedSampler ? (
                <label>
                  <span className="inline-label-text">
                    {t("video.negativePrompt", { defaultValue: "Negative prompt" })}
                    <InfoTooltip text={t("videoStudio.negativePrompt.tooltip", {
                      defaultValue:
                        "Tells the model what to avoid. A generic prompt is pre-filled and tuned for most video models — clear or edit it if you have a model-specific preference. More specificity usually helps more than it hurts.",
                    })} />
                  </span>
                  <input
                    className="text-input"
                    type="text"
                    value={videoNegativePrompt}
                    onChange={(event) => onVideoNegativePromptChange(event.target.value)}
                    placeholder={t("videoStudio.negativePrompt.placeholder", {
                      defaultValue: "Optional: things to avoid (low quality, watermark, etc.)",
                    })}
                  />
                </label>
              ) : null}

              {selectedVideoVariant?.acceptsImageInput ? (
                <InputImagePicker
                  label={t("inputImage.endLabel")}
                  hint={t("inputImage.endHint")}
                  images={videoEndImage ? [videoEndImage] : []}
                  max={1}
                  onChange={(next) => onVideoEndImageChange(next[0] ?? null)}
                  gallery={imageGallery}
                />
              ) : null}

              {/*
                Per-run knobs. We expose these because Wan 2.1 / LTX defaults at
                full resolution + step count can detonate Apple Silicon's MPS
                backend (the attention QK^T matrix scales with width × height ×
                num_frames squared — a 73 GB allocation killed the sidecar at
                832x480 × 96 frames × 50 steps during testing). Letting the user
                dial down resolution / frames / steps is the only way to keep
                consumer hardware in the safe envelope.

                ``numFrames`` step is 4 because Wan-family pipelines require
                ``(num_frames - 1) % 4 == 0``; the parent hook re-snaps on
                generate as a defensive backstop.
              */}
              <div className="field-grid image-field-grid">
                <label>
                  <span className="inline-label-text">
                    {t("videoStudio.fields.width", { defaultValue: "Width" })}
                    <InfoTooltip text={t("videoStudio.fields.widthTooltip", {
                      defaultValue:
                        "Horizontal resolution in pixels. Must be divisible by 8. Higher = sharper + slower + more VRAM. Try an Aspect ratio preset above for safe values.",
                    })} />
                  </span>
                  <input
                    className="text-input"
                    type="number"
                    min={256}
                    max={videoMaxSide}
                    step={64}
                    value={displayNumber(videoWidth)}
                    onChange={(event) => onNumericChange(event, onVideoWidthChange)}
                    onBlur={() => onVideoWidthChange(clampStudioSide(videoWidth, { maxSide: videoMaxSide }, 832))}
                  />
                </label>
                <label>
                  <span className="inline-label-text">
                    {t("videoStudio.fields.height", { defaultValue: "Height" })}
                    <InfoTooltip text={t("videoStudio.fields.heightTooltip", {
                      defaultValue:
                        "Vertical resolution in pixels. Must be divisible by 8. Higher = sharper + slower + more VRAM. Try an Aspect ratio preset above for safe values.",
                    })} />
                  </span>
                  <input
                    className="text-input"
                    type="number"
                    min={256}
                    max={videoMaxSide}
                    step={64}
                    value={displayNumber(videoHeight)}
                    onChange={(event) => onNumericChange(event, onVideoHeightChange)}
                    onBlur={() => onVideoHeightChange(clampStudioSide(videoHeight, { maxSide: videoMaxSide }, 480))}
                  />
                </label>
                <label>
                  <span className="inline-label-text">
                    {t("videoStudio.fields.frames", { defaultValue: "Frames" })}
                    <InfoTooltip text={t("videoStudio.fields.framesTooltip", {
                      defaultValue:
                        "How many frames to render. Wan / LTX require (frames-1) to be divisible by 4 — valid values are 1, 5, 9, 13, …, 161. Clip length in seconds = Frames ÷ FPS.",
                    })} />
                  </span>
                  <input
                    className="text-input"
                    type="number"
                    min={1}
                    max={257}
                    step={4}
                    value={displayNumber(videoNumFrames)}
                    onChange={(event) => onNumericChange(event, onVideoNumFramesChange)}
                    onBlur={() => onNumericBlur(videoNumFrames, onVideoNumFramesChange, 33)}
                  />
                </label>
                <label>
                  <span className="inline-label-text">
                    {t("videoStudio.fields.fps", { defaultValue: "FPS" })}
                    <InfoTooltip text={t("videoStudio.fields.fpsTooltip", {
                      defaultValue:
                        "Frames per second for playback. 24 is cinematic, 30 is smoother. Doesn't affect generation cost — only how fast the clip plays back.",
                    })} />
                  </span>
                  <input
                    className="text-input"
                    type="number"
                    min={1}
                    max={60}
                    value={displayNumber(videoFps)}
                    onChange={(event) => onNumericChange(event, onVideoFpsChange)}
                    onBlur={() => onNumericBlur(videoFps, onVideoFpsChange, 24)}
                  />
                </label>
              </div>

              <div className="button-row">
                <label className="inline-label" style={{ display: "flex", alignItems: "center", gap: ".4rem" }}>
                  <input
                    type="checkbox"
                    checked={videoUseRandomSeed}
                    onChange={(event) => onVideoUseRandomSeedChange(event.target.checked)}
                  />
                  {t("videoStudio.fields.randomSeed", { defaultValue: "Random seed" })}
                </label>
                {!videoUseRandomSeed ? (
                  <input
                    className="text-input"
                    type="number"
                    value={videoSeedInput}
                    onChange={(event) => onVideoSeedInputChange(event.target.value)}
                    placeholder={t("videoStudio.fields.seedPlaceholder", { defaultValue: "Seed (integer)" })}
                    style={{ maxWidth: 200 }}
                  />
                ) : null}
              </div>
            </Disclosure>

            <Disclosure id="expert" title={t("disclosure.expert", { defaultValue: "Expert" })} summary={expertSummary}>
              {selectedVideoVariant?.availableLocally ? (
                <RecommendedSettingsPanel
                  recommendation={selectedVideoVariant.recommendedSettings}
                  appliers={recommendationAppliers}
                  current={recommendationCurrent}
                  labelFor={(kind, id) =>
                    kind === "cacheStrategy"
                      ? t(`videoStudio.cacheStrategies.${id}.label`, {
                          defaultValue: IMAGE_CACHE_STRATEGIES.find((strategy) => strategy.id === id)?.label ?? id,
                        })
                      : id
                  }
                  disabled={videoBusy}
                />
              ) : null}

              {fastPreviewSibling ? (
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={fastPreviewActive}
                    onChange={(event) => onVideoFastPreviewChange(event.target.checked)}
                  />
                  <span>
                    <strong>{t("videoStudio.fastPreview.label", { defaultValue: "Fast preview" })}</strong>
                    {" · "}
                    {t("videoStudio.fastPreview.via", { defaultValue: "via" })}
                    {" "}
                    <span className="muted-text">{fastPreviewSibling.name}</span>
                    <InfoTooltip
                      text={t("videoStudio.fastPreview.tooltip", {
                        defaultValue:
                          "Renders this generation through {sibling} instead of {parent} using the same prompt + seed. Distilled fixed-step sampler — typically 6–9× faster than the full quality dev render. Untick when you want the dev variant's full quality.",
                        sibling: fastPreviewSibling.name,
                        parent: selectedVideoVariant?.name ?? t("videoStudio.fastPreview.devVariantFallback", { defaultValue: "the dev variant" }),
                      })}
                    />
                  </span>
                </label>
              ) : null}

              <div className="field-grid image-field-grid">
                {!fixedSampler ? (
                  <label>
                    <span className="inline-label-text">
                      {t("videoStudio.fields.steps", { defaultValue: "Steps" })}
                      <InfoTooltip text={t("videoStudio.fields.stepsTooltip", {
                        defaultValue:
                          "Denoising steps — how many passes the model makes to clean up noise into an image. More = sharper and more coherent, but linearly slower. 20 is draft quality, 30 is standard, 50+ is high quality with diminishing returns.",
                      })} />
                    </span>
                    <div className="slider-number-row">
                      <input
                        type="range"
                        min={1}
                        max={100}
                        step={1}
                        value={Number.isFinite(videoSteps) ? videoSteps : 30}
                        onChange={(event) => onVideoStepsChange(Number(event.target.value))}
                      />
                      <input
                        className="text-input"
                        type="number"
                        min={1}
                        max={100}
                        value={displayNumber(videoSteps)}
                        onChange={(event) => onNumericChange(event, onVideoStepsChange)}
                        onBlur={() => onNumericBlur(videoSteps, onVideoStepsChange, 30)}
                      />
                    </div>
                  </label>
                ) : null}
                {!fixedSampler ? (
                  <label>
                    <span className="inline-label-text">
                      {t("videoStudio.fields.guidance", { defaultValue: "Guidance" })}
                      <InfoTooltip text={t("videoStudio.fields.guidanceTooltip", {
                        defaultValue:
                          "How strongly the model follows your prompt. Too low = ignores the prompt; too high = rigid or distorted output. Recommended: LTX-Video ≈ 3, Wan ≈ 5, HunyuanVideo ≈ 6. The prompt's 'negative' direction comes from the Negative prompt above.",
                      })} />
                    </span>
                    <div className="slider-number-row">
                      <input
                        type="range"
                        min={1}
                        max={15}
                        step={0.5}
                        value={Number.isFinite(videoGuidance) ? videoGuidance : 5}
                        onChange={(event) => onVideoGuidanceChange(Number(event.target.value))}
                      />
                      <input
                        className="text-input"
                        type="number"
                        min={1}
                        max={20}
                        step={0.5}
                        value={displayNumber(videoGuidance)}
                        onChange={(event) => onNumericChange(event, onVideoGuidanceChange)}
                        onBlur={() => onNumericBlur(videoGuidance, onVideoGuidanceChange, 5)}
                      />
                    </div>
                    {selectedVideoVariant?.repo === "Lightricks/LTX-Video" && videoGuidance > 4 ? (
                      <p className="caution-text" role="alert">
                        {t("videoStudio.fields.guidanceLtxWarning", {
                          defaultValue:
                            "LTX-Video is a flow-matching model — CFG above ~3.5 over-saturates and produces blurred / rainbow output. Lower to 3 for the cleanest results.",
                        })}
                      </p>
                    ) : null}
                  </label>
                ) : null}
              </div>

              {selectedVideoVariant?.repo === "Lightricks/LTX-Video" ? (
                <p className="muted-text">
                  {t("videoStudio.ltxAutoTuneNote", {
                    defaultValue:
                      "Backend auto-tunes LTX decode parameters (frame_rate as model conditioning, decode_timestep, decode_noise_scale, guidance_rescale) to the Lightricks reference defaults — no extra sliders needed.",
                  })}
                </p>
              ) : null}

              {showCudaControls ? (
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={videoUseNf4}
                    onChange={(event) => onVideoUseNf4Change(event.target.checked)}
                  />
                  <span>
                    <strong>{t("videoStudio.toggles.nf4.label", { defaultValue: "4-bit (NF4)" })}</strong>
                    <InfoTooltip text={t("videoStudio.toggles.nf4.tooltip", {
                      defaultValue:
                        "bitsandbytes 4-bit weight quantization for the video DiT transformer. Fits Wan 2.1 14B in <24 GB VRAM with negligible quality loss. CUDA only — bitsandbytes ships no Metal kernels, so the toggle is ignored on macOS (MPS) and CPU. Stacks with First Block Cache for additional wall-time win.",
                    })} />
                  </span>
                </label>
              ) : null}

              {selectedVideoVariant?.repo === "Lightricks/LTX-Video" ? (
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={videoEnableLtxRefiner}
                    onChange={(event) => onVideoEnableLtxRefinerChange(event.target.checked)}
                  />
                  <span>
                    <strong>{t("videoStudio.toggles.ltxRefiner.label", {
                      defaultValue: "LTX two-stage spatial upscale",
                    })}</strong>
                    <InfoTooltip text={t("videoStudio.toggles.ltxRefiner.tooltip", {
                      defaultValue:
                        "Renders the base sample at the requested resolution, then refines through Lightricks/LTX-Video-0.9.5-spatial-upscaler at 2× spatial resolution. Frame budget grows ~1.5×. Sharper micro-detail and cleaner motion edges; off by default because the wall-time hit is real.",
                    })} />
                  </span>
                </label>
              ) : null}

              {!fixedSampler ? (
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={videoCfgDecay}
                    onChange={(event) => onVideoCfgDecayChange(event.target.checked)}
                  />
                  <span>
                    <strong>{t("videoStudio.toggles.cfgDecay.label", { defaultValue: "CFG decay" })}</strong>
                    <InfoTooltip text={t("videoStudio.toggles.cfgDecay.tooltip", {
                      defaultValue:
                        "Linearly drops guidance_scale from your slider value at step 0 toward 1.5 (the floor that keeps classifier-free guidance enabled end-to-end) at the final step. Flow-match video models (Wan, LTX, HunyuanVideo) oversaturate when CFG stays high throughout sampling; decay lets early steps lock semantics and late steps preserve fine detail. Default on for video — the runtime gates non-flow-match repos automatically.",
                    })} />
                  </span>
                </label>
              ) : null}

              {/*
                FU-018: TAESD/TAEHV preview-decode VAE swap. Off by
                default — video users typically want full fidelity.
                Backend maps the loaded repo to the matching tiny VAE
                (taew2_2 for Wan, taeltx2_3_wide for LTX, taehv1_5 for
                HunyuanVideo, taecogvideox / taemochi for the others);
                unmapped repos no-op silently.
              */}
              {!fixedSampler ? (
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={videoPreviewVae}
                    onChange={(event) => onVideoPreviewVaeChange(event.target.checked)}
                  />
                  <span>
                    <strong>{t("videoStudio.toggles.previewVae.label", { defaultValue: "Preview VAE" })}</strong>
                    <InfoTooltip text={t("videoStudio.toggles.previewVae.tooltip", {
                      defaultValue:
                        "Swaps the full VAE for the matching tiny VAE (madebyollin/taew2_2 for Wan, taeltx2_3_wide for LTX, taehv1_5 for HunyuanVideo, taecogvideox / taemochi for the others) so each step decodes in a fraction of the wall-time. Trades final fidelity for iteration speed. Off by default; backend silently no-ops on repos without a mapped tiny VAE.",
                    })} />
                  </span>
                </label>
              ) : null}

              {/*
                FU-024: FP8 layerwise casting on CUDA SM 8.9+ (Ada / Hopper /
                Blackwell). Halves transformer VRAM with negligible quality
                drift. Hidden entirely on Apple Silicon — there's no CUDA
                path so the toggle would be unreachable. Still rendered on
                Linux / Windows where the user may or may not have an Ada+
                GPU; backend silently skips on pre-Ada hardware.
              */}
              {showCudaControls ? (
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={videoFp8LayerwiseCasting}
                    onChange={(event) => onVideoFp8LayerwiseCastingChange(event.target.checked)}
                  />
                  <span>
                    <strong>{t("videoStudio.toggles.fp8Layerwise.label", { defaultValue: "FP8 layerwise (CUDA Ada+)" })}</strong>
                    <InfoTooltip text={t("videoStudio.toggles.fp8Layerwise.tooltip", {
                      defaultValue:
                        "diffusers' enable_layerwise_casting. Family-correct dtype: E5M2 for HunyuanVideo, E4M3 for Wan / LTX / FLUX / Qwen-Image. Backend checks GPU compute capability before applying — pre-Ada GPUs lack hardware fp8 and skip with a runtimeNote. Best stacked with the GGUF or Nunchaku quant paths for the smallest VRAM footprint.",
                    })} />
                  </span>
                </label>
              ) : null}

              {/*
                FU-015: diffusion cache strategy. First Block Cache works
                on every diffusers DiT pipeline (Wan / LTX / Hunyuan /
                Mochi / CogVideoX) regardless of platform — macOS (MPS),
                Windows (CUDA), Linux (CUDA / CPU). Hidden when the
                placeholder engine is active (no transformer to attach to)
                but otherwise always available. The mlx-video LTX-2
                subprocess path ignores the field because cache hooks
                attach to the diffusers transformer; the backend swallows
                the no-op silently.
              */}
              {!fixedSampler ? (
                <div className="control-stack">
                  <span className="eyebrow">
                    {t("videoStudio.diffusionCache.label", { defaultValue: "Diffusion cache" })}
                    <InfoTooltip text={t("videoStudio.diffusionCache.tooltip", {
                      defaultValue:
                        "Speed up generation by reusing transformer block outputs between similar timesteps. First Block Cache works on every DiT pipeline (Wan, LTX, Hunyuan, CogVideoX, Mochi) on macOS / Windows / Linux. TeaCache only applies to FLUX-family video models (Hunyuan / LTX / CogVideoX / Mochi) — hidden for Wan because the upstream patch targets a different transformer layout. The mlx-video LTX-2 subprocess path renders outside the diffusers hook system, so caching is unavailable there.",
                    })} />
                  </span>
                  <select
                    className="text-input"
                    value={videoCacheStrategy}
                    onChange={(event) =>
                      onVideoCacheStrategyChange(event.target.value as VideoCacheStrategyId)
                    }
                    disabled={isMlxVideoSubprocessPath}
                  >
                    {availableCacheStrategies.map((strategy) => (
                      <option key={strategy.id} value={strategy.id}>
                        {t(`videoStudio.cacheStrategies.${strategy.id}.label`, { defaultValue: strategy.label })}
                        {" · "}
                        {t(`videoStudio.cacheStrategies.${strategy.id}.hint`, { defaultValue: strategy.hint })}
                      </option>
                    ))}
                  </select>
                  {isMlxVideoSubprocessPath ? (
                    <span className="muted-text" style={{ fontSize: 11 }}>
                      {t("videoStudio.diffusionCache.mlxSubprocessNote", {
                        defaultValue:
                          "mlx-video LTX-2 runs as a subprocess outside the diffusers hook system — caching strategies are not available here. Switch to a diffusers Wan / LTX / Hunyuan variant to use First Block Cache.",
                      })}
                    </span>
                  ) : null}
                  {isWanRepo ? (
                    <span className="muted-text" style={{ fontSize: 11 }}>
                      {t("videoStudio.diffusionCache.wanNote", {
                        defaultValue:
                          "TeaCache hidden for Wan — its calibration tables target a different transformer layout. First Block Cache covers Wan via the diffusers 0.36 generic hook.",
                      })}
                    </span>
                  ) : null}
                  {videoCacheStrategy !== "none" ? (
                    <label className="control-stack-inline">
                      <span className="muted-text">
                        {t("videoStudio.diffusionCache.threshold", {
                          defaultValue: "Threshold ({value})",
                          value: videoCacheRelL1Thresh ??
                            VIDEO_CACHE_STRATEGY_DEFAULT_THRESH[videoCacheStrategy],
                        })}
                        <InfoTooltip text={t("videoStudio.diffusionCache.thresholdTooltip", {
                          defaultValue:
                            "Lower = stricter (less speedup, less quality drift). Higher = more aggressive caching. Video DiTs are more sensitive to drift than image DiTs, so the default is tighter (0.08 vs 0.12).",
                        })} />
                      </span>
                      <input
                        className="text-input"
                        type="number"
                        min={0.01}
                        max={0.6}
                        step={0.01}
                        value={
                          videoCacheRelL1Thresh ??
                          VIDEO_CACHE_STRATEGY_DEFAULT_THRESH[videoCacheStrategy]
                        }
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          onVideoCacheRelL1ThreshChange(
                            Number.isFinite(value) && value > 0 ? value : null,
                          );
                        }}
                      />
                    </label>
                  ) : null}
                </div>
              ) : null}

              {/*
                STG (Spatial-Temporal Guidance) — mlx-video LTX-2 only. Adds
                a perturbed forward pass per sampler step (skipping the
                final transformer blocks) that the backend mixes in to
                reduce object breakup / chroma drift. 1.0 = upstream's
                recommended quality default; 0.0 disables the perturbed
                pass, freeing ~33 % wall time per step on dev pipelines.
                Distilled pipelines run a fixed sampler that ignores the
                value; we still expose the slider on distilled so users see
                the cost they would pay if they switched. Hidden entirely
                for non-LTX-2 variants since other runtimes do not consume
                the flag.
              */}
              {isMlxVideoVariant ? (
                <label>
                  <span className="inline-label-text">
                    {t("videoStudio.stgScale.label", { defaultValue: "STG scale" })}
                    <InfoTooltip text={t("videoStudio.stgScale.tooltip", {
                      defaultValue:
                        "Spatial-Temporal Guidance. Adds an extra perturbed forward pass per sampler step on the LTX-2 dev MLX path to reduce object breakup and chroma drift. 1.0 matches upstream's recommended default; 0.0 disables for ~33% faster dev runs at a mild quality cost. Distilled pipelines run a fixed sampler and ignore the value.",
                    })} />
                  </span>
                  <div className="slider-number-row">
                    <input
                      type="range"
                      min={0}
                      max={3}
                      step={0.1}
                      value={Number.isFinite(videoStgScale) ? videoStgScale : 1}
                      onChange={(event) => onVideoStgScaleChange(Number(event.target.value))}
                      disabled={isLtx2DistilledVariant}
                    />
                    <input
                      className="text-input"
                      type="number"
                      min={0}
                      max={3}
                      step={0.1}
                      value={displayNumber(videoStgScale)}
                      onChange={(event) => {
                        const parsed = Number(event.target.value);
                        if (Number.isFinite(parsed)) {
                          onVideoStgScaleChange(Math.max(0, Math.min(3, parsed)));
                        }
                      }}
                      disabled={isLtx2DistilledVariant}
                    />
                  </div>
                  {isLtx2DistilledVariant ? (
                    <span className="muted-text" style={{ fontSize: 11 }}>
                      {t("videoStudio.stgScale.distilledNote", {
                        defaultValue: "Distilled pipelines run a fixed sampler — STG is ignored. Switch to a dev variant to use this knob.",
                      })}
                    </span>
                  ) : null}
                </label>
              ) : null}
            </Disclosure>
          </div>
        </div>
      </Panel>
    </div>
    </DisclosureMemoryContext.Provider>
  );
}
