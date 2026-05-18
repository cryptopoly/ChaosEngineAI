import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { InfoTooltip } from "../../components/InfoTooltip";
import { ImageOutputCard } from "../../components/ImageOutputCard";
import { PromptEnhanceButton } from "../../components/PromptEnhanceButton";
import { ImageStudioRuntimeBanner } from "./ImageStudioRuntimeBanner";
import type { CudaTorchInstallResult, DownloadStatus, GpuBundleJobState, InstallResult } from "../../api";
import type {
  ImageCacheStrategyId,
  ImageModelFamily,
  ImageModelVariant,
  ImageOutputArtifact,
  ImageQualityPreset,
  ImageSamplerId,
  ImageRuntimeStatus,
  TabId,
  TauriBackendInfo,
} from "../../types";
import type { NativeBackendStatus } from "../../types/server";
import type { SystemStats } from "../../types/system";
import {
  sizeLabel,
  downloadProgressLabel,
  formatImageAccessError,
  isGatedImageAccessError,
} from "../../utils";
import {
  imageOrVideoVariantPlatformGate,
  isVariantCompatibleWithHost,
} from "../../utils/platform";
import { assessImageGenerationSafety, imageVariantSizeForMemoryEstimate } from "../../utils/images";
import {
  IMAGE_RATIO_PRESETS,
  IMAGE_QUALITY_PRESETS,
  IMAGE_SAMPLERS,
  IMAGE_CACHE_STRATEGY_DEFAULT_THRESH,
  imageCacheStrategiesForRepo,
  isFlowMatchingRepo,
  isUnetImageRepo,
} from "../../constants";

export interface ImageStudioTabProps {
  imageCatalog: ImageModelFamily[];
  selectedImageModelId: string;
  onSelectedImageModelIdChange: (id: string) => void;
  selectedImageVariant: ImageModelVariant | null;
  selectedImageFamily: ImageModelFamily | null;
  selectedImageLoaded: boolean;
  selectedImageWillLoadOnGenerate: boolean;
  imageRuntimeLoadedDifferentModel: boolean;
  loadedImageVariant: ImageModelVariant | null;
  imageRuntimeStatus: ImageRuntimeStatus;
  tauriBackend: TauriBackendInfo | null;
  busy: boolean;
  busyAction: string | null;
  imageBusy: boolean;
  imageBusyLabel: string | null;
  backendOnline: boolean;
  /** FU-056 Phase 3: capability snapshot used by the runtime banner's
   * Performance boosters sub-section to gate Install / Installed pills
   * on the accelerator cards. */
  nativeBackends?: NativeBackendStatus;
  activeImageDownloads: Record<string, DownloadStatus>;
  imagePrompt: string;
  onImagePromptChange: (value: string) => void;
  imageNegativePrompt: string;
  onImageNegativePromptChange: (value: string) => void;
  imageQualityPreset: ImageQualityPreset;
  imageRatioId: (typeof IMAGE_RATIO_PRESETS)[number]["id"];
  imageWidth: number;
  onImageWidthChange: (value: number) => void;
  imageHeight: number;
  onImageHeightChange: (value: number) => void;
  imageSteps: number;
  onImageStepsChange: (value: number) => void;
  imageGuidance: number;
  onImageGuidanceChange: (value: number) => void;
  imageBatchSize: number;
  onImageBatchSizeChange: (value: number) => void;
  imageUseRandomSeed: boolean;
  onImageUseRandomSeedChange: (value: boolean) => void;
  imageSeedInput: string;
  onImageSeedInputChange: (value: string) => void;
  imageOutputs: ImageOutputArtifact[];
  recentImageOutputs: ImageOutputArtifact[];
  onActiveTabChange: (tab: TabId) => void;
  onOpenImageStudio: (modelId?: string) => void;
  onOpenImageGallery: (modelId?: string) => void;
  onSubmitImageGeneration: () => void;
  onApplyImageRatioPreset: (presetId: (typeof IMAGE_RATIO_PRESETS)[number]["id"]) => void;
  onApplyImageQuality: (presetId: ImageQualityPreset) => void;
  imageDraftMode: boolean;
  onImageDraftModeChange: (value: boolean) => void;
  imageSampler: ImageSamplerId;
  onImageSamplerChange: (value: ImageSamplerId) => void;
  /** FU-015: diffusion cache strategy id ("none" / "fbcache" / "teacache"). */
  imageCacheStrategy: ImageCacheStrategyId;
  onImageCacheStrategyChange: (value: ImageCacheStrategyId) => void;
  /** Optional threshold override; null defers to strategy default. */
  imageCacheRelL1Thresh: number | null;
  onImageCacheRelL1ThreshChange: (value: number | null) => void;
  /** FU-021: opt-in CFG decay for flow-match image models. */
  imageCfgDecay: boolean;
  onImageCfgDecayChange: (value: boolean) => void;
  imagePreviewVae: boolean;
  onImagePreviewVaeChange: (value: boolean) => void;
  /** FU-024: opt-in FP8 layerwise casting (CUDA SM 8.9+). */
  imageFp8LayerwiseCasting: boolean;
  onImageFp8LayerwiseCastingChange: (value: boolean) => void;
  /** Hide CUDA-only controls (FP8 layerwise) on Apple Silicon hosts. */
  isAppleSiliconHost: boolean;
  /** Filter platform-incompatible variants out of the model dropdown
   * (per the FU-056 hide-unrecoverable-options policy). Threaded from
   * App.tsx → workspace.system. Optional because some test harnesses
   * mount the component without a full SystemStats. */
  hostSystem?: Pick<SystemStats, "platform" | "arch">;
  onPreloadImageModel: (variant: ImageModelVariant) => void;
  onUnloadImageModel: (variant?: ImageModelVariant) => void;
  onInstallImageRuntime: () => Promise<InstallResult>;
  /** Trigger /api/setup/install-cuda-torch directly from the GPU
   * acceleration warning. Lets the user fix the +cpu wheel without
   * navigating away to Settings > Setup. */
  onInstallCudaTorch?: () => void;
  installingCudaTorch?: boolean;
  /** Raw result from the most recent install attempt. Drives the
   * collapsible terminal log under the Install button so users can
   * inspect per-attempt pip output for debugging. ``null`` until an
   * install has been kicked off in this session. */
  cudaTorchResult?: CudaTorchInstallResult | null;
  // Live state of the GPU bundle install job — drives the InstallLogPanel
  // under the install button so users see per-step pip output instead of a
  // generic "failed" toast. Null when no install has been kicked off yet
  // in this session.
  gpuBundleJob: GpuBundleJobState | null;
  onImageDownload: (repo: string) => void;
  onCancelImageDownload: (repo: string) => void;
  onDeleteImageDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRestartServer: () => void;
  onUseSameImageSettings: (artifact: ImageOutputArtifact, closeModal?: boolean) => void;
  onVaryImageSeed: (artifact: ImageOutputArtifact) => void;
  onRevealPath: (path: string) => void;
  onDeleteImageArtifact: (id: string) => void;
}

export function ImageStudioTab({
  imageCatalog,
  selectedImageModelId,
  onSelectedImageModelIdChange,
  selectedImageVariant,
  selectedImageFamily,
  selectedImageLoaded,
  selectedImageWillLoadOnGenerate,
  imageRuntimeLoadedDifferentModel,
  loadedImageVariant,
  imageRuntimeStatus,
  tauriBackend,
  busy,
  busyAction,
  imageBusy,
  imageBusyLabel,
  backendOnline,
  nativeBackends,
  activeImageDownloads,
  imagePrompt,
  onImagePromptChange,
  imageNegativePrompt,
  onImageNegativePromptChange,
  imageQualityPreset,
  imageRatioId,
  imageWidth,
  onImageWidthChange,
  imageHeight,
  onImageHeightChange,
  imageSteps,
  onImageStepsChange,
  imageGuidance,
  onImageGuidanceChange,
  imageBatchSize,
  onImageBatchSizeChange,
  imageUseRandomSeed,
  onImageUseRandomSeedChange,
  imageSeedInput,
  onImageSeedInputChange,
  imageOutputs,
  recentImageOutputs,
  onActiveTabChange,
  onOpenImageStudio,
  onOpenImageGallery,
  onSubmitImageGeneration,
  onApplyImageRatioPreset,
  onApplyImageQuality,
  imageDraftMode,
  onImageDraftModeChange,
  imageSampler,
  onImageSamplerChange,
  imageCacheStrategy,
  onImageCacheStrategyChange,
  imageCacheRelL1Thresh,
  onImageCacheRelL1ThreshChange,
  imageCfgDecay,
  onImageCfgDecayChange,
  imagePreviewVae,
  onImagePreviewVaeChange,
  imageFp8LayerwiseCasting,
  onImageFp8LayerwiseCastingChange,
  isAppleSiliconHost,
  hostSystem,
  onPreloadImageModel,
  onUnloadImageModel,
  onInstallImageRuntime,
  onInstallCudaTorch,
  installingCudaTorch,
  cudaTorchResult,
  gpuBundleJob,
  onImageDownload,
  onCancelImageDownload,
  onDeleteImageDownload,
  onOpenExternalUrl,
  onRestartServer,
  onUseSameImageSettings,
  onVaryImageSeed,
  onRevealPath,
  onDeleteImageArtifact,
}: ImageStudioTabProps) {
  const { t } = useTranslation("studio");
  const [installingImageRuntime, setInstallingImageRuntime] = useState(false);
  // Per-configuration acknowledgement that unlocks Generate when the image
  // safety heuristic flags a danger-level run. Image OOMs on MPS are less
  // catastrophic than video (the sidecar may just die rather than kernel-
  // panic the Mac), but FLUX dev at 2K on a 32 GB Mac is still a reliable
  // way to nuke the backend — worth a conscious "I know what I'm doing"
  // step. Resets whenever variant / width / height change.
  const [dangerOverrideAck, setDangerOverrideAck] = useState(false);

  async function handleInstallImageRuntime() {
    if (installingImageRuntime) return;
    setInstallingImageRuntime(true);
    try {
      const result = await onInstallImageRuntime();
      if (result.ok && result.output.toLowerCase().includes("restart")) {
        onRestartServer();
      }
    } finally {
      setInstallingImageRuntime(false);
    }
  }

  // Only offer models that are actually downloaded in the picker. The
  // Image Studio is the "generate right now" surface — a user selecting an
  // unavailable model here would hit a download-required callout and be
  // bounced to Discover anyway. Families that end up with zero installed
  // variants disappear entirely.
  const installedCatalog = useMemo(() => {
    return imageCatalog
      .map((family) => ({
        ...family,
        variants: family.variants.filter((variant) => {
          if (!variant.availableLocally) return false;
          // FU-056: hide platform-incompatible variants entirely so
          // macOS users don't see Nunchaku INT4 (CUDA) etc. and
          // Win/Linux users don't see mflux (MLX) / sd.cpp MLX-only
          // entries they can never load.
          if (
            !isVariantCompatibleWithHost(
              imageOrVideoVariantPlatformGate(variant),
              hostSystem,
            )
          ) {
            return false;
          }
          return true;
        }),
      }))
      .filter((family) => family.variants.length > 0);
  }, [imageCatalog, hostSystem]);

  const hasInstalledImageModels = installedCatalog.length > 0;

  // If the currently-selected model is no longer in the installed list
  // (e.g. the user just deleted it, or the default picked an uninstalled
  // variant), fall back to the first installed one so the dropdown stays
  // in sync with what's on disk.
  useEffect(() => {
    if (!hasInstalledImageModels) return;
    const stillInstalled = installedCatalog.some((family) =>
      family.variants.some((variant) => variant.id === selectedImageModelId),
    );
    if (!stillInstalled) {
      const firstInstalled = installedCatalog[0].variants[0];
      if (firstInstalled) {
        onSelectedImageModelIdChange(firstInstalled.id);
      }
    }
  }, [installedCatalog, hasInstalledImageModels, selectedImageModelId, onSelectedImageModelIdChange]);

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

  // Safety estimate for the chosen model × resolution against the active
  // device's memory budget. Surfaces BEFORE the user clicks Generate so
  // FLUX-dev-on-32GB-MPS and similar detonations are caught pre-flight
  // instead of killing the sidecar mid-generate. Same pattern as the
  // video-safety heuristic — see assessImageGenerationSafety docs for the
  // calibration points.
  const imageSafety = useMemo(
    () =>
      assessImageGenerationSafety({
        width: imageWidth,
        height: imageHeight,
        device: imageRuntimeStatus.device ?? imageRuntimeStatus.expectedDevice,
        deviceMemoryGb: imageRuntimeStatus.deviceMemoryGb,
        baseModelFootprintGb: selectedImageVariant
          ? imageVariantSizeForMemoryEstimate(selectedImageVariant)
          : undefined,
        runtimeFootprintGb: selectedImageVariant?.runtimeFootprintGb,
        runtimeFootprintMpsGb: selectedImageVariant?.runtimeFootprintMpsGb,
        runtimeFootprintCudaGb: selectedImageVariant?.runtimeFootprintCudaGb,
        runtimeFootprintCpuGb: selectedImageVariant?.runtimeFootprintCpuGb,
        repo: selectedImageVariant?.repo,
        ggufFile: selectedImageVariant?.ggufFile,
      }),
    [
      imageWidth,
      imageHeight,
      imageRuntimeStatus.device,
      imageRuntimeStatus.expectedDevice,
      imageRuntimeStatus.deviceMemoryGb,
      selectedImageVariant?.repo,
      selectedImageVariant?.ggufFile,
      selectedImageVariant?.sizeGb,
      selectedImageVariant?.coreWeightsGb,
      selectedImageVariant?.onDiskGb,
      selectedImageVariant?.repoSizeGb,
      selectedImageVariant?.runtimeFootprintGb,
      selectedImageVariant?.runtimeFootprintMpsGb,
      selectedImageVariant?.runtimeFootprintCudaGb,
      selectedImageVariant?.runtimeFootprintCpuGb,
    ],
  );

  useEffect(() => {
    setDangerOverrideAck(false);
  }, [selectedImageVariant?.id, imageWidth, imageHeight]);

  // FU-015: image cache strategy filter. Match the video-side gating —
  // hide TeaCache for non-FLUX DiTs (calibrated forward exists for
  // FLUX only) and hide both strategies entirely for UNet pipelines
  // (SDXL / SD1.5 / SD2 — no .transformer attribute to attach to).
  // Auto-reset to "none" if the user previously picked something
  // that no longer applies after switching variants.
  const selectedImageRepo = selectedImageVariant?.repo ?? "";
  const isUnetVariant = isUnetImageRepo(selectedImageRepo);
  const availableImageCacheStrategies = useMemo(
    () => imageCacheStrategiesForRepo(selectedImageRepo),
    [selectedImageRepo],
  );
  useEffect(() => {
    const allowedIds = new Set(availableImageCacheStrategies.map((s) => s.id));
    if (!allowedIds.has(imageCacheStrategy)) {
      onImageCacheStrategyChange("none");
    }
  }, [availableImageCacheStrategies, imageCacheStrategy, onImageCacheStrategyChange]);

  function handleApplySafeImageSettings() {
    const suggestion = imageSafety.suggestion;
    if (!suggestion) return;
    onImageWidthChange(suggestion.width);
    onImageHeightChange(suggestion.height);
  }

  const imageGenerateBlockedByDanger =
    imageSafety.riskLevel === "danger" && !dangerOverrideAck;
  const imageGenerateDisabled =
    imageBusy || !selectedImageVariant || imageGenerateBlockedByDanger;
  const imageGenerateTitle = imageGenerateBlockedByDanger
    ? t("imageStudio.generateTitle.dangerBlocked", {
        defaultValue: "Danger-level configuration — tick the acknowledgement below the safety callout to proceed.",
      })
    : imageBusy
      ? t("imageStudio.generateTitle.generating", { defaultValue: "Generating..." })
      : !selectedImageVariant
        ? t("imageStudio.generateTitle.selectModel", { defaultValue: "Select a model first." })
        : t("imageStudio.generateTitle.ready", { defaultValue: "Generate this image." });

  // Localized aspect-ratio + quality preset labels. Static config arrays
  // in src/constants/image.ts stay english-only (shared with non-React
  // call sites); we resolve the visible label here so the studio renders
  // the user's chosen locale.
  const ratioLabelFor = (id: string): string =>
    t(`imageStudio.aspectRatios.${id}.label`, {
      defaultValue:
        IMAGE_RATIO_PRESETS.find((p) => p.id === id)?.label ?? id,
    });
  const qualityLabelFor = (id: string): string =>
    t(`imageStudio.quality.${id}.label`, {
      defaultValue:
        IMAGE_QUALITY_PRESETS.find((p) => p.id === id)?.label ?? id,
    });
  const qualityHintFor = (id: string): string =>
    t(`imageStudio.quality.${id}.hint`, {
      defaultValue:
        IMAGE_QUALITY_PRESETS.find((p) => p.id === id)?.hint ?? "",
    });
  const samplerLabelFor = (id: string): string =>
    t(`imageStudio.samplers.${id}.label`, {
      defaultValue:
        IMAGE_SAMPLERS.find((s) => s.id === id)?.label ?? id,
    });
  const samplerHintFor = (id: string): string =>
    t(`imageStudio.samplers.${id}.hint`, {
      defaultValue:
        IMAGE_SAMPLERS.find((s) => s.id === id)?.hint ?? "",
    });
  const cacheStrategyLabelFor = (id: string): string =>
    t(`imageStudio.cacheStrategies.${id}.label`, {
      defaultValue:
        availableImageCacheStrategies.find((s) => s.id === id)?.label ?? id,
    });
  const cacheStrategyHintFor = (id: string): string =>
    t(`imageStudio.cacheStrategies.${id}.hint`, {
      defaultValue:
        availableImageCacheStrategies.find((s) => s.id === id)?.hint ?? "",
    });

  const formatImageGb = (gb: number): string =>
    gb >= 10 ? `${gb.toFixed(0)} GB` : `${gb.toFixed(1)} GB`;

  return (
    <div className="content-grid image-page-grid">
      <Panel
        title={t("image.title")}
        subtitle={selectedImageVariant
          ? `${selectedImageVariant.name} / ${selectedImageVariant.runtime} / ${imageOutputs.length} ${t("image.savedOutputsLabel", { defaultValue: "saved outputs" })}`
          : t("image.subtitle", { defaultValue: "Choose a model, prompt it, and iterate on saved outputs" })}
        className="span-2"
        actions={
          <div className="button-row">
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-discover")}>
              {t("imageStudio.actions.discover", { defaultValue: "Discover" })}
            </button>
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-models")}>
              {t("imageStudio.actions.installed", { defaultValue: "Installed" })}
            </button>
            <button className="secondary-button" type="button" onClick={() => onOpenImageGallery()}>
              {t("imageStudio.actions.gallery", { defaultValue: "Gallery" })}
            </button>
          </div>
        }
      >
        <div className="image-studio-hero">
          <div>
            <span className="eyebrow">{t("imageStudio.hero.currentRuntime", { defaultValue: "Current Runtime" })}</span>
            <h3>{selectedImageVariant?.name ?? t("imageStudio.hero.placeholder", { defaultValue: "Select an image model" })}</h3>
          </div>
          {selectedImageVariant ? (
            <div className="image-studio-hero-stats">
              <span className="badge muted">{selectedImageFamily?.name ?? selectedImageVariant.provider}</span>
              <span className="badge muted">{selectedImageVariant.recommendedResolution}</span>
              <span className="badge muted">{sizeLabel(selectedImageVariant.sizeGb)}</span>
              {selectedImageVariant.availableLocally ? <span className="badge success">{t("imageStudio.badges.installed", { defaultValue: "Installed" })}</span> : null}
              {selectedImageLoaded ? <span className="badge success">{t("imageStudio.badges.loadedInMemory", { defaultValue: "Loaded In Memory" })}</span> : null}
              {selectedImageWillLoadOnGenerate ? <span className="badge subtle">{t("imageStudio.badges.loadsOnFirstGenerate", { defaultValue: "Loads On First Generate" })}</span> : null}
              {imageBusy && selectedImageWillLoadOnGenerate ? <span className="badge accent">{t("imageStudio.badges.loadingIntoMemory", { defaultValue: "Loading Into Memory" })}</span> : null}
              {!selectedImageVariant.availableLocally && selectedImageDownloadComplete ? <span className="badge success">{t("imageStudio.badges.downloaded", { defaultValue: "Downloaded" })}</span> : null}
            </div>
          ) : null}
        </div>
        <ImageStudioRuntimeBanner
          imageRuntimeStatus={imageRuntimeStatus}
          selectedImageVariant={selectedImageVariant}
          selectedImageLoaded={selectedImageLoaded}
          selectedImageWillLoadOnGenerate={selectedImageWillLoadOnGenerate}
          imageRuntimeLoadedDifferentModel={imageRuntimeLoadedDifferentModel}
          loadedImageVariant={loadedImageVariant}
          busy={busy}
          busyAction={busyAction}
          imageBusy={imageBusy}
          backendOnline={backendOnline}
          onRestartServer={onRestartServer}
          onPreloadImageModel={onPreloadImageModel}
          onUnloadImageModel={onUnloadImageModel}
          onInstallCudaTorch={onInstallCudaTorch}
          installingCudaTorch={installingCudaTorch}
          cudaTorchResult={cudaTorchResult}
          installingImageRuntime={installingImageRuntime}
          gpuBundleJob={gpuBundleJob}
          onInstallImageRuntime={() => void handleInstallImageRuntime()}
          nativeBackends={nativeBackends}
        />
      </Panel>

      <Panel
        title={t("image.promptPanelTitle", { defaultValue: "Prompt" })}
        subtitle={t("image.promptPanelSubtitle", {
          defaultValue: "Choose a model, set the aspect ratio and quality, then generate into the local gallery.",
        })}
        className="image-studio-form-panel"
        actions={
          <button
            className="primary-button"
            type="button"
            onClick={() => onSubmitImageGeneration()}
            disabled={imageGenerateDisabled}
            title={imageGenerateTitle}
          >
            {imageBusy
              ? t("image.generating", { defaultValue: "Generating..." })
              : t("image.generate", { defaultValue: "Generate" })}
          </button>
        }
      >
        <div className="image-form-stack">
          <label>
            {t("image.modelLabel", { defaultValue: "Model" })}
            <select
              className="text-input"
              value={hasInstalledImageModels ? selectedImageModelId : ""}
              onChange={(event) => onSelectedImageModelIdChange(event.target.value)}
              disabled={!hasInstalledImageModels}
            >
              {hasInstalledImageModels ? (
                installedCatalog.map((family) => (
                  <optgroup key={family.id} label={family.name}>
                    {family.variants.map((variant) => (
                      <option key={variant.id} value={variant.id}>
                        {variant.name}
                      </option>
                    ))}
                  </optgroup>
                ))
              ) : (
                <option value="">{t("image.noModelsOption", { defaultValue: "No models installed — download one from Discover" })}</option>
              )}
            </select>
          </label>

          {!hasInstalledImageModels ? (
            <div className="callout image-callout">
              <p>
                {t("imageStudio.noModelsCallout.message", {
                  defaultValue:
                    "You don't have any image models downloaded yet. Head to Image Discover to browse and install one, then come back here to generate.",
                })}
              </p>
              <div className="button-row">
                <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-discover")}>
                  {t("imageStudio.noModelsCallout.openDiscover", { defaultValue: "Open Image Discover" })}
                </button>
              </div>
            </div>
          ) : null}

          {!selectedImageVariant?.availableLocally && selectedImageVariant ? (
            <div className="callout image-callout">
              <p>
                {selectedImageDownloadFailed
                  ? t("imageStudio.download.failedMessage", {
                      defaultValue:
                        "{name} did not finish downloading correctly. ChaosEngineAI only found a partial local snapshot, so it cannot load the real image pipeline yet.",
                      name: selectedImageVariant.name,
                    })
                  : selectedImageDownloadPaused
                  ? t("imageStudio.download.pausedMessage", {
                      defaultValue:
                        "{name} is partially downloaded. Resume when you're ready and ChaosEngineAI will continue from the files already on disk.",
                      name: selectedImageVariant.name,
                    })
                  : selectedImageDownloadComplete
                  ? t("imageStudio.download.completeMessage", {
                      defaultValue:
                        "{name} finished downloading. The installed-model scan will refresh automatically.",
                      name: selectedImageVariant.name,
                    })
                  : t("imageStudio.download.notInstalledMessage", {
                      defaultValue:
                        "{name} is not installed locally. Download it from Discover to enable local generation.",
                      name: selectedImageVariant.name,
                    })}
              </p>
              {selectedImageDownloadFailed && selectedImageDownload?.error ? (
                <>
                  <p className="muted-text">{selectedImageFriendlyDownloadError}</p>
                  {selectedImageNeedsGatedAccess ? (
                    <div className="button-row">
                      <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(selectedImageVariant.link)}>
                        {t("imageStudio.download.huggingFace", { defaultValue: "Hugging Face" })}
                      </button>
                      <button className="secondary-button" type="button" onClick={() => onActiveTabChange("settings")}>
                        {t("imageStudio.download.settings", { defaultValue: "Settings" })}
                      </button>
                    </div>
                  ) : null}
                  {selectedImageFriendlyDownloadError !== selectedImageDownload.error ? (
                    <details className="debug-details">
                      <summary>{t("imageStudio.download.technicalDetails", { defaultValue: "Technical details" })}</summary>
                      <p className="mono-text">{selectedImageDownload.error}</p>
                    </details>
                  ) : null}
                </>
              ) : null}
              <div className="button-row">
                {selectedImageDownload?.state === "downloading" ? (
                  <>
                    <span className="badge accent">{downloadProgressLabel(selectedImageDownload)}</span>
                    <button className="secondary-button" type="button" onClick={() => onCancelImageDownload(selectedImageVariant.repo)}>
                      {t("imageStudio.download.pause", { defaultValue: "Pause" })}
                    </button>
                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(selectedImageVariant.repo)}>
                      {t("imageStudio.download.cancel", { defaultValue: "Cancel" })}
                    </button>
                  </>
                ) : selectedImageDownloadPaused ? (
                  <>
                    <span className="badge warning">{downloadProgressLabel(selectedImageDownload)}</span>
                    <button className="secondary-button" type="button" onClick={() => onImageDownload(selectedImageVariant.repo)}>
                      {t("imageStudio.download.resume", { defaultValue: "Resume" })}
                    </button>
                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(selectedImageVariant.repo)}>
                      {t("imageStudio.download.delete", { defaultValue: "Delete" })}
                    </button>
                  </>
                ) : selectedImageDownloadFailed ? (
                  <>
                    <button className="secondary-button" type="button" onClick={() => onImageDownload(selectedImageVariant.repo)}>
                      {t("imageStudio.download.retry", { defaultValue: "Retry Download" })}
                    </button>
                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(selectedImageVariant.repo)}>
                      {t("imageStudio.download.delete", { defaultValue: "Delete" })}
                    </button>
                  </>
                ) : selectedImageDownloadComplete ? (
                  <span className="badge success">{t("imageStudio.download.complete", { defaultValue: "Download complete" })}</span>
                ) : (
                  <>
                    <button className="secondary-button" type="button" onClick={() => onImageDownload(selectedImageVariant.repo)}>
                      {t("imageStudio.download.downloadModel", { defaultValue: "Download Model" })}
                    </button>
                    {selectedImageVariant.hasLocalData ? (
                      <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(selectedImageVariant.repo)}>
                        {t("imageStudio.download.delete", { defaultValue: "Delete" })}
                      </button>
                    ) : null}
                  </>
                )}
                <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(selectedImageVariant.link)}>
                  {t("imageStudio.download.huggingFace", { defaultValue: "Hugging Face" })}
                </button>
              </div>
            </div>
          ) : null}

          <label>
            <span className="prompt-label-row">
              {t("imageStudio.prompt.label", { defaultValue: "Prompt" })}
              <PromptEnhanceButton
                prompt={imagePrompt}
                repo={selectedImageVariant?.repo ?? ""}
                onEnhanced={onImagePromptChange}
              />
            </span>
            <textarea
              className="text-input prompt-area"
              rows={5}
              placeholder={t("imageStudio.prompt.placeholder", {
                defaultValue:
                  "Moody cinematic alleyway after rain, neon reflections, 35mm photo, shallow depth of field",
              })}
              value={imagePrompt}
              onChange={(event) => onImagePromptChange(event.target.value)}
            />
          </label>

          <label>
            {t("image.negativePrompt", { defaultValue: "Negative prompt" })}
            <textarea
              className="text-input prompt-area prompt-area--secondary"
              rows={3}
              placeholder={t("imageStudio.negativePrompt.placeholder", {
                defaultValue: "blurry, deformed hands, extra limbs, overexposed",
              })}
              value={imageNegativePrompt}
              onChange={(event) => onImageNegativePromptChange(event.target.value)}
            />
          </label>

          <div className="control-stack">
            <span className="eyebrow">{t("image.aspectRatio", { defaultValue: "Aspect Ratio" })}</span>
            <div className="image-pill-row">
              {IMAGE_RATIO_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  className={selectedRatioPreset.id === preset.id ? "pill-button active" : "pill-button"}
                  type="button"
                  onClick={() => onApplyImageRatioPreset(preset.id)}
                >
                  <strong>{ratioLabelFor(preset.id)}</strong>
                  <span>{preset.hint}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="control-stack">
            <span className="eyebrow">{t("image.qualityPreset", { defaultValue: "Quality Preset" })}</span>
            <div className="image-pill-row">
              {IMAGE_QUALITY_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  className={selectedQuality.id === preset.id ? "pill-button active" : "pill-button"}
                  type="button"
                  onClick={() => onApplyImageQuality(preset.id)}
                >
                  <strong>{qualityLabelFor(preset.id)}</strong>
                  <span>{qualityHintFor(preset.id)}</span>
                </button>
              ))}
              <button
                className={imageDraftMode ? "pill-button active" : "pill-button"}
                type="button"
                onClick={() => onImageDraftModeChange(!imageDraftMode)}
                title={t("imageStudio.draft.tooltip", {
                  defaultValue:
                    "Force a 512px long-edge render for fast prompt iteration. Output saves at the draft size — disable for a full-resolution final pass.",
                })}
              >
                <strong>{t("image.previewLabel", { defaultValue: "Preview" })}</strong>
                <span>{imageDraftMode
                  ? t("image.draftOn", { defaultValue: "512px · on" })
                  : t("image.draftOff", { defaultValue: "Draft @ 512px" })}</span>
              </button>
            </div>
          </div>

          {selectedImageVariant && !isFlowMatchingRepo(selectedImageVariant.repo) ? (
            <div className="control-stack">
              <span className="eyebrow">
                {t("image.sampler", { defaultValue: "Sampler" })}
                <InfoTooltip text={t("imageStudio.sampler.tooltip", {
                  defaultValue:
                    "Scheduler / sampler algorithm used during denoising. ‘Model default’ keeps whatever the pipeline shipped with. AYS DPM++ 2M (SD1.5 / SDXL) uses NVIDIA’s Align Your Steps schedule — wins detail at 7-10 steps where Karras / Euler look soft. Hidden for FLUX, SD3, Qwen-Image, Sana and HiDream — those flow-matching pipelines ship locked schedulers and swapping produces noise.",
                })} />
              </span>
              <select
                className="text-input"
                value={imageSampler}
                onChange={(event) => onImageSamplerChange(event.target.value as ImageSamplerId)}
              >
                {IMAGE_SAMPLERS.map((sampler) => (
                  <option key={sampler.id} value={sampler.id}>
                    {samplerLabelFor(sampler.id)} · {samplerHintFor(sampler.id)}
                  </option>
                ))}
              </select>
            </div>
          ) : null}

          {/*
            FU-015: diffusion cache strategy. Cross-platform — runs on
            macOS (MPS), Windows (CUDA / DirectML) and Linux (CUDA / CPU)
            because both FBCache and TeaCache attach to the diffusers
            transformer regardless of device. Hidden for the placeholder
            engine and for variants that lack a transformer attribute
            (UNet-based SD1.5/SDXL fall through gracefully on the
            backend with a runtimeNote).
          */}
          {selectedImageVariant && !isUnetVariant ? (
            <div className="control-stack">
              <span className="eyebrow">
                {t("image.diffusionCache", { defaultValue: "Diffusion cache" })}
                <InfoTooltip text={t("imageStudio.diffusionCache.tooltip", {
                  defaultValue:
                    "Speed up generation by reusing transformer block outputs between similar sampling steps. First Block Cache is the cross-platform default and works on every DiT pipeline (FLUX, SD3, Qwen-Image, Sana, HiDream, Z-Image, ERNIE-Image, GLM-Image) on macOS / Windows / Linux — typical 1.5-2× wall-time win at default threshold with imperceptible quality drift. TeaCache only ships calibrated forwards for the FLUX family on the image side — hidden for other DiTs so the dropdown reflects what the backend will actually apply. Hidden entirely for UNet pipelines (SDXL / SD1.5 / SD2) which lack the transformer attachment point.",
                })} />
              </span>
              <select
                className="text-input"
                value={imageCacheStrategy}
                onChange={(event) =>
                  onImageCacheStrategyChange(event.target.value as ImageCacheStrategyId)
                }
              >
                {availableImageCacheStrategies.map((strategy) => (
                  <option key={strategy.id} value={strategy.id}>
                    {cacheStrategyLabelFor(strategy.id)} · {cacheStrategyHintFor(strategy.id)}
                  </option>
                ))}
              </select>
              {availableImageCacheStrategies.length === 2 ? (
                <span className="muted-text" style={{ fontSize: 11 }}>
                  {t("imageStudio.diffusionCache.teaCacheHidden", {
                    defaultValue:
                      "TeaCache hidden — its image-side calibration only covers the FLUX family. First Block Cache works on every DiT pipeline shipped today (cross-platform).",
                  })}
                </span>
              ) : null}
              {imageCacheStrategy !== "none" ? (
                <label className="control-stack-inline">
                  <span className="muted-text">
                    {t("imageStudio.diffusionCache.threshold", {
                      defaultValue: "Threshold ({value})",
                      value: imageCacheRelL1Thresh ??
                        IMAGE_CACHE_STRATEGY_DEFAULT_THRESH[imageCacheStrategy],
                    })}
                    <InfoTooltip text={t("imageStudio.diffusionCache.thresholdTooltip", {
                      defaultValue:
                        "Relative L1 distance between consecutive block-cache states. Lower = stricter (less speedup, less drift). Higher = more aggressive caching (more speedup, may shimmer fine detail). Defaults: First Block Cache 0.12, TeaCache 0.4 — both calibrated against the diffusers blog / upstream papers for negligible quality loss on FLUX.1-dev.",
                    })} />
                  </span>
                  <input
                    className="text-input"
                    type="number"
                    min={0.01}
                    max={0.6}
                    step={0.01}
                    value={
                      imageCacheRelL1Thresh ??
                      IMAGE_CACHE_STRATEGY_DEFAULT_THRESH[imageCacheStrategy]
                    }
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      onImageCacheRelL1ThreshChange(
                        Number.isFinite(value) && value > 0 ? value : null,
                      );
                    }}
                  />
                </label>
              ) : null}
            </div>
          ) : null}

          {/*
            FU-021: opt-in CFG decay schedule. Applies only to
            flow-match models (FLUX, SD3, Qwen-Image, Sana, HiDream)
            where late-step high CFG drifts toward oversaturation.
            Backend gates non-flow-match repos automatically; we hide
            the toggle for SD1.5/SDXL so the UI matches behaviour.
          */}
          {selectedImageVariant && isFlowMatchingRepo(selectedImageVariant.repo) ? (
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={imageCfgDecay}
                onChange={(event) => onImageCfgDecayChange(event.target.checked)}
              />
              <span>
                <strong>{t("imageStudio.toggles.cfgDecay.label", { defaultValue: "CFG decay" })}</strong>
                {" "}— {t("imageStudio.toggles.cfgDecay.description", {
                  defaultValue:
                    "linearly relax guidance from your slider value toward 1.5 across the schedule. Reduces oversaturation on late steps without changing semantics from early steps. Off by default; backend ignores this on SD1.5 / SDXL.",
                })}
                <InfoTooltip text={t("imageStudio.toggles.cfgDecay.tooltip", {
                  defaultValue:
                    "Flow-match models (FLUX, SD3, Qwen-Image, Sana, HiDream) tend to ‘burn’ highlights when classifier-free guidance stays high through every step. Decaying lets early steps lock semantics (high CFG) while late steps preserve fine detail (low CFG). Floor stays at 1.5 — dropping to 1.0 mid-schedule swaps the pipeline from 2-batch to 1-batch shape and crashes diffusers. Same shape as the video runtime knob you already use.",
                })} />
              </span>
            </label>
          ) : null}

          {/*
            FU-018: TAESD preview-decode VAE swap. Off by default —
            image users typically want full fidelity. Backend maps
            the loaded repo to the matching tiny VAE
            (taef1/taef2/taesd3/taesdxl/taesd/taeqwenimage); unmapped
            repos no-op silently.
          */}
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={imagePreviewVae}
              onChange={(event) => onImagePreviewVaeChange(event.target.checked)}
            />
            <span>
              <strong>{t("imageStudio.toggles.previewVae.label", { defaultValue: "Preview VAE" })}</strong>
              {" "}— {t("imageStudio.toggles.previewVae.description", {
                defaultValue:
                  "swap the full VAE for the matching tiny VAE (TAESD / TAEHV) so each step decodes in a fraction of the wall-time. Trades final fidelity for iteration speed. Off by default.",
              })}
              <InfoTooltip text={t("imageStudio.toggles.previewVae.tooltip", {
                defaultValue:
                  "Tiny VAEs (madebyollin/taef1, taef2, taesd3, taesdxl, taesd, taeqwenimage) are 1-2 order of magnitude faster than the full VAE but lose some fine-detail fidelity. Best for fast iteration / drafting; flip off when you're ready to ship the final image. Backend silently no-ops on repos without a mapped tiny VAE so you can leave it on without surprises.",
              })} />
            </span>
          </label>

          {/*
            FU-024: FP8 layerwise casting on CUDA SM 8.9+ (Ada/Hopper/
            Blackwell). Halves transformer VRAM by storing fp8 weights +
            promoting to bf16 inside the matmul. Hidden entirely on
            Apple Silicon — there's no CUDA path so the toggle would be
            unreachable. Still rendered on Linux / Windows where backend
            checks compute capability + skips on pre-Ada hardware.
          */}
          {!isAppleSiliconHost ? (
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={imageFp8LayerwiseCasting}
                onChange={(event) => onImageFp8LayerwiseCastingChange(event.target.checked)}
              />
              <span>
                <strong>{t("imageStudio.toggles.fp8Layerwise.label", { defaultValue: "FP8 layerwise (CUDA Ada+)" })}</strong>
                {" "}— {t("imageStudio.toggles.fp8Layerwise.description", {
                  defaultValue:
                    "store transformer weights in fp8 + promote to bf16 inside the matmul. Halves VRAM with negligible quality drift on modern GPUs. Apple Silicon / pre-Ada GPUs no-op cleanly.",
                })}
                <InfoTooltip text={t("imageStudio.toggles.fp8Layerwise.tooltip", {
                  defaultValue:
                    "diffusers' enable_layerwise_casting. Family-correct dtype: E5M2 for HunyuanVideo, E4M3 for FLUX / Wan / Qwen-Image / SD3 / LTX. Backend checks GPU compute capability before applying — pre-Ada (SM <8.9) lacks hardware fp8 and skips with a runtimeNote. Best stacked with Nunchaku INT4 for the smallest footprint.",
                })} />
              </span>
            </label>
          ) : null}

          <div className="field-grid image-field-grid">
            <label>
              {t("imageStudio.fields.width", { defaultValue: "Width" })}
              <input
                className="text-input"
                type="number"
                min={256}
                max={2048}
                step={64}
                value={imageWidth}
                onChange={(event) => onImageWidthChange(Number(event.target.value) || 1024)}
              />
            </label>
            <label>
              {t("imageStudio.fields.height", { defaultValue: "Height" })}
              <input
                className="text-input"
                type="number"
                min={256}
                max={2048}
                step={64}
                value={imageHeight}
                onChange={(event) => onImageHeightChange(Number(event.target.value) || 1024)}
              />
            </label>
            <label>
              {t("imageStudio.fields.steps", { defaultValue: "Steps" })}
              <input
                className="text-input"
                type="number"
                min={1}
                max={100}
                value={imageSteps}
                onChange={(event) => onImageStepsChange(Number(event.target.value) || 24)}
              />
            </label>
            <label>
              {t("imageStudio.fields.guidance", { defaultValue: "Guidance" })}
              <input
                className="text-input"
                type="number"
                min={1}
                max={20}
                step={0.5}
                value={imageGuidance}
                onChange={(event) => onImageGuidanceChange(Number(event.target.value) || 6)}
              />
            </label>
            <label>
              {t("imageStudio.fields.images", { defaultValue: "Images" })}
              <input
                className="text-input"
                type="number"
                min={1}
                max={4}
                value={imageBatchSize}
                onChange={(event) => onImageBatchSizeChange(Math.max(1, Math.min(4, Number(event.target.value) || 1)))}
              />
            </label>
            <label className="checkbox-card">
              <span className="checkbox-card-label">{t("imageStudio.fields.randomSeed", { defaultValue: "Random seed" })}</span>
              <input
                type="checkbox"
                checked={imageUseRandomSeed}
                onChange={(event) => onImageUseRandomSeedChange(event.target.checked)}
              />
            </label>
          </div>

          {/*
            Pre-flight memory callout. Same structure as the video studio's
            safety callout. Shows for caution AND danger, but only the
            danger level blocks Generate — caution just warns. The
            checkbox below the warning text is the explicit override for
            danger runs (auto-resets on variant / width / height change).

            We keep this above the Seed input (below the field-grid) so
            users who just edited W/H see the consequence right there
            instead of having to hunt at the bottom of the form. Mirrors
            the video tab's placement.
          */}
          {imageSafety.riskLevel !== "safe" && selectedImageVariant ? (
            <div
              className={`callout image-callout ${
                imageSafety.riskLevel === "danger" ? "error" : "warning"
              }`}
              role="alert"
            >
              <p>
                <strong>
                  {imageSafety.riskLevel === "danger"
                    ? t("imageStudio.safety.dangerTitle", { defaultValue: "Likely to crash the backend" })
                    : t("imageStudio.safety.warningTitle", { defaultValue: "Heads up — may struggle on this device" })}
                  :
                </strong>{" "}
                {imageSafety.reason}
              </p>
              {imageSafety.modelFootprintGb > 0 ? (
                <p className="muted-text">
                  {t("imageStudio.safety.resourceEstimate", {
                    defaultValue:
                      "Model ≈ {model} · this run peak ≈ {peak} of ~{total} total.",
                    model: formatImageGb(imageSafety.modelFootprintGb),
                    peak: formatImageGb(imageSafety.estimatedPeakGb),
                    total: formatImageGb(imageSafety.deviceMemoryGb),
                  })}
                </p>
              ) : null}
              {imageSafety.suggestion ? (
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={handleApplySafeImageSettings}
                    disabled={imageBusy}
                    title={t("imageStudio.safety.applyTitle", {
                      defaultValue: "Apply {label}",
                      label: imageSafety.suggestion.label,
                    })}
                  >
                    {t("imageStudio.safety.useSafer", {
                      defaultValue: "Use safer settings ({label})",
                      label: imageSafety.suggestion.label,
                    })}
                  </button>
                </div>
              ) : (
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => onActiveTabChange("image-discover")}
                    disabled={imageBusy}
                  >
                    {t("imageStudio.safety.browseSmaller", { defaultValue: "Browse smaller models" })}
                  </button>
                </div>
              )}
              {imageSafety.riskLevel === "danger" ? (
                <label
                  className="inline-label"
                  style={{ display: "flex", alignItems: "center", gap: ".4rem", marginTop: ".6rem" }}
                >
                  <input
                    type="checkbox"
                    checked={dangerOverrideAck}
                    onChange={(event) => setDangerOverrideAck(event.target.checked)}
                  />
                  <span>
                    {t("imageStudio.safety.overrideAck", {
                      defaultValue: "Generate anyway — I accept that the backend may crash.",
                    })}
                  </span>
                </label>
              ) : null}
            </div>
          ) : null}

          {!imageUseRandomSeed ? (
            <label>
              {t("imageStudio.fields.seed", { defaultValue: "Seed" })}
              <input
                className="text-input"
                type="number"
                min={0}
                max={2147483647}
                value={imageSeedInput}
                onChange={(event) => onImageSeedInputChange(event.target.value)}
              />
            </label>
          ) : null}

          {imageBusyLabel ? (
            <p className="busy-indicator"><span className="busy-dot" />{imageBusyLabel}</p>
          ) : null}
        </div>
      </Panel>

      <Panel
        title={t("image.recentOutputsTitle", { defaultValue: "Recent Outputs" })}
        subtitle={imageOutputs.length > 0
          ? t("imageStudio.recent.subtitle", {
              defaultValue: "{recent} newest of {total} saved generations",
              recent: recentImageOutputs.length,
              total: imageOutputs.length,
            })
          : t("imageStudio.recent.emptySubtitle", { defaultValue: "Generated images will appear here" })}
        className="image-gallery-panel"
        actions={
          <button className="secondary-button" type="button" onClick={() => onOpenImageGallery()}>
            {t("imageStudio.recent.openGallery", { defaultValue: "Open Gallery" })}
          </button>
        }
      >
        {imageOutputs.length === 0 ? (
          <div className="empty-state image-empty-state">
            <p>{t("imageStudio.recent.emptyHint", {
              defaultValue: "Generate a prompt to create the first saved image artifact for this branch.",
            })}</p>
          </div>
        ) : (
          <div className="image-gallery-grid">
            {recentImageOutputs.map((artifact) => (
              <ImageOutputCard
                key={artifact.artifactId}
                artifact={artifact}
                imageCatalog={imageCatalog}
                imageBusy={imageBusy}
                onOpenExternalUrl={(url) => onOpenExternalUrl(url)}
                onUseSameSettings={onUseSameImageSettings}
                onVarySeed={(a) => onVaryImageSeed(a)}
                onRevealPath={(path) => onRevealPath(path)}
                onDelete={(id) => onDeleteImageArtifact(id)}
                onNavigateSettings={() => onActiveTabChange("settings")}
              />
            ))}
          </div>
        )}
        {imageOutputs.length > recentImageOutputs.length ? (
          <p className="muted-text image-gallery-footnote">
            {t("imageStudio.recent.footnote", {
              defaultValue:
                "Showing the newest {recent} saved images here. Open Image Gallery to browse everything, filter by model, and manage older runs.",
              recent: recentImageOutputs.length,
            })}
          </p>
        ) : null}
      </Panel>
    </div>
  );
}
