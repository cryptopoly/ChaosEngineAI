import { useEffect, useMemo, useState } from "react";
import { Panel } from "../../components/Panel";
import { InstallLogPanel } from "../../components/InstallLogPanel";
import { ImageOutputCard } from "../../components/ImageOutputCard";
import type { DownloadStatus, GpuBundleJobState, InstallResult } from "../../api";
import type {
  ImageModelFamily,
  ImageModelVariant,
  ImageOutputArtifact,
  ImageQualityPreset,
  ImageSamplerId,
  ImageRuntimeStatus,
  TabId,
  TauriBackendInfo,
} from "../../types";
import {
  sizeLabel,
  downloadProgressLabel,
  formatImageAccessError,
  isGatedImageAccessError,
} from "../../utils";
import { assessImageGenerationSafety, imageVariantSizeForMemoryEstimate } from "../../utils/images";
import { IMAGE_RATIO_PRESETS, IMAGE_QUALITY_PRESETS, IMAGE_SAMPLERS, isFlowMatchingRepo } from "../../constants";

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
  onPreloadImageModel: (variant: ImageModelVariant) => void;
  onUnloadImageModel: (variant?: ImageModelVariant) => void;
  onInstallImageRuntime: () => Promise<InstallResult>;
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
  onPreloadImageModel,
  onUnloadImageModel,
  onInstallImageRuntime,
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
        variants: family.variants.filter((variant) => variant.availableLocally),
      }))
      .filter((family) => family.variants.length > 0);
  }, [imageCatalog]);

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
    ? "Danger-level configuration — tick the acknowledgement below the safety callout to proceed."
    : imageBusy
      ? "Generating..."
      : !selectedImageVariant
        ? "Select a model first."
        : "Generate this image.";

  const formatImageGb = (gb: number): string =>
    gb >= 10 ? `${gb.toFixed(0)} GB` : `${gb.toFixed(1)} GB`;

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
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-discover")}>
              Discover
            </button>
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-models")}>
              Installed
            </button>
            <button className="secondary-button" type="button" onClick={() => onOpenImageGallery()}>
              Gallery
            </button>
          </div>
        }
      >
        <div className="image-studio-hero">
          <div>
            <span className="eyebrow">Current Runtime</span>
            <h3>{selectedImageVariant?.name ?? "Select an image model"}</h3>
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
          <div className="chip-row">
            <span className={`badge ${imageRuntimeStatus.realGenerationAvailable ? "success" : "warning"}`}>
              {imageRuntimeStatus.realGenerationAvailable
                ? "Real local generation available"
                : imageRuntimeStatus.activeEngine === "unavailable"
                  ? "Runtime unavailable"
                  : "Using placeholder outputs"}
            </span>
            <span className="badge muted">Engine: {imageRuntimeStatus.activeEngine}</span>
            {/* Prefer the actual-loaded device; fall back to the
              * predicted expectedDevice computed cheaply via
              * nvidia-smi + find_spec (no torch import). When
              * nothing is loaded yet the badge reads 'Device: cuda
              * (expected)' which tells the user what will happen on
              * first Generate instead of leaving them to guess. */}
            {(() => {
              const resolved =
                imageRuntimeStatus.device
                ?? (imageRuntimeStatus.expectedDevice
                  ? `${imageRuntimeStatus.expectedDevice} (expected)`
                  : null);
              return resolved ? <span className="badge muted">Device: {resolved}</span> : null;
            })()}
          </div>
          {selectedImageVariant && imageRuntimeStatus.realGenerationAvailable ? (
            <div className="image-runtime-summary">
              <p className="muted-text">
                {selectedImageLoaded
                  ? `${selectedImageVariant.name} is loaded and ready to generate.`
                  : imageRuntimeLoadedDifferentModel && loadedImageVariant
                    ? `${loadedImageVariant.name} is loaded. Generating with ${selectedImageVariant.name} will swap the pipeline.`
                    : selectedImageWillLoadOnGenerate
                      ? `${selectedImageVariant.name} is installed locally but not loaded yet. The first generate will take longer while the diffusion pipeline warms up.`
                      : !selectedImageVariant.availableLocally
                        ? `${selectedImageVariant.name} is not installed locally. Download it from Discover to enable local generation.`
                        : "Model will load on demand when you generate."}
              </p>
              {imageBusy && selectedImageWillLoadOnGenerate ? (
                <p className="busy-indicator"><span className="busy-dot" />Loading model into memory...</p>
              ) : null}
              {(selectedImageVariant.availableLocally || loadedImageVariant) ? (
                <div className="button-row image-runtime-control-row">
                  {selectedImageVariant.availableLocally && !selectedImageLoaded ? (
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => onPreloadImageModel(selectedImageVariant)}
                      disabled={imageBusy || busy || !backendOnline}
                    >
                      Preload Model
                    </button>
                  ) : null}
                  {selectedImageLoaded ? (
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => onUnloadImageModel(selectedImageVariant)}
                      disabled={imageBusy || busy || !backendOnline}
                    >
                      Unload Model
                    </button>
                  ) : null}
                  {!selectedImageLoaded && loadedImageVariant ? (
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => onUnloadImageModel()}
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
            <>
              <div className="image-runtime-actions">
                {/* Two display modes for the same not-available state:
                  * (a) Fresh / broken install → offer Install GPU runtime.
                  * (b) Install just completed this session but backend
                  *     hasn't been restarted yet → PYTHONPATH in the
                  *     running backend is frozen from spawn time, so
                  *     find_spec still can't see the freshly-installed
                  *     torch. The user doesn't need to install again,
                  *     they need to restart. Keep the restart button
                  *     prominent and explain why. */}
                {gpuBundleJob?.phase === "done" && gpuBundleJob.requiresRestart ? (
                  <>
                    <p className="muted-text">
                      GPU runtime installed to{" "}
                      <code>{gpuBundleJob.targetDir ?? "extras"}</code>. The running backend
                      still has its old import cache — click Restart Backend to activate the
                      new runtime, then image generation will use your GPU.
                    </p>
                    <div className="button-row">
                      <button
                        className="primary-button"
                        type="button"
                        onClick={() => onRestartServer()}
                        disabled={busy}
                      >
                        {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend to activate"}
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="muted-text">
                      Install the GPU image runtime (torch + diffusers + accelerate + transformers,
                      ~2.5 GB) to enable real local generation. Writes to a persistent user-local
                      folder so app updates don't wipe it.
                    </p>
                    <div className="button-row">
                      <button
                        className="primary-button"
                        type="button"
                        onClick={() => void handleInstallImageRuntime()}
                        disabled={installingImageRuntime || !backendOnline}
                      >
                        {installingImageRuntime ? "Installing..." : "Install GPU runtime"}
                      </button>
                      <button className="secondary-button" type="button" onClick={() => onRestartServer()} disabled={busy}>
                        {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend"}
                      </button>
                    </div>
                  </>
                )}
              </div>
              <InstallLogPanel job={gpuBundleJob} />
            </>
          ) : null}
        </div>
      </Panel>

      <Panel
        title="Prompt"
        subtitle="Choose a model, set the aspect ratio and quality, then generate into the local gallery."
        className="image-studio-form-panel"
        actions={
          <button
            className="primary-button"
            type="button"
            onClick={() => onSubmitImageGeneration()}
            disabled={imageGenerateDisabled}
            title={imageGenerateTitle}
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
                <option value="">No models installed — download one from Discover</option>
              )}
            </select>
          </label>

          {!hasInstalledImageModels ? (
            <div className="callout image-callout">
              <p>
                You don't have any image models downloaded yet. Head to Image Discover to
                browse and install one, then come back here to generate.
              </p>
              <div className="button-row">
                <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-discover")}>
                  Open Image Discover
                </button>
              </div>
            </div>
          ) : null}

          {!selectedImageVariant?.availableLocally && selectedImageVariant ? (
            <div className="callout image-callout">
              <p>
                {selectedImageDownloadFailed
                  ? `${selectedImageVariant.name} did not finish downloading correctly. ChaosEngineAI only found a partial local snapshot, so it cannot load the real image pipeline yet.`
                  : selectedImageDownloadPaused
                  ? `${selectedImageVariant.name} is partially downloaded. Resume when you're ready and ChaosEngineAI will continue from the files already on disk.`
                  : selectedImageDownloadComplete
                  ? `${selectedImageVariant.name} finished downloading. The installed-model scan will refresh automatically.`
                  : `${selectedImageVariant.name} is not installed locally. Download it from Discover to enable local generation.`}
              </p>
              {selectedImageDownloadFailed && selectedImageDownload?.error ? (
                <>
                  <p className="muted-text">{selectedImageFriendlyDownloadError}</p>
                  {selectedImageNeedsGatedAccess ? (
                    <div className="button-row">
                      <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(selectedImageVariant.link)}>
                        Hugging Face
                      </button>
                      <button className="secondary-button" type="button" onClick={() => onActiveTabChange("settings")}>
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
                    <button className="secondary-button" type="button" onClick={() => onCancelImageDownload(selectedImageVariant.repo)}>
                      Pause
                    </button>
                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(selectedImageVariant.repo)}>
                      Cancel
                    </button>
                  </>
                ) : selectedImageDownloadPaused ? (
                  <>
                    <span className="badge warning">{downloadProgressLabel(selectedImageDownload)}</span>
                    <button className="secondary-button" type="button" onClick={() => onImageDownload(selectedImageVariant.repo)}>
                      Resume
                    </button>
                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(selectedImageVariant.repo)}>
                      Delete
                    </button>
                  </>
                ) : selectedImageDownloadFailed ? (
                  <>
                    <button className="secondary-button" type="button" onClick={() => onImageDownload(selectedImageVariant.repo)}>
                      Retry Download
                    </button>
                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(selectedImageVariant.repo)}>
                      Delete
                    </button>
                  </>
                ) : selectedImageDownloadComplete ? (
                  <span className="badge success">Download complete</span>
                ) : (
                  <>
                    <button className="secondary-button" type="button" onClick={() => onImageDownload(selectedImageVariant.repo)}>
                      Download Model
                    </button>
                    {selectedImageVariant.hasLocalData ? (
                      <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(selectedImageVariant.repo)}>
                        Delete
                      </button>
                    ) : null}
                  </>
                )}
                <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(selectedImageVariant.link)}>
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
              onChange={(event) => onImagePromptChange(event.target.value)}
            />
          </label>

          <label>
            Negative prompt
            <textarea
              className="text-input prompt-area prompt-area--secondary"
              rows={3}
              placeholder="blurry, deformed hands, extra limbs, overexposed"
              value={imageNegativePrompt}
              onChange={(event) => onImageNegativePromptChange(event.target.value)}
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
                  onClick={() => onApplyImageRatioPreset(preset.id)}
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
                  onClick={() => onApplyImageQuality(preset.id)}
                >
                  <strong>{preset.label}</strong>
                  <span>{preset.hint}</span>
                </button>
              ))}
              <button
                className={imageDraftMode ? "pill-button active" : "pill-button"}
                type="button"
                onClick={() => onImageDraftModeChange(!imageDraftMode)}
                title="Force a 512px long-edge render for fast prompt iteration. Output saves at the draft size — disable for a full-resolution final pass."
              >
                <strong>Preview</strong>
                <span>{imageDraftMode ? "512px · on" : "Draft @ 512px"}</span>
              </button>
            </div>
          </div>

          {selectedImageVariant && !isFlowMatchingRepo(selectedImageVariant.repo) ? (
            <div className="control-stack">
              <span className="eyebrow">Sampler</span>
              <select
                className="text-input"
                value={imageSampler}
                onChange={(event) => onImageSamplerChange(event.target.value as ImageSamplerId)}
                title="Scheduler / sampler algorithm. Hidden for FLUX, SD3 and other flow-matching models where swapping produces noise."
              >
                {IMAGE_SAMPLERS.map((sampler) => (
                  <option key={sampler.id} value={sampler.id}>
                    {sampler.label} · {sampler.hint}
                  </option>
                ))}
              </select>
            </div>
          ) : null}

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
                onChange={(event) => onImageWidthChange(Number(event.target.value) || 1024)}
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
                onChange={(event) => onImageHeightChange(Number(event.target.value) || 1024)}
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
                onChange={(event) => onImageStepsChange(Number(event.target.value) || 24)}
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
                onChange={(event) => onImageGuidanceChange(Number(event.target.value) || 6)}
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
                onChange={(event) => onImageBatchSizeChange(Math.max(1, Math.min(4, Number(event.target.value) || 1)))}
              />
            </label>
            <label className="checkbox-card">
              <span className="checkbox-card-label">Random seed</span>
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
                    ? "Likely to crash the backend"
                    : "Heads up — may struggle on this device"}
                  :
                </strong>{" "}
                {imageSafety.reason}
              </p>
              {imageSafety.modelFootprintGb > 0 ? (
                <p className="muted-text">
                  Model ≈ {formatImageGb(imageSafety.modelFootprintGb)} · this run peak ≈ {" "}
                  {formatImageGb(imageSafety.estimatedPeakGb)} of ~{formatImageGb(imageSafety.deviceMemoryGb)} total.
                </p>
              ) : null}
              {imageSafety.suggestion ? (
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={handleApplySafeImageSettings}
                    disabled={imageBusy}
                    title={`Apply ${imageSafety.suggestion.label}`}
                  >
                    Use safer settings ({imageSafety.suggestion.label})
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
                    Browse smaller models
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
                    Generate anyway — I accept that the backend may crash.
                  </span>
                </label>
              ) : null}
            </div>
          ) : null}

          {!imageUseRandomSeed ? (
            <label>
              Seed
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
        title="Recent Outputs"
        subtitle={imageOutputs.length > 0 ? `${recentImageOutputs.length} newest of ${imageOutputs.length} saved generations` : "Generated images will appear here"}
        className="image-gallery-panel"
        actions={
          <button className="secondary-button" type="button" onClick={() => onOpenImageGallery()}>
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
            Showing the newest {recentImageOutputs.length} saved images here. Open Image Gallery to browse everything,
            filter by model, and manage older runs.
          </p>
        ) : null}
      </Panel>
    </div>
  );
}
