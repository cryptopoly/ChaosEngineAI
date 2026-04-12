import { Panel } from "../../components/Panel";
import { ImageOutputCard } from "../../components/ImageOutputCard";
import type { DownloadStatus } from "../../api";
import type {
  ImageModelFamily,
  ImageModelVariant,
  ImageOutputArtifact,
  ImageQualityPreset,
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
import { IMAGE_RATIO_PRESETS, IMAGE_QUALITY_PRESETS } from "../../constants";

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
  onPreloadImageModel: (variant: ImageModelVariant) => void;
  onUnloadImageModel: (variant?: ImageModelVariant) => void;
  onImageDownload: (repo: string) => void;
  onCancelImageDownload: (repo: string) => void;
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
  onPreloadImageModel,
  onUnloadImageModel,
  onImageDownload,
  onCancelImageDownload,
  onOpenExternalUrl,
  onRestartServer,
  onUseSameImageSettings,
  onVaryImageSeed,
  onRevealPath,
  onDeleteImageArtifact,
}: ImageStudioTabProps) {
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
            {imageRuntimeStatus.device ? <span className="badge muted">Device: {imageRuntimeStatus.device}</span> : null}
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
            <div className="image-runtime-actions">
              <p className="muted-text">
                {imageRuntimeStatus.activeEngine === "unavailable"
                  ? "Install the diffusers runtime to enable local image generation."
                  : "Restart the backend if you recently installed image packages."}
              </p>
              <div className="button-row">
                {tauriBackend?.managedByTauri ? (
                  <button className="secondary-button" type="button" onClick={() => onRestartServer()} disabled={busy}>
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
        subtitle="Choose a model, set the aspect ratio and quality, then generate into the local gallery."
        className="image-studio-form-panel"
        actions={
          <button
            className="primary-button"
            type="button"
            onClick={() => onSubmitImageGeneration()}
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
              onChange={(event) => onSelectedImageModelIdChange(event.target.value)}
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
                  </>
                ) : selectedImageDownloadPaused ? (
                  <>
                    <span className="badge warning">{downloadProgressLabel(selectedImageDownload)}</span>
                    <button className="secondary-button" type="button" onClick={() => onImageDownload(selectedImageVariant.repo)}>
                      Resume
                    </button>
                  </>
                ) : selectedImageDownloadComplete ? (
                  <span className="badge success">Download complete</span>
                ) : (
                  <button className="secondary-button" type="button" onClick={() => onImageDownload(selectedImageVariant.repo)}>
                    {selectedImageDownloadFailed ? "Retry Download" : "Download Model"}
                  </button>
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
