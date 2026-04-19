import { LiveProgress, type LiveProgressPhase } from "./LiveProgress";
import { number, findImageVariantById, formatImageTimestamp, formatImageAccessError, isGatedImageAccessError } from "../utils";
import { useGenerationProgress } from "../hooks/useGenerationProgress";
import type { ImageModelFamily, ImageModelVariant, ImageOutputArtifact, TabId } from "../types";

export interface ImageGenerationRunInfo {
  modelName: string;
  prompt: string;
  batchSize: number;
  steps: number;
  needsPipelineLoad: boolean;
}

export interface ImageGenerationModalProps {
  showImageGenerationModal: boolean;
  imageBusy: boolean;
  imageGenerationStartedAt: number | null;
  imageGenerationError: string | null;
  imageGenerationArtifacts: ImageOutputArtifact[];
  selectedImageGenerationArtifact: ImageOutputArtifact | null;
  imageGenerationRunInfo: ImageGenerationRunInfo | null;
  imageCatalog: ImageModelFamily[];
  imageSteps: number;
  selectedImageVariant: ImageModelVariant | null;
  onShowImageGenerationModalChange: (show: boolean) => void;
  onSelectedImageGenerationArtifactIdChange: (id: string) => void;
  onActiveTabChange: (tab: TabId) => void;
  onUseSameSettings: (artifact: ImageOutputArtifact, closeModal: boolean) => void;
  onVarySeed: (artifact: ImageOutputArtifact) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
  onDeleteArtifact: (artifactId: string) => void;
}

export function ImageGenerationModal({
  showImageGenerationModal,
  imageBusy,
  imageGenerationStartedAt,
  imageGenerationError,
  imageGenerationArtifacts,
  selectedImageGenerationArtifact,
  imageGenerationRunInfo,
  imageCatalog,
  imageSteps,
  selectedImageVariant,
  onShowImageGenerationModalChange,
  onSelectedImageGenerationArtifactIdChange,
  onActiveTabChange,
  onUseSameSettings,
  onVarySeed,
  onOpenExternalUrl,
  onRevealPath,
  onDeleteArtifact,
}: ImageGenerationModalProps) {
  // Hook ordering rule: react requires hooks to run in the same order on every
  // render, so we always invoke ``useGenerationProgress`` even when the modal
  // is hidden. The hook itself short-circuits when ``active`` is false.
  const realProgress = useGenerationProgress("image", imageBusy && Boolean(imageGenerationStartedAt));

  if (!showImageGenerationModal) {
    return null;
  }

  const activeArtifact = selectedImageGenerationArtifact;
  const runInfo = imageGenerationRunInfo;
  const activeArtifactVariant = activeArtifact ? findImageVariantById(imageCatalog, activeArtifact.modelId) : null;
  const activeArtifactRuntimeNote = formatImageAccessError(activeArtifact?.runtimeNote, activeArtifactVariant);
  const activeArtifactNeedsGatedAccess = isGatedImageAccessError(activeArtifact?.runtimeNote);
  const steps = runInfo?.steps ?? imageSteps;
  const batch = runInfo?.batchSize ?? 1;
  // Estimates are now a *fallback* — when the backend tracker is publishing
  // step counts, LiveProgress drives the bar from the real step / total. The
  // estimates still anchor the per-phase widths for the proportional fill so
  // diffusion doesn't visually consume the whole bar in one go.
  const diffuseEstimate = Math.max(30, Math.round(steps * 3 * batch));
  // Phase IDs are the same strings the backend ProgressTracker publishes
  // (loading / encoding / diffusing / decoding / saving) so LiveProgress can
  // match them up with the real-time signal. Keep the labels human-friendly.
  const imagePhases: LiveProgressPhase[] = [
    ...(runInfo?.needsPipelineLoad
      ? [{ id: "loading", label: "Loading model into memory", estimatedSeconds: 30 }]
      : []),
    { id: "encoding", label: "Encoding prompt", estimatedSeconds: 5 },
    {
      id: "diffusing",
      label: `Diffusing ${batch} image${batch > 1 ? "s" : ""}`,
      estimatedSeconds: diffuseEstimate,
    },
    { id: "decoding", label: "Decoding pixels", estimatedSeconds: 8 },
    { id: "saving", label: "Saving to output gallery", estimatedSeconds: 3 },
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
              realProgress={realProgress}
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
                        onClick={() => onSelectedImageGenerationArtifactIdChange(artifact.artifactId)}
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
                      onClick={() => void onOpenExternalUrl(activeArtifactVariant.link)}
                    >
                      Hugging Face
                    </button>
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => {
                        onShowImageGenerationModalChange(false);
                        onActiveTabChange("settings");
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
                    onClick={() => onUseSameSettings(activeArtifact, true)}
                  >
                    Use Same Settings
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => void onVarySeed(activeArtifact)}
                    disabled={imageBusy}
                  >
                    Vary Seed
                  </button>
                </div>
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => void onOpenExternalUrl(activeArtifact.imagePath ?? activeArtifact.previewUrl)}
                  >
                    Open
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => activeArtifact.imagePath ? void onRevealPath(activeArtifact.imagePath) : void onOpenExternalUrl(activeArtifact.previewUrl)}
                  >
                    Reveal
                  </button>
                  <button
                    className="secondary-button danger-button"
                    type="button"
                    onClick={() => void onDeleteArtifact(activeArtifact.artifactId)}
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
              onClick={() => onShowImageGenerationModalChange(false)}
            >
              {imageGenerationError ? "Close" : "Done"}
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
