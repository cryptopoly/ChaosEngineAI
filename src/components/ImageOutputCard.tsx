import type { ImageModelFamily, ImageOutputArtifact } from "../types";
import {
  findImageVariantById,
  imageOrientation,
} from "../utils/images";
import {
  number,
  formatImageTimestamp,
  formatImageAccessError,
  isGatedImageAccessError,
} from "../utils/format";

export interface ImageOutputCardProps {
  artifact: ImageOutputArtifact;
  imageCatalog: ImageModelFamily[];
  imageBusy: boolean;
  onOpenExternalUrl: (url: string) => void;
  onUseSameSettings: (artifact: ImageOutputArtifact) => void;
  onVarySeed: (artifact: ImageOutputArtifact) => void;
  onRevealPath: (path: string) => void;
  onDelete: (artifactId: string) => void;
  onNavigateSettings: () => void;
}

export function ImageOutputCard({
  artifact,
  imageCatalog,
  imageBusy,
  onOpenExternalUrl,
  onUseSameSettings,
  onVarySeed,
  onRevealPath,
  onDelete,
  onNavigateSettings,
}: ImageOutputCardProps) {
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
            <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(artifactVariant.link)}>
              Hugging Face
            </button>
            <button className="secondary-button" type="button" onClick={onNavigateSettings}>
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
          <button className="secondary-button" type="button" onClick={() => onUseSameSettings(artifact)}>
            Use Same Settings
          </button>
          <button className="secondary-button" type="button" onClick={() => onVarySeed(artifact)} disabled={imageBusy}>
            Vary Seed
          </button>
        </div>
        <div className="button-row">
          <button
            className="secondary-button"
            type="button"
            onClick={() => onOpenExternalUrl(artifact.imagePath ?? artifact.previewUrl)}
          >
            Open
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={() => artifact.imagePath ? onRevealPath(artifact.imagePath) : onOpenExternalUrl(artifact.previewUrl)}
          >
            Reveal
          </button>
          <button className="secondary-button danger-button" type="button" onClick={() => onDelete(artifact.artifactId)}>
            Delete
          </button>
        </div>
      </div>
    </article>
  );
}
