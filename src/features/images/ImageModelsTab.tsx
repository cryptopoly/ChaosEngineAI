import { Panel } from "../../components/Panel";
import type { DownloadStatus } from "../../api";
import type {
  ImageModelFamily,
  ImageModelVariant,
  TabId,
} from "../../types";
import {
  sizeLabel,
  downloadProgressLabel,
} from "../../utils";

export interface ImageModelsTabProps {
  installedImageVariants: ImageModelVariant[];
  imageCatalog: ImageModelFamily[];
  activeImageDownloads: Record<string, DownloadStatus>;
  onActiveTabChange: (tab: TabId) => void;
  onOpenImageStudio: (modelId?: string) => void;
  onImageDownload: (repo: string) => void;
  onCancelImageDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
}

export function ImageModelsTab({
  installedImageVariants,
  imageCatalog,
  activeImageDownloads,
  onActiveTabChange,
  onOpenImageStudio,
  onImageDownload,
  onCancelImageDownload,
  onOpenExternalUrl,
}: ImageModelsTabProps) {
  return (
    <div className="content-grid image-page-grid">
      <Panel
        title="Installed Image Models"
        subtitle={installedImageVariants.length > 0
          ? `${installedImageVariants.length} model${installedImageVariants.length !== 1 ? "s" : ""} with local data`
          : "No image models detected locally yet"}
        className="span-2"
        actions={
          <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-discover")}>
            Browse Catalog
          </button>
        }
      >
        {installedImageVariants.length === 0 ? (
          <div className="empty-state image-empty-state">
            <p>Download an image model from Image Discover to get started.</p>
          </div>
        ) : (
          <div className="image-library-grid">
            {installedImageVariants.map((variant) => {
              const family = imageCatalog.find((item) => item.variants.some((candidate) => candidate.id === variant.id));
              const isComplete = variant.availableLocally;
              const isPartial = !isComplete && variant.hasLocalData;
              const downloadState = activeImageDownloads[variant.repo];
              const isDownloading = downloadState?.state === "downloading";
              const isPaused = downloadState?.state === "cancelled";
              const isDownloadComplete = downloadState?.state === "completed";
              const isDownloadFailed = downloadState?.state === "failed";
              return (
                <article key={variant.id} className="image-library-card">
                  <div className="image-library-card-head">
                    <div>
                      <h3>{variant.name}</h3>
                      <p>{family?.name ?? variant.provider}</p>
                    </div>
                    {isComplete || isDownloadComplete ? (
                      <span className="badge success">Installed</span>
                    ) : isDownloading ? (
                      <span className="badge accent">{downloadProgressLabel(downloadState)}</span>
                    ) : isPaused ? (
                      <span className="badge warning">{downloadProgressLabel(downloadState)}</span>
                    ) : isDownloadFailed ? (
                      <span className="badge warning">Download Failed</span>
                    ) : isPartial ? (
                      <span className="badge warning">Incomplete</span>
                    ) : null}
                  </div>
                  <div className="image-library-stats">
                    <span>{sizeLabel(variant.sizeGb)}</span>
                    <span>{variant.recommendedResolution}</span>
                    {variant.styleTags.slice(0, 3).map((tag) => (
                      <span key={tag} className="badge subtle">{tag}</span>
                    ))}
                  </div>
                  {isDownloadFailed && downloadState?.error ? (
                    <p className="muted-text" style={{ color: "var(--error, #e26d6d)" }}>{downloadState.error}</p>
                  ) : null}
                  <div className="button-row">
                    {isComplete || isDownloadComplete ? (
                      <button className="primary-button" type="button" onClick={() => onOpenImageStudio(variant.id)}>
                        Generate
                      </button>
                    ) : isDownloading ? (
                      <button className="secondary-button" type="button" onClick={() => onCancelImageDownload(variant.repo)}>
                        Pause
                      </button>
                    ) : isPaused ? (
                      <button className="secondary-button" type="button" onClick={() => onImageDownload(variant.repo)}>
                        Resume
                      </button>
                    ) : (
                      <button className="secondary-button" type="button" onClick={() => onImageDownload(variant.repo)}>
                        {isDownloadFailed ? "Retry" : isPartial ? "Resume Download" : "Download"}
                      </button>
                    )}
                    <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(variant.link)}>
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
