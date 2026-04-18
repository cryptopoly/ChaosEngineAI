import { Panel } from "../../components/Panel";
import type { DownloadStatus } from "../../api";
import type {
  TabId,
  VideoModelFamily,
  VideoModelVariant,
  VideoRuntimeStatus,
} from "../../types";
import { downloadProgressLabel, number, sizeLabel } from "../../utils";

export interface VideoModelsTabProps {
  installedVideoVariants: VideoModelVariant[];
  videoCatalog: VideoModelFamily[];
  activeVideoDownloads: Record<string, DownloadStatus>;
  videoRuntimeStatus: VideoRuntimeStatus;
  videoBusy: boolean;
  videoBusyLabel: string | null;
  loadedVideoVariant: VideoModelVariant | null;
  fileRevealLabel: string;
  onActiveTabChange: (tab: TabId) => void;
  onOpenVideoStudio: (modelId?: string) => void;
  onVideoDownload: (repo: string) => void;
  onCancelVideoDownload: (repo: string) => void;
  onDeleteVideoDownload: (repo: string) => void;
  onPreloadVideoModel: (variant: VideoModelVariant) => void;
  onUnloadVideoModel: (variant?: VideoModelVariant) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
}

export function VideoModelsTab({
  installedVideoVariants,
  videoCatalog,
  activeVideoDownloads,
  videoRuntimeStatus,
  videoBusy,
  videoBusyLabel,
  loadedVideoVariant,
  fileRevealLabel,
  onActiveTabChange,
  onOpenVideoStudio,
  onVideoDownload,
  onCancelVideoDownload,
  onDeleteVideoDownload,
  onPreloadVideoModel,
  onUnloadVideoModel,
  onOpenExternalUrl,
  onRevealPath,
}: VideoModelsTabProps) {
  return (
    <div className="content-grid image-page-grid">
      <Panel
        title="Installed Video Models"
        subtitle={installedVideoVariants.length > 0
          ? `${installedVideoVariants.length} model${installedVideoVariants.length !== 1 ? "s" : ""} with local data`
          : "No video models detected locally yet"}
        className="span-2"
        actions={
          <button className="secondary-button" type="button" onClick={() => onActiveTabChange("video-discover")}>
            Browse Catalog
          </button>
        }
      >
        {installedVideoVariants.length === 0 ? (
          <div className="empty-state image-empty-state">
            <p>Download a video model from Video Discover to get started.</p>
          </div>
        ) : (
          <div className="image-library-grid">
            {installedVideoVariants.map((variant) => {
              const family = videoCatalog.find((item) =>
                item.variants.some((candidate) => candidate.id === variant.id),
              );
              const isComplete = variant.availableLocally;
              const isPartial = !isComplete && variant.hasLocalData;
              const downloadState = activeVideoDownloads[variant.repo];
              const isDownloading = downloadState?.state === "downloading";
              const isPaused = downloadState?.state === "cancelled";
              const isDownloadComplete = downloadState?.state === "completed";
              const isDownloadFailed = downloadState?.state === "failed";
              const canDeleteLocalData = Boolean(
                isComplete || isDownloadComplete || isPaused || isDownloadFailed || isPartial,
              );
              const isLoadedInMemory = loadedVideoVariant?.id === variant.id;
              const canPreload = isComplete && videoRuntimeStatus.realGenerationAvailable && !isLoadedInMemory;
              return (
                <article key={variant.id} className="image-library-card">
                  <div className="image-library-card-head">
                    <div>
                      <h3>{variant.name}</h3>
                      <p>{family?.name ?? variant.provider}</p>
                    </div>
                    {isLoadedInMemory ? (
                      <span className="badge accent">In Memory</span>
                    ) : isComplete || isDownloadComplete ? (
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
                    <span>{number(variant.defaultDurationSeconds)}s clip</span>
                    {variant.styleTags.slice(0, 3).map((tag) => (
                      <span key={tag} className="badge subtle">{tag}</span>
                    ))}
                  </div>
                  {isDownloadFailed && downloadState?.error ? (
                    <p className="muted-text" style={{ color: "var(--error, #e26d6d)" }}>{downloadState.error}</p>
                  ) : null}
                  <div className="button-row">
                    {isComplete || isDownloadComplete ? (
                      <button className="primary-button" type="button" onClick={() => onOpenVideoStudio(variant.id)}>
                        Open in Studio
                      </button>
                    ) : isDownloading ? (
                      <button className="secondary-button" type="button" onClick={() => onCancelVideoDownload(variant.repo)}>
                        Pause
                      </button>
                    ) : isPaused ? (
                      <button className="secondary-button" type="button" onClick={() => onVideoDownload(variant.repo)}>
                        Resume
                      </button>
                    ) : (
                      <button className="secondary-button" type="button" onClick={() => onVideoDownload(variant.repo)}>
                        {isDownloadFailed ? "Retry" : isPartial ? "Resume Download" : "Download"}
                      </button>
                    )}
                    {canPreload ? (
                      <button
                        className="secondary-button"
                        type="button"
                        disabled={videoBusy}
                        onClick={() => onPreloadVideoModel(variant)}
                      >
                        {videoBusy && videoBusyLabel?.includes(variant.name) ? "Loading..." : "Load into memory"}
                      </button>
                    ) : null}
                    {isLoadedInMemory ? (
                      <button
                        className="secondary-button"
                        type="button"
                        disabled={videoBusy}
                        onClick={() => onUnloadVideoModel(variant)}
                      >
                        {videoBusy && videoBusyLabel?.includes("Unloading") ? "Unloading..." : "Unload"}
                      </button>
                    ) : null}
                    {isDownloading || canDeleteLocalData ? (
                      <button className="secondary-button danger-button" type="button" onClick={() => onDeleteVideoDownload(variant.repo)}>
                        {isDownloading ? "Cancel" : "Delete"}
                      </button>
                    ) : null}
                    {variant.localPath ? (
                      <button
                        className="secondary-button icon-button"
                        type="button"
                        title={fileRevealLabel}
                        onClick={() => onRevealPath(variant.localPath as string)}
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                          <polyline points="15 3 21 3 21 9" />
                          <line x1="10" y1="14" x2="21" y2="3" />
                        </svg>
                      </button>
                    ) : null}
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
