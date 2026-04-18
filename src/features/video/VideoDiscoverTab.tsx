import { Panel } from "../../components/Panel";
import type { DownloadStatus } from "../../api";
import type {
  TabId,
  TauriBackendInfo,
  VideoModelVariant,
  VideoRuntimeStatus,
} from "../../types";
import type { VideoDiscoverTaskFilter } from "../../types/video";
import {
  downloadProgressLabel,
  downloadSizeTooltip,
  number,
  sizeLabel,
} from "../../utils";

export interface VideoDiscoverTabProps {
  combinedVideoDiscoverResults: VideoModelVariant[];
  videoDiscoverSearchInput: string;
  onVideoDiscoverSearchInputChange: (value: string) => void;
  videoDiscoverTaskFilter: VideoDiscoverTaskFilter;
  onVideoDiscoverTaskFilterChange: (value: VideoDiscoverTaskFilter) => void;
  videoDiscoverHasActiveFilters: boolean;
  videoDiscoverSearchQuery: string;
  videoRuntimeStatus: VideoRuntimeStatus;
  tauriBackend: TauriBackendInfo | null;
  busy: boolean;
  busyAction: string | null;
  activeVideoDownloads: Record<string, DownloadStatus>;
  selectedVideoVariant: VideoModelVariant | null;
  onActiveTabChange: (tab: TabId) => void;
  onOpenVideoStudio: (modelId?: string) => void;
  onVideoDownload: (repo: string) => void;
  onCancelVideoDownload: (repo: string) => void;
  onDeleteVideoDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRestartServer: () => void;
}

export function VideoDiscoverTab({
  combinedVideoDiscoverResults,
  videoDiscoverSearchInput,
  onVideoDiscoverSearchInputChange,
  videoDiscoverTaskFilter,
  onVideoDiscoverTaskFilterChange,
  videoDiscoverHasActiveFilters,
  videoDiscoverSearchQuery,
  videoRuntimeStatus,
  tauriBackend,
  busy,
  busyAction,
  activeVideoDownloads,
  selectedVideoVariant,
  onActiveTabChange,
  onOpenVideoStudio,
  onVideoDownload,
  onCancelVideoDownload,
  onDeleteVideoDownload,
  onOpenExternalUrl,
  onRestartServer,
}: VideoDiscoverTabProps) {
  return (
    <div className="image-discover-stack">
      <Panel
        title="Video Discover"
        subtitle={`${combinedVideoDiscoverResults.length} curated video models`}
      >
        <div className="image-hero">
          <div>
            <h3>Browse and download video models for local generation.</h3>
            <p className="muted-text">
              First-wave engines only. Download any model to use it in Video Studio.
            </p>
          </div>
          <div className="image-hero-actions">
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("video-models")}>
              Installed Models
            </button>
            <button className="primary-button" type="button" onClick={() => onOpenVideoStudio(selectedVideoVariant?.id)}>
              Open Studio
            </button>
          </div>
        </div>

        <div className="callout image-callout image-runtime-callout">
          <p>{videoRuntimeStatus.message}</p>
          <div className="chip-row">
            <span className={`badge ${videoRuntimeStatus.realGenerationAvailable ? "success" : "warning"}`}>
              {videoRuntimeStatus.realGenerationAvailable
                ? "Real engine ready"
                : videoRuntimeStatus.activeEngine === "unavailable"
                  ? "Runtime unavailable"
                  : "Fallback active"}
            </span>
            <span className="badge muted">Engine: {videoRuntimeStatus.activeEngine}</span>
            {videoRuntimeStatus.device ? <span className="badge muted">Device: {videoRuntimeStatus.device}</span> : null}
            {(videoRuntimeStatus.missingDependencies ?? []).slice(0, 4).map((dependency) => (
              <span key={dependency} className="badge subtle">{dependency}</span>
            ))}
          </div>
          {!videoRuntimeStatus.realGenerationAvailable ? (
            <div className="image-runtime-actions">
              {videoRuntimeStatus.pythonExecutable ?? tauriBackend?.pythonExecutable ? (
                <span className="mono-text muted-text">
                  Backend Python: {videoRuntimeStatus.pythonExecutable ?? tauriBackend?.pythonExecutable}
                </span>
              ) : null}
              {tauriBackend?.managedByTauri ? (
                <button className="secondary-button" type="button" onClick={() => onRestartServer()} disabled={busy}>
                  {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend"}
                </button>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="image-discover-filter-row">
          <label className="image-discover-search">
            Search
            <input
              className="text-input"
              type="search"
              value={videoDiscoverSearchInput}
              onChange={(event) => onVideoDiscoverSearchInputChange(event.target.value)}
              placeholder="Search LTX, Wan, Mochi, provider, tags..."
            />
          </label>
          <label>
            Task
            <select
              className="text-input"
              value={videoDiscoverTaskFilter}
              onChange={(event) => onVideoDiscoverTaskFilterChange(event.target.value as VideoDiscoverTaskFilter)}
            >
              <option value="all">All tasks</option>
              <option value="txt2video">Text to video</option>
              <option value="img2video">Image to video</option>
              <option value="video2video">Video to video</option>
            </select>
          </label>
          <div className="image-discover-filter-actions">
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                onVideoDiscoverSearchInputChange("");
                onVideoDiscoverTaskFilterChange("all");
              }}
              disabled={!videoDiscoverHasActiveFilters}
            >
              Clear Filters
            </button>
          </div>
        </div>

        <div className="image-discover-results-summary">
          <span>
            {combinedVideoDiscoverResults.length} model{combinedVideoDiscoverResults.length !== 1 ? "s" : ""}
          </span>
          {videoDiscoverSearchQuery ? (
            <span className="badge subtle">Search: {videoDiscoverSearchInput.trim()}</span>
          ) : null}
          {videoDiscoverTaskFilter !== "all" ? (
            <span className="badge muted">Task: {videoDiscoverTaskFilter}</span>
          ) : null}
        </div>
      </Panel>

      {combinedVideoDiscoverResults.length === 0 ? (
        <Panel title="Video Models" subtitle="No models match the current filters" className="image-discover-section-panel">
          <div className="empty-state image-empty-state">
            <p>Try broadening the filters or search terms.</p>
          </div>
        </Panel>
      ) : (
        <div className="image-discover-grid image-discover-grid--latest">
          {combinedVideoDiscoverResults.map((variant) => {
            const downloadState = activeVideoDownloads[variant.repo];
            const isDownloading = downloadState?.state === "downloading";
            const isPaused = downloadState?.state === "cancelled";
            const isDownloadComplete = downloadState?.state === "completed";
            const isDownloadFailed = downloadState?.state === "failed";
            const isComplete = variant.availableLocally || isDownloadComplete;
            const isPartial = !isComplete && variant.hasLocalData;
            const canDeleteLocalData = Boolean(
              isComplete || isDownloadComplete || isPaused || isDownloadFailed || isPartial,
            );
            return (
              <article key={variant.id} className="image-library-card">
                <div className="image-library-card-head">
                  <div>
                    <h3>{variant.name}</h3>
                    <p>{variant.familyName ?? variant.provider}</p>
                  </div>
                  {isComplete ? (
                    <span className="badge success">Installed</span>
                  ) : isDownloading ? (
                    <span className="badge accent" title={downloadSizeTooltip(downloadState)}>
                      {downloadProgressLabel(downloadState)}
                    </span>
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
                <p className="muted-text">{variant.note}</p>
                {isDownloadFailed && downloadState?.error ? (
                  <p className="muted-text" style={{ color: "var(--error, #e26d6d)" }}>{downloadState.error}</p>
                ) : null}
                <div className="button-row">
                  {isComplete ? (
                    <button className="primary-button" type="button" onClick={() => onOpenVideoStudio(variant.id)}>
                      Generate
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
                  {isDownloading || canDeleteLocalData ? (
                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteVideoDownload(variant.repo)}>
                      {isDownloading ? "Cancel" : "Delete"}
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
    </div>
  );
}
