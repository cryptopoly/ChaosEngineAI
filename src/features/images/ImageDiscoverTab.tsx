import { Panel } from "../../components/Panel";
import { LatestImageDiscoverCard } from "../../components/LatestImageDiscoverCard";
import type { DownloadStatus } from "../../api";
import type {
  ImageModelVariant,
  ImageRuntimeStatus,
  TabId,
  TauriBackendInfo,
} from "../../types";
import type {
  ImageDiscoverTaskFilter,
  ImageDiscoverAccessFilter,
} from "../../types/image";

export interface ImageDiscoverTabProps {
  combinedImageDiscoverResults: ImageModelVariant[];
  imageDiscoverSearchInput: string;
  onImageDiscoverSearchInputChange: (value: string) => void;
  imageDiscoverTaskFilter: ImageDiscoverTaskFilter;
  onImageDiscoverTaskFilterChange: (value: ImageDiscoverTaskFilter) => void;
  imageDiscoverAccessFilter: ImageDiscoverAccessFilter;
  onImageDiscoverAccessFilterChange: (value: ImageDiscoverAccessFilter) => void;
  imageDiscoverHasActiveFilters: boolean;
  imageDiscoverSearchQuery: string;
  imageRuntimeStatus: ImageRuntimeStatus;
  tauriBackend: TauriBackendInfo | null;
  busy: boolean;
  busyAction: string | null;
  activeImageDownloads: Record<string, DownloadStatus>;
  selectedImageVariant: ImageModelVariant | null;
  onActiveTabChange: (tab: TabId) => void;
  onOpenImageStudio: (modelId?: string) => void;
  onImageDownload: (repo: string) => void;
  onCancelImageDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRestartServer: () => void;
}

export function ImageDiscoverTab({
  combinedImageDiscoverResults,
  imageDiscoverSearchInput,
  onImageDiscoverSearchInputChange,
  imageDiscoverTaskFilter,
  onImageDiscoverTaskFilterChange,
  imageDiscoverAccessFilter,
  onImageDiscoverAccessFilterChange,
  imageDiscoverHasActiveFilters,
  imageDiscoverSearchQuery,
  imageRuntimeStatus,
  tauriBackend,
  busy,
  busyAction,
  activeImageDownloads,
  selectedImageVariant,
  onActiveTabChange,
  onOpenImageStudio,
  onImageDownload,
  onCancelImageDownload,
  onOpenExternalUrl,
  onRestartServer,
}: ImageDiscoverTabProps) {
  return (
    <div className="image-discover-stack">
      <Panel
        title="Image Discover"
        subtitle={`${combinedImageDiscoverResults.length} models / live Hugging Face metadata`}
      >
        <div className="image-hero">
          <div>
            <h3>Browse and download image models for local generation.</h3>
            <p className="muted-text">
              Models are sorted by most recently updated. Download any model to use it in Image Studio.
            </p>
          </div>
          <div className="image-hero-actions">
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-models")}>
              Installed Models
            </button>
            <button className="primary-button" type="button" onClick={() => onOpenImageStudio(selectedImageVariant?.id)}>
              Open Studio
            </button>
          </div>
        </div>
        <div className="callout image-callout image-runtime-callout">
          <p>{imageRuntimeStatus.message}</p>
          <div className="chip-row">
            <span className={`badge ${imageRuntimeStatus.realGenerationAvailable ? "success" : "warning"}`}>
              {imageRuntimeStatus.realGenerationAvailable
                ? "Real engine ready"
                : imageRuntimeStatus.activeEngine === "unavailable"
                  ? "Runtime unavailable"
                  : "Fallback active"}
            </span>
            <span className="badge muted">Engine: {imageRuntimeStatus.activeEngine}</span>
            {imageRuntimeStatus.device ? <span className="badge muted">Device: {imageRuntimeStatus.device}</span> : null}
            {(imageRuntimeStatus.missingDependencies ?? []).slice(0, 4).map((dependency) => (
              <span key={dependency} className="badge subtle">{dependency}</span>
            ))}
          </div>
          {!imageRuntimeStatus.realGenerationAvailable ? (
            <div className="image-runtime-actions">
              {imageRuntimeStatus.pythonExecutable ?? tauriBackend?.pythonExecutable ? (
                <span className="mono-text muted-text">
                  Backend Python: {imageRuntimeStatus.pythonExecutable ?? tauriBackend?.pythonExecutable}
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
              value={imageDiscoverSearchInput}
              onChange={(event) => onImageDiscoverSearchInputChange(event.target.value)}
              placeholder="Search FLUX, SDXL, provider, task, tags, license..."
            />
          </label>
          <label>
            Task
            <select
              className="text-input"
              value={imageDiscoverTaskFilter}
              onChange={(event) => onImageDiscoverTaskFilterChange(event.target.value as ImageDiscoverTaskFilter)}
            >
              <option value="all">All tasks</option>
              <option value="txt2img">Text to image</option>
              <option value="img2img">Image to image</option>
              <option value="inpaint">Inpaint</option>
            </select>
          </label>
          <label>
            Access
            <select
              className="text-input"
              value={imageDiscoverAccessFilter}
              onChange={(event) => onImageDiscoverAccessFilterChange(event.target.value as ImageDiscoverAccessFilter)}
            >
              <option value="all">Open + gated</option>
              <option value="open">Open only</option>
              <option value="gated">Gated only</option>
            </select>
          </label>
          <div className="image-discover-filter-actions">
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                onImageDiscoverSearchInputChange("");
                onImageDiscoverTaskFilterChange("all");
                onImageDiscoverAccessFilterChange("all");
              }}
              disabled={!imageDiscoverHasActiveFilters}
            >
              Clear Filters
            </button>
          </div>
        </div>

        <div className="image-discover-results-summary">
          <span>
            {combinedImageDiscoverResults.length} model{combinedImageDiscoverResults.length !== 1 ? "s" : ""} sorted by most recent
          </span>
          {imageDiscoverSearchQuery ? (
            <span className="badge subtle">Search: {imageDiscoverSearchInput.trim()}</span>
          ) : null}
          {imageDiscoverTaskFilter !== "all" ? (
            <span className="badge muted">Task: {imageDiscoverTaskFilter}</span>
          ) : null}
          {imageDiscoverAccessFilter !== "all" ? (
            <span className="badge muted">
              Access: {imageDiscoverAccessFilter === "open" ? "Open only" : "Gated only"}
            </span>
          ) : null}
        </div>
      </Panel>

      {combinedImageDiscoverResults.length === 0 ? (
        <Panel title="Image Models" subtitle="No models match the current filters" className="image-discover-section-panel">
          <div className="empty-state image-empty-state">
            <p>Try broadening the filters or search terms.</p>
          </div>
        </Panel>
      ) : (
        <div className="image-discover-grid image-discover-grid--latest">
          {combinedImageDiscoverResults.map((variant) => (
            <LatestImageDiscoverCard
              key={variant.id}
              variant={variant}
              downloadState={activeImageDownloads[variant.repo]}
              onDownload={(repo) => onImageDownload(repo)}
              onCancelDownload={(repo) => onCancelImageDownload(repo)}
              onOpenExternalUrl={(url) => onOpenExternalUrl(url)}
              onNavigateSettings={() => onActiveTabChange("settings")}
            />
          ))}
        </div>
      )}
    </div>
  );
}
