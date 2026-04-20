import type { ImageModelVariant } from "../types";
import type { DownloadStatus } from "../api";
import {
  imagePrimarySizeLabel,
  imageSecondarySizeLabel,
  formatImageLicenseLabel,
  formatImageAccessError,
  formatReleaseLabel,
  isGatedImageAccessError,
} from "../utils/format";
import { downloadProgressLabel, downloadSizeTooltip } from "../utils/downloads";

export interface LatestImageDiscoverCardProps {
  variant: ImageModelVariant;
  downloadState?: DownloadStatus;
  fileRevealLabel?: string;
  onDownload: (repo: string) => void;
  onCancelDownload: (repo: string) => void;
  onDeleteDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onNavigateSettings: () => void;
  onRevealPath?: (path: string) => void;
}

export function LatestImageDiscoverCard({
  variant,
  downloadState,
  fileRevealLabel,
  onDownload,
  onCancelDownload,
  onDeleteDownload,
  onOpenExternalUrl,
  onNavigateSettings,
  onRevealPath,
}: LatestImageDiscoverCardProps) {
  const isDownloadPaused = downloadState?.state === "cancelled";
  const isDownloadComplete = downloadState?.state === "completed";
  const isDownloadFailed = downloadState?.state === "failed";
  const hasLocalData = Boolean(variant.hasLocalData || isDownloadComplete || isDownloadPaused || isDownloadFailed);
  const friendlyDownloadError = formatImageAccessError(downloadState?.error, variant);
  const needsGatedAccess = isGatedImageAccessError(downloadState?.error);
  return (
    <article key={variant.id} className="image-family-card image-family-card--latest">
      <div className="image-family-card-head">
        <div>
          <div className="image-family-title-row">
            <h3>{variant.name}</h3>
            <span className="badge muted">{variant.provider}</span>
            {variant.source === "curated" ? <span className="badge accent">Curated</span> : null}
            {!variant.availableLocally && isDownloadComplete ? <span className="badge success">Downloaded</span> : null}
            {isDownloadPaused ? <span className="badge warning">Paused</span> : null}
            {isDownloadFailed ? <span className="badge warning">Download Failed</span> : null}
          </div>
          <p>{variant.note}</p>
        </div>
        <span className="badge muted">{variant.updatedLabel ?? "Recently updated"}</span>
      </div>

      <div className="image-family-meta">
        <span>{imagePrimarySizeLabel(variant)}</span>
        {imageSecondarySizeLabel(variant) ? <span>{imageSecondarySizeLabel(variant)}</span> : null}
        <span>{variant.recommendedResolution}</span>
        {variant.pipelineTag ? <span>{variant.pipelineTag}</span> : null}
      </div>

      <div className="image-family-meta">
        {formatReleaseLabel(variant.releaseLabel, variant.releaseDate ?? variant.createdAt) ? (
          <span>{formatReleaseLabel(variant.releaseLabel, variant.releaseDate ?? variant.createdAt)}</span>
        ) : null}
        {variant.downloadsLabel ? <span>{variant.downloadsLabel}</span> : null}
        {variant.likesLabel ? <span>{variant.likesLabel}</span> : null}
        {variant.license ? <span>{formatImageLicenseLabel(variant.license)}</span> : null}
        {typeof variant.gated === "boolean" ? <span>{variant.gated ? "Gated access" : "Open access"}</span> : null}
      </div>

      <div className="chip-row">
        {variant.taskSupport.map((task) => (
          <span key={task} className="badge muted">{task}</span>
        ))}
        {variant.styleTags.map((tag) => (
          <span key={tag} className="badge subtle">{tag}</span>
        ))}
      </div>

      {isDownloadFailed && downloadState?.error ? (
        <div className="callout error image-callout">
          <p>{friendlyDownloadError}</p>
          {needsGatedAccess ? (
            <div className="button-row">
              <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(variant.link)}>
                Hugging Face
              </button>
              <button className="secondary-button" type="button" onClick={onNavigateSettings}>
                Settings
              </button>
            </div>
          ) : null}
          {friendlyDownloadError !== downloadState.error ? (
            <details className="debug-details">
              <summary>Technical details</summary>
              <p className="mono-text">{downloadState.error}</p>
            </details>
          ) : null}
        </div>
      ) : null}

      <div className="button-row">
        {variant.availableLocally ? (
          <span className="badge success">Installed</span>
        ) : downloadState?.state === "downloading" ? (
          <>
            <span className="badge accent" title={downloadSizeTooltip(downloadState)}>{downloadProgressLabel(downloadState)}</span>
            <button className="secondary-button" type="button" onClick={() => onCancelDownload(variant.repo)}>
              Pause
            </button>
            <button className="secondary-button danger-button" type="button" onClick={() => onDeleteDownload(variant.repo)}>
              Cancel
            </button>
          </>
        ) : isDownloadPaused ? (
          <>
            <span className="badge warning" title={downloadSizeTooltip(downloadState)}>{downloadProgressLabel(downloadState)}</span>
            <button className="secondary-button" type="button" onClick={() => onDownload(variant.repo)}>
              Resume
            </button>
            <button className="secondary-button danger-button" type="button" onClick={() => onDeleteDownload(variant.repo)}>
              Delete
            </button>
          </>
        ) : isDownloadFailed ? (
          <>
            <button className="secondary-button" type="button" onClick={() => onDownload(variant.repo)}>
              Retry Download
            </button>
            <button className="secondary-button danger-button" type="button" onClick={() => onDeleteDownload(variant.repo)}>
              Delete
            </button>
          </>
        ) : isDownloadComplete ? (
          <span className="badge success">Download complete</span>
        ) : (
          <>
            <button className="secondary-button" type="button" onClick={() => onDownload(variant.repo)}>
              Download
            </button>
            {hasLocalData ? (
              <button className="secondary-button danger-button" type="button" onClick={() => onDeleteDownload(variant.repo)}>
                Delete
              </button>
            ) : null}
          </>
        )}
        {variant.localPath && onRevealPath ? (
          <button
            className="secondary-button icon-button"
            type="button"
            title={fileRevealLabel ?? "Show in folder"}
            onClick={() => onRevealPath(variant.localPath as string)}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
              <polyline points="15 3 21 3 21 9" />
              <line x1="10" y1="14" x2="21" y2="3" />
            </svg>
          </button>
        ) : null}
        <button className="secondary-button icon-link-button" type="button" onClick={() => onOpenExternalUrl(variant.link)}>
          Hugging Face <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
        </button>
      </div>
    </article>
  );
}
