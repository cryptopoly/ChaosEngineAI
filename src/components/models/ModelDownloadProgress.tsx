import { downloadProgressLabel, downloadProgressPercent, downloadSizeTooltip } from "../../utils";
import type { ResolvedModelStatus } from "./modelStatus";

/**
 * The progress line under a row that is downloading or paused: bar, percent label and "downloaded / total GB".
 * Reads the same `DownloadStatus` the status icon tooltip reads, so every Models page shows it alike.
 */
export function ModelDownloadProgress({ status }: { status: ResolvedModelStatus }) {
  if (status.state !== "downloading" && status.state !== "paused") return null;
  const download = status.download;
  const paused = status.state === "paused";
  const percent = downloadProgressPercent(download);
  const label = downloadProgressLabel(download ?? (paused ? { repo: "", state: "cancelled", progress: 0, downloadedGb: 0, totalGb: null, error: null } : null));
  const size = downloadSizeTooltip(download);
  return (
    <div className={`models-card__progress${paused ? " is-paused" : ""}`} data-testid="model-download-progress">
      <div
        className={`progress-track models-card__progress-track${percent === null && !paused ? " is-indeterminate" : ""}`}
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent ?? undefined}
      >
        <div className="progress-fill" style={{ width: `${percent ?? (paused ? 0 : 100)}%` }} />
      </div>
      <div className="models-card__progress-meta">
        <span>{label}</span>
        {size ? <span className="muted-text">{size}</span> : null}
      </div>
    </div>
  );
}
