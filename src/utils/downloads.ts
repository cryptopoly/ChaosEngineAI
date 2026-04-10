import type { DownloadStatus } from "../api";
import { number } from "./format";

export function buildDownloadStatusMap(statuses: DownloadStatus[]): Record<string, DownloadStatus> {
  return Object.fromEntries(statuses.map((status) => [status.repo, status]));
}

export function pendingDownloadStatus(repo: string, existing?: DownloadStatus | null): DownloadStatus {
  return {
    repo,
    state: "downloading",
    progress: Math.max(0, existing?.progress ?? 0),
    downloadedGb: Math.max(0, existing?.downloadedGb ?? 0),
    totalGb: typeof existing?.totalGb === "number" ? existing.totalGb : null,
    error: null,
  };
}

export function failedDownloadStatus(repo: string, error: string): DownloadStatus {
  return {
    repo,
    state: "failed",
    progress: 0,
    downloadedGb: 0,
    totalGb: null,
    error,
  };
}

export function downloadProgressLabel(download?: DownloadStatus | null): string {
  if (!download) return "Preparing download...";
  const prefix = download.state === "cancelled" ? "Paused" : download.state === "downloading" ? "Downloading" : "";
  if (!prefix) return "";
  const totalGb = typeof download.totalGb === "number" && download.totalGb > 0 ? download.totalGb : null;
  const downloadedGb = Math.max(0, download.downloadedGb ?? 0);
  if (totalGb !== null) {
    const pct = Math.max(downloadedGb > 0 ? 1 : 0, Math.round((download.progress ?? 0) * 100));
    return `${prefix} ${pct}%`;
  }
  if (downloadedGb > 0) {
    return download.state === "cancelled" ? `${prefix} at ${number(downloadedGb)} GB` : `${prefix} ${number(downloadedGb)} GB`;
  }
  return download.state === "cancelled" ? "Paused" : "Preparing download...";
}

export function downloadSizeTooltip(download?: DownloadStatus | null): string {
  if (!download) return "";
  const downloadedGb = Math.max(0, download.downloadedGb ?? 0);
  const totalGb = typeof download.totalGb === "number" && download.totalGb > 0 ? download.totalGb : null;
  if (totalGb !== null && downloadedGb > 0) return `${number(downloadedGb)} / ${number(totalGb)} GB`;
  if (totalGb !== null) return `${number(totalGb)} GB total`;
  if (downloadedGb > 0) return `${number(downloadedGb)} GB downloaded`;
  return "";
}
