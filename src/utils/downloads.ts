import i18next from "i18next";
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
  const preparing = () => i18next.t("library:download.preparing", { defaultValue: "Preparing download..." });
  const pausedLabel = () => i18next.t("library:download.paused", { defaultValue: "Paused" });
  if (!download) return preparing();
  const paused = download.state === "cancelled";
  if (!paused && download.state !== "downloading") return "";
  const totalGb = typeof download.totalGb === "number" && download.totalGb > 0 ? download.totalGb : null;
  const downloadedGb = Math.max(0, download.downloadedGb ?? 0);
  if (downloadedGb <= 0 && (download.progress ?? 0) <= 0) {
    return paused ? pausedLabel() : preparing();
  }
  if (totalGb !== null) {
    const pct = Math.max(downloadedGb > 0 ? 1 : 0, Math.round((download.progress ?? 0) * 100));
    return paused
      ? i18next.t("library:download.pausedPercent", { percent: pct, defaultValue: "Paused {percent}%" })
      : i18next.t("library:download.downloadingPercent", { percent: pct, defaultValue: "Downloading {percent}%" });
  }
  if (downloadedGb > 0) {
    return paused
      ? i18next.t("library:download.pausedAtSize", { size: number(downloadedGb), defaultValue: "Paused at {size} GB" })
      : i18next.t("library:download.downloadingSize", { size: number(downloadedGb), defaultValue: "Downloading {size} GB" });
  }
  return paused ? pausedLabel() : preparing();
}

export function downloadSizeTooltip(download?: DownloadStatus | null): string {
  if (!download) return "";
  const downloadedGb = Math.max(0, download.downloadedGb ?? 0);
  const totalGb = typeof download.totalGb === "number" && download.totalGb > 0 ? download.totalGb : null;
  if (totalGb !== null && downloadedGb > 0) return `${number(downloadedGb)} / ${number(totalGb)} GB`;
  if (totalGb !== null) return i18next.t("library:download.sizeTotalGb", { size: number(totalGb), defaultValue: "{size} GB total" });
  if (downloadedGb > 0) return i18next.t("library:download.sizeDownloadedGb", { size: number(downloadedGb), defaultValue: "{size} GB downloaded" });
  return "";
}
