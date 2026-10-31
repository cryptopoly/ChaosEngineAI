import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { StatusIcon } from "../ModelActionIcons";
import { downloadProgressLabel, downloadSizeTooltip } from "../../utils";
import type { ResolvedModelStatus } from "./modelStatus";
import type { ModelStatusBucket } from "./modelEntry";

export function modelStatusLabel(status: ResolvedModelStatus, t: TFunction<"library">): string {
  if (status.loaded) return t("models.status.loaded", { defaultValue: "Loaded in memory" });
  switch (status.state) {
    case "installed":
      return t("models.status.installed", { defaultValue: "Installed" });
    case "downloading":
      return t("models.status.downloading", { defaultValue: "Downloading" });
    case "paused":
      return t("models.status.paused", { defaultValue: "Paused" });
    case "failed":
      return t("models.status.failed", { defaultValue: "Failed" });
    case "broken":
      return t("models.status.broken", { defaultValue: "Broken" });
    case "partial":
      return t("models.status.partial", { defaultValue: "Incomplete" });
    default:
      return t("models.status.available", { defaultValue: "Not installed" });
  }
}

export function modelBucketLabel(bucket: ModelStatusBucket | "all", t: TFunction<"library">): string {
  switch (bucket) {
    case "installed":
      return t("models.availability.installed", { defaultValue: "Installed" });
    case "available":
      return t("models.availability.available", { defaultValue: "Available" });
    case "downloading":
      return t("models.availability.downloading", { defaultValue: "Downloading" });
    case "attention":
      return t("models.availability.attention", { defaultValue: "Needs attention" });
    default:
      return t("models.availability.all", { defaultValue: "All" });
  }
}

/** Progress / error detail shown in the chip's tooltip. */
export function modelStatusDetail(status: ResolvedModelStatus, t: TFunction<"library">, reason?: string | null): string | null {
  const download = status.download;
  if (status.state === "downloading" || status.state === "paused") {
    if (!download) return null;
    return [downloadProgressLabel(download), downloadSizeTooltip(download)].filter(Boolean).join(" / ") || null;
  }
  if (status.state === "failed") {
    return download?.error ?? t("models.status.failedDetail", { defaultValue: "Download failed" });
  }
  if (status.state === "broken") return reason ?? null;
  return null;
}

export function ModelStatusChip({ status, reason }: { status: ResolvedModelStatus; reason?: string | null }) {
  const { t } = useTranslation("library");
  return (
    <StatusIcon
      status={status.chip}
      label={modelStatusLabel(status, t)}
      detail={modelStatusDetail(status, t, reason)}
    />
  );
}
