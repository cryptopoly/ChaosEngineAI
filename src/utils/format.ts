import type { ImageModelVariant, ImageRuntimeStatus, VideoModelVariant } from "../types";

export function number(value: number, digits = 1) {
  return value.toFixed(digits);
}

export function sizeLabel(sizeGb: number) {
  return sizeGb > 0 ? `${number(sizeGb)} GB` : "Unknown";
}

export function signedDelta(value: number, digits = 1, suffix = "") {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(digits)}${suffix}`;
}

export function formatImageTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

const MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/**
 * Return a short "Released MMM YYYY" label from a curated ``YYYY-MM`` /
 * ``YYYY-MM-DD`` release date *or* a Hugging Face ISO ``createdAt`` value.
 *
 * Prefers an already-computed label from the backend (``releaseLabel``) so the
 * Python ``_format_release_label`` helper stays the source of truth. Falls
 * back to parsing the raw date in the browser when the backend label is
 * missing, which keeps older cached responses working.
 */
export function formatReleaseLabel(
  primary?: string | null,
  secondary?: string | null,
): string | null {
  if (primary && primary.trim().length > 0) return primary;
  const raw = (secondary ?? "").trim();
  if (!raw) return null;
  const shortMatch = /^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$/.exec(raw);
  if (shortMatch) {
    const year = Number(shortMatch[1]);
    const monthIndex = shortMatch[2] ? Number(shortMatch[2]) - 1 : 0;
    if (Number.isFinite(year) && monthIndex >= 0 && monthIndex <= 11) {
      return `Released ${MONTH_SHORT[monthIndex]} ${year}`;
    }
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return null;
  return `Released ${MONTH_SHORT[parsed.getUTCMonth()]} ${parsed.getUTCFullYear()}`;
}

export function formatImageLicenseLabel(value?: string | null) {
  if (!value) return null;
  return value
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function imagePrimarySizeLabel(variant: ImageModelVariant) {
  if (typeof variant.onDiskGb === "number" && variant.onDiskGb > 0) {
    return `${sizeLabel(variant.onDiskGb)} on disk`;
  }
  if (typeof variant.coreWeightsGb === "number" && variant.coreWeightsGb > 0) {
    return `${sizeLabel(variant.coreWeightsGb)} weights`;
  }
  if (typeof variant.repoSizeGb === "number" && variant.repoSizeGb > 0) {
    return `${sizeLabel(variant.repoSizeGb)} download`;
  }
  return sizeLabel(variant.sizeGb);
}

export function imageSecondarySizeLabel(variant: ImageModelVariant) {
  if (
    typeof variant.repoSizeGb === "number" &&
    variant.repoSizeGb > 0 &&
    typeof variant.coreWeightsGb === "number" &&
    variant.coreWeightsGb > 0 &&
    Math.abs(variant.repoSizeGb - variant.coreWeightsGb) > 0.2
  ) {
    return `${sizeLabel(variant.repoSizeGb)} full repo`;
  }
  return null;
}

export function videoPrimarySizeLabel(variant: VideoModelVariant) {
  if (typeof variant.onDiskGb === "number" && variant.onDiskGb > 0) {
    return `${sizeLabel(variant.onDiskGb)} on disk`;
  }
  return sizeLabel(variant.sizeGb);
}

export function imageRuntimeErrorStatus(error: unknown): ImageRuntimeStatus {
  return {
    activeEngine: "unavailable",
    realGenerationAvailable: false,
    message: error instanceof Error ? error.message : "Image runtime unavailable.",
    missingDependencies: [],
  };
}

export function isGatedImageAccessError(message: string | null | undefined) {
  if (!message) return false;
  const lowered = message.toLowerCase();
  return (
    lowered.includes("cannot access gated repo")
    || lowered.includes("gated repo")
    || lowered.includes("authorized list")
    || (lowered.includes("access to model") && lowered.includes("restricted"))
  );
}

export function formatImageAccessError(
  message: string | null | undefined,
  variant?: Pick<ImageModelVariant, "name" | "link"> | null,
) {
  if (!message) return "";
  if (!isGatedImageAccessError(message)) {
    return message;
  }
  return `${variant?.name ?? "This model"} is gated on Hugging Face. Your account or token is not approved for it yet. Open Hugging Face, request or accept access, add a read-enabled HF token in Settings, then retry.`;
}
