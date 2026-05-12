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

export function formatImageTimestamp(value: string, locale?: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  // FU-042: when a locale is passed (from ``i18n.language``), use it so
  // CJK / Cyrillic / Latin date formats render in the user's writing
  // system.  Caller omits the arg → browser default (back-compat with
  // the pre-i18n call sites that haven't been migrated yet).
  return new Intl.DateTimeFormat(locale ?? [], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

/**
 * FU-042 — locale-aware byte formatter.  Uses the ``Intl.NumberFormat``
 * "unit" style with ``gigabyte`` / ``megabyte`` / ``kilobyte`` so the
 * unit symbol + decimal separator + thousands grouping all flip with
 * the active locale (e.g. ``1,8 GB`` in de vs ``1.8 GB`` in en).
 */
export function formatBytes(bytes: number, locale?: string): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const formatter = (unit: "gigabyte" | "megabyte" | "kilobyte" | "byte", value: number) =>
    new Intl.NumberFormat(locale ?? [], {
      style: "unit",
      unit,
      maximumFractionDigits: value >= 100 ? 0 : 1,
    }).format(value);
  if (bytes >= 1e9) return formatter("gigabyte", bytes / 1e9);
  if (bytes >= 1e6) return formatter("megabyte", bytes / 1e6);
  if (bytes >= 1e3) return formatter("kilobyte", bytes / 1e3);
  return formatter("byte", bytes);
}

/**
 * FU-042 — locale-aware number formatter for everything that isn't a
 * byte / unit count (e.g. tokens/sec values, percentages, plain ints).
 * Honours the locale's decimal + thousands separators.
 */
export function formatNumber(
  value: number,
  locale?: string,
  options?: Intl.NumberFormatOptions,
): string {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(locale ?? [], options).format(value);
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

export function compactReleaseLabel(value?: string | null) {
  if (!value) return null;
  return value.replace(/^Released\s+/i, "").trim() || null;
}

export function formatImageLicenseLabel(value?: string | null) {
  if (!value) return null;
  return value
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function compactModelSizeLabel(value: string | null | undefined) {
  if (!value) return "Unknown";
  return value.replace(/\s+(on disk|weights|download|full repo)$/i, "").trim();
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
  // Preference order (best to worst):
  //   1. onDiskGb — actual bytes on disk once downloaded. Ground truth.
  //   2. coreWeightsGb — sum of weight-file siblings from HF. Matches what
  //      the diffusers allow-pattern download actually pulls.
  //   3. repoSizeGb — total repo size. Overestimates on video repos with
  //      legacy non-diffusers checkpoints (Wan 2.2 ships ~109 GB but the
  //      diffusers subtree is ~14 GB).
  //   4. curated sizeGb — hardcoded catalog guess, often stale.
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

export function videoSecondarySizeLabel(variant: VideoModelVariant) {
  // When the full repo is meaningfully larger than just the weight files —
  // which is the case for Wan 2.2 (109 GB repo vs ~14 GB weights) — show
  // the full-repo figure so users aren't surprised by a 5-10× disk balloon
  // when HF pulls the entire tree.
  if (
    typeof variant.repoSizeGb === "number" &&
    variant.repoSizeGb > 0 &&
    typeof variant.coreWeightsGb === "number" &&
    variant.coreWeightsGb > 0 &&
    Math.abs(variant.repoSizeGb - variant.coreWeightsGb) > 0.5
  ) {
    return `${sizeLabel(variant.repoSizeGb)} full repo`;
  }
  return null;
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
