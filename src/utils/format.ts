import type { ImageModelVariant, ImageRuntimeStatus } from "../types";

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

export function formatImageLicenseLabel(value?: string | null) {
  if (!value) return null;
  return value
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function imagePrimarySizeLabel(variant: ImageModelVariant) {
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
