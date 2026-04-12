import type { LibraryItem, ModelFamily, ModelVariant } from "../types";
import { flattenVariants } from "./models";

export function tokenSet(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length >= 3 && !["gguf", "mlx", "bf16", "fp8", "instruct", "community"].includes(token));
}

export function normalizeQuantizationLabel(value: string | null | undefined): string {
  return (value ?? "").toLowerCase().replace(/[\s-]+/g, "");
}

export function inferQuantizationLabel(text: string): string | null {
  const lowered = text.toLowerCase();
  const qMatch = lowered.match(/\b(q\d(?:_[a-z0-9]+)*)\b/);
  if (qMatch) return qMatch[1].toUpperCase();
  const bitMatch = lowered.match(/\b(\d+)[-_ ]?bit\b/);
  if (bitMatch) return `${bitMatch[1]}-bit`;
  if (/(^|[^a-z])bf16([^a-z]|$)|bfloat16/.test(lowered)) return "BF16";
  if (/fp16|float16|(^|[^a-z])f16([^a-z]|$)/.test(lowered)) return "FP16";
  if (/fp8|float8/.test(lowered)) return "FP8";
  if (/fp32|float32/.test(lowered)) return "FP32";
  return null;
}

export function libraryItemSourceKind(item: LibraryItem): string {
  if (item.sourceKind) return item.sourceKind;
  if (item.path.includes("/models--")) return "HF cache";
  return /\.(gguf|safetensors)$/i.test(item.path) ? "File" : "Directory";
}

export function libraryItemFormat(item: LibraryItem, matchedVariant?: ModelVariant | null): string {
  const explicit = (item.format ?? "").trim();
  if (explicit && explicit.toLowerCase() !== "hf cache" && explicit.toLowerCase() !== "unknown") {
    return explicit;
  }
  const haystack = `${item.name} ${item.path}`.toLowerCase();
  if (item.backend === "llama.cpp" || haystack.includes("gguf")) return "GGUF";
  if (explicit.toLowerCase() === "mlx" || /(^|[^a-z])mlx([^a-z]|$)|mlx-community/.test(haystack)) return "MLX";
  if (matchedVariant?.format) return matchedVariant.format;
  return explicit || "Unknown";
}

export function libraryItemQuantization(item: LibraryItem, matchedVariant?: ModelVariant | null): string | null {
  return item.quantization ?? matchedVariant?.quantization ?? inferQuantizationLabel(`${item.name} ${item.path}`);
}

export function libraryItemBackend(item: LibraryItem, matchedVariant?: ModelVariant | null): string {
  if (item.backend) return item.backend;
  const format = libraryItemFormat(item, matchedVariant).toLowerCase();
  if (format === "gguf") return "llama.cpp";
  if (matchedVariant?.backend) return matchedVariant.backend;
  return "mlx";
}

export function libraryVariantMatchScore(item: LibraryItem, variant: ModelVariant): number {
  const haystack = `${item.name} ${item.path}`.toLowerCase();
  let score = 0;
  const exactCandidates = [variant.id, variant.repo, variant.name, variant.link]
    .map((value) => value.toLowerCase())
    .filter(Boolean);
  if (exactCandidates.some((candidate) => haystack.includes(candidate))) {
    score += 80;
  }

  const compactCandidates = exactCandidates
    .flatMap((candidate) => [candidate.split("/").pop() ?? "", candidate.replace(/\//g, "-")])
    .filter(Boolean);
  if (compactCandidates.some((candidate) => haystack.includes(candidate))) {
    score += 40;
  }

  const hits = tokenSet(`${variant.repo} ${variant.name}`).filter((token) => haystack.includes(token));
  score += hits.length * 6;

  const itemFormat = libraryItemFormat(item).toLowerCase();
  if (itemFormat && variant.format) {
    if (variant.format.toLowerCase() === itemFormat) score += 14;
    else if ((item.format ?? "").toLowerCase() !== "hf cache") score -= 6;
  }

  const itemQuant = normalizeQuantizationLabel(libraryItemQuantization(item));
  const variantQuant = normalizeQuantizationLabel(variant.quantization);
  if (itemQuant && variantQuant) {
    score += itemQuant === variantQuant ? 18 : -8;
  }

  if (haystack.includes("gguf")) {
    score += variant.format === "GGUF" ? 8 : -4;
  }
  if (haystack.includes("mlx")) {
    score += variant.format === "MLX" ? 8 : -4;
  }

  return score;
}

export function libraryItemMatchesVariant(item: LibraryItem, variant: ModelVariant): boolean {
  return libraryVariantMatchScore(item, variant) >= 12;
}

export function findLibraryItemForVariant(library: LibraryItem[], variant: ModelVariant): LibraryItem | null {
  let best: { item: LibraryItem; score: number } | null = null;
  for (const item of library) {
    const score = libraryVariantMatchScore(item, variant);
    if (!best || score > best.score) {
      best = { item, score };
    }
  }
  return best && best.score >= 12 ? best.item : null;
}

export function findCatalogVariantForLibraryItem(families: ModelFamily[], item: LibraryItem): ModelVariant | null {
  let best: { variant: ModelVariant; score: number } | null = null;
  for (const variant of flattenVariants(families)) {
    const score = libraryVariantMatchScore(item, variant);
    if (!best || score > best.score) {
      best = { variant, score };
    }
  }
  return best && best.score >= 12 ? best.variant : null;
}
