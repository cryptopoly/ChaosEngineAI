import type { LibraryItem, ModelFamily, ModelVariant } from "../types";
import { flattenVariants } from "./models";

export function tokenSet(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length >= 3 && !["gguf", "mlx", "bf16", "fp8", "instruct", "community"].includes(token));
}

function paramScaleTokens(value: string): Set<string> {
  const tokens = new Set<string>();
  const pattern = /(^|[^a-z0-9])(\d+(?:\.\d+)?)b(?=$|[^a-z0-9])/gi;
  for (const match of value.matchAll(pattern)) {
    const token = match[2]?.toLowerCase();
    if (token) tokens.add(`${token}b`);
  }
  return tokens;
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

export function inferHfRepoFromLocalPath(path: string | null | undefined): string | null {
  if (!path) return null;
  const match = path.match(/models--([^/]+(?:--[^/]+)+)/);
  if (!match) return null;
  const repo = match[1].replace(/--/g, "/").replace(/^\/+|\/+$/g, "");
  return repo || null;
}

const CHAT_EXCLUDED_MODEL_TYPES = new Set(["image", "video", "draft"]);

const VIDEO_LIBRARY_KEYWORDS = [
  "hunyuanvideo",
  "wan-ai/",
  "wan2.",
  "wan2-",
  "-t2v-",
  "-i2v-",
  "-v2v-",
  "mochi-1",
  "cogvideo",
  "ltx-video",
  "ltx-2",
  "zeroscope",
  "animatediff",
];

const IMAGE_LIBRARY_KEYWORDS = [
  "stable-diffusion",
  "sdxl",
  "flux.",
  "flux1",
  "flux-",
  "dall-e",
  "imagen",
  "kandinsky",
  "wuerstchen",
  "diffusion-pipe",
  "qwen-image",
  "qwen/qwen-image",
  "sana_sprint",
  "sana-sprint",
  "sana sprint",
  "sana_1600m",
  "sana-1600m",
];

export function isChatLibraryItem(item: LibraryItem): boolean {
  const modelType = String(item.modelType ?? "").trim().toLowerCase();
  if (modelType && CHAT_EXCLUDED_MODEL_TYPES.has(modelType)) return false;

  const repo = inferHfRepoFromLocalPath(item.path);
  const haystack = `${item.name} ${item.path} ${repo ?? ""}`.toLowerCase();
  if (VIDEO_LIBRARY_KEYWORDS.some((keyword) => haystack.includes(keyword))) return false;
  if (IMAGE_LIBRARY_KEYWORDS.some((keyword) => haystack.includes(keyword))) return false;
  return true;
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

  const itemScaleTokens = paramScaleTokens(haystack);
  const variantScaleTokens = paramScaleTokens(`${variant.repo} ${variant.name}`);
  if (itemScaleTokens.size > 0 && variantScaleTokens.size > 0) {
    const sameScale = [...itemScaleTokens].some((token) => variantScaleTokens.has(token));
    score += sameScale ? 18 : -24;
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

function roundGb(value: number): number {
  return Math.round(value * 10) / 10;
}

/**
 * Estimate peak resident memory when this library item is loaded for inference.
 *
 * Prefers the on-disk size (ground truth — MLX, GGUF, and safetensors formats
 * all store weights at their runtime precision) plus modest KV-cache and
 * framework overheads sized for typical single-user desktop usage.
 *
 * Assumptions (deliberately rough — the Studio and Launch panels model
 * specific workloads more precisely):
 *  - Weights sit in memory at their stored precision. Size on disk ≈ resident weights.
 *  - KV cache ≈ 4% of weights at an 8K context for a modern GQA model. Scale
 *    linearly with chosen context length elsewhere.
 *  - Framework overhead is small and largely size-invariant (~0.6 GB).
 *
 * Falls back to the matched catalog variant's estimate only when the on-disk
 * size is unknown or zero.
 */
export function estimateLibraryItemResidentGb(
  item: LibraryItem,
  matchedVariant?: ModelVariant | null,
): number | null {
  const sizeGb = Number.isFinite(item.sizeGb) && item.sizeGb > 0 ? item.sizeGb : null;
  if (sizeGb == null) {
    const fallback = matchedVariant?.estimatedMemoryGb;
    return fallback != null && Number.isFinite(fallback) ? fallback : null;
  }
  const kvCacheGb = sizeGb * 0.04;
  const frameworkOverheadGb = 0.6;
  return roundGb(sizeGb + kvCacheGb + frameworkOverheadGb);
}

/**
 * Same baseline as estimateLibraryItemResidentGb but assumes a compressed KV
 * cache strategy (ChaosEngine / TurboQuant / RotorQuant), which roughly halves
 * the KV term. Weights and framework overhead are unchanged. At short contexts
 * the delta against the uncompressed estimate is small by design — compression
 * pays off at long contexts, which this column foreshadows.
 */
export function estimateLibraryItemCompressedGb(
  item: LibraryItem,
  matchedVariant?: ModelVariant | null,
): number | null {
  const sizeGb = Number.isFinite(item.sizeGb) && item.sizeGb > 0 ? item.sizeGb : null;
  if (sizeGb == null) {
    const fallback = matchedVariant?.estimatedCompressedMemoryGb;
    return fallback != null && Number.isFinite(fallback) ? fallback : null;
  }
  const kvCacheGb = sizeGb * 0.04 * 0.5;
  const frameworkOverheadGb = 0.6;
  return roundGb(sizeGb + kvCacheGb + frameworkOverheadGb);
}
