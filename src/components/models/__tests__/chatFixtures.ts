import type { DownloadStatus } from "../../../api";
import type { HubModel, LibraryItem, ModelFamily, ModelVariant } from "../../../types";

export function makeChatVariant(overrides: Partial<ModelVariant> = {}): ModelVariant {
  return {
    id: "mlx-community/Qwen3-4B-4bit",
    familyId: "qwen-3",
    name: "Qwen3 4B 4-bit",
    repo: "mlx-community/Qwen3-4B-4bit",
    link: "https://huggingface.co/mlx-community/Qwen3-4B-4bit",
    paramsB: 4,
    sizeGb: 2.3,
    format: "MLX",
    quantization: "4-bit",
    capabilities: ["chat", "reasoning"],
    note: "Small reasoning model.",
    contextWindow: "32K",
    estimatedMemoryGb: 3.1,
    estimatedCompressedMemoryGb: 2.1,
    availableLocally: false,
    launchMode: "direct",
    backend: "mlx",
    releaseDate: "2025-04-29",
    ...overrides,
  };
}

export function makeChatFamily(overrides: Partial<ModelFamily> = {}, variants?: ModelVariant[]): ModelFamily {
  const list = variants ?? [makeChatVariant()];
  return {
    id: "qwen-3",
    name: "Qwen3",
    provider: "Qwen",
    headline: "Hybrid reasoning family.",
    summary: "Dense and MoE models.",
    description: "Qwen3 family.",
    updatedLabel: "Updated Apr 2025",
    popularityLabel: "1M downloads",
    likesLabel: "2k likes",
    badges: [],
    capabilities: ["chat", "reasoning"],
    defaultVariantId: list[0]?.id ?? "",
    variants: list,
    readme: ["First readme line.", "Second readme line.", "Third line not shown."],
    ...overrides,
  };
}

export function makeLibraryItem(overrides: Partial<LibraryItem> = {}): LibraryItem {
  return {
    name: "mlx-community/Qwen3-4B-4bit",
    path: "/Users/dan/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit",
    format: "MLX",
    sourceKind: "HF cache",
    quantization: "4-bit",
    backend: "mlx",
    modelType: "text",
    sizeGb: 2.3,
    lastModified: "2026-04-16 17:00",
    actions: [],
    directoryLabel: "Hugging Face cache",
    broken: false,
    brokenReason: null,
    ...overrides,
  };
}

export function makeHubModel(overrides: Partial<HubModel> = {}): HubModel {
  return {
    id: "zai-org/GLM-4.7-Flash",
    repo: "zai-org/GLM-4.7-Flash",
    name: "GLM-4.7-Flash",
    provider: "zai-org",
    link: "https://huggingface.co/zai-org/GLM-4.7-Flash",
    format: "Transformers",
    tags: ["text-generation"],
    downloads: 729180,
    likes: 1665,
    downloadsLabel: "729,180 downloads",
    likesLabel: "1,665 likes",
    lastModified: "2026-04-16T10:00:00Z",
    updatedLabel: "Updated Apr 16",
    availableLocally: false,
    launchMode: "convert",
    backend: "mlx",
    ...overrides,
  };
}

export function transfer(repo: string, state: DownloadStatus["state"], overrides: Partial<DownloadStatus> = {}): DownloadStatus {
  return { repo, state, progress: 0.42, downloadedGb: 1, totalGb: 2.3, error: null, ...overrides };
}

export const MAC = { platform: "Darwin", arch: "arm64", gpuVramTotalGb: null };
export const WINDOWS_CUDA = { platform: "Windows", arch: "x86_64", gpuVramTotalGb: 24 };
