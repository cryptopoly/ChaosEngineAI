export type ModelLaunchMode = "direct" | "convert";

export interface ModelVariant {
  id: string;
  familyId: string;
  name: string;
  repo: string;
  ggufRepo?: string | null;
  ggufFile?: string | null;
  link: string;
  paramsB: number;
  sizeGb: number;
  format: string;
  quantization: string;
  capabilities: string[];
  note: string;
  contextWindow: string;
  estimatedMemoryGb: number | null;
  estimatedCompressedMemoryGb: number | null;
  availableLocally: boolean;
  launchMode: ModelLaunchMode;
  backend: "mlx" | "llama.cpp" | "auto";
  maxContext?: number | null;
  releaseDate?: string | null;
  releaseLabel?: string | null;
}

export interface ModelFamily {
  id: string;
  name: string;
  provider: string;
  headline: string;
  summary: string;
  description: string;
  updatedLabel: string;
  popularityLabel: string;
  likesLabel: string;
  badges: string[];
  capabilities: string[];
  defaultVariantId: string;
  variants: ModelVariant[];
  readme: string[];
}

export interface LibraryItem {
  name: string;
  path: string;
  format: string;
  sourceKind?: string | null;
  quantization?: string | null;
  backend?: string | null;
  modelType?: string | null;
  sizeGb: number;
  lastModified: string;
  actions: string[];
  directoryId?: string;
  directoryLabel?: string;
  directoryPath?: string;
  maxContext?: number | null;
  broken?: boolean;
  brokenReason?: string | null;
}

export interface ModelDirectorySetting {
  id: string;
  label: string;
  path: string;
  enabled: boolean;
  source: "default" | "user";
  exists?: boolean;
  modelCount?: number;
}

export interface LaunchPreferences {
  contextTokens: number;
  maxTokens: number;
  temperature: number;
  cacheBits: number;
  fp16Layers: number;
  fusedAttention: boolean;
  cacheStrategy: string;
  fitModelInMemory: boolean;
  speculativeDecoding: boolean;
  treeBudget: number;
  /** FU-002: TriAttention MLX kv_budget — number of KV positions kept
   * per layer; older positions get scored + evicted by the
   * apply_triattention_mlx compressor. Only consulted when
   * cacheStrategy === "triattention"; ignored otherwise. Default
   * 2048 matches the upstream default + the spike-validated value
   * on Qwen2.5-0.5B (2.6× speedup, identical output). */
  kvBudget: number;
}
