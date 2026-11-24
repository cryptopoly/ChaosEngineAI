export interface LogEntry {
  ts: string;
  source: string;
  level: string;
  message: string;
}

export interface ActivityItem {
  time: string;
  title: string;
  detail: string;
  /** Optional i18n key under `dashboard` namespace for the title. */
  titleKey?: string;
  /** Optional i18n key under `dashboard` namespace for the detail line. */
  detailKey?: string;
  /** ICU MessageFormat variables for the keyed strings. */
  payload?: Record<string, unknown>;
}

export interface PreviewMetrics {
  bits: number;
  fp16Layers: number;
  numLayers: number;
  numHeads: number;
  numKvHeads: number;
  hiddenSize: number;
  contextTokens: number;
  paramsB: number;
  baselineCacheGb: number;
  optimizedCacheGb: number;
  compressionRatio: number;
  estimatedTokS: number;
  speedRatio: number;
  qualityPercent: number;
  diskSizeGb: number;
  summary: string;
  /** Where the cache size came from: the model's own config / GGUF header,
   * or an estimate from its parameter count. */
  archSource?: "config" | "gguf" | "hub" | "estimate";
  /** "measured": this machine's own past runs of the model; "estimate": from memory bandwidth. */
  speedSource?: "measured" | "estimate";
  bandwidthGbps?: number;
  bandwidthLabel?: string;
  bandwidthKnown?: boolean;
  /** A mixture-of-experts GGUF on a discrete NVIDIA card (FU-550): its layer count and how many layers' experts
   * would stay in system RAM to fit the card's free memory now (``autoCpuLayers``, null when that cannot be read). */
  moeOffload?: MoeOffloadPreview | null;
  /** The ceiling a launch has to fit under here: Mac Metal's working-set limit, an NVIDIA card's VRAM (with the
   * free part), or 80 % of RAM. */
  memory?: MemoryBudget | null;
  /** NVIDIA only: what the TensorFold 1.x binary needs at this window and the largest window that fits now. */
  tensorfoldFit?: TensorfoldNativeFit | null;
}

export interface MemoryBudget {
  kind: "metal" | "vram" | "ram";
  limitGb: number;
  totalGb: number;
  freeGb: number | null;
}

export interface TensorfoldNativeFit {
  needGb: number;
  freeGb: number;
  fits: boolean;
  maxContext: number;
}

export interface MoeOffloadPreview {
  blocks: number;
  moeLayers: number;
  expertGb: number;
  autoCpuLayers: number | null;
  gpuGb: number;
  ramGb: number;
  freeGpuGb: number | null;
}
