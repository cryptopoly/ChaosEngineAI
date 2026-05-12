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
}
