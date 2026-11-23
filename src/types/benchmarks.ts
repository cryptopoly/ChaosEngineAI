import type { RuntimeStatus } from "./server";


/**
 * Phase 3.5: per-turn host telemetry snapshot. Captured at stream
 * finalisation so the values reflect the load the turn generated,
 * not idle baseline. Any field can be null when the underlying
 * sampler is unavailable on this OS.
 */
export interface PerfTelemetry {
  cpuPercent?: number | null;
  gpuPercent?: number | null;
  thermalState?: "nominal" | "moderate" | "critical" | null;
  availableMemoryGb?: number | null;
}

export interface GenerationMetrics {
  finishReason: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  tokS: number;
  responseSeconds?: number | null;
  /** Phase 3.5: host telemetry sampled at turn finalisation. */
  perfTelemetry?: PerfTelemetry | null;
  /**
   * Phase 3.1: DDTree accepted-span overlay data. `acceptedSpans` is
   * a run-length-encoded list over `acceptedTokenText` describing
   * which character ranges came from accepted draft tokens vs
   * verifier-decoded tokens. Only populated when speculative
   * decoding ran (DFLASH path).
   */
  acceptedSpans?: Array<{ start: number; length: number; accepted: boolean }> | null;
  acceptedTokenText?: string | null;
  /** Time-to-first-token in seconds (Phase 2.0). Time from generation start
   * to the moment the model produced its first reasoning or text token.
   * Useful for diagnosing slow prompt-eval phases on long contexts. */
  ttftSeconds?: number | null;
  runtimeNote: string | null;
  dflashAcceptanceRate?: number | null;
  model?: string | null;
  modelRef?: string | null;
  canonicalRepo?: string | null;
  backend?: string | null;
  engineLabel?: string | null;
  cacheLabel?: string | null;
  cacheStrategy?: string | null;
  cacheBits?: number | null;
  fp16Layers?: number | null;
  fusedAttention?: boolean | null;
  fitModelInMemory?: boolean | null;
  requestedCacheLabel?: string | null;
  requestedCacheStrategy?: string | null;
  requestedCacheBits?: number | null;
  requestedFp16Layers?: number | null;
  requestedFitModelInMemory?: boolean | null;
  requestedSpeculativeDecoding?: boolean | null;
  requestedTreeBudget?: number | null;
  speculativeDecoding?: boolean | null;
  dflashDraftModel?: string | null;
  treeBudget?: number | null;
  modelSource?: string | null;
  modelPath?: string | null;
  contextTokens?: number | null;
  generatedAt?: string | null;
}

export type BenchmarkMode = "throughput" | "perplexity" | "task_accuracy";

export interface BenchmarkResult {
  id: string;
  mode?: BenchmarkMode;
  label: string;
  model: string;
  modelRef?: string | null;
  backend: string;
  engineLabel: string;
  source: string;
  measuredAt: string;
  bits: number;
  fp16Layers: number;
  cacheStrategy: string;
  cacheLabel: string;
  cacheGb: number;
  baselineCacheGb: number;
  compression: number;
  tokS: number;
  /** Runs saved before v0.10.0 carry an estimated "quality %" here. It was never measured and is not shown. */
  quality?: number | null;
  responseSeconds: number;
  loadSeconds: number;
  totalSeconds: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  contextTokens: number;
  maxTokens: number;
  notes?: string | null;
  // Perplexity mode
  perplexity?: number | null;
  perplexityStdError?: number | null;
  perplexityDataset?: string | null;
  perplexityNumSamples?: number | null;
  evalTokensPerSecond?: number | null;
  evalSeconds?: number | null;
  // Task accuracy mode
  taskName?: string | null;
  taskAccuracy?: number | null;
  taskCorrect?: number | null;
  taskTotal?: number | null;
  taskNumShots?: number | null;
  /**
   * How the answer was picked: "next_token" (the choice letter the model scores highest) or
   * "generated" (three generated tokens searched for a letter). Runs saved before the field
   * existed used "generated"; accuracy from the two methods is not comparable.
   */
  taskScoringMethod?: "next_token" | "generated" | null;
}

export interface BenchmarkRunPayload {
  mode?: BenchmarkMode;
  modelRef?: string;
  modelName?: string;
  source?: string;
  backend?: string;
  path?: string;
  label?: string;
  prompt?: string;
  cacheBits: number;
  fp16Layers: number;
  fusedAttention: boolean;
  cacheStrategy: string;
  fitModelInMemory: boolean;
  speculativeDecoding: boolean;
  treeBudget: number;
  /** FU-002: TriAttention MLX kv_budget. Defaults to 2048 server-side. */
  kvBudget: number;
  /** Shared with ``LaunchPreferences`` for the launch controls; benchmark loads use automatic (FU-550). */
  moeCpuLayers?: number | null;
  moeCpuLayersFor?: string | null;
  contextTokens: number;
  maxTokens: number;
  temperature: number;
  // Perplexity mode
  perplexityDataset?: string;
  perplexityNumSamples?: number;
  perplexitySeqLength?: number;
  perplexityBatchSize?: number;
  // Task accuracy mode
  taskName?: string;
  taskLimit?: number;
  taskNumShots?: number;
}

export interface BenchmarkRunResponse {
  /** The saved run, or null when the run was cancelled (nothing is saved then). */
  result: BenchmarkResult | null;
  benchmarks: BenchmarkResult[];
  runtime: RuntimeStatus;
  /** True when Cancel stopped the run. */
  cancelled?: boolean;
  /** With ``cancelled``: the model was unloaded (the run loaded it, or stopping the run needed it). */
  modelUnloaded?: boolean;
}

/** How a cancelled benchmark ended, for the dialog. ``confirmed`` is false when the backend never answered the cancel. */
export interface BenchmarkCancelOutcome {
  confirmed: boolean;
  modelUnloaded: boolean;
}
