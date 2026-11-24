import type { TensorfoldExtra, TensorfoldJobState } from "../api";
import type { SystemStats } from "../types";

/**
 * TensorFold engine helpers (exact speculative decoding on Apple Silicon).
 *
 * The backend owns the list of checkpoints TensorFold serves
 * (``backend_service/inference/_tensorfold.py``) and publishes it in the
 * system snapshot; everything here only reads that list. Matching is by
 * *exact* repo id, deliberately unlike the fuzzy ``candidateKeys`` matching
 * DFlash and MTPLX use: TensorFold is tested with specific conversions, and
 * a different conversion of the same model is not evidence that it reads it.
 */

export type TensorfoldInfo = NonNullable<SystemStats["tensorfold"]>;

/**
 * - ``exclusive`` — no other engine the app ships can load it, so it always
 *   runs on TensorFold (and needs it installed).
 * - ``tested`` — stock MLX also loads it; ticking speculative decoding moves
 *   it to TensorFold when installed.
 */
export type TensorfoldTier = "exclusive" | "tested";

export interface TensorfoldModelSupport {
  tier: TensorfoldTier;
  /** The registry's spelling of the repo that matched. */
  repo: string;
  /** Unified memory the checkpoint's docs call for, when they state one. */
  minMemoryGb: number | null;
  /** TensorFold can read images for this checkpoint once the vision extra is installed. */
  vision: boolean;
}

/**
 * The app-level TensorFold state the launch surfaces receive as one prop
 * (system snapshot slice + install job), instead of threading four.
 */
export interface TensorfoldLaunchControls {
  info?: SystemStats["tensorfold"];
  /** Install TensorFold, or add optional ``extras`` to the existing install. */
  onInstall?: (extras?: TensorfoldExtra[]) => void;
  installing?: boolean;
  job?: TensorfoldJobState | null;
}

/** What ``RuntimeControls`` needs about one selected checkpoint. */
export function tensorfoldLaunchInfoFor(
  controls: TensorfoldLaunchControls | null | undefined,
  candidates: Array<string | null | undefined>,
): TensorfoldLaunchInfo | null {
  const support = resolveTensorfoldSupport(controls?.info, candidates);
  if (!controls?.info || !support) return null;
  return {
    available: controls.info.available,
    version: controls.info.version,
    extras: controls.info.extras ?? [],
    support,
    native: controls.info.engine === "native",
  };
}

/** Launch-settings wiring the ``RuntimeControls`` TensorFold block reads. */
export interface TensorfoldLaunchInfo {
  available: boolean;
  version?: string | null;
  /** Optional extras already installed ("vision", "grammar"). */
  extras: string[];
  support: TensorfoldModelSupport;
  /** TensorFold 1.x native binary on a CUDA host (not the Mac Python engine). */
  native?: boolean;
}

function lowered(values: Array<string | null | undefined>): string[] {
  return values
    .map((value) => (value ?? "").trim().toLowerCase())
    .filter((value) => value.length > 0);
}

/** The TensorFold support level of a model, or ``null`` when it is not one TensorFold serves. */
export function resolveTensorfoldSupport(
  info: TensorfoldInfo | null | undefined,
  candidates: Array<string | null | undefined>,
): TensorfoldModelSupport | null {
  if (!info) return null;
  const wanted = lowered(candidates);
  if (wanted.length === 0) return null;
  const repo = info.supportedModels.find((supported) => wanted.includes(supported.toLowerCase()));
  if (!repo) return null;
  const exclusive = info.exclusiveModels.some((entry) => entry.toLowerCase() === repo.toLowerCase());
  return {
    tier: exclusive ? "exclusive" : "tested",
    repo,
    minMemoryGb: info.minMemoryGb?.[repo] ?? null,
    vision: (info.visionModels ?? []).some((entry) => entry.toLowerCase() === repo.toLowerCase()),
  };
}

/** Whether an optional extra is already in the TensorFold install. */
export function tensorfoldHasExtra(
  launch: Pick<TensorfoldLaunchInfo, "extras"> | null | undefined,
  extra: TensorfoldExtra,
): boolean {
  return launch?.extras.includes(extra) ?? false;
}

/**
 * Whether the launch settings should offer to add ``extra``: TensorFold itself
 * is installed (the base install has its own button), the extra is not, and —
 * for image input — the selected checkpoint has a vision tower to read.
 */
export function canAddTensorfoldExtra(
  launch: Pick<TensorfoldLaunchInfo, "available" | "extras" | "support"> | null | undefined,
  extra: TensorfoldExtra,
): boolean {
  if (!launch?.available || tensorfoldHasExtra(launch, extra)) return false;
  return extra === "grammar" || launch.support.vision;
}

/** True when a model of this support level needs more memory than the Mac has. */
export function tensorfoldMemoryShortfall(
  support: TensorfoldModelSupport | null | undefined,
  totalMemoryGb: number | null | undefined,
): boolean {
  if (!support?.minMemoryGb || !totalMemoryGb || totalMemoryGb <= 0) return false;
  return totalMemoryGb < support.minMemoryGb;
}

/**
 * Whether TensorFold is what actually serves the model under the current
 * launch settings — it is installed and either the model needs it or the
 * user asked for speculative decoding. Mirrors the backend rule in
 * ``RuntimeController._select_tensorfold``.
 */
export function tensorfoldEngaged(
  launch: Pick<TensorfoldLaunchInfo, "available" | "support"> | null | undefined,
  speculativeDecoding: boolean,
): boolean {
  if (!launch?.available) return false;
  return launch.support.tier === "exclusive" || speculativeDecoding;
}

/** Repo ids of every checkpoint TensorFold serves, for the Discover / My Models filters. */
export function tensorfoldModelList(info: TensorfoldInfo | null | undefined): string[] {
  return info?.supportedModels ?? [];
}

/** Exact, case-insensitive membership test against ``tensorfoldModelList``. */
export function isTensorfoldRepo(
  models: readonly string[] | null | undefined,
  candidates: Array<string | null | undefined>,
): boolean {
  if (!models || models.length === 0) return false;
  const wanted = lowered(candidates);
  return models.some((model) => wanted.includes(model.toLowerCase()));
}

/**
 * TensorFold keeps its own caches (the target's, the draft model's and one prompt-scratch block), so its memory
 * need at a long window is about this many times the standard KV estimate. Measured on a 64 GB M4 Max with
 * Qwen3.8 27B 4-bit and its DFlash 2 draft: 34.1 GiB needed at a 262,144-token window against a 16.0 GB standard
 * estimate (2.1x); 196,608 tokens started, 262,144 did not.
 */
export const TENSORFOLD_CACHE_FACTOR = 2.2;
/** TensorFold keeps this much free beside the weights (its own start-up check says "an 8 GiB margin"). */
export const TENSORFOLD_MARGIN_GB = 8;
/** Share of unified memory used as the limit when the backend cannot read Metal's own (51.8 of 64 GB measured). */
const TENSORFOLD_USABLE_SHARE = 0.8;
const TENSORFOLD_CONTEXT_STEP = 4096;
const TENSORFOLD_MIN_CONTEXT = 8192;

export interface TensorfoldContextFit {
  /** GB TensorFold's caches would need at the asked window. */
  needGb: number;
  /** GB left for them beside the weights and the margin. */
  budgetGb: number;
  /** The ceiling everything is measured against, and the weights taken out of it. */
  limitGb: number;
  weightsGb: number;
  fits: boolean;
  /** The window to use instead: the asked one when it fits, else the largest that does (a multiple of 4,096). */
  maxContext: number;
}

/**
 * Whether TensorFold will start at ``contextTokens``. It refuses at start-up when its caches do not fit, and the
 * app then quietly runs standard MLX; this predicts that so the launch dialog can cap the window first.
 * ``cacheGbPerToken`` is the standard estimate per token (the preview's cache GB over its window). ``limitGb`` is
 * the machine's own GPU limit (Metal's working set, sent by the backend); without it 80 % of RAM stands in. Returns
 * null when an input is unknown, so no limit is invented.
 */
export function tensorfoldContextFit(input: {
  cacheGbPerToken: number;
  contextTokens: number;
  weightsGb: number;
  totalMemoryGb: number;
  limitGb?: number | null;
}): TensorfoldContextFit | null {
  const { cacheGbPerToken, contextTokens, weightsGb, totalMemoryGb } = input;
  if (!(cacheGbPerToken > 0) || !(contextTokens > 0) || !(weightsGb > 0) || !(totalMemoryGb > 0)) return null;
  const limitGb = input.limitGb && input.limitGb > 0 ? input.limitGb : totalMemoryGb * TENSORFOLD_USABLE_SHARE;
  const perToken = cacheGbPerToken * TENSORFOLD_CACHE_FACTOR;
  const budgetGb = limitGb - weightsGb - TENSORFOLD_MARGIN_GB;
  const needGb = perToken * contextTokens;
  if (needGb <= budgetGb) return { needGb, budgetGb, limitGb, weightsGb, fits: true, maxContext: contextTokens };
  // Sized under the budget with a 10 % reserve, because the real need is not exactly linear.
  const raw = Math.floor((budgetGb * 0.9) / perToken / TENSORFOLD_CONTEXT_STEP) * TENSORFOLD_CONTEXT_STEP;
  const maxContext = Math.max(TENSORFOLD_MIN_CONTEXT, Math.min(raw, contextTokens));
  return { needGb, budgetGb, limitGb, weightsGb, fits: false, maxContext };
}
