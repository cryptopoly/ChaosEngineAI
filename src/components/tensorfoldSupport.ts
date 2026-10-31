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
  };
}

/** Launch-settings wiring the ``RuntimeControls`` TensorFold block reads. */
export interface TensorfoldLaunchInfo {
  available: boolean;
  version?: string | null;
  /** Optional extras already installed ("vision", "grammar"). */
  extras: string[];
  support: TensorfoldModelSupport;
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
