import type {
  RecommendationBasis,
  RecommendedSettingKey,
  RecommendedSettings,
  RecommendedSettingValues,
} from "../types/recommendations";

/** Studio setters the panel may call, one per setting. A setting without a setter is not
 * shown: the Studio cannot apply it for this model (the control is hidden). */
export type RecommendationAppliers = {
  [K in RecommendedSettingKey]?: (value: NonNullable<RecommendedSettingValues[K]>) => void;
};

/** The Studio's current values, to mark rows that are already in use. */
export type RecommendationCurrent = Partial<RecommendedSettingValues>;

export type RecommendationRowId =
  | "qualityPreset"
  | "steps"
  | "guidance"
  | "sampler"
  | "size"
  | "numFrames"
  | "fps"
  | "cache"
  | "cfgDecay"
  | "previewVae"
  | "fp8LayerwiseCasting";

/** Rows in display and apply order. Width and height share a row, as do the cache
 * strategy and its threshold; the preset comes before steps so explicit steps win. */
export const RECOMMENDATION_ROWS: ReadonlyArray<{
  id: RecommendationRowId;
  keys: readonly RecommendedSettingKey[];
  /** Keys that must be present for the row to show (the others are optional). */
  required: readonly RecommendedSettingKey[];
}> = [
  { id: "qualityPreset", keys: ["qualityPreset"], required: ["qualityPreset"] },
  { id: "steps", keys: ["steps"], required: ["steps"] },
  { id: "guidance", keys: ["guidance"], required: ["guidance"] },
  { id: "sampler", keys: ["sampler"], required: ["sampler"] },
  { id: "size", keys: ["width", "height"], required: ["width", "height"] },
  { id: "numFrames", keys: ["numFrames"], required: ["numFrames"] },
  { id: "fps", keys: ["fps"], required: ["fps"] },
  { id: "cache", keys: ["cacheStrategy", "cacheThreshold"], required: ["cacheStrategy"] },
  { id: "cfgDecay", keys: ["cfgDecay"], required: ["cfgDecay"] },
  { id: "previewVae", keys: ["previewVae"], required: ["previewVae"] },
  { id: "fp8LayerwiseCasting", keys: ["fp8LayerwiseCasting"], required: ["fp8LayerwiseCasting"] },
];

export interface RecommendationRow {
  id: RecommendationRowId;
  /** The keys of this row the recommendation sets. */
  keys: RecommendedSettingKey[];
  values: RecommendedSettingValues;
  reason: string | null;
  /** ``measured`` only when every key of the row was measured. */
  basis: RecommendationBasis;
  /** True when the Studio already uses these values. */
  inUse: boolean;
}

function present(value: unknown): boolean {
  return value !== undefined && value !== null && !(typeof value === "number" && !Number.isFinite(value));
}

function sameValue(current: unknown, recommended: unknown): boolean {
  if (typeof current === "number" && typeof recommended === "number") {
    return Math.abs(current - recommended) < 1e-6;
  }
  return current === recommended;
}

function keyBasis(rec: RecommendedSettings, key: RecommendedSettingKey): RecommendationBasis {
  return rec.basisByKey?.[key] ?? rec.basis;
}

/** The rows the panel shows: recommended, and applicable through a Studio setter. */
export function recommendationRows(
  rec: RecommendedSettings | null | undefined,
  appliers: RecommendationAppliers,
  current: RecommendationCurrent = {},
): RecommendationRow[] {
  if (!rec?.settings) return [];
  const rows: RecommendationRow[] = [];
  for (const row of RECOMMENDATION_ROWS) {
    const keys = row.keys.filter((key) => present(rec.settings[key]));
    if (!row.required.every((key) => keys.includes(key))) continue;
    if (!keys.every((key) => typeof appliers[key] === "function")) continue;
    const values: RecommendedSettingValues = {};
    for (const key of keys) {
      (values as Record<string, unknown>)[key] = rec.settings[key];
    }
    const reason = keys.map((key) => rec.reasons?.[key]).find((text) => Boolean(text)) ?? null;
    const basis = keys.every((key) => keyBasis(rec, key) === "measured") ? "measured" : "estimated";
    const inUse = keys.every((key) => sameValue(current[key], rec.settings[key]));
    rows.push({ id: row.id, keys, values, reason, basis, inUse });
  }
  return rows;
}

/** Apply the given rows (all applicable rows when omitted) through the Studio setters,
 * in ``RECOMMENDATION_ROWS`` order. Returns the keys that were set. */
export function applyRecommendationRows(
  rows: readonly RecommendationRow[],
  appliers: RecommendationAppliers,
): RecommendedSettingKey[] {
  const applied: RecommendedSettingKey[] = [];
  const ordered = [...rows].sort(
    (a, b) =>
      RECOMMENDATION_ROWS.findIndex((row) => row.id === a.id) -
      RECOMMENDATION_ROWS.findIndex((row) => row.id === b.id),
  );
  for (const row of ordered) {
    for (const key of row.keys) {
      const value = row.values[key];
      const setter = appliers[key] as ((next: unknown) => void) | undefined;
      if (!present(value) || !setter) continue;
      setter(value);
      applied.push(key);
    }
  }
  return applied;
}

export const DEFAULT_MAX_SIDE = 4096;
export const DEFAULT_MIN_SIDE = 256;

/** Clamp a width / height to the model's limits; ``fallback`` replaces an empty or invalid value. */
export function clampStudioSide(
  value: number,
  limits: { maxSide?: number | null; minSide?: number | null },
  fallback: number,
): number {
  const max = limits.maxSide && limits.maxSide > 0 ? limits.maxSide : DEFAULT_MAX_SIDE;
  const min = Math.min(max, Math.max(DEFAULT_MIN_SIDE, limits.minSide ?? DEFAULT_MIN_SIDE));
  if (!Number.isFinite(value) || value <= 0) return Math.max(min, Math.min(max, fallback));
  return Math.max(min, Math.min(max, Math.round(value)));
}
