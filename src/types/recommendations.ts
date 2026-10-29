/** Settings the Image / Video Studio can apply from the "Recommended for this machine"
 * panel. Every key is optional: the backend sends only what the variant's engine reads
 * (``backend_service/helpers/recommendations.py``). */
export interface RecommendedSettingValues {
  steps?: number;
  guidance?: number;
  width?: number;
  height?: number;
  /** Video only. */
  numFrames?: number;
  /** Video only. */
  fps?: number;
  /** Image only: one of the Studio's quality presets. */
  qualityPreset?: string;
  /** Image only (UNet pipelines). */
  sampler?: string;
  cacheStrategy?: string;
  cacheThreshold?: number;
  cfgDecay?: boolean;
  previewVae?: boolean;
  fp8LayerwiseCasting?: boolean;
}

export type RecommendedSettingKey = keyof RecommendedSettingValues;

export type RecommendationBasis = "measured" | "estimated";

export interface RecommendedSettings {
  /** Hardware tier the table was read for, e.g. ``apple-64`` or ``cuda-24``. */
  hardwareBucket: string;
  /** Human label for the tier, e.g. "Apple Silicon, 64 GB". */
  hardwareLabel?: string;
  /** ``measured`` only when every setting was timed on a machine like this one. */
  basis: RecommendationBasis;
  basisByKey?: Partial<Record<RecommendedSettingKey, RecommendationBasis>>;
  /** Machine the measured settings were timed on, e.g. "Apple M4 Max, 64 GB". */
  measuredOn?: string | null;
  settings: RecommendedSettingValues;
  /** One line per setting saying why. */
  reasons: Partial<Record<RecommendedSettingKey, string>>;
}
