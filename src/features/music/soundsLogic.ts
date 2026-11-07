/**
 * Sounds mode of the Music Studio: the form and its limits, validation, the request, the plain readiness
 * answer ("what stands between you and a sound"), and the run state. Sounds are short clips from Stable Audio
 * Open (Small: 1 to 11 s, 1.0: 1 to 47 s; each catalog row carries its own limits); the models are gated, so
 * readiness also asks whether the licence was accepted.
 */
import type { MusicLibraryItem, MusicModelVariant, MusicResult, MusicSoundPayload } from "../../types";
import { classifyMusicError, SEED_MAX, type MusicErrorInfo } from "./musicLogic";

/** Limits used when no model row is known yet (Small's, the default sound model). */
export const SOUND_SECONDS_MIN = 1;
export const SOUND_SECONDS_MAX = 11;
export const SOUND_CAPTION_MAX = 300;

type SoundRange = Partial<Pick<MusicModelVariant, "minSeconds" | "maxSeconds">> | null | undefined;

export interface SoundLimits {
  min: number;
  max: number;
}

/** The selected model's own length range (`minSeconds` / `maxSeconds` of its catalog row). */
export function soundLimits(variant?: SoundRange): SoundLimits {
  const min = Number(variant?.minSeconds);
  const max = Number(variant?.maxSeconds);
  return {
    min: Number.isFinite(min) && min > 0 ? min : SOUND_SECONDS_MIN,
    max: Number.isFinite(max) && max > 0 ? max : SOUND_SECONDS_MAX,
  };
}

/** Length pills for a model: Small's familiar set, and a longer ladder ending at the maximum for longer models. */
export function soundLengthPresets(variant?: SoundRange): number[] {
  const { min, max } = soundLimits(variant);
  const ladder = max <= 12 ? [2, 4, 6, 8] : [5, 10, 20, 30];
  return [...ladder.filter((n) => n >= min && n < max), max];
}

export function clampSoundSeconds(seconds: number, variant?: SoundRange): number {
  const { min, max } = soundLimits(variant);
  if (!Number.isFinite(seconds)) return Math.min(5, max);
  return Math.min(max, Math.max(min, seconds));
}
export const SOUND_VARIATION_CHOICES = [1, 2, 4] as const;
/** Pack size shown on the Models page (the install is about 0.9 GB of Python packages). */
export const SOUNDS_PACK_GB = 1;

export interface SoundForm {
  caption: string;
  seconds: number;
  loop: boolean;
  variations: 1 | 2 | 4;
  /** `null` = a new random seed every run. */
  seed: number | null;
}

export type SoundIssue = "captionEmpty" | "captionTooLong" | "secondsRange" | "seedRange" | "variations";

export function defaultSoundForm(variant?: (Pick<MusicModelVariant, "defaultSettings"> & SoundRange) | null): SoundForm {
  const seconds = Number(variant?.defaultSettings?.seconds);
  const { min, max } = soundLimits(variant);
  return {
    caption: "",
    seconds: Number.isFinite(seconds) && seconds >= min && seconds <= max ? seconds : Math.min(5, max),
    loop: Boolean(variant?.defaultSettings?.loop),
    variations: 1,
    seed: null,
  };
}

export function validateSoundForm(form: SoundForm, variant?: SoundRange): SoundIssue[] {
  const { min, max } = soundLimits(variant);
  const issues: SoundIssue[] = [];
  if (!form.caption.trim()) issues.push("captionEmpty");
  else if (form.caption.length > SOUND_CAPTION_MAX) issues.push("captionTooLong");
  if (!Number.isFinite(form.seconds) || form.seconds < min || form.seconds > max) issues.push("secondsRange");
  if (form.seed !== null && (!Number.isInteger(form.seed) || form.seed < 0 || form.seed > SEED_MAX)) issues.push("seedRange");
  if (!(SOUND_VARIATION_CHOICES as readonly number[]).includes(form.variations)) issues.push("variations");
  return issues;
}

export function buildSoundPayload(form: SoundForm, modelId: string | null): MusicSoundPayload {
  const payload: MusicSoundPayload = {
    modelId,
    caption: form.caption.trim(),
    seconds: form.seconds,
    loop: form.loop,
    variations: form.variations,
  };
  if (form.seed !== null) payload.seed = form.seed;
  return payload;
}

/** "Make another like this" for a sound in the Library. */
export function soundFormFromItem(item: MusicLibraryItem, variant?: SoundRange): SoundForm {
  const seconds = clampSoundSeconds(Math.round(item.durationS ?? 5), variant);
  return { caption: item.caption ?? "", seconds, loop: Boolean(item.loop), variations: 1, seed: item.seed ?? null };
}

export function againSoundForm(form: SoundForm): SoundForm {
  return { ...form, seed: null };
}

export function varySoundForm(form: SoundForm, seed: number): SoundForm {
  return { ...form, seed: Math.min(SEED_MAX, seed + 1) };
}

// ---------------------------------------------------------------------------
// Readiness
// ---------------------------------------------------------------------------

export type SoundBlocker = "offline" | "loading" | "unsupported" | "licence" | "pack" | "download" | "textEncoder" | null;

export interface SoundReadinessInput {
  catalogLoaded: boolean;
  variant: MusicModelVariant | null;
  packInstalled: boolean;
  unavailableReason: string | null;
  licenceAccepted: boolean;
}

/** The first thing missing before a sound can be made, in the order the user fixes it; `null` = ready. */
export function soundBlocker({ catalogLoaded, variant, packInstalled, unavailableReason, licenceAccepted }: SoundReadinessInput): SoundBlocker {
  if (!catalogLoaded || !variant) return "loading";
  if (unavailableReason) return "unsupported";
  if (variant.licence?.gated && !licenceAccepted && !variant.availableLocally) return "licence";
  if (!variant.availableLocally) return variant.textEncoderReady === false && variant.weightsSource ? "textEncoder" : "download";
  if (!packInstalled) return "pack";
  return null;
}

// ---------------------------------------------------------------------------
// Run state
// ---------------------------------------------------------------------------

export interface SoundRunState {
  running: boolean;
  startedAt: number | null;
  cancelling: boolean;
  error: MusicErrorInfo | null;
  results: MusicResult[];
  lastForm: SoundForm | null;
}

export const INITIAL_SOUND_RUN: SoundRunState = { running: false, startedAt: null, cancelling: false, error: null, results: [], lastForm: null };

export type SoundRunAction =
  | { type: "start"; at: number; form: SoundForm }
  | { type: "cancelRequested" }
  | { type: "succeeded"; results: MusicResult[] }
  | { type: "failed"; message: string; status: number | null }
  | { type: "dismiss" }
  | { type: "forget"; ids: string[] };

export const MAX_SOUND_RESULTS = 12;

export function soundRunReducer(state: SoundRunState, action: SoundRunAction): SoundRunState {
  switch (action.type) {
    case "start":
      return state.running ? state : { ...state, running: true, startedAt: action.at, cancelling: false, error: null, lastForm: action.form };
    case "cancelRequested":
      return state.running ? { ...state, cancelling: true } : state;
    case "succeeded":
      if (!state.running) return state;
      return { ...state, running: false, startedAt: null, cancelling: false, results: [...action.results, ...state.results].slice(0, MAX_SOUND_RESULTS) };
    case "failed":
      if (!state.running) return state;
      return { ...state, running: false, startedAt: null, cancelling: false, error: classifyMusicError(action.message, action.status) };
    case "dismiss":
      return state.running ? state : { ...state, error: null };
    case "forget":
      return { ...state, results: state.results.filter((result) => !action.ids.includes(result.item.id)) };
    default:
      return state;
  }
}
