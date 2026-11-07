/**
 * Pure rules of the Music screens: the form and its limits, the request payload, the run estimate as the
 * strip shows it, queue and progress text, plain-language errors, the generation lifecycle as a reducer,
 * and the Library's filter and sort. No React and no network, so all of it is unit tested.
 */

import type { EstimateBasis, EstimateRisk, RunEstimateQueue } from "../../components/RunEstimateStrip";
import { memoryRisk } from "../../components/RunEstimateStrip";
import { ALL_PROJECTS, filterByProject } from "../../utils/galleryProject";
import type {
  MusicEstimateResponse,
  MusicGeneratePayload,
  MusicLibraryItem,
  MusicModelFamily,
  MusicModelVariant,
  MusicProgressSnapshot,
  MusicQueueSnapshot,
  MusicResult,
} from "../../types";

// ---------------------------------------------------------------------------
// Limits (the request model in backend_service/routes/music.py)
// ---------------------------------------------------------------------------

/** The music pack download, as the install banner says it. */
export const MUSIC_PACK_GB = 1.6;

export const CAPTION_MAX = 512;
export const LYRICS_MAX = 4096;
export const SECONDS_MIN = 10;
export const SECONDS_MAX = 600;
export const BPM_MIN = 30;
export const BPM_MAX = 300;
export const SEED_MAX = 2 ** 31 - 1;
export const VARIATION_CHOICES = [1, 2, 4] as const;
export type Variations = (typeof VARIATION_CHOICES)[number];
export const LENGTH_PRESETS = [15, 30, 60, 120] as const;
/** The keys the backend accepts: a note and a mode. */
const NOTES = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"];
export const MUSIC_KEYS: readonly string[] = NOTES.flatMap((note) => [`${note} major`, `${note} minor`]);

// ---------------------------------------------------------------------------
// Form
// ---------------------------------------------------------------------------

export type MusicMode = "song" | "instrumental";

export interface MusicForm {
  mode: MusicMode;
  caption: string;
  lyrics: string;
  seconds: number;
  /** The Custom pill is on: the length is a typed number. */
  customLength: boolean;
  /** `null` = automatic. */
  bpm: number | null;
  key: string | null;
  variations: Variations;
  /** `null` = a random seed. */
  seed: number | null;
}

export function defaultForm(variant?: Pick<MusicModelVariant, "defaultSettings" | "minSeconds" | "maxSeconds"> | null): MusicForm {
  const wanted = Number(variant?.defaultSettings?.seconds);
  const seconds = Number.isFinite(wanted) && wanted > 0 ? clampSeconds(wanted, variant) : 60;
  return {
    mode: "song",
    caption: "",
    lyrics: "",
    seconds,
    customLength: !(LENGTH_PRESETS as readonly number[]).includes(seconds),
    bpm: null,
    key: null,
    variations: 1,
    seed: null,
  };
}

export function clampSeconds(
  seconds: number,
  limits?: Partial<Pick<MusicModelVariant, "minSeconds" | "maxSeconds">> | null,
): number {
  const min = Math.max(SECONDS_MIN, limits?.minSeconds ?? SECONDS_MIN);
  const max = Math.min(SECONDS_MAX, limits?.maxSeconds ?? SECONDS_MAX);
  if (!Number.isFinite(seconds)) return min;
  return Math.min(max, Math.max(min, Math.round(seconds)));
}

export type FormIssue =
  | "captionEmpty"
  | "captionTooLong"
  | "lyricsEmpty"
  | "lyricsTooLong"
  | "secondsRange"
  | "bpmRange"
  | "seedRange"
  | "variations";

export function validateForm(form: MusicForm): FormIssue[] {
  const issues: FormIssue[] = [];
  if (!form.caption.trim()) issues.push("captionEmpty");
  else if (form.caption.length > CAPTION_MAX) issues.push("captionTooLong");
  if (form.mode === "song") {
    if (!form.lyrics.trim()) issues.push("lyricsEmpty");
    else if (form.lyrics.length > LYRICS_MAX) issues.push("lyricsTooLong");
  }
  if (!Number.isFinite(form.seconds) || form.seconds < SECONDS_MIN || form.seconds > SECONDS_MAX) issues.push("secondsRange");
  if (form.bpm !== null && (!Number.isInteger(form.bpm) || form.bpm < BPM_MIN || form.bpm > BPM_MAX)) issues.push("bpmRange");
  if (form.seed !== null && (!Number.isInteger(form.seed) || form.seed < 0 || form.seed > SEED_MAX)) issues.push("seedRange");
  if (!(VARIATION_CHOICES as readonly number[]).includes(form.variations)) issues.push("variations");
  return issues;
}

export function buildPayload(form: MusicForm, modelId: string | null): MusicGeneratePayload {
  const instrumental = form.mode === "instrumental";
  const payload: MusicGeneratePayload = {
    modelId,
    caption: form.caption.trim(),
    instrumental,
    seconds: form.seconds,
    variations: form.variations,
  };
  if (!instrumental) payload.lyrics = form.lyrics;
  if (form.bpm !== null) payload.bpm = form.bpm;
  if (form.key) payload.key = form.key;
  if (form.seed !== null) payload.seed = form.seed;
  return payload;
}

/** "Again": the same settings with a new random seed. */
export function againForm(form: MusicForm): MusicForm {
  return { ...form, seed: null };
}

/** "Vary": the seed plus one, everything else kept. `seed` is the seed the finished run used. */
export function varyForm(form: MusicForm, seed: number): MusicForm {
  return { ...form, seed: Math.min(SEED_MAX, seed + 1) };
}

/** "Make another like this": the form that would remake a Library item (its seed kept). */
export function formFromItem(item: MusicLibraryItem, limits?: Partial<Pick<MusicModelVariant, "minSeconds" | "maxSeconds">> | null): MusicForm {
  const seconds = clampSeconds(item.durationS ?? 60, limits);
  return {
    mode: item.instrumental ? "instrumental" : "song",
    caption: item.caption ?? "",
    lyrics: item.lyrics ?? "",
    seconds,
    customLength: !(LENGTH_PRESETS as readonly number[]).includes(seconds),
    bpm: item.bpm ?? null,
    key: item.key ?? null,
    variations: 1,
    seed: item.seed ?? null,
  };
}

// ---------------------------------------------------------------------------
// Length and summaries
// ---------------------------------------------------------------------------

export type TFn = (key: string, options?: Record<string, unknown>) => string;

export function lengthPillLabel(t: TFn, seconds: number): string {
  if (seconds >= 60 && seconds % 60 === 0) {
    return t("musicStudio.length.minutes", { defaultValue: "{n} min", n: seconds / 60 });
  }
  return t("musicStudio.length.seconds", { defaultValue: "{n} s", n: seconds });
}

/** "2:30" style clock for a track length. */
export function clock(seconds: number | null | undefined): string {
  const total = Math.max(0, Math.round(seconds ?? 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/** Megabytes as the result line shows them: "11 MB" (one decimal below 10). */
export function megabytes(bytes: number | null | undefined): string | null {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return null;
  const mb = bytes / 1_000_000;
  return mb >= 10 ? `${Math.round(mb)} MB` : `${Math.round(mb * 10) / 10} MB`;
}

export function diskLabel(bytes: number): string {
  if (bytes >= 1e9) return `${Math.round(bytes / 1e8) / 10} GB`;
  return `${Math.max(0, Math.round(bytes / 1e6))} MB`;
}

// ---------------------------------------------------------------------------
// Estimate strip, queue and progress
// ---------------------------------------------------------------------------

export interface StripModel {
  seconds: number | null;
  peakGb: number | null;
  totalGb: number | null;
  diskMb: number | null;
  basis: EstimateBasis | null;
  risk: EstimateRisk;
  queue: RunEstimateQueue | null;
}

const EMPTY_STRIP: StripModel = { seconds: null, peakGb: null, totalGb: null, diskMb: null, basis: null, risk: "safe", queue: null };

function positive(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

/** What is queued ahead of a new job: the strip's second line. `null` when the queue is empty. */
export function queueForStrip(
  snapshot: MusicQueueSnapshot | null | undefined,
  progressQueue: MusicProgressSnapshot["queue"] | null | undefined,
): RunEstimateQueue | null {
  if (progressQueue?.waiting) return { what: "", secondsLeft: null, position: progressQueue.position };
  if (!snapshot) return null;
  const ahead = (snapshot.running ? 1 : 0) + snapshot.waiting.length;
  return ahead > 0 ? { what: "", secondsLeft: null, position: ahead } : null;
}

/** The estimate response as the `RunEstimateStrip` props: unknown stays null (the strip says so). */
export function stripFromEstimate(
  response: MusicEstimateResponse | null | undefined,
  progressQueue?: MusicProgressSnapshot["queue"] | null,
  /** This Studio has a run in flight: its progress panel shows the queue, so the strip shows none. */
  running = false,
): StripModel {
  if (!response) return { ...EMPTY_STRIP, queue: queueForStrip(null, progressQueue) };
  const { estimate, fit, totalGb } = response;
  const peakGb = positive(estimate.peakGb);
  const total = positive(totalGb);
  const fallback: EstimateRisk = fit?.status === "wont-fit" ? "danger" : fit?.status === "tight" ? "caution" : "safe";
  const seconds = positive(estimate.seconds);
  return {
    seconds,
    peakGb,
    totalGb: total,
    diskMb: positive(estimate.writesMb),
    basis: seconds == null && peakGb == null ? null : estimate.basis === "measured" ? "measured" : "estimated",
    risk: memoryRisk(peakGb, total, fallback),
    // While this Studio runs, the progress panel owns the queue line (and the snapshot lists our own job).
    queue: running ? null : queueForStrip(response.queue, progressQueue),
  };
}

/** The extra time a cold worker needs, for the strip's tooltip text; 0 when the model is loaded. */
export function coldStartSeconds(response: MusicEstimateResponse | null | undefined): number {
  return response && !response.estimate.warm ? response.estimate.coldStartSeconds : 0;
}

/** True when a run is waiting behind another job (the Studio shows "Queued"). */
export function isQueued(progress: MusicProgressSnapshot | null | undefined): boolean {
  return Boolean(progress?.queue?.waiting);
}

/** The five phases of a run, in order; `LiveProgress` matches them against the backend's phase id. */
export const MUSIC_PHASE_IDS = ["load", "encode", "compose", "decode", "save"] as const;
export const MUSIC_PHASE_SECONDS: Record<(typeof MUSIC_PHASE_IDS)[number], number> = {
  load: 25,
  encode: 2,
  compose: 8,
  decode: 4,
  save: 1,
};

// ---------------------------------------------------------------------------
// Errors in plain words
// ---------------------------------------------------------------------------

export type MusicErrorKind = "oom" | "notReady" | "cancelled" | "offline" | "invalid" | "generic";

export interface MusicErrorInfo {
  kind: MusicErrorKind;
  /** The backend's own sentence (plain by design); shown under the headline when it adds something. */
  detail: string;
  /** Raw text for "Technical details". */
  technical: string;
  status: number | null;
}

/** Sorts a failure by what the user can do about it. */
export function classifyMusicError(message: string, status: number | null): MusicErrorInfo {
  const text = (message || "").trim();
  const lower = text.toLowerCase();
  let kind: MusicErrorKind = "generic";
  if (status === 409 && lower === "cancelled") kind = "cancelled";
  else if (status === 0 || /failed to fetch|networkerror|load failed|timed out/.test(lower)) kind = "offline";
  else if (status === 503 && /memory|free|unload|gb/.test(lower)) kind = "oom";
  else if (/out of memory|ran out of memory|insufficient memory/.test(lower)) kind = "oom";
  else if (status === 409 || /not installed|not downloaded|pack/.test(lower)) kind = "notReady";
  else if (status === 400 || status === 422) kind = "invalid";
  const technical = [status != null && status > 0 ? `HTTP ${status}` : null, text].filter(Boolean).join(" · ");
  return { kind, detail: kind === "cancelled" ? "" : text, technical, status };
}

/** The next shorter length for the "Use a shorter length" fix; stays at the minimum. */
export function shorterSeconds(seconds: number): number {
  const lower = [...LENGTH_PRESETS].reverse().find((preset) => preset < seconds);
  return lower ?? SECONDS_MIN;
}

// ---------------------------------------------------------------------------
// Generation lifecycle
// ---------------------------------------------------------------------------

export type GenerationPhase = "idle" | "running" | "done" | "error";

export interface GenerationState {
  phase: GenerationPhase;
  startedAt: number | null;
  /** Cancel was pressed; the run is stopping. */
  cancelling: boolean;
  error: MusicErrorInfo | null;
  /** Finished tracks of this session, newest first. */
  results: MusicResult[];
  /** The settings of the run in flight or the last run, for Again / Vary / the error fix. */
  lastForm: MusicForm | null;
}

export const INITIAL_GENERATION: GenerationState = {
  phase: "idle",
  startedAt: null,
  cancelling: false,
  error: null,
  results: [],
  lastForm: null,
};

export type GenerationAction =
  | { type: "start"; at: number; form: MusicForm }
  | { type: "cancelRequested" }
  | { type: "succeeded"; results: MusicResult[] }
  | { type: "failed"; error: MusicErrorInfo }
  | { type: "dismiss" }
  | { type: "forget"; ids: string[] };

/** Results kept for the session. */
export const MAX_SESSION_RESULTS = 12;

export function generationReducer(state: GenerationState, action: GenerationAction): GenerationState {
  switch (action.type) {
    case "start":
      if (state.phase === "running") return state;
      return { ...state, phase: "running", startedAt: action.at, cancelling: false, error: null, lastForm: action.form };
    case "cancelRequested":
      return state.phase === "running" ? { ...state, cancelling: true } : state;
    case "succeeded":
      if (state.phase !== "running") return state;
      return {
        ...state,
        phase: "done",
        startedAt: null,
        cancelling: false,
        error: null,
        results: [...action.results, ...state.results].slice(0, MAX_SESSION_RESULTS),
      };
    case "failed":
      if (state.phase !== "running") return state;
      // A cancel is not an error: back to idle with the cancelled note kept for one line.
      return { ...state, phase: "error", startedAt: null, cancelling: false, error: action.error };
    case "dismiss":
      return state.phase === "running" ? state : { ...state, phase: "idle", error: null };
    case "forget":
      return { ...state, results: state.results.filter((result) => !action.ids.includes(result.item.id)) };
    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export function flattenVariants(families: MusicModelFamily[]): MusicModelVariant[] {
  return families.flatMap((family) =>
    family.variants.map((variant) => ({ ...variant, familyName: variant.familyName ?? family.name })),
  );
}

export function isInstalled(variant: MusicModelVariant): boolean {
  return Boolean(variant.availableLocally) && (variant.missingRequires ?? []).length === 0;
}

/** The variant to start on: the best-fit installed one, else any installed, else the family default. */
export function pickDefaultVariant(families: MusicModelFamily[]): MusicModelVariant | null {
  const variants = flattenVariants(families);
  if (variants.length === 0) return null;
  const installed = variants.filter(isInstalled);
  const best = new Set(families.map((family) => family.bestFitVariantId).filter(Boolean));
  const rank = (variant: MusicModelVariant) => (variant.fit?.status === "fits" ? 0 : variant.fit?.status === "tight" ? 1 : 2);
  const pool = installed.length > 0 ? installed : variants;
  const sorted = [...pool].sort(
    (a, b) => Number(best.has(b.id)) - Number(best.has(a.id)) || rank(a) - rank(b),
  );
  const defaults = new Set(families.map((family) => family.defaultVariantId));
  return installed.length > 0 ? sorted[0] : variants.find((variant) => defaults.has(variant.id)) ?? sorted[0];
}

/** "about N s of audio per second" from the catalog row; null when not measured. */
export function audioPerSecond(variant: Pick<MusicModelVariant, "estimates">): number | null {
  const per10 = variant.estimates?.secondsPer10sAudio;
  if (variant.estimates?.basis !== "measured" || !per10 || per10 <= 0) return null;
  return Math.round((10 / per10) * 10) / 10;
}

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

export type LibrarySort = "newest" | "oldest" | "longest";

export interface LibraryFilter {
  query: string;
  /** Empty = any tag. */
  tag: string;
  sort: LibrarySort;
  /** ALL_PROJECTS, NO_PROJECT or a project id. */
  project?: string;
}

export const DEFAULT_LIBRARY_FILTER: LibraryFilter = { query: "", tag: "", sort: "newest", project: ALL_PROJECTS };

export function filterLibrary(items: MusicLibraryItem[], filter: LibraryFilter): MusicLibraryItem[] {
  const needle = filter.query.trim().toLowerCase();
  const tag = filter.tag.trim().toLowerCase();
  const kept = filterByProject(items, filter.project ?? ALL_PROJECTS).filter((item) => {
    if (tag && !item.tags.some((entry) => entry.toLowerCase() === tag)) return false;
    if (!needle) return true;
    const haystack = `${item.title}\n${item.caption}\n${item.tags.join(" ")}`.toLowerCase();
    return haystack.includes(needle);
  });
  const byDate = (a: MusicLibraryItem, b: MusicLibraryItem) => a.createdAt - b.createdAt || a.id.localeCompare(b.id);
  return [...kept].sort((a, b) => {
    if (filter.sort === "oldest") return byDate(a, b);
    if (filter.sort === "longest") return (b.durationS ?? 0) - (a.durationS ?? 0) || byDate(b, a);
    return byDate(b, a);
  });
}

/** Every tag in use, case-insensitively unique, alphabetical. */
export function allTags(items: MusicLibraryItem[]): string[] {
  const seen = new Map<string, string>();
  for (const item of items) {
    for (const tag of item.tags) {
      if (!seen.has(tag.toLowerCase())) seen.set(tag.toLowerCase(), tag);
    }
  }
  return [...seen.values()].sort((a, b) => a.localeCompare(b));
}

/** Add a tag the way the backend cleans them: trimmed, one line, no duplicates (any case). */
export function withTag(tags: string[], raw: string): string[] {
  const tag = raw.replace(/\s+/g, " ").trim().slice(0, 40);
  if (!tag || tags.some((entry) => entry.toLowerCase() === tag.toLowerCase())) return tags;
  return [...tags, tag];
}

/** The library's counts line parts. */
export function libraryCounts(items: MusicLibraryItem[]): { tracks: number; sounds: number } {
  const sounds = items.filter((item) => item.kind === "sound").length;
  return { tracks: items.length - sounds, sounds };
}

/** Where a Library item's audio file is on this machine, for "Send to storyboard". */
export function itemFilePath(libraryDir: string | null | undefined, item: Pick<MusicLibraryItem, "id" | "audio">): string | undefined {
  if (!libraryDir) return undefined;
  const separator = libraryDir.includes("\\") && !libraryDir.includes("/") ? "\\" : "/";
  return `${libraryDir.replace(/[\\/]+$/, "")}${separator}${item.id}${item.audio}`;
}

// ---- polling ----
export const DOWNLOAD_POLL_MS = 2000;
export const POLL_MAX_MS = 30_000;
export const MAX_POLL_FAILURES = 6;

/** Wait before the next status poll: 2 s while it answers, then 4, 8, 16 and at most 30 s after each failure. */
export function pollDelay(failures: number, baseMs = DOWNLOAD_POLL_MS): number {
  return Math.min(POLL_MAX_MS, baseMs * 2 ** Math.max(0, failures));
}
