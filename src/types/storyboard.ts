/** Storyboard project schema and API payloads (mirrors backend_service/storyboard/models.py). */

/** The aspect ratio (called the export preset in the stored project). */
export type StoryboardExportPreset = "16:9" | "9:16" | "1:1";
export type StoryboardResolution = "480p" | "720p" | "1080p" | "1440p" | "2160p";
export type StoryboardUpscale = "none" | "1440p" | "2160p";

export interface StoryboardDefaults {
  width: number;
  height: number;
  fps: number;
  imageModel: string | null;
  videoModel: string | null;
  voice: string | null;
  exportPreset: StoryboardExportPreset;
  resolution: StoryboardResolution;
}

/** One picture of a character or place (project format 3). */
export interface StoryboardRefImage {
  path: string;
  hash: string | null;
}

/** A character or place: a name, a few words and up to four pictures (front, side, detail...). */
export interface StoryboardReference {
  id: string;
  name: string;
  description: string;
  images: StoryboardRefImage[];
}

export interface StoryboardReferences {
  characters: StoryboardReference[];
  places: StoryboardReference[];
}

export interface StoryboardTransition {
  type: "cut" | "crossfade";
  seconds: number;
}

export interface StoryboardTake {
  id: string;
  still: string | null;
  clip: string | null;
  hashes: Partial<Record<"still" | "clip", string>>;
  settings: Record<string, string | number | boolean | null>;
  measuredSeconds: number | null;
  createdAt: number;
  origin: "generate" | "upload" | "batch";
  basis: string | null;
}

export type StoryboardShotStatus = "idle" | "queued" | "running" | "done" | "failed" | "stale" | "missing";
export type StoryboardStartFrom = "generate" | "upload" | "previous-shot" | "text";

export interface StoryboardShot {
  id: string;
  order: number;
  title: string;
  status: StoryboardShotStatus;
  stillPrompt: string;
  motionPrompt: string;
  referenceImages: string[];
  startFrom: StoryboardStartFrom;
  endImage: string | null;
  seconds: number;
  seed: number | null;
  locked: boolean;
  trimIn: number;
  trimOut: number | null;
  speed: number;
  takes: StoryboardTake[];
  activeTake: string | null;
  transition: StoryboardTransition;
  nativeSoundDb: number;
  redo: boolean;
}

export interface StoryboardAudioSource {
  path: string;
  hash: string | null;
  measuredSeconds: number | null;
  clonedVoice: boolean;
  label: string;
}

export interface StoryboardAudioClip {
  id: string;
  kind: "music" | "sfx" | "voice" | "import";
  source: StoryboardAudioSource;
  anchor: { shotId: string; offset: number } | null;
  start: number | null;
  trimIn: number;
  trimOut: number | null;
  gainDb: number;
  fadeIn: number;
  fadeOut: number;
  duck: boolean;
  loop: boolean;
  status: "ok" | "missing";
}

export interface StoryboardExportSettings {
  aiLabel: { embedTag: boolean };
  loudnessLufs: number | null;
  /** Write the captions as an .srt next to the film. */
  captionsSrt: boolean;
  /** Draw the captions into the picture. */
  captionsBurn: boolean;
  /** Also write each audio lane as its own .wav. */
  stems: boolean;
  /** Scale the finished film up (lanczos) at the final encode; "none" keeps the project size. */
  upscale: StoryboardUpscale;
}

/** One caption. With `clipId` the times are seconds in that clip's source; without, seconds on the timeline. */
export interface StoryboardCaption {
  id: string;
  clipId: string | null;
  start: number;
  end: number;
  text: string;
}

export type StoryboardBatchState = "queued" | "running" | "done" | "failed" | "cancelled" | "interrupted";

export interface StoryboardBatchResult {
  shotId: string;
  kind: "still" | "clip";
  state: "queued" | "running" | "done" | "failed" | "skipped" | "cancelled";
  takeId: string | null;
  error: string | null;
}

export interface StoryboardBatchRecord {
  id: string;
  what: "still" | "clip" | "both";
  force: boolean;
  state: StoryboardBatchState;
  createdAt: number;
  finishedAt: number | null;
  results: StoryboardBatchResult[];
}

export interface StoryboardProject {
  schemaVersion: number;
  revision: number;
  id: string;
  name: string;
  createdAt: number;
  defaults: StoryboardDefaults;
  references: StoryboardReferences;
  shots: StoryboardShot[];
  audio: StoryboardAudioClip[];
  export: StoryboardExportSettings;
  captions: StoryboardCaption[];
  batches: StoryboardBatchRecord[];
  undoStack: number[];
  redoStack: number[];
}

export interface StoryboardShotSpan {
  id: string;
  start: number;
  end: number;
  duration: number;
  source_in: number;
  source_out: number;
  transition_type: string;
  transition_seconds: number;
  measured: boolean;
  ready: boolean;
}

export interface StoryboardAudioSpan {
  id: string;
  start: number;
  duration: number;
  trim_in: number;
  end: number;
}

export interface StoryboardTimeline {
  total: number;
  shots: StoryboardShotSpan[];
  audio: StoryboardAudioSpan[];
  problems: string[];
}

/** What every project route answers: the project and its computed timeline. */
export interface StoryboardProjectPayload {
  project: StoryboardProject;
  timeline: StoryboardTimeline;
}

export type StoryboardListRow =
  | { id: string; name: string; revision: number; shots: number; createdAt: number; error?: undefined }
  | { id: string; error: string; name?: undefined; revision?: undefined; shots?: undefined; createdAt?: undefined };

export type StoryboardBatchWhat = "still" | "clip" | "both";

export interface StoryboardBatchStatus {
  batch: StoryboardBatchRecord;
  revision: number;
  /** Live batches only: where the run is, elapsed seconds, and time left when earlier steps give a basis. */
  progress?: import("./chat").JobProgress | null;
}

export type StoryboardRenderState = "running" | "done" | "failed" | "cancelled" | "interrupted";

/** One render job as the backend describes it (`RenderJob.describe`). */
export interface StoryboardRenderJob {
  renderId: string;
  projectId: string;
  preview: boolean;
  state: StoryboardRenderState;
  /** 0 to 1. */
  progress: number;
  /** The assembler's current stage: shots, join, measure or mix. */
  message: string;
  /** Project-relative path of the finished film (`exports/...`), when done. */
  file?: string;
  error?: string;
  /** Epoch seconds. */
  startedAt?: number;
  /** Files written next to the film (project-relative): the caption file and the lane stems. */
  extras?: string[];
  warnings?: string[];
  /** An interrupted, failed or cancelled render can be started again; finished shots are reused. */
  resumable?: boolean;
  shotsReady?: number;
  shotsTotal?: number;
}

/** Waveform peaks of one project file: 0 to 1 per bucket (empty when the file has no sound). */
export interface StoryboardPeaks {
  peaks: number[];
  durationS: number;
}

export interface StoryboardShotFields {
  title?: string;
  stillPrompt?: string;
  motionPrompt?: string;
  referenceImages?: string[];
  startFrom?: StoryboardStartFrom;
  seconds?: number;
  seed?: number | null;
  trimIn?: number;
  trimOut?: number | null;
  speed?: number;
  transition?: StoryboardTransition;
  nativeSoundDb?: number;
}

export interface StoryboardAudioFields {
  anchor?: { shotId: string; offset: number };
  start?: number;
  trimIn?: number;
  trimOut?: number;
  gainDb?: number;
  fadeIn?: number;
  fadeOut?: number;
  duck?: boolean;
  loop?: boolean;
  label?: string;
}

/** Arguments per op name (backend_service/storyboard/ops.py). */
export interface StoryboardOpArgs {
  edit_project: { name?: string; defaults?: Partial<StoryboardDefaults>; export?: Record<string, unknown> };
  /** `image` and `images` take local paths, `upload:` handles, or a `media/` path of this project (a shot still). */
  add_reference: { kind: "character" | "place"; name?: string; description?: string; image?: string; images?: string[] };
  /** `image` replaces every picture; `addImage` appends one (four at most); `removeImage` drops one by position. */
  edit_reference: { refId: string; name?: string; description?: string; image?: string; addImage?: string; removeImage?: number };
  remove_reference: { refId: string };
  add_shot: StoryboardShotFields & { index?: number };
  edit_shot: StoryboardShotFields & { shotId: string };
  remove_shot: { shotId: string };
  reorder: { order: string[] };
  lock: { shotId: string; locked?: boolean };
  pick_take: { shotId: string; takeId: string };
  mark_redo: { shotId: string; redo?: boolean };
  import_take: { shotId: string; clip?: string; still?: string };
  relink: { id: string; path: string };
  add_audio: StoryboardAudioFields & {
    kind: StoryboardAudioClip["kind"];
    path: string;
    clonedVoice?: boolean;
  };
  edit_audio: StoryboardAudioFields & { clipId: string };
  remove_audio: { clipId: string };
  set_captions: { clipId?: string | null; cues: { start: number; end: number; text: string }[] };
  add_caption: { start: number; end: number; text: string; clipId?: string | null };
  edit_caption: { captionId: string; start?: number; end?: number; text?: string };
  remove_caption: { captionId: string };
}

export type StoryboardOpName = keyof StoryboardOpArgs;
