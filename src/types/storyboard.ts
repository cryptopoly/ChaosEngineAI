/** Storyboard project schema and API payloads (mirrors backend_service/storyboard/models.py). */

export type StoryboardExportPreset = "16:9" | "9:16" | "1:1";

export interface StoryboardDefaults {
  width: number;
  height: number;
  fps: number;
  imageModel: string | null;
  videoModel: string | null;
  voice: string | null;
  exportPreset: StoryboardExportPreset;
}

export interface StoryboardReference {
  id: string;
  name: string;
  description: string;
  image: string | null;
  hash: string | null;
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
export type StoryboardStartFrom = "generate" | "upload" | "previous-shot";

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
  aiLabel: { embedTag: boolean; endCard: boolean };
  loudnessLufs: number | null;
  endCardText: string;
  endCardSeconds: number;
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
}

export type StoryboardRenderState = "running" | "done" | "failed" | "cancelled";

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
  add_reference: { kind: "character" | "place"; name?: string; description?: string; image?: string };
  edit_reference: { refId: string; name?: string; description?: string; image?: string };
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
}

export type StoryboardOpName = keyof StoryboardOpArgs;
