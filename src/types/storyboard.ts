/** Storyboard project schema and API payloads (mirrors backend_service/storyboard/models.py). */

import type { JobProgress } from "./chat";

/** The aspect ratio (called the export preset in the stored project). */
export type StoryboardExportPreset = "16:9" | "9:16" | "1:1";
export type StoryboardResolution = "480p" | "720p" | "1080p" | "1440p" | "2160p";
/** What a project's size reads as: a preset, another short side ("576p" for 1024x576) or "custom" (FU-213). */
export type StoryboardResolutionLabel = StoryboardResolution | `${number}p` | "custom";
export type StoryboardUpscale = "none" | "1440p" | "2160p";

export interface StoryboardDefaults {
  width: number;
  height: number;
  fps: number;
  /** Stills for Make and Make drafts; null is Automatic (the fastest installed model). */
  imageModel: string | null;
  /** Stills for the final pass on kept shots; null is Automatic (the best-quality installed model). */
  finalImageModel?: string | null;
  videoModel: string | null;
  voice: string | null;
  exportPreset: StoryboardExportPreset;
  resolution: StoryboardResolutionLabel;
}

/** One picture of a character or place (project format 3). */
export interface StoryboardRefImage {
  path: string;
  hash: string | null;
}

/**
 * Character lock field names (Storyboard UX audit 2026-11). The only place they are spelled: every use goes
 * through these constants, so a backend rename is a one-line change here.
 */
/** On a character: it appears in every shot unless a shot leaves it out. */
export const IN_EVERY_SHOT = "inEveryShot";
/** On a shot: ids of in-every-shot characters this shot leaves out. */
export const CAST_OUT = "castOut";
/** On a planned shot from the planner: the reference ids of the characters in it. */
export const PLANNED_CHARACTERS = "characters";

/** A character, place or style: a name, a few words and up to four pictures (front, side, detail...). */
export interface StoryboardReference {
  id: string;
  name: string;
  description: string;
  images: StoryboardRefImage[];
  /** A style's main colours (`#rrggbb`), read from its pictures (S15). Empty for characters and places. */
  palette?: string[];
  /** A character in every shot (absent reads as false). */
  [IN_EVERY_SHOT]?: boolean;
}

export interface StoryboardReferences {
  characters: StoryboardReference[];
  places: StoryboardReference[];
  /** The look of the film (S15); absent in projects saved before it. */
  styles?: StoryboardReference[];
}

export type StoryboardReferenceKind = "character" | "place" | "style";

/** How captions look when drawn into the picture (S15). */
export interface StoryboardCaptionLook {
  font: string | null;
  colour: string;
  position: "bottom" | "middle" | "top";
}

export type StoryboardLogoCorner = "top-left" | "top-right" | "bottom-left" | "bottom-right";
/** How style references reach the still model: as pictures when it takes them (auto), as words, or not at all. */
export type StoryboardStyleUse = "auto" | "words" | "off";

/** The project's brand kit (S15). */
export interface StoryboardBrand {
  logo: StoryboardRefImage | null;
  logoCorner: StoryboardLogoCorner;
  captions: StoryboardCaptionLook;
  styleUse: StoryboardStyleUse;
}

/** The idea behind the film (S8). */
export interface StoryboardPlan {
  brief: string;
  filmSeconds: number | null;
}

/** One line of the direction chat (kept with the project, not part of Undo). */
export interface StoryboardPlanMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  at: number;
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
  /** A quick draft or a full take; absent (an older project) reads as final. */
  renderPass?: StoryboardRenderPass;
}

export type StoryboardRenderPass = "draft" | "final";

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
  /** Lip sync (D-103): the vocal clip (WAV or FLAC, under media/) the shot's clip follows. */
  vocalClip?: string | null;
  /** FU-373: where in the vocal file the clip starts, in seconds. */
  vocalStart?: number;
  /** The vocal is sung: singing wording and, by default, 3 takes kept by the sync score (D-110). */
  vocalSinging?: boolean;
  /** 1 to 4 takes; null = 3 for a sung vocal, 1 for speech. */
  vocalTakes?: number | null;
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
  /** Chosen for the final pass. */
  kept?: boolean;
  /** In-every-shot characters this shot leaves out (absent reads as none). */
  [CAST_OUT]?: string[];
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
  /** The brand kit on the film (S15); absent in projects saved before it. Draw the brand logo in its corner. */
  showLogo?: boolean;
  /** A title card before the first shot and an end card after the last, each `cardSeconds` long. */
  titleCard?: boolean;
  endCard?: boolean;
  /** Card text; empty means the project name. */
  titleText?: string;
  endText?: string;
  cardSeconds?: number;
}

/** One caption. With `clipId` the times are seconds in that clip's source; without, seconds on the timeline. */
export interface StoryboardCaption {
  id: string;
  clipId: string | null;
  start: number;
  end: number;
  text: string;
}

/** `paused`: stopped between steps (Pause, or the app stopped while it ran) and keeps its remaining steps (S-M1). */
export type StoryboardBatchState = "queued" | "running" | "paused" | "done" | "failed" | "cancelled" | "interrupted";

export interface StoryboardBatchResult {
  shotId: string;
  kind: "still" | "clip";
  state: "queued" | "running" | "done" | "failed" | "skipped" | "cancelled";
  takeId: string | null;
  /** The raw error text (Technical details). */
  error: string | null;
  /** Why a failed step failed (`backend_service/storyboard/failure.py`); absent or null before codes were stored. */
  code?: StoryboardFailureCode | null;
}

/** A failed step's stable code. A newer backend may send a code this list lacks: treat it as "unknown". */
export type StoryboardFailureCode =
  | "out-of-memory"
  | "model-missing"
  | "lipsync-model"
  | "download-blocked"
  | "video-tools-missing"
  | "size-over-cap"
  | "references-unsupported"
  | "no-prompt"
  | "disk-full"
  | "cancelled"
  | "unknown";

/** The one button a failure offers. */
export type StoryboardFailureAction = "smaller-size" | "open-models" | "install-video-tools" | "edit-shot" | "free-disk" | "retry";

/** A failed step for the app (`GET .../batches/{id}` `failures`, and `guide.failures` with `batchId`). */
export interface StoryboardFailure {
  shotId: string;
  kind: "still" | "clip";
  code: StoryboardFailureCode;
  /** One plain English sentence (the app shows its own translated text for the code). */
  message: string;
  action: StoryboardFailureAction;
  /** The raw error text. */
  error: string | null;
  batchId?: string;
}

export interface StoryboardBatchRecord {
  id: string;
  what: StoryboardBatchWhat;
  force: boolean;
  /** A Make drafts run. */
  draft?: boolean;
  state: StoryboardBatchState;
  createdAt: number;
  finishedAt: number | null;
  results: StoryboardBatchResult[];
  /** S-M1: the last pause (epoch seconds), why it paused, the last resume and how many resumes. */
  pausedAt?: number | null;
  pauseReason?: "user" | "restart" | null;
  resumedAt?: number | null;
  resumes?: number;
}

/** A paused batch's bar data (`GET .../batches/{id}` `progress`). */
export interface StoryboardPausedProgress extends JobProgress {
  paused: true;
  pauseReason?: "user" | "restart" | null;
  totalSteps: number;
  step: number;
  remainingSteps: number;
  remaining: { still: number; clip: number };
  etaSeconds?: number;
  /** "clips": from steps this batch finished; "project": from other takes of this project. */
  etaBasis?: "clips" | "project";
}

/** One part of the attached song (S-M1). Labels: intro, outro, or a letter (parts that sound alike share it). */
export interface StoryboardSongSection {
  start: number;
  end: number;
  label: string;
  /** 0 to 1: how loud this part is within the song. */
  energy?: number | null;
}

/** The song a music video is cut to (S-M1). */
export interface StoryboardSong {
  path: string;
  hash: string | null;
  title: string;
  seconds: number;
  bpm: number | null;
  beats: number[];
  sections: StoryboardSongSection[];
  lyrics: string[];
  /** The music clip it put on the mixer, or null. */
  clipId: string | null;
  libraryId: string | null;
  /** The analysis is an estimate. */
  rough: boolean;
  note: string;
  analysedAt: number;
  /** The last "Give each shot its vocal" run, per shot (audit change 3); absent or null before one. */
  coverage?: StoryboardVocalCoverage | null;
}

/** Why a shot is not lip-synced (`backend_service/storyboard/coverage.py` `SKIP_CODES`). */
export type StoryboardLipSyncReason =
  | "other-speed"
  | "locked"
  | "outside-song"
  | "too-short"
  | "too-long"
  | "no-singing"
  | "not-assigned";

/** The fix a coverage row offers (null: nothing to fix, the shot plays as an ordinary clip). */
export type StoryboardLipSyncFix = "real-speed" | "unlock" | "lengthen" | "shorten" | "assign-vocals" | "make-lipsync";

/** synced: the clip follows the vocal; waiting: it has a vocal, no current clip; skipped: a reason code; unassigned: no vocal given yet. */
export type StoryboardLipSyncState = "synced" | "waiting" | "skipped" | "unassigned";

/** The stored record of one lip-sync run (`project.song.coverage`). */
export interface StoryboardVocalCoverage {
  source: "stem" | "song";
  at: number;
  shots: { shotId: string; code: StoryboardLipSyncReason | null; message: string; songStart: number; length: number }[];
}

/** One row of the coverage table (`guide.coverage.shots`). */
export interface StoryboardLipSyncRow {
  shotId: string;
  number: number;
  /** Song seconds the shot covers. */
  songStart: number;
  songEnd: number;
  seconds: number;
  state: StoryboardLipSyncState;
  /** null when the shot has a vocal. */
  code: StoryboardLipSyncReason | null;
  message: string;
  fix: StoryboardLipSyncFix | null;
  /** The lip-sync length rule for this shot: at least 2 s, at most 8 s less its lead-in. */
  minSeconds: number;
  maxSeconds: number;
}

/** Lip-sync coverage, read live from the shots and the stored record (null without a song). */
export interface StoryboardLipSyncCoverage {
  source: "stem" | "song" | null;
  at: number | null;
  counts: Record<StoryboardLipSyncState, number> & { total: number; sings: number };
  shots: StoryboardLipSyncRow[];
}

export interface StoryboardSize {
  width: number;
  height: number;
}

/** What one take was made at (audit change 2). Sizes are null when not recorded (uploads, old takes). */
export interface StoryboardTakeQuality {
  pass: StoryboardRenderPass;
  draft: boolean;
  still: StoryboardSize | null;
  clip: (StoryboardSize & { fps?: number }) | null;
  /** The half no longer matches the project's format (a final at another size or rate, a draft at another shape). */
  stillStale: boolean;
  clipStale: boolean;
}

export interface StoryboardQuality {
  /** The film's size and rate. */
  target: StoryboardSize & { fps: number };
  /** Half size before a model's minimum side; a draft take records its real size. */
  draftSize: StoryboardSize;
  /** Shots whose active take is a draft. */
  draftShots: string[];
  /** Shots whose active take no longer matches the format. */
  staleShots: string[];
  /** takes[shotId][takeId]. */
  takes: Record<string, Record<string, StoryboardTakeQuality>>;
}

export type StoryboardReadinessCode =
  | "no-shots"
  | "no-clips"
  | "failed-shots"
  | "missing-clips"
  | "drafts"
  | "stale-shots"
  | "lipsync-waiting"
  | "lipsync-gap"
  | "song-length"
  | "clip-sound-audible"
  | "format"
  | "lipsync-coverage"
  | "lipsync-no-singing"
  | "clip-sound-muted";

/** An action code a readiness item or a stage maps to its button. */
export type StoryboardGuideAction =
  | "attach-song"
  | "open-cast"
  | "cut-plan"
  | "plan-shots"
  | "open-storyboard"
  | "watch-batch"
  | "resume-batch"
  | "retry-failed"
  | "make-drafts"
  | "make-missing"
  | "make-final"
  | "choose-final"
  | "make-clips"
  | "remake-stale"
  | "assign-vocals"
  | "make-lipsync"
  | "open-coverage"
  | "open-mixer"
  | "open-check"
  | "export"
  | StoryboardLipSyncFix;

export interface StoryboardReadinessItem {
  code: StoryboardReadinessCode;
  message: string;
  shotIds: string[];
  action: StoryboardGuideAction | null;
  /** Numbers behind the sentence (sizes, seconds, counts, `reason` of a lip-sync gap). */
  data: Record<string, number | string>;
}

/** Ready to export? (audit change 6). `ready` is false only when something stops the export. */
export interface StoryboardReadiness {
  ready: boolean;
  stops: StoryboardReadinessItem[];
  worthFixing: StoryboardReadinessItem[];
  forInfo: StoryboardReadinessItem[];
}

export type StoryboardStageId = "song" | "look" | "shots" | "drafts" | "final" | "lipsync" | "sound" | "check" | "export";
export type StoryboardStageStatus = "not-started" | "in-progress" | "done" | "needs-attention";

export interface StoryboardStage {
  id: StoryboardStageId;
  /** false: hidden for this project (no song: song and lipsync). */
  applies: boolean;
  optional: boolean;
  status: StoryboardStageStatus;
  /** The one-sentence next step. */
  message: string;
  action: StoryboardGuideAction | null;
}

/** The stage rail (audit changes 1 and 9); `current` is null when every stage is done. */
export interface StoryboardStages {
  current: StoryboardStageId | null;
  stages: StoryboardStage[];
}

/** The guide block on every project payload and `GET .../guide` (`backend_service/storyboard/guide.py`). */
export interface StoryboardGuide {
  quality: StoryboardQuality;
  coverage: StoryboardLipSyncCoverage | null;
  failures: StoryboardFailure[];
  readiness: StoryboardReadiness;
  stages: StoryboardStages;
}

/** Where one card sits on the freestyle board, in board pixels at zoom 1 (S-M2). */
export interface StoryboardBoardPos {
  x: number;
  y: number;
}

export interface StoryboardBoard {
  pos: Record<string, StoryboardBoardPos>;
}

export interface StoryboardProject {
  schemaVersion: number;
  revision: number;
  id: string;
  name: string;
  createdAt: number;
  defaults: StoryboardDefaults;
  references: StoryboardReferences;
  /** S8 and S15; absent in projects saved before them. */
  plan?: StoryboardPlan;
  brand?: StoryboardBrand;
  planChat?: StoryboardPlanMessage[];
  /** S-M1: the song the film is cut to; absent or null when none. */
  song?: StoryboardSong | null;
  /** S-M2: the freestyle board's card positions (shot or reference id to board pixels); absent in older projects. */
  board?: StoryboardBoard;
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

/** What every project route answers: the project, its computed timeline and the guide block. */
export interface StoryboardProjectPayload {
  project: StoryboardProject;
  timeline: StoryboardTimeline;
  /** Optional only so older fixtures type-check; the backend always sends it. */
  guide?: StoryboardGuide;
}

export type StoryboardListRow =
  | { id: string; name: string; revision: number; shots: number; createdAt: number; error?: undefined }
  | { id: string; error: string; name?: undefined; revision?: undefined; shots?: undefined; createdAt?: undefined };

/** `final`: the kept shots again with the quality model, then their clips. */
export type StoryboardBatchWhat = "still" | "clip" | "both" | "final" | "redo-clip" | "redo-still";

export interface StoryboardBatchStatus {
  batch: StoryboardBatchRecord;
  revision: number;
  /** Where the run is, elapsed seconds, and time left when earlier steps give a basis; a paused batch gives its
   *  remaining steps instead (S-M1). */
  progress?: import("./chat").JobProgress | StoryboardPausedProgress | null;
  /** Each failed step as a code, a sentence, an action and the raw text (audit change 5). */
  failures?: StoryboardFailure[];
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
  vocalStart?: number;
  vocalSinging?: boolean;
  vocalTakes?: number | null;
  [CAST_OUT]?: string[];
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
  add_reference: { kind: StoryboardReferenceKind; name?: string; description?: string; image?: string; images?: string[]; [IN_EVERY_SHOT]?: boolean };
  /** `image` replaces every picture; `addImage` appends one (four at most); `removeImage` drops one by position; `palette` trims a style's colours. */
  edit_reference: { refId: string; name?: string; description?: string; image?: string; addImage?: string; removeImage?: number; palette?: string[]; [IN_EVERY_SHOT]?: boolean };
  remove_reference: { refId: string };
  add_shot: StoryboardShotFields & { index?: number };
  edit_shot: StoryboardShotFields & { shotId: string };
  remove_shot: { shotId: string };
  reorder: { order: string[] };
  lock: { shotId: string; locked?: boolean };
  pick_take: { shotId: string; takeId: string };
  mark_redo: { shotId: string; redo?: boolean };
  set_kept: { shotId: string; kept?: boolean };
  import_take: { shotId: string; clip?: string; still?: string };
  /** The picture a clip should end on (LTX-2 models only); `image` is a path, an `upload:` handle or a project `media/` path, empty or absent clears it. */
  set_end_image: { shotId: string; image?: string | null };
  set_vocal_clip: { shotId: string; audio?: string | null };
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
  /** S8: the brief and the length aimed for. */
  set_plan: { brief?: string; filmSeconds?: number; clearFilmSeconds?: boolean };
  /** S8: planned shots in one step (one Undo); up to 500 (a cut plan). */
  apply_plan: { mode: "replace" | "append"; shots: StoryboardPlannedShot[] };
  /** S-M1: change the attached song by hand (sections, lyrics, tempo, title). */
  edit_song: { title?: string; bpm?: number; clearBpm?: boolean; sections?: StoryboardSongSection[]; lyrics?: string[] };
  /** S-M1: take the song off the project (and, by default, its mixer clip). */
  clear_song: { removeClip?: boolean };
  /** S-M2: move freestyle board cards (shot or reference ids), one Undo. */
  place_cards: { positions: Record<string, StoryboardBoardPos> };
  /** S-M2: a picture (path, `upload:` handle) as a new shot with that still as its take, placed on the board; one Undo. */
  board_add_shot: { still: string; title?: string; index?: number; x?: number; y?: number };
  /** S8: the direction chat's changes in one step (one Undo). */
  plan_changes: StoryboardPlanChanges;
  add_plan_message: { role: "user" | "assistant"; text: string };
  clear_plan_chat: Record<string, never>;
  /** S15: the brand kit; `logo` is a path, an `upload:` handle or a project `media/` path. */
  set_brand: {
    logo?: string;
    clearLogo?: boolean;
    logoCorner?: StoryboardLogoCorner;
    captions?: { font?: string; clearFont?: boolean; colour?: string; position?: StoryboardCaptionLook["position"] };
    styleUse?: StoryboardStyleUse;
  };
}

/** A shot the chat model proposed (S8). */
export interface StoryboardPlannedShot {
  title: string;
  stillPrompt: string;
  motionPrompt: string;
  seconds: number;
  referenceImages?: string[];
  /** A cut plan sets the out point so the clip ends on the music (S-M1). */
  trimOut?: number | null;
  /** The planner's characters for this shot (reference ids); applied as the shot's references. */
  [PLANNED_CHARACTERS]?: string[];
  /** In-every-shot characters this shot leaves out (the planner or cut plan sets it). */
  [CAST_OUT]?: string[];
}

/** `POST .../cut-plan`: shots cut to the song; apply with `apply_plan` (mode replace). */
export interface StoryboardCutPlan {
  shots: StoryboardPlannedShot[];
  mode: "replace";
  totalSeconds: number;
  clipSeconds: number;
  clipSecondsSource: "model" | "request" | "default";
  videoModel: string | null;
  sections: (StoryboardSongSection & { shots: number })[];
  note: string;
  rough: boolean;
}

/** What `POST .../song` found in the audio. */
export interface StoryboardSongAnalysis {
  seconds: number;
  bpm: number | null;
  beats: number;
  sections: StoryboardSongSection[];
  rough: boolean;
  note: string;
}

/** What `POST .../song/vocals` did (FU-371): shots that got their slice of the vocal, and why the others did not. */
export interface StoryboardSongVocals {
  source: "stem" | "song";
  attached: number;
  /** `reason` is the plain sentence, `code` the machine-readable reason. */
  skipped: { shotId: string; number: number; code: StoryboardLipSyncReason; reason: string; songStart: number; length: number }[];
  /** Cuts moved to phrase starts (with a stem). */
  snapped?: number;
  /** The coverage table after the run (also in the payload's `guide.coverage`). */
  coverage?: StoryboardLipSyncCoverage | null;
}

export interface StoryboardPlanChanges {
  edits: ({ shotId: string } & Partial<Omit<StoryboardPlannedShot, "referenceImages">>)[];
  add: (StoryboardPlannedShot & { after: string | null })[];
  remove: string[];
}

export type StoryboardOpName = keyof StoryboardOpArgs;
