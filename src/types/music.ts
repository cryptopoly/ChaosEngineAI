/**
 * Music payloads: the catalog, the runtime and install state, the estimate, progress, results and the
 * Music Library. Shapes follow `backend_service/routes/music.py`, `routes/music_library.py` and
 * `catalog/music_models.py`; an older or newer backend may send less, so readers treat most fields as
 * optional.
 */

import type { ModelFit } from "./models";
import type { GenerationProgressSnapshot } from "./progress";

export type MusicMakes = "songs" | "instrumental" | "sfx" | "loops";
export type MusicCommercial = "yes" | "limited" | "no";

export interface MusicLicence {
  id: string;
  name: string;
  commercial: MusicCommercial;
  gated: boolean;
  summaryLines?: string[];
  url?: string;
  /** A line the licence asks you to show where the sounds are shared ("Powered by Stability AI"). */
  attribution?: string;
  /** Yearly revenue (US$) above which the free commercial licence ends. */
  revenueCapUsd?: number;
  note?: string;
}

export interface MusicEstimates {
  basis: "measured" | "estimated";
  measuredOn?: string;
  wallSecondsAt?: Array<[number, number]>;
  /** The model works on its whole window whatever length is asked, so every clip costs the same. */
  fixedWindow?: boolean;
  secondsPer10sAudio?: number;
  coldStartSeconds?: number;
  peakGb?: number | null;
  idleGb?: number | null;
  diskGb?: number | null;
  outputMbPerMinute?: number | null;
}

export interface MusicDefaultSettings {
  seconds?: number;
  variations?: number;
  steps?: number;
  format?: string;
  sampleRate?: number;
  channels?: number;
  [key: string]: unknown;
}

export interface MusicModelVariant {
  id: string;
  /** `sound` rows (Stable Audio Open) make short effects and loops in the Studio's Sounds mode. */
  kind?: "music" | "sound";
  /** Which pack runs it: the music pack by default, or the separate sounds pack. */
  pack?: "music" | "sound";
  familyId: string;
  familyName?: string;
  name: string;
  provider: string;
  repo: string;
  revision?: string;
  link: string;
  engine: string;
  /** The engine's own model name; the runtime reports it as `loadedModel`. */
  engineModel?: string;
  runtime?: string;
  makes: MusicMakes[];
  platforms: string[];
  taskSupport?: string[];
  sizeGb: number;
  sizeNote?: string;
  requires?: string[];
  large?: boolean;
  minMemoryGb?: number | null;
  maxSeconds: number;
  minSeconds: number;
  licence: MusicLicence;
  estimates: MusicEstimates;
  defaultSettings: MusicDefaultSettings;
  note?: string;
  // Added by the backend per variant.
  weightsSource?: "checkpoints" | "hf-cache" | null;
  availableLocally?: boolean;
  missingRequires?: string[];
  packInstalled?: boolean;
  textEncoderReady?: boolean;
  ready?: boolean;
  fit?: ModelFit;
}

export interface MusicModelFamily {
  id: string;
  kind?: "music" | "sound";
  name: string;
  provider: string;
  headline?: string;
  summary?: string;
  badges?: string[];
  defaultVariantId: string;
  bestFitVariantId?: string | null;
  variants: MusicModelVariant[];
}

export interface MusicCatalogResponse {
  families: MusicModelFamily[];
}

export interface MusicInstallJob {
  phase: string;
  phaseLabel: string;
  phaseIndex: number;
  phaseTotal: number;
  percent: number;
  message: string;
  error: string | null;
  log: string[];
  startedAt: number;
  finishedAt: number;
  running: boolean;
}

export interface MusicSettings {
  idleUnloadMinutes: number;
}

/** The sounds pack and worker (Stable Audio Open), reported next to the music ones. */
export interface MusicSoundRuntime {
  lane: string;
  unavailableReason: string | null;
  installed: boolean;
  engine: string;
  workerRunning: boolean;
  loadedModel: string | null;
  install: MusicInstallJob;
}

export interface MusicRuntimeStatus {
  sound?: MusicSoundRuntime;
  lane: string;
  unavailableReason: string | null;
  installed: boolean;
  engine: string;
  engineVersion?: string;
  workerRunning: boolean;
  loadedModel: string | null;
  busy?: boolean | null;
  footprintGb?: number | null;
  idleUnloadMinutes: number;
  embedTag: boolean;
  libraryDir: string;
  settings: MusicSettings;
  install: MusicInstallJob;
}

export interface MusicRunEstimate {
  model: string;
  seconds: number;
  coldStartSeconds: number;
  warm: boolean;
  peakGb: number | null;
  idleGb: number | null;
  writesMb: number | null;
  modelDiskGb: number | null;
  basis: "measured" | "estimated";
  measuredOn?: string | null;
}

export interface MusicQueueTicket {
  id: string;
  kind: string;
  source: string;
  label: string;
  queuedAt?: number;
  startedAt?: number | null;
}

export interface MusicQueueSnapshot {
  running: MusicQueueTicket | null;
  waiting: MusicQueueTicket[];
}

export interface MusicEstimateResponse {
  estimate: MusicRunEstimate;
  fit: ModelFit | null;
  totalGb: number;
  queue: MusicQueueSnapshot;
}

/** The music progress route answers with the shared snapshot shape (`kind: "music"`, phases load, encode, compose, decode, save). */
export type MusicProgressSnapshot = GenerationProgressSnapshot;

export interface MusicGeneratePayload {
  modelId?: string | null;
  caption: string;
  lyrics?: string;
  instrumental?: boolean;
  seconds: number;
  variations?: 1 | 2 | 4;
  bpm?: number | null;
  key?: string | null;
  seed?: number | null;
  title?: string | null;
}

export interface MusicSoundPayload {
  modelId?: string | null;
  caption: string;
  seconds: number;
  loop?: boolean;
  variations?: 1 | 2 | 4;
  seed?: number | null;
  title?: string | null;
}

export interface MusicLibraryItem {
  id: string;
  kind: "track" | "sound";
  /** Sounds only: made as a seamless loop. */
  loop?: boolean;
  /** Sounds only: the licence and attribution line of the model that made it. */
  licence?: string;
  title: string;
  caption: string;
  lyrics: string;
  tags: string[];
  createdAt: number;
  updatedAt: number;
  audio: string;
  model?: string;
  modelName?: string;
  seed?: number;
  durationS?: number;
  sampleRate?: number;
  bpm?: number | null;
  key?: string | null;
  instrumental?: boolean;
  variation?: number;
  jobSeconds?: number;
  sizeBytes?: number;
  tagged?: boolean;
  /** The Storyboard project it was made for (set by the backend, absent otherwise). */
  projectId?: string | null;
  projectName?: string | null;
}

export interface MusicGenerateResponse {
  items: MusicLibraryItem[];
  seed: number;
  jobId: string;
  model: string;
  seconds: number | null;
  timings?: Record<string, number>;
  peakFootprintGb?: number | null;
}

export interface MusicLibraryListResponse {
  items: MusicLibraryItem[];
  count: number;
  diskBytes: number;
}

/** One finished track in the Studio's session list. */
export interface MusicResult {
  item: MusicLibraryItem;
  /** Wall seconds of the whole run (all variations). */
  jobSeconds: number | null;
  seed: number;
}
