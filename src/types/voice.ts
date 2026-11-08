export interface SttModel {
  /** Hugging Face repo the model loads from on this machine. */
  id: string;
  /** Catalog key, the same on every platform. */
  key?: string;
  name: string;
  languages?: string;
  sizeGb: number;
  installed: boolean;
  default: boolean;
  /** "parakeet-mlx" | "nemotron-asr-mlx" | "mlx-whisper" | "faster-whisper". */
  backend?: string;
  /** Can translate speech into English. */
  translate?: boolean;
  /** False until the voice pack is installed. */
  backendInstalled?: boolean;
}

export interface TtsVoice {
  id: string;
  name: string;
  gender?: "female" | "male";
  language: string;
  languageLabel?: string;
  /** False when this machine needs an extra install for the language. */
  available?: boolean;
}

export interface VoiceSettings {
  sttModel: string;
  language: string;
  ttsVoice: string;
  /** The voice last used on the Speak screen, apart from the chat voice; "" until Speak has been used. */
  speakVoice?: string;
  speed: number;
  autoSend: boolean;
  autoRead: boolean;
  /** Conversation mode: silence that ends a turn. */
  vadSilenceMs: number;
  /** Conversation mode: speaking over the model interrupts it. */
  bargeIn: boolean;
  idleUnloadMinutes: number;
  /** Agents and other machines may use cloned voices (off by default; only this window can change it). */
  allowAgentCloning?: boolean;
}

export interface VoiceInstallJob {
  phase: string;
  phaseLabel: string;
  phaseIndex: number;
  phaseTotal: number;
  percent: number;
  message: string;
  error: string | null;
  log: string[];
  running: boolean;
  extras: string[];
}

export interface VoiceExtra {
  id: string;
  label: string;
  installed: boolean;
}

export interface VoiceRuntime {
  platform: string;
  /** "mlx" on Apple Silicon, "portable" on Windows / Linux / Intel Macs. */
  lane: "mlx" | "portable";
  /** The voice pack (its own Python environment) is installed. */
  installed: boolean;
  sttAvailable: boolean;
  ttsAvailable: boolean;
  ttsModelInstalled: boolean;
  /** Conversation mode's turn detector is installed (voice packs from before it need an update). */
  vadAvailable: boolean;
  /** Speaker labels can be added to transcripts. */
  speakersAvailable: boolean;
  /** The installed pack lacks something this version provides. */
  updateAvailable: boolean;
  sttBackend: string | null;
  ttsBackend: string | null;
  cuda: boolean;
  nvidiaGpu: boolean;
  extras: VoiceExtra[];
  /** When the voice pack install last passed its check (local time, ISO), if known. */
  installedAt?: string | null;
  workerPid: number | null;
  install: VoiceInstallJob;
  sttModels: SttModel[];
  ttsVoices: TtsVoice[];
  settings: VoiceSettings;
}

export interface TranscriptWord {
  start: number;
  end: number;
  text: string;
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  /** Speaker number (0, 1, ...) when speaker labels were asked for. */
  speaker?: number | null;
  words?: TranscriptWord[];
}

export type VoiceLibraryKind = "transcript" | "speech";

/** A Library item as listed (no segments; ``preview`` is the start of the text). */
export interface VoiceLibrarySummary {
  id: string;
  kind: VoiceLibraryKind;
  title: string;
  preview: string;
  createdAt: number;
  updatedAt: number;
  /** Audio file extension, or null when the item has no audio. */
  audio: string | null;
  speakers: Record<string, string>;
  speakerCount: number;
  language?: string | null;
  model?: string;
  voice?: string | null;
  durationS?: number | null;
  source?: string | null;
  task?: "transcribe" | "translate";
  /** The Storyboard project it was made for (set by the backend, absent otherwise). */
  projectId?: string | null;
  projectName?: string | null;
}

export interface VoiceLibraryItem extends Omit<VoiceLibrarySummary, "preview" | "speakerCount"> {
  text: string;
  segments: TranscriptSegment[];
  /** Where the audio file is on this machine (null when the item has none); only the single-item route sends it. */
  audioPath?: string | null;
}

export interface VoiceLibraryPatch {
  title?: string;
  text?: string;
  segments?: TranscriptSegment[];
  speakers?: Record<string, string>;
}

/** What a voice's model can do (set by the catalog). */
export interface VoiceCapabilities {
  speak: boolean;
  style: boolean;
  clone: boolean;
  design: boolean;
}

export type VoiceEntryKind = "builtin" | "preset" | "design" | "clone";

/** One row of ``GET /api/voice/voices``: built-in, preset, designed or cloned. */
export interface VoiceEntry {
  id: string;
  name: string;
  kind: VoiceEntryKind;
  language: string;
  languageLabel?: string;
  gender?: "female" | "male" | null;
  about?: string;
  /** Catalog key of the model used by default (``kokoro`` for built-ins). */
  model: string;
  /** Every catalog key that can speak this voice. */
  models: string[];
  capabilities: VoiceCapabilities;
  available: boolean;
  /** Quick (Kokoro) voices: the model author's quality grade ("A", "B-", ...); null when ungraded. */
  grade?: string | null;
  /** Quick voices that need a voice pack add-on to run here ("chinese"); null otherwise. */
  addon?: string | null;
  /** Plain words for why it cannot be used yet. */
  needs?: string | null;
  consent?: { speakerName: string; basis: "own" | "permission"; consentedAt: string; appVersion?: string };
}

/** Speed and memory of a speech model from the catalog; ``basis`` says whether it was measured, and where. */
export interface VoiceModelEstimates {
  basis: "measured" | "estimate";
  /** Machine and date of the run, e.g. "Apple M4 Max, 64 GB, 2026-11-08". */
  measuredOn?: string;
  coldStartSeconds?: number;
  /** Wall seconds to make 10 s of speech. */
  secondsPer10sAudio?: number;
  wallSecondsShortLine?: number;
  /** Worker memory while speaking and after it finished. */
  peakGb?: number;
  idleGb?: number;
}

/** One row of ``GET /api/voice/models``: a Qwen3-TTS model with its install state. */
export interface VoiceModelRow {
  key: string;
  name: string;
  summary: string;
  repo: string;
  sizeGb: number;
  languages: string;
  default: boolean;
  kind: "base" | "custom" | "design";
  capabilities: VoiceCapabilities;
  /** Null (or missing) when the model has not been measured. */
  estimates?: VoiceModelEstimates | null;
  supported: boolean;
  installed: boolean;
  licence?: { id: string; name: string; url: string };
  download: { state: "idle" | "downloading" | "done" | "failed"; error: string | null; phase?: "weights" | "watermark" | null };
}

/** Body of ``POST /api/voice/speak``. */
export interface SpeakBody {
  text: string;
  voice: string;
  style?: string;
  speed?: number;
  pitch?: number;
  seed?: number | null;
  model?: string;
}

/** Answer of ``POST /api/voice/speak``. */
export interface SpeakResult {
  /** Base64 WAV. */
  audio: string;
  durationS: number;
  chunks: number;
  voice: string;
  kind: VoiceEntryKind;
  model: string;
  modelName: string;
  clonedVoice: boolean;
  libraryId: string | null;
  notes: string[];
}
