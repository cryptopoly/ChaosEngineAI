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
  /** "parakeet-mlx" | "nemotron-asr-mlx" | "mlx-whisper" | "faster-whisper" | "transformers-asr" (NVIDIA GPU). */
  backend?: string;
  /** Can translate speech into English. */
  translate?: boolean;
  /** False until the voice pack is installed. */
  backendInstalled?: boolean;
  /** The weights' licence id ("Apache-2.0", "MIT", ...); GPU rows also a note in English ("Untested here.") and the GPU memory they need. */
  licence?: string | null;
  note?: string | null;
  minVramGb?: number;
  /** GPU rows: ``minVramGb`` comes from a run on an RTX 4090 ("measured") or from the weights' size ("estimate"). */
  vramBasis?: "measured" | "estimate";
  /** GPU rows: word timings need the forced aligner (``VoiceRuntime.aligner``). */
  needsAligner?: boolean;
  /** GPU rows: why this GPU cannot run it (memory or compute capability), or null. */
  gpuFacts?: { kind: "memory" | "compute"; need: number; have: number } | null;
}

export interface SttAligner {
  key: string;
  /** Hugging Face repo, downloaded through ``/api/models/download``. */
  id: string;
  name: string;
  sizeGb: number;
  licence: string;
  /** GPU memory it adds while loaded. */
  vramGb: number;
  installed: boolean;
}

export interface TtsVoice {
  id: string;
  name: string;
  gender?: "female" | "male";
  language: string;
  languageLabel?: string;
  /** False when this machine needs an extra install for the language. */
  available?: boolean;
  /** Nothing can make it run on this machine (Japanese on the MLX lane). */
  neverAvailable?: boolean;
}

export interface VoiceSettings {
  sttModel: string;
  language: string;
  /** Chat read-aloud: any listed voice whose ``chatUse.read`` is true (D-72). */
  ttsVoice: string;
  /** Voice calls: "" means the chat voice; any listed voice whose ``chatUse.call`` is true. */
  callVoice?: string;
  /** What reads when the chosen chat or call voice cannot (D-75): an installed voice, "" for none; Kokoro's Heart by default. */
  fallbackVoice?: string;
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

/** One part of the voice runtime: the speech packages, the Kokoro quick voices, turn detection, speaker labels. */
export interface VoiceRuntimePart {
  id: "packages" | "kokoro" | "turns" | "speakers";
  sizeGb: number;
}

/** A model the speech worker has in memory; ``peakGb`` is the measured peak (null when not measured). */
export interface VoiceInMemory {
  lane: "stt" | "tts";
  name: string;
  peakGb: number | null;
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
  /** The runtime's parts and their size on disk (an older backend sends none). */
  parts?: VoiceRuntimePart[];
  /** Models the speech worker holds now. */
  inMemory?: VoiceInMemory[];
  install: VoiceInstallJob;
  sttModels: SttModel[];
  /** The forced aligner, the "Word timings" download for the GPU dictation models (D-101); null or absent elsewhere. */
  aligner?: SttAligner | null;
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
  /** Speech saved since D-67 W7: the voice's display name and kind (older items have only ``voice``). */
  voiceName?: string | null;
  voiceKind?: VoiceEntryKind | null;
  /** Made with a cloned voice: labelled as AI-made and watermarked. */
  clonedVoice?: boolean | null;
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

/** What a voice's model can do (set by the catalog). The last four are set on model rows only. */
export interface VoiceCapabilities {
  speak: boolean;
  style: boolean;
  clone: boolean;
  design: boolean;
  /** The Expression slider (Chatterbox V3). */
  expression?: boolean;
  /** Sound tags such as [laugh] in the text (Chatterbox-Turbo, Orpheus). */
  tags?: boolean;
  /** Named voices of its own (Orpheus, the Chatterbox default voices). */
  presets?: boolean;
  /** The request's language picks the language (Chatterbox V3). */
  languageSelect?: boolean;
}

/** A slider's range from the catalog. */
export interface VoiceSliderRange {
  min: number;
  max: number;
  step: number;
  default: number;
}

/** What the Speak screen draws for a model, from data (no per-model code). */
export interface VoiceModelControls {
  /** Documented sound tags, inserted as [name]. */
  tags?: string[];
  /** Chatterbox-Turbo's style tokens: in its tokenizer, not documented by its authors. */
  styleTags?: string[];
  expression?: VoiceSliderRange;
  guidance?: VoiceSliderRange;
  /** Language codes the model can be told to speak. */
  languages?: string[];
}

export type VoiceFamily = "kokoro" | "qwen3-tts" | "voxcpm2" | "chatterbox" | "chatterbox-turbo" | "orpheus";

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
  /** The Speak picker's group. */
  group?: "quick" | "natural" | "designed" | "cloned";
  /** The model family that speaks it by default. */
  family?: VoiceFamily;
  /** Cloned voices: the voice sample's length, for the per-model minimum. */
  durationS?: number | null;
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
  /** ffmpeg is missing here: pitch (every voice) and speed (all but quick voices) cannot be changed. */
  needsFfmpeg?: boolean;
  consent?: { speakerName: string; basis: "own" | "permission"; consentedAt: string; appVersion?: string };
  /** Where it may speak in Chat (D-72): read-aloud and calls; cloned voices read aloud only. */
  chatUse?: { read: boolean; call: boolean };
  /** Measured first-audio latency and memory for one chat sentence; null when not measured. */
  chatCost?: VoiceChatCost | null;
}

/** ``scripts/measure-voice-models.py --chat`` on the catalog row (Kokoro's own constant for quick voices). */
export interface VoiceChatCost {
  /** Seconds to the first audio with the model already loaded. */
  firstAudioS: number;
  /** Seconds to the first audio when the speech worker has to start and load the model. */
  firstAudioColdS?: number | null;
  peakGb?: number | null;
  measuredOn?: string | null;
  /** Warm first audio above 2 s: allowed for calls, but labelled. */
  slowForCalls: boolean;
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

/** One row of ``GET /api/voice/models``: a speech model with its install state. */
export interface VoiceModelRow {
  key: string;
  name: string;
  summary: string;
  repo: string;
  sizeGb: number;
  /** Model plus companion repos: what Download fetches. */
  downloadGb?: number;
  languages: string;
  default: boolean;
  kind: "base" | "custom" | "design" | "turbo" | "voxcpm" | "chatterbox" | "orpheus";
  family?: VoiceFamily;
  /** Where the row runs: "mlx" (Apple Silicon) or "cuda" (the speech GPU add-on on an NVIDIA PC). */
  lanes?: string[];
  capabilities: VoiceCapabilities;
  controls?: VoiceModelControls;
  /** Output sample rate (48 kHz for VoxCPM2). */
  sampleRate?: number;
  /** The shortest voice sample it clones from, in seconds; null for models that do not clone. */
  minReferenceS?: number | null;
  /** A watermark the model adds itself ("perth" on the CUDA Chatterbox rows); null otherwise. */
  builtinWatermark?: string | null;
  /** Null (or missing) when the model has not been measured. */
  estimates?: VoiceModelEstimates | null;
  supported: boolean;
  /** Why a model that suits this machine's lane still cannot run on its GPU (memory, compute capability); null otherwise. */
  unsupportedReason?: string | null;
  /** The same as data: `memory` (GB needed and present) or `compute` (capability needed and present); null otherwise. */
  unsupportedFacts?: { kind: "memory" | "compute"; need: number; have: number } | null;
  installed: boolean;
  /** Fit on this machine from the measured peak (``helpers/voice_fit.py``); read with ``readFit``. */
  fit?: unknown;
  licence?: { id: string; name: string; url: string; note?: string; baseUrl?: string };
  download: {
    state: "idle" | "downloading" | "done" | "failed" | "cancelled";
    error: string | null;
    /** The failure as a code, plain sentence and action (``download_blocked``: sign in to Hugging Face; ``download_failed``: retry). */
    errorInfo?: VoiceErrorInfo | null;
    phase?: "weights" | "watermark" | null;
    /** While downloading: 0..1 and GB on disk against the whole download (model, companions). */
    progress?: number | null;
    downloadedGb?: number | null;
    totalGb?: number | null;
  };
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
  /** Language code or "auto" (Chatterbox V3 speaks the chosen one). */
  language?: string;
  /** Chatterbox V3: calm (0) to dramatic (1). */
  expression?: number;
  /** Chatterbox V3: how closely a voice sample is followed. */
  guidance?: number;
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

/**
 * Voice errors (voice studio UX audit, change 6). A failed voice request keeps its HTTP status and ``detail`` (the
 * raw text) and gains ``code``, ``message`` (one plain sentence) and ``action`` (the one button to offer).
 * The text on screen comes from ``features/voice/voiceErrorText.ts``; ``message`` is the backend's English fallback.
 */
export const VOICE_ERROR_CODES = [
  "runtime_missing",
  "runtime_outdated",
  "addon_missing",
  "model_missing",
  "watermark_missing",
  "gpu_memory",
  "gpu_unsupported",
  "unsupported_here",
  "ffmpeg_missing",
  "reference_too_short",
  "reference_too_long",
  "reference_unreadable",
  "audio_unreadable",
  "language_unsupported",
  "permission_required",
  "cloning_not_allowed",
  "voice_not_found",
  "no_backup_voice",
  "download_blocked",
  "download_failed",
  "worker_busy",
  "worker_timeout",
  "worker_stopped",
  "unknown",
] as const;
export type VoiceErrorCode = (typeof VOICE_ERROR_CODES)[number];

export const VOICE_ERROR_ACTIONS = [
  "install-voice-runtime",
  "install-addon",
  "download-model",
  "smaller-model",
  "pick-another-model",
  "install-ffmpeg",
  "pick-another-language",
  "change-sample",
  "confirm-permission",
  "add-hf-token",
  "retry",
  "none",
] as const;
export type VoiceErrorAction = (typeof VOICE_ERROR_ACTIONS)[number];

/** The voice install add-on to tick when the code is ``addon_missing`` (``VoiceRuntime`` extras); null otherwise. */
export type VoiceErrorExtra = "kokoro" | "chinese" | "speech-gpu" | "stt-gpu";

export interface VoiceErrorInfo {
  code: VoiceErrorCode;
  /** One plain English sentence; the screen prefers its own translated text for ``code``. */
  message: string;
  action: VoiceErrorAction;
  extra: VoiceErrorExtra | null;
}

/** A voice error response body: ``detail`` is the raw text (shown under Details), the rest is ``VoiceErrorInfo``. */
export interface VoiceErrorBody extends VoiceErrorInfo {
  detail: string;
}
