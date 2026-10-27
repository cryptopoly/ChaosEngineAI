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
  /** "parakeet-mlx" | "mlx-whisper" | "faster-whisper". */
  backend?: string;
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
  speed: number;
  autoSend: boolean;
  autoRead: boolean;
  /** Conversation mode: silence that ends a turn. */
  vadSilenceMs: number;
  /** Conversation mode: speaking over the model interrupts it. */
  bargeIn: boolean;
  idleUnloadMinutes: number;
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
  sttBackend: string | null;
  ttsBackend: string | null;
  cuda: boolean;
  nvidiaGpu: boolean;
  extras: VoiceExtra[];
  workerPid: number | null;
  install: VoiceInstallJob;
  sttModels: SttModel[];
  ttsVoices: TtsVoice[];
  settings: VoiceSettings;
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}
