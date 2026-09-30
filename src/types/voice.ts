export interface SttModel {
  id: string;
  name: string;
  sizeGb: number;
  installed: boolean;
  default: boolean;
}

export interface TtsVoice {
  id: string;
  name: string;
  language: string;
}

export interface VoiceRuntime {
  sttAvailable: boolean;
  ttsAvailable: boolean;
  platform: string;
  sttBackend: string | null;
  ttsBackend: string | null;
  sttModels: SttModel[];
  ttsVoices: TtsVoice[];
  /** Pip package keys to pass to install-package when backend is missing. */
  sttInstallPackages: string[] | null;
  ttsInstallPackages: string[] | null;
  /** Voice-model data TTS loads: HF repo id on Apple Silicon (downloadable
   * via the standard model machinery), null elsewhere (kokoro-onnx GitHub
   * release files, downloaded via the /api/voice/kokoro endpoints). */
  ttsModelRepo: string | null;
  ttsModelInstalled: boolean;
}
