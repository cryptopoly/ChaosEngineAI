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
}
