import { createContext, useContext, type ReactNode } from "react";
import type { VoiceRuntime, VoiceSettings } from "../../types";

/**
 * What chat, compare and the Voice tabs need to know about voice: whether
 * the voice pack is installed, the user's voice settings, and how to send
 * the user to the install screen.
 */
export interface VoiceContextValue {
  runtime: VoiceRuntime | null;
  /** OS name from the Tauri shell ("Darwin", "Windows", "Linux"), for help text. */
  platform: string;
  refresh: () => void;
  saveSettings: (patch: Partial<VoiceSettings>) => Promise<void>;
  openVoiceModels: () => void;
}

const DEFAULT_SETTINGS: VoiceSettings = {
  sttModel: "",
  language: "",
  ttsVoice: "af_heart",
  speed: 1,
  autoSend: false,
  autoRead: false,
  vadSilenceMs: 700,
  bargeIn: true,
  idleUnloadMinutes: 10,
};

const VoiceContext = createContext<VoiceContextValue>({
  runtime: null,
  platform: "",
  refresh: () => {},
  saveSettings: async () => {},
  openVoiceModels: () => {},
});

export function VoiceProvider({ value, children }: { value: VoiceContextValue; children: ReactNode }) {
  return <VoiceContext.Provider value={value}>{children}</VoiceContext.Provider>;
}

export function useVoice(): VoiceContextValue & { settings: VoiceSettings } {
  const value = useContext(VoiceContext);
  return { ...value, settings: value.runtime?.settings ?? DEFAULT_SETTINGS };
}
