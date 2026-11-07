export type VoicePackMode = "install" | "update" | "repair" | "healthy";

export interface VoicePackHealthInput {
  installed: boolean;
  updateAvailable: boolean;
  vadAvailable: boolean;
  ttsModelInstalled: boolean;
  /** Phase of the most recent install job ("done", "error", ...). */
  jobPhase: string;
}

/**
 * Which call to action the voice pack panel leads with. Only a missing pack, an out-of-date pack
 * or a failed check gets a primary button; a healthy pack gets a quiet "Reinstall" link.
 */
export function voicePackMode(input: VoicePackHealthInput): VoicePackMode {
  if (!input.installed) return "install";
  if (input.updateAvailable || !input.vadAvailable) return "update";
  if (input.jobPhase === "error" || !input.ttsModelInstalled) return "repair";
  return "healthy";
}

/** The Mandarin add-on is offered only where the backend lists it and it is not installed yet. */
export function chineseAddonPending(extras: Array<{ id: string; installed: boolean }>): boolean {
  return extras.some((extra) => extra.id === "chinese" && !extra.installed);
}
