export type VoicePackMode = "install" | "update" | "repair" | "healthy";

export interface VoicePackHealthInput {
  installed: boolean;
  updateAvailable: boolean;
  vadAvailable: boolean;
  /** Kokoro's files are on disk. Not part of the health check: Kokoro is an optional add-on (D-75). */
  ttsModelInstalled?: boolean;
  /** Phase of the most recent install job ("done", "error", ...). */
  jobPhase: string;
}

/**
 * Which call to action the voice pack panel leads with. Only a missing pack, an out-of-date pack
 * or a failed check gets a primary button; a healthy pack gets a quiet "Reinstall" link. A runtime without the
 * Kokoro quick voices is healthy: they are an add-on that can be left out (D-75).
 */
export function voicePackMode(input: VoicePackHealthInput): VoicePackMode {
  if (!input.installed) return "install";
  if (input.updateAvailable || !input.vadAvailable) return "update";
  if (input.jobPhase === "error") return "repair";
  return "healthy";
}

/** Whether the runtime lists the Kokoro add-on as installed (D-75); an older backend without the add-on means yes. */
export function kokoroInstalled(extras: Array<{ id: string; installed: boolean }>, ttsModelInstalled: boolean): boolean {
  const addon = extras.find((extra) => extra.id === "kokoro");
  return addon ? addon.installed : ttsModelInstalled;
}

/** How many quick (Kokoro) voices and languages this machine can offer; voices that can never run here are not counted. */
export function quickVoiceSummary(voices: Array<{ language: string; neverAvailable?: boolean }>): { count: number; languages: number } {
  const offered = voices.filter((v) => !v.neverAvailable);
  return { count: offered.length, languages: new Set(offered.map((v) => v.language)).size };
}

/** The runtime's parts as display numbers (GB, one or two decimals); null when the backend sent no sizes. */
export function runtimePartsSummary(
  parts: Array<{ id: string; sizeGb: number }>,
): { total: string; packages: string; kokoro: string; small: string } | null {
  if (parts.length === 0) return null;
  const size = (id: string) => parts.find((p) => p.id === id)?.sizeGb ?? 0;
  const small = size("turns") + size("speakers");
  const total = parts.reduce((sum, p) => sum + p.sizeGb, 0);
  const gb = (value: number) => (value > 0 && value < 0.1 ? value.toFixed(2) : value.toFixed(1));
  return { total: gb(total), packages: gb(size("packages")), kokoro: gb(size("kokoro")), small: gb(small) };
}

/** The Mandarin add-on is offered only where the backend lists it and it is not installed yet. */
export function chineseAddonPending(extras: Array<{ id: string; installed: boolean }>): boolean {
  return extras.some((extra) => extra.id === "chinese" && !extra.installed);
}
