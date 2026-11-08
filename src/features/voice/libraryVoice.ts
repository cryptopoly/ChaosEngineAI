/** Which voice made a Library speech item, for its card, and the Library filter another screen asks for. */

import type { VoiceEntryKind, VoiceLibrarySummary } from "../../types";

/**
 * The kind of a saved voice id: ``clone:`` ids are cloned, ``qwen3:`` ids natural presets, ``design`` the designed
 * voice, anything else a quick (Kokoro) voice. Used for items saved before ``voiceKind`` was stored.
 */
export function kindFromVoiceId(id: string): VoiceEntryKind {
  if (id.startsWith("clone:")) return "clone";
  if (id.startsWith("qwen3:")) return "preset";
  if (id === "design") return "design";
  return "builtin";
}

/** "af_heart" -> "Heart", "qwen3:Uncle_Fu" -> "Uncle Fu"; a clone id has no readable name, so it gives "". */
export function nameFromVoiceId(id: string): string {
  if (id.startsWith("clone:") || id === "design") return "";
  const bare = id.startsWith("qwen3:") ? id.slice("qwen3:".length) : /^[a-z]{2}_/.test(id) ? id.slice(3) : id;
  const spaced = bare.replace(/_/g, " ").trim();
  return spaced ? spaced.charAt(0).toUpperCase() + spaced.slice(1) : "";
}

export interface SpeechVoice {
  /** Empty when nothing readable is known (an old cloned-voice item). */
  name: string;
  kind: VoiceEntryKind;
}

/** The voice of a speech item: the stored name and kind, or what the id says for older items; null when none. */
export function speechVoice(item: Partial<Pick<VoiceLibrarySummary, "voice" | "voiceName" | "voiceKind" | "clonedVoice">>): SpeechVoice | null {
  const id = item.voice ?? "";
  if (!id && !item.voiceName) return null;
  const kind = item.voiceKind ?? (item.clonedVoice ? "clone" : kindFromVoiceId(id));
  return { name: item.voiceName ?? nameFromVoiceId(id), kind };
}

// The Library opens on a filter once when another screen asks (Speak's "Saved to Library" link opens Speech). The
// Library reads it when it mounts and clears it after, so opening the Library later starts on All again.
let pendingFilter: "speech" | "transcript" | null = null;

export function requestLibraryFilter(kind: "speech" | "transcript"): void {
  pendingFilter = kind;
}

export function peekLibraryFilter(): "speech" | "transcript" | "" {
  return pendingFilter ?? "";
}

export function clearLibraryFilter(): void {
  pendingFilter = null;
}
