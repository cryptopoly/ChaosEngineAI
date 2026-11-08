/**
 * Pure rules for the project voice picker: the four groups of `GET /api/voice/voices` in a fixed order, the
 * search, and whether a row can play a sample. No React here so the rules are unit-tested directly.
 */
import type { VoiceEntry, VoiceEntryKind } from "../../types";

export const VOICE_GROUP_ORDER: VoiceEntryKind[] = ["builtin", "preset", "design", "clone"];

export interface VoiceGroup {
  kind: VoiceEntryKind;
  voices: VoiceEntry[];
}

/** Voices whose name, language or description holds every word of the query, grouped; empty groups dropped. */
export function groupVoices(voices: readonly VoiceEntry[], query: string): VoiceGroup[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const matches = (voice: VoiceEntry) => {
    const text = [voice.name, voice.language, voice.languageLabel ?? "", voice.about ?? ""].join(" ").toLowerCase();
    return words.every((word) => text.includes(word));
  };
  return VOICE_GROUP_ORDER.map((kind) => ({
    kind,
    voices: voices.filter((voice) => voice.kind === kind && matches(voice)).sort((a, b) => a.name.localeCompare(b.name)),
  })).filter((group) => group.voices.length > 0);
}

/**
 * Play a sample only for a voice that can speak now and is not cloned: a cloned voice's speech is tagged and
 * saved to the Voice Library, which a quick sample must not do.
 */
export function canPlaySample(voice: VoiceEntry): boolean {
  return voice.available && voice.kind !== "clone";
}

/** The label of the stored voice id: its name when the list knows it, else the id as stored. */
export function voiceLabel(id: string | null, voices: readonly VoiceEntry[]): string | null {
  if (!id) return null;
  return voices.find((voice) => voice.id === id)?.name ?? id;
}
