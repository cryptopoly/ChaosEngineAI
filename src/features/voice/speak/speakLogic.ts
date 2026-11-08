/** Pure rules for the Speak and Voices screens: which model runs, whether the style box is usable. */

import type { VoiceEntry, VoiceModelRow } from "../../../types";

export const DESIGN_VOICE_ID = "design";
export const MAX_SPEAK_CHARS = 200_000;
/** One request carries at most this much text (the backend limit). */
export const MAX_REQUEST_CHARS = 20_000;

/** Style chips: the label key and the instruction sent to the model (empty clears the box). */
export const STYLE_CHIPS: Array<{ id: string; instruction: string }> = [
  { id: "neutral", instruction: "" },
  { id: "angry", instruction: "Speak in an angry tone." },
  { id: "sad", instruction: "Speak in a sad, quiet tone." },
  { id: "excited", instruction: "Speak in an excited, energetic tone." },
  { id: "whisper", instruction: "Speak in a soft whisper." },
  { id: "calm", instruction: "Speak in a calm, relaxed tone." },
];

export type VoiceFilter = "all" | "mine" | "builtin";

/** The model that will run: mirrors the backend rule (style needs a style model; installed ones come first). */
export function chooseModel(
  voice: VoiceEntry | null,
  models: VoiceModelRow[],
  wantsStyle: boolean,
  preferredKey = "",
): VoiceModelRow | null {
  if (!voice || voice.kind === "builtin") return null;
  const rows = voice.models.map((key) => models.find((m) => m.key === key)).filter((m): m is VoiceModelRow => Boolean(m));
  // A cloning model the person picked, when it is downloaded.
  const preferred = voice.kind === "clone" ? rows.find((m) => m.key === preferredKey && m.installed) : undefined;
  if (preferred) return preferred;
  const rank = (m: VoiceModelRow) => [wantsStyle && !m.capabilities.style ? 1 : 0, m.sizeGb] as const;
  const sort = (list: VoiceModelRow[]) =>
    [...list].sort((a, b) => {
      const [ax, as] = rank(a);
      const [bx, bs] = rank(b);
      return ax - bx || as - bs;
    });
  const installed = rows.filter((m) => m.installed);
  return sort(installed.length > 0 ? installed : rows)[0] ?? null;
}

/** The downloaded cloning models for a cloned voice; the Speak screen offers a choice when there are two or more. */
export function installedCloneModels(voice: VoiceEntry | null, models: VoiceModelRow[]): VoiceModelRow[] {
  if (!voice || voice.kind !== "clone") return [];
  return voice.models
    .map((key) => models.find((m) => m.key === key))
    .filter((m): m is VoiceModelRow => Boolean(m && m.installed));
}

export interface StyleState {
  /** Whether the style box can be edited. */
  enabled: boolean;
  /** Why it cannot, in a key the screen translates. */
  reason: "builtin" | "clone" | "smallModel" | "none";
  /** A model to download to unlock style (preset voices only). */
  unlockWith: VoiceModelRow | null;
}

/** Style is usable for a design voice (the box is its description) and for a preset when a style model is installed. */
export function styleState(voice: VoiceEntry | null, models: VoiceModelRow[]): StyleState {
  if (!voice) return { enabled: false, reason: "none", unlockWith: null };
  if (voice.kind === "builtin") return { enabled: false, reason: "builtin", unlockWith: null };
  if (voice.kind === "clone") return { enabled: false, reason: "clone", unlockWith: null };
  if (voice.kind === "design") return { enabled: true, reason: "none", unlockWith: null };
  const styled = voice.models.map((key) => models.find((m) => m.key === key)).filter((m): m is VoiceModelRow => Boolean(m && m.capabilities.style));
  if (styled.some((m) => m.installed)) return { enabled: true, reason: "none", unlockWith: null };
  return { enabled: false, reason: "smallModel", unlockWith: styled[0] ?? null };
}

export interface ModelNote {
  /** The model that runs, or null for a built-in voice (part of the voice pack). */
  model: VoiceModelRow | null;
  /** The model must be downloaded before Generate works. */
  needsDownload: boolean;
  downloading: boolean;
}

export function modelNote(voice: VoiceEntry | null, models: VoiceModelRow[], wantsStyle: boolean, preferredKey = ""): ModelNote {
  const model = chooseModel(voice, models, wantsStyle, preferredKey);
  return {
    model,
    needsDownload: Boolean(voice && voice.kind !== "builtin" && (!model || !model.installed)),
    downloading: model?.download.state === "downloading",
  };
}

/** Whether Generate can run. A design voice needs its description in the style box. */
export function canGenerate(voice: VoiceEntry | null, text: string, style: string, note: ModelNote): boolean {
  if (!voice || !text.trim() || note.needsDownload) return false;
  if (voice.kind === "design" && !style.trim()) return false;
  return true;
}

/** Whether a request-sized piece of text is too long for one cloned-voice request. */
export function cloneTextTooLong(text: string): boolean {
  return text.length > MAX_REQUEST_CHARS;
}

/** The text of a cloned-voice request: one request, so a [pause] marker becomes a paragraph break. */
export function cloneRequestText(text: string): string {
  return text.replace(/\s*\[pause\]\s*/gi, "\n\n").trim();
}

export function matchesFilter(voice: VoiceEntry, filter: VoiceFilter): boolean {
  if (filter === "all") return true;
  if (filter === "builtin") return voice.kind === "builtin" || voice.kind === "preset";
  return voice.kind === "clone";
}

/** Voices shown in the picker: built-in ones that cannot be used here are left out, as before. */
export function pickerVoices(voices: VoiceEntry[]): VoiceEntry[] {
  return voices.filter((v) => v.kind !== "builtin" || v.available);
}

/** Which voice to select first: the chat voice if it is listed, else the first usable one. */
export function defaultVoice(voices: VoiceEntry[], preferred: string): string {
  const list = pickerVoices(voices);
  return list.find((v) => v.id === preferred)?.id ?? list.find((v) => v.kind === "builtin")?.id ?? list[0]?.id ?? preferred;
}
