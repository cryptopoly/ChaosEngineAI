/** Pure rules for the Clone a voice panel, kept apart so they can be tested without a DOM. */

import type { CloneModel } from "./cloneApi";

export type CloneBasis = "own" | "permission";

export interface NewVoiceDetails {
  basis: CloneBasis | null;
  speakerName: string;
  voiceName: string;
  /** "This is my own voice, or the speaker has given me permission to clone it." */
  confirmed: boolean;
}

export const MIN_SAMPLE_S = 3;
export const MAX_SAMPLE_S = 30;
/** File types the voice sample upload takes (the microphone recording is WAV). */
export const SAMPLE_EXTENSIONS = [".wav", ".mp3", ".m4a", ".flac", ".ogg"] as const;

/** Whether a file name has one of the sample types. */
export function isSampleFile(name: string): boolean {
  const lower = name.toLowerCase();
  return SAMPLE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

/** A voice can be saved once whose voice it is, the speaker's name, the tick and a usable sample are given. */
export function canSaveVoice(details: NewVoiceDetails, hasSample: boolean): boolean {
  return details.basis !== null && details.speakerName.trim().length > 0 && details.confirmed && hasSample;
}

/** A microphone recording must be 3 to 30 seconds (an uploaded file is measured by the backend). */
export function sampleLengthProblem(seconds: number | null): "short" | "long" | null {
  if (seconds == null) return null;
  if (seconds < MIN_SAMPLE_S) return "short";
  if (seconds > MAX_SAMPLE_S) return "long";
  return null;
}

/** Seconds of audio in a 16-bit mono WAV blob size (what the mic capture makes), header excluded. */
export function wavSeconds(bytes: number, sampleRate = 16_000): number {
  return Math.max(0, bytes - 44) / (2 * sampleRate);
}

/** The model to preselect: the downloaded default, else any downloaded one, else the default. */
export function pickCloneModel(models: CloneModel[]): CloneModel | null {
  const usable = models.filter((m) => m.supported);
  return (
    usable.find((m) => m.default && m.installed) ??
    usable.find((m) => m.installed) ??
    usable.find((m) => m.default) ??
    usable[0] ??
    null
  );
}
