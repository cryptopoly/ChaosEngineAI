/** Pure rules for the Clone a voice panel, kept apart so they can be tested without a DOM. */

import type { CloneModel } from "./cloneApi";

export type CloneBasis = "own" | "permission";

export interface ConsentForm {
  basis: CloneBasis | null;
  speakerName: string;
  voiceName: string;
  /** "This is my voice, or the speaker gave me permission." */
  confirmed: boolean;
}

export const MIN_SAMPLE_S = 3;
export const MAX_SAMPLE_S = 30;

/** The consent sentence can only be asked for once whose voice it is, a name and the statement are given. */
export function canStartConsent(form: ConsentForm): boolean {
  return form.basis !== null && form.speakerName.trim().length > 0 && form.confirmed;
}

/** A recording of the consent sentence (or a separate sample) must be 3 to 30 seconds. */
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

/** "4721" -> "4 7 2 1", read one digit at a time. */
export function spacedCode(code: string): string {
  return code.split("").join(" ");
}
