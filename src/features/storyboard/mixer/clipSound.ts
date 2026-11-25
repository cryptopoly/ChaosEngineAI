/**
 * Clip sound under a song (audit change 7). A film cut to a song plays the song, so each shot's own sound starts
 * muted (-60 dB). These rules say whether the clips are heard and what the one switch writes through `edit_shot`.
 * No React here so the rules are unit-tested directly.
 */
import type { StoryboardProject } from "../../../types";

/** What a shot's own sound is set to when it is muted and when it plays. */
export const CLIP_SOUND_MUTED_DB = -60;
export const CLIP_SOUND_ON_DB = 0;
/** A shot's own sound at or above this is heard over the song (the backend's `readiness.AUDIBLE_DB`). */
export const CLIP_SOUND_AUDIBLE_DB = -40;

export type ClipSoundState = "muted" | "playing" | "mixed";

/** Whether the clips carry a song: the lane header only talks about the song then. */
export function hasSong(project: Pick<StoryboardProject, "song">): boolean {
  return Boolean(project.song);
}

/** muted: no shot is heard; playing: every shot is; mixed: some are. */
export function clipSoundState(shots: { nativeSoundDb: number }[]): ClipSoundState {
  const heard = shots.filter((shot) => shot.nativeSoundDb > CLIP_SOUND_AUDIBLE_DB).length;
  if (heard === 0) return "muted";
  return heard === shots.length ? "playing" : "mixed";
}

/**
 * The `edit_shot` arguments for the "Let clip sound play" switch. On: every shot that is muted goes to 0 dB (a shot
 * the user already set to a level keeps it). Off: every shot that is heard goes to -60 dB. Only shots that change.
 */
export function clipSoundPatches(
  shots: { id: string; nativeSoundDb: number }[],
  on: boolean,
): { shotId: string; nativeSoundDb: number }[] {
  return shots
    .filter((shot) => (on ? shot.nativeSoundDb <= CLIP_SOUND_AUDIBLE_DB : shot.nativeSoundDb > CLIP_SOUND_AUDIBLE_DB))
    .map((shot) => ({ shotId: shot.id, nativeSoundDb: on ? CLIP_SOUND_ON_DB : CLIP_SOUND_MUTED_DB }));
}

/** The level text a shot editor draft gets from its switch: 0 when turned on, -60 when off. */
export function clipSoundDraftValue(on: boolean): string {
  return String(on ? CLIP_SOUND_ON_DB : CLIP_SOUND_MUTED_DB);
}

/** True when a shot editor draft's level text is heard over the song (a number above the audible line). */
export function clipSoundDraftOn(nativeSoundDb: string): boolean {
  const value = Number(nativeSoundDb);
  return nativeSoundDb.trim() !== "" && Number.isFinite(value) && value > CLIP_SOUND_AUDIBLE_DB;
}
