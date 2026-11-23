/**
 * Lip sync in the Video Studio (FU-372) and the Storyboard: the 3 to 8 s rule, the take count (D-110: 3 takes for a
 * sung vocal, 1 for speech, 4 as the best-quality setting) and the time estimate, mirroring
 * `backend_service/lipsync_runtime.py` (`shot_frames`, `estimate_seconds`) and `lipsync_takes.py`.
 */

export const LIP_SYNC_MIN_SECONDS = 3;
export const LIP_SYNC_MAX_SECONDS = 8;
export const LIP_SYNC_TAKES_SPEECH = 1;
export const LIP_SYNC_TAKES_SUNG = 3;
export const LIP_SYNC_TAKES_BEST = 4;
const SCORE_SECONDS = 10;
const LOAD_SECONDS = 35;

/** The catalog row's measured lip-sync figures (`lipSync` on the 4-bit LTX-2.3 row). */
export interface LipSyncBlock {
  engine?: string;
  basis?: string;
  secondsPerShot?: number;
  secondsPerShotWarm?: number;
  estimateRef?: { width: number; height: number; frames: number };
  minSeconds?: number;
  maxSeconds?: number;
}

/** What the Video Studio sends with a lip-sync run. `takes` null = the default for the vocal. */
export interface LipSyncInput {
  audioPath: string | null;
  start: number;
  singing: boolean;
  takes: number | null;
}

export const EMPTY_LIP_SYNC: LipSyncInput = { audioPath: null, start: 0, singing: false, takes: null };

export function lipSyncTakeCount(takes: number | null | undefined, singing: boolean): number {
  if (takes == null) return singing ? LIP_SYNC_TAKES_SUNG : LIP_SYNC_TAKES_SPEECH;
  return Math.max(1, Math.min(LIP_SYNC_TAKES_BEST, Math.round(takes)));
}

/** Frames the backend renders (8k+1, rounded down) and the seconds they last. */
export function lipSyncClip(numFrames: number, fps: number): { frames: number; seconds: number; ok: boolean } {
  const frames = Math.max(9, Math.floor((Math.round(numFrames) - 1) / 8) * 8 + 1);
  const seconds = fps > 0 ? (frames - 1) / fps : 0;
  const ok = seconds >= LIP_SYNC_MIN_SECONDS - 1e-6 && seconds <= LIP_SYNC_MAX_SECONDS + 1e-6;
  return { frames, seconds, ok };
}

/** Seconds for one shot of `takes` takes, as the backend estimates it; null when the row has no measurement. */
export function lipSyncEstimateSeconds(
  block: LipSyncBlock | null | undefined,
  width: number,
  height: number,
  frames: number,
  takes: number,
  warm = false,
): number | null {
  const ref = block?.estimateRef;
  const perShot = block?.secondsPerShot;
  if (!ref || typeof perShot !== "number") return null;
  const scale = (width * height * frames) / (ref.width * ref.height * ref.frames);
  const cold = LOAD_SECONDS + (perShot - LOAD_SECONDS) * scale;
  const warmTake = typeof block?.secondsPerShotWarm === "number" ? block.secondsPerShotWarm * scale : cold;
  const count = Math.max(1, Math.round(takes));
  const total = (warm ? warmTake : cold) + (count - 1) * warmTake + (count > 1 ? SCORE_SECONDS * count : 0);
  return Math.round(total * 10) / 10;
}

/** "about 5 min" / "about 50 s" for an estimate. */
export function formatLipSyncMinutes(seconds: number): { value: number; unit: "s" | "min" } {
  if (seconds < 90) return { value: Math.round(seconds), unit: "s" };
  return { value: Math.round(seconds / 60), unit: "min" };
}

/** The request fields a lip-sync run adds (none without a vocal or on a row that cannot lip-sync). */
export function lipSyncPayloadFields(
  rowCanLipSync: boolean,
  input: LipSyncInput,
): { audioPath?: string; audioStartSeconds?: number; lipSyncSinging?: boolean; lipSyncTakes?: number } {
  if (!rowCanLipSync || !input.audioPath) return {};
  return {
    audioPath: input.audioPath,
    audioStartSeconds: input.start,
    ...(input.singing ? { lipSyncSinging: true } : {}),
    ...(input.takes != null ? { lipSyncTakes: input.takes } : {}),
  };
}
