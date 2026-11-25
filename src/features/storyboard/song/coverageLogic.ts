/**
 * Lip-sync coverage and the "Does someone sing on screen?" flow (Storyboard UX audit changes 3 and 8), as pure
 * helpers: the length rule in plain words, each coverage row's state and reason as text, the fix code mapped to the
 * action that already exists for it, and which cut the answer asks for.
 */
import type {
  StoryboardLipSyncCoverage,
  StoryboardLipSyncFix,
  StoryboardLipSyncReason,
  StoryboardLipSyncRow,
  StoryboardLipSyncState,
  StoryboardProject,
} from "../../../types";

/** The shortest shot that can lip-sync (`coverage.py` `MIN_SHOT_SECONDS`). */
export const LIP_SYNC_MIN_SHOT_SECONDS = 2;
/** A lip-sync clip lasts 3 to 8 s; a 2 to 3 s shot gets silent room up to 3 s (`lipsync_runtime`). */
export const LIP_SYNC_CLIP_MIN_SECONDS = 3;
export const LIP_SYNC_CLIP_MAX_SECONDS = 8;
/** The shot lengths "Yes, lip-sync" cuts the song into. */
export const SUNG_SHOT_MIN_SECONDS = 3;
export const SUNG_SHOT_MAX_SECONDS = 5;

/** A translatable sentence: the key, its English, and the values it takes. */
export interface Text {
  key: string;
  defaultValue: string;
  values?: Record<string, string | number>;
}

/** The length rule, said once for the song panel and the shot editor. */
export const LIP_SYNC_RANGE: Text = {
  key: "board.song.lipSyncRange",
  defaultValue:
    "A shot of {min} s or more can lip-sync. Its clip lasts {clipMin} to {clipMax} s: a shot shorter than {clipMin} s gets silent room to fill it.",
  values: { min: LIP_SYNC_MIN_SHOT_SECONDS, clipMin: LIP_SYNC_CLIP_MIN_SECONDS, clipMax: LIP_SYNC_CLIP_MAX_SECONDS },
};

export type SingAnswer = "yes" | "no";

/** The answer a project already gave: lip sync used before means yes; otherwise not asked yet. */
export function initialSingAnswer(project: StoryboardProject): SingAnswer | null {
  if (project.song?.coverage) return "yes";
  return project.shots.some((shot) => Boolean(shot.vocalClip)) ? "yes" : null;
}

/** The cut the answer asks for: short singing shots, or the ordinary cut to the beat. */
export function cutOptions(answer: SingAnswer | null): { minShotSeconds?: number; maxShotSeconds?: number } {
  return answer === "yes" ? { minShotSeconds: SUNG_SHOT_MIN_SECONDS, maxShotSeconds: SUNG_SHOT_MAX_SECONDS } : {};
}

/** True once lip sync has been used on the project, so the table stays on show whatever the answer is now. */
export function coverageInUse(coverage: StoryboardLipSyncCoverage | null | undefined): coverage is StoryboardLipSyncCoverage {
  return Boolean(coverage && (coverage.source !== null || coverage.counts.sings > 0));
}

/** What a fix button does, using actions that already exist. */
export type FixAction =
  | { kind: "open-length"; shotId: string }
  | { kind: "unlock"; shotId: string }
  | { kind: "real-speed"; shotId: string }
  | { kind: "assign-vocals" }
  | { kind: "make-lipsync"; shotIds: string[] };

export function fixAction(fix: StoryboardLipSyncFix, shotId: string): FixAction {
  switch (fix) {
    case "lengthen":
    case "shorten":
      return { kind: "open-length", shotId };
    case "unlock":
      return { kind: "unlock", shotId };
    case "real-speed":
      return { kind: "real-speed", shotId };
    case "assign-vocals":
      return { kind: "assign-vocals" };
    case "make-lipsync":
      return { kind: "make-lipsync", shotIds: [shotId] };
  }
}

/** The row's button: a skipped shot's fix, or making a waiting shot's clip. Unassigned rows share the one
 * "Give each shot its vocal" button under the table, since that run covers every shot. Lengthen and Shorten open
 * the shot editor, so they need a way to open it. */
export function rowFix(row: StoryboardLipSyncRow, canOpenShot = true): StoryboardLipSyncFix | null {
  if (row.state !== "skipped" && row.state !== "waiting") return null;
  if (!canOpenShot && (row.fix === "lengthen" || row.fix === "shorten")) return null;
  return row.fix;
}

const round1 = (n: number) => Math.round(n * 10) / 10;

export function fixLabel(fix: StoryboardLipSyncFix, row: Pick<StoryboardLipSyncRow, "minSeconds" | "maxSeconds">): Text {
  switch (fix) {
    case "lengthen":
      return { key: "board.song.coverage.fix.lengthen", defaultValue: "Lengthen to {n} s or more", values: { n: round1(row.minSeconds) } };
    case "shorten":
      return { key: "board.song.coverage.fix.shorten", defaultValue: "Shorten to {n} s or less", values: { n: round1(row.maxSeconds) } };
    case "unlock":
      return { key: "board.song.coverage.fix.unlock", defaultValue: "Unlock" };
    case "real-speed":
      return { key: "board.song.coverage.fix.realSpeed", defaultValue: "Play at real speed" };
    case "assign-vocals":
      return { key: "board.song.coverage.fix.assign", defaultValue: "Give each shot its vocal" };
    case "make-lipsync":
      return { key: "board.song.coverage.fix.make", defaultValue: "Make its clip" };
  }
}

export function stateLabel(state: StoryboardLipSyncState): Text {
  switch (state) {
    case "synced":
      return { key: "board.song.coverage.state.synced", defaultValue: "Synced" };
    case "waiting":
      return { key: "board.song.coverage.state.waiting", defaultValue: "Waiting for its clip" };
    case "skipped":
      return { key: "board.song.coverage.state.skipped", defaultValue: "Not lip-synced" };
    case "unassigned":
      return { key: "board.song.coverage.state.unassigned", defaultValue: "No vocal" };
  }
}

/** Why a shot is not lip-synced, in plain words (from the reason code, not the backend's English). */
export function reasonText(code: StoryboardLipSyncReason, row: Pick<StoryboardLipSyncRow, "seconds" | "minSeconds" | "maxSeconds">): Text {
  switch (code) {
    case "other-speed":
      return { key: "board.song.coverage.reason.otherSpeed", defaultValue: "Plays at another speed; lip sync needs real time." };
    case "locked":
      return { key: "board.song.coverage.reason.locked", defaultValue: "The shot is locked." };
    case "outside-song":
      return { key: "board.song.coverage.reason.outsideSong", defaultValue: "Outside the song; it plays as an ordinary clip." };
    case "too-short":
      return {
        key: "board.song.coverage.reason.tooShort",
        defaultValue: "Too short: {seconds} s, a lip-sync shot needs at least {min} s.",
        values: { seconds: round1(row.seconds), min: round1(row.minSeconds) },
      };
    case "too-long":
      return {
        key: "board.song.coverage.reason.tooLong",
        defaultValue: "Too long: {seconds} s, a lip-sync shot here lasts at most {max} s.",
        values: { seconds: round1(row.seconds), max: round1(row.maxSeconds) },
      };
    case "no-singing":
      return { key: "board.song.coverage.reason.noSinging", defaultValue: "No singing here; it plays as an ordinary clip." };
    case "not-assigned":
      return { key: "board.song.coverage.reason.notAssigned", defaultValue: "No vocal; it plays as an ordinary clip." };
  }
}

/** Shots with a vocal and no current clip, for "Make N lip-sync clips". */
export function waitingShotIds(coverage: StoryboardLipSyncCoverage): string[] {
  return coverage.shots.filter((row) => row.state === "waiting").map((row) => row.shotId);
}
