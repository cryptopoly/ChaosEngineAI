/**
 * What the guide block says about one shot card (Storyboard UX audit, changes 2 to 5): the quality label of each
 * take, the shot's latest failure, its lip-sync row and whether it has no character. `shotCardGuide` cuts that out
 * of `guide` so the card takes one optional prop; the chip texts are pure so they are tested without rendering.
 */
import type {
  StoryboardGuide,
  StoryboardLipSyncReason,
  StoryboardLipSyncRow,
  StoryboardProject,
  StoryboardShot,
  StoryboardFailure,
  StoryboardSize,
  StoryboardTakeQuality,
} from "../../types";
import type { TFn } from "./qualityText";
import { castOutOf, isInEveryShot } from "./castLock";

export interface ShotCardGuide {
  /** takeId to what it was made at (`guide.quality.takes[shotId]`). */
  takes?: Record<string, StoryboardTakeQuality>;
  /** The film's size. */
  target?: StoryboardSize | null;
  /** The size a draft is made at. */
  draftSize?: StoryboardSize | null;
  /** The shot's latest failure while it is failed. */
  failure?: StoryboardFailure | null;
  /** The shot's lip-sync row, when lip sync is in use in this project. */
  sync?: StoryboardLipSyncRow | null;
  /** A character is locked "In every shot", this shot leaves it out and its effective cast has no character. */
  noCharacter?: boolean;
}

/** Lip sync is in use once a run recorded its result or a shot holds a vocal. */
export function lipSyncInUse(guide: StoryboardGuide | null | undefined): boolean {
  const coverage = guide?.coverage;
  return Boolean(coverage && (coverage.source != null || coverage.counts.sings > 0));
}

export function shotCardGuide(
  project: StoryboardProject,
  guide: StoryboardGuide | null | undefined,
  shot: StoryboardShot,
): ShotCardGuide {
  if (!guide) return {};
  const characters = new Set(project.references.characters.map((c) => c.id));
  // Only a shot that left out a locked ("In every shot") character is worth a chip: without a lock, a shot with no
  // character is normal (its prompt can describe the singer in words).
  const leftOut = castOutOf(shot);
  const lockedOut = project.references.characters.some((c) => isInEveryShot(c) && leftOut.includes(c.id));
  const cast = guide.cast?.[shot.id];
  return {
    takes: guide.quality?.takes?.[shot.id],
    target: guide.quality?.target ?? null,
    draftSize: guide.quality?.draftSize ?? null,
    failure: guide.failures?.find((f) => f.shotId === shot.id) ?? null,
    sync: lipSyncInUse(guide) ? (guide.coverage?.shots.find((row) => row.shotId === shot.id) ?? null) : null,
    noCharacter: lockedOut && cast != null && !cast.some((id) => characters.has(id)),
  };
}

export interface SyncChip {
  text: string;
  title: string;
  /** `sings`: the shot lip-syncs (or will); `quiet`: it plays as an ordinary clip. */
  tone: "sings" | "quiet";
}

function reasonShort(t: TFn, code: StoryboardLipSyncReason | null): string {
  switch (code) {
    case "other-speed": return t("board.chip.reasonSpeed", { defaultValue: "other speed" });
    case "locked": return t("board.chip.reasonLocked", { defaultValue: "locked" });
    case "outside-song": return t("board.chip.reasonOutside", { defaultValue: "outside the song" });
    case "too-short": return t("board.chip.reasonShort", { defaultValue: "too short" });
    case "too-long": return t("board.chip.reasonLong", { defaultValue: "too long" });
    case "no-singing": return t("board.chip.reasonNoSinging", { defaultValue: "no singing here" });
    default: return t("board.chip.reasonNoVocal", { defaultValue: "no vocal yet" });
  }
}

function reasonSentence(t: TFn, row: StoryboardLipSyncRow): string {
  switch (row.code) {
    case "other-speed":
      return t("board.chip.titleSpeed", { defaultValue: "This shot plays at another speed. Lip sync needs real time." });
    case "locked":
      return t("board.chip.titleLocked", { defaultValue: "This shot is locked, so lip sync leaves it alone." });
    case "outside-song":
      return t("board.chip.titleOutside", { defaultValue: "This shot is outside the song, so there is nothing to sing." });
    case "too-short":
      return t("board.chip.titleShort", { defaultValue: "This shot is too short to lip-sync. It needs at least {min} s.", min: row.minSeconds });
    case "too-long":
      return t("board.chip.titleLong", { defaultValue: "This shot is too long to lip-sync. It can last at most {max} s.", max: row.maxSeconds });
    case "no-singing":
      return t("board.chip.titleNoSinging", { defaultValue: "There is no singing in the vocal for this part of the song." });
    case "not-assigned":
    case null:
      return t("board.chip.titleNoVocal", { defaultValue: "This shot has no vocal yet. Give each shot its vocal to lip-sync it." });
    default:
      // A reason from a newer backend: its own sentence.
      return row.message || t("board.chip.titleNoVocal", { defaultValue: "This shot has no vocal yet. Give each shot its vocal to lip-sync it." });
  }
}

/** "Sings" once the clip is synced, "Will sing" while it waits for its clip, a quiet "No lip sync: reason" otherwise. */
export function syncChip(t: TFn, row: StoryboardLipSyncRow | null | undefined): SyncChip | null {
  if (!row) return null;
  if (row.state === "synced") {
    return {
      text: t("board.chip.sings", { defaultValue: "Sings" }),
      title: t("board.chip.singsTitle", { defaultValue: "This shot's clip is lip-synced to the song." }),
      tone: "sings",
    };
  }
  if (row.state === "waiting") {
    return {
      text: t("board.chip.willSing", { defaultValue: "Will sing" }),
      title: t("board.chip.willSingTitle", { defaultValue: "This shot has its vocal. Make its lip-sync clip to see it sing." }),
      tone: "sings",
    };
  }
  return {
    text: t("board.chip.noLipSync", { defaultValue: "No lip sync: {reason}", reason: reasonShort(t, row.code) }),
    title: `${reasonSentence(t, row)} ${t("board.chip.plainClip", { defaultValue: "It plays as an ordinary clip." })}`,
    tone: "quiet",
  };
}
