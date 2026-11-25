/**
 * Take compare (S10) and redo one half of a take (S11): pure helpers. A take's settings hold what made it
 * (`stillModel`, `stillSeed`, `clipModel`, `clipSeed`, size, steps); takes made before S11 hold only one `seed`,
 * which belongs to the clip when the take has one (the clip ran last) and to the still otherwise.
 */
import type { StoryboardShot, StoryboardTake } from "../../types";
import { shotSeconds } from "./planLogic";

/** The active take's clip came from another still than the one the take holds now (after Redo still only). */
export function clipOutdated(shot: StoryboardShot): boolean {
  const take = shot.takes.find((item) => item.id === shot.activeTake);
  if (!take?.clip) return false;
  const madeFrom = take.settings?.clipStill;
  const still = take.hashes.still;
  return typeof madeFrom === "string" && Boolean(still) && madeFrom !== still;
}

/** Redo clip only: a take to keep the still of. A shot that starts from a new still needs that still first. */
export function canRedoClip(shot: StoryboardShot): boolean {
  const take = shot.takes.find((item) => item.id === shot.activeTake);
  if (!take || shot.startFrom === "upload" || shot.status === "missing") return false;
  return shot.startFrom !== "generate" || Boolean(take.still);
}

/** Redo still only: only a shot that starts from a still it makes itself. */
export function canRedoStill(shot: StoryboardShot): boolean {
  const take = shot.takes.find((item) => item.id === shot.activeTake);
  return Boolean(take) && shot.startFrom === "generate" && shot.status !== "missing";
}

// No "steps": a take holds one steps figure for whichever half ran last, so it would compare a still with a clip.
export type CompareKey = "pass" | "stillModel" | "stillSeed" | "clipModel" | "clipSeed" | "size" | "length";
export const COMPARE_KEYS: CompareKey[] = ["pass", "stillModel", "stillSeed", "clipModel", "clipSeed", "size", "length"];

export type TakeFacts = Record<CompareKey, string | null>;

function text(value: unknown): string | null {
  if (value === null || value === undefined || value === "") return null;
  return String(value);
}

/** What made a take, as short strings (null when the take does not record it). */
export function takeFacts(take: StoryboardTake): TakeFacts {
  const s = take.settings ?? {};
  const seed = text(s.seed);
  const width = s.width ?? null;
  const height = s.height ?? null;
  return {
    pass: take.renderPass === "draft" ? "draft" : "final",
    stillModel: take.still ? text(s.stillModel) ?? (take.clip ? null : text(s.model)) : null,
    stillSeed: take.still ? text(s.stillSeed) ?? (take.clip ? null : seed) : null,
    clipModel: take.clip ? text(s.clipModel) ?? text(s.model) : null,
    clipSeed: take.clip ? text(s.clipSeed) ?? seed : null,
    size: width && height ? `${width} x ${height}` : null,
    length: take.measuredSeconds ? shotSeconds(take.measuredSeconds) : null,
  };
}

export interface CompareRow {
  key: CompareKey;
  a: string | null;
  b: string | null;
  differs: boolean;
}

/** One row per fact either take records; `differs` when both record it and the values are not the same. */
export function compareRows(a: StoryboardTake, b: StoryboardTake): CompareRow[] {
  const fa = takeFacts(a);
  const fb = takeFacts(b);
  return COMPARE_KEYS.filter((key) => fa[key] !== null || fb[key] !== null).map((key) => ({
    key,
    a: fa[key],
    b: fb[key],
    differs: fa[key] !== fb[key],
  }));
}

export type CompareSummary = "sameStill" | "sameClip" | "identical" | "different";

/** How two takes relate: they share a still (a redo of the clip), share a clip (a redo of the still), or neither. */
export function compareSummary(a: StoryboardTake, b: StoryboardTake): CompareSummary {
  const sameStill = Boolean(a.hashes.still) && a.hashes.still === b.hashes.still;
  const sameClip = Boolean(a.hashes.clip) && a.hashes.clip === b.hashes.clip;
  if (sameStill && sameClip) return "identical";
  if (sameStill) return "sameStill";
  if (sameClip) return "sameClip";
  return "different";
}

/** The two takes the compare opens on: the active one and the newest other one (or the two newest). */
export function defaultComparePair(shot: StoryboardShot): [string, string] | null {
  if (shot.takes.length < 2) return null;
  const newest = [...shot.takes].sort((x, y) => y.createdAt - x.createdAt);
  const active = shot.takes.find((take) => take.id === shot.activeTake) ?? newest[0];
  const other = newest.find((take) => take.id !== active.id) as StoryboardTake;
  return [active.id, other.id];
}

/** What the compare shows: clips when both takes have one, else stills when both have one, else what there is. */
export function compareView(a: StoryboardTake, b: StoryboardTake): "clip" | "still" {
  if (a.clip && b.clip) return "clip";
  if (a.still && b.still) return "still";
  return a.clip || b.clip ? "clip" : "still";
}

/** Where the follower should be to stay with the leader, or null when it is close enough (drift under `slack`). */
export function syncTime(leader: number, follower: number, followerDuration: number, slack = 0.12): number | null {
  const target = Number.isFinite(followerDuration) && followerDuration > 0 ? Math.min(leader, followerDuration) : leader;
  return Math.abs(target - follower) > slack ? target : null;
}
