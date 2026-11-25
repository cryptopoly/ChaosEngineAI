/**
 * Character lock (Storyboard UX audit 2026-11): a character marked "In every shot" appears in every shot unless
 * that shot leaves it out. Pure helpers for the Cast card, the shot editor and applying a plan proposal.
 */
import type { StoryboardPlannedShot, StoryboardProject, StoryboardReference, StoryboardShot } from "../../types";
import { CAST_OUT, IN_EVERY_SHOT, PLANNED_CHARACTERS } from "../../types/storyboard";

/** A shot uses at most this many references (the backend's limit). */
const MAX_REFS = 8;

export function isInEveryShot(reference: StoryboardReference): boolean {
  return reference[IN_EVERY_SHOT] === true;
}

export function castOutOf(shot: Pick<StoryboardShot, typeof CAST_OUT>): string[] {
  return shot[CAST_OUT] ?? [];
}

/** A new character starts in every shot when the project has a song and has no character yet. */
export function startsInEveryShot(project: StoryboardProject): boolean {
  return Boolean(project.song) && project.references.characters.length === 0;
}

/** The characters in a shot: those ticked plus the in-every-shot ones, minus those the shot leaves out. */
export function castInShot(
  characters: StoryboardReference[],
  shot: { referenceImages: string[]; leftOut: string[] },
): StoryboardReference[] {
  return characters.filter(
    (c) => !shot.leftOut.includes(c.id) && (isInEveryShot(c) || shot.referenceImages.includes(c.id)),
  );
}

/** Leave a character out of the shot, or put it back. */
export function toggleCastOut(ids: string[], refId: string): string[] {
  return ids.includes(refId) ? ids.filter((id) => id !== refId) : [...ids, refId];
}

/** A proposed shot as `apply_plan` takes it: its characters (ones this project has) become its references. */
export function plannedShotFromProposal(shot: StoryboardPlannedShot, project: StoryboardProject): StoryboardPlannedShot {
  const known = new Set([...project.references.characters, ...project.references.places].map((r) => r.id));
  const refs = [...(shot.referenceImages ?? []), ...(shot[PLANNED_CHARACTERS] ?? [])].filter((id) => known.has(id));
  const planned: StoryboardPlannedShot = {
    title: shot.title,
    stillPrompt: shot.stillPrompt,
    motionPrompt: shot.motionPrompt,
    seconds: shot.seconds,
  };
  const unique = [...new Set(refs)].slice(0, MAX_REFS);
  if (unique.length > 0) planned.referenceImages = unique;
  const characterIds = new Set(project.references.characters.map((r) => r.id));
  const out = [...new Set((shot[CAST_OUT] ?? []).filter((id) => characterIds.has(id)))];
  if (out.length > 0) planned[CAST_OUT] = out;
  return planned;
}
