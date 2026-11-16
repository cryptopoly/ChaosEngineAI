/**
 * Pure helpers for the song panel and the board's song ruler (S-M2): clock times, the ruler's parts as shares of
 * the song, a section relabelled for `edit_song`, lyrics text to lines and back, and a cut plan grouped by section
 * for review.
 */
import type { StoryboardCutPlan, StoryboardPlannedShot, StoryboardSongSection } from "../../../types";

/** "3:05" for 185 s; "0:04" for 4.2 s. */
export function clock(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export interface RulerPart {
  index: number;
  label: string;
  start: number;
  end: number;
  /** Left edge and width as a percentage of the song. */
  left: number;
  width: number;
  /** 0 to 1, or null when the analysis gave none. */
  energy: number | null;
}

/** The sections as ruler parts, clipped to the song and in order. */
export function rulerParts(sections: StoryboardSongSection[], seconds: number): RulerPart[] {
  if (!(seconds > 0)) return [];
  return sections
    .map((section, index) => ({ section, index }))
    .filter(({ section }) => section.end > section.start && section.start < seconds)
    .sort((a, b) => a.section.start - b.section.start)
    .map(({ section, index }) => {
      const start = Math.max(0, section.start);
      const end = Math.min(seconds, section.end);
      return {
        index,
        label: section.label,
        start,
        end,
        left: (start / seconds) * 100,
        width: ((end - start) / seconds) * 100,
        energy: typeof section.energy === "number" ? section.energy : null,
      };
    });
}

/** The sections with one label changed (trimmed, at most 40 characters), or null when nothing changes. */
export function relabel(sections: StoryboardSongSection[], index: number, label: string): StoryboardSongSection[] | null {
  const next = label.replace(/\s+/g, " ").trim().slice(0, 40);
  const current = sections[index];
  if (!current || !next || next === current.label) return null;
  return sections.map((section, i) => (i === index ? { ...section, label: next } : section));
}

/** Lyrics typed in the box as caption lines: one per line, blank lines dropped, at most 200 characters a line. */
export function lyricsToLines(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.replace(/\s+/g, " ").trim().slice(0, 200))
    .filter(Boolean)
    .slice(0, 500);
}

export function sameLines(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((line, i) => line === b[i]);
}

export interface CutPlanGroup {
  label: string;
  start: number;
  end: number;
  shots: { number: number; shot: StoryboardPlannedShot }[];
}

/** The proposed shots under the section they fall in (by the section's shot count, in order). */
export function cutPlanGroups(plan: StoryboardCutPlan): CutPlanGroup[] {
  const groups: CutPlanGroup[] = [];
  let next = 0;
  for (const section of plan.sections) {
    const count = Math.max(0, section.shots);
    const shots = plan.shots.slice(next, next + count).map((shot, i) => ({ number: next + i + 1, shot }));
    next += count;
    groups.push({ label: section.label, start: section.start, end: section.end, shots });
  }
  if (next < plan.shots.length) {
    const rest = plan.shots.slice(next).map((shot, i) => ({ number: next + i + 1, shot }));
    const last = groups[groups.length - 1];
    if (last) last.shots.push(...rest);
    else groups.push({ label: "", start: 0, end: plan.totalSeconds, shots: rest });
  }
  return groups;
}

/** How many shots with takes a replace would take off the board (Undo brings them back). */
export function shotsWithTakes(shots: { takes: unknown[] }[]): number {
  return shots.filter((shot) => shot.takes.length > 0).length;
}
