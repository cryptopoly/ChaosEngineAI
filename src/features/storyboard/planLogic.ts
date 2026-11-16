/**
 * Pure rules for the Plan tab and the style and brand kit (S8, S15): the length choices, how a proposal's
 * length compares with the aim, what the direction chat's changes say in words, which fields a reworked shot
 * changes, where a brief picture goes, and the brand kit with its defaults filled in for older projects.
 * No React here so the rules are unit-tested directly.
 */
import type {
  StoryboardBrand,
  StoryboardPlan,
  StoryboardPlanChanges,
  StoryboardPlannedShot,
  StoryboardProject,
  StoryboardReference,
  StoryboardShot,
} from "../../types";
import { MAX_PICTURES } from "./pickerLogic";

/** Film lengths offered next to the brief, in seconds. */
export const LENGTH_CHOICES = [15, 30, 60, 90, 120] as const;
/** Shot counts the model may be asked for; null lets it choose between 4 and 8. */
export const SHOT_COUNT_CHOICES = [4, 5, 6, 7, 8] as const;
/** Style references a project keeps (mirrors References.styles in models.py). */
export const MAX_STYLES = 20;
/** Colours kept per style (mirrors MAX_PALETTE in models.py). */
export const MAX_PALETTE = 8;

export const DEFAULT_BRAND: StoryboardBrand = {
  logo: null,
  logoCorner: "bottom-right",
  captions: { font: null, colour: "#ffffff", position: "bottom" },
  styleUse: "auto",
};

export const DEFAULT_PLAN: StoryboardPlan = { brief: "", filmSeconds: null };

export function planOf(project: Pick<StoryboardProject, "plan">): StoryboardPlan {
  return { ...DEFAULT_PLAN, ...(project.plan ?? {}) };
}

export function brandOf(project: Pick<StoryboardProject, "brand">): StoryboardBrand {
  const brand = project.brand;
  if (!brand) return DEFAULT_BRAND;
  return { ...DEFAULT_BRAND, ...brand, captions: { ...DEFAULT_BRAND.captions, ...(brand.captions ?? {}) } };
}

export function stylesOf(project: Pick<StoryboardProject, "references">): StoryboardReference[] {
  return project.references.styles ?? [];
}

/** Every colour of every style, once each, in order. */
export function projectPalette(project: Pick<StoryboardProject, "references">, limit = 12): string[] {
  const out: string[] = [];
  for (const ref of stylesOf(project)) for (const colour of ref.palette ?? []) if (!out.includes(colour)) out.push(colour);
  return out.slice(0, limit);
}

/** The sum of the shots' seconds, to one decimal. */
export function totalSeconds(shots: readonly Pick<StoryboardPlannedShot, "seconds">[]): number {
  return Math.round(shots.reduce((sum, shot) => sum + (Number.isFinite(shot.seconds) ? shot.seconds : 0), 0) * 10) / 10;
}

export type LengthVerdict = "match" | "short" | "long" | "none";

/** Within 15 % of the aim is a match; no aim gives "none". */
export function lengthVerdict(total: number, target: number | null | undefined): LengthVerdict {
  if (!target) return "none";
  if (total < target * 0.85) return "short";
  if (total > target * 1.15) return "long";
  return "match";
}

/** "1 min", "1 min 30 s", "45 s". */
export function lengthLabel(seconds: number): { minutes: number; seconds: number } {
  const whole = Math.round(seconds);
  return { minutes: Math.floor(whole / 60), seconds: whole % 60 };
}

export function changeCount(changes: StoryboardPlanChanges | null | undefined): number {
  if (!changes) return 0;
  return changes.edits.length + changes.add.length + changes.remove.length;
}

export type ChangeRow =
  | { kind: "edit"; number: number; title: string; fields: string[] }
  | { kind: "add"; after: number; title: string }
  | { kind: "remove"; number: number; title: string };

/** The direction chat's changes as rows the user can read before applying them. Unknown shots are dropped. */
export function changeRows(changes: StoryboardPlanChanges, shots: readonly Pick<StoryboardShot, "id" | "title">[]): ChangeRow[] {
  const numberOf = (id: string | null) => (id === null ? 0 : shots.findIndex((s) => s.id === id) + 1);
  const rows: ChangeRow[] = [];
  for (const edit of changes.edits) {
    const number = numberOf(edit.shotId);
    if (number <= 0) continue;
    const fields = (["title", "stillPrompt", "motionPrompt", "seconds"] as const).filter((key) => edit[key] !== undefined);
    rows.push({ kind: "edit", number, title: shots[number - 1].title, fields });
  }
  for (const id of changes.remove) {
    const number = numberOf(id);
    if (number > 0) rows.push({ kind: "remove", number, title: shots[number - 1].title });
  }
  for (const added of changes.add) {
    const after = numberOf(added.after);
    if (after >= 0) rows.push({ kind: "add", after, title: added.title });
  }
  return rows;
}

export type ShotField = "title" | "stillPrompt" | "motionPrompt" | "seconds";

/** The fields a reworked shot changes, compared with the shot as it is. */
export function changedFields(shot: Pick<StoryboardShot, ShotField>, suggestion: Pick<StoryboardPlannedShot, ShotField>): ShotField[] {
  return (["title", "stillPrompt", "motionPrompt", "seconds"] as const).filter((key) => {
    if (key === "seconds") return Math.abs(shot.seconds - suggestion.seconds) > 0.01;
    return (shot[key] ?? "").trim() !== (suggestion[key] ?? "").trim() && (suggestion[key] ?? "").trim() !== "";
  });
}

/** `edit_shot` arguments for a reworked shot: only what changed. */
export function reworkEdit(shot: Pick<StoryboardShot, "id" | ShotField>, suggestion: Pick<StoryboardPlannedShot, ShotField>) {
  const args: { shotId: string; title?: string; stillPrompt?: string; motionPrompt?: string; seconds?: number } = { shotId: shot.id };
  for (const key of changedFields(shot, suggestion)) {
    if (key === "seconds") args.seconds = suggestion.seconds;
    else args[key] = suggestion[key];
  }
  return args;
}

/** The style a brief picture joins: the first with room, or null when a new style is needed. */
export function styleWithRoom(project: Pick<StoryboardProject, "references">): StoryboardReference | null {
  return stylesOf(project).find((ref) => ref.images.length < MAX_PICTURES) ?? null;
}

/** True when an apply would replace shots that already have pictures or clips. */
export function replaceLosesTakes(project: Pick<StoryboardProject, "shots">): number {
  return project.shots.filter((shot) => shot.takes.length > 0).length;
}

export function lockedCount(project: Pick<StoryboardProject, "shots">): number {
  return project.shots.filter((shot) => shot.locked).length;
}

/** `#rrggbb` only (what the backend takes). */
export function isHexColour(value: string): boolean {
  return /^#[0-9a-f]{6}$/.test(value);
}

/** Black or white text, whichever reads better on the colour (for swatch labels and the caption preview). */
export function inkOn(hex: string): "#000000" | "#ffffff" {
  if (!isHexColour(hex)) return "#ffffff";
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const luminance = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return luminance > 0.4 ? "#000000" : "#ffffff";
}

/** What the backend's error code means for the Plan tab. */
export type PlanFailure = "noModel" | "busy" | "other";

export function planFailure(code: string | null | undefined): PlanFailure {
  if (code === "no-model") return "noModel";
  if (code === "busy") return "busy";
  return "other";
}

/** The style word on an active take (`settings.style`), when the batch recorded one. */
export function takeStyle(shot: Pick<StoryboardShot, "takes" | "activeTake">): "pictures" | "words" | null {
  const take = shot.takes.find((t) => t.id === shot.activeTake);
  const value = take?.settings?.style;
  return value === "pictures" || value === "words" ? value : null;
}
