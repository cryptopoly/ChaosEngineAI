/**
 * "Ready to export?" on the Export tab (audit change 6). The backend's `guide.readiness` has three groups of items
 * (`stops`, `worthFixing`, `forInfo`), each with a plain sentence, the shots it is about and an action code. This file
 * turns them into what the tab shows: the three groups, one fix button per item mapped from the action code to a tab
 * (or one shot's editor), and a few facts only the app knows (loudness, captions, the AI tag, the preview) as extra
 * "for your information" lines. The old preflight in `overviewLogic.ts` is not used here.
 */
import type {
  StoryboardGuideAction,
  StoryboardProject,
  StoryboardReadiness,
  StoryboardReadinessItem,
} from "../../types";
import type { ProjectTabId } from "./projectRoute";

export type ReadinessGroupId = "stops" | "worthFixing" | "forInfo";

/** The three groups in the order they are shown. */
export const READINESS_GROUPS: ReadinessGroupId[] = ["stops", "worthFixing", "forInfo"];

/** Which button an action gives: where it goes and what the button says (the words come from the tab). */
export type FixKind = "plan" | "song" | "cast" | "storyboard" | "shot" | "mixer";

export interface ReadinessFix {
  kind: FixKind;
  tab: ProjectTabId;
  /** Set for `kind: "shot"` when the item is about exactly one shot: the editor of that shot opens. */
  shotId: string | null;
}

/** Action codes that open the Storyboard (where shots are made, remade, chosen for final or retried). */
const STORYBOARD_ACTIONS: StoryboardGuideAction[] = [
  "open-storyboard",
  "watch-batch",
  "resume-batch",
  "retry-failed",
  "make-drafts",
  "make-missing",
  "make-final",
  "choose-final",
  "make-clips",
  "remake-stale",
];
/** Lip-sync fixes that change one shot (its length, its lock, its speed). */
const SHOT_ACTIONS: StoryboardGuideAction[] = ["real-speed", "unlock", "lengthen", "shorten"];
/** The song step in Plan: attach, cut, give each shot its vocal, make the lip-sync clips, read the coverage. */
const SONG_ACTIONS: StoryboardGuideAction[] = ["attach-song", "cut-plan", "assign-vocals", "make-lipsync", "open-coverage"];

/**
 * The button for an action code, or null when there is none to press (no action, or the action is "check" or
 * "export", which is this tab). `shotIds` is what the item is about: a single shot opens its own editor.
 */
export function fixForAction(action: StoryboardGuideAction | null, shotIds: string[] = []): ReadinessFix | null {
  if (!action) return null;
  if (action === "plan-shots") return { kind: "plan", tab: "plan", shotId: null };
  if (SONG_ACTIONS.includes(action)) return { kind: "song", tab: "plan", shotId: null };
  if (action === "open-cast") return { kind: "cast", tab: "cast", shotId: null };
  if (action === "open-mixer") return { kind: "mixer", tab: "mixer", shotId: null };
  if (SHOT_ACTIONS.includes(action)) return { kind: "shot", tab: "storyboard", shotId: shotIds.length === 1 ? shotIds[0] : null };
  if (STORYBOARD_ACTIONS.includes(action)) return { kind: "storyboard", tab: "storyboard", shotId: null };
  return null;
}

/** One line of a group: the backend's sentence and its fix, or a fact the app adds. */
export type ReadinessLine =
  | { source: "guide"; code: StoryboardReadinessItem["code"]; message: string; shotIds: string[]; fix: ReadinessFix | null }
  | { source: "local"; local: LocalLine };

/** Facts only the app knows, shown under "For your information". */
export type LocalLine =
  | { id: "upscale"; size: string; method: "fast" | "lanczos" }
  | { id: "loudness"; lufs: number | null }
  | { id: "captions"; count: number; how: "srt+burn" | "srt" | "burn" | "none" }
  | { id: "aiTag"; on: boolean; cloned: boolean }
  | { id: "preview" };

export interface LocalFacts {
  /** The upscaled size ("3840 × 2160") when the export scales the film up, else null. */
  upscaledTo: string | null;
  method: "fast" | "lanczos";
  previewStale: boolean;
  clonedVoice: boolean;
  /** Captions inside the film (`filmCaptionCount`). */
  captions: number;
}

export function localLines(project: Pick<StoryboardProject, "export">, facts: LocalFacts): LocalLine[] {
  const lines: LocalLine[] = [];
  if (facts.upscaledTo) lines.push({ id: "upscale", size: facts.upscaledTo, method: facts.method });
  const lufs = project.export?.loudnessLufs;
  lines.push({ id: "loudness", lufs: lufs ?? null });
  const srt = Boolean(project.export?.captionsSrt);
  const burn = Boolean(project.export?.captionsBurn);
  lines.push({ id: "captions", count: facts.captions, how: srt && burn ? "srt+burn" : srt ? "srt" : burn ? "burn" : "none" });
  lines.push({ id: "aiTag", on: facts.clonedVoice || Boolean(project.export?.aiLabel?.embedTag), cloned: facts.clonedVoice });
  if (facts.previewStale) lines.push({ id: "preview" });
  return lines;
}

export interface ReadinessGroup {
  id: ReadinessGroupId;
  lines: ReadinessLine[];
}

/**
 * The groups to show. Empty groups are kept (the tab says "none" for Stops the export and Worth fixing) except that
 * "For your information" always has the local lines. `readiness` null (an older payload without the guide) gives the
 * local lines only.
 */
export function readinessGroups(readiness: StoryboardReadiness | null, local: LocalLine[]): ReadinessGroup[] {
  const fromGuide = (items: StoryboardReadinessItem[]): ReadinessLine[] =>
    items.map((item) => ({
      source: "guide",
      code: item.code,
      message: item.message,
      shotIds: item.shotIds,
      fix: fixForAction(item.action, item.shotIds),
    }));
  return [
    { id: "stops", lines: readiness ? fromGuide(readiness.stops) : [] },
    { id: "worthFixing", lines: readiness ? fromGuide(readiness.worthFixing) : [] },
    {
      id: "forInfo",
      lines: [...(readiness ? fromGuide(readiness.forInfo) : []), ...local.map((line): ReadinessLine => ({ source: "local", local: line }))],
    },
  ];
}

/** The Export button is off only when the backend says something stops the export. Without the guide: when no shot has a clip. */
export function exportAllowed(readiness: StoryboardReadiness | null, hasClips: boolean): boolean {
  return readiness ? readiness.ready : hasClips;
}

/** With things worth fixing the button says "Export anyway"; the choice stays the user's. */
export function exportAnyway(readiness: StoryboardReadiness | null): boolean {
  return Boolean(readiness && readiness.ready && readiness.worthFixing.length > 0);
}

/** The first thing that stops the export, in the backend's words, for the line under a disabled button. */
export function stopReason(readiness: StoryboardReadiness | null): string | null {
  return readiness && !readiness.ready ? (readiness.stops[0]?.message ?? null) : null;
}
