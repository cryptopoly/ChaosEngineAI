/**
 * Pure rules for "Use in storyboard": which projects the menu lists, which targets a project offers, the
 * storyboard ops that put a gallery image, clip or sound into a project, and how those ops are applied
 * one by one with a single retry when the project changed underneath them. No React and no network here
 * (the caller passes the function that talks to the server), so all of it is unit tested.
 */
import { StoryboardRevisionConflict } from "../../api/storyboard";
import type { StoryboardOpArgs, StoryboardProject, StoryboardProjectPayload } from "../../types";

/** What the menu is asked to send. `music` and `sfx` are for the Music Library. */
export type SendKind = "image" | "video" | "voice" | "music" | "sfx";

export interface SendItem {
  kind: SendKind;
  /** The file on this machine. Leave it out and give `resolvePath` when the file must be made first. */
  path?: string;
  resolvePath?: () => Promise<string>;
  /** The prompt of an image or clip. */
  prompt?: string;
  /** A name for a sound or a transcript line; used to name a new project when there is no prompt. */
  label?: string;
  /** Length of a clip in seconds. */
  seconds?: number | null;
  clonedVoice?: boolean;
}

export type SendTarget =
  | { type: "new-shot" }
  | { type: "new-take"; shotId: string }
  | { type: "audio" };

/** A project as the menu needs it: a name and the shots it could take a new take for. */
export interface SendProject {
  id: string;
  name: string;
  shots: { id: string; title: string; number: number }[];
}

export type PlannedStep =
  | { op: "add_shot"; args: StoryboardOpArgs["add_shot"] }
  | { op: "import_take"; args: StoryboardOpArgs["import_take"] }
  | { op: "add_audio"; args: StoryboardOpArgs["add_audio"] };

/** Stands for the shot the previous step created; replaced by its real id when the step runs. */
export const NEW_SHOT = "@new-shot";

export const RECENT_PROJECTS = 3;
export const TITLE_LENGTH = 40;
/** The server accepts 0.5 to 60 s for a shot. */
const MIN_SHOT_SECONDS = 0.5;
const MAX_SHOT_SECONDS = 60;

export function isAudioKind(kind: SendKind): boolean {
  return kind === "voice" || kind === "music" || kind === "sfx";
}

/** The menu is not rendered at all unless the Music and Storyboard areas are switched on. */
export function sendMenuVisible(enabled: boolean, item: SendItem | null | undefined): boolean {
  return enabled && item != null && (item.path != null || item.resolvePath != null);
}

/** The project tab to open after sending: Storyboard for pictures and clips, Mixer for sound. */
export function destinationTab(kind: SendKind): "storyboard" | "mixer" {
  return isAudioKind(kind) ? "mixer" : "storyboard";
}

function oneLine(text: string | undefined | null): string {
  return (text ?? "").replace(/\s+/g, " ").trim();
}

/** The first 40 characters of a prompt, on one line; empty when there is none. */
export function shotTitleFrom(prompt: string | undefined | null): string {
  const line = oneLine(prompt);
  return line.slice(0, TITLE_LENGTH).trimEnd();
}

/** The name of a project made from an item; `fallback` is the translated "Untitled storyboard". */
export function projectNameFor(item: SendItem, fallback: string): string {
  return shotTitleFrom(item.prompt) || shotTitleFrom(item.label) || fallback;
}

/** The three projects the menu lists: the open one first, then the newest. Damaged rows are left out. */
export function pickRecentProjects<T extends { id: string; createdAt?: number; error?: string }>(
  rows: readonly T[],
  openId: string | null,
  limit = RECENT_PROJECTS,
): T[] {
  const readable = rows.filter((row) => row.error === undefined);
  const newest = [...readable].sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));
  const open = openId ? newest.find((row) => row.id === openId) : undefined;
  const rest = open ? newest.filter((row) => row.id !== open.id) : newest;
  return (open ? [open, ...rest] : rest).slice(0, limit);
}

export function sendProjectFrom(project: StoryboardProject, _untitled?: (n: number) => string): SendProject {
  return {
    id: project.id,
    name: project.name,
    // An untitled shot keeps an empty title; the menu then names it by its number only.
    shots: project.shots.map((shot, index) => ({ id: shot.id, title: shot.title, number: index + 1 })),
  };
}

/** What a project offers for this item: new shot or a new take of a shot, or just "add the sound". */
export function targetsFor(item: SendItem, project: SendProject): SendTarget[] {
  if (isAudioKind(item.kind)) return [{ type: "audio" }];
  return [{ type: "new-shot" }, ...project.shots.map((shot): SendTarget => ({ type: "new-take", shotId: shot.id }))];
}

/** True when the project step is skipped because there is only one thing to do. */
export function hasSingleTarget(item: SendItem): boolean {
  return isAudioKind(item.kind);
}

function clampSeconds(seconds: number | null | undefined): number | undefined {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return undefined;
  return Math.min(MAX_SHOT_SECONDS, Math.max(MIN_SHOT_SECONDS, Math.round(seconds * 100) / 100));
}

/**
 * The ops that put `item` (its `path` already known) into `project` for `target`. A new shot is two ops:
 * the shot with its prompt and title, then the file as its first take; the second names the shot with
 * `NEW_SHOT` because its id does not exist yet.
 */
export function planSend(item: SendItem, project: Pick<StoryboardProject, "shots">, target: SendTarget): PlannedStep[] {
  const path = item.path;
  if (!path) throw new Error("planSend needs the file path");
  if (isAudioKind(item.kind)) {
    const first = project.shots[0];
    const kind = item.kind as "voice" | "music" | "sfx";
    const args: StoryboardOpArgs["add_audio"] = { kind, path };
    if (item.label) args.label = shotTitleFrom(item.label);
    if (item.clonedVoice) args.clonedVoice = true;
    if (first) args.anchor = { shotId: first.id, offset: 0 };
    else args.start = 0;
    return [{ op: "add_audio", args }];
  }
  const media = item.kind === "video" ? { clip: path } : { still: path };
  if (target.type === "new-take") {
    return [{ op: "import_take", args: { shotId: target.shotId, ...media } }];
  }
  const add: StoryboardOpArgs["add_shot"] = {};
  const title = shotTitleFrom(item.prompt);
  if (title) add.title = title;
  const prompt = oneLine(item.prompt);
  if (prompt) add.stillPrompt = item.prompt!.trim();
  if (item.kind === "video") {
    const seconds = clampSeconds(item.seconds);
    if (seconds !== undefined) add.seconds = seconds;
  }
  return [
    { op: "add_shot", args: add },
    { op: "import_take", args: { shotId: NEW_SHOT, ...media } },
  ];
}

/** The id that appeared when a shot was added: the first in `after` that `before` did not have. */
export function newShotId(before: readonly string[], after: readonly string[]): string | null {
  const known = new Set(before);
  return after.find((id) => !known.has(id)) ?? null;
}

/** Fills in the id of the shot the add_shot step created (found by comparing the shot lists, not by position). */
export function resolveStep(step: PlannedStep, createdShotId: string | null): PlannedStep {
  if (step.op !== "import_take" || step.args.shotId !== NEW_SHOT) return step;
  if (!createdShotId) throw new Error("The new shot is missing");
  return { op: "import_take", args: { ...step.args, shotId: createdShotId } };
}

export type RunStep = (step: PlannedStep, revision: number) => Promise<StoryboardProjectPayload>;

/**
 * Runs the steps in order against `start`, each with the revision of the project as the last answer left
 * it. A step that meets a revision conflict is tried once more with the fresh project the server sent;
 * a second conflict, or any other failure, is thrown. Returns the project after the last step.
 */
export async function applyPlan(
  start: Pick<StoryboardProject, "revision" | "shots">,
  steps: readonly PlannedStep[],
  run: RunStep,
): Promise<Pick<StoryboardProject, "revision" | "shots">> {
  let current: Pick<StoryboardProject, "revision" | "shots"> = start;
  let created: string | null = null;
  for (const step of steps) {
    let retried = false;
    for (;;) {
      const resolved = resolveStep(step, created);
      const before = current.shots.map((shot) => shot.id);
      try {
        current = (await run(resolved, current.revision)).project;
        if (step.op === "add_shot") created = newShotId(before, current.shots.map((shot) => shot.id));
        break;
      } catch (err) {
        if (err instanceof StoryboardRevisionConflict && !retried) {
          retried = true;
          current = err.project;
          continue;
        }
        throw err;
      }
    }
  }
  return current;
}

// ---- keyboard model ----
export type MenuKeyAction = { type: "focus"; index: number } | { type: "close" } | null;

/** What a key does inside the open menu: arrows and Home / End move, Escape and Tab close. */
export function menuKeyAction(key: string, index: number, count: number): MenuKeyAction {
  if (count <= 0) return key === "Escape" || key === "Tab" ? { type: "close" } : null;
  switch (key) {
    case "ArrowDown":
      return { type: "focus", index: index < 0 ? 0 : (index + 1) % count };
    case "ArrowUp":
      return { type: "focus", index: index <= 0 ? count - 1 : index - 1 };
    case "Home":
      return { type: "focus", index: 0 };
    case "End":
      return { type: "focus", index: count - 1 };
    case "Escape":
    case "Tab":
      return { type: "close" };
    default:
      return null;
  }
}

/** The server calls a send needs; the hook passes the real ones and the tests fakes. */
export interface SendApi {
  create: (name: string) => Promise<StoryboardProjectPayload>;
  get: (id: string) => Promise<StoryboardProjectPayload>;
  run: (projectId: string, step: PlannedStep, revision: number) => Promise<StoryboardProjectPayload>;
  remove: (projectId: string, revision: number) => Promise<unknown>;
}

/**
 * Puts `item` (its `path` known) into a project: the one given, or a new one named from the item. If the
 * send fails after making a new project, that project is deleted again so a failed send (or each retry)
 * leaves nothing behind. Returns the project's id.
 */
export async function sendIntoProject(
  item: SendItem,
  projectId: string | null,
  target: SendTarget,
  untitledName: string,
  api: SendApi,
): Promise<string> {
  const madeHere = projectId === null;
  const fresh = (await (projectId ? api.get(projectId) : api.create(projectNameFor(item, untitledName)))).project;
  try {
    await applyPlan(fresh, planSend(item, fresh, target), (step, revision) => api.run(fresh.id, step, revision));
  } catch (err) {
    if (madeHere) {
      try {
        // Its revision moved with the steps that did land, so read it again.
        await api.remove(fresh.id, (await api.get(fresh.id)).project.revision);
      } catch {
        // the original failure is the one to report
      }
    }
    throw err;
  }
  return fresh.id;
}
