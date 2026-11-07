/**
 * "Use in storyboard" for one screen: the three most recent projects for the menu (cached for the session),
 * the send itself through the storyboard ops, and the move to the Board or the Mixer afterwards.
 *
 * A send reads the project again first, applies its ops with that revision (one retry when the project
 * changed meanwhile, see `applyPlan`), then opens the project. The Board and the Mixer keep their selection
 * to themselves, so the new shot or clip is not pre-selected: the screen opens on the project.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { createStoryboardProject, deleteStoryboardProject, getStoryboardProject, listStoryboardProjects, runStoryboardOp } from "../../api";
import { HOME_STUDIO_ENABLED } from "../../constants/features";
import type { TabId } from "../../types";
import {
  destinationTab,
  pickRecentProjects,
  sendIntoProject,
  sendProjectFrom,
  type PlannedStep,
  type SendItem,
  type SendProject,
  type SendTarget,
} from "./sendToStoryboardLogic";
import { readOpenProjectId, writeOpenProjectId } from "./storyboardLogic";

/** How long the cached project list is trusted when a menu opens. Sending drops it at once. */
export const PROJECT_CACHE_MS = 60_000;

let cache: { at: number; projects: SendProject[] } | null = null;

export function clearSendProjectCache(): void {
  cache = null;
}

/** Runs one planned step; each branch keeps the op name and its arguments matched. */
function runPlannedStep(projectId: string, revision: number, step: PlannedStep) {
  switch (step.op) {
    case "add_shot":
      return runStoryboardOp(projectId, revision, "add_shot", step.args);
    case "import_take":
      return runStoryboardOp(projectId, revision, "import_take", step.args);
    case "add_audio":
      return runStoryboardOp(projectId, revision, "add_audio", step.args);
  }
}

export interface UseSendToStoryboard {
  enabled: boolean;
  projects: SendProject[];
  /** True once the list has been read, so "no projects" is known and not just not loaded yet. */
  loaded: boolean;
  busy: boolean;
  /** A plain sentence when the last send failed; null otherwise. */
  notice: string | null;
  dismissNotice: () => void;
  loadProjects: () => Promise<void>;
  /** `projectId` null makes a new project named from the item. */
  send: (item: SendItem, projectId: string | null, target: SendTarget) => Promise<boolean>;
}

export function useSendToStoryboard(
  onNavigate: (tab: TabId) => void,
  enabled: boolean = HOME_STUDIO_ENABLED,
): UseSendToStoryboard {
  const { t } = useTranslation("studio");
  const [projects, setProjects] = useState<SendProject[]>(() => cache?.projects ?? []);
  const [loaded, setLoaded] = useState(() => cache !== null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true; // set again on a remount (React strict mode runs the cleanup first)
    return () => {
      alive.current = false;
    };
  }, []);

  const untitled = useCallback((n: number) => t("board.shot.untitled", { defaultValue: "Shot {n}", n }), [t]);

  const loadProjects = useCallback(async () => {
    if (!enabled) return;
    if (cache && Date.now() - cache.at < PROJECT_CACHE_MS) {
      setProjects(cache.projects);
      setLoaded(true);
      return;
    }
    try {
      const rows = await listStoryboardProjects();
      const picked = pickRecentProjects(rows, readOpenProjectId());
      const loadedProjects = await Promise.all(
        picked.map(async (row) => sendProjectFrom((await getStoryboardProject(row.id)).project, untitled)),
      );
      cache = { at: Date.now(), projects: loadedProjects };
      if (alive.current) {
        setProjects(loadedProjects);
        setLoaded(true);
      }
    } catch {
      // The menu then offers a new project only; sending reports a real failure.
      if (alive.current) setLoaded(true);
    }
  }, [enabled, untitled]);

  useEffect(() => {
    if (enabled) void loadProjects();
  }, [enabled, loadProjects]);

  const send = useCallback(
    async (item: SendItem, projectId: string | null, target: SendTarget): Promise<boolean> => {
      setBusy(true);
      setNotice(null);
      try {
        const path = item.path ?? (await item.resolvePath?.());
        if (!path) throw new Error(t("sendToStoryboard.noFile", { defaultValue: "This item has no file to send." }));
        const ready = { ...item, path };
        const openedId = await sendIntoProject(ready, projectId, target, t("sendToStoryboard.untitled", { defaultValue: "Untitled storyboard" }), {
          create: (name) => createStoryboardProject(name),
          get: getStoryboardProject,
          run: (id, step, revision) => runPlannedStep(id, revision, step),
          remove: deleteStoryboardProject,
        });
        cache = null;
        writeOpenProjectId(openedId);
        onNavigate(destinationTab(item.kind));
        return true;
      } catch (err) {
        const detail = err instanceof Error ? err.message : String(err);
        const conflict = err instanceof Error && err.name === "StoryboardRevisionConflict";
        if (alive.current) {
          setNotice(
            conflict
              ? t("sendToStoryboard.conflict", { defaultValue: "The project changed while sending. Try again." })
              : t("sendToStoryboard.failed", { defaultValue: "Could not add this to the storyboard: {detail}", detail }),
          );
        }
        cache = null;
        return false;
      } finally {
        if (alive.current) setBusy(false);
      }
    },
    [onNavigate, t],
  );

  const dismissNotice = useCallback(() => setNotice(null), []);

  return { enabled, projects, loaded, busy, notice, dismissNotice, loadProjects, send };
}
