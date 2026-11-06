/** Storyboard Board: write, generate and arrange the shots (UX spec D.5). */
import { useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { PlusIcon, RedoIcon, UndoIcon } from "../../components/icons/ActionIcons";
import { Panel } from "../../components/Panel";
import { RunEstimateStrip } from "../../components/RunEstimateStrip";
import { StudioShell } from "../../components/StudioShell";
import type { ImageOutputArtifact, StoryboardBatchRecord, StoryboardShot, VideoOutputArtifact } from "../../types";
import { BatchBar } from "./BatchBar";
import { ConfirmDialog } from "./ConfirmDialog";
import { ProjectDefaultsPanel, type ModelChoice } from "./ProjectDefaultsPanel";
import { ReferencesPanel } from "./ReferencesPanel";
import { ShotCard } from "./ShotCard";
import { ShotEditorDrawer } from "./ShotEditorDrawer";
import {
  effectiveShotStatus,
  filmSeconds,
  isBatchRunning,
  liveShotStates,
  missingShotIds,
  reorderByDrop,
  reorderByStep,
  staleAfterShot,
  staleCount,
  unlockedShotIds,
} from "./storyboardLogic";
import type { UseStoryboardResult } from "./useStoryboard";

export interface BoardTabProps {
  backendOnline: boolean;
  story: UseStoryboardResult;
  imageGallery: ImageOutputArtifact[];
  videoGallery: VideoOutputArtifact[];
  imageModels: ModelChoice[];
  videoModels: ModelChoice[];
  onOpenProjects: () => void;
  onOpenMixer: () => void;
}

/** The error text the last batch recorded for a shot, if that batch failed it. */
export function lastFailure(batches: StoryboardBatchRecord[], shotId: string): string | null {
  const last = batches[batches.length - 1];
  const failed = last?.results.find((r) => r.shotId === shotId && r.state === "failed");
  return failed ? (failed.error ?? "") : null;
}

export function BoardTab({
  backendOnline, story, imageGallery, videoGallery, imageModels, videoModels, onOpenProjects, onOpenMixer,
}: BoardTabProps) {
  const { t } = useTranslation("studio");
  const { project, timeline, batch, busy } = story;
  const [editingId, setEditingId] = useState<string | null>(null);
  const [removeId, setRemoveId] = useState<string | null>(null);
  const [kept, setKept] = useState<Set<string>>(() => new Set());
  const [dragId, setDragId] = useState<string | null>(null);
  const [overId, setOverId] = useState<string | null>(null);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [announce, setAnnounce] = useState("");
  const list = useRef<HTMLDivElement>(null);

  const live = useMemo(() => liveShotStates(batch), [batch]);
  const queuedOrder = useMemo(() => (project ? project.shots.filter((s) => live.get(s.id) === "queued").map((s) => s.id) : []), [project, live]);
  const running = isBatchRunning(batch);

  if (!story.projectId || story.notFound) {
    return (
      <div className="content-grid storyboard-page">
        <Panel title={t("board.noProjectTitle", { defaultValue: "Board" })} className="span-2">
          <div className="empty-state">
            <p className="muted-text">
              {story.notFound
                ? t("board.notFound", { defaultValue: "This project is not there any more. Pick another one." })
                : t("board.noProject", { defaultValue: "No project is open. Open one from Projects, or start a new one." })}
            </p>
            <button type="button" className="primary-button" onClick={onOpenProjects}>
              {t("board.openProjects", { defaultValue: "Open Projects" })}
            </button>
          </div>
        </Panel>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="content-grid storyboard-page">
        <Panel title={t("board.noProjectTitle", { defaultValue: "Board" })} className="span-2">
          <div className="empty-state">
            <p className="muted-text">
              {!backendOnline
                ? t("storyboard.offline", { defaultValue: "Waiting for the backend…" })
                : story.error ?? t("board.loading", { defaultValue: "Loading the project…" })}
            </p>
            {story.error ? (
              <button type="button" className="secondary-button" onClick={() => void story.reload()}>
                {t("board.reload", { defaultValue: "Reload" })}
              </button>
            ) : null}
          </div>
        </Panel>
      </div>
    );
  }

  const shots = project.shots;
  const ids = shots.map((s) => s.id);
  const editing = shots.find((s) => s.id === editingId) ?? null;
  const toRemove = shots.find((s) => s.id === removeId) ?? null;
  const missing = missingShotIds(shots);
  const unlocked = unlockedShotIds(shots);
  const stale = staleCount(project);
  const tabbableId = focusId && ids.includes(focusId) ? focusId : ids[0];

  function focusCard(id: string | undefined) {
    if (!id) return;
    list.current?.querySelector<HTMLElement>(`[data-shot-id="${id}"]`)?.focus();
  }

  async function reorderTo(order: string[] | null, shotId: string) {
    if (!order) return;
    const ok = await story.op("reorder", { order });
    if (ok) {
      setAnnounce(t("board.shot.moved", { defaultValue: "Shot {n} moved to position {position}", n: ids.indexOf(shotId) + 1, position: order.indexOf(shotId) + 1 }));
      requestAnimationFrame(() => focusCard(shotId));
    }
  }

  function rerun(shot: StoryboardShot) {
    void story.op("mark_redo", { shotId: shot.id, redo: true });
  }

  function onListKey(event: KeyboardEvent<HTMLDivElement>) {
    const card = event.target as HTMLElement;
    if (!card.matches?.("article.shot-card")) return;
    const id = card.dataset.shotId as string;
    const index = ids.indexOf(id);
    const shot = shots[index];
    if (!shot) return;
    switch (event.key) {
      case "ArrowDown":
      case "ArrowRight":
        event.preventDefault();
        focusCard(ids[Math.min(ids.length - 1, index + 1)]);
        break;
      case "ArrowUp":
      case "ArrowLeft":
        event.preventDefault();
        focusCard(ids[Math.max(0, index - 1)]);
        break;
      case "Enter":
        event.preventDefault();
        setEditingId(id);
        break;
      case "l":
      case "L":
        void story.op("lock", { shotId: id, locked: !shot.locked });
        break;
      case "r":
      case "R":
        void story.op("mark_redo", { shotId: id, redo: !shot.redo });
        break;
      case "Delete":
        setRemoveId(id);
        break;
      default:
    }
  }

  const header = (
    <>
      <div className="board-header__title">
        <h2>{t("board.title", { defaultValue: "{project} · Board", project: project.name })}</h2>
        <p className="muted-text">
          {t("board.subtitle", {
            defaultValue: "{shots, plural, one {# shot} other {# shots}} · {seconds} s · {stale} stale",
            shots: shots.length,
            seconds: filmSeconds(timeline),
            stale,
          })}
        </p>
      </div>
      <div className="board-header__actions">
        {story.notice === "conflict" ? (
          <span className="badge warning conflict-chip" role="status">
            {t("board.conflictNotice", { defaultValue: "Changed elsewhere; showing the latest" })}
            <button
              type="button"
              className="conflict-chip__dismiss"
              aria-label={t("storyboard.dismiss", { defaultValue: "Dismiss" })}
              onClick={story.dismissNotice}
            >
              ×
            </button>
          </span>
        ) : null}
        <button type="button" className="secondary-button" disabled={busy || project.undoStack.length === 0} onClick={() => void story.undo()}>
          <UndoIcon /> {t("board.undo", { defaultValue: "Undo" })}
        </button>
        <button type="button" className="secondary-button" disabled={busy || project.redoStack.length === 0} onClick={() => void story.redo()}>
          <RedoIcon /> {t("board.redo", { defaultValue: "Redo" })}
        </button>
        <button type="button" className="secondary-button" onClick={onOpenProjects}>
          {t("board.projects", { defaultValue: "Projects" })}
        </button>
        <button type="button" className="secondary-button" onClick={onOpenMixer}>
          {t("board.mixer", { defaultValue: "Mixer" })}
        </button>
      </div>
    </>
  );

  return (
    <div className="content-grid storyboard-page">
      <section className="panel span-2 board-tab">
      {running && batch ? <BatchBar batch={batch} onCancel={() => void story.cancelBatch()} /> : null}
      {story.error ? (
        <div className="callout error" role="alert">
          <p>{story.error}</p>
        </div>
      ) : null}
      <StudioShell
        studioId="story-board"
        header={header}
        inputs={
          <>
            <ProjectDefaultsPanel project={project} busy={busy} op={story.op} imageModels={imageModels} videoModels={videoModels} />
            <ReferencesPanel project={project} busy={busy} op={story.op} />
          </>
        }
        estimate={
          <>
            <p className="board-estimate__need">
              {missing.length > 0
                ? t("board.estimate.need", {
                    defaultValue: "{n, plural, one {# shot needs a new take} other {# shots need a new take}}",
                    n: missing.length,
                  })
                : t("board.estimate.none", { defaultValue: "Every unlocked shot has a take." })}
            </p>
            <RunEstimateStrip
              seconds={null}
              peakGb={null}
              totalGb={null}
              diskMb={null}
              basis={null}
              risk="safe"
              action={
                <div className="board-estimate__actions">
                  <button
                    type="button"
                    className="primary-button"
                    disabled={running || busy || missing.length === 0}
                    onClick={() => void story.startBatch("both", missing)}
                  >
                    {t("board.makeMissing", { defaultValue: "Make missing" })}
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={running || busy || unlocked.length === 0}
                    title={t("board.makeAllTitle", { defaultValue: "Adds a new take to every unlocked shot. Nothing is overwritten." })}
                    onClick={() => void story.startBatch("both", unlocked)}
                  >
                    {t("board.makeAll", { defaultValue: "Make all unlocked" })}
                  </button>
                </div>
              }
            />
          </>
        }
        results={
          <>
            <div className="board-shots__head">
              <h3>{t("board.shots", { defaultValue: "Shots" })}</h3>
              <button type="button" className="secondary-button" disabled={busy} onClick={() => void story.op("add_shot", {})}>
                <PlusIcon /> {t("board.addShot", { defaultValue: "Add shot" })}
              </button>
            </div>
            {shots.length === 0 ? (
              <div className="empty-state">
                <p className="muted-text">{t("board.empty", { defaultValue: "No shots yet. Add a shot, then give it a prompt or import footage." })}</p>
                <button type="button" className="primary-button" disabled={busy} onClick={() => void story.op("add_shot", {})}>
                  {t("board.addShot", { defaultValue: "Add shot" })}
                </button>
              </div>
            ) : (
              <div
                className="shot-list"
                role="list"
                ref={list}
                onKeyDown={onListKey}
                onFocusCapture={(event) => {
                  const id = (event.target as HTMLElement).closest<HTMLElement>("[data-shot-id]")?.dataset.shotId;
                  if (id) setFocusId(id);
                }}
              >
                {shots.map((shot, index) => (
                  <div role="listitem" key={shot.id} className="shot-list__item">
                    <ShotCard
                      projectId={project.id}
                      shot={shot}
                      index={index}
                      count={shots.length}
                      status={effectiveShotStatus(shot, live.get(shot.id))}
                      queuePosition={queuedOrder.indexOf(shot.id) + 1 || undefined}
                      staleAfter={staleAfterShot(shots, index)}
                      kept={kept.has(shot.id)}
                      failure={lastFailure(project.batches, shot.id)}
                      tabbable={shot.id === tabbableId}
                      dragOver={overId === shot.id && dragId !== shot.id}
                      disabled={busy}
                      onRename={(title) => void story.op("edit_shot", { shotId: shot.id, title })}
                      onPickTake={(takeId) => void story.op("pick_take", { shotId: shot.id, takeId })}
                      onToggleLock={() => void story.op("lock", { shotId: shot.id, locked: !shot.locked })}
                      onToggleRedo={() => void story.op("mark_redo", { shotId: shot.id, redo: !shot.redo })}
                      onTransition={(transition) => void story.op("edit_shot", { shotId: shot.id, transition })}
                      onMove={(step) => void reorderTo(reorderByStep(ids, shot.id, step), shot.id)}
                      onEdit={() => setEditingId(shot.id)}
                      onKeep={() => setKept((current) => new Set(current).add(shot.id))}
                      onRerun={() => rerun(shot)}
                      onRetry={() => void story.startBatch("both", [shot.id])}
                      onRemove={() => setRemoveId(shot.id)}
                      onDragStart={setDragId}
                      onDragEnd={() => {
                        setDragId(null);
                        setOverId(null);
                      }}
                      onDragOverCard={setOverId}
                      onDropOn={(drag, over) => {
                        setDragId(null);
                        void reorderTo(reorderByDrop(ids, drag, over), drag);
                      }}
                    />
                  </div>
                ))}
              </div>
            )}
            <p className="sr-only" role="status" aria-live="polite">{announce}</p>
          </>
        }
      />
      </section>
      {editing ? (
        <ShotEditorDrawer
          key={editing.id}
          project={project}
          shot={editing}
          index={ids.indexOf(editing.id)}
          imageGallery={imageGallery}
          videoGallery={videoGallery}
          busy={busy}
          batchRunning={running}
          op={story.op}
          onMake={(shotId) => void story.startBatch("both", [shotId])}
          onClose={() => setEditingId(null)}
        />
      ) : null}
      {toRemove ? (
        <ConfirmDialog
          title={t("board.removeTitle", { defaultValue: "Delete this shot?" })}
          body={
            <p>
              {t("board.removeBody", {
                defaultValue: "{name} leaves the film. Its takes stay in the project folder, and Undo brings it back.",
                name: toRemove.title || t("board.shot.untitled", { defaultValue: "Shot {n}", n: ids.indexOf(toRemove.id) + 1 }),
              })}
            </p>
          }
          confirmLabel={t("board.shot.menu.delete", { defaultValue: "Delete" })}
          danger
          busy={busy}
          onConfirm={() => {
            const id = toRemove.id;
            setRemoveId(null);
            if (editingId === id) setEditingId(null);
            void story.op("remove_shot", { shotId: id });
          }}
          onCancel={() => setRemoveId(null)}
        />
      ) : null}
    </div>
  );
}
