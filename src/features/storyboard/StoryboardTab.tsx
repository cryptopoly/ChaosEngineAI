/**
 * The Storyboard tab of a project (D-65): only the shots. Write, make and arrange them; project settings live
 * on Overview and the cast on Cast. A shot needs no character: it can stand alone.
 */
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { PlusIcon } from "../../components/icons/ActionIcons";
import type { ImageOutputArtifact, StoryboardBatchRecord, StoryboardProject, StoryboardShot, StoryboardTimeline, VideoOutputArtifact } from "../../types";
import { RunBar } from "./RunBar";
import { useProjectEstimate } from "./useProjectEstimate";
import { ConfirmDialog } from "./ConfirmDialog";
import { ShotCard } from "./ShotCard";
import { ShotEditorDrawer } from "./ShotEditorDrawer";
import {
  activeTakeOf,
  effectiveShotStatus,
  filmSeconds,
  isBareKey,
  isBatchRunning,
  isStillOnly,
  liveShotStates,
  missingShotIds,
  reorderByDrop,
  reorderByStep,
  staleAfterShot,
  staleCount,
  unlockedShotIds,
} from "./storyboardLogic";
import type { UseStoryboardResult } from "./useStoryboard";

export interface StoryboardTabProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  timeline: StoryboardTimeline;
  imageGallery: ImageOutputArtifact[];
  videoGallery: VideoOutputArtifact[];
  /** A shot to focus on arrival (a picture block clicked in the Mixer, a thumbnail on Overview). */
  focusShotId?: string | null;
  onFocused?: () => void;
}

/** The error text the last batch recorded for a shot, if that batch failed it. */
export function lastFailure(batches: StoryboardBatchRecord[], shotId: string): string | null {
  const last = batches[batches.length - 1];
  const failed = last?.results.find((r) => r.shotId === shotId && r.state === "failed");
  return failed ? (failed.error ?? "") : null;
}

export function StoryboardTab({ story, project, timeline, imageGallery, videoGallery, focusShotId, onFocused }: StoryboardTabProps) {
  const { t } = useTranslation("studio");
  const { batch, busy } = story;
  const [editingId, setEditingId] = useState<string | null>(null);
  const [removeId, setRemoveId] = useState<string | null>(null);
  const [kept, setKept] = useState<Set<string>>(() => new Set());
  const [dragId, setDragId] = useState<string | null>(null);
  const [overId, setOverId] = useState<string | null>(null);
  const [focusId, setFocusId] = useState<string | null>(focusShotId ?? null);
  const [announce, setAnnounce] = useState("");
  const list = useRef<HTMLDivElement>(null);

  const live = useMemo(() => liveShotStates(batch), [batch]);
  const queuedOrder = useMemo(() => project.shots.filter((s) => live.get(s.id) === "queued").map((s) => s.id), [project, live]);
  const running = isBatchRunning(batch);

  const shots = project.shots;
  const ids = shots.map((s) => s.id);
  const editing = shots.find((s) => s.id === editingId) ?? null;
  const toRemove = shots.find((s) => s.id === removeId) ?? null;
  const missing = missingShotIds(shots);
  const unlocked = unlockedShotIds(shots);
  const stillOnly = shots.filter((shot) => missing.includes(shot.id) && isStillOnly(shot)).length;
  const estimate = useProjectEstimate(project.id, project.revision, missing, !running);
  const stale = staleCount(project);
  const tabbableId = focusId && ids.includes(focusId) ? focusId : ids[0];
  const castById = new Map([...project.references.characters, ...project.references.places].map((r) => [r.id, r]));
  const castNamesOf = (shot: StoryboardShot) =>
    shot.referenceImages.map((id) => castById.get(id)?.name || t("cast.unnamed", { defaultValue: "Unnamed" })).filter(Boolean);

  function focusCard(id: string | undefined) {
    if (!id) return;
    const card = list.current?.querySelector<HTMLElement>(`[data-shot-id="${id}"]`);
    card?.focus();
    card?.scrollIntoView?.({ block: "nearest" });
  }

  // A shot asked for by another tab is focused once, on arrival.
  useEffect(() => {
    if (!focusShotId) return;
    requestAnimationFrame(() => focusCard(focusShotId));
    onFocused?.();
  }, [focusShotId]); // once per request: the callbacks are not dependencies

  async function addShot() {
    const before = new Set(ids);
    const ok = await story.op("add_shot", {});
    if (ok) {
      requestAnimationFrame(() => {
        const added = Array.from(list.current?.querySelectorAll<HTMLElement>("[data-shot-id]") ?? []).find(
          (el) => !before.has(el.dataset.shotId ?? ""),
        );
        added?.focus();
      });
    }
  }

  async function reorderTo(order: string[] | null, shotId: string) {
    if (!order) return;
    const ok = await story.op("reorder", { order });
    if (ok) {
      setAnnounce(t("board.shot.moved", { defaultValue: "Shot {n} moved to position {position}", n: ids.indexOf(shotId) + 1, position: order.indexOf(shotId) + 1 }));
      requestAnimationFrame(() => focusCard(shotId));
    }
  }

  /** A cast entry from this shot's still, named after the shot; the Cast tab shows it. */
  async function useAsReference(shot: StoryboardShot, index: number) {
    const still = activeTakeOf(shot)?.still;
    if (!still) return;
    const name = shot.title || t("board.shot.untitled", { defaultValue: "Shot {n}", n: index + 1 });
    const ok = await story.op("add_reference", { kind: "character", name, images: [still] });
    if (ok) setAnnounce(t("board.shot.referenceAdded", { defaultValue: "{name} was added to the Cast with this still.", name }));
  }

  function rerun(shot: StoryboardShot) {
    void story.op("mark_redo", { shotId: shot.id, redo: true });
  }

  function onListKey(event: KeyboardEvent<HTMLDivElement>) {
    const card = event.target as HTMLElement;
    if (!card.matches?.("article.shot-card")) return;
    if (!isBareKey(event)) return;
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

  return (
    <div className="board-tab">
      <div className="board-shots__head">
        <div>
          <h3>{t("board.shots", { defaultValue: "Shots" })}</h3>
          <p className="muted-text">
            {t("board.subtitleShots", {
              defaultValue: "{shots, plural, one {# shot} other {# shots}} · {seconds} s · {stale} stale",
              shots: shots.length,
              seconds: filmSeconds(timeline),
              stale,
            })}
          </p>
        </div>
        <button type="button" className="secondary-button" disabled={busy} onClick={() => void addShot()}>
          <PlusIcon /> {t("board.addShot", { defaultValue: "Add shot" })}
        </button>
      </div>
      {shots.length > 1 && timeline.total > 0 ? (
        <div className="length-strip" role="group" aria-label={t("board.lengthStrip", { defaultValue: "Shot lengths" })}>
          {timeline.shots.map((span) => {
            const index = ids.indexOf(span.id);
            if (index < 0) return null;
            return (
              <button
                key={span.id}
                type="button"
                className={`length-strip__part length-strip__part--${shots[index].status}`}
                style={{ flexGrow: Math.max(0.2, span.duration) }}
                title={t("board.lengthStripPart", { defaultValue: "Shot {n} · {seconds} s", n: index + 1, seconds: Math.round(span.duration * 10) / 10 })}
                aria-label={t("board.lengthStripPart", { defaultValue: "Shot {n} · {seconds} s", n: index + 1, seconds: Math.round(span.duration * 10) / 10 })}
                onClick={() => focusCard(span.id)}
              >
                {index + 1}
              </button>
            );
          })}
        </div>
      ) : null}
      {shots.length === 0 ? (
        <div className="empty-state">
          <p className="muted-text">
            {t("board.emptyNoCast", {
              defaultValue: "No shots yet. Add a shot and describe it, or import footage. A shot does not need a character: a street or a crowd works on its own.",
            })}
          </p>
          <button type="button" className="primary-button" disabled={busy} onClick={() => void addShot()}>
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
                castNames={castNamesOf(shot)}
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
                onMakeClip={running ? undefined : () => void story.startBatch("clip", [shot.id])}
                onUseAsReference={() => void useAsReference(shot, index)}
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
          <div role="listitem" className="shot-list__item">
            <button type="button" className="shot-ghost" disabled={busy} onClick={() => void addShot()}>
              <PlusIcon aria-hidden="true" /> {t("board.addShot", { defaultValue: "Add shot" })}
            </button>
          </div>
        </div>
      )}
      {shots.length > 0 ? (
        <RunBar
          missing={missing.length}
          stills={stillOnly}
          stale={stale}
          unlocked={unlocked.length}
          estimate={estimate}
          batch={batch}
          batchProgress={story.batchProgress ?? null}
          running={running}
          busy={busy}
          onMakeMissing={() => void story.startBatch("both", missing)}
          onMakeAll={() => void story.startBatch("both", unlocked)}
          onCancel={() => void story.cancelBatch()}
        />
      ) : null}
      <p className="sr-only" role="status" aria-live="polite">{announce}</p>
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
