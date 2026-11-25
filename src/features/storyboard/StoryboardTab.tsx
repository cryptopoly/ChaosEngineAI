/**
 * The Storyboard tab of a project (D-65): only the shots. Write, make and arrange them; project settings live
 * on Overview and the cast on Cast. A shot needs no character: it can stand alone.
 */
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { PlusIcon } from "../../components/icons/ActionIcons";
import type {
  ImageOutputArtifact,
  StoryboardBatchRecord,
  StoryboardFailureAction,
  StoryboardGuide,
  StoryboardProject,
  StoryboardShot,
  StoryboardTimeline,
  VideoOutputArtifact,
} from "../../types";
import { shotCardGuide } from "./cardGuide";
import { shotSeconds } from "./planLogic";
import { RunBar } from "./RunBar";
import { RoughCutButton } from "./preview/RoughCutButton";
import { useProjectEstimate } from "./useProjectEstimate";
import { ConfirmDialog } from "./ConfirmDialog";
import { ShotCard } from "./ShotCard";
import { ShotEditorDrawer } from "./ShotEditorDrawer";
import {
  activeTakeOf,
  effectiveShotStatus,
  filmSeconds,
  finalPassShotIds,
  isBareKey,
  isBatchRunning,
  isStillOnly,
  keptShotIds,
  liveShotStates,
  missingShotIds,
  primaryPass,
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
  /** A shot whose editor another tab asked for (a Lengthen or Shorten fix), and the field to focus in it. */
  editRequest?: { shotId: string; field?: "length" } | null;
  onEditRequestHandled?: () => void;
  /** The project's guide block (`story.guide`, or the shell's fallback read): quality, failures, coverage, cast. */
  guide?: StoryboardGuide | null;
  /** Opens one of the app's other tabs (Image Models, Video Models, Setup); without it those fixes just retry. */
  onOpenAppTab?: (tab: "image-models" | "video-models" | "settings") => void;
}

/** The error text the last batch recorded for a shot, if that batch failed it. */
export function lastFailure(batches: StoryboardBatchRecord[], shotId: string): string | null {
  const last = batches[batches.length - 1];
  const failed = last?.results.find((r) => r.shotId === shotId && r.state === "failed");
  return failed ? (failed.error ?? "") : null;
}

export function StoryboardTab({
  story, project, timeline, imageGallery, videoGallery, focusShotId, onFocused, editRequest, onEditRequestHandled, guide, onOpenAppTab,
}: StoryboardTabProps) {
  const { t } = useTranslation("studio");
  const { batch, busy } = story;
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editField, setEditField] = useState<"length" | undefined>(undefined);
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
  const finalIds = finalPassShotIds(shots);
  const keptCount = keptShotIds(shots).length;
  const estimate = useProjectEstimate(project.id, project.revision, missing, !running && (missing.length > 0 || finalIds.length > 0));
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

  // Another tab asked for one shot's editor, on one of its fields.
  useEffect(() => {
    if (!editRequest) return;
    setEditField(editRequest.field);
    setEditingId(editRequest.shotId);
    requestAnimationFrame(() => focusCard(editRequest.shotId));
    onEditRequestHandled?.();
  }, [editRequest]); // once per request: the callbacks are not dependencies

  function openEditor(shotId: string) {
    setEditField(undefined);
    setEditingId(shotId);
  }

  /** What a failed card's button does. `retry` and `edit-shot` are the card's own; the rest come here. */
  function failureAction(shot: StoryboardShot, action: StoryboardFailureAction) {
    const kind = guide?.failures?.find((failure) => failure.shotId === shot.id)?.kind ?? "still";
    switch (action) {
      case "smaller-size":
        void story.startBatch(kind === "clip" ? "both" : "still", [shot.id], false, true);
        break;
      case "open-models":
        if (onOpenAppTab) onOpenAppTab(kind === "clip" ? "video-models" : "image-models");
        else void story.startBatch("both", [shot.id]);
        break;
      case "install-video-tools":
        if (onOpenAppTab) onOpenAppTab("settings");
        else void story.startBatch("both", [shot.id]);
        break;
      case "edit-shot":
        openEditor(shot.id);
        break;
      case "free-disk":
        break; // the sentence says what to free; Retry follows once it is freed
      default:
        void story.startBatch("both", [shot.id]);
    }
  }

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
      case "Home":
        event.preventDefault();
        focusCard(ids[0]);
        break;
      case "End":
        event.preventDefault();
        focusCard(ids[ids.length - 1]);
        break;
      case "Enter":
        event.preventDefault();
        openEditor(id);
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
        <div className="board-shots__actions">
          <RoughCutButton project={project} timeline={timeline} quality={guide?.quality} />
          <button type="button" className="secondary-button" disabled={busy} onClick={() => void addShot()}>
            <PlusIcon /> {t("board.addShot", { defaultValue: "Add shot" })}
          </button>
        </div>
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
                // A pointer shortcut: keyboard users move between the cards with the arrow keys, so the strip
                // adds no tab stop per shot.
                tabIndex={-1}
                style={{ flexGrow: Math.max(0.2, span.duration) }}
                title={t("board.lengthStripPart", { defaultValue: "Shot {n} · {seconds} s", n: index + 1, seconds: shotSeconds(span.duration) })}
                aria-label={t("board.lengthStripPart", { defaultValue: "Shot {n} · {seconds} s", n: index + 1, seconds: shotSeconds(span.duration) })}
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
                guide={shotCardGuide(project, guide, shot)}
                onFailureAction={(action) => failureAction(shot, action)}
                onRename={(title) => story.op("edit_shot", { shotId: shot.id, title })}
                onPickTake={(takeId) => void story.op("pick_take", { shotId: shot.id, takeId })}
                onToggleLock={() => void story.op("lock", { shotId: shot.id, locked: !shot.locked })}
                onToggleRedo={() => void story.op("mark_redo", { shotId: shot.id, redo: !shot.redo })}
                onToggleKept={() => void story.op("set_kept", { shotId: shot.id, kept: !shot.kept })}
                onTransition={(transition) => void story.op("edit_shot", { shotId: shot.id, transition })}
                onMove={(step) => void reorderTo(reorderByStep(ids, shot.id, step), shot.id)}
                onEdit={() => openEditor(shot.id)}
                onKeep={() => setKept((current) => new Set(current).add(shot.id))}
                onRerun={() => rerun(shot)}
                onRetry={() => void story.startBatch("both", [shot.id])}
                onMakeClip={running ? undefined : () => void story.startBatch("clip", [shot.id])}
                onRedoClip={running ? undefined : () => void story.startBatch("redo-clip", [shot.id])}
                onRedoStill={running ? undefined : () => void story.startBatch("redo-still", [shot.id])}
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
          onPause={story.pauseBatch ? () => void story.pauseBatch?.() : undefined}
          onResume={story.resumeBatch ? () => void story.resumeBatch?.() : undefined}
          pausing={story.pausing}
          kept={keptCount}
          finalShots={finalIds.length}
          onMakeDrafts={(withClips) => void story.startBatch(withClips ? "both" : "still", missing, false, true)}
          onFinalPass={() => void story.startBatch("final", finalIds)}
          primary={primaryPass(shots, missing.length, estimate)}
          quality={guide?.quality}
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
          onRedo={(what, shotId) => void story.startBatch(what, [shotId])}
          onClose={() => setEditingId(null)}
          focusField={editField}
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
