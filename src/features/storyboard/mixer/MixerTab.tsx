/** Storyboard Mixer: hear and see the film, place sound, export (UX spec D.6). */
import { isTauri } from "@tauri-apps/api/core";
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { fetchStoryboardFileBlobUrl, fetchStoryboardFileSize, storyboardFilePath } from "../../../api";
import { RedoIcon, UndoIcon } from "../../../components/icons/ActionIcons";
import { Panel } from "../../../components/Panel";
import type { StoryboardExportPreset, StoryboardProject, StoryboardTimeline } from "../../../types";
import { PathField } from "../PathField";
import type { UseStoryboardResult } from "../useStoryboard";
import { useStoryboardRender } from "../useStoryboardRender";
import { ClipInspector, ShotInspector } from "./ClipInspector";
import { AddAudioPanel } from "./AddAudioPanel";
import { ExportPanel, RenderProgress, type ExportResult } from "./ExportPanel";
import {
  aiLabelPatch,
  canRender,
  exportSize,
  fileNameOf,
  isRenderRunning,
  leftOutShotCount,
  mixerKeyAction,
  presetPatch,
  previewIsStale,
  projectFilePath,
  seekedTime,
  addAudioArgs,
  type MixerKeyAction,
  type TargetKind,
} from "./mixerLogic";
import { Timeline, type ClipCommit } from "./Timeline";
import {
  DEFAULT_ZOOM_INDEX,
  ZOOM_LEVELS,
  buildLanes,
  clampZoom,
  clipSpan,
  dropToAnchor,
  fitZoomIndex,
  pxPerSecond,
  shiftPosition,
  snapEdges,
  snapRange,
  trimAtPlayhead,
  trimEndPatch,
  trimStartPatch,
  type AudioLaneId,
  type ClipModel,
  type LaneId,
} from "./timelineGeometry";
import { Transport } from "./Transport";

function clipById(project: StoryboardProject, id: string) {
  return project.audio.find((clip) => clip.id === id) ?? null;
}

const PREVIEW_FILE = "exports/preview.mp4";
const NUDGE_DEBOUNCE_MS = 250;
const LABEL_COLUMN_PX = 120;

export interface MixerTabProps {
  backendOnline: boolean;
  story: UseStoryboardResult;
  /** The app's data directory, to build a path for "Copy path". */
  dataDirectory?: string;
  onOpenProjects: () => void;
  onOpenBoard: () => void;
}

type Selection = { type: "clip" | "shot"; id: string } | null;

export function MixerTab({ backendOnline, story, dataDirectory, onOpenProjects, onOpenBoard }: MixerTabProps) {
  const { t } = useTranslation("studio");
  const { project, timeline } = story;

  if (!story.projectId || story.notFound) {
    return (
      <div className="content-grid storyboard-page">
        <Panel title={t("mixer.noProjectTitle", { defaultValue: "Mixer" })} className="span-2">
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
  if (!project || !timeline) {
    return (
      <div className="content-grid storyboard-page">
        <Panel title={t("mixer.noProjectTitle", { defaultValue: "Mixer" })} className="span-2">
          <div className="empty-state">
            <p className="muted-text">
              {!backendOnline
                ? t("storyboard.offline", { defaultValue: "Waiting for the backend…" })
                : (story.error ?? t("board.loading", { defaultValue: "Loading the project…" }))}
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
  return (
    <MixerScreen
      key={project.id}
      story={story}
      project={project}
      timeline={timeline}
      dataDirectory={dataDirectory}
      onOpenProjects={onOpenProjects}
      onOpenBoard={onOpenBoard}
    />
  );
}

interface MixerScreenProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  timeline: StoryboardTimeline;
  dataDirectory?: string;
  onOpenProjects: () => void;
  onOpenBoard: () => void;
}

function targetKindOf(target: EventTarget | null): TargetKind {
  const el = target as HTMLElement | null;
  if (!el || !el.closest) return "other";
  if (el.closest("input, select, textarea, button, summary, a, [contenteditable='true']")) return "control";
  if (el.closest(".timeline__clip")) return "clip";
  return "other";
}

function MixerScreen({ story, project, timeline, dataDirectory, onOpenProjects, onOpenBoard }: MixerScreenProps) {
  const { t } = useTranslation("studio");
  const { busy } = story;
  const render = useStoryboardRender(project.id, true, () => void story.reload());
  const [zoomIndex, setZoomIndex] = useState(DEFAULT_ZOOM_INDEX);
  const [playheadS, setPlayheadS] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [selection, setSelection] = useState<Selection>(null);
  const [addLane, setAddLane] = useState<AudioLaneId | null>(null);
  const [relinkId, setRelinkId] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewRevision, setPreviewRevision] = useState<number | null>(null);
  const [exportBytes, setExportBytes] = useState<{ id: string; bytes: number | null } | null>(null);
  const [copied, setCopied] = useState(false);
  const [announce, setAnnounce] = useState("");
  const video = useRef<HTMLVideoElement>(null);
  const root = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const zoomed = useRef(false);
  const projectRef = useRef(project);
  projectRef.current = project;
  const selectNew = useRef<number | null>(null);
  const nudge = useRef<{ clipId: string; delta: number; timer: ReturnType<typeof setTimeout> } | null>(null);

  const pps = pxPerSecond(zoomIndex);
  const spans = timeline.shots;
  const untitled = t("board.shot.untitled", { defaultValue: "Shot {n}", n: "{n}" });
  const lanes = useMemo(() => buildLanes(project, timeline, untitled), [project, timeline, untitled]);
  const laneLabels: Record<LaneId, string> = {
    picture: t("mixer.lane.picture", { defaultValue: "Picture" }),
    native: t("mixer.lane.native", { defaultValue: "Shot sound" }),
    voice: t("mixer.lane.voice", { defaultValue: "Voice" }),
    sfx: t("mixer.lane.sfx", { defaultValue: "Effects" }),
    music: t("mixer.lane.music", { defaultValue: "Music" }),
  };
  const total = timeline.total;
  const size = exportSize(project.defaults, project.defaults.exportPreset, true);
  const rendering = isRenderRunning(render.job) || render.starting;
  const hasPreview = previewUrl !== null;
  const stale = hasPreview && previewIsStale(previewRevision, project.revision);
  const leftOut = leftOutShotCount(timeline);
  const selectedClip = selection?.type === "clip" ? clipById(project, selection.id) : null;
  const selectedShot = selection?.type === "shot" ? project.shots.find((s) => s.id === selection.id) : null;
  const disabled = busy;

  // Start at the zoom level where the whole film fits.
  useEffect(() => {
    if (zoomed.current || total <= 0) return;
    zoomed.current = true;
    const width = (scroller.current?.clientWidth ?? 0) - LABEL_COLUMN_PX;
    setZoomIndex(fitZoomIndex(total, width));
  }, [total]);

  // The preview file: load it when a render finishes, and once on arrival if one is already there.
  useEffect(() => {
    let cancelled = false;
    let made: string | null = null;
    fetchStoryboardFileBlobUrl(project.id, PREVIEW_FILE)
      .then((url) => {
        if (cancelled) URL.revokeObjectURL(url);
        else {
          made = url;
          setPreviewUrl(url);
        }
      })
      .catch(() => {
        if (!cancelled && render.previewVersion === 0) setPreviewUrl(null);
      });
    return () => {
      cancelled = true;
      if (made) URL.revokeObjectURL(made);
    };
  }, [project.id, render.previewVersion]);

  // Export size for the result card.
  const exportedId = render.exported?.renderId ?? null;
  const exportedFile = render.exported?.file ?? null;
  useEffect(() => {
    if (!exportedId || !exportedFile) return;
    let cancelled = false;
    fetchStoryboardFileSize(project.id, exportedFile)
      .then((bytes) => !cancelled && setExportBytes({ id: exportedId, bytes }))
      .catch(() => !cancelled && setExportBytes({ id: exportedId, bytes: null }));
    return () => {
      cancelled = true;
    };
  }, [project.id, exportedId, exportedFile]);

  // The playhead follows the video while it plays.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      const el = video.current;
      if (el) setPlayheadS(el.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  // Keep the playhead on screen while it moves.
  useEffect(() => {
    const el = scroller.current;
    if (!el || !playing) return;
    const x = playheadS * pps;
    const view = el.clientWidth - LABEL_COLUMN_PX;
    if (x < el.scrollLeft || x > el.scrollLeft + view - 24) el.scrollLeft = Math.max(0, x - 48);
  }, [playheadS, playing, pps]);

  // A clip added from this screen is selected once the project comes back with it.
  const audioCount = project.audio.length;
  useEffect(() => {
    if (selectNew.current !== null && audioCount > selectNew.current) {
      const last = projectRef.current.audio[projectRef.current.audio.length - 1];
      if (last) setSelection({ type: "clip", id: last.id });
      selectNew.current = null;
    }
  }, [audioCount]);

  // A selection that no longer exists (undo, remove) is dropped.
  useEffect(() => {
    if (selection?.type === "clip" && !clipById(project, selection.id)) setSelection(null);
    if (selection?.type === "shot" && !project.shots.some((s) => s.id === selection.id)) setSelection(null);
  }, [project, selection]);

  const seek = useCallback(
    (timeS: number) => {
      const next = Math.max(0, Math.min(total, timeS));
      setPlayheadS(next);
      const el = video.current;
      if (el && Number.isFinite(el.duration)) el.currentTime = Math.min(next, el.duration);
    },
    [total],
  );

  const togglePlay = useCallback(async () => {
    const el = video.current;
    if (!el) return;
    if (!el.paused) {
      el.pause();
      return;
    }
    try {
      if (el.ended || el.currentTime >= el.duration - 0.05) el.currentTime = 0;
      await el.play();
    } catch {
      setPlaying(false);
    }
  }, []);

  const stop = useCallback(() => {
    const el = video.current;
    if (el) {
      el.pause();
      el.currentTime = 0;
    }
    setPlaying(false);
    setPlayheadS(0);
  }, []);

  const zoom = useCallback((direction: 1 | -1) => setZoomIndex((z) => clampZoom(z + direction)), []);

  async function startRender(isPreview: boolean) {
    const ok = await render.start(project.revision, isPreview);
    if (ok && isPreview) {
      setPreviewRevision(project.revision);
      stop();
    }
  }

  // ---- edits ----
  const edgeList = useMemo(() => snapEdges(spans, playheadS), [spans, playheadS]);

  function commitClip(model: ClipModel, commit: ClipCommit) {
    const clip = model.clip;
    if (!clip) return;
    if (commit.mode === "move") {
      const raw = Math.max(0, model.start + commit.deltaS);
      const start = commit.alt ? raw : snapRange(raw, model.duration, pps, edgeList);
      void story.op("edit_audio", { clipId: clip.id, ...dropToAnchor(start, spans, commit.alt) });
    } else if (commit.mode === "trim-start") {
      void story.op("edit_audio", { clipId: clip.id, ...trimStartPatch(clip, commit.deltaS) });
    } else {
      void story.op("edit_audio", { clipId: clip.id, ...trimEndPatch(clip, commit.deltaS) });
    }
  }

  function flushNudge() {
    const pending = nudge.current;
    if (!pending) return;
    nudge.current = null;
    const clip = clipById(projectRef.current, pending.clipId);
    if (clip) void story.op("edit_audio", { clipId: clip.id, ...shiftPosition(clip, pending.delta) });
  }

  function nudgeClip(clipId: string, delta: number) {
    const pending = nudge.current;
    if (pending && pending.clipId === clipId) {
      clearTimeout(pending.timer);
      pending.delta += delta;
      pending.timer = setTimeout(flushNudge, NUDGE_DEBOUNCE_MS);
      return;
    }
    flushNudge();
    nudge.current = { clipId, delta, timer: setTimeout(flushNudge, NUDGE_DEBOUNCE_MS) };
  }

  useEffect(
    () => () => {
      if (nudge.current) clearTimeout(nudge.current.timer);
    },
    [],
  );

  function runAction(action: MixerKeyAction) {
    switch (action.type) {
      case "toggle-play":
        if (hasPreview) void togglePlay();
        break;
      case "seek":
        seek(seekedTime(playheadS, action.deltaS, total));
        break;
      case "seek-to":
        seek(action.where === "start" ? 0 : total);
        break;
      case "nudge-clip":
        if (selectedClip) nudgeClip(selectedClip.id, action.deltaS);
        break;
      case "zoom":
        zoom(action.direction);
        break;
      case "remove-clip":
        if (selectedClip) {
          void story.op("remove_audio", { clipId: selectedClip.id });
          setSelection(null);
        }
        break;
      case "trim":
        if (selectedClip) {
          const patch = trimAtPlayhead(selectedClip, clipSpan(selectedClip, timeline, spans), playheadS, action.side);
          if (patch) void story.op("edit_audio", { clipId: selectedClip.id, ...patch });
        }
        break;
      case "undo":
        void story.undo();
        break;
      case "redo":
        void story.redo();
        break;
      case "deselect":
        setSelection(null);
        setAddLane(null);
        setRelinkId(null);
        break;
      default:
    }
  }

  const onKey = (event: KeyboardEvent | ReactKeyboardEvent) => {
    const action = mixerKeyAction({
      key: event.key,
      shift: event.shiftKey,
      alt: event.altKey,
      mod: event.metaKey || event.ctrlKey,
      target: targetKindOf(event.target),
      clipSelected: selectedClip !== null,
      fps: project.defaults.fps,
    });
    if (!action) return;
    event.preventDefault();
    runAction(action);
  };
  const keyHandler = useRef(onKey);
  keyHandler.current = onKey;

  // Keys work anywhere on the screen, and when nothing in particular has focus.
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const inside = !!target && !!root.current && root.current.contains(target);
      const idle = !target || target === document.body;
      if (!inside && !idle) return;
      if (target?.closest?.(".modal-overlay")) return;
      keyHandler.current(event);
    };
    document.addEventListener("keydown", listener);
    return () => document.removeEventListener("keydown", listener);
  }, []);

  function selectModel(model: ClipModel) {
    if (model.type === "audio") {
      setSelection({ type: "clip", id: model.id });
    } else {
      setSelection({ type: "shot", id: model.id });
      seek(model.start);
    }
  }

  async function addAudio(lane: AudioLaneId, picked: { path: string; name?: string }, cloned: boolean) {
    selectNew.current = project.audio.length;
    const ok = await story.op("add_audio", addAudioArgs(lane, picked, playheadS, spans, cloned));
    if (ok) setAddLane(null);
    else selectNew.current = null;
  }

  async function relinkClip(path: string) {
    if (!relinkId) return;
    const ok = await story.op("relink", { id: relinkId, path });
    if (ok) setRelinkId(null);
  }

  async function chooseReveal(file: string) {
    const absolute = projectFilePath(dataDirectory, project.id, file);
    if (isTauri()) {
      try {
        const [{ revealItemInDir }, { homeDir }] = await Promise.all([import("@tauri-apps/plugin-opener"), import("@tauri-apps/api/path")]);
        const resolved = absolute.startsWith("~/") ? `${(await homeDir()).replace(/[\\/]+$/, "")}/${absolute.slice(2)}` : absolute;
        await revealItemInDir(resolved);
        return;
      } catch {
        // fall through to the download
      }
    }
    try {
      const url = await fetchStoryboardFileBlobUrl(project.id, file);
      const link = document.createElement("a");
      link.href = url;
      link.download = fileNameOf(file);
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (err) {
      setAnnounce(err instanceof Error ? err.message : String(err));
    }
  }

  async function copyPath(file: string) {
    try {
      await navigator.clipboard.writeText(projectFilePath(dataDirectory, project.id, file));
      setCopied(true);
      setAnnounce(t("mixer.export.copied", { defaultValue: "Path copied" }));
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setAnnounce(storyboardFilePath(project.id, file));
    }
  }

  const exportResult: ExportResult | null = render.exported
    ? { job: render.exported, bytes: exportBytes?.id === render.exported.renderId ? exportBytes.bytes : null }
    : null;
  const exportJob = render.job && !render.job.preview ? render.job : null;
  const previewJob = render.job?.preview ? render.job : null;
  const empty = project.shots.length === 0;
  const ratio = { "--stage-w": size.width, "--stage-h": size.height } as CSSProperties; // preview frame shape: computed from the export size
  const inspectorOpen = selection !== null;
  const relinkClipTarget = relinkId ? clipById(project, relinkId) : null;

  return (
    <div className="content-grid storyboard-page">
      <section className="panel span-2 mixer-tab" ref={root}>
        <header className="mixer-tab__header">
          <div className="board-header__title">
            <h2>{t("mixer.title", { defaultValue: "{project} · Mixer", project: project.name })}</h2>
            <p className="muted-text">
              {t("mixer.subtitle", {
                defaultValue: "{seconds} s · {aspect} · preview {width} × {height}",
                seconds: Math.round(total * 10) / 10,
                aspect: project.defaults.exportPreset,
                width: size.width,
                height: size.height,
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
            <button type="button" className="secondary-button" onClick={onOpenBoard}>
              {t("mixer.board", { defaultValue: "Board" })}
            </button>
            <button type="button" className="secondary-button" onClick={onOpenProjects}>
              {t("board.projects", { defaultValue: "Projects" })}
            </button>
          </div>
        </header>

        {story.error ? (
          <div className="callout error" role="alert">
            <p>{story.error}</p>
          </div>
        ) : null}

        {empty ? (
          <div className="empty-state">
            <p className="muted-text">{t("mixer.empty", { defaultValue: "Nothing to mix yet. Make shots on the Board first." })}</p>
            <button type="button" className="primary-button" onClick={onOpenBoard}>
              {t("mixer.openBoard", { defaultValue: "Open Board" })}
            </button>
          </div>
        ) : (
          <>
            <div className="mixer__top">
              <div className="mixer__stage-wrap">
                <div className="mixer__stage" style={ratio}>
                  {previewUrl ? (
                    <video
                      ref={video}
                      className="mixer__video"
                      src={previewUrl}
                      loop={loop}
                      playsInline
                      preload="auto"
                      onPlay={() => setPlaying(true)}
                      onPause={() => setPlaying(false)}
                      onEnded={() => setPlaying(false)}
                      onSeeked={(event) => setPlayheadS(event.currentTarget.currentTime)}
                    />
                  ) : (
                    <p className="mixer__stage-empty muted-text">
                      {t("mixer.preview.none", { defaultValue: "No preview yet. Render one to watch and hear the film." })}
                    </p>
                  )}
                </div>
                <div className="mixer__stage-actions">
                  <button type="button" className="primary-button" disabled={disabled || rendering || !canRender(timeline)} onClick={() => void startRender(true)}>
                    {t("mixer.preview.render", { defaultValue: "Render preview" })}
                  </button>
                  {previewJob?.state === "running" ? (
                    <button type="button" className="secondary-button" onClick={() => void render.cancel()}>
                      {t("mixer.preview.cancel", { defaultValue: "Cancel" })}
                    </button>
                  ) : null}
                  {stale ? (
                    <span className="badge warning" role="status">
                      {t("mixer.preview.stale", { defaultValue: "Out of date. Render again to hear your latest changes." })}
                    </span>
                  ) : null}
                </div>
                {previewJob?.state === "running" ? (
                  <RenderProgress
                    job={previewJob}
                    startedAt={render.startedAt}
                    filmSeconds={total}
                    title={t("mixer.preview.rendering", { defaultValue: "Rendering preview…" })}
                  />
                ) : null}
                {previewJob?.state === "failed" ? (
                  <div className="callout error" role="alert">
                    <p>{t("mixer.preview.failed", { defaultValue: "The preview did not render." })}</p>
                    {previewJob.error ? <p className="muted-text">{previewJob.error}</p> : null}
                  </div>
                ) : null}
                {render.error ? (
                  <div className="callout warning" role="alert">
                    <p>{render.error}</p>
                    <button type="button" className="secondary-button" onClick={render.clearError}>
                      {t("storyboard.dismiss", { defaultValue: "Dismiss" })}
                    </button>
                  </div>
                ) : null}
                {leftOut > 0 ? (
                  <p className="muted-text mixer__note">
                    {t("mixer.leftOut", {
                      defaultValue: "{n, plural, one {# shot has no clip yet and is left out of the preview and the export.} other {# shots have no clip yet and are left out of the preview and the export.}}",
                      n: leftOut,
                    })}
                  </p>
                ) : null}
              </div>
              <aside className={`mixer__inspector${inspectorOpen ? " mixer__inspector--open" : ""}`}>
                {selectedClip ? (
                  <ClipInspector
                    clip={selectedClip}
                    span={clipSpan(selectedClip, timeline, spans)}
                    shots={project.shots}
                    shotSpans={spans}
                    disabled={disabled}
                    op={story.op}
                    onRemove={() => {
                      void story.op("remove_audio", { clipId: selectedClip.id });
                      setSelection(null);
                    }}
                    onClose={() => setSelection(null)}
                  />
                ) : selectedShot ? (
                  <ShotInspector
                    shot={selectedShot}
                    index={project.shots.indexOf(selectedShot)}
                    span={spans.find((s) => s.id === selectedShot.id)}
                    disabled={disabled}
                    op={story.op}
                    onOpenBoard={onOpenBoard}
                    onClose={() => setSelection(null)}
                  />
                ) : (
                  <p className="muted-text clip-inspector--empty">
                    {t("mixer.inspector.empty", { defaultValue: "Select a clip on the timeline to edit its sound." })}
                  </p>
                )}
              </aside>
            </div>

            <Transport
              playing={playing}
              timeS={playheadS}
              durationS={total}
              loop={loop}
              canPlay={hasPreview}
              canZoomIn={zoomIndex < ZOOM_LEVELS.length - 1}
              canZoomOut={zoomIndex > 0}
              onPlay={() => void togglePlay()}
              onStop={stop}
              onLoop={() => setLoop((on) => !on)}
              onZoom={zoom}
            />

            <Timeline
              projectId={project.id}
              durationS={total}
              pxPerSecond={pps}
              playheadS={playheadS}
              lanes={lanes}
              laneLabels={laneLabels}
              selectedId={selection?.id ?? null}
              selectedType={selection?.type ?? null}
              onSelect={selectModel}
              onCommit={commitClip}
              onRelink={(model) => setRelinkId(model.id)}
              onKeyDown={(event, model) => {
                if (event.key === "Enter" && event.target === event.currentTarget) {
                  event.preventDefault();
                  selectModel(model);
                }
              }}
              onAdd={(lane) => setAddLane((current) => (current === lane ? null : lane))}
              onSeek={seek}
              onClearSelection={() => setSelection(null)}
              scrollRef={scroller}
            />
            <div className="mixer__links">
              <button type="button" className="mixer__link" onClick={onOpenBoard}>
                {t("mixer.openOnBoard", { defaultValue: "Open on Board" })}
              </button>
              <span className="muted-text">{t("mixer.transitionsNote", { defaultValue: "Transitions shorten the film by their length; the ruler shows the real total." })}</span>
            </div>

            {addLane ? (
              <AddAudioPanel
                key={addLane}
                lane={addLane}
                laneLabel={laneLabels[addLane]}
                disabled={disabled}
                onAdd={(picked, cloned) => void addAudio(addLane, picked, cloned)}
                onClose={() => setAddLane(null)}
              />
            ) : null}
            {relinkClipTarget ? (
              <div className="add-audio">
                <PathField
                  label={t("mixer.relinkFor", { defaultValue: "Find the file for {name}", name: relinkClipTarget.source.label || fileNameOf(relinkClipTarget.source.path) })}
                  buttonLabel={t("mixer.clip.relink", { defaultValue: "Relink" })}
                  extensions={["wav", "mp3", "flac", "m4a", "mp4", "mov"]}
                  disabled={disabled}
                  onSubmit={(path) => void relinkClip(path)}
                />
                <button type="button" className="secondary-button" onClick={() => setRelinkId(null)}>
                  {t("storyboard.cancel", { defaultValue: "Cancel" })}
                </button>
              </div>
            ) : null}

            <ExportPanel
              project={project}
              filmSeconds={total}
              empty={!canRender(timeline)}
              rendering={rendering}
              exportJob={exportJob}
              startedAt={render.startedAt}
              result={exportResult}
              disabled={disabled}
              copied={copied}
              revealLabel={isTauri() ? t("mixer.export.reveal", { defaultValue: "Reveal file" }) : t("mixer.export.download", { defaultValue: "Download" })}
              onPreset={(preset: StoryboardExportPreset) => void story.op("edit_project", presetPatch(preset))}
              onAiLabel={(change) => {
                const patch = aiLabelPatch(project, change);
                if (patch) void story.op("edit_project", patch);
              }}
              onLoudness={(lufs) => void story.op("edit_project", { export: { loudnessLufs: lufs } })}
              onExport={() => void startRender(false)}
              onCancel={() => void render.cancel()}
              onReveal={(file) => void chooseReveal(file)}
              onCopyPath={(file) => void copyPath(file)}
            />
          </>
        )}
        <p className="sr-only" role="status" aria-live="polite">{announce}</p>
      </section>
    </div>
  );
}
