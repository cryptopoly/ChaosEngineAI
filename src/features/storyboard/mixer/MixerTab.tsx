/** The Mixer tab of a project: hear and see the film, place sound, captions (UX spec D.6; export is its own tab, D-65). */
import { isTauri } from "@tauri-apps/api/core";
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import {
  exportStoryboardCaptions,
  fetchStoryboardFileBlobUrl,
  generateStoryboardCaptions,
} from "../../../api";
import type { StoryboardProject, StoryboardTimeline } from "../../../types";
import { PathField } from "../PathField";
import { revealProjectFile } from "../projectFiles";
import type { UseStoryboardResult } from "../useStoryboard";
import { useStoryboardRender, type UseStoryboardRenderResult } from "../useStoryboardRender";
import { ClipInspector, ShotInspector } from "./ClipInspector";
import { AddAudioPanel } from "./AddAudioPanel";
import { CaptionsPanel, type CaptionsState } from "./CaptionsPanel";
import { RenderProgress } from "./ExportPanel";
import {
  accumulateNudge,
  canRender,
  exportSize,
  fileNameOf,
  isRenderRunning,
  leftOutShotCount,
  mixerKeyAction,
  nudgeArgs,
  previewIsStale,
  readRenderedRevision,
  readoutDue,
  seekedTime,
  addAudioArgs,
  writeRenderedRevision,
  type MixerKeyAction,
  type PendingNudge,
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
import { VideoToolsPrompt } from "../../videotools/VideoToolsPrompt";
import { useVideoTools } from "../../videotools/useVideoTools";

function clipById(project: StoryboardProject, id: string) {
  return project.audio.find((clip) => clip.id === id) ?? null;
}

const PREVIEW_FILE = "exports/preview.mp4";
const NUDGE_DEBOUNCE_MS = 250;
const LABEL_COLUMN_PX = 120;

export interface MixerTabProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  timeline: StoryboardTimeline;
  /** The app's data directory, to build a path for "Copy path". */
  dataDirectory?: string;
  /** Opens the Storyboard tab, focused on a shot when one is given. */
  onOpenShot: (shotId: string | null) => void;
  /** The shell's render state, shared with the Export tab; without it the tab tracks renders itself. */
  render?: UseStoryboardRenderResult;
}

type Selection = { type: "clip" | "shot"; id: string } | null;

/** The id of the audio clip block that holds focus, if one does. */
function focusedClipIdOf(target: EventTarget | null): string | null {
  const el = target as HTMLElement | null;
  const block = el?.closest?.('.timeline__clip[data-clip-type="audio"]');
  return block?.getAttribute("data-clip-id") ?? null;
}

function targetKindOf(target: EventTarget | null): TargetKind {
  const el = target as HTMLElement | null;
  if (!el || !el.closest) return "other";
  if (el.closest("input, select, textarea, button, summary, a, [contenteditable='true']")) return "control";
  if (el.closest(".timeline__clip")) return "clip";
  return "other";
}

export function MixerTab({ story, project, timeline, dataDirectory, onOpenShot, render: shared }: MixerTabProps) {
  const { t } = useTranslation("studio");
  const { busy } = story;
  const own = useStoryboardRender(project.id, !shared, () => void story.reload());
  const render = shared ?? own;
  const videoTools = useVideoTools(true, () => render.clearError());
  const [zoomIndex, setZoomIndex] = useState(DEFAULT_ZOOM_INDEX);
  const [playheadS, setPlayheadS] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(false);
  const [selection, setSelection] = useState<Selection>(null);
  const [addLane, setAddLane] = useState<AudioLaneId | null>(null);
  const [relinkId, setRelinkId] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewRevision, setPreviewRevision] = useState<number | null>(() => readRenderedRevision(project.id));
  const [announce, setAnnounce] = useState("");
  const [captionsState, setCaptionsState] = useState<CaptionsState>({ busy: false, notes: [], error: null, file: null });
  const video = useRef<HTMLVideoElement>(null);
  const root = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const zoomed = useRef(false);
  const projectRef = useRef(project);
  projectRef.current = project;
  const selectNew = useRef<number | null>(null);
  const nudge = useRef<{ pending: PendingNudge; timer: ReturnType<typeof setTimeout> } | null>(null);
  const playheadRef = useRef(0);
  const readoutAt = useRef(0);
  const ppsRef = useRef(0);
  const timelineInner = useRef<HTMLDivElement>(null);
  const previewOwned = useRef<string | null>(null);
  const startedRevision = useRef<number | null>(null);

  const pps = pxPerSecond(zoomIndex);
  ppsRef.current = pps;
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
  // A failed refetch keeps the preview that is already showing.
  useEffect(() => {
    let cancelled = false;
    fetchStoryboardFileBlobUrl(project.id, PREVIEW_FILE)
      .then((url) => {
        if (cancelled) URL.revokeObjectURL(url);
        else setPreviewUrl(url);
      })
      .catch(() => {
        if (!cancelled && render.previewVersion === 0) setPreviewUrl(null);
      });
    return () => {
      cancelled = true;
    };
  }, [project.id, render.previewVersion]);

  // This screen owns the blob URLs it made: one is released once the player has moved on from it.
  useEffect(() => {
    const before = previewOwned.current;
    previewOwned.current = previewUrl;
    if (before && before !== previewUrl) URL.revokeObjectURL(before);
  }, [previewUrl]);
  useEffect(
    () => () => {
      if (previewOwned.current) URL.revokeObjectURL(previewOwned.current);
    },
    [],
  );

  // The revision a preview shows is whatever the project was when that render started, noted only once it finished.
  useEffect(() => {
    if (render.previewVersion === 0 || startedRevision.current === null) return;
    setPreviewRevision(startedRevision.current);
    writeRenderedRevision(project.id, startedRevision.current);
    startedRevision.current = null;
  }, [project.id, render.previewVersion]);

  /**
   * The playhead line moves every frame through a CSS variable, with no render; the number and everything
   * else that reads the playhead catch up at most ten times a second (at once when it is not playing).
   */
  const movePlayhead = useCallback((timeS: number, immediate: boolean) => {
    playheadRef.current = timeS;
    timelineInner.current?.style.setProperty("--playhead-x", String(timeS * ppsRef.current));
    const now = performance.now();
    if (immediate || readoutDue(readoutAt.current, now)) {
      readoutAt.current = now;
      setPlayheadS(timeS);
    }
  }, []);

  // The playhead follows the video while it plays.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      const el = video.current;
      if (el) movePlayhead(el.currentTime, false);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, movePlayhead]);

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
      movePlayhead(next, true);
      const el = video.current;
      if (el && Number.isFinite(el.duration)) el.currentTime = Math.min(next, el.duration);
    },
    [total, movePlayhead],
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
    movePlayhead(0, true);
  }, [movePlayhead]);

  const zoom = useCallback((direction: 1 | -1) => setZoomIndex((z) => clampZoom(z + direction)), []);

  async function startRender(isPreview: boolean) {
    const ok = await render.start(project.revision, isPreview);
    if (ok && isPreview) {
      startedRevision.current = project.revision;
      stop();
    }
  }

  // ---- captions and the export for an editor ----
  async function makeCaptions() {
    setCaptionsState((s) => ({ ...s, busy: true, error: null, notes: [] }));
    try {
      const done = await generateStoryboardCaptions(project.id, project.revision);
      setCaptionsState((s) => ({ ...s, busy: false, notes: done.notes }));
    } catch (err) {
      setCaptionsState((s) => ({ ...s, busy: false, error: err instanceof Error ? err.message : String(err) }));
    }
    await story.reload();
  }

  async function writeSrt() {
    setCaptionsState((s) => ({ ...s, error: null }));
    try {
      const done = await exportStoryboardCaptions(project.id);
      setCaptionsState((s) => ({ ...s, file: done.file }));
    } catch (err) {
      setCaptionsState((s) => ({ ...s, error: err instanceof Error ? err.message : String(err) }));
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

  // A nudge is sent as a relative change, worked out when its turn comes in the edit queue, so presses
  // sent while an earlier nudge is still on its way add to it instead of starting from the old position.
  const flushNudge = useCallback(() => {
    const queued = nudge.current;
    if (!queued) return;
    nudge.current = null;
    clearTimeout(queued.timer);
    void story.opFrom("edit_audio", (current) => nudgeArgs(current, queued.pending));
  }, [story]);
  const flushRef = useRef(flushNudge);
  flushRef.current = flushNudge;

  function nudgeClip(clipId: string, delta: number) {
    const { pending, flush } = accumulateNudge(nudge.current?.pending ?? null, clipId, delta);
    if (flush) flushNudge();
    if (nudge.current) clearTimeout(nudge.current.timer);
    nudge.current = { pending, timer: setTimeout(() => flushRef.current(), NUDGE_DEBOUNCE_MS) };
  }

  // Leaving the screen sends what is still waiting.
  useEffect(() => () => flushRef.current(), []);

  /** Undo and redo run after any nudge still waiting, never before it. */
  function undoNow() {
    flushNudge();
    void story.undo();
  }
  function redoNow() {
    flushNudge();
    void story.redo();
  }

  function runAction(action: MixerKeyAction) {
    switch (action.type) {
      case "toggle-play":
        if (hasPreview) void togglePlay();
        break;
      case "seek":
        seek(seekedTime(playheadRef.current, action.deltaS, total));
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
          const patch = trimAtPlayhead(selectedClip, clipSpan(selectedClip, timeline, spans), playheadRef.current, action.side);
          if (patch) void story.op("edit_audio", { clipId: selectedClip.id, ...patch });
        }
        break;
      case "undo":
        undoNow();
        break;
      case "redo":
        redoNow();
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
      selectedClipId: selectedClip?.id ?? null,
      focusedClipId: focusedClipIdOf(event.target),
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
    const ok = await story.op("add_audio", addAudioArgs(lane, picked, playheadRef.current, spans, cloned));
    if (ok) setAddLane(null);
    else selectNew.current = null;
  }

  async function relinkClip(path: string) {
    if (!relinkId) return;
    const ok = await story.op("relink", { id: relinkId, path });
    if (ok) setRelinkId(null);
  }

  async function chooseReveal(file: string) {
    const problem = await revealProjectFile(project.id, file, dataDirectory);
    if (problem) setAnnounce(problem);
  }

  const previewJob = render.job?.preview ? render.job : null;
  const empty = project.shots.length === 0;
  const ratio = { "--stage-w": size.width, "--stage-h": size.height } as CSSProperties; // preview frame shape: computed from the export size
  const inspectorOpen = selection !== null;
  const relinkClipTarget = relinkId ? clipById(project, relinkId) : null;

  return (
    <div className="mixer-tab" ref={root}>
        <p className="muted-text mixer-tab__summary">
          {t("mixer.subtitle", {
            defaultValue: "{seconds} s · {aspect} · preview {width} × {height}",
            seconds: Math.round(total * 10) / 10,
            aspect: project.defaults.exportPreset,
            width: size.width,
            height: size.height,
          })}
        </p>

        {empty ? (
          <div className="empty-state">
            <p className="muted-text">{t("mixer.emptyShots", { defaultValue: "Nothing to mix yet. Make shots on the Storyboard tab first." })}</p>
            <button type="button" className="primary-button" onClick={() => onOpenShot(null)}>
              {t("mixer.openStoryboard", { defaultValue: "Open Storyboard" })}
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
                      onPause={(event) => {
                        setPlaying(false);
                        movePlayhead(event.currentTarget.currentTime, true);
                      }}
                      onEnded={() => setPlaying(false)}
                      onSeeked={(event) => movePlayhead(event.currentTarget.currentTime, true)}
                      onTimeUpdate={(event) => movePlayhead(event.currentTarget.currentTime, false)}
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
                <VideoToolsPrompt
                  status={videoTools.status}
                  busy={videoTools.busy}
                  percent={videoTools.percent}
                  error={videoTools.error}
                  forced={render.needsVideoTools}
                  onInstall={() => void videoTools.install()}
                  onChooseFile={(path) => void videoTools.chooseFile(path)}
                  onDismissError={videoTools.clearError}
                />
                {render.error && !render.needsVideoTools ? (
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
                    onOpenBoard={() => onOpenShot(selectedShot.id)}
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
              innerRef={timelineInner}
            />
            <div className="mixer__links">
              <button type="button" className="mixer__link" onClick={() => onOpenShot(selectedShot?.id ?? null)}>
                {t("mixer.openInStoryboard", { defaultValue: "Open in Storyboard" })}
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

            <CaptionsPanel
              project={project}
              timeline={timeline}
              state={captionsState}
              disabled={disabled || rendering}
              playheadS={playheadS}
              revealLabel={isTauri() ? t("mixer.export.reveal", { defaultValue: "Reveal file" }) : t("mixer.export.download", { defaultValue: "Download" })}
              onGenerate={() => void makeCaptions()}
              onAdd={(startS) =>
                void story.op("add_caption", { start: startS, end: startS + 2, text: t("mixer.captions.newText", { defaultValue: "New caption" }) })
              }
              onEditText={(id, text) => void story.op("edit_caption", { captionId: id, text })}
              onRemove={(id) => void story.op("remove_caption", { captionId: id })}
              onExportSrt={() => void writeSrt()}
              onReveal={(file) => void chooseReveal(file)}
            />
          </>
        )}
        <p className="sr-only" role="status" aria-live="polite">{announce}</p>
    </div>
  );
}
