/**
 * The Mixer timeline (UX spec D.6): a ruler, the picture row, the shot sound row and three audio
 * lanes. Blocks are placed with CSS custom properties (`--clip-x`, `--clip-w`, `--playhead-x`,
 * `--timeline-w`): their pixel values come from the zoom level and the project's times, so they cannot
 * be classes. Everything else is classes on the shared tokens.
 */
import { useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent, type Ref } from "react";
import { useTranslation } from "react-i18next";
import { PlusIcon } from "../../../components/icons/ActionIcons";
import { WaveformCanvas } from "../../../components/WaveformCanvas";
import { ProjectImage } from "../ProjectMedia";
import { thumbnailOf } from "../storyboardLogic";
import { usePeaks } from "./usePeaks";
import {
  AUDIO_LANES,
  bucketsFor,
  formatTimecode,
  levelFraction,
  peaksForBlock,
  rulerTicks,
  seekFromPointer,
  tickLabel,
  tickStep,
  type AudioLaneId,
  type ClipModel,
  type LaneId,
  type LaneModel,
} from "./timelineGeometry";

const DRAG_THRESHOLD_PX = 3;
const MIN_BLOCK_PX = 6;
const TRACK_PAD_PX = 48;

// ---- ruler ----
export interface TimeRulerProps {
  durationS: number;
  pxPerSecond: number;
  onSeek: (timeS: number) => void;
}

export function TimeRuler({ durationS, pxPerSecond, onSeek }: TimeRulerProps) {
  const { t } = useTranslation("studio");
  const ref = useRef<HTMLDivElement>(null);
  const scrubbing = useRef(false);
  const step = tickStep(pxPerSecond);
  const seek = (event: PointerEvent<HTMLDivElement>) => {
    const left = ref.current?.getBoundingClientRect().left ?? 0;
    onSeek(seekFromPointer(event.clientX, left, pxPerSecond, durationS));
  };
  return (
    <div
      ref={ref}
      className="time-ruler"
      role="slider"
      tabIndex={-1}
      aria-label={t("mixer.ruler", { defaultValue: "Timeline ruler. Click to move the playhead." })}
      aria-valuemin={0}
      aria-valuemax={Math.round(durationS * 10) / 10}
      aria-valuenow={0}
      onPointerDown={(event) => {
        scrubbing.current = true;
        event.currentTarget.setPointerCapture?.(event.pointerId);
        seek(event);
      }}
      onPointerMove={(event) => {
        if (scrubbing.current) seek(event);
      }}
      onPointerUp={() => {
        scrubbing.current = false;
      }}
      onPointerCancel={() => {
        scrubbing.current = false;
      }}
    >
      {rulerTicks(durationS, pxPerSecond).map((tick) => (
        <span
          key={tick.timeS}
          className={`time-ruler__tick${tick.major ? " time-ruler__tick--major" : ""}`}
          style={{ "--clip-x": tick.timeS * pxPerSecond } as CSSProperties /* tick position: computed from zoom */}
        >
          {tick.major ? <span className="time-ruler__label">{tickLabel(tick.timeS, step)}</span> : null}
        </span>
      ))}
    </div>
  );
}

// ---- one block ----
type DragMode = "move" | "trim-start" | "trim-end";

export interface ClipCommit {
  mode: DragMode;
  /** Seconds the block moved (negative = earlier). */
  deltaS: number;
  alt: boolean;
}

export interface TimelineClipProps {
  projectId: string;
  model: ClipModel;
  laneLabel: string;
  pxPerSecond: number;
  selected: boolean;
  onSelect: (model: ClipModel) => void;
  onCommit: (model: ClipModel, commit: ClipCommit) => void;
  onRelink: (model: ClipModel) => void;
  onKeyDown: (event: KeyboardEvent<HTMLElement>, model: ClipModel) => void;
}

function ClipWave({ projectId, model, pxPerSecond }: { projectId: string; model: ClipModel; pxPerSecond: number }) {
  const { t } = useTranslation("studio");
  const state = usePeaks(projectId, model.missing ? null : model.sourcePath, model.sourceKey, bucketsFor(model.sourceLength));
  if (!model.sourcePath || model.missing) return null;
  if (state.loading) return <span className="timeline__reading">{t("mixer.clip.reading", { defaultValue: "Reading…" })}</span>;
  const sourceLength = state.peaks?.durationS || model.sourceLength || 0;
  const shown = model.type === "shot" && model.span ? model.span.source_out - model.span.source_in : model.duration;
  const peaks = state.peaks ? peaksForBlock(state.peaks.peaks, sourceLength, model.trimIn, shown, !!model.clip?.loop) : [];
  if (peaks.length === 0) return <span className="timeline__reading timeline__reading--silent" aria-hidden="true" />;
  return (
    <div className="timeline__wave" data-width={Math.round(model.duration * pxPerSecond)}>
      <WaveformCanvas peaks={peaks} height={36} />
    </div>
  );
}

export function TimelineClip(props: TimelineClipProps) {
  const { projectId, model, laneLabel, pxPerSecond, selected, onSelect, onCommit, onRelink, onKeyDown } = props;
  const { t } = useTranslation("studio");
  const drag = useRef<{ mode: DragMode; x: number; moved: boolean } | null>(null);
  const [dx, setDx] = useState(0);
  const [mode, setMode] = useState<DragMode | null>(null);
  const editable = model.type === "audio" && !model.missing;

  let x = model.start * pxPerSecond;
  let width = Math.max(MIN_BLOCK_PX, model.duration * pxPerSecond);
  if (mode === "move") x += dx;
  else if (mode === "trim-start") {
    const applied = Math.min(dx, width - MIN_BLOCK_PX);
    x += applied;
    width -= applied;
  } else if (mode === "trim-end") width = Math.max(MIN_BLOCK_PX, width + dx);

  function begin(event: PointerEvent<HTMLElement>, next: DragMode) {
    if (event.button !== 0 || !editable) return;
    event.stopPropagation();
    drag.current = { mode: next, x: event.clientX, moved: false };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    onSelect(model);
  }
  function move(event: PointerEvent<HTMLElement>) {
    const current = drag.current;
    if (!current) return;
    const delta = event.clientX - current.x;
    if (!current.moved && Math.abs(delta) < DRAG_THRESHOLD_PX) return;
    if (!current.moved) {
      current.moved = true;
      setMode(current.mode);
    }
    setDx(delta);
  }
  function end(event: PointerEvent<HTMLElement>, cancelled: boolean) {
    const current = drag.current;
    drag.current = null;
    if (!current) return;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    const delta = event.clientX - current.x;
    setMode(null);
    setDx(0);
    if (!cancelled && current.moved) onCommit(model, { mode: current.mode, deltaS: delta / pxPerSecond, alt: event.altKey });
  }

  const label = model.clip
    ? t("mixer.clip.ariaLabel", {
        defaultValue: "{lane}, {name}, from {start} to {end} seconds, {gain} dB",
        lane: laneLabel,
        name: model.label,
        start: Math.round(model.start * 10) / 10,
        end: Math.round(model.end * 10) / 10,
        gain: model.clip.gainDb,
      })
    : t("mixer.block.ariaLabel", {
        defaultValue: "{lane}, {name}, from {start} to {end} seconds",
        lane: laneLabel,
        name: model.label,
        start: Math.round(model.start * 10) / 10,
        end: Math.round(model.end * 10) / 10,
      });

  const classes = [
    "timeline__clip",
    `timeline__clip--${model.laneId}`,
    selected ? "timeline__clip--selected" : "",
    model.missing ? "timeline__clip--missing" : "",
    model.clip?.duck ? "timeline__clip--ducked" : "",
    mode ? "timeline__clip--dragging" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const thumb = model.laneId === "picture" && model.shot ? thumbnailOf(model.shot) : null;

  return (
    <div
      role="listitem"
      className={classes}
      tabIndex={0}
      aria-label={label}
      aria-current={selected ? "true" : undefined}
      data-clip-id={model.id}
      style={{ "--clip-x": x, "--clip-w": width } as CSSProperties /* block position: computed from time and zoom */}
      onClick={() => !drag.current && onSelect(model)}
      onKeyDown={(event) => onKeyDown(event, model)}
      onPointerDown={(event) => begin(event, "move")}
      onPointerMove={move}
      onPointerUp={(event) => end(event, false)}
      onPointerCancel={(event) => end(event, true)}
    >
      {editable ? (
        <span
          className="timeline__clip-handle timeline__clip-handle--start"
          aria-hidden="true"
          onPointerDown={(event) => begin(event, "trim-start")}
          onPointerMove={move}
          onPointerUp={(event) => end(event, false)}
          onPointerCancel={(event) => end(event, true)}
        />
      ) : null}
      {model.laneId === "picture" ? null : <ClipWave projectId={projectId} model={model} pxPerSecond={pxPerSecond} />}
      {thumb?.kind === "still" ? <ProjectImage projectId={projectId} rel={thumb.path} className="timeline__thumb" /> : null}
      {model.duckRegions.map((region) => (
        <span
          key={`${region.from}-${region.to}`}
          className="timeline__duck"
          aria-hidden="true"
          style={{ "--clip-x": region.from * pxPerSecond, "--clip-w": (region.to - region.from) * pxPerSecond } as CSSProperties /* ducked span: computed from time and zoom */}
        />
      ))}
      <span className="timeline__clip-name">{model.label}</span>
      {model.laneId === "native" && model.shot ? (
        <span className="timeline__level" title={`${model.shot.nativeSoundDb} dB`}>
          <span
            className="timeline__level-fill"
            style={{ "--level": Math.round(levelFraction(model.shot.nativeSoundDb) * 100) } as CSSProperties /* level bar: computed from the shot's gain */}
          />
          <span className="timeline__level-text">{model.shot.nativeSoundDb} dB</span>
        </span>
      ) : null}
      {model.missing ? (
        <span className="timeline__missing">
          <span>{t("mixer.clip.missing", { defaultValue: "Missing" })}</span>
          {model.type === "audio" ? (
            <button
              type="button"
              className="secondary-button timeline__relink"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                onRelink(model);
              }}
            >
              {t("mixer.clip.relink", { defaultValue: "Relink" })}
            </button>
          ) : null}
        </span>
      ) : null}
      {editable && !model.clip?.loop ? (
        <span
          className="timeline__clip-handle timeline__clip-handle--end"
          aria-hidden="true"
          onPointerDown={(event) => begin(event, "trim-end")}
          onPointerMove={move}
          onPointerUp={(event) => end(event, false)}
          onPointerCancel={(event) => end(event, true)}
        />
      ) : null}
    </div>
  );
}

// ---- one lane ----
export interface TimelineLaneProps {
  projectId: string;
  lane: LaneModel;
  label: string;
  pxPerSecond: number;
  selectedId: string | null;
  selectedType: "clip" | "shot" | null;
  onSelect: (model: ClipModel) => void;
  onCommit: (model: ClipModel, commit: ClipCommit) => void;
  onRelink: (model: ClipModel) => void;
  onKeyDown: (event: KeyboardEvent<HTMLElement>, model: ClipModel) => void;
  onAdd?: (lane: AudioLaneId) => void;
  /** Crossfade overlaps to mark on the picture row: start and length in seconds. */
  overlaps?: { start: number; seconds: number }[];
}

const isAudioLane = (id: LaneId): id is AudioLaneId => (AUDIO_LANES as readonly string[]).includes(id);

export function TimelineLane(props: TimelineLaneProps) {
  const { lane, label, pxPerSecond, selectedId, selectedType, onAdd, overlaps } = props;
  const { t } = useTranslation("studio");
  return (
    <div className={`timeline__lane timeline__lane--${lane.id}`}>
      <div className="timeline__lane-label">
        <span className="timeline__lane-name">{label}</span>
        {isAudioLane(lane.id) && onAdd ? (
          <button
            type="button"
            className="icon-button timeline__lane-add"
            aria-label={t("mixer.addAudioTo", { defaultValue: "Add audio to {lane}", lane: label })}
            title={t("mixer.addAudioTo", { defaultValue: "Add audio to {lane}", lane: label })}
            onClick={() => onAdd(lane.id as AudioLaneId)}
          >
            <PlusIcon />
          </button>
        ) : null}
      </div>
      <div className="timeline__track" role="list" aria-label={label}>
        {lane.clips.map((model) => (
          <TimelineClip
            key={`${model.laneId}-${model.id}`}
            projectId={props.projectId}
            model={model}
            laneLabel={label}
            pxPerSecond={pxPerSecond}
            selected={selectedId === model.id && selectedType === (model.type === "audio" ? "clip" : "shot")}
            onSelect={props.onSelect}
            onCommit={props.onCommit}
            onRelink={props.onRelink}
            onKeyDown={props.onKeyDown}
          />
        ))}
        {overlaps?.map((overlap) => (
          <span
            key={overlap.start}
            className="timeline__xfade"
            aria-hidden="true"
            title={t("mixer.crossfade", { defaultValue: "Crossfade {seconds} s", seconds: Math.round(overlap.seconds * 100) / 100 })}
            style={{ "--clip-x": overlap.start * pxPerSecond, "--clip-w": overlap.seconds * pxPerSecond } as CSSProperties /* overlap: computed from time and zoom */}
          />
        ))}
      </div>
    </div>
  );
}

// ---- the whole timeline ----
export interface TimelineProps {
  projectId: string;
  durationS: number;
  pxPerSecond: number;
  playheadS: number;
  lanes: LaneModel[];
  laneLabels: Record<LaneId, string>;
  selectedId: string | null;
  selectedType: "clip" | "shot" | null;
  onSelect: (model: ClipModel) => void;
  onCommit: (model: ClipModel, commit: ClipCommit) => void;
  onRelink: (model: ClipModel) => void;
  onKeyDown: (event: KeyboardEvent<HTMLElement>, model: ClipModel) => void;
  onAdd: (lane: AudioLaneId) => void;
  onSeek: (timeS: number) => void;
  onClearSelection: () => void;
  scrollRef?: Ref<HTMLDivElement>;
}

export function Timeline(props: TimelineProps) {
  const { durationS, pxPerSecond, playheadS, lanes, laneLabels } = props;
  const { t } = useTranslation("studio");
  const farthest = lanes.reduce((max, lane) => lane.clips.reduce((m, c) => Math.max(m, c.end), max), durationS);
  const widthPx = Math.ceil(farthest * pxPerSecond) + TRACK_PAD_PX;
  const picture = lanes.find((lane) => lane.id === "picture");
  const overlaps = (picture?.clips ?? [])
    .filter((clip) => clip.span && clip.span.transition_type === "crossfade" && clip.span.transition_seconds > 0)
    .map((clip) => ({ start: clip.start, seconds: clip.span?.transition_seconds ?? 0 }));

  return (
    <div className="timeline" aria-label={t("mixer.timeline", { defaultValue: "Timeline" })}>
      <div className="timeline__scroll" ref={props.scrollRef}>
        <div
          className="timeline__inner"
          style={{ "--timeline-w": widthPx, "--playhead-x": playheadS * pxPerSecond } as CSSProperties /* track width and playhead: computed from time and zoom */}
          onClick={(event) => {
            if (event.target === event.currentTarget || (event.target as HTMLElement).classList.contains("timeline__track")) {
              props.onClearSelection();
            }
          }}
        >
          <div className="timeline__lane timeline__lane--ruler">
            <div className="timeline__lane-label timeline__lane-label--corner">
              <span className="timeline__clock" aria-hidden="true">{formatTimecode(playheadS)}</span>
            </div>
            <div className="timeline__track">
              <TimeRuler durationS={farthest} pxPerSecond={pxPerSecond} onSeek={props.onSeek} />
            </div>
          </div>
          {lanes.map((lane) => (
            <TimelineLane
              key={lane.id}
              projectId={props.projectId}
              lane={lane}
              label={laneLabels[lane.id]}
              pxPerSecond={pxPerSecond}
              selectedId={props.selectedId}
              selectedType={props.selectedType}
              onSelect={props.onSelect}
              onCommit={props.onCommit}
              onRelink={props.onRelink}
              onKeyDown={props.onKeyDown}
              onAdd={props.onAdd}
              overlaps={lane.id === "picture" ? overlaps : undefined}
            />
          ))}
          <div className="timeline__playhead" aria-hidden="true" />
        </div>
      </div>
    </div>
  );
}
