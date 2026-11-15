/**
 * The rough-cut player (S9): a dialog that plays the active takes in order with the Mixer's sound,
 * straight from the project's files, before any export. Shot markers on the scrubber show what each
 * shot will play (a clip, a draft, a held still, or nothing yet); the stage labels the shot playing.
 */
import { useMemo, useRef, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { useTranslation } from "react-i18next";
import { CloseIcon, PauseIcon, PlayIcon } from "../../../components/icons/ActionIcons";
import { useDialogFocus } from "../../../hooks/useDialogFocus";
import type { StoryboardProject, StoryboardTimeline } from "../../../types";
import { exportSize } from "../mixer/mixerLogic";
import { formatTimecode } from "../mixer/timelineGeometry";
import {
  buildRoughCut,
  captionAt,
  cutMarkers,
  pieceIndexAt,
  previewKeyAction,
  shotJump,
  type CutPiece,
  type MarkerState,
} from "./roughCutLogic";
import { useRoughCut } from "./useRoughCut";

export interface RoughCutPlayerProps {
  project: StoryboardProject;
  timeline: StoryboardTimeline;
  onClose: () => void;
}

function isButton(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  return !!el?.closest?.("button, input, select, textarea, a");
}

export function RoughCutPlayer({ project, timeline, onClose }: RoughCutPlayerProps) {
  const { t } = useTranslation("studio");
  // A snapshot: the film as it was when the player opened.
  const [cut] = useState(() => buildRoughCut(project, timeline));
  const player = useRoughCut(project.id, cut);
  const [captionsOn, setCaptionsOn] = useState(cut.captions.length > 0);
  const dialog = useRef<HTMLDivElement>(null);
  const track = useRef<HTMLDivElement>(null);
  useDialogFocus(dialog, { onEscape: onClose, initialFocus: "container" });

  const markers = useMemo(() => cutMarkers(cut), [cut]);
  const size = exportSize(project.defaults, project.defaults.exportPreset);
  const stageStyle = { "--cut-w": size.width, "--cut-h": size.height } as CSSProperties; // stage shape: the export frame
  const { timeS, playing, ready } = player;
  const current: CutPiece | undefined = cut.pieces[pieceIndexAt(cut, timeS)];
  const caption = captionsOn ? captionAt(cut, timeS) : null;
  const total = cut.total;
  const shotLabel = (piece: CutPiece) =>
    piece.title.trim()
      ? t("board.preview.shotTitled", { defaultValue: "Shot {n} · {title}", n: piece.index + 1, title: piece.title })
      : t("board.preview.shot", { defaultValue: "Shot {n}", n: piece.index + 1 });
  const stateLabel = (state: MarkerState) =>
    state === "draft"
      ? t("board.preview.legendDraft", { defaultValue: "Draft take" })
      : state === "still"
        ? t("board.preview.legendStill", { defaultValue: "No clip yet: still held" })
        : state === "none"
          ? t("board.preview.legendNone", { defaultValue: "No picture yet" })
          : t("board.preview.legendClip", { defaultValue: "Clip" });
  const states = [...new Set(markers.map((m) => m.state))];

  function seekFromPointer(event: ReactPointerEvent<HTMLDivElement>) {
    const rect = track.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    player.seek(((event.clientX - rect.left) / rect.width) * total);
  }

  function onKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if ((event.key === " " || event.key === "Enter") && isButton(event.target)) return;
    const action = previewKeyAction(event.key, event.shiftKey);
    if (!action) return;
    event.preventDefault();
    switch (action.type) {
      case "toggle-play":
        player.toggle();
        break;
      case "seek":
        player.seek(player.now() + action.deltaS);
        break;
      case "seek-to":
        player.seek(action.where === "start" ? 0 : total);
        break;
      case "shot":
        player.seek(shotJump(cut, player.now(), action.direction));
        break;
      case "captions":
        if (cut.captions.length) setCaptionsOn((on) => !on);
        break;
      default:
    }
  }

  const percent = total > 0 ? Math.min(100, (timeS / total) * 100) : 0;
  const timeText = t("board.preview.timeOf", {
    defaultValue: "{current} of {total}",
    current: formatTimecode(timeS),
    total: formatTimecode(total),
  });
  const title = t("board.preview.title", { defaultValue: "Rough cut" });

  return (
    <div className="modal-overlay rough-cut-overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div
        ref={dialog}
        className="modal-content rough-cut"
        role="dialog"
        aria-modal="true"
        aria-labelledby="rough-cut-title"
        aria-describedby="rough-cut-keys"
        tabIndex={-1}
        onKeyDown={onKeyDown}
        data-drift-p50-ms={Math.round(player.drift.p50S * 1000)}
        data-drift-p95-ms={Math.round(player.drift.p95S * 1000)}
        data-drift-max-ms={Math.round(player.drift.maxS * 1000)}
        data-drift-bias-ms={Math.round(player.drift.biasS * 1000)}
        data-drift-seeks={player.drift.seeks}
        data-latency-ms={Math.round(player.drift.latencyS * 1000)}
      >
        <div className="rough-cut__head">
          <div>
            <h3 id="rough-cut-title">{title}</h3>
            <p className="muted-text">
              {t("board.preview.subtitle", {
                defaultValue: "{name} · {shots, plural, one {# shot} other {# shots}} · {seconds} s · played from the takes, nothing is rendered",
                name: project.name,
                shots: cut.pieces.length,
                seconds: Math.round(total * 10) / 10,
              })}
            </p>
          </div>
          <button
            type="button"
            className="secondary-button rough-cut__close"
            onClick={onClose}
            aria-label={t("board.preview.close", { defaultValue: "Close the rough cut" })}
            title={t("board.preview.close", { defaultValue: "Close the rough cut" })}
          >
            <CloseIcon />
          </button>
        </div>

        <div className="rough-cut__stage" style={stageStyle}>
          {cut.pieces.map((piece) => {
            const label = shotLabel(piece);
            if (piece.media.kind === "clip") {
              const url = player.urlOf(piece.media.clip);
              return (
                <video
                  key={piece.shotId}
                  ref={(el) => player.registerLayer(piece.shotId, el)}
                  className="rough-cut__layer"
                  src={url ?? undefined}
                  muted
                  playsInline
                  preload="auto"
                  aria-hidden="true"
                />
              );
            }
            if (piece.media.kind === "still") {
              const url = player.urlOf(piece.media.still);
              return (
                <div key={piece.shotId} ref={(el) => player.registerLayer(piece.shotId, el)} className="rough-cut__layer rough-cut__layer--still">
                  {url ? <img src={url} alt="" /> : null}
                  <span className="rough-cut__hold">
                    {t("board.preview.heldStill", { defaultValue: "No clip yet · still held for {seconds} s", seconds: Math.round(piece.duration * 10) / 10 })}
                  </span>
                </div>
              );
            }
            return (
              <div key={piece.shotId} ref={(el) => player.registerLayer(piece.shotId, el)} className="rough-cut__layer rough-cut__layer--card">
                <strong>{label}</strong>
                <span>{t("board.preview.noPicture", { defaultValue: "No picture yet · {seconds} s", seconds: Math.round(piece.duration * 10) / 10 })}</span>
              </div>
            );
          })}
          {current ? (
            <div className="rough-cut__badges">
              <span className="rough-cut__chip">{shotLabel(current)}</span>
              {current.draft ? <span className="rough-cut__chip rough-cut__chip--draft">{t("board.preview.draft", { defaultValue: "Draft" })}</span> : null}
            </div>
          ) : null}
          {caption ? (
            <p className="rough-cut__caption" aria-live="off">
              {caption}
            </p>
          ) : null}
          {!ready ? (
            <div className="rough-cut__loading" role="status">
              <span>
                {t("board.preview.loading", { defaultValue: "Loading the film: {done} of {total} files", done: player.load.done, total: player.load.total })}
              </span>
              <span className="rough-cut__loading-bar" aria-hidden="true">
                <span style={{ width: `${player.load.total ? (player.load.done / player.load.total) * 100 : 0}%` }} />
              </span>
            </div>
          ) : null}
        </div>

        <div className="rough-cut__transport">
          <button
            type="button"
            className="primary-button rough-cut__play"
            disabled={!ready || total <= 0}
            onClick={player.toggle}
            aria-label={playing ? t("board.preview.pause", { defaultValue: "Pause" }) : t("board.preview.play", { defaultValue: "Play" })}
            title={playing ? t("board.preview.pause", { defaultValue: "Pause" }) : t("board.preview.play", { defaultValue: "Play" })}
          >
            {playing ? <PauseIcon /> : <PlayIcon />}
          </button>
          <span className="rough-cut__time" aria-hidden="true">
            {formatTimecode(timeS)} / {formatTimecode(total)}
          </span>
          <div
            ref={track}
            className="rough-cut__scrubber"
            role="slider"
            tabIndex={0}
            aria-label={t("board.preview.scrubber", { defaultValue: "Position in the film" })}
            aria-valuemin={0}
            aria-valuemax={Math.round(total * 10) / 10}
            aria-valuenow={Math.round(timeS * 10) / 10}
            aria-valuetext={current ? `${timeText}, ${shotLabel(current)}` : timeText}
            aria-disabled={!ready}
            onPointerDown={(event) => {
              if (!ready) return;
              event.currentTarget.setPointerCapture(event.pointerId);
              seekFromPointer(event);
            }}
            onPointerMove={(event) => {
              if (ready && event.currentTarget.hasPointerCapture(event.pointerId)) seekFromPointer(event);
            }}
          >
            {markers.map((marker) => (
              <span
                key={marker.shotId}
                className={`rough-cut__marker rough-cut__marker--${marker.state}${current?.shotId === marker.shotId ? " is-current" : ""}`}
                style={{ left: `${marker.left}%`, width: `${marker.width}%` }}
                title={`${shotLabel(cut.pieces.find((p) => p.shotId === marker.shotId) as CutPiece)} · ${stateLabel(marker.state)}`}
              >
                <span className="rough-cut__marker-n" aria-hidden="true">{marker.index + 1}</span>
              </span>
            ))}
            <span className="rough-cut__played" style={{ width: `${percent}%` }} aria-hidden="true" />
            <span className="rough-cut__head-line" style={{ left: `${percent}%` }} aria-hidden="true" />
          </div>
          <button
            type="button"
            className={`secondary-button rough-cut__cc${captionsOn ? " active" : ""}`}
            disabled={cut.captions.length === 0}
            aria-pressed={captionsOn}
            onClick={() => setCaptionsOn((on) => !on)}
            title={
              cut.captions.length
                ? t("board.captions.toggle", { defaultValue: "Show captions (C)" })
                : t("board.captions.none", { defaultValue: "No captions yet. Make them from the voice lane in the Mixer." })
            }
          >
            {t("board.captions.short", { defaultValue: "CC" })}
          </button>
        </div>

        <div className="rough-cut__foot">
          <ul className="rough-cut__legend" aria-label={t("board.preview.legend", { defaultValue: "What the markers mean" })}>
            {states.map((state) => (
              <li key={state}>
                <span className={`rough-cut__swatch rough-cut__marker--${state}`} aria-hidden="true" />
                {stateLabel(state)}
              </li>
            ))}
          </ul>
          <p id="rough-cut-keys" className="muted-text rough-cut__keys">
            {t("board.preview.keys", {
              defaultValue: "Space or K: play / pause · J and L: 5 s · arrows: 1 s (Shift 10 s) · Page Up / Down: shot · Home / End · C: captions · Esc: close",
            })}
          </p>
        </div>
        {cut.missing > 0 ? (
          <p className="muted-text rough-cut__note">
            {t("board.preview.missingNote", {
              defaultValue: "{n, plural, one {# shot has no clip yet. It holds its still here (or a card) for its planned length; the export leaves it out.} other {# shots have no clip yet. They hold their still here (or a card) for their planned length; the export leaves them out.}}",
              n: cut.missing,
            })}
          </p>
        ) : null}
        {cut.drafts > 0 ? (
          <p className="muted-text rough-cut__note">
            {t("board.preview.draftNote", {
              defaultValue: "{n, plural, one {# shot plays a draft take.} other {# shots play a draft take.}} Run the final pass on kept shots for full quality.",
              n: cut.drafts,
            })}
          </p>
        ) : null}
        {player.load.failed.length > 0 ? (
          <p className="callout warning" role="alert">
            {t("board.preview.failed", {
              defaultValue: "{n, plural, one {# file could not be read; its part plays blank or silent.} other {# files could not be read; their parts play blank or silent.}}",
              n: player.load.failed.length,
            })}
          </p>
        ) : null}
        {player.silent ? (
          <p className="muted-text rough-cut__note">{t("board.preview.noSound", { defaultValue: "This window cannot play sound, so the rough cut plays without it." })}</p>
        ) : null}
        <p className="sr-only" role="status" aria-live="polite">
          {ready && !playing && current ? `${timeText}, ${shotLabel(current)}` : ""}
        </p>
      </div>
    </div>
  );
}
