/**
 * The rough-cut player (S9): a dialog that plays the active takes in order with the Mixer's sound,
 * straight from the project's files, before any export. Shot markers on the scrubber show what each
 * shot will play (a clip, a draft, a held still, or nothing yet); the stage labels the shot playing.
 */
import { useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { useTranslation } from "react-i18next";
import { CloseIcon, PauseIcon, PlayIcon } from "../../../components/icons/ActionIcons";
import { useDialogFocus } from "../../../hooks/useDialogFocus";
import type { StoryboardProject, StoryboardQuality, StoryboardTimeline } from "../../../types";
import { draftTag, type TFn } from "../qualityText";
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
import { RoughCutStage } from "./RoughCutStage";
import { useRoughCut } from "./useRoughCut";

export interface RoughCutPlayerProps {
  project: StoryboardProject;
  timeline: StoryboardTimeline;
  /** `guide.quality`: the sizes the takes were made at, for the Draft tags. Left out, they are read from the takes. */
  quality?: StoryboardQuality | null;
  onClose: () => void;
}

function isButton(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  return !!el?.closest?.("button, input, select, textarea, a");
}

export function RoughCutPlayer({ project, timeline, quality, onClose }: RoughCutPlayerProps) {
  const { t } = useTranslation("studio");
  // A snapshot: the film as it was when the player opened.
  const [cut] = useState(() => buildRoughCut(project, timeline, { quality }));
  const player = useRoughCut(project.id, cut);
  const [captionsOn, setCaptionsOn] = useState(cut.captions.length > 0);
  const dialog = useRef<HTMLDivElement>(null);
  const track = useRef<HTMLDivElement>(null);
  useDialogFocus(dialog, { onEscape: onClose, initialFocus: "container" });

  const markers = useMemo(() => cutMarkers(cut), [cut]);
  const size = exportSize(project.defaults, project.defaults.exportPreset);
  const { timeS, playing, ready } = player;
  const current: CutPiece | undefined = cut.pieces[pieceIndexAt(cut, timeS)];
  const caption = captionsOn ? captionAt(cut, timeS) : null;
  const total = cut.total;
  const shotLabel = (piece: CutPiece) =>
    piece.media.kind === "card"
      ? piece.media.which === "title"
        ? t("board.preview.titleCard", { defaultValue: "Title card" })
        : t("board.preview.endCard", { defaultValue: "End card" })
      : piece.title.trim()
      ? t("board.preview.shotTitled", { defaultValue: "Shot {n} · {title}", n: piece.index + 1, title: piece.title })
      : t("board.preview.shot", { defaultValue: "Shot {n}", n: piece.index + 1 });
  const stateLabel = (state: MarkerState) =>
    state === "draft"
      ? t("board.preview.legendDraft", { defaultValue: "Draft take" })
      : state === "still"
        ? t("board.preview.legendStill", { defaultValue: "No clip yet: still held" })
        : state === "none"
          ? t("board.preview.legendNone", { defaultValue: "No picture yet" })
          : state === "card"
            ? t("board.preview.legendCard", { defaultValue: "Title or end card (brand kit)" })
          : t("board.preview.legendClip", { defaultValue: "Clip" });
  const states = [...new Set(markers.map((m) => m.state))];
  const target = quality?.target ?? null;
  // A draft marker names the size the take was made at: "Shot 2 · Draft take · 640 x 384".
  const markerTitle = (shotId: string, state: MarkerState) => {
    const piece = cut.pieces.find((p) => p.shotId === shotId) as CutPiece;
    const base = `${shotLabel(piece)} · ${stateLabel(state)}`;
    const size = state === "draft" ? draftTag(t as unknown as TFn, piece.quality, target)?.size : null;
    return size ? `${base} · ${size.width} x ${size.height}` : base;
  };

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
                shots: cut.pieces.filter((p) => p.media.kind !== "card").length,
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

        <RoughCutStage
          cut={cut}
          player={player}
          width={size.width}
          height={size.height}
          caption={caption}
          captionLook={project.brand?.captions ?? null}
          current={current}
          shotLabel={shotLabel}
          target={quality?.target ?? null}
        />

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
                title={markerTitle(marker.shotId, marker.state)}
              >
                {marker.state === "card" ? null : <span className="rough-cut__marker-n" aria-hidden="true">{marker.index + 1}</span>}
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
            {t("board.preview.draftNoteSized", {
              defaultValue: "{n, plural, one {# shot plays a draft take.} other {# shots play a draft take.}} Drafts are smaller than the film. Choose them for final and run the final pass for the full size.",
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
