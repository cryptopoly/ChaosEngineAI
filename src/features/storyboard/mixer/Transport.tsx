/** The Mixer transport row: play / pause, stop, loop, the time readout and the zoom buttons (UX spec D.6). */
import { useTranslation } from "react-i18next";
import { LoopIcon, MinusIcon, PauseIcon, PlayIcon, PlusIcon, StopIcon } from "../../../components/icons/ActionIcons";
import { formatTimecode } from "./timelineGeometry";

export interface TransportProps {
  playing: boolean;
  timeS: number;
  durationS: number;
  loop: boolean;
  /** False until a preview has been rendered: there is nothing to play yet. */
  canPlay: boolean;
  canZoomIn: boolean;
  canZoomOut: boolean;
  onPlay: () => void;
  onStop: () => void;
  onLoop: () => void;
  onZoom: (direction: 1 | -1) => void;
}

export function Transport(props: TransportProps) {
  const { playing, timeS, durationS, loop, canPlay, canZoomIn, canZoomOut } = props;
  const { t } = useTranslation("studio");
  const playLabel = playing
    ? t("mixer.transport.pause", { defaultValue: "Pause" })
    : t("mixer.transport.play", { defaultValue: "Play" });
  const readout = `${formatTimecode(timeS)} / ${formatTimecode(durationS)}`;
  return (
    <div
      className="transport"
      role="group"
      aria-label={t("mixer.transport.label", { defaultValue: "Transport" })}
      title={t("mixer.transport.keys", {
        defaultValue: "Space or K: play / pause. J and L: 5 s back / forward. Left and Right: 1 s (Shift 10 s, Alt one frame). Home / End. + and -: zoom.",
      })}
    >
      <button
        type="button"
        className="secondary-button transport__button"
        disabled={!canPlay}
        aria-label={playLabel}
        title={canPlay ? playLabel : t("mixer.transport.needPreview", { defaultValue: "Render a preview to play the film." })}
        onClick={props.onPlay}
      >
        {playing ? <PauseIcon /> : <PlayIcon />}
      </button>
      <button
        type="button"
        className="secondary-button transport__button"
        disabled={!canPlay}
        aria-label={t("mixer.transport.stop", { defaultValue: "Stop" })}
        title={t("mixer.transport.stop", { defaultValue: "Stop" })}
        onClick={props.onStop}
      >
        <StopIcon />
      </button>
      <button
        type="button"
        className={`secondary-button transport__button${loop ? " active" : ""}`}
        aria-pressed={loop}
        aria-label={t("mixer.transport.loop", { defaultValue: "Loop" })}
        title={t("mixer.transport.loop", { defaultValue: "Loop" })}
        onClick={props.onLoop}
      >
        <LoopIcon />
      </button>
      <span className="transport__time" aria-label={t("mixer.transport.timeOf", { defaultValue: "{current} of {total}", current: formatTimecode(timeS), total: formatTimecode(durationS) })}>
        {readout}
      </span>
      <span className="transport__zoom">
        <button
          type="button"
          className="secondary-button transport__button"
          disabled={!canZoomOut}
          aria-label={t("mixer.transport.zoomOut", { defaultValue: "Zoom out" })}
          title={t("mixer.transport.zoomOut", { defaultValue: "Zoom out" })}
          onClick={() => props.onZoom(-1)}
        >
          <MinusIcon />
        </button>
        <button
          type="button"
          className="secondary-button transport__button"
          disabled={!canZoomIn}
          aria-label={t("mixer.transport.zoomIn", { defaultValue: "Zoom in" })}
          title={t("mixer.transport.zoomIn", { defaultValue: "Zoom in" })}
          onClick={() => props.onZoom(1)}
        >
          <PlusIcon />
        </button>
      </span>
    </div>
  );
}
