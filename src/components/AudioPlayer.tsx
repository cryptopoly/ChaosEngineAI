/**
 * Play / stop, loop, time and a waveform for one audio file. Only one
 * player plays at a time across the app (see `audioPlayback`). Keyboard:
 * Space on the focused play button toggles playback, L toggles loop.
 */
import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { formatClock } from "../features/voice/audio";
import { claimPlayback, releasePlayback } from "./audioPlayback";
import { LoopIcon, PlayIcon, StopIcon } from "./icons/ActionIcons";
import { WaveformCanvas } from "./WaveformCanvas";

export interface AudioPlayerProps {
  src: string;
  durationS: number;
  peaks?: number[];
  loop?: boolean;
  onLoopChange?: (loop: boolean) => void;
  /** Names the player for screen readers, e.g. the track title. */
  label: string;
}

const NO_PEAKS: number[] = [];

export function AudioPlayer({ src, durationS, peaks, loop = false, onLoopChange, label }: AudioPlayerProps) {
  const { t } = useTranslation("studio");
  const id = useId();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [loopOn, setLoopOn] = useState(loop);

  useEffect(() => setLoopOn(loop), [loop]);

  const stop = useCallback(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    setPlaying(false);
    setCurrent(0);
    releasePlayback(id);
  }, [id]);

  // Leaving the screen must not leave sound playing or an owner behind.
  useEffect(() => () => {
    audioRef.current?.pause();
    releasePlayback(id);
  }, [id]);

  async function toggle() {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) {
      stop();
      return;
    }
    claimPlayback(id, stop);
    try {
      await audio.play();
      setPlaying(true);
    } catch {
      releasePlayback(id);
      setPlaying(false);
    }
  }

  function toggleLoop() {
    const next = !loopOn;
    setLoopOn(next);
    onLoopChange?.(next);
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if ((event.key === "l" || event.key === "L") && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault();
      toggleLoop();
    }
  }

  const total = Math.max(0, durationS);
  const progress = total > 0 ? Math.min(1, current / total) : 0;
  const clock = `${formatClock(current)} / ${formatClock(total)}`;
  const spoken = t("audioPlayer.timeOf", { defaultValue: "{current} of {total}", current: formatClock(current), total: formatClock(total) });

  // The container only listens for the L shortcut; the buttons inside are the focus targets.
  return (
    <div className="audio-player" onKeyDown={onKeyDown}>
      <audio
        ref={audioRef}
        src={src}
        preload="none"
        loop={loopOn}
        onTimeUpdate={(event) => setCurrent(event.currentTarget.currentTime)}
        onEnded={() => {
          setPlaying(false);
          setCurrent(0);
          releasePlayback(id);
        }}
      />
      <div className="audio-player__controls">
        <button
          type="button"
          className="secondary-button audio-player__button"
          aria-label={
            playing
              ? t("musicStudio.result.stop", { defaultValue: "Stop" })
              : t("musicStudio.result.play", { defaultValue: "Play {title}, {seconds} seconds", title: label, seconds: Math.round(total) })
          }
          onClick={() => void toggle()}
        >
          {playing ? <StopIcon /> : <PlayIcon />}
        </button>
        <button
          type="button"
          className={`secondary-button audio-player__button${loopOn ? " active" : ""}`}
          aria-pressed={loopOn}
          aria-label={t("musicStudio.result.loop", { defaultValue: "Loop" })}
          title={t("musicStudio.result.loop", { defaultValue: "Loop" })}
          onClick={toggleLoop}
        >
          <LoopIcon />
        </button>
      </div>
      <div className="audio-player__wave">
        <WaveformCanvas peaks={peaks ?? NO_PEAKS} progress={progress} />
      </div>
      <span className="audio-player__time" aria-label={spoken}>
        {clock}
      </span>
    </div>
  );
}
