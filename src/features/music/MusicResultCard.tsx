/**
 * One finished track in the Studio: player with waveform and loop, the measured line, and the actions.
 * Save to Library is automatic (the backend saves every track), so the actions are Open in Library, Again
 * (same settings, new seed), Vary (seed plus one) and Send to storyboard.
 */
import { useTranslation } from "react-i18next";
import { formatDuration } from "../../components/RunEstimateStrip";
import { AudioPlayer } from "../../components/AudioPlayer";
import { SendToStoryboardMenu } from "../../components/SendToStoryboardMenu";
import type { UseSendToStoryboard } from "../storyboard/useSendToStoryboard";
import type { MusicResult } from "../../types";
import { clock, itemFilePath, megabytes } from "./musicLogic";
import { useTrackAudio } from "./useTrackAudio";

export interface MusicResultCardProps {
  result: MusicResult;
  libraryDir: string | null;
  storyboard: UseSendToStoryboard;
  /** A run is in flight: Again and Vary wait. */
  busy: boolean;
  onOpenLibrary: () => void;
  onAgain: (result: MusicResult) => void;
  onVary: (result: MusicResult) => void;
}

export function MusicResultCard({ result, libraryDir, storyboard, busy, onOpenLibrary, onAgain, onVary }: MusicResultCardProps) {
  const { t } = useTranslation("studio");
  const { item } = result;
  const duration = item.durationS ?? 0;
  const audio = useTrackAudio(item.id, duration);
  const size = megabytes(item.sizeBytes);
  const time = result.jobSeconds != null ? formatDuration(t as unknown as Parameters<typeof formatDuration>[0], result.jobSeconds) : null;
  const line = [
    time ? t("musicStudio.result.doneIn", { defaultValue: "Done in {time}", time }) : t("musicStudio.result.done", { defaultValue: "Done" }),
    size,
    t("musicStudio.result.seed", { defaultValue: "seed {seed}", seed: result.seed }),
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <article className="music-result" aria-label={item.title}>
      <div className="music-result__head">
        <strong className="music-result__title">{item.title}</strong>
        <span className="muted-text">{clock(duration)}</span>
      </div>
      {audio.url ? (
        <AudioPlayer src={audio.url} durationS={duration} peaks={audio.peaks} label={item.title} />
      ) : (
        <p className="muted-text music-result__loading" role="status">
          {audio.failed
            ? t("musicStudio.result.audioFailed", { defaultValue: "The audio file could not be loaded." })
            : t("musicStudio.result.audioLoading", { defaultValue: "Loading the audio…" })}
        </p>
      )}
      <p className="muted-text music-result__line">{line}</p>
      <div className="button-row music-result__actions">
        <button type="button" className="secondary-button" onClick={onOpenLibrary}>
          {t("musicStudio.result.openLibrary", { defaultValue: "Open in Library" })}
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={busy}
          title={t("musicStudio.result.againTitle", { defaultValue: "Same settings, a new random seed." })}
          onClick={() => onAgain(result)}
        >
          {t("musicStudio.result.again", { defaultValue: "Again" })}
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={busy}
          title={t("musicStudio.result.varyTitle", { defaultValue: "Same settings, the next seed ({seed}).", seed: result.seed + 1 })}
          onClick={() => onVary(result)}
        >
          {t("musicStudio.result.vary", { defaultValue: "Vary" })}
        </button>
        <SendToStoryboardMenu
          item={{ kind: "music", path: itemFilePath(libraryDir, item), label: item.title, seconds: duration || null }}
          projects={storyboard.projects}
          loaded={storyboard.loaded}
          busy={storyboard.busy}
          enabled={storyboard.enabled}
          notice={storyboard.notice}
          onOpen={() => void storyboard.loadProjects()}
          onSend={(projectId, target) => storyboard.send({ kind: "music", path: itemFilePath(libraryDir, item), label: item.title, seconds: duration || null }, projectId, target)}
          onDismissNotice={storyboard.dismissNotice}
        />
      </div>
    </article>
  );
}
