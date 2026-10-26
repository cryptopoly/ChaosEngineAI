import { useTranslation } from "react-i18next";
import { speechPlayer, useSpeechPlayer } from "./speechPlayer";
import { useVoice } from "./VoiceContext";

/** Speaker button for a message: reads it aloud, or stops it if it is playing. */
export function ReadAloudButton({ ownerId, text, className = "message-action-btn" }: { ownerId: string; text: string; className?: string }) {
  const { t } = useTranslation("chat");
  const { runtime, settings, openVoiceModels } = useVoice();
  const player = useSpeechPlayer();
  const active = player.ownerId === ownerId && player.status !== "idle" && player.status !== "error";
  const ready = Boolean(runtime?.ttsAvailable);

  const title = !ready
    ? t("voice.installToRead", { defaultValue: "Read aloud needs the voice pack. Click to install it." })
    : active
      ? t("voice.stopReading", { defaultValue: "Stop reading" })
      : t("voice.readAloud", { defaultValue: "Read aloud" });

  return (
    <button
      type="button"
      className={`${className}${active ? " message-action-btn--active" : ""}`}
      title={title}
      aria-label={title}
      aria-pressed={active}
      onClick={() => {
        if (!ready) openVoiceModels();
        else if (active) speechPlayer.stop();
        else void speechPlayer.play(ownerId, text, settings.ttsVoice, settings.speed);
      }}
    >
      {active ? (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <rect x="6" y="6" width="12" height="12" rx="1" />
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
          <path d="M15.5 8.5a5 5 0 0 1 0 7" />
          <path d="M19 5a10 10 0 0 1 0 14" />
        </svg>
      )}
    </button>
  );
}

/** Bar above the composer while something is being read aloud. */
export function SpeechPlayerBar() {
  const { t } = useTranslation("chat");
  const player = useSpeechPlayer();
  if (player.status === "idle") return null;

  if (player.status === "error") {
    return (
      <div className="speech-bar speech-bar--error" role="alert">
        <span className="speech-bar__text">{player.error}</span>
        <button type="button" className="secondary-button speech-bar__button" onClick={() => speechPlayer.stop()}>
          {t("voice.dismiss", { defaultValue: "Dismiss" })}
        </button>
      </div>
    );
  }

  return (
    <div className="speech-bar" aria-live="polite">
      <button
        type="button"
        className="secondary-button speech-bar__button"
        disabled={player.status === "loading"}
        onClick={() => speechPlayer.togglePause()}
      >
        {player.status === "paused" ? t("voice.resume", { defaultValue: "Resume" }) : t("voice.pause", { defaultValue: "Pause" })}
      </button>
      <button type="button" className="secondary-button speech-bar__button" onClick={() => speechPlayer.stop()}>
        {t("voice.stop", { defaultValue: "Stop" })}
      </button>
      <span className="speech-bar__text">
        {player.status === "loading"
          ? t("voice.preparing", { defaultValue: "Preparing speech…" })
          : player.sentence}
      </span>
      {player.total > 0 ? (
        <span className="speech-bar__count">
          {player.index + 1}/{player.total}
        </span>
      ) : null}
    </div>
  );
}
