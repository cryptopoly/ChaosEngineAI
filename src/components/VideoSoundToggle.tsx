import { useTranslation } from "react-i18next";

interface VideoSoundToggleProps {
  muted: boolean;
  onToggle: () => void;
}

/** Speaker button laid over a clip that has sound: crossed out while muted, click to hear it. */
export function VideoSoundToggle({ muted, onToggle }: VideoSoundToggleProps) {
  const { t } = useTranslation("studio");
  const label = muted
    ? t("videoGallery.card.soundOn", { defaultValue: "Turn sound on" })
    : t("videoGallery.card.soundOff", { defaultValue: "Mute" });
  return (
    <button
      type="button"
      className="video-sound-toggle"
      aria-label={label}
      aria-pressed={!muted}
      title={label}
      onClick={onToggle}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M2.5 6.2h2.4L8 3.4v9.2L4.9 9.8H2.5z" />
        {muted ? (
          <path d="M10.6 6.2l3 3.6M13.6 6.2l-3 3.6" />
        ) : (
          <>
            <path d="M10.4 5.6a3.2 3.2 0 0 1 0 4.8" />
            <path d="M12.2 3.8a5.8 5.8 0 0 1 0 8.4" />
          </>
        )}
      </svg>
    </button>
  );
}
