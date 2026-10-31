import { useTranslation } from "react-i18next";

/** Marks a clip that carries a sound track. Monochrome speaker, inherits the text colour. */
export function SoundBadge() {
  const { t } = useTranslation("studio");
  return (
    <span
      className="badge subtle"
      title={t("videoStudio.sound.badgeTooltip", {
        defaultValue: "This clip has sound. Turn it on with the volume control on the player.",
      })}
      style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
    >
      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M2.5 6.2h2.4L8 3.4v9.2L4.9 9.8H2.5z" />
        <path d="M10.4 5.6a3.2 3.2 0 0 1 0 4.8" />
        <path d="M12.2 3.8a5.8 5.8 0 0 1 0 8.4" />
      </svg>
      {t("videoStudio.sound.badge", { defaultValue: "Sound" })}
    </span>
  );
}
