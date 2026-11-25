/**
 * The card a Models page opens with (voice studio UX audit, change 7): the one speech model recommended for this
 * machine, with what it does, its download size and the GPU memory it uses, and one button. Every figure comes from
 * the catalogue; a figure that is not there is left out.
 */
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

export function SpeechPickCard({
  name,
  does,
  sizeGb,
  memory,
  notice,
  actions,
  error,
}: {
  name: string;
  /** What it makes, in plain words ("Natural voices, style"). */
  does: string;
  /** The download size in GB; null when nothing downloads (the quick voices come with the runtime). */
  sizeGb: string | null;
  /** "about 4.7 GB GPU memory", or null while the model has not been measured. */
  memory: string | null;
  notice?: string | null;
  actions: ReactNode;
  error?: ReactNode;
}) {
  const { t } = useTranslation("studio");
  const title = t("voice.models.pick.title", { defaultValue: "Recommended for this machine" });
  const facts = [
    sizeGb ? t("voice.models.pick.download", { defaultValue: "{gb} GB download", gb: sizeGb }) : null,
    memory,
  ].filter(Boolean);
  return (
    <section className="voice-install-card voice-models__pick" aria-label={title} data-speech-pick={name}>
      <h4 className="voice-install-card__title voice-models__pick-title">{title}</h4>
      <p className="voice-install-card__body">
        <strong>{name}</strong>: {does}
      </p>
      {facts.length > 0 ? <p className="muted-text voice-install-card__cost">{facts.join(", ")}.</p> : null}
      {notice ? <p className="muted-text">{notice}</p> : null}
      <div className="button-row">{actions}</div>
      {error}
    </section>
  );
}
