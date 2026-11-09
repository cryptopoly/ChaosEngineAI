import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { VoiceEntry } from "../../../types";
import { canPlaySample } from "./useVoiceSample";

export interface CardCost {
  text: string;
  tone: "ok" | "need" | "plain";
}

/** One selectable voice card: name, a line about it, the cost chip, and a sample button where a sample exists. */
export function VoiceCard({
  voice,
  title,
  detail,
  selected,
  disabled,
  onSelect,
  playing,
  loading,
  tabStop,
  onSample,
  cost,
}: {
  voice: VoiceEntry;
  title: string;
  detail: string;
  selected: boolean;
  disabled?: boolean;
  onSelect: () => void;
  playing: boolean;
  loading: boolean;
  /** The one card in the picker that is a Tab stop (arrow keys move between cards). */
  tabStop: boolean;
  onSample: () => void;
  /** Cost chip: memory and measured speed, or the download still needed. */
  cost?: CardCost | null;
}) {
  const { t } = useTranslation("studio");
  return (
    <div className={`voice-card${selected ? " voice-card--selected" : ""}${disabled ? " voice-card--disabled" : ""}`}>
      <button
        type="button"
        role="radio"
        aria-checked={selected}
        tabIndex={tabStop ? 0 : -1}
        data-voice-id={voice.id}
        data-autofocus={tabStop ? "" : undefined}
        className="voice-card__main"
        disabled={disabled}
        onClick={onSelect}
      >
        <span className="voice-card__title">{title}</span>
        <span className="voice-card__detail muted-text">{detail}</span>
        {cost ? <span className={`voice-card__cost voice-card__cost--${cost.tone}`}>{cost.text}</span> : null}
      </button>
      {canPlaySample(voice) ? (
        <button
          type="button"
          className="voice-card__play"
          aria-label={
            loading
              ? t("voice.speak.sampleLoading", { defaultValue: "Making a sample of {name}. Click to stop.", name: title })
              : t("voice.speak.sampleOf", { defaultValue: "Play a sample of {name}", name: title })
          }
          aria-busy={loading}
          onClick={onSample}
        >
          <SampleIcon playing={playing} loading={loading} />
        </button>
      ) : null}
    </div>
  );
}

/** The monochrome play / stop / making-a-sample icon on every sample button. */
export function SampleIcon({ playing, loading }: { playing: boolean; loading: boolean }) {
  if (loading) {
    return (
      <svg className="voice-card__spinner" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="8" cy="8" r="5.5" strokeOpacity="0.3" />
        <path d="M8 2.5a5.5 5.5 0 0 1 5.5 5.5" strokeLinecap="round" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="currentColor">
      {playing ? <rect x="3" y="3" width="10" height="10" rx="1" /> : <path d="M4 2.5v11l9-5.5z" />}
    </svg>
  );
}

/** A card that does something rather than select a voice (New cloned voice, Get more voices, Add Mandarin). */
export function ActionCard({ title, detail, children }: { title: string; detail: string; children: ReactNode }) {
  return (
    <div className="voice-card voice-card--action">
      <span className="voice-card__title">{title}</span>
      <span className="voice-card__detail muted-text">{detail}</span>
      {children}
    </div>
  );
}

/** A group heading with its one-line explanation. */
export function GroupHead({ title, note, children }: { title: string; note?: string; children?: ReactNode }) {
  return (
    <div className="voice-group__head">
      <h4 className="voice-group__title">{title}</h4>
      {note ? <span className="muted-text voice-group__note">{note}</span> : null}
      {children}
    </div>
  );
}
