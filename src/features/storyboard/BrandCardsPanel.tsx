/**
 * The brand kit on the film (S15), in the Export tab: the logo in its corner, a title card at the start and an
 * end card at the end, with their text and length. The look comes from the brand kit on the Cast tab (typeface
 * and colour) and the style references (background); the small card here shows it. "Play film" and the export
 * both draw what is set here.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardProject } from "../../types";
import { cardLook, cardText } from "./preview/brandCards";
import { fontStack } from "./preview/RoughCutStage";
import type { SaveResult } from "./unsavedText";
import { useUnsavedText } from "./useUnsavedText";
import "./preview/brand.css";

export const CARD_SECONDS = [1, 1.5, 2, 3, 4, 5];

export interface BrandCardsPanelProps {
  project: StoryboardProject;
  disabled: boolean;
  onChange: (change: Partial<StoryboardProject["export"]>) => SaveResult;
  onOpenCast: () => void;
}

function CardTextField({ id, value, placeholder, label, disabled, onSave }: {
  id: string;
  value: string;
  placeholder: string;
  label: string;
  disabled: boolean;
  onSave: (text: string) => SaveResult;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  const unsaved = useUnsavedText(`cards:${id}`, {
    text: draft,
    dirty: draft.replace(/\s+/g, " ").trim() !== value,
    send: onSave,
    restore: setDraft,
  });
  return (
    <>
      <input
        type="text"
        className="text-input brand-cards__text"
        value={draft}
        placeholder={placeholder}
        maxLength={200}
        aria-label={label}
        disabled={disabled}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => {
          const next = draft.replace(/\s+/g, " ").trim();
          if (next !== value) unsaved.save(next);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") (event.target as HTMLInputElement).blur();
        }}
      />
      {unsaved.notice}
    </>
  );
}

export function BrandCardsPanel({ project, disabled, onChange, onOpenCast }: BrandCardsPanelProps) {
  const { t } = useTranslation("studio");
  const settings = project.export;
  const hasLogo = Boolean(project.brand?.logo);
  const look = cardLook(project);
  const seconds = settings.cardSeconds ?? 2;
  const anyCard = Boolean(settings.titleCard || settings.endCard);
  return (
    <section className="brand-cards" aria-labelledby="brand-cards-title">
      <div className="brand-cards__head">
        <h4 id="brand-cards-title">{t("board.brandFilm.title", { defaultValue: "Brand on the film" })}</h4>
        <button type="button" className="mixer__link" onClick={onOpenCast}>
          {t("board.brandFilm.editKit", { defaultValue: "Edit the brand kit" })}
        </button>
      </div>
      <fieldset className="brand-cards__options" disabled={disabled}>
        <legend className="sr-only">{t("board.brandFilm.title", { defaultValue: "Brand on the film" })}</legend>
        <label className="check-row" title={hasLogo ? undefined : t("board.brandFilm.noLogo", { defaultValue: "Add a logo in the brand kit on the Cast tab first." })}>
          <input
            type="checkbox"
            checked={hasLogo && settings.showLogo !== false}
            disabled={!hasLogo}
            onChange={(event) => onChange({ showLogo: event.target.checked })}
          />{" "}
          {hasLogo
            ? t("board.brandFilm.logo", { defaultValue: "Logo in the corner of every shot" })
            : t("board.brandFilm.logoNone", { defaultValue: "Logo in the corner (no logo in the brand kit yet)" })}
        </label>
        <div className="brand-cards__card-row">
          <label className="check-row">
            <input type="checkbox" checked={Boolean(settings.titleCard)} onChange={(event) => onChange({ titleCard: event.target.checked })} />{" "}
            {t("board.brandFilm.titleCard", { defaultValue: "Title card at the start" })}
          </label>
          <CardTextField
            id="title"
            value={settings.titleText ?? ""}
            placeholder={project.name}
            label={t("board.brandFilm.titleText", { defaultValue: "Title card text (empty: the project name)" })}
            disabled={disabled || !settings.titleCard}
            onSave={(text) => onChange({ titleText: text })}
          />
        </div>
        <div className="brand-cards__card-row">
          <label className="check-row">
            <input type="checkbox" checked={Boolean(settings.endCard)} onChange={(event) => onChange({ endCard: event.target.checked })} />{" "}
            {t("board.brandFilm.endCard", { defaultValue: "End card at the end" })}
          </label>
          <CardTextField
            id="end"
            value={settings.endText ?? ""}
            placeholder={project.name}
            label={t("board.brandFilm.endText", { defaultValue: "End card text (empty: the project name)" })}
            disabled={disabled || !settings.endCard}
            onSave={(text) => onChange({ endText: text })}
          />
        </div>
        <label className="brand-cards__seconds">
          {t("board.brandFilm.seconds", { defaultValue: "Each card lasts" })}{" "}
          <select
            className="text-input brand-cards__select"
            value={String(seconds)}
            disabled={disabled || !anyCard}
            onChange={(event) => onChange({ cardSeconds: Number(event.target.value) })}
          >
            {CARD_SECONDS.map((value) => (
              <option key={value} value={String(value)}>
                {t("board.brandFilm.secondsValue", { defaultValue: "{seconds} s", seconds: value })}
              </option>
            ))}
          </select>
        </label>
      </fieldset>
      {anyCard ? (
        <div className="brand-cards__sample" aria-hidden="true">
          <span
            className="brand-cards__swatch"
            style={{ background: look.background, color: look.colour, fontFamily: fontStack(look.font) }}
          >
            {cardText(project, settings.titleCard ? "title" : "end")}
          </span>
          <span className="muted-text">
            {t("board.brandFilm.lookNote", {
              defaultValue: "Typeface and text colour from the brand kit, background from the style pictures. Play film shows the cards too.",
            })}
          </span>
        </div>
      ) : null}
    </section>
  );
}
