/**
 * The shot count suggestion under the brief (FU-188 item 5): "Suggestion: an ad of 30 s is about 6 to 8 shots."
 * Rule based (`shotSuggestion`): the length the brief states, else the Length choice, else the usual length for
 * the kind of film. When the length came from the brief or the usual length and is one of the Length choices, a
 * button sets it. The model is free to choose differently.
 */
import { useTranslation } from "react-i18next";
import { LENGTH_CHOICES, shotSuggestion } from "./planLogic";

export interface PlanSuggestionProps {
  brief: string;
  target: number | null;
  /** The length formatted as the Length chips show it ("30 s", "1 min"). */
  lengthText: (seconds: number) => string;
  onUseLength: (seconds: number) => void;
}

export function PlanSuggestion({ brief, target, lengthText, onUseLength }: PlanSuggestionProps) {
  const { t } = useTranslation("studio");
  const suggestion = shotSuggestion(brief, target);
  if (!suggestion) return null;
  const { seconds, source, kind, low, high } = suggestion;
  const length = lengthText(seconds);
  const shots =
    low === high
      ? t("board.plan.suggest.shotsOne", { defaultValue: "{n, plural, one {# shot} other {# shots}}", n: low })
      : t("board.plan.suggest.shotsRange", { defaultValue: "{low} to {high} shots", low, high });
  const sentence =
    source === "typical"
      ? t("board.plan.suggest.typical", {
          defaultValue: "{kind, select, ad {An ad} trailer {A trailer} other {A short film}} usually runs about {length}, which is about {shots}.",
          kind,
          length,
          shots,
        })
      : t("board.plan.suggest.count", {
          defaultValue: "{kind, select, ad {An ad} trailer {A trailer} other {A film}} of {length} is about {shots}.",
          kind,
          length,
          shots,
        });
  const offerLength = source !== "target" && seconds !== target && (LENGTH_CHOICES as readonly number[]).includes(seconds);
  return (
    <p
      className="plan-suggest muted-text"
      title={t("board.plan.suggest.title", { defaultValue: "A rule of thumb from the length: shots of 4 to 5 s. The model may choose differently." })}
    >
      <strong>{t("board.plan.suggest.label", { defaultValue: "Suggestion:" })}</strong> {sentence}
      {offerLength ? (
        <>
          {" "}
          {/* Never disabled while busy: a click right after typing the brief blurs it first (its save makes the tab
              busy), and the op queue runs the length after the brief anyway. */}
          <button type="button" className="plan-link" onClick={() => onUseLength(seconds)}>
            {t("board.plan.suggest.useLength", { defaultValue: "Use {length}", length })}
          </button>
        </>
      ) : null}
    </p>
  );
}
