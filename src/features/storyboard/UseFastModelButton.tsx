/**
 * "Use the fast model" under the project's still model (FU-188 item 4): one click writes the fastest installed
 * still model to the project through the ordinary `edit_project` op, so Undo works. Shown only when the project
 * uses a slower model (`fasterStillModel`).
 */
import { useTranslation } from "react-i18next";
import type { StoryboardStillModels } from "../../api/storyboard";
import { formatDuration } from "../../components/RunEstimateStrip";
import { fasterStillModel } from "./storyboardLogic";

type TFn = (key: string, options: Record<string, unknown>) => string;

export interface UseFastModelButtonProps {
  models: StoryboardStillModels | null;
  saved: string | null | undefined;
  disabled: boolean;
  onUse: (modelId: string) => void;
}

export function UseFastModelButton({ models, saved, disabled, onUse }: UseFastModelButtonProps) {
  const { t } = useTranslation("studio");
  const fast = models ? fasterStillModel(models.models, saved, models.fast) : null;
  if (!fast?.id || fast.seconds == null) return null;
  const id = fast.id;
  return (
    <div className="project-defaults__fast">
      <button type="button" className="secondary-button" disabled={disabled} onClick={() => onUse(id)}>
        {t("board.defaults.useFast", { defaultValue: "Use the fast model" })}
      </button>
      <span className="muted-text">
        {t("board.defaults.useFastHint", {
          defaultValue: "{name}, about {time} per still",
          name: fast.name,
          time: formatDuration(t as unknown as TFn, fast.seconds),
        })}
      </span>
    </div>
  );
}
