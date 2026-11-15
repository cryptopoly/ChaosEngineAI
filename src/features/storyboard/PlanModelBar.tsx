/**
 * The chat model line at the top of the Plan tab (S8). With a model loaded: its name, and whether it reads the
 * style pictures or only their words. Without one: a plain sentence and a way to load one of the installed
 * chat models right here (the app stays on this tab while it loads).
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardPlanStatus } from "../../api/storyboardPlan";

export interface ChatModelChoice {
  key: string;
  label: string;
  detail?: string;
}

export interface PlanModelBarProps {
  chatModel: StoryboardPlanStatus["chatModel"] | null;
  /** Null while the status has not arrived. */
  known: boolean;
  choices: ChatModelChoice[];
  preferredKey?: string | null;
  loading: boolean;
  onLoad?: (key: string) => void;
}

export function PlanModelBar({ chatModel, known, choices, preferredKey, loading, onLoad }: PlanModelBarProps) {
  const { t } = useTranslation("studio");
  const [pick, setPick] = useState<string>(preferredKey && choices.some((c) => c.key === preferredKey) ? preferredKey : choices[0]?.key ?? "");
  useEffect(() => {
    if (!choices.some((c) => c.key === pick)) setPick(preferredKey && choices.some((c) => c.key === preferredKey) ? preferredKey : choices[0]?.key ?? "");
  }, [choices, pick, preferredKey]);

  if (!known) {
    return (
      <p className="plan-model plan-model--quiet muted-text" role="status">
        {t("board.plan.model.checking", { defaultValue: "Checking the chat model…" })}
      </p>
    );
  }
  if (chatModel) {
    return (
      <p className="plan-model" role="status">
        <span className="plan-model__dot" aria-hidden="true" />
        <span>
          {t("board.plan.model.using", { defaultValue: "Planning with {name}", name: chatModel.name })}
        </span>
        <span className="muted-text">
          {chatModel.vision
            ? t("board.plan.model.vision", { defaultValue: "· reads your pictures" })
            : t("board.plan.model.noVision", { defaultValue: "· reads your pictures as their words and colours" })}
        </span>
      </p>
    );
  }
  return (
    <div className="callout warning plan-model-missing" role="status">
      <p>
        <strong>{t("board.plan.model.noneTitle", { defaultValue: "No chat model is loaded." })}</strong>{" "}
        {t("board.plan.model.noneBody", { defaultValue: "Planning asks a chat model to write the shots. Load one to start; nothing renders until you say so." })}
      </p>
      {choices.length > 0 && onLoad ? (
        <div className="plan-model-missing__load">
          <label className="sr-only" htmlFor="plan-model-pick">
            {t("board.plan.model.pickLabel", { defaultValue: "Chat model to load" })}
          </label>
          <select
            id="plan-model-pick"
            className="text-input"
            value={pick}
            disabled={loading}
            onChange={(event) => setPick(event.target.value)}
          >
            {choices.map((choice) => (
              <option key={choice.key} value={choice.key}>
                {choice.detail ? `${choice.label} · ${choice.detail}` : choice.label}
              </option>
            ))}
          </select>
          <button type="button" className="primary-button" disabled={loading || !pick} onClick={() => onLoad(pick)}>
            {loading ? t("board.plan.model.loading", { defaultValue: "Loading…" }) : t("board.plan.model.load", { defaultValue: "Load" })}
          </button>
        </div>
      ) : (
        <p className="muted-text">
          {t("board.plan.model.noneInstalled", { defaultValue: "No chat model is installed yet. Download one under Chat > Models, then come back." })}
        </p>
      )}
    </div>
  );
}
