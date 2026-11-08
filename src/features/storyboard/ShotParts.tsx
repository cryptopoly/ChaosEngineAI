/** Small parts of a shot card: the stale badge and the transition picker. */
import { useTranslation } from "react-i18next";
import type { StoryboardTransition } from "../../types";
import { transitionWith } from "./storyboardLogic";

export function StaleBadge({ reason, title }: { reason: string; title: string }) {
  return (
    <span className="stale-badge badge warning" title={title}>
      {reason}
    </span>
  );
}

export interface TransitionPickerProps {
  value: StoryboardTransition;
  onChange: (next: StoryboardTransition) => void;
  disabled?: boolean;
}

export function TransitionPicker({ value, onChange, disabled }: TransitionPickerProps) {
  const { t } = useTranslation("studio");
  return (
    <div className="transition-picker">
      <label className="transition-picker__type">
        <span className="sr-only">{t("board.shot.transitionLabel", { defaultValue: "Transition" })}</span>
        <select
          value={value.type}
          disabled={disabled}
          title={t("board.shot.transitionLabel", { defaultValue: "Transition" })}
          onChange={(event) => onChange(transitionWith(value, { type: event.target.value as StoryboardTransition["type"] }))}
        >
          <option value="cut">{t("board.shot.transitionType.cut", { defaultValue: "Cut" })}</option>
          <option value="crossfade">{t("board.shot.transitionType.crossfade", { defaultValue: "Crossfade" })}</option>
        </select>
      </label>
      {value.type === "crossfade" ? (
        <label className="transition-picker__seconds">
          <span className="sr-only">{t("board.shot.transitionSeconds", { defaultValue: "Crossfade length in seconds" })}</span>
          <input
            type="number"
            min={0.1}
            max={5}
            step={0.1}
            value={value.seconds}
            disabled={disabled}
            onChange={(event) => onChange(transitionWith(value, { seconds: event.target.valueAsNumber }))}
          />
          <span aria-hidden="true">s</span>
        </label>
      ) : null}
    </div>
  );
}
