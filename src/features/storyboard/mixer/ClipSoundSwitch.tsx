/**
 * The Shot sound lane's header when the clips carry a song (audit change 7): it says plainly that the clips' own
 * sound is muted under the song, and one switch lets it play. The switch changes every shot at once through
 * `edit_shot` (the caller does that); the per-shot level stays in the inspector.
 */
import { useTranslation } from "react-i18next";
import type { ClipSoundState } from "./clipSound";
import "./clipSound.css";

export interface ClipSoundSwitchProps {
  state: ClipSoundState;
  disabled?: boolean;
  /** true: let the clip sound play; false: mute it under the song. */
  onToggle: (on: boolean) => void;
}

export function ClipSoundSwitch({ state, disabled, onToggle }: ClipSoundSwitchProps) {
  const { t } = useTranslation("studio");
  const status =
    state === "muted"
      ? t("mixer.clipSound.muted", { defaultValue: "Muted under the song" })
      : state === "playing"
        ? t("mixer.clipSound.playing", { defaultValue: "Plays over the song" })
        : t("mixer.clipSound.mixed", { defaultValue: "Plays in some shots" });
  return (
    <span className="clip-sound">
      <span className="clip-sound__status">{status}</span>
      <label className="clip-sound__switch">
        <input
          type="checkbox"
          role="switch"
          checked={state !== "muted"}
          disabled={disabled}
          onChange={(event) => onToggle(event.target.checked)}
        />
        <span>{t("mixer.clipSound.switch", { defaultValue: "Let clip sound play" })}</span>
      </label>
    </span>
  );
}
