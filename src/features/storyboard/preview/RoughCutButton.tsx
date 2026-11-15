/** "Play film": opens the rough-cut player (S9). Used on the Storyboard, Mixer and Export tabs. */
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { PlayIcon } from "../../../components/icons/ActionIcons";
import type { StoryboardProject, StoryboardTimeline } from "../../../types";
import { RoughCutPlayer } from "./RoughCutPlayer";

export interface RoughCutButtonProps {
  project: StoryboardProject;
  timeline: StoryboardTimeline;
  /** "primary" for the main action of a screen; secondary otherwise. */
  tone?: "primary" | "secondary";
}

export function RoughCutButton({ project, timeline, tone = "secondary" }: RoughCutButtonProps) {
  const { t } = useTranslation("studio");
  const [open, setOpen] = useState(false);
  const empty = timeline.shots.length === 0 || timeline.total <= 0;
  const title = empty
    ? t("board.preview.emptyHint", { defaultValue: "Add a shot to play the film." })
    : t("board.preview.buttonHint", { defaultValue: "Play the active takes in order with the Mixer's sound, without rendering anything." });
  return (
    <>
      <button
        type="button"
        className={`${tone === "primary" ? "primary-button" : "secondary-button"} rough-cut-button`}
        disabled={empty}
        title={title}
        aria-haspopup="dialog"
        onClick={() => setOpen(true)}
      >
        <PlayIcon /> {t("board.preview.button", { defaultValue: "Play film" })}
      </button>
      {open ? <RoughCutPlayer project={project} timeline={timeline} onClose={() => setOpen(false)} /> : null}
    </>
  );
}
