/**
 * The takes of one shot as small thumbnails. One radio group with a roving tabindex: Tab enters on the
 * active take, arrows / Home / End move focus, Enter or Space picks it.
 */
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardTake } from "../../types";
import { ProjectImage, ProjectVideo } from "./ProjectMedia";
import { nextTakeIndex, rovingTakeIndex } from "./storyboardLogic";

export interface TakesStripProps {
  projectId: string;
  takes: StoryboardTake[];
  activeId: string | null;
  onPick: (takeId: string) => void;
  disabled?: boolean;
  /** Opens the side-by-side compare (S10); shown from two takes up. */
  onCompare?: () => void;
  /** False while the card is not the shot list's current card: no take is a Tab stop then. */
  tabStops?: boolean;
}

export function TakesStrip({ projectId, takes, activeId, onPick, disabled, onCompare, tabStops = true }: TakesStripProps) {
  const { t } = useTranslation("studio");
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const [focused, setFocused] = useState<number | null>(null);
  if (takes.length === 0) return null;
  const tabbable = rovingTakeIndex(takes.map((take) => take.id), activeId, focused);

  const strip = (
    <div
      className="takes-strip"
      role="radiogroup"
      aria-label={t("board.takes.label", { defaultValue: "Takes" })}
    >
      {takes.map((take, index) => {
        const active = take.id === activeId;
        return (
          <button
            key={take.id}
            ref={(node) => {
              buttons.current[index] = node;
            }}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={tabStops && index === tabbable ? 0 : -1}
            disabled={disabled}
            className={`takes-strip__take${active ? " takes-strip__take--active" : ""}`}
            title={active ? t("board.takes.active", { defaultValue: "Active take" }) : t("board.takes.pick", { defaultValue: "Use this take" })}
            aria-label={`${t("board.takes.take", { defaultValue: "Take {n}", n: index + 1 })}${active ? ` · ${t("board.takes.active", { defaultValue: "Active take" })}` : ""}`}
            onFocus={() => setFocused(index)}
            onKeyDown={(event) => {
              const next = nextTakeIndex(index, takes.length, event.key);
              if (next == null) return;
              event.preventDefault();
              buttons.current[next]?.focus();
            }}
            onClick={() => {
              if (!active) onPick(take.id);
            }}
          >
            {take.still ? (
              <ProjectImage projectId={projectId} rel={take.still} version={take.hashes.still} className="takes-strip__thumb" thumb={128} />
            ) : take.clip ? (
              <ProjectVideo projectId={projectId} rel={take.clip} version={take.hashes.clip} className="takes-strip__thumb" />
            ) : (
              <span className="takes-strip__thumb project-media-empty" />
            )}
            <span className="takes-strip__index">{index + 1}</span>
          </button>
        );
      })}
    </div>
  );
  if (!onCompare || takes.length < 2) return strip;
  return (
    <div className="takes-strip__row">
      {strip}
      <button
        type="button"
        className="secondary-button takes-strip__compare"
        title={t("board.compare.openTitle", { defaultValue: "Two takes side by side, with what made each one." })}
        onClick={onCompare}
      >
        {t("board.compare.short", { defaultValue: "Compare" })}
      </button>
    </div>
  );
}
