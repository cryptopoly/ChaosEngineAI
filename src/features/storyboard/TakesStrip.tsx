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
}

export function TakesStrip({ projectId, takes, activeId, onPick, disabled }: TakesStripProps) {
  const { t } = useTranslation("studio");
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const [focused, setFocused] = useState<number | null>(null);
  if (takes.length === 0) return null;
  const tabbable = rovingTakeIndex(takes.map((take) => take.id), activeId, focused);

  return (
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
            tabIndex={index === tabbable ? 0 : -1}
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
              <ProjectImage projectId={projectId} rel={take.still} version={take.hashes.still} className="takes-strip__thumb" />
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
}
