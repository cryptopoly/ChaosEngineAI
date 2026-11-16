/**
 * The attached song's sections as a ruler (S-M2): one part per section, as wide as its share of the song, shaded by
 * how loud it is, with its label and times. With `onRename` each label is a button that turns into a field (Enter
 * or leaving it saves through `edit_song`, Escape keeps the old label). Used by the song panel and along the
 * bottom of the freestyle board.
 */
import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardSongSection } from "../../../types";
import { clock, rulerParts } from "./songLogic";

export interface SongRulerProps {
  seconds: number;
  sections: StoryboardSongSection[];
  /** Left out, the labels are plain text. */
  onRename?: (index: number, label: string) => void;
  disabled?: boolean;
  compact?: boolean;
}

export function SongRuler({ seconds, sections, onRename, disabled, compact }: SongRulerProps) {
  const { t } = useTranslation("studio");
  const [editing, setEditing] = useState<number | null>(null);
  const [value, setValue] = useState("");
  const parts = rulerParts(sections, seconds);
  if (parts.length === 0) {
    return <p className="muted-text song-ruler__none">{t("board.song.noSections", { defaultValue: "No sections were found in this song." })}</p>;
  }

  function finish(save: boolean) {
    if (editing !== null && save && onRename) onRename(editing, value);
    setEditing(null);
  }

  return (
    <ol
      className={`song-ruler${compact ? " song-ruler--compact" : ""}`}
      aria-label={t("board.song.rulerLabel", { defaultValue: "Song sections" })}
    >
      {parts.map((part) => {
        const name = t("board.song.sectionName", { defaultValue: "Section {label}, {from} to {to}", label: part.label || "?", from: clock(part.start), to: clock(part.end) });
        const shade = part.energy == null ? 0.18 : 0.1 + part.energy * 0.32;
        return (
          <li
            key={part.index}
            className="song-ruler__part"
            style={{ left: `${part.left}%`, width: `${part.width}%`, ["--song-energy" as string]: String(shade) }}
            title={name}
          >
            {editing === part.index ? (
              <input
                className="text-input song-ruler__input"
                value={value}
                maxLength={40}
                autoFocus
                aria-label={t("board.song.renameSection", { defaultValue: "Name of the section from {from} to {to}", from: clock(part.start), to: clock(part.end) })}
                onChange={(event) => setValue(event.target.value)}
                onBlur={() => finish(true)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") finish(true);
                  if (event.key === "Escape") {
                    event.stopPropagation();
                    finish(false);
                  }
                }}
              />
            ) : onRename ? (
              <button
                type="button"
                className="song-ruler__label"
                disabled={disabled}
                aria-label={`${name}. ${t("board.song.renameHint", { defaultValue: "Rename" })}`}
                onClick={() => {
                  setValue(part.label);
                  setEditing(part.index);
                }}
              >
                {part.label || "?"}
              </button>
            ) : (
              <span className="song-ruler__label" aria-label={name}>{part.label || "?"}</span>
            )}
            {compact ? null : <span className="song-ruler__time" aria-hidden="true">{clock(part.start)}</span>}
          </li>
        );
      })}
    </ol>
  );
}
