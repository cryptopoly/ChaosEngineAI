/**
 * Pictures dropped on or added to the freestyle board (S-M2): ask once what they become, then send each one. New shots
 * (the default) are added with that picture as their still, placed where they were dropped, side by side, one Undo
 * each (`board_add_shot`). Reference pictures become characters, places or styles (`add_reference`), named from the
 * file and laid out in the reference row. The backend reads the type from the bytes (PNG, JPEG or WebP).
 */
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { uploadStoryboardPicture } from "../../../api/storyboard";
import { useDialogFocus } from "../../../hooks/useDialogFocus";
import type { StoryboardReferenceKind } from "../../../types";
import type { UseStoryboardResult } from "../useStoryboard";
import { CARD_W, GAP, titleFromFile, type Pos } from "./boardLogic";

export type PictureUse = "shot" | StoryboardReferenceKind;

/** Send the pictures one by one; returns the names of those that failed, with the reason. */
export async function addPictures(story: UseStoryboardResult, files: File[], at: Pos, use: PictureUse): Promise<string[]> {
  const failed: string[] = [];
  let column = 0;
  for (const file of files) {
    try {
      const { handle } = await uploadStoryboardPicture(file);
      const title = titleFromFile(file.name);
      const ok =
        use === "shot"
          ? await story.op("board_add_shot", { still: handle, title, x: at.x + column * (CARD_W + GAP), y: at.y })
          : await story.op("add_reference", { kind: use, name: title || undefined, image: handle });
      if (ok) column += 1;
      else failed.push(file.name);
    } catch (err) {
      failed.push(`${file.name}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  return failed;
}

export interface BoardAddPicturesProps {
  story: UseStoryboardResult;
  files: File[];
  at: Pos;
  onDone: (failed: string[]) => void;
  onCancel: () => void;
}

export function BoardAddPictures({ story, files, at, onDone, onCancel }: BoardAddPicturesProps) {
  const { t } = useTranslation("studio");
  const dialog = useRef<HTMLDivElement>(null);
  const [use, setUse] = useState<PictureUse>("shot");
  const [sending, setSending] = useState(false);
  useDialogFocus(dialog, { onEscape: sending ? undefined : onCancel });
  const title = t("board.freestyle.addTitle", { defaultValue: "{n, plural, one {Add # picture} other {Add # pictures}}", n: files.length });
  const options: { value: PictureUse; label: string }[] = [
    { value: "shot", label: t("board.freestyle.asShots", { defaultValue: "As new shots (the picture is the shot's still)" }) },
    { value: "character", label: t("board.freestyle.asCharacter", { defaultValue: "As characters (reference pictures)" }) },
    { value: "place", label: t("board.freestyle.asPlace", { defaultValue: "As places (reference pictures)" }) },
    { value: "style", label: t("board.freestyle.asStyle", { defaultValue: "As a style (reference pictures)" }) },
  ];

  async function send() {
    setSending(true);
    try {
      onDone(await addPictures(story, files, at, use));
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="modal-overlay" onMouseDown={(event) => !sending && event.target === event.currentTarget && onCancel()}>
      <div ref={dialog} className="modal-content board-add" role="dialog" aria-modal="true" aria-label={title}>
        <div className="modal-header">
          <h3>{title}</h3>
        </div>
        <div className="modal-body">
          <p className="muted-text board-add__files">{files.map((file) => file.name).join(", ")}</p>
          <fieldset className="board-add__choices">
            <legend>{t("board.freestyle.addAs", { defaultValue: "What should they become?" })}</legend>
            {options.map((option) => (
              <label key={option.value} className="board-add__choice">
                <input type="radio" name="board-add-use" value={option.value} checked={use === option.value} onChange={() => setUse(option.value)} />
                {option.label}
              </label>
            ))}
          </fieldset>
        </div>
        <div className="modal-footer">
          <button type="button" className="secondary-button" disabled={sending} onClick={onCancel}>
            {t("storyboard.cancel", { defaultValue: "Cancel" })}
          </button>
          <button type="button" className="primary-button" disabled={sending} onClick={() => void send()} data-autofocus="">
            {sending ? t("board.freestyle.adding", { defaultValue: "Adding…" }) : t("board.freestyle.add", { defaultValue: "Add" })}
          </button>
        </div>
      </div>
    </div>
  );
}
