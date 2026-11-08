/**
 * One character or place on the Cast tab: up to four pictures (each removable), the name, "What it looks
 * like", how many shots use it, and Remove. A picture is added through the picker, or by dropping or pasting
 * an image on the card. Fields save when they lose focus.
 */
import { useEffect, useRef, useState, type ClipboardEvent, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import { PlusIcon, TrashIcon, CloseIcon } from "../../components/icons/ActionIcons";
import type { StoryboardRefImage, StoryboardReference } from "../../types";
import { MAX_PICTURES, isSmallPicture, picturesFrom } from "./pickerLogic";
import { useProjectFileUrl } from "./ProjectMedia";

export interface ReferenceCardProps {
  projectId: string;
  reference: StoryboardReference;
  usedBy: number;
  disabled: boolean;
  /** Focus the name field on arrival (a card just added). */
  autoFocus?: boolean;
  onEdit: (patch: { name?: string; description?: string }) => void;
  onRemove: () => void;
  onRemovePicture: (index: number) => void;
  onOpenPicker: () => void;
  onFiles: (files: File[]) => void;
}

function Picture({ projectId, image, index, name, disabled, onRemove }: {
  projectId: string;
  image: StoryboardRefImage;
  index: number;
  name: string;
  disabled: boolean;
  onRemove: () => void;
}) {
  const { t } = useTranslation("studio");
  const url = useProjectFileUrl(projectId, image.path, image.hash, { thumb: 512 });
  const [small, setSmall] = useState(false);
  return (
    <figure className="cast-card__picture">
      {url ? (
        <img
          src={url}
          alt={t("cast.pictureAlt", { defaultValue: "{name}, picture {n}", name: name || t("cast.unnamed", { defaultValue: "Unnamed" }), n: index + 1 })}
          onLoad={(event) => setSmall(isSmallPicture(event.currentTarget.naturalWidth, event.currentTarget.naturalHeight))}
        />
      ) : (
        <span className="cast-card__picture-empty" aria-hidden="true" />
      )}
      <button
        type="button"
        className="cast-card__picture-remove"
        disabled={disabled}
        aria-label={t("cast.removePicture", { defaultValue: "Remove picture {n}", n: index + 1 })}
        title={t("cast.removePicture", { defaultValue: "Remove picture {n}", n: index + 1 })}
        onClick={onRemove}
      >
        <CloseIcon aria-hidden="true" />
      </button>
      {small ? <figcaption className="cast-card__small">{t("cast.small", { defaultValue: "Small picture; results may drift." })}</figcaption> : null}
    </figure>
  );
}

export function ReferenceCard({
  projectId, reference, usedBy, disabled, autoFocus, onEdit, onRemove, onRemovePicture, onOpenPicker, onFiles,
}: ReferenceCardProps) {
  const { t } = useTranslation("studio");
  const [name, setName] = useState(reference.name);
  const [description, setDescription] = useState(reference.description);
  const [over, setOver] = useState(false);
  useEffect(() => setName(reference.name), [reference.name]);
  useEffect(() => setDescription(reference.description), [reference.description]);
  const room = MAX_PICTURES - reference.images.length;
  // What was typed and not saved yet. The fields stay editable while an edit is on its way (the edits queue up),
  // and anything typed is saved before the picker opens, so nothing typed is lost.
  const saved = useRef({ name: reference.name, description: reference.description });
  useEffect(() => {
    saved.current = { name: reference.name, description: reference.description };
  }, [reference.name, reference.description]);
  const nameField = useRef<HTMLInputElement>(null);
  // A card just added takes the focus on its name, so typing goes straight in.
  useEffect(() => {
    if (autoFocus) nameField.current?.focus();
  }, [autoFocus]);
  function flush() {
    const patch: { name?: string; description?: string } = {};
    if (name !== saved.current.name) patch.name = name;
    if (description !== saved.current.description) patch.description = description;
    if (Object.keys(patch).length === 0) return;
    saved.current = { name, description };
    onEdit(patch);
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    setOver(false);
    const files = picturesFrom(event.dataTransfer?.files, room);
    if (files.length > 0) onFiles(files);
  }

  function onPaste(event: ClipboardEvent) {
    const target = event.target as HTMLElement;
    if (target.matches("input, textarea")) return; // pasting text into a field stays text
    const files = picturesFrom(event.clipboardData?.files, room);
    if (files.length === 0) return;
    event.preventDefault();
    onFiles(files);
  }

  const label = name || t("cast.unnamed", { defaultValue: "Unnamed" });
  return (
    <article
      className={`cast-card${over ? " cast-card--over" : ""}`}
      aria-label={label}
      data-ref-id={reference.id}
      onDragOver={(event) => {
        if (room <= 0) return;
        event.preventDefault();
        event.stopPropagation();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      onPaste={onPaste}
    >
      <div className="cast-card__pictures">
        {reference.images.map((image, index) => (
          <Picture
            key={`${image.path}-${index}`}
            projectId={projectId}
            image={image}
            index={index}
            name={name}
            disabled={disabled}
            onRemove={() => onRemovePicture(index)}
          />
        ))}
        {room > 0 ? (
          <button
            type="button"
            className="cast-card__add-picture"
            aria-label={t("cast.addPictureFor", { defaultValue: "Add a picture of {name}", name: label })}
            onMouseDown={flush} // stays enabled while the save lands, or the click after it would be lost
            onClick={() => {
              flush();
              onOpenPicker();
            }}
          >
            <PlusIcon aria-hidden="true" />
            <span>{t("cast.addPicture", { defaultValue: "Add picture" })}</span>
            <span className="muted-text">{t("cast.room", { defaultValue: "{n} of {max}", n: reference.images.length, max: MAX_PICTURES })}</span>
          </button>
        ) : null}
      </div>
      <div className="cast-card__fields">
        <input
          type="text"
          className="text-input"
          value={name}
          maxLength={200}
          ref={nameField}
          placeholder={t("board.references.name", { defaultValue: "Name" })}
          aria-label={t("board.references.name", { defaultValue: "Name" })}
          onChange={(event) => setName(event.target.value)}
          onBlur={flush}
          onKeyDown={(event) => {
            if (event.key === "Enter") flush();
          }}
        />
        <textarea
          className="text-area"
          rows={2}
          value={description}
          maxLength={4000}
          placeholder={t("board.references.description", { defaultValue: "What it looks like" })}
          aria-label={t("board.references.description", { defaultValue: "What it looks like" })}
          onChange={(event) => setDescription(event.target.value)}
          onBlur={flush}
        />
        <p className="muted-text cast-card__used">
          {t("cast.usedBy", { defaultValue: "{n, plural, =0 {Not used by a shot yet} one {Used by # shot} other {Used by # shots}}", n: usedBy })}
        </p>
      </div>
      <button
        type="button"
        className="secondary-button icon-button cast-card__remove"
        disabled={disabled}
        aria-label={t("cast.remove", { defaultValue: "Remove {name}", name: label })}
        title={t("cast.remove", { defaultValue: "Remove {name}", name: label })}
        onClick={onRemove}
      >
        <TrashIcon />
      </button>
    </article>
  );
}
