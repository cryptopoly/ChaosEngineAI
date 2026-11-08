/**
 * The Cast picture picker (D-65): choose a file (its thumbnail shows at once), drop or paste one, pick from
 * the Image gallery, or use a still from one of this project's shots. A typed path stays as a small "Paste a
 * path" link for agents' outputs and remote folders. A modal dialog: focus stays inside and Escape closes it;
 * on a narrow screen it opens as a sheet from the bottom.
 */
import { useEffect, useRef, useState, type ClipboardEvent, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import { uploadStoryboardPicture } from "../../api";
import { useDialogFocus } from "../../hooks/useDialogFocus";
import type { ImageOutputArtifact, StoryboardProject } from "../../types";
import { nextTabIndex } from "./projectShellLogic";
import { PathField } from "./PathField";
import { PICKER_GRID_LIMIT, PICTURE_MIME, galleryChoices, picturesFrom, shotStillChoices } from "./pickerLogic";
import { ProjectImage } from "./ProjectMedia";

type Source = "file" | "gallery" | "shot";
const SOURCES: Source[] = ["file", "gallery", "shot"];

export interface ReferenceImagePickerProps {
  title: string;
  project: StoryboardProject;
  imageGallery: ImageOutputArtifact[];
  /** How many more pictures fit (four at most per character or place). */
  room: number;
  /** Adds one picture from a path, an upload handle or a project still; resolves true when it landed. */
  onPick: (source: string) => Promise<boolean>;
  onClose: () => void;
}

export function ReferenceImagePicker({ title, project, imageGallery, room, onPick, onClose }: ReferenceImagePickerProps) {
  const { t } = useTranslation("studio");
  const [limit, setLimit] = useState(PICKER_GRID_LIMIT);
  const allStills = shotStillChoices(project, Number.POSITIVE_INFINITY);
  const stills = allStills.slice(0, limit);
  const [source, setSource] = useState<Source>(stills.length > 0 ? "shot" : "file");
  const [query, setQuery] = useState("");
  const [pathOpen, setPathOpen] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const dialog = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const tabRow = useRef<HTMLDivElement>(null);
  useDialogFocus(dialog, { onEscape: onClose });

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  async function run(task: () => Promise<boolean>) {
    setWorking(true);
    setError(null);
    try {
      if (await task()) onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setWorking(false);
    }
  }

  async function sendFiles(files: File[]) {
    if (files.length === 0) {
      setError(t("picker.notPicture", { defaultValue: "Pick a PNG, JPEG or WebP picture." }));
      return;
    }
    setPreview(URL.createObjectURL(files[0]));
    await run(async () => {
      let ok = true;
      for (const file of files) {
        const { handle } = await uploadStoryboardPicture(file);
        ok = (await onPick(handle)) && ok;
      }
      return ok;
    });
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    void sendFiles(picturesFrom(event.dataTransfer?.files, room));
  }

  function onPaste(event: ClipboardEvent) {
    const files = picturesFrom(event.clipboardData?.files, room);
    if (files.length === 0) return;
    event.preventDefault();
    event.stopPropagation();
    void sendFiles(files);
  }

  /** "Paste from clipboard": reads a picture through the clipboard API where the app may; else says how to paste. */
  async function pasteFromClipboard() {
    setError(null);
    try {
      const items = await navigator.clipboard.read();
      const files: File[] = [];
      for (const item of items) {
        const type = item.types.find((kind) => (PICTURE_MIME as readonly string[]).includes(kind));
        if (type) files.push(new File([await item.getType(type)], `pasted.${type.split("/")[1]}`, { type }));
      }
      if (files.length === 0) {
        setError(t("picker.clipboardEmpty", { defaultValue: "The clipboard holds no picture. Copy one first, then try again." }));
        return;
      }
      await sendFiles(files.slice(0, room));
    } catch {
      setError(
        t("picker.clipboardBlocked", {
          defaultValue: "The clipboard could not be read here. Click the paste box and press Cmd+V (Ctrl+V on Windows and Linux).",
        }),
      );
    }
  }

  const labels: Record<Source, string> = {
    file: t("picker.source.computer", { defaultValue: "This computer" }),
    gallery: t("picker.source.gallery", { defaultValue: "Image gallery" }),
    shot: t("picker.source.shot", { defaultValue: "From a shot" }),
  };
  const allGallery = source === "gallery" ? galleryChoices(imageGallery, query, project.id, Number.POSITIVE_INFINITY) : [];
  const gallery = allGallery.slice(0, limit);
  const more = source === "gallery" ? allGallery.length > limit : source === "shot" ? allStills.length > limit : false;

  return (
    <div className="modal-overlay picker-overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div
        ref={dialog}
        className="modal-content picker-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onPaste={onPaste}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div className="modal-header">
          <h3>{title}</h3>
        </div>
        <div className="modal-body">
          <div
            className="picker-dialog__sources"
            role="tablist"
            aria-label={t("picker.sources", { defaultValue: "Where the picture comes from" })}
            ref={tabRow}
            onKeyDown={(event) => {
              const next = nextTabIndex(SOURCES.indexOf(source), SOURCES.length, event.key);
              if (next === null) return;
              event.preventDefault();
              setSource(SOURCES[next]);
              requestAnimationFrame(() => tabRow.current?.querySelector<HTMLElement>(`[data-source="${SOURCES[next]}"]`)?.focus());
            }}
          >
            {SOURCES.map((id) => (
              <button
                key={id}
                type="button"
                role="tab"
                data-source={id}
                aria-selected={source === id}
                tabIndex={source === id ? 0 : -1}
                className={`picker-dialog__source${source === id ? " active" : ""}`}
                onClick={() => setSource(id)}
              >
                {labels[id]}
              </button>
            ))}
          </div>

          <div role="tabpanel" aria-label={labels[source]} className="picker-dialog__panel">
            {source === "file" ? (
              <div className={`picker-drop${dragging ? " picker-drop--over" : ""}`}>
                {preview ? <img src={preview} alt="" className="picker-drop__preview" /> : null}
                <p className="muted-text">
                  {t("picker.dropHint", { defaultValue: "Drop a picture here, or paste one (PNG, JPEG or WebP)." })}
                </p>
                <button type="button" className="primary-button" disabled={working} onClick={() => fileInput.current?.click()} data-autofocus="">
                  {t("picker.chooseFile", { defaultValue: "Choose file" })}
                </button>
                {/* A text box takes a paste in every webview; a plain element only does in some. */}
                <input
                  type="text"
                  className="text-input picker-drop__paste"
                  value=""
                  readOnly={false}
                  onChange={() => undefined}
                  onPaste={onPaste}
                  placeholder={t("picker.pasteHere", { defaultValue: "Click here, then paste a picture (Cmd+V)" })}
                  aria-label={t("picker.pasteHereLabel", { defaultValue: "Paste a picture here" })}
                />
                <button type="button" className="secondary-button" disabled={working} onClick={() => void pasteFromClipboard()}>
                  {t("picker.pasteClipboard", { defaultValue: "Paste from clipboard" })}
                </button>
                <input
                  ref={fileInput}
                  type="file"
                  accept={PICTURE_MIME.join(",")}
                  multiple={room > 1}
                  className="sr-only"
                  tabIndex={-1}
                  aria-label={labels.file}
                  onChange={(event) => {
                    const files = picturesFrom(event.target.files, room);
                    event.target.value = "";
                    void sendFiles(files);
                  }}
                />
              </div>
            ) : null}

            {source === "gallery" ? (
              <>
                <input
                  type="search"
                  className="text-input"
                  value={query}
                  placeholder={t("picker.searchGallery", { defaultValue: "Search the Image gallery" })}
                  aria-label={t("picker.searchGallery", { defaultValue: "Search the Image gallery" })}
                  onChange={(event) => setQuery(event.target.value)}
                />
                {gallery.length === 0 ? (
                  <p className="muted-text">{t("picker.galleryEmpty", { defaultValue: "No saved images match." })}</p>
                ) : (
                  <ul className="picker-grid">
                    {gallery.map((item) => (
                      <li key={item.id}>
                        <button
                          type="button"
                          className="picker-grid__item"
                          disabled={working}
                          aria-label={t("picker.useImage", { defaultValue: "Use {name}", name: item.label })}
                          title={item.label}
                          onClick={() => void run(() => onPick(item.path))}
                        >
                          <img src={item.previewUrl} alt="" />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            ) : null}

            {source === "shot" ? (
              stills.length === 0 ? (
                <p className="muted-text">{t("picker.noStills", { defaultValue: "No shot has a still yet." })}</p>
              ) : (
                <ul className="picker-grid">
                  {stills.map((still) => {
                    const label = t("picker.shotTake", { defaultValue: "Shot {shot} · take {take}", shot: still.shotNumber, take: still.takeNumber });
                    return (
                      <li key={still.key}>
                        <button
                          type="button"
                          className="picker-grid__item"
                          disabled={working}
                          aria-label={t("picker.useImage", { defaultValue: "Use {name}", name: label })}
                          onClick={() => void run(() => onPick(still.path))}
                        >
                          <ProjectImage projectId={project.id} rel={still.path} version={still.takeId} className="picker-grid__img" thumb={256} />
                          <span className="picker-grid__label">{label}</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )
            ) : null}
          </div>

          {more ? (
            <button type="button" className="secondary-button" onClick={() => setLimit((n) => n + PICKER_GRID_LIMIT)}>
              {t("picker.showMore", { defaultValue: "Show more" })}
            </button>
          ) : null}
          {pathOpen ? (
            <PathField
              label={t("picker.pathLabel", { defaultValue: "Picture path" })}
              buttonLabel={t("picker.usePath", { defaultValue: "Use this picture" })}
              extensions={["png", "jpg", "jpeg", "webp"]}
              disabled={working}
              onSubmit={(path) => void run(() => onPick(path))}
            />
          ) : null}
          {working ? <p className="muted-text" role="status">{t("picker.adding", { defaultValue: "Adding the picture…" })}</p> : null}
          {error ? <p className="error-text" role="alert">{error}</p> : null}
        </div>
        <div className="modal-footer picker-dialog__footer">
          {!pathOpen ? (
            <button type="button" className="mixer__link" onClick={() => setPathOpen(true)}>
              {t("picker.pastePath", { defaultValue: "Paste a path instead" })}
            </button>
          ) : (
            <span />
          )}
          <button type="button" className="secondary-button" onClick={onClose}>
            {t("storyboard.cancel", { defaultValue: "Cancel" })}
          </button>
        </div>
      </div>
    </div>
  );
}
