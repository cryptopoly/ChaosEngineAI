/** The side drawer for one shot: prompts, length, seed, trims, references, and importing footage. */
import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { CloseIcon } from "../../components/icons/ActionIcons";
import { RunEstimateStrip } from "../../components/RunEstimateStrip";
import { ConfirmDialog } from "./ConfirmDialog";
import type { ImageOutputArtifact, StoryboardProject, StoryboardShot, VideoOutputArtifact } from "../../types";
import { PathField } from "./PathField";
import type { RunOp } from "./ReferencesPanel";
import {
  draftFromShot,
  drawerEscapeAction,
  importKindOf,
  shotPatchFromDraft,
  toggleReference,
  trapTabTarget,
  type ShotDraft,
} from "./storyboardLogic";

const GALLERY_LIMIT = 8;
const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export interface ShotEditorDrawerProps {
  project: StoryboardProject;
  shot: StoryboardShot;
  index: number;
  imageGallery: ImageOutputArtifact[];
  videoGallery: VideoOutputArtifact[];
  busy: boolean;
  batchRunning: boolean;
  op: RunOp;
  onMake: (shotId: string) => void;
  onClose: () => void;
}

export function ShotEditorDrawer({
  project, shot, index, imageGallery, videoGallery, busy, batchRunning, op, onMake, onClose,
}: ShotEditorDrawerProps) {
  const { t } = useTranslation("studio");
  const [draft, setDraft] = useState<ShotDraft>(() => draftFromShot(shot));
  const [importError, setImportError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const opener = useRef<HTMLElement | null>(null);
  const root = useRef<HTMLElement>(null);
  const references = [...project.references.characters, ...project.references.places];
  const patch = shotPatchFromDraft(shot, draft);
  const dirty = Object.keys(patch).length > 0;
  const number = index + 1;

  // Focus goes into the drawer, and back to the card that opened it when it closes.
  useEffect(() => {
    opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    root.current?.focus();
    return () => {
      const back = opener.current;
      if (back && back.isConnected) back.focus();
    };
  }, []);

  /** Close, but ask first when there are edits that were not saved. */
  function requestClose() {
    if (dirty) setConfirming(true);
    else onClose();
  }
  const closeRef = useRef(requestClose);
  closeRef.current = requestClose;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const action = drawerEscapeAction({ key: event.key, isComposing: event.isComposing || event.keyCode === 229, dirty, confirming });
      if (action !== "none") closeRef.current();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [dirty, confirming]);

  /** Tab stays inside the drawer. */
  function trapTab(event: ReactKeyboardEvent<HTMLElement>) {
    if (event.key !== "Tab" || !root.current) return;
    const stops = Array.from(root.current.querySelectorAll<HTMLElement>(FOCUSABLE));
    const next = trapTabTarget(stops.length, stops.indexOf(document.activeElement as HTMLElement), event.shiftKey);
    if (next === null) return;
    event.preventDefault();
    stops[next]?.focus();
  }

  function set<K extends keyof ShotDraft>(key: K, value: ShotDraft[K]) {
    setDraft((current) => ({ ...current, [key]: value }));
  }

  async function save() {
    if (!dirty) return;
    await op("edit_shot", { shotId: shot.id, ...patch });
  }

  async function importFootage(args: { still?: string; clip?: string }) {
    setImportError(null);
    const ok = await op("import_take", { shotId: shot.id, ...args });
    if (!ok) setImportError(t("board.editor.importFailed", { defaultValue: "That file could not be imported." }));
  }

  async function importPath(path: string) {
    const kind = importKindOf(path);
    if (!kind) {
      setImportError(t("board.editor.importKind", { defaultValue: "Pick an image (png, jpg, webp) or a video (mp4, mov, webm)." }));
      return;
    }
    if (shot.status === "missing") {
      setImportError(null);
      await op("relink", { id: shot.id, path });
      return;
    }
    await importFootage(kind === "still" ? { still: path } : { clip: path });
  }

  const stillGallery = imageGallery.filter((item) => item.imagePath).slice(0, GALLERY_LIMIT);
  const clipGallery = videoGallery.filter((item) => item.videoPath).slice(0, GALLERY_LIMIT);

  return (
    <>
    <aside
      className="drawer shot-editor"
      ref={root}
      tabIndex={-1}
      role="dialog"
      aria-modal="true"
      aria-labelledby="shot-editor-title"
      onKeyDown={trapTab}
    >
      <div className="drawer__head">
        <h3 id="shot-editor-title">{t("board.editor.title", { defaultValue: "Shot {n}", n: number })}</h3>
        <button
          type="button"
          className="secondary-button icon-button"
          aria-label={t("storyboard.close", { defaultValue: "Close" })}
          title={t("storyboard.close", { defaultValue: "Close" })}
          onClick={requestClose}
        >
          <CloseIcon />
        </button>
      </div>

      <div className="drawer__body">
        <label className="control-stack">
          <span className="eyebrow">{t("board.editor.stillPrompt", { defaultValue: "Still prompt" })}</span>
          <textarea rows={4} maxLength={4000} value={draft.stillPrompt} onChange={(e) => set("stillPrompt", e.target.value)} />
        </label>
        <label className="control-stack">
          <span className="eyebrow">{t("board.editor.motionPrompt", { defaultValue: "Motion prompt" })}</span>
          <textarea rows={3} maxLength={4000} value={draft.motionPrompt} onChange={(e) => set("motionPrompt", e.target.value)} />
        </label>

        <div className="shot-editor__grid">
          <label className="control-stack">
            <span className="eyebrow">{t("board.editor.startFrom", { defaultValue: "Start from" })}</span>
            <select value={draft.startFrom} onChange={(e) => set("startFrom", e.target.value as ShotDraft["startFrom"])}>
              <option value="generate">{t("board.editor.startFromGenerate", { defaultValue: "A new still" })}</option>
              <option value="upload">{t("board.editor.startFromUpload", { defaultValue: "My image" })}</option>
              <option value="previous-shot">{t("board.editor.startFromPrevious", { defaultValue: "Last frame of the previous shot" })}</option>
            </select>
          </label>
          <label className="control-stack">
            <span className="eyebrow">{t("board.editor.length", { defaultValue: "Length (s)" })}</span>
            <input type="number" min={0.5} max={60} step={0.5} value={draft.seconds} onChange={(e) => set("seconds", e.target.value)} />
          </label>
          <label className="control-stack">
            <span className="eyebrow">{t("board.editor.seed", { defaultValue: "Seed" })}</span>
            <input
              type="number"
              min={0}
              step={1}
              value={draft.seed}
              placeholder={t("board.editor.seedRandom", { defaultValue: "Random" })}
              onChange={(e) => set("seed", e.target.value)}
            />
          </label>
          <label className="control-stack">
            <span className="eyebrow">{t("board.editor.nativeSound", { defaultValue: "Shot sound level (dB)" })}</span>
            <input type="number" min={-60} max={12} step={1} value={draft.nativeSoundDb} onChange={(e) => set("nativeSoundDb", e.target.value)} />
          </label>
          <label className="control-stack">
            <span className="eyebrow">{t("board.editor.trimIn", { defaultValue: "Trim in (s)" })}</span>
            <input type="number" min={0} step={0.1} value={draft.trimIn} onChange={(e) => set("trimIn", e.target.value)} />
          </label>
          <label className="control-stack">
            <span className="eyebrow">{t("board.editor.trimOut", { defaultValue: "Trim out (s)" })}</span>
            <input
              type="number"
              min={0}
              step={0.1}
              value={draft.trimOut}
              placeholder={t("board.editor.trimOutEnd", { defaultValue: "End of take" })}
              onChange={(e) => set("trimOut", e.target.value)}
            />
          </label>
          <label className="control-stack">
            <span className="eyebrow">{t("board.editor.speed", { defaultValue: "Speed" })}</span>
            <input type="number" min={0.25} max={4} step={0.25} value={draft.speed} onChange={(e) => set("speed", e.target.value)} />
          </label>
        </div>

        <fieldset className="shot-editor__refs">
          <legend className="eyebrow">{t("board.editor.references", { defaultValue: "References" })}</legend>
          {references.length === 0 ? (
            <p className="muted-text">
              {t("board.editor.noReferences", { defaultValue: "No references in this project yet. Add them in the References panel." })}
            </p>
          ) : (
            references.map((reference) => (
              <label key={reference.id} className="check-row">
                <input
                  type="checkbox"
                  checked={draft.referenceImages.includes(reference.id)}
                  onChange={() => set("referenceImages", toggleReference(draft.referenceImages, reference.id))}
                />
                {reference.name || reference.id}
              </label>
            ))
          )}
        </fieldset>

        <div className="shot-editor__actions">
          <button type="button" className="primary-button" disabled={!dirty || busy} onClick={() => void save()}>
            {t("board.editor.save", { defaultValue: "Save changes" })}
          </button>
          <button type="button" className="secondary-button" disabled={!dirty} onClick={() => setDraft(draftFromShot(shot))}>
            {t("board.editor.revert", { defaultValue: "Revert" })}
          </button>
        </div>

        <section className="shot-editor__import">
          <h4>{t("board.shot.menu.import", { defaultValue: "Import footage" })}</h4>
          <p className="muted-text">
            {shot.status === "missing"
              ? t("board.editor.relinkHint", { defaultValue: "Point the missing take at a new file." })
              : t("board.editor.importHint", { defaultValue: "Adds a new take from a file or from your galleries. Nothing is overwritten." })}
          </p>
          <PathField
            label={t("board.editor.importFile", { defaultValue: "Image or video file" })}
            buttonLabel={
              shot.status === "missing"
                ? t("board.shot.relink", { defaultValue: "Relink" })
                : t("board.editor.import", { defaultValue: "Import" })
            }
            extensions={["png", "jpg", "jpeg", "webp", "mp4", "mov", "m4v", "webm"]}
            disabled={busy}
            onSubmit={(path) => void importPath(path)}
          />
          {stillGallery.length > 0 ? (
            <div className="shot-editor__gallery">
              <span className="eyebrow">{t("board.editor.fromImageGallery", { defaultValue: "From the Image Gallery" })}</span>
              <div className="input-image-gallery">
                {stillGallery.map((item) => (
                  <button
                    key={item.artifactId}
                    type="button"
                    className="input-image-choice"
                    title={item.prompt}
                    disabled={busy}
                    onClick={() => void importFootage({ still: item.imagePath as string })}
                  >
                    <img src={item.previewUrl} alt={item.prompt} className="input-image-thumb" />
                  </button>
                ))}
              </div>
            </div>
          ) : null}
          {clipGallery.length > 0 ? (
            <div className="shot-editor__gallery">
              <span className="eyebrow">{t("board.editor.fromVideoGallery", { defaultValue: "From the Video Gallery" })}</span>
              <ul className="shot-editor__clips">
                {clipGallery.map((item) => (
                  <li key={item.artifactId}>
                    <span className="shot-editor__clip-name" title={item.prompt}>{item.prompt || item.modelName}</span>
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={busy}
                      onClick={() => void importFootage({ clip: item.videoPath as string })}
                    >
                      {t("board.editor.useClip", { defaultValue: "Use as clip" })}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {importError ? <p className="error-text">{importError}</p> : null}
        </section>
      </div>

      <div className="drawer__foot">
        <RunEstimateStrip
          seconds={null}
          peakGb={null}
          totalGb={null}
          diskMb={null}
          basis={null}
          risk="safe"
          action={
            <button
              type="button"
              className="primary-button"
              disabled={busy || batchRunning || shot.locked || dirty}
              title={
                shot.locked
                  ? t("board.shot.lockedNote", { defaultValue: "Locked shots are skipped by Make shots." })
                  : dirty
                    ? t("board.editor.saveFirst", { defaultValue: "Save your changes first." })
                    : undefined
              }
              onClick={() => onMake(shot.id)}
            >
              {t("board.makeOne", { defaultValue: "Make this shot" })}
            </button>
          }
        />
      </div>
    </aside>
    {confirming ? (
      <ConfirmDialog
        title={t("board.editor.discardTitle", { defaultValue: "Discard your changes?" })}
        body={<p>{t("board.editor.discardBody", { defaultValue: "This shot has edits that were not saved. Closing now throws them away." })}</p>}
        confirmLabel={t("board.editor.discard", { defaultValue: "Discard changes" })}
        danger
        onConfirm={() => {
          setConfirming(false);
          onClose();
        }}
        onCancel={() => setConfirming(false)}
      />
    ) : null}
    </>
  );
}
