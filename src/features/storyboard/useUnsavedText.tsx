/**
 * The one hook every blur-save field of a project uses to keep typed text whose save did not land (unsavedText.ts).
 * `save(text)` sends through the field's own `send` and follows the answer; `notice` is the field's "Your text was
 * not saved" line with the text and "Put it back" (after a conflict) or "Retry" (the backend did not answer), which
 * puts the text back in the field and saves it again. A field that goes away with unsaved typing (the tab hides
 * while the backend is away) leaves its text in the project's store and finds it when it comes back.
 */
import { useCallback, useContext, useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { createUnsavedTextStore, trackSave, UnsavedTextContext, type SaveResult, type UnsavedEntry, type UnsavedTextStore } from "./unsavedText";
import "./unsaved.css";

export interface UnsavedTextOptions {
  /** What the field shows now. */
  text: string;
  /** True while the field holds typing that was not sent yet. */
  dirty: boolean;
  /** Sends `text` the way the field saves (an op, or the parent's callback); a promise of whether it landed. */
  send: (text: string) => SaveResult;
  /** Puts `text` back in the field. */
  restore: (text: string) => void;
  /** A store shared with a sibling that sends the saves itself (`useUnsavedStore`); default: the project's. */
  store?: UnsavedTextStore;
}

export interface UnsavedText {
  /** Send `text` with the field's `send` and keep it when the save is refused. */
  save: (text: string) => void;
  /** Follow a save the field sent itself (one op for two fields). */
  track: (text: string, result: SaveResult) => void;
  entry: UnsavedEntry | null;
  /** The notice to render under the field, or null. */
  notice: ReactNode;
}

/** The open project's store, or one of the caller's own outside a project (tests, a lone panel). */
export function useUnsavedStore(): UnsavedTextStore {
  const shared = useContext(UnsavedTextContext);
  const [own] = useState(createUnsavedTextStore);
  return shared ?? own;
}

export function useUnsavedText(key: string, options: UnsavedTextOptions): UnsavedText {
  const fallback = useUnsavedStore();
  const store = options.store ?? fallback;
  const subscribe = useCallback((listener: () => void) => store.subscribe(listener), [store]);
  const read = useCallback(() => store.get(key), [store, key]);
  const entry = useSyncExternalStore(subscribe, read, read);

  const live = useRef(options);
  live.current = options;
  // Unsent typing in a field that goes away (no blur came first): keep it for when the field is back.
  useEffect(
    () => () => {
      const { dirty, text } = live.current;
      if (dirty && !store.get(key)) store.set(key, { text, kind: "failed" });
    },
    [store, key],
  );

  const track = useCallback((text: string, result: SaveResult) => trackSave(store, key, text, result), [store, key]);
  const save = useCallback((text: string) => track(text, live.current.send(text)), [track]);
  const putBack = useCallback(
    (text: string) => {
      live.current.restore(text);
      save(text);
    },
    [save],
  );
  const dismiss = useCallback(() => store.set(key, null), [store, key]);

  return {
    save,
    track,
    entry,
    notice: entry ? <UnsavedTextNotice entry={entry} onPutBack={() => putBack(entry.text)} onDismiss={dismiss} /> : null,
  };
}

export function UnsavedTextNotice({ entry, onPutBack, onDismiss }: { entry: UnsavedEntry; onPutBack: () => void; onDismiss: () => void }) {
  const { t } = useTranslation("studio");
  const conflict = entry.kind === "conflict";
  return (
    <div className="unsaved-text" role="status">
      <p className="unsaved-text__head">
        <strong>{t("board.unsaved.title", { defaultValue: "Your text was not saved" })}</strong>{" "}
        <span className="muted-text">
          {conflict
            ? t("board.unsaved.conflict", { defaultValue: "It changed elsewhere first; the field shows the latest." })
            : t("board.unsaved.failed", { defaultValue: "The backend did not answer." })}
        </span>
      </p>
      <blockquote className="unsaved-text__text">{entry.text || t("board.unsaved.empty", { defaultValue: "(empty)" })}</blockquote>
      <div className="unsaved-text__actions">
        <button type="button" className="secondary-button" onClick={onPutBack}>
          {conflict ? t("board.unsaved.putBack", { defaultValue: "Put it back" }) : t("board.unsaved.retry", { defaultValue: "Retry" })}
        </button>
        <button type="button" className="unsaved-text__dismiss" onClick={onDismiss}>
          {conflict ? t("board.unsaved.keepLatest", { defaultValue: "Keep the latest" }) : t("board.unsaved.discard", { defaultValue: "Discard my text" })}
        </button>
      </div>
    </div>
  );
}
