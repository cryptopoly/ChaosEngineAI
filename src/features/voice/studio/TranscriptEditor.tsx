import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { updateLibraryItem } from "../../../api";
import type { VoiceLibraryItem, VoiceLibraryPatch, TranscriptSegment } from "../../../types";
import { downloadBlob, formatClock } from "../audio";
import {
  HAND_OFF_PROMPTS,
  editSegment,
  exportTranscript,
  joinedText,
  segmentAt,
  speakerNamer,
  speakersIn,
  toPlainText,
  wordAt,
  type ExportFormat,
  type HandOff,
} from "./transcript";

export interface TranscriptEditorProps {
  item: VoiceLibraryItem;
  /** Object URL of the item's audio, or null when it has none. */
  audioUrl: string | null;
  onSaved: (item: VoiceLibraryItem) => void;
  onSendToChat: (text: string) => void;
}

const EXPORTS: ExportFormat[] = ["txt", "md", "srt", "vtt", "json"];
const HAND_OFFS: HandOff[] = ["summary", "actions", "notes"];
const SAVE_DELAY_MS = 800;

/**
 * A transcript synced to its audio: click a time or word to play from there,
 * edit a line in place (its times stay), name the speakers, export, or hand
 * the text to chat. Changes save to the Library on their own.
 */
export function TranscriptEditor({ item, audioUrl, onSaved, onSendToChat }: TranscriptEditorProps) {
  const { t } = useTranslation("studio");
  const [title, setTitle] = useState(item.title);
  const [segments, setSegments] = useState<TranscriptSegment[]>(item.segments);
  const [speakers, setSpeakers] = useState<Record<string, string>>(item.speakers ?? {});
  const [time, setTime] = useState(0);
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [saveState, setSaveState] = useState<"saved" | "pending" | "error">("saved");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const pending = useRef<VoiceLibraryPatch>({});
  const timer = useRef<number | null>(null);
  const itemId = item.id;

  const name = useMemo(
    () => speakerNamer(speakers, (n) => t("voice.editor.speaker", { defaultValue: "Speaker {n}", n: n + 1 })),
    [speakers, t],
  );
  const speakerList = useMemo(() => speakersIn(segments), [segments]);
  const active = audioUrl ? segmentAt(segments, time) : -1;
  const activeWord = active >= 0 ? wordAt(segments[active], time) : -1;

  const flush = useCallback(async () => {
    timer.current = null;
    const patch = pending.current;
    pending.current = {};
    if (Object.keys(patch).length === 0) return;
    try {
      onSaved(await updateLibraryItem(itemId, patch));
      setSaveState("saved");
    } catch {
      pending.current = { ...patch, ...pending.current };
      setSaveState("error");
    }
  }, [itemId, onSaved]);

  const queueSave = useCallback(
    (patch: VoiceLibraryPatch) => {
      pending.current = { ...pending.current, ...patch };
      setSaveState("pending");
      if (timer.current != null) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => void flush(), SAVE_DELAY_MS);
    },
    [flush],
  );

  // Save what is pending when leaving the item.
  useEffect(
    () => () => {
      if (timer.current != null) {
        window.clearTimeout(timer.current);
        void flush();
      }
    },
    [flush],
  );

  const seek = (seconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, seconds);
    void audio.play().catch(() => {});
  };

  const commitEdit = () => {
    if (editing == null) return;
    const next = segments.map((s, i) => (i === editing ? editSegment(s, draft) : s));
    setEditing(null);
    if (next[editing] === segments[editing]) return;
    setSegments(next);
    queueSave({ segments: next, text: joinedText(next, item.language) });
  };

  const exportAs = (format: ExportFormat) => {
    const out = exportTranscript(format, { ...item, title, speakers }, segments, name);
    const base = (title || "transcript").replace(/[\\/:*?"<>|]+/g, "_").slice(0, 80);
    downloadBlob(new Blob([out.text], { type: out.type }), `${base}.${format}`);
  };

  const handOff = (kind: HandOff) => {
    // i18n-extract: voice.handoff.summaryPrompt
    // i18n-extract: voice.handoff.actionsPrompt
    // i18n-extract: voice.handoff.notesPrompt
    const prompt = t(`voice.handoff.${kind}Prompt`, { defaultValue: HAND_OFF_PROMPTS[kind] });
    onSendToChat(`${prompt}\n\n"""\n${toPlainText(segments, name, item.text)}\n"""`);
  };

  return (
    <div className="voice-editor">
      <div className="voice-editor__head">
        <input
          className="text-input voice-editor__title"
          value={title}
          maxLength={200}
          aria-label={t("voice.editor.title", { defaultValue: "Title" })}
          onChange={(e) => {
            setTitle(e.target.value);
            queueSave({ title: e.target.value.trim() || item.title });
          }}
        />
        <span className="muted-text voice-editor__save" aria-live="polite">
          {saveState === "pending"
            ? t("voice.editor.saving", { defaultValue: "Saving…" })
            : saveState === "error"
              ? t("voice.editor.saveFailed", { defaultValue: "Not saved" })
              : t("voice.editor.savedToLibrary", { defaultValue: "Saved to Library" })}
        </span>
      </div>

      {audioUrl ? (
        // eslint-disable-next-line jsx-a11y/media-has-caption -- the transcript below is the caption
        <audio
          ref={audioRef}
          className="voice-editor__player"
          controls
          src={audioUrl}
          onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
          onSeeked={(e) => setTime(e.currentTarget.currentTime)}
        />
      ) : null}

      {speakerList.length > 0 ? (
        <div className="voice-editor__speakers">
          {speakerList.map((n) => (
            <label key={n} className="voice-editor__speaker">
              <span className={`voice-editor__dot voice-editor__dot--${n % 6}`} aria-hidden="true" />
              <input
                className="text-input"
                value={speakers[String(n)] ?? ""}
                maxLength={60}
                placeholder={t("voice.editor.speaker", { defaultValue: "Speaker {n}", n: n + 1 })}
                aria-label={t("voice.editor.renameSpeaker", { defaultValue: "Name for speaker {n}", n: n + 1 })}
                onChange={(e) => {
                  const next = { ...speakers, [String(n)]: e.target.value };
                  setSpeakers(next);
                  queueSave({ speakers: next });
                }}
              />
            </label>
          ))}
        </div>
      ) : null}

      {segments.length === 0 ? (
        <p className="voice-editor__plain">{item.text || t("voice.editor.empty", { defaultValue: "No speech was found." })}</p>
      ) : (
        <ol className="voice-editor__lines">
          {segments.map((segment, index) => (
            <li key={`${segment.start}-${index}`} className={`voice-editor__line${index === active ? " voice-editor__line--active" : ""}`}>
              <button type="button" className="voice-editor__time" onClick={() => seek(segment.start)} disabled={!audioUrl}>
                {formatClock(segment.start)}
              </button>
              <div className="voice-editor__body">
                {segment.speaker != null && (index === 0 || segments[index - 1].speaker !== segment.speaker) ? (
                  <span className={`voice-editor__who voice-editor__who--${segment.speaker % 6}`}>{name(segment.speaker)}</span>
                ) : null}
                {editing === index ? (
                  <textarea
                    className="text-input voice-editor__input"
                    value={draft}
                    autoFocus
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={commitEdit}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                        e.preventDefault();
                        commitEdit();
                      } else if (e.key === "Escape") {
                        setEditing(null);
                      }
                    }}
                  />
                ) : (
                  <p
                    className="voice-editor__text"
                    title={t("voice.editor.editHint", { defaultValue: "Double-click to edit" })}
                    onDoubleClick={() => {
                      setEditing(index);
                      setDraft(segment.text);
                    }}
                  >
                    {segment.words?.length && audioUrl
                      ? segment.words.map((word, w) => (
                          <span
                            key={w}
                            className={`voice-editor__word${index === active && w === activeWord ? " voice-editor__word--active" : ""}`}
                            onClick={() => seek(word.start)}
                          >
                            {word.text}{" "}
                          </span>
                        ))
                      : segment.text}
                  </p>
                )}
              </div>
              {editing !== index ? (
                <button
                  type="button"
                  className="voice-editor__edit"
                  aria-label={t("voice.editor.edit", { defaultValue: "Edit line" })}
                  title={t("voice.editor.edit", { defaultValue: "Edit line" })}
                  onClick={() => {
                    setEditing(index);
                    setDraft(segment.text);
                  }}
                >
                  <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
                    <path d="M4 20h4L19 9l-4-4L4 16v4z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
                  </svg>
                </button>
              ) : null}
            </li>
          ))}
        </ol>
      )}

      <div className="voice-editor__actions">
        <div className="button-row">
          {HAND_OFFS.map((kind) => (
            <button key={kind} type="button" className="secondary-button" onClick={() => handOff(kind)}>
              {kind === "summary"
                ? t("voice.handoff.summary", { defaultValue: "Summarise in chat" })
                : kind === "actions"
                  ? t("voice.handoff.actions", { defaultValue: "Action items" })
                  : t("voice.handoff.notes", { defaultValue: "Meeting notes" })}
            </button>
          ))}
          <button
            type="button"
            className="secondary-button"
            onClick={() => void navigator.clipboard.writeText(toPlainText(segments, name, item.text))}
          >
            {t("voice.copy", { defaultValue: "Copy" })}
          </button>
        </div>
        <div className="button-row voice-editor__exports" role="group" aria-label={t("voice.editor.export", { defaultValue: "Export" })}>
          <span className="muted-text">{t("voice.editor.export", { defaultValue: "Export" })}</span>
          {EXPORTS.map((format) => (
            <button key={format} type="button" className="secondary-button" disabled={format !== "txt" && format !== "md" && segments.length === 0} onClick={() => exportAs(format)}>
              {format.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
