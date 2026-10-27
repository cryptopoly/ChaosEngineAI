import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { deleteLibraryItems, getLibrary, getLibraryAudio, getLibraryItem } from "../../api";
import type { VoiceLibraryKind, VoiceLibrarySummary } from "../../types";
import { downloadBlob, formatClock } from "./audio";
import { exportTranscript, speakerNamer, type ExportFormat } from "./studio/transcript";

export interface VoiceLibraryTabProps {
  backendOnline: boolean;
  /** Open a transcript in Voice Studio's editor. */
  onOpenTranscript: (id: string) => void;
}

type Filter = "" | VoiceLibraryKind;

function safeName(title: string): string {
  return (title || "voice").replace(/[\\/:*?"<>|]+/g, "_").slice(0, 80);
}

function LibraryCard({
  item,
  selected,
  onSelect,
  onOpen,
  onDelete,
}: {
  item: VoiceLibrarySummary;
  selected: boolean;
  onSelect: (on: boolean) => void;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const { t } = useTranslation("studio");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(
    () => () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    },
    [audioUrl],
  );

  const run = async (work: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await work();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const play = () => run(async () => setAudioUrl(URL.createObjectURL(await getLibraryAudio(item.id))));

  const exportTranscriptAs = (format: ExportFormat) =>
    run(async () => {
      const full = await getLibraryItem(item.id);
      const name = speakerNamer(full.speakers ?? {}, (n) => t("voice.editor.speaker", { defaultValue: "Speaker {n}", n: n + 1 }));
      const out = exportTranscript(format, full, full.segments, name);
      downloadBlob(new Blob([out.text], { type: out.type }), `${safeName(full.title)}.${format}`);
    });

  const exportAudio = (format: "" | "mp3") =>
    run(async () => {
      const blob = await getLibraryAudio(item.id, format);
      downloadBlob(blob, `${safeName(item.title)}${format ? ".mp3" : item.audio ?? ".wav"}`);
    });

  const meta = [
    new Date(item.createdAt * 1000).toLocaleString(),
    item.durationS != null ? formatClock(item.durationS) : null,
    item.speakerCount > 0 ? t("voice.library.speakers", { defaultValue: "{n} speakers", n: item.speakerCount }) : null,
    item.kind === "speech" && item.voice ? item.voice : null,
  ].filter(Boolean);

  return (
    <article className={`voice-library-card${selected ? " voice-library-card--selected" : ""}`}>
      <div className="voice-library-card__head">
        <input
          type="checkbox"
          checked={selected}
          onChange={(e) => onSelect(e.target.checked)}
          aria-label={t("voice.library.select", { defaultValue: "Select {title}", title: item.title })}
        />
        <div className="voice-library-card__title">
          <strong>{item.title}</strong>
          <span className="muted-text">
            <span className="badge subtle">
              {item.kind === "transcript" ? t("voice.library.transcript", { defaultValue: "Transcript" }) : t("voice.library.speech", { defaultValue: "Speech" })}
            </span>{" "}
            {meta.join(" · ")}
          </span>
        </div>
      </div>
      {item.preview ? <p className="voice-library-card__preview">{item.preview}</p> : null}
      {audioUrl ? <audio controls autoPlay src={audioUrl} /> : null}
      {error ? (
        <p className="voice-error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="button-row">
        {item.kind === "transcript" ? (
          <button type="button" className="secondary-button" onClick={onOpen}>
            {t("voice.queue.open", { defaultValue: "Open" })}
          </button>
        ) : null}
        {item.audio && !audioUrl ? (
          <button type="button" className="secondary-button" disabled={busy} onClick={() => void play()}>
            {t("voice.library.play", { defaultValue: "Play" })}
          </button>
        ) : null}
        <select
          className="text-input voice-library-card__export"
          value=""
          disabled={busy}
          aria-label={t("voice.editor.export", { defaultValue: "Export" })}
          onChange={(e) => {
            const value = e.target.value;
            if (value === "audio") void exportAudio("");
            else if (value === "mp3") void exportAudio("mp3");
            else if (value) void exportTranscriptAs(value as ExportFormat);
          }}
        >
          <option value="">{busy ? t("voice.library.working", { defaultValue: "Working…" }) : t("voice.editor.export", { defaultValue: "Export" })}</option>
          {item.kind === "transcript"
            ? (["txt", "md", "srt", "vtt", "json"] as const).map((f) => (
                <option key={f} value={f}>
                  {f.toUpperCase()}
                </option>
              ))
            : null}
          {item.audio ? (
            <option value="audio">
              {t("voice.library.originalAudio", { defaultValue: "Audio ({ext})", ext: item.audio.replace(".", "").toUpperCase() })}
            </option>
          ) : null}
          {item.audio && item.audio !== ".mp3" ? <option value="mp3">MP3</option> : null}
        </select>
        <button type="button" className="secondary-button" onClick={onDelete}>
          {t("voice.library.delete", { defaultValue: "Delete" })}
        </button>
      </div>
    </article>
  );
}

export function VoiceLibraryTab({ backendOnline, onOpenTranscript }: VoiceLibraryTabProps) {
  const { t } = useTranslation("studio");
  const [items, setItems] = useState<VoiceLibrarySummary[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const request = useRef(0);

  const load = useCallback(async () => {
    const mine = ++request.current;
    try {
      const next = await getLibrary(query.trim(), filter);
      if (mine !== request.current) return;
      setItems(next);
      setSelected((current) => new Set([...current].filter((id) => next.some((i) => i.id === id))));
      setError(null);
    } catch (err) {
      if (mine === request.current) setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (mine === request.current) setLoaded(true);
    }
  }, [query, filter]);

  useEffect(() => {
    if (!backendOnline) return;
    const timer = window.setTimeout(() => void load(), 200);
    return () => window.clearTimeout(timer);
  }, [backendOnline, load]);

  const remove = async (ids: string[]) => {
    const message =
      ids.length === 1
        ? t("voice.library.confirmDelete", { defaultValue: "Delete this item and its audio? This cannot be undone." })
        : t("voice.library.confirmDeleteMany", { defaultValue: "Delete {n} items and their audio? This cannot be undone.", n: ids.length });
    if (!window.confirm(message)) return;
    try {
      const deleted = new Set(await deleteLibraryItems(ids));
      setItems((all) => all.filter((i) => !deleted.has(i.id)));
      setSelected((current) => new Set([...current].filter((id) => !deleted.has(id))));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const filters = useMemo(
    () =>
      [
        ["", t("voice.library.all", { defaultValue: "All" })],
        ["transcript", t("voice.library.transcripts", { defaultValue: "Transcripts" })],
        ["speech", t("voice.library.speechFilter", { defaultValue: "Speech" })],
      ] as const,
    [t],
  );

  return (
    <div className="content-grid discover-page voice-page">
      <Panel
        title={t("voice.library.title", { defaultValue: "Voice Library" })}
        subtitle={t("voice.library.subtitle", { defaultValue: "Transcripts and speech you have made, kept on this computer" })}
        className="span-2"
      >
        {!backendOnline ? (
          <div className="empty-state">
            <p className="muted-text">{t("voice.offline", { defaultValue: "Waiting for the backend…" })}</p>
          </div>
        ) : (
          <div className="voice-studio">
            <div className="voice-library__toolbar">
              <input
                type="search"
                className="text-input"
                value={query}
                placeholder={t("voice.library.search", { defaultValue: "Search titles and text" })}
                aria-label={t("voice.library.search", { defaultValue: "Search titles and text" })}
                onChange={(e) => setQuery(e.target.value)}
              />
              <div className="segmented" role="radiogroup" aria-label={t("voice.library.filter", { defaultValue: "Show" })}>
                {filters.map(([value, label]) => (
                  <button
                    key={value || "all"}
                    type="button"
                    role="radio"
                    aria-checked={filter === value}
                    className={filter === value ? "segment active" : "segment"}
                    onClick={() => setFilter(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {selected.size > 0 ? (
                <button type="button" className="secondary-button" onClick={() => void remove([...selected])}>
                  {t("voice.library.deleteSelected", { defaultValue: "Delete selected ({n})", n: selected.size })}
                </button>
              ) : null}
            </div>
            {error ? (
              <p className="voice-error" role="alert">
                {error}
              </p>
            ) : null}
            {!loaded ? (
              <p className="muted-text">{t("voice.library.loading", { defaultValue: "Loading…" })}</p>
            ) : items.length === 0 ? (
              <div className="empty-state">
                <p className="muted-text">
                  {query || filter
                    ? t("voice.library.noMatch", { defaultValue: "Nothing matches." })
                    : t("voice.library.empty", { defaultValue: "Nothing here yet. Transcripts made in Voice Studio are kept here, and so is speech you save." })}
                </p>
              </div>
            ) : (
              <div className="voice-library__grid">
                {items.map((item) => (
                  <LibraryCard
                    key={item.id}
                    item={item}
                    selected={selected.has(item.id)}
                    onSelect={(on) =>
                      setSelected((current) => {
                        const next = new Set(current);
                        if (on) next.add(item.id);
                        else next.delete(item.id);
                        return next;
                      })
                    }
                    onOpen={() => onOpenTranscript(item.id)}
                    onDelete={() => void remove([item.id])}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </Panel>
    </div>
  );
}
