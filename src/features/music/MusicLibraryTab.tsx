/**
 * Music > Library: every track the Studio made, as cards (or rows) with a player, an inline rename, tags,
 * the style prompt, Send to storyboard, "Make another like this" and delete. Filter by text and tag, sort by
 * newest, oldest or longest. Cards are draggable (`application/x-chaos-media`); the keyboard path is the
 * Send to storyboard menu.
 */
import { useEffect, useMemo, useState, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import type { TabRequest } from "../../constants";
import { AudioPlayer } from "../../components/AudioPlayer";
import { CloseIcon, PlusIcon } from "../../components/icons/ActionIcons";
import { MediaLibraryCard } from "../../components/MediaLibraryCard";
import { Panel } from "../../components/Panel";
import { SendToStoryboardMenu } from "../../components/SendToStoryboardMenu";
import type { UseMusicState } from "../../hooks/useMusicState";
import type { MusicLibraryItem } from "../../types";
import { ConfirmDialog } from "../storyboard/ConfirmDialog";
import { useSendToStoryboard } from "../storyboard/useSendToStoryboard";
import {
  DEFAULT_LIBRARY_FILTER,
  allTags,
  clock,
  diskLabel,
  filterLibrary,
  itemFilePath,
  libraryCounts,
  megabytes,
  withTag,
  type LibraryFilter,
  type LibrarySort,
} from "./musicLogic";
import { useTrackAudio } from "./useTrackAudio";

export interface MusicLibraryTabProps {
  music: UseMusicState;
  backendOnline: boolean;
  onTabChange: (request: TabRequest) => void;
}

/**
 * A card fetches and decodes its audio only when the user shows interest (hover, focus or a press), not when it
 * scrolls into view, so a long library opens without reading every file.
 */
export function LibraryPlayer({ item }: { item: MusicLibraryItem }) {
  const { t } = useTranslation("studio");
  const [engaged, setEngaged] = useState(false);
  const audio = useTrackAudio(item.id, item.durationS ?? 0, engaged);
  const engage = () => setEngaged(true);
  return (
    <div className="music-library__player" onPointerEnter={engage} onFocusCapture={engage}>
      {audio.url ? (
        <AudioPlayer src={audio.url} durationS={item.durationS ?? 0} peaks={audio.peaks} label={item.title} />
      ) : audio.failed ? (
        <p className="muted-text">{t("musicStudio.result.audioFailed", { defaultValue: "The audio file could not be loaded." })}</p>
      ) : engaged ? (
        <p className="muted-text">{t("musicStudio.result.audioLoading", { defaultValue: "Loading the audio…" })}</p>
      ) : (
        <button type="button" className="secondary-button" onClick={engage}>
          {t("musicLibrary.loadAudio", { defaultValue: "Load the audio" })}
        </button>
      )}
    </div>
  );
}

function TagEditor({ item, onChange }: { item: MusicLibraryItem; onChange: (tags: string[]) => void }) {
  const { t } = useTranslation("studio");
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState("");
  function commit() {
    const next = withTag(item.tags, draft);
    setDraft("");
    setAdding(false);
    if (next !== item.tags) onChange(next);
  }
  return (
    <ul className="media-library-card__tags music-library__tags">
      {item.tags.map((tag) => (
        <li key={tag} className="badge subtle music-library__tag">
          {tag}
          <button
            type="button"
            className="music-library__tag-remove"
            aria-label={t("musicLibrary.removeTag", { defaultValue: "Remove tag {tag}", tag })}
            onClick={() => onChange(item.tags.filter((entry) => entry !== tag))}
          >
            <CloseIcon />
          </button>
        </li>
      ))}
      <li>
        {adding ? (
          <input
            className="text-input music-library__tag-input"
            autoFocus
            maxLength={40}
            value={draft}
            aria-label={t("musicLibrary.addTag", { defaultValue: "Add tag" })}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={commit}
            onKeyDown={(event) => {
              if (event.key === "Enter") commit();
              if (event.key === "Escape") {
                setDraft("");
                setAdding(false);
              }
            }}
          />
        ) : (
          <button type="button" className="secondary-button music-library__add-tag" onClick={() => setAdding(true)}>
            <PlusIcon /> {t("musicLibrary.addTag", { defaultValue: "Add tag" })}
          </button>
        )}
      </li>
    </ul>
  );
}

function RenameField({ item, onRename, onDone }: { item: MusicLibraryItem; onRename: (title: string) => void; onDone: () => void }) {
  const { t } = useTranslation("studio");
  const [draft, setDraft] = useState(item.title);
  function commit() {
    const next = draft.trim();
    onDone();
    if (next && next !== item.title) onRename(next);
  }
  return (
    <input
      className="text-input music-library__rename"
      autoFocus
      maxLength={200}
      value={draft}
      aria-label={t("musicLibrary.rename", { defaultValue: "Rename" })}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") commit();
        if (event.key === "Escape") onDone();
      }}
    />
  );
}

export function MusicLibraryTab({ music, backendOnline, onTabChange }: MusicLibraryTabProps) {
  const { t } = useTranslation("studio");
  const { library, runtime } = music;
  const storyboard = useSendToStoryboard((tab) => onTabChange(tab));
  const [filter, setFilter] = useState<LibraryFilter>(DEFAULT_LIBRARY_FILTER);
  const [layout, setLayout] = useState<"grid" | "list">("grid");
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [renaming, setRenaming] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<string[] | null>(null);

  // Opening the Library shows what the backend has now (a track made in another window, a delete).
  const { refreshLibrary } = library;
  useEffect(() => {
    if (backendOnline) void refreshLibrary();
  }, [backendOnline, refreshLibrary]);

  const items = useMemo(() => filterLibrary(library.items, filter), [library.items, filter]);
  const tags = useMemo(() => allTags(library.items), [library.items]);
  const counts = libraryCounts(library.items);

  function toggle(id: string, on: boolean) {
    setSelected((previous) => {
      const next = new Set(previous);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });
  }

  async function confirmDelete() {
    const ids = pendingDelete ?? [];
    setPendingDelete(null);
    const deleted = await library.deleteItems(ids);
    setSelected((previous) => new Set([...previous].filter((id) => !deleted.includes(id))));
    music.studio.forgetResults(deleted);
  }

  function onDragStart(event: DragEvent<HTMLElement>, item: MusicLibraryItem) {
    event.dataTransfer.setData("application/x-chaos-media", JSON.stringify({ kind: "music", id: item.id }));
    event.dataTransfer.effectAllowed = "copy";
  }

  const subtitle = t("musicLibrary.subtitle", {
    defaultValue: "{tracks, plural, one {# track} other {# tracks}} · {sounds, plural, one {# sound} other {# sounds}} · {size} on disk",
    tracks: counts.tracks,
    sounds: counts.sounds,
    size: diskLabel(library.diskBytes),
  });

  let body;
  if (!backendOnline) {
    body = (
      <div className="empty-state">
        <p className="muted-text">{t("musicStudio.offline", { defaultValue: "Waiting for the backend…" })}</p>
      </div>
    );
  } else if (!library.loaded) {
    body = (
      <div className="empty-state">
        <p className="muted-text">{t("musicLibrary.loading", { defaultValue: "Loading your library…" })}</p>
      </div>
    );
  } else if (library.items.length === 0) {
    body = (
      <div className="empty-state">
        <p className="muted-text">{t("musicLibrary.empty", { defaultValue: "No music yet. Describe a track and press Make track." })}</p>
        <button type="button" className="primary-button" onClick={() => onTabChange("music-studio")}>
          {t("musicLibrary.openStudio", { defaultValue: "Open Studio" })}
        </button>
      </div>
    );
  } else if (items.length === 0) {
    body = (
      <div className="empty-state">
        <p className="muted-text">{t("musicLibrary.noMatch", { defaultValue: "Nothing matches." })}</p>
        <button type="button" className="secondary-button" onClick={() => setFilter(DEFAULT_LIBRARY_FILTER)}>
          {t("musicLibrary.clearFilters", { defaultValue: "Clear filters" })}
        </button>
      </div>
    );
  } else {
    body = (
      <div className={`music-library__grid${layout === "list" ? " music-library__grid--list" : ""}`} role="list">
        {items.map((item) => {
          const sendItem = {
            kind: "music" as const,
            path: itemFilePath(runtime?.libraryDir, item),
            label: item.title,
            seconds: item.durationS ?? null,
          };
          const size = megabytes(item.sizeBytes);
          const date = new Date(item.createdAt * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
          const meta = [clock(item.durationS), item.modelName, size, date, item.seed != null ? t("musicStudio.result.seed", { defaultValue: "seed {seed}", seed: item.seed }) : null]
            .filter(Boolean)
            .join(" · ");
          return (
            <div role="listitem" key={item.id} className="music-library__item">
              <MediaLibraryCard
                title={
                  renaming === item.id ? (
                    <RenameField item={item} onRename={(title) => void library.renameItem(item.id, title)} onDone={() => setRenaming(null)} />
                  ) : (
                    item.title
                  )
                }
                badge={item.kind === "sound" ? t("musicLibrary.kind.sound", { defaultValue: "Sound" }) : t("musicLibrary.kind.track", { defaultValue: "Track" })}
                meta={meta}
                preview={item.caption}
                selected={selected.has(item.id)}
                onSelect={(on) => toggle(item.id, on)}
                selectLabel={t("musicLibrary.select", { defaultValue: "Select {title}", title: item.title })}
                tagsSlot={<TagEditor item={item} onChange={(next) => void library.setTags(item.id, next)} />}
                player={<LibraryPlayer item={item} />}
                draggable
                onDragStart={(event) => onDragStart(event, item)}
                actions={
                  <>
                    <button type="button" className="secondary-button" onClick={() => music.makeAnotherLike(item)}>
                      {t("musicLibrary.makeAnother", { defaultValue: "Make another like this" })}
                    </button>
                    <SendToStoryboardMenu
                      item={sendItem}
                      projects={storyboard.projects}
                      loaded={storyboard.loaded}
                      busy={storyboard.busy}
                      notice={storyboard.notice}
                      enabled={storyboard.enabled}
                      onOpen={() => void storyboard.loadProjects()}
                      onDismissNotice={storyboard.dismissNotice}
                      onSend={(projectId, target) => storyboard.send(sendItem, projectId, target)}
                    />
                    <button type="button" className="secondary-button" onClick={() => setRenaming(item.id)}>
                      {t("musicLibrary.rename", { defaultValue: "Rename" })}
                    </button>
                    <button type="button" className="secondary-button danger-button" onClick={() => setPendingDelete([item.id])}>
                      {t("musicLibrary.delete", { defaultValue: "Delete" })}
                    </button>
                  </>
                }
              />
            </div>
          );
        })}
      </div>
    );
  }

  const toolbar =
    backendOnline && library.items.length > 0 ? (
      <div className="voice-library__toolbar music-library__toolbar">
        <input
          type="search"
          className="text-input"
          value={filter.query}
          placeholder={t("musicLibrary.search", { defaultValue: "Search titles, prompts, tags" })}
          aria-label={t("musicLibrary.search", { defaultValue: "Search titles, prompts, tags" })}
          onChange={(event) => setFilter({ ...filter, query: event.target.value })}
        />
        <select
          className="text-input music-library__select"
          value={filter.tag}
          aria-label={t("musicLibrary.filter.tag", { defaultValue: "Tag" })}
          onChange={(event) => setFilter({ ...filter, tag: event.target.value })}
        >
          <option value="">{t("musicLibrary.filter.anyTag", { defaultValue: "Any tag" })}</option>
          {tags.map((tag) => (
            <option key={tag} value={tag}>
              {tag}
            </option>
          ))}
        </select>
        <select
          className="text-input music-library__select"
          value={filter.sort}
          aria-label={t("musicLibrary.sort.label", { defaultValue: "Sort" })}
          onChange={(event) => setFilter({ ...filter, sort: event.target.value as LibrarySort })}
        >
          <option value="newest">{t("musicLibrary.sort.newest", { defaultValue: "Newest first" })}</option>
          <option value="oldest">{t("musicLibrary.sort.oldest", { defaultValue: "Oldest first" })}</option>
          <option value="longest">{t("musicLibrary.sort.longest", { defaultValue: "Longest first" })}</option>
        </select>
        <div className="segmented" role="group" aria-label={t("musicLibrary.layout.label", { defaultValue: "Layout" })}>
          {(["grid", "list"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              className={`segment${layout === mode ? " active" : ""}`}
              aria-pressed={layout === mode}
              onClick={() => setLayout(mode)}
            >
              {mode === "grid"
                ? t("musicLibrary.layout.grid", { defaultValue: "Grid" })
                : t("musicLibrary.layout.list", { defaultValue: "List" })}
            </button>
          ))}
        </div>
        {selected.size > 0 ? (
          <button type="button" className="secondary-button danger-button" onClick={() => setPendingDelete([...selected])}>
            {t("musicLibrary.deleteSelected", { defaultValue: "Delete selected ({n})", n: selected.size })}
          </button>
        ) : null}
      </div>
    ) : null;

  return (
    <div className="content-grid music-page">
      <Panel
        title={t("musicLibrary.title", { defaultValue: "Music Library" })}
        subtitle={subtitle}
        className="span-2 music-library"
        actions={
          <button type="button" className="primary-button" onClick={() => onTabChange("music-studio")}>
            {t("musicLibrary.openStudio", { defaultValue: "Open Studio" })}
          </button>
        }
      >
        {toolbar}
        {body}
      </Panel>
      {pendingDelete ? (
        <ConfirmDialog
          title={t("musicLibrary.delete", { defaultValue: "Delete" })}
          body={
            <p>
              {pendingDelete.length === 1
                ? t("musicLibrary.confirmDelete", { defaultValue: "Delete this item and its audio? This cannot be undone." })
                : t("musicLibrary.confirmDeleteMany", { defaultValue: "Delete {n} items and their audio? This cannot be undone.", n: pendingDelete.length })}
            </p>
          }
          confirmLabel={t("musicLibrary.delete", { defaultValue: "Delete" })}
          danger
          onConfirm={() => void confirmDelete()}
          onCancel={() => setPendingDelete(null)}
        />
      ) : null}
    </div>
  );
}
