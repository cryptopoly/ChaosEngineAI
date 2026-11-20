/**
 * D-87: the prompt library, folded into Chat. A "Prompts" button beside the composer opens a panel to browse and
 * search the saved templates, fill a template's variables, insert it into the message or use it as the system
 * prompt, save the current message as a template, and edit or delete one. It reads and writes the same
 * `/api/prompts` store the old Prompts page used. The system prompt is still one app-wide value (FU-224); the
 * panel shows it with a way to stop it.
 */
import { useCallback, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { apiFetch, fetchJson } from "../../api";
import {
  applyVariables,
  formatTemplateTime,
  insertIntoDraft,
  jsonBoxProblem,
  matchesTemplateSearch,
  nameFromDraft,
  parseVariablesBox,
  seedVariableValues,
  type PromptTemplate,
  type VariableValues,
} from "./promptLogic";

export interface ChatPromptsPanelProps {
  backendOnline: boolean;
  draftMessage: string;
  onDraftMessageChange: (message: string) => void;
  /** The app-wide system prompt Chat sends ("" when none). */
  systemPrompt: string;
  onSystemPromptChange: (prompt: string) => void;
}

type View = { kind: "list" } | { kind: "detail"; id: string } | { kind: "edit"; id: string | null };

interface EditorFields {
  name: string;
  category: string;
  tags: string;
  prompt: string;
  variables: string;
}

const EMPTY_EDITOR: EditorFields = { name: "", category: "General", tags: "", prompt: "", variables: "" };

export function ChatPromptsPanel({
  backendOnline,
  draftMessage,
  onDraftMessageChange,
  systemPrompt,
  onSystemPromptChange,
}: ChatPromptsPanelProps) {
  const { t, i18n } = useTranslation("common");
  const panelId = useId();
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<View>({ kind: "list" });
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [loadFailed, setLoadFailed] = useState(false);
  const [query, setQuery] = useState("");
  const [fill, setFill] = useState<VariableValues>({});
  const [editor, setEditor] = useState<EditorFields>(EMPTY_EDITOR);
  const [saveError, setSaveError] = useState<string | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const backRef = useRef<HTMLButtonElement>(null);

  const close = useCallback((refocus: boolean) => {
    setOpen(false);
    if (refocus) triggerRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!open || !backendOnline) return;
    let live = true;
    fetchJson<{ templates?: PromptTemplate[] }>("/api/prompts")
      .then((data) => {
        if (!live) return;
        setTemplates(data.templates ?? []);
        setLoadFailed(false);
      })
      .catch(() => {
        if (live) setLoadFailed(true);
      });
    return () => {
      live = false;
    };
  }, [open, backendOnline]);

  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) close(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open, close]);

  // Focus follows the view: the search box on the list, the Back button on a template or the editor (the row or
  // button that opened it is gone, and Escape must still reach the panel).
  useEffect(() => {
    if (!open) return;
    if (view.kind === "list") searchRef.current?.focus();
    else if (view.kind === "detail") backRef.current?.focus();
  }, [open, view]);

  const filtered = useMemo(() => templates.filter((template) => matchesTemplateSearch(template, query)), [templates, query]);
  const selected = view.kind === "detail" ? templates.find((template) => template.id === view.id) ?? null : null;
  const resolved = selected ? applyVariables(selected.systemPrompt, fill) : "";

  function openPanel() {
    setView({ kind: "list" });
    setSaveError(null);
    setOpen(true);
  }

  function showDetail(template: PromptTemplate) {
    setFill(seedVariableValues(template.variables));
    setView({ kind: "detail", id: template.id });
  }

  function startEdit(template: PromptTemplate | null, prefill?: Partial<EditorFields>) {
    setSaveError(null);
    setEditor(
      template
        ? {
            name: template.name,
            category: template.category || "General",
            tags: template.tags.join(", "),
            prompt: template.systemPrompt,
            variables: template.variables?.length ? JSON.stringify(template.variables, null, 2) : "",
          }
        : { ...EMPTY_EDITOR, ...prefill },
    );
    setView({ kind: "edit", id: template?.id ?? null });
  }

  function insertSelected() {
    if (!selected) return;
    onDraftMessageChange(insertIntoDraft(draftMessage, resolved));
    const textarea = wrapRef.current?.closest(".composer")?.querySelector("textarea");
    setOpen(false);
    if (textarea) textarea.focus();
    else triggerRef.current?.focus();
  }

  function useAsSystemPrompt() {
    if (!selected) return;
    onSystemPromptChange(resolved);
  }

  async function handleSave() {
    if (!editor.name.trim()) {
      setSaveError(t("promptLibrary.nameRequired", { defaultValue: "Give the template a name before saving." }));
      return;
    }
    if (jsonBoxProblem(editor.variables, "array")) {
      setSaveError(t("promptLibrary.variablesInvalid", { defaultValue: "Variables must be a JSON array (a list in square brackets). Nothing was saved." }));
      return;
    }
    const failed = t("promptLibrary.saveFailed", { defaultValue: "The template was not saved. Check the fields and try again." });
    const editingId = view.kind === "edit" ? view.id : null;
    const existing = editingId ? templates.find((template) => template.id === editingId) : undefined;
    // Fields this panel does not edit are sent back as they were, so an edit never drops them.
    const body: Record<string, unknown> = {
      name: editor.name.trim(),
      systemPrompt: editor.prompt,
      category: editor.category.trim() || "General",
      tags: editor.tags.split(",").map((tag) => tag.trim()).filter(Boolean),
      variables: parseVariablesBox(editor.variables),
      presetSamplers: existing?.presetSamplers ?? null,
      presetModelRef: existing?.presetModelRef ?? null,
      fewShotExamples: existing?.fewShotExamples ?? [],
    };
    if (editingId) body.id = editingId;
    setSaveError(null);
    try {
      const response = await apiFetch("/api/prompts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        setSaveError(failed);
        return;
      }
      const data = (await response.json()) as { template?: PromptTemplate };
      const saved = data.template;
      if (!saved) {
        setSaveError(failed);
        return;
      }
      setTemplates((prev) => (prev.some((template) => template.id === saved.id)
        ? prev.map((template) => (template.id === saved.id ? saved : template))
        : [...prev, saved]));
      showDetail(saved);
    } catch {
      setSaveError(failed);
    }
  }

  async function handleDelete(template: PromptTemplate) {
    const confirmed = window.confirm(
      t("promptLibrary.deleteConfirm", { name: template.name, defaultValue: "Delete the template \"{name}\"? This cannot be undone." }),
    );
    if (!confirmed) return;
    try {
      const response = await apiFetch(`/api/prompts/${encodeURIComponent(template.id)}`, { method: "DELETE" });
      if (!response.ok) return;
      setTemplates((prev) => prev.filter((row) => row.id !== template.id));
      setView({ kind: "list" });
    } catch {
      /* the row stays; nothing was deleted */
    }
  }

  function onPanelKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      if (view.kind === "list") close(true);
      else setView({ kind: "list" });
      return;
    }
    if (view.kind !== "list" || (event.key !== "ArrowDown" && event.key !== "ArrowUp")) return;
    // Up and Down move between the search box and the rows.
    const rows = Array.from(listRef.current?.querySelectorAll<HTMLButtonElement>("button.chat-prompts__row") ?? []);
    const stops: HTMLElement[] = [...(searchRef.current ? [searchRef.current] : []), ...rows];
    const at = stops.indexOf(document.activeElement as HTMLElement);
    if (stops.length < 2) return;
    event.preventDefault();
    const next = event.key === "ArrowDown" ? (at + 1) % stops.length : (at - 1 + stops.length) % stops.length;
    stops[next]?.focus();
  }

  const title = view.kind === "edit"
    ? view.id
      ? t("panels.editTemplate", { defaultValue: "Edit Template" })
      : t("panels.newTemplate", { defaultValue: "New Template" })
    : selected?.name ?? t("promptLibrary.chatPanelTitle", { defaultValue: "Prompts" });

  return (
    <div className="chat-prompts" ref={wrapRef}>
      <button
        ref={triggerRef}
        type="button"
        className={`secondary-button chat-prompts__trigger${systemPrompt ? " chat-prompts__trigger--active" : ""}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        title={systemPrompt
          ? t("promptLibrary.chatButtonActiveTooltip", { defaultValue: "Saved prompts. A system prompt is in use." })
          : t("promptLibrary.chatButtonTooltip", { defaultValue: "Saved prompts: insert one into the message, or use one as the system prompt" })}
        onClick={() => (open ? close(false) : openPanel())}
      >
        {t("promptLibrary.chatButton", { defaultValue: "Prompts" })}
        {systemPrompt ? <span className="chat-prompts__dot" aria-hidden="true" /> : null}
      </button>
      {open ? (
        <div
          id={panelId}
          className="chat-prompts__panel"
          role="dialog"
          aria-label={t("promptLibrary.chatPanelTitle", { defaultValue: "Prompts" })}
          onKeyDown={onPanelKeyDown}
        >
          <div className="chat-prompts__header">
            {view.kind !== "list" ? (
              <button ref={backRef} type="button" className="chat-prompts__icon-button" onClick={() => setView({ kind: "list" })}
                aria-label={t("promptLibrary.back", { defaultValue: "Back to the list" })}>
                &larr;
              </button>
            ) : null}
            <strong className="chat-prompts__title">{title}</strong>
            <button type="button" className="chat-prompts__icon-button" onClick={() => close(true)}
              aria-label={t("promptLibrary.close", { defaultValue: "Close" })}>
              &times;
            </button>
          </div>

          {systemPrompt ? (
            <div className="chat-prompts__in-use" role="status">
              <span>{t("promptLibrary.inUseNote", { defaultValue: "Chat is using this system prompt in every thread until you stop it or restart the app:" })}</span>
              <pre>{systemPrompt}</pre>
              <button type="button" className="secondary-button" onClick={() => onSystemPromptChange("")}>
                {t("promptLibrary.stopUsing", { defaultValue: "Stop using it" })}
              </button>
            </div>
          ) : null}

          {view.kind === "list" ? (
            <>
              <div className="chat-prompts__actions">
                <button type="button" className="secondary-button" disabled={!draftMessage.trim()}
                  onClick={() => startEdit(null, { name: nameFromDraft(draftMessage), prompt: draftMessage })}
                  title={draftMessage.trim() ? undefined : t("promptLibrary.saveDraftEmpty", { defaultValue: "Type a message first" })}>
                  {t("promptLibrary.saveDraft", { defaultValue: "Save message as prompt" })}
                </button>
                <button type="button" className="secondary-button" onClick={() => startEdit(null)}>
                  {t("promptLibrary.new", { defaultValue: "New" })}
                </button>
              </div>
              <input
                ref={searchRef}
                type="search"
                className="text-input"
                aria-label={t("promptLibrary.searchPlaceholder", { defaultValue: "Search templates..." })}
                placeholder={t("promptLibrary.searchPlaceholder", { defaultValue: "Search templates..." })}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              <span className="chat-prompts__count" aria-live="polite">
                {t("promptLibrary.templatesCount", { count: filtered.length, defaultValue: "{count, plural, one {# template} other {# templates}}" })}
              </span>
              <ul className="chat-prompts__list" ref={listRef}>
                {filtered.map((template) => (
                  <li key={template.id}>
                    <button type="button" className="chat-prompts__row" onClick={() => showDetail(template)}>
                      <strong>{template.name}</strong>
                      <span>{[template.category, ...template.tags].filter(Boolean).join(" · ")}</span>
                    </button>
                  </li>
                ))}
              </ul>
              {loadFailed ? (
                <p role="alert" className="error-text">
                  {t("promptLibrary.loadFailed", { defaultValue: "The templates could not be loaded. Check that the backend is running." })}
                </p>
              ) : filtered.length === 0 ? (
                <p className="muted-text">
                  {query.trim()
                    ? t("promptLibrary.noMatches", { defaultValue: "No template matches the search." })
                    : t("promptLibrary.empty", { defaultValue: "No templates yet. Use New to make one." })}
                </p>
              ) : null}
            </>
          ) : null}

          {view.kind === "detail" && selected ? (
            <div className="chat-prompts__detail">
              <span className="chat-prompts__meta">
                {[selected.category, ...selected.tags].filter(Boolean).join(" · ")}
                {selected.variables?.length
                  ? ` · ${t("promptLibrary.variableCount", { count: selected.variables.length, defaultValue: "{count, plural, one {# variable} other {# variables}}" })}`
                  : ""}
              </span>
              {selected.variables?.length ? (
                <fieldset className="chat-prompts__fill">
                  <legend>{t("promptLibrary.fillTitle", { defaultValue: "Fill template variables" })}</legend>
                  {selected.variables.map((variable) => (
                    <label key={variable.name} className="chat-prompts__field">
                      <span>{variable.name}{variable.description ? ` (${variable.description})` : ""}</span>
                      {variable.type === "boolean" ? (
                        <input type="checkbox" checked={Boolean(fill[variable.name])}
                          onChange={(event) => setFill((prev) => ({ ...prev, [variable.name]: event.target.checked }))} />
                      ) : variable.type === "number" ? (
                        <input type="number" className="text-input" value={Number(fill[variable.name] ?? 0)}
                          onChange={(event) => setFill((prev) => ({ ...prev, [variable.name]: parseFloat(event.target.value) || 0 }))} />
                      ) : (
                        <input type="text" className="text-input" value={String(fill[variable.name] ?? "")}
                          onChange={(event) => setFill((prev) => ({ ...prev, [variable.name]: event.target.value }))} />
                      )}
                    </label>
                  ))}
                </fieldset>
              ) : null}
              <span className="chat-prompts__label">
                {selected.variables?.length
                  ? t("promptLibrary.resolvedPreview", { defaultValue: "Resolved prompt preview" })
                  : t("promptLibrary.editor.systemPrompt", { defaultValue: "System Prompt" })}
              </span>
              <pre className="chat-prompts__preview" data-testid="chat-prompts-preview">{resolved}</pre>
              <div className="chat-prompts__actions">
                <button type="button" className="primary-button" onClick={insertSelected}>
                  {t("promptLibrary.insert", { defaultValue: "Insert into message" })}
                </button>
                <button type="button" className="secondary-button" onClick={useAsSystemPrompt}
                  aria-pressed={systemPrompt !== "" && systemPrompt === resolved}>
                  {t("promptLibrary.useAsSystem", { defaultValue: "Use as system prompt" })}
                </button>
                <button type="button" className="secondary-button" onClick={() => startEdit(selected)}>
                  {t("promptLibrary.edit", { defaultValue: "Edit" })}
                </button>
                <button type="button" className="secondary-button message-action-delete" onClick={() => void handleDelete(selected)}>
                  {t("promptLibrary.delete", { defaultValue: "Delete" })}
                </button>
              </div>
              <span className="chat-prompts__meta">
                {t("promptLibrary.createdUpdated", {
                  created: formatTemplateTime(selected.createdAt, i18n.language),
                  updated: formatTemplateTime(selected.updatedAt, i18n.language),
                  defaultValue: "Created: {created} | Updated: {updated}",
                })}
              </span>
            </div>
          ) : null}

          {view.kind === "edit" ? (
            <form className="chat-prompts__editor" onSubmit={(event) => { event.preventDefault(); void handleSave(); }}>
              <label className="chat-prompts__field">
                <span>{t("promptLibrary.editor.name", { defaultValue: "Name" })}</span>
                <input className="text-input" value={editor.name} autoFocus
                  onChange={(event) => setEditor((prev) => ({ ...prev, name: event.target.value }))} />
              </label>
              <label className="chat-prompts__field">
                <span>{t("promptLibrary.editor.category", { defaultValue: "Category" })}</span>
                <input className="text-input" value={editor.category}
                  onChange={(event) => setEditor((prev) => ({ ...prev, category: event.target.value }))} />
              </label>
              <label className="chat-prompts__field">
                <span>{t("promptLibrary.editor.tags", { defaultValue: "Tags (comma-separated)" })}</span>
                <input className="text-input" value={editor.tags}
                  onChange={(event) => setEditor((prev) => ({ ...prev, tags: event.target.value }))} />
              </label>
              <label className="chat-prompts__field">
                <span>{t("promptLibrary.editor.promptText", { defaultValue: "Prompt" })}</span>
                <textarea className="text-input" rows={6} value={editor.prompt} aria-describedby={`${panelId}-hint`}
                  onChange={(event) => setEditor((prev) => ({ ...prev, prompt: event.target.value }))} />
              </label>
              <small id={`${panelId}-hint`} className="chat-prompts__meta">
                {t("promptLibrary.editor.placeholderHint", { defaultValue: "Use '{{'name'}}' placeholders for variables you declare below." })}
              </small>
              <label className="chat-prompts__field">
                <span>{t("promptLibrary.editor.variables", { defaultValue: "Variables (JSON array)" })}</span>
                <textarea className="text-input chat-prompts__mono" rows={3} value={editor.variables}
                  placeholder={'[{"name": "topic", "type": "string", "default": "AI"}]'}
                  onChange={(event) => setEditor((prev) => ({ ...prev, variables: event.target.value }))} />
              </label>
              {saveError ? <p role="alert" className="error-text">{saveError}</p> : null}
              <div className="chat-prompts__actions">
                <button type="submit" className="primary-button">{t("promptLibrary.editor.save", { defaultValue: "Save" })}</button>
                <button type="button" className="secondary-button"
                  onClick={() => (view.id ? setView({ kind: "detail", id: view.id }) : setView({ kind: "list" }))}>
                  {t("promptLibrary.editor.cancel", { defaultValue: "Cancel" })}
                </button>
              </div>
            </form>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
