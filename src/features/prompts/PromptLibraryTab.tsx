import { useEffect, useMemo, useState } from "react";
import { apiFetch, fetchJson } from "../../api";
import { Panel } from "../../components/Panel";

/**
 * Phase 2.7: variable declaration shape. `default` is the seed value
 * shown in the fill-form before Use in Chat; `description` surfaces
 * as a hint underneath the input. Boolean variables render as a
 * checkbox; number variables as `<input type="number">`; string as
 * a textarea.
 */
interface PromptVariable {
  name: string;
  type: "string" | "number" | "boolean";
  default?: string | number | boolean | null;
  description?: string;
}

interface PromptTemplate {
  id: string;
  name: string;
  systemPrompt: string;
  tags: string[];
  category: string;
  fewShotExamples: Array<{ role: string; content: string }>;
  variables?: PromptVariable[];
  presetSamplers?: Record<string, unknown> | null;
  presetModelRef?: string | null;
  createdAt: string;
  updatedAt: string;
}

/**
 * Phase 2.7: replace `{{name}}` placeholders with user-supplied
 * values. Mirrors backend `apply_variables` so the frontend can
 * preview the resolved prompt before sending. Missing names stay
 * as the literal placeholder so the user notices the gap.
 */
const PLACEHOLDER_PATTERN = /\{\{\s*([A-Za-z0-9_-]+)\s*\}\}/g;

function applyVariables(
  text: string,
  values: Record<string, string | number | boolean | null | undefined>,
): string {
  if (!text) return text;
  return text.replace(PLACEHOLDER_PATTERN, (placeholder, name) => {
    if (!(name in values)) return placeholder;
    const value = values[name];
    if (value == null) return "";
    if (typeof value === "boolean") return value ? "true" : "false";
    return String(value);
  });
}

interface PromptLibraryTabProps {
  backendOnline: boolean;
  onApplyTemplate: (systemPrompt: string) => void;
}

export function PromptLibraryTab({ backendOnline, onApplyTemplate }: PromptLibraryTabProps) {
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [editMode, setEditMode] = useState(false);
  const [editName, setEditName] = useState("");
  const [editPrompt, setEditPrompt] = useState("");
  const [editCategory, setEditCategory] = useState("");
  const [editTags, setEditTags] = useState("");
  // Phase 2.7: raw JSON in the variables editor — keeps the surface
  // tight while still allowing full control. The fill-form parses it
  // back into PromptVariable[] when the user clicks Use in Chat.
  const [editVariables, setEditVariables] = useState("");
  const [editPresetModelRef, setEditPresetModelRef] = useState("");
  const [editPresetSamplers, setEditPresetSamplers] = useState("");
  // Variable fill state for Use in Chat. When the selected template
  // declares variables, clicking Use opens this form rather than
  // applying the raw template. The resolved prompt is what reaches the
  // composer.
  const [fillValues, setFillValues] = useState<Record<string, string | number | boolean>>({});
  const [fillOpen, setFillOpen] = useState(false);

  const selected = templates.find((t) => t.id === selectedId) ?? null;
  const selectedVariables = useMemo(() => selected?.variables ?? [], [selected]);
  const resolvedFillPrompt = useMemo(() => {
    if (!selected) return "";
    return applyVariables(selected.systemPrompt, fillValues);
  }, [selected, fillValues]);

  useEffect(() => {
    if (!backendOnline) return;
    fetchJson<{ templates?: PromptTemplate[] }>("/api/prompts")
      .then((data) => setTemplates(data.templates ?? []))
      .catch(() => {});
  }, [backendOnline]);

  const filtered = templates.filter((t) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      t.name.toLowerCase().includes(q) ||
      t.category.toLowerCase().includes(q) ||
      t.tags.some((tag) => tag.toLowerCase().includes(q))
    );
  });

  function startEdit(template: PromptTemplate | null) {
    setEditMode(true);
    setEditName(template?.name ?? "");
    setEditPrompt(template?.systemPrompt ?? "");
    setEditCategory(template?.category ?? "General");
    setEditTags(template?.tags?.join(", ") ?? "");
    setEditVariables(
      template?.variables?.length
        ? JSON.stringify(template.variables, null, 2)
        : "",
    );
    setEditPresetModelRef(template?.presetModelRef ?? "");
    setEditPresetSamplers(
      template?.presetSamplers
        ? JSON.stringify(template.presetSamplers, null, 2)
        : "",
    );
  }

  function parseEditVariables(): PromptVariable[] {
    if (!editVariables.trim()) return [];
    try {
      const parsed = JSON.parse(editVariables);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(
        (v): v is PromptVariable =>
          v && typeof v === "object" && typeof v.name === "string",
      );
    } catch {
      return [];
    }
  }

  function parseEditPresetSamplers(): Record<string, unknown> | null {
    if (!editPresetSamplers.trim()) return null;
    try {
      const parsed = JSON.parse(editPresetSamplers);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
      return null;
    } catch {
      return null;
    }
  }

  function openFillForm() {
    if (!selected) return;
    if (!selectedVariables.length) {
      // No variables → apply raw prompt directly
      onApplyTemplate(selected.systemPrompt);
      return;
    }
    const seed: Record<string, string | number | boolean> = {};
    for (const variable of selectedVariables) {
      const fallback = variable.default ?? (variable.type === "boolean" ? false : variable.type === "number" ? 0 : "");
      seed[variable.name] = fallback as string | number | boolean;
    }
    setFillValues(seed);
    setFillOpen(true);
  }

  function applyFilledTemplate() {
    if (!selected) return;
    onApplyTemplate(applyVariables(selected.systemPrompt, fillValues));
    setFillOpen(false);
  }

  async function handleSave() {
    const body: Record<string, unknown> = {
      name: editName,
      systemPrompt: editPrompt,
      category: editCategory,
      tags: editTags.split(",").map((t) => t.trim()).filter(Boolean),
      variables: parseEditVariables(),
      presetSamplers: parseEditPresetSamplers(),
      presetModelRef: editPresetModelRef.trim() || null,
    };
    if (selectedId) body.id = selectedId;

    try {
      const resp = await apiFetch("/api/prompts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        return;
      }
      const data = await resp.json();
      if (data.template) {
        setTemplates((prev) => {
          const exists = prev.find((t) => t.id === data.template.id);
          if (exists) return prev.map((t) => (t.id === data.template.id ? data.template : t));
          return [...prev, data.template];
        });
        setSelectedId(data.template.id);
      }
    } catch { /* ignore */ }
    setEditMode(false);
  }

  async function handleDelete(id: string) {
    try {
      const response = await apiFetch(`/api/prompts/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!response.ok) {
        return;
      }
      setTemplates((prev) => prev.filter((t) => t.id !== id));
      if (selectedId === id) setSelectedId(null);
    } catch { /* ignore */ }
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 12, height: "100%" }}>
      {/* Left: Template list */}
      <Panel
        title="Templates"
        subtitle={`${filtered.length} templates`}
        actions={
          <button className="secondary-button" type="button" onClick={() => { setSelectedId(null); startEdit(null); }} style={{ fontSize: 11 }}>
            New
          </button>
        }
      >
        <div style={{ padding: "8px 8px 4px" }}>
          <input
            type="text"
            className="text-input"
            placeholder="Search templates..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{ width: "100%", fontSize: 12 }}
          />
        </div>
        <div style={{ overflow: "auto", flex: 1 }}>
          {filtered.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => { setSelectedId(t.id); setEditMode(false); }}
              className={`session-button${selectedId === t.id ? " active" : ""}`}
              style={{ width: "100%", textAlign: "left" }}
            >
              <div style={{ fontWeight: 600, fontSize: 13 }}>{t.name}</div>
              <div style={{ fontSize: 11, color: "#7a8594" }}>{t.category} {t.tags.length ? `| ${t.tags.join(", ")}` : ""}</div>
            </button>
          ))}
        </div>
      </Panel>

      {/* Right: Detail/Editor */}
      <Panel title={editMode ? (selectedId ? "Edit Template" : "New Template") : (selected?.name ?? "Select a template")} subtitle="">
        {editMode ? (
          <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
            <div>
              <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>Name</label>
              <input className="text-input" value={editName} onChange={(e) => setEditName(e.target.value)} style={{ width: "100%" }} />
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>Category</label>
              <input className="text-input" value={editCategory} onChange={(e) => setEditCategory(e.target.value)} style={{ width: "100%" }} />
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>Tags (comma-separated)</label>
              <input className="text-input" value={editTags} onChange={(e) => setEditTags(e.target.value)} style={{ width: "100%" }} />
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>System Prompt</label>
              <textarea
                className="text-input"
                value={editPrompt}
                onChange={(e) => setEditPrompt(e.target.value)}
                style={{ width: "100%", minHeight: 200, resize: "vertical", fontFamily: "monospace", fontSize: 12 }}
              />
              <small style={{ fontSize: 10, color: "#5a6574" }}>
                Use {"{{name}}"} placeholders for variables you declare below.
              </small>
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>
                Variables (JSON array — Phase 2.7)
              </label>
              <textarea
                className="text-input"
                value={editVariables}
                onChange={(e) => setEditVariables(e.target.value)}
                placeholder={'[{"name": "topic", "type": "string", "default": "AI"}]'}
                style={{ width: "100%", minHeight: 80, resize: "vertical", fontFamily: "monospace", fontSize: 11 }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>
                Preset model ref (optional)
              </label>
              <input
                className="text-input"
                value={editPresetModelRef}
                onChange={(e) => setEditPresetModelRef(e.target.value)}
                placeholder="e.g. Qwen3-7B-Instruct"
                style={{ width: "100%" }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>
                Preset samplers (JSON object — optional)
              </label>
              <textarea
                className="text-input"
                value={editPresetSamplers}
                onChange={(e) => setEditPresetSamplers(e.target.value)}
                placeholder={'{"topP": 0.9, "topK": 40}'}
                style={{ width: "100%", minHeight: 60, resize: "vertical", fontFamily: "monospace", fontSize: 11 }}
              />
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="primary-button" type="button" onClick={() => void handleSave()}>Save</button>
              <button className="secondary-button" type="button" onClick={() => setEditMode(false)}>Cancel</button>
            </div>
          </div>
        ) : selected ? (
          <div style={{ padding: 16 }}>
            <div style={{ marginBottom: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
              <span className="badge">{selected.category}</span>
              {selected.tags.map((tag) => <span key={tag} className="badge" style={{ background: "#1e3a5f", color: "#8fb4ff" }}>{tag}</span>)}
              {selected.presetModelRef ? (
                <span className="badge" style={{ background: "#1f2a44", color: "#9bc7ff" }} title="Preset model — applied when you Use in Chat">
                  preset: {selected.presetModelRef}
                </span>
              ) : null}
              {selected.variables?.length ? (
                <span className="badge" style={{ background: "#2a3a1f", color: "#b9d18f" }} title="Template has variable placeholders">
                  {selected.variables.length} variable{selected.variables.length === 1 ? "" : "s"}
                </span>
              ) : null}
            </div>
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>System Prompt</label>
              <pre style={{ background: "#0f1215", borderRadius: 8, padding: 12, color: "#c8d0da", whiteSpace: "pre-wrap", fontSize: 12, maxHeight: 300, overflow: "auto" }}>
                {selected.systemPrompt}
              </pre>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="primary-button" type="button" onClick={openFillForm}>
                {selectedVariables.length ? "Use in Chat..." : "Use in Chat"}
              </button>
              <button className="secondary-button" type="button" onClick={() => startEdit(selected)}>Edit</button>
              <button className="secondary-button message-action-delete" type="button" onClick={() => void handleDelete(selected.id)}>Delete</button>
            </div>
            <div style={{ marginTop: 12, fontSize: 11, color: "#5a6574" }}>
              Created: {selected.createdAt} | Updated: {selected.updatedAt}
            </div>
            {fillOpen && selectedVariables.length ? (
              <div style={{ marginTop: 16, padding: 12, background: "#0f1215", borderRadius: 8, border: "1px solid #1f2a3a" }}>
                <strong style={{ fontSize: 12 }}>Fill template variables</strong>
                <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 8 }}>
                  {selectedVariables.map((variable) => (
                    <div key={variable.name}>
                      <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>
                        {variable.name}
                        {variable.description ? <span style={{ color: "#5a6574" }}> — {variable.description}</span> : null}
                      </label>
                      {variable.type === "boolean" ? (
                        <input
                          type="checkbox"
                          checked={Boolean(fillValues[variable.name])}
                          onChange={(e) => setFillValues((prev) => ({ ...prev, [variable.name]: e.target.checked }))}
                        />
                      ) : variable.type === "number" ? (
                        <input
                          type="number"
                          className="text-input"
                          value={Number(fillValues[variable.name] ?? 0)}
                          onChange={(e) => setFillValues((prev) => ({ ...prev, [variable.name]: parseFloat(e.target.value) || 0 }))}
                          style={{ width: "100%" }}
                        />
                      ) : (
                        <textarea
                          className="text-input"
                          value={String(fillValues[variable.name] ?? "")}
                          onChange={(e) => setFillValues((prev) => ({ ...prev, [variable.name]: e.target.value }))}
                          rows={2}
                          style={{ width: "100%", fontFamily: "inherit", fontSize: 12 }}
                        />
                      )}
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 12 }}>
                  <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>Resolved prompt preview</label>
                  <pre style={{ background: "#080a0c", borderRadius: 6, padding: 10, color: "#c8d0da", whiteSpace: "pre-wrap", fontSize: 11, maxHeight: 200, overflow: "auto" }}>
                    {resolvedFillPrompt}
                  </pre>
                </div>
                <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                  <button className="primary-button" type="button" onClick={applyFilledTemplate}>
                    Apply to chat
                  </button>
                  <button className="secondary-button" type="button" onClick={() => setFillOpen(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        ) : (
          <div style={{ padding: 24, textAlign: "center" }}>
            <p className="muted-text">Select a template from the list or create a new one.</p>
          </div>
        )}
      </Panel>
    </div>
  );
}
