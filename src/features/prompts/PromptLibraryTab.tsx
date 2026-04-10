import { useEffect, useState } from "react";
import { Panel } from "../../components/Panel";

interface PromptTemplate {
  id: string;
  name: string;
  systemPrompt: string;
  tags: string[];
  category: string;
  fewShotExamples: Array<{ role: string; content: string }>;
  createdAt: string;
  updatedAt: string;
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

  const selected = templates.find((t) => t.id === selectedId) ?? null;

  useEffect(() => {
    if (!backendOnline) return;
    fetch("/api/prompts")
      .then((r) => r.json())
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
  }

  async function handleSave() {
    const body: Record<string, unknown> = {
      name: editName,
      systemPrompt: editPrompt,
      category: editCategory,
      tags: editTags.split(",").map((t) => t.trim()).filter(Boolean),
    };
    if (selectedId) body.id = selectedId;

    try {
      const resp = await fetch("/api/prompts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
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
      await fetch(`/api/prompts/${encodeURIComponent(id)}`, { method: "DELETE" });
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
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="primary-button" type="button" onClick={() => void handleSave()}>Save</button>
              <button className="secondary-button" type="button" onClick={() => setEditMode(false)}>Cancel</button>
            </div>
          </div>
        ) : selected ? (
          <div style={{ padding: 16 }}>
            <div style={{ marginBottom: 12, display: "flex", gap: 8 }}>
              <span className="badge">{selected.category}</span>
              {selected.tags.map((tag) => <span key={tag} className="badge" style={{ background: "#1e3a5f", color: "#8fb4ff" }}>{tag}</span>)}
            </div>
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 11, color: "#7a8594", display: "block", marginBottom: 4 }}>System Prompt</label>
              <pre style={{ background: "#0f1215", borderRadius: 8, padding: 12, color: "#c8d0da", whiteSpace: "pre-wrap", fontSize: 12, maxHeight: 300, overflow: "auto" }}>
                {selected.systemPrompt}
              </pre>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="primary-button" type="button" onClick={() => onApplyTemplate(selected.systemPrompt)}>
                Use in Chat
              </button>
              <button className="secondary-button" type="button" onClick={() => startEdit(selected)}>Edit</button>
              <button className="secondary-button message-action-delete" type="button" onClick={() => void handleDelete(selected.id)}>Delete</button>
            </div>
            <div style={{ marginTop: 12, fontSize: 11, color: "#5a6574" }}>
              Created: {selected.createdAt} | Updated: {selected.updatedAt}
            </div>
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
