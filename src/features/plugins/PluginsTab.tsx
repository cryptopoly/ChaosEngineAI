import { useEffect, useState } from "react";
import { Panel } from "../../components/Panel";

interface PluginInfo {
  id: string;
  name: string;
  type: string;
  version: string;
  author: string;
  description: string;
  builtin: boolean;
  enabled: boolean;
}

interface PluginsTabProps {
  backendOnline: boolean;
}

const TYPE_LABELS: Record<string, string> = {
  cache_strategy: "Cache Strategies",
  inference_engine: "Inference Engines",
  tool: "Agent Tools",
  model_source: "Model Sources",
  post_processor: "Post Processors",
};

const TYPE_COLORS: Record<string, string> = {
  cache_strategy: "#8fb4ff",
  tool: "#8fcf9f",
  inference_engine: "#f0b060",
  model_source: "#c084fc",
  post_processor: "#f87171",
};

export function PluginsTab({ backendOnline }: PluginsTabProps) {
  const [plugins, setPlugins] = useState<Record<string, PluginInfo[]>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!backendOnline) return;
    fetch("/api/plugins")
      .then((r) => r.json())
      .then((data) => {
        setPlugins(data.plugins ?? {});
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [backendOnline]);

  async function togglePlugin(pluginId: string, currentlyEnabled: boolean) {
    const action = currentlyEnabled ? "disable" : "enable";
    try {
      await fetch(`/api/plugins/${encodeURIComponent(pluginId)}/${action}`, { method: "POST" });
      setPlugins((prev) => {
        const next = { ...prev };
        for (const [type, list] of Object.entries(next)) {
          next[type] = list.map((p) =>
            p.id === pluginId ? { ...p, enabled: !currentlyEnabled } : p,
          );
        }
        return next;
      });
    } catch { /* ignore */ }
  }

  const totalCount = Object.values(plugins).reduce((sum, list) => sum + list.length, 0);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Panel title="Plugin System" subtitle={loading ? "Loading..." : `${totalCount} plugins registered`}>
        {loading ? (
          <p className="muted-text" style={{ padding: 16 }}>Discovering plugins...</p>
        ) : totalCount === 0 ? (
          <p className="muted-text" style={{ padding: 24, textAlign: "center" }}>No plugins found.</p>
        ) : (
          Object.entries(plugins).map(([type, list]) => (
            <div key={type} style={{ marginBottom: 16, padding: "0 8px" }}>
              <h4 style={{ color: TYPE_COLORS[type] ?? "#c8d0da", fontSize: 13, marginBottom: 8, fontWeight: 600 }}>
                {TYPE_LABELS[type] ?? type} ({list.length})
              </h4>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 8 }}>
                {list.map((plugin) => (
                  <div
                    key={plugin.id}
                    style={{
                      border: "1px solid #27303a",
                      borderRadius: 8,
                      padding: 12,
                      background: plugin.enabled ? "#1a1f26" : "#0f1215",
                      opacity: plugin.enabled ? 1 : 0.6,
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 10,
                    }}
                  >
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
                        <span style={{ fontWeight: 600, color: "#c8d0da", fontSize: 13 }}>{plugin.name}</span>
                        {plugin.builtin && (
                          <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 3, background: "#27303a", color: "#7a8594" }}>
                            BUILT-IN
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: 12, color: "#7a8594", marginBottom: 2 }}>{plugin.description}</div>
                      <div style={{ fontSize: 11, color: "#5a6574" }}>v{plugin.version}</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => void togglePlugin(plugin.id, plugin.enabled)}
                      style={{
                        border: "1px solid #27303a",
                        borderRadius: 6,
                        padding: "4px 10px",
                        background: plugin.enabled ? "#1e3a5f" : "#15191e",
                        color: plugin.enabled ? "#8fb4ff" : "#5a6574",
                        cursor: "pointer",
                        fontSize: 11,
                        flexShrink: 0,
                      }}
                    >
                      {plugin.enabled ? "ON" : "OFF"}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </Panel>

      <Panel title="External Plugins" subtitle="Install from directory">
        <div style={{ padding: 24, textAlign: "center" }}>
          <p className="muted-text" style={{ marginBottom: 8 }}>
            Place plugin directories with a manifest.json in:
          </p>
          <code style={{ background: "#0f1215", padding: "4px 12px", borderRadius: 6, color: "#8fb4ff", fontSize: 12 }}>
            ~/.chaosengine/plugins/
          </code>
          <p style={{ color: "#5a6574", fontSize: 12, marginTop: 8 }}>
            Each plugin needs a manifest.json with id, name, type, and entry_point fields.
          </p>
        </div>
      </Panel>
    </div>
  );
}
