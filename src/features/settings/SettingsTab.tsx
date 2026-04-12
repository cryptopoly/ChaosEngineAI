import type { SetStateAction } from "react";
import { Panel } from "../../components/Panel";
import type { SettingsDraft } from "../../types/chat";

export interface SettingsTabProps {
  settingsDraft: SettingsDraft;
  onSettingsDraftChange: (action: SetStateAction<SettingsDraft>) => void;
  newDirectoryLabel: string;
  onNewDirectoryLabelChange: (label: string) => void;
  newDirectoryPath: string;
  onNewDirectoryPathChange: (path: string) => void;
  onPickDataDirectory: () => void;
  onSaveSettings: () => void;
  onPickDirectory: (currentPath?: string) => Promise<string | null>;
  onAddDirectory: () => void;
  onUpdateDirectoryPath: (directoryId: string, nextPath: string) => void;
  onToggleDirectory: (directoryId: string) => void;
  onRemoveDirectory: (directoryId: string) => void;
  onCopyText: (text: string) => void;
  serverLocalhostUrl: string | undefined;
  serverPort: number;
  loadedModelName: string | undefined;
}

export function SettingsTab({
  settingsDraft,
  onSettingsDraftChange,
  newDirectoryLabel,
  onNewDirectoryLabelChange,
  newDirectoryPath,
  onNewDirectoryPathChange,
  onPickDataDirectory,
  onSaveSettings,
  onPickDirectory,
  onAddDirectory,
  onUpdateDirectoryPath,
  onToggleDirectory,
  onRemoveDirectory,
  onCopyText,
  serverLocalhostUrl,
  serverPort,
  loadedModelName,
}: SettingsTabProps) {
  return (
    <div className="content-grid">
      <Panel
        title="Data Directory"
        subtitle="Where ChaosEngineAI stores chat history, settings, and benchmark runs. Change to a cloud-synced folder (Dropbox, iCloud) to back up or share across machines."
      >
        <div className="control-stack">
          <div className="directory-add-row">
            <input
              className="text-input directory-add-path mono-text"
              type="text"
              readOnly
              value={settingsDraft.dataDirectory || "~/.chaosengine"}
            />
            <button className="secondary-button" type="button" onClick={() => void onPickDataDirectory()}>
              Browse...
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={() => onSettingsDraftChange((current) => ({ ...current, dataDirectory: "" }))}
            >
              Reset to default
            </button>
          </div>
          <p className="help-text">
            Changes take effect after the backend restarts. Existing data will be copied to the new location; the
            old files are left in place.
          </p>
        </div>
      </Panel>
      <Panel
        title="Model Directories"
        subtitle="Add the folders ChaosEngineAI should scan for local models, including custom, Ollama, LM Studio, or shared model paths."
        actions={
          <button className="primary-button" type="button" onClick={() => void onSaveSettings()}>
            Save settings
          </button>
        }
      >
        <div className="control-stack directory-stack">
          <div className="directory-add-row">
            <input
              className="text-input directory-add-label"
              type="text"
              placeholder="Label (e.g. LM Studio models)"
              value={newDirectoryLabel}
              onChange={(event) => onNewDirectoryLabelChange(event.target.value)}
            />
            <input
              className="text-input directory-add-path"
              type="text"
              placeholder="/Users/dan/AI_Models"
              value={newDirectoryPath}
              onChange={(event) => onNewDirectoryPathChange(event.target.value)}
            />
            <button
              className="secondary-button"
              type="button"
              onClick={async () => {
                const picked = await onPickDirectory(newDirectoryPath);
                if (picked) {
                  onNewDirectoryPathChange(picked);
                  if (!newDirectoryLabel.trim()) {
                    const leaf = picked.split(/[\\/]/).filter(Boolean).pop();
                    if (leaf) onNewDirectoryLabelChange(leaf);
                  }
                }
              }}
            >
              Browse…
            </button>
            <button className="secondary-button" type="button" onClick={onAddDirectory}>
              Add
            </button>
          </div>
          <div className="list scrollable-list directory-list">
            {settingsDraft.modelDirectories.map((directory) => (
              <div className="list-row directory-row" key={directory.id}>
                <div className="directory-row-info">
                  <strong>{directory.label}</strong>
                  <p className="mono-text">{directory.path}</p>
                </div>
                <div className="directory-row-actions">
                  <span className={`badge ${directory.exists ? "success" : "warning"}`}>
                    {directory.exists ? "Found" : "Missing"}
                  </span>
                  <span className="badge muted">{directory.modelCount ?? 0} models</span>
                  <button
                    className="secondary-button small-button"
                    type="button"
                    onClick={async () => {
                      const picked = await onPickDirectory(directory.path);
                      if (picked) onUpdateDirectoryPath(directory.id, picked);
                    }}
                  >
                    Change…
                  </button>
                  <button className="secondary-button small-button" type="button" onClick={() => onToggleDirectory(directory.id)}>
                    {directory.enabled ? "Disable" : "Enable"}
                  </button>
                  {directory.source === "user" ? (
                    <button className="secondary-button small-button" type="button" onClick={() => onRemoveDirectory(directory.id)}>
                      Remove
                    </button>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Panel>

      <Panel
        title="Remote Providers"
        subtitle="Configure cloud OpenAI-compatible APIs as a fallback. Keys are stored locally with 0600 permissions."
        actions={
          <button className="secondary-button" type="button" onClick={() => {
            const id = `remote-${Date.now()}`;
            onSettingsDraftChange((c) => ({
              ...c,
              remoteProviders: [...(c.remoteProviders ?? []), { id, label: "New Provider", apiBase: "https://api.openai.com/v1", apiKey: "", model: "gpt-4o-mini" }],
            }));
          }}>
            + Add Provider
          </button>
        }
      >
        <div className="control-stack">
          {(settingsDraft.remoteProviders ?? []).length === 0 ? (
            <p className="empty-state">No remote providers configured. Add one to use cloud models as a fallback.</p>
          ) : null}
          {(settingsDraft.remoteProviders ?? []).map((provider, idx) => (
            <div key={provider.id} className="remote-provider-card">
              <div className="field-grid">
                <label>
                  Label
                  <input
                    className="text-input"
                    type="text"
                    value={provider.label}
                    onChange={(event) => {
                      const next = [...(settingsDraft.remoteProviders ?? [])];
                      next[idx] = { ...next[idx], label: event.target.value };
                      onSettingsDraftChange((c) => ({ ...c, remoteProviders: next }));
                    }}
                  />
                </label>
                <label>
                  Model name
                  <input
                    className="text-input"
                    type="text"
                    value={provider.model}
                    onChange={(event) => {
                      const next = [...(settingsDraft.remoteProviders ?? [])];
                      next[idx] = { ...next[idx], model: event.target.value };
                      onSettingsDraftChange((c) => ({ ...c, remoteProviders: next }));
                    }}
                  />
                </label>
                <label>
                  API base URL (must be HTTPS)
                  <input
                    className="text-input"
                    type="url"
                    placeholder="https://api.openai.com/v1"
                    value={provider.apiBase}
                    onChange={(event) => {
                      const next = [...(settingsDraft.remoteProviders ?? [])];
                      next[idx] = { ...next[idx], apiBase: event.target.value };
                      onSettingsDraftChange((c) => ({ ...c, remoteProviders: next }));
                    }}
                  />
                </label>
                <label>
                  API key
                  <input
                    className="text-input"
                    type="password"
                    placeholder={provider.hasApiKey ? provider.apiKeyMasked || "•••• stored ••••" : "sk-..."}
                    value={provider.apiKey ?? ""}
                    onChange={(event) => {
                      const next = [...(settingsDraft.remoteProviders ?? [])];
                      next[idx] = { ...next[idx], apiKey: event.target.value };
                      onSettingsDraftChange((c) => ({ ...c, remoteProviders: next }));
                    }}
                  />
                </label>
              </div>
              <div className="button-row">
                <button className="secondary-button" type="button" onClick={() => {
                  const next = (settingsDraft.remoteProviders ?? []).filter((_, i) => i !== idx);
                  onSettingsDraftChange((c) => ({ ...c, remoteProviders: next }));
                }}>
                  Remove
                </button>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel
        title="Hugging Face"
        subtitle="Required for gated models like Mistral, Llama, Gemma. Get one at huggingface.co/settings/tokens"
        actions={
          <button className="primary-button" type="button" onClick={() => void onSaveSettings()}>
            Save settings
          </button>
        }
      >
        <div className="control-stack">
          <label>
            Hugging Face token
            <input
              className="text-input"
              type="password"
              placeholder={
                settingsDraft.hasHuggingFaceToken
                  ? settingsDraft.huggingFaceTokenMasked || "•••• stored ••••"
                  : "hf_..."
              }
              value={settingsDraft.huggingFaceToken}
              onChange={(event) =>
                onSettingsDraftChange((c) => ({ ...c, huggingFaceToken: event.target.value }))
              }
            />
          </label>
          <p className="muted-text">
            Stored locally. Used by MLX conversion when fetching gated models from Hugging Face.
          </p>
        </div>
      </Panel>

      <Panel
        title="Integrations"
        subtitle="Connect external tools to ChaosEngineAI's OpenAI-compatible API."
        className="settings-integrations-panel"
      >
        <div className="control-stack">
          <p className="muted-text">
            Use these snippets to connect popular tools to ChaosEngineAI as their LLM backend. The server must be running on{" "}
            <span className="mono-text">{serverLocalhostUrl ?? `http://127.0.0.1:${serverPort}/v1`}</span>.
          </p>
          {[
            { name: "Continue.dev (VS Code)", config: `{\n  "models": [{\n    "title": "ChaosEngineAI",\n    "provider": "openai",\n    "model": "${loadedModelName ?? "current-model"}",\n    "apiBase": "${serverLocalhostUrl ?? `http://127.0.0.1:${serverPort}/v1`}",\n    "apiKey": "not-needed"\n  }]\n}` },
            { name: "Goose", config: `# In ~/.config/goose/config.yaml\nGOOSE_PROVIDER: openai\nGOOSE_MODEL: ${loadedModelName ?? "current-model"}\nOPENAI_BASE_URL: ${serverLocalhostUrl ?? `http://127.0.0.1:${serverPort}/v1`}\nOPENAI_API_KEY: not-needed` },
            { name: "Cursor", config: `1. Settings → Models → Add Model\n2. OpenAI API Key: not-needed\n3. Override OpenAI Base URL: ${serverLocalhostUrl ?? `http://127.0.0.1:${serverPort}/v1`}\n4. Add custom model: ${loadedModelName ?? "current-model"}` },
            { name: "Claude Code (via OpenAI proxy)", config: `# Set environment variables before running claude\nexport ANTHROPIC_BASE_URL=${serverLocalhostUrl ?? `http://127.0.0.1:${serverPort}/v1`}\nexport ANTHROPIC_AUTH_TOKEN=not-needed` },
          ].map((item) => (
            <div key={item.name} className="integration-card">
              <div className="integration-card-header">
                <strong>{item.name}</strong>
                <button className="secondary-button" type="button" onClick={() => onCopyText(item.config)}>
                  Copy
                </button>
              </div>
              <pre className="integration-snippet">{item.config}</pre>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
