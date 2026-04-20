import { useState, type SetStateAction } from "react";
import { Panel } from "../../components/Panel";
import type { SettingsDraft } from "../../types/chat";
import type { SidebarMode } from "../../types";

export interface SettingsTabProps {
  settingsDraft: SettingsDraft;
  onSettingsDraftChange: (action: SetStateAction<SettingsDraft>) => void;
  newDirectoryLabel: string;
  onNewDirectoryLabelChange: (label: string) => void;
  newDirectoryPath: string;
  onNewDirectoryPathChange: (path: string) => void;
  onPickDataDirectory: () => void;
  onPickImageOutputsDirectory: () => void;
  onPickVideoOutputsDirectory: () => void;
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
  apiToken: string | null;
  sidebarMode: SidebarMode;
  onSidebarModeChange: (mode: SidebarMode) => void;
}

// The Settings page used to be one long stack of seven panels — Appearance,
// Data Directory, Delivery Folders, Model Directories, Remote Providers,
// Hugging Face, and Integrations — which felt squished on the 2-column grid
// and forced users to scroll past unrelated controls to reach the one they
// wanted. We now group those panels into four logical sections and navigate
// between them with a horizontal sub-tab bar. (We tried matching the user's
// ``sidebarMode`` preference with a side-menu variant for collapsible users
// but it felt clunky at the viewport widths this app runs at — tabs always,
// for Settings only.)
type SettingsSectionId = "general" | "storage" | "providers" | "integrations";

interface SettingsSectionDef {
  id: SettingsSectionId;
  label: string;
}

const SETTINGS_SECTIONS: SettingsSectionDef[] = [
  { id: "general", label: "General" },
  { id: "storage", label: "Storage" },
  { id: "providers", label: "Providers" },
  { id: "integrations", label: "Integrations" },
];

export function SettingsTab({
  settingsDraft,
  onSettingsDraftChange,
  newDirectoryLabel,
  onNewDirectoryLabelChange,
  newDirectoryPath,
  onNewDirectoryPathChange,
  onPickDataDirectory,
  onPickImageOutputsDirectory,
  onPickVideoOutputsDirectory,
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
  apiToken,
  sidebarMode,
  onSidebarModeChange,
}: SettingsTabProps) {
  const integrationApiToken = apiToken ?? "<chaosengine-api-token>";
  // Section selection lives in component state because it's purely a UI
  // concern — there's no need to thread it through the App-level workspace
  // or persist it across reloads (the user lands on "General" each time,
  // which matches the macOS Settings idiom of opening to the first pane).
  const [activeSection, setActiveSection] = useState<SettingsSectionId>("general");

  // Resolve the effective paths for the delivery folders so we can show the
  // real value in each input rather than a placeholder-flavoured hint. The
  // frontend doesn't expand ``~`` — we leave that for the backend — so the
  // user sees the same shape the backend prints in logs. When the user
  // hasn't set an override we mark the row with a small "default" tag so
  // it's unambiguous whether the path is inherited or explicit.
  const effectiveDataDirectory = settingsDraft.dataDirectory || "~/.chaosengine";
  const imagesOutputsOverride = (settingsDraft.imageOutputsDirectory ?? "").trim();
  const videosOutputsOverride = (settingsDraft.videoOutputsDirectory ?? "").trim();
  const effectiveImagesOutputs = imagesOutputsOverride || `${effectiveDataDirectory}/images/outputs`;
  const effectiveVideosOutputs = videosOutputsOverride || `${effectiveDataDirectory}/videos/outputs`;

  const generalPanels = (
    <div className="content-grid settings-section-grid">
      <Panel
        title="Appearance"
        subtitle="Choose how the sidebar organises grouped tabs. Switches are instant and remembered across restarts."
      >
        <div className="control-stack">
          <div className="segmented" role="radiogroup" aria-label="Sidebar style">
            <button
              type="button"
              role="radio"
              aria-checked={sidebarMode === "collapsible"}
              className={sidebarMode === "collapsible" ? "segment active" : "segment"}
              onClick={() => onSidebarModeChange("collapsible")}
            >
              Collapsible
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={sidebarMode === "tabs"}
              className={sidebarMode === "tabs" ? "segment active" : "segment"}
              onClick={() => onSidebarModeChange("tabs")}
            >
              Tabs
            </button>
          </div>
          <p className="help-text">
            <strong>Collapsible</strong> shows all groups expanded with children listed inline — one click per
            destination. <strong>Tabs</strong> keeps the sidebar compact: groups behave like single buttons that jump
            to their last-used tab, with a sub-tab bar above the content.
          </p>
        </div>
      </Panel>
    </div>
  );

  // Show a "default" badge beside the path when the user hasn't set an
  // override so it's unambiguous whether the row is inherited from the data
  // directory or explicit. The data directory itself also gets a default
  // marker when it's empty (the fallback path is shown in the input).
  const dataDirectoryIsDefault = !settingsDraft.dataDirectory?.trim();
  const imagesOutputsIsDefault = !imagesOutputsOverride;
  const videosOutputsIsDefault = !videosOutputsOverride;

  const storagePanels = (
    <div className="settings-storage-grid">
      <div className="settings-storage-col">
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
                value={effectiveDataDirectory}
              />
              {dataDirectoryIsDefault ? <span className="badge muted">default</span> : null}
              <button className="secondary-button" type="button" onClick={() => void onPickDataDirectory()}>
                Browse...
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={dataDirectoryIsDefault}
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
          title="Delivery Folders"
          subtitle="Where newly generated images and videos land. Override the defaults to drop finished renders straight into a client folder, Dropbox sync, or an external SSD."
        >
          <div className="control-stack">
            <div className="field-label-row">
              <label className="field-label">Images</label>
              {imagesOutputsIsDefault ? <span className="badge muted">default</span> : null}
            </div>
            <div className="directory-add-row">
              <input
                className="text-input directory-add-path mono-text"
                type="text"
                readOnly
                value={effectiveImagesOutputs}
              />
              <button className="secondary-button" type="button" onClick={() => void onPickImageOutputsDirectory()}>
                Browse...
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={imagesOutputsIsDefault}
                onClick={() => onSettingsDraftChange((current) => ({ ...current, imageOutputsDirectory: "" }))}
              >
                Reset to default
              </button>
            </div>
            <div className="field-label-row">
              <label className="field-label">Videos</label>
              {videosOutputsIsDefault ? <span className="badge muted">default</span> : null}
            </div>
            <div className="directory-add-row">
              <input
                className="text-input directory-add-path mono-text"
                type="text"
                readOnly
                value={effectiveVideosOutputs}
              />
              <button className="secondary-button" type="button" onClick={() => void onPickVideoOutputsDirectory()}>
                Browse...
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={videosOutputsIsDefault}
                onClick={() => onSettingsDraftChange((current) => ({ ...current, videoOutputsDirectory: "" }))}
              >
                Reset to default
              </button>
            </div>
            <p className="help-text">
              New artifacts go to the folder you pick right away — no backend restart needed. Existing renders stay where
              they were written. Reset returns the row to the default under the Data Directory.
            </p>
          </div>
        </Panel>
      </div>
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
              placeholder="~/AI_Models"
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
    </div>
  );

  const providerPanels = (
    <div className="content-grid settings-section-grid">
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
    </div>
  );

  const integrationPanels = (
    <div className="content-grid settings-section-grid">
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
            { name: "Continue.dev (VS Code)", config: `{\n  "models": [{\n    "title": "ChaosEngineAI",\n    "provider": "openai",\n    "model": "${loadedModelName ?? "current-model"}",\n    "apiBase": "${serverLocalhostUrl ?? `http://127.0.0.1:${serverPort}/v1`}",\n    "apiKey": "${integrationApiToken}"\n  }]\n}` },
            { name: "Goose", config: `# In ~/.config/goose/config.yaml\nGOOSE_PROVIDER: openai\nGOOSE_MODEL: ${loadedModelName ?? "current-model"}\nOPENAI_BASE_URL: ${serverLocalhostUrl ?? `http://127.0.0.1:${serverPort}/v1`}\nOPENAI_API_KEY: ${integrationApiToken}` },
            { name: "Cursor", config: `1. Settings → Models → Add Model\n2. OpenAI API Key: ${integrationApiToken}\n3. Override OpenAI Base URL: ${serverLocalhostUrl ?? `http://127.0.0.1:${serverPort}/v1`}\n4. Add custom model: ${loadedModelName ?? "current-model"}` },
            { name: "Claude Code (via OpenAI proxy)", config: `# Set environment variables before running claude\nexport ANTHROPIC_BASE_URL=${serverLocalhostUrl ?? `http://127.0.0.1:${serverPort}/v1`}\nexport ANTHROPIC_AUTH_TOKEN=${integrationApiToken}` },
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

  // Keep this dispatch exhaustive — TypeScript's ``SettingsSectionId``
  // discriminant makes it a compile error to forget a branch when we add a
  // new section.
  const sectionContent =
    activeSection === "general"
      ? generalPanels
      : activeSection === "storage"
        ? storagePanels
        : activeSection === "providers"
          ? providerPanels
          : integrationPanels;

  // Horizontal sub-tab bar, always — the vertical side-menu variant we
  // tried originally felt clunky at the viewport widths this app runs at
  // (too much wasted horizontal space for four shortish labels). The
  // app-wide ``sidebarMode`` preference still controls the top-level
  // sidebar, but Settings always uses tabs regardless; the hint line on
  // each section lives in the panel subtitles instead.
  //
  // The inner body is a bare ``settings-content`` frame — each section
  // brings its own grid wrapper. That lets Storage use a 2-column layout
  // that stacks Data + Delivery on the left and gives Model Directories
  // the full right column to breathe, while Providers and Integrations
  // keep the standard 2-col ``content-grid``.
  return (
    <div className="settings-layout">
      <div className="subtab-bar settings-subtab-bar" role="tablist" aria-label="Settings sections">
        {SETTINGS_SECTIONS.map((section) => {
          const isActive = section.id === activeSection;
          return (
            <button
              key={section.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              className={isActive ? "subtab active" : "subtab"}
              onClick={() => setActiveSection(section.id)}
            >
              {section.label}
            </button>
          );
        })}
      </div>
      <div className="settings-content">{sectionContent}</div>
    </div>
  );
}
