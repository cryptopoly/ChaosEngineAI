import { useEffect, useRef, useState } from "react";
import { Panel } from "../../components/Panel";
import { StatCard } from "../../components/StatCard";
import { ModelLoadingProgress } from "../../components/ModelLoadingProgress";
import type { LogEntry, ModelLoadingState, OrphanedWorker, WarmModel } from "../../types";
import type { SettingsDraft } from "../../types/chat";

export interface ServerTabProps {
  serverStatus: "running" | "idle";
  serverPort: number;
  localServerUrl: string;
  primaryLanUrl: string | null;
  primaryLanOrigin: string | null;
  remoteAccessActive: boolean;
  remoteAccessRequested: boolean;
  preferredPortUnavailable: boolean;
  busyAction: string | null;
  busy: boolean;
  backendOnline: boolean;
  warmModels: WarmModel[];
  serverLoading: ModelLoadingState | null;
  requestsServed: number;
  activeConnections: number;
  engineLabel: string;
  settingsDraft: SettingsDraft;
  serverLogEntries: LogEntry[];
  showRemoteTest: boolean;
  testModelId: string | null;
  apiToken: string | null;
  localHealthCurl: string;
  localModelsCurl: string;
  remoteHealthCurl: string | null;
  remoteModelsCurl: string | null;
  recentOrphanedWorkers: OrphanedWorker[];
  selectedServerOptionKey: string | undefined;
  onOpenModelSelector: (action: "chat" | "server" | "thread", preselectedKey?: string) => void;
  onRestartServer: () => void;
  onStopServer: () => void;
  onLoadModel: (payload: {
    modelRef: string;
    modelName?: string;
    source?: string;
    backend?: string;
    path?: string;
    busyLabel?: string;
  }) => void;
  onUnloadWarmModel: (ref: string) => void;
  onSaveSettings: () => void;
  onSettingsDraftChange: React.Dispatch<React.SetStateAction<SettingsDraft>>;
  onShowRemoteTestChange: (show: boolean) => void;
  onTestModelIdChange: (id: string | null) => void;
}

export function ServerTab({
  serverStatus,
  serverPort,
  localServerUrl,
  primaryLanUrl,
  primaryLanOrigin,
  remoteAccessActive,
  remoteAccessRequested,
  preferredPortUnavailable,
  busyAction,
  busy,
  backendOnline,
  warmModels,
  serverLoading,
  requestsServed,
  activeConnections,
  engineLabel,
  settingsDraft,
  serverLogEntries,
  showRemoteTest,
  testModelId,
  apiToken,
  localHealthCurl,
  localModelsCurl,
  remoteHealthCurl,
  remoteModelsCurl,
  recentOrphanedWorkers,
  selectedServerOptionKey,
  onOpenModelSelector,
  onRestartServer,
  onStopServer,
  onLoadModel,
  onUnloadWarmModel,
  onSaveSettings,
  onSettingsDraftChange,
  onShowRemoteTestChange,
  onTestModelIdChange,
}: ServerTabProps) {
  const serverLogRef = useRef<HTMLDivElement>(null);
  const [serverLogAtBottom, setServerLogAtBottom] = useState(true);
  const [orphansDismissed, setOrphansDismissed] = useState(false);

  // Reset the local dismissal whenever a fresh batch of orphans arrives
  // (first PID in the list changes = new event), and auto-hide the banner
  // 12s after the latest batch so it behaves like a notification instead
  // of persistent state.
  const latestOrphanPid = recentOrphanedWorkers[0]?.pid;
  useEffect(() => {
    if (latestOrphanPid == null) return;
    setOrphansDismissed(false);
    const timer = window.setTimeout(() => setOrphansDismissed(true), 12_000);
    return () => window.clearTimeout(timer);
  }, [latestOrphanPid]);

  function handleServerLogScroll() {
    const el = serverLogRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    setServerLogAtBottom(atBottom);
  }

  function scrollServerLogToBottom() {
    if (serverLogRef.current) {
      serverLogRef.current.scrollTop = serverLogRef.current.scrollHeight;
      setServerLogAtBottom(true);
    }
  }

  useEffect(() => {
    if (serverLogAtBottom && serverLogRef.current) {
      serverLogRef.current.scrollTop = serverLogRef.current.scrollHeight;
    }
  }, [serverLogEntries, serverLogAtBottom]);

  function copyText(text: string) {
    void navigator.clipboard.writeText(text);
  }

  const [showApiToken, setShowApiToken] = useState(false);
  const maskedToken = apiToken ? `${apiToken.slice(0, 4)}${"•".repeat(Math.max(apiToken.length - 8, 4))}${apiToken.slice(-4)}` : null;

  const loadingRef = serverLoading?.modelRef ?? null;

  return (
    <div className="content-grid">
      <Panel
        title="Server"
        subtitle="OpenAI-compatible local API"
        className="span-2"
      >
        <div className="server-layout">
          <div className="server-main-col">
            <div className="server-status-strip">
              <div className="server-status-copy">
                <div className="server-status-heading">
                  <span className={`badge ${serverStatus === "running" ? "success" : "warning"}`}>
                    {serverStatus.toUpperCase()}
                  </span>
                  <h3>{localServerUrl}</h3>
                </div>
                {remoteAccessActive && primaryLanUrl && (
                  <p className="mono-text muted-text">{primaryLanUrl}</p>
                )}
                {busyAction ? (
                  <p className="busy-indicator"><span className="busy-dot" />{busyAction}</p>
                ) : null}
              </div>
              <div className="button-row server-actions">
                <button
                  className="primary-button"
                  type="button"
                  onClick={() => onOpenModelSelector("server", selectedServerOptionKey)}
                  disabled={busy || !backendOnline}
                >
                  Load Model
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => void onRestartServer()}
                  disabled={busy || !backendOnline}
                >
                  Restart
                </button>
                <button
                  className="secondary-button danger-button"
                  type="button"
                  onClick={() => void onStopServer()}
                  disabled={busy || !backendOnline}
                >
                  Stop
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => { onTestModelIdChange(null); onShowRemoteTestChange(true); }}
                >
                  Test
                </button>
              </div>
            </div>

            <div className="server-api-token-row">
              <div className="server-api-token-label">
                <strong>API key</strong>
                <small className="muted-text">Required as <span className="mono-text">Authorization: Bearer …</span> on every /v1 and /api call.</small>
              </div>
              <div className="server-api-token-value">
                <span className="mono-text">
                  {apiToken ? (showApiToken ? apiToken : maskedToken) : "(waiting for backend…)"}
                </span>
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={!apiToken}
                    onClick={() => setShowApiToken((v) => !v)}
                  >
                    {showApiToken ? "Hide" : "Reveal"}
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={!apiToken}
                    onClick={() => apiToken && copyText(apiToken)}
                  >
                    Copy
                  </button>
                </div>
              </div>
            </div>

            {recentOrphanedWorkers.length > 0 && !orphansDismissed ? (
              <div className="callout warning server-warning-callout">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <strong>Orphaned backend workers were cleaned up</strong>
                  <button
                    type="button"
                    className="callout-dismiss-btn"
                    onClick={() => setOrphansDismissed(true)}
                    title="Dismiss"
                    aria-label="Dismiss orphan worker notification"
                  >
                    &times;
                  </button>
                </div>
                <p>
                  ChaosEngineAI recently found and terminated {recentOrphanedWorkers.length} untracked worker
                  {recentOrphanedWorkers.length === 1 ? "" : "s"} left behind by an older load or crash.
                </p>
                <p className="mono-text muted-text">
                  {recentOrphanedWorkers
                    .slice(0, 4)
                    .map((worker) => `${worker.label} pid ${worker.pid} (${worker.action})`)
                    .join(" · ")}
                </p>
              </div>
            ) : null}

            {(() => {
              const loadingName = serverLoading?.modelName ?? null;
              if (warmModels.length === 0) {
                return null;
              }
              return (
                <div className="warm-pool-list">
                  {warmModels.map((w) => {
                    const isLoading = loadingRef === w.ref;
                    const badgeClass = w.active ? "success" : isLoading ? "accent" : "muted";
                    const badgeLabel = w.active ? "ACTIVE" : isLoading ? "LOADING" : "WARM";
                    const endpoint = `${localServerUrl}  ${w.ref}`;
                    return (
                      <div key={w.ref} className={`warm-pool-row${w.active ? " active" : ""}${isLoading ? " loading" : ""}`}>
                        <div className="row-meta">
                          <div className="row-meta-head">
                            <span className={`badge ${badgeClass}`}>{badgeLabel}</span>
                            <h4>{w.name}</h4>
                            <small className="row-engine">{w.engine}</small>
                          </div>
                          <div className="row-endpoint">
                            <p className="mono-text">{localServerUrl}</p>
                            <p className="mono-text muted-text">model id: {w.ref}</p>
                            <button
                              className="secondary-button"
                              type="button"
                              onClick={() => copyText(endpoint)}
                            >
                              Copy
                            </button>
                          </div>
                          {isLoading && serverLoading ? (
                            <ModelLoadingProgress loading={serverLoading} />
                          ) : null}
                        </div>
                        <div className="row-actions button-row">
                          <button
                            className="primary-button"
                            type="button"
                            disabled={w.active || busy || !backendOnline}
                            onClick={() => void onLoadModel({ modelRef: w.ref, modelName: w.name, source: "warm-pool" })}
                          >
                            Activate
                          </button>
                          <button
                            className="secondary-button"
                            type="button"
                            disabled={busy || !backendOnline}
                            onClick={() => void onUnloadWarmModel(w.ref)}
                          >
                            Unload
                          </button>
                          <button
                            className="secondary-button"
                            type="button"
                            onClick={() => { onTestModelIdChange(w.ref); onShowRemoteTestChange(true); }}
                          >
                            Test
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })()}

            <div className="stat-grid server-stat-grid">
              <StatCard label="Port" value={String(serverPort)} hint={preferredPortUnavailable ? "Preferred port is busy" : "Active"} />
              <StatCard
                label="Active"
                value={warmModels.find((m) => m.active)?.name ?? "None"}
                hint={engineLabel}
              />
              <StatCard
                label="Warm pool"
                value={String(warmModels.length)}
                hint={`${warmModels.filter((m) => m.warm).length} warm`}
              />
              <StatCard
                label="LAN"
                value={remoteAccessActive ? "Enabled" : "Local only"}
                hint={
                  remoteAccessRequested && !remoteAccessActive
                    ? "Restart to enable LAN"
                    : remoteAccessActive
                      ? primaryLanOrigin ?? "0.0.0.0"
                      : "Localhost"
                }
              />
              <StatCard label="Requests" value={String(requestsServed)} hint={`${activeConnections} active`} />
            </div>

            <div className="server-compact-settings">
              <div className="server-settings-row">
                <label>
                  Port
                  <input
                    className="text-input"
                    type="number"
                    min="1024"
                    max="65535"
                    disabled={busy}
                    value={settingsDraft.preferredServerPort}
                    onChange={(event) => onSettingsDraftChange((c) => ({ ...c, preferredServerPort: Number(event.target.value) }))}
                  />
                </label>
                <label className="check-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.allowRemoteConnections}
                    disabled={busy}
                    onChange={(event) => onSettingsDraftChange((c) => ({ ...c, allowRemoteConnections: event.target.checked }))}
                  />
                  LAN access
                </label>
                <label className="check-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.autoStartServer}
                    disabled={busy}
                    onChange={(event) => onSettingsDraftChange((c) => ({ ...c, autoStartServer: event.target.checked }))}
                  />
                  Auto-start
                </label>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => void onSaveSettings()}
                  disabled={busy || !backendOnline}
                >
                  Save
                </button>
              </div>
            </div>

            <div className="server-log-panel">
              <span className="eyebrow">Dev Log</span>
              <div className="server-log-scroll" ref={serverLogRef} onScroll={handleServerLogScroll}>
                {serverLogEntries.length > 0 ? (
                  serverLogEntries.map((entry, i) => (
                    <div className="server-log-line" key={`${entry.ts}-${entry.source}-${i}`}>
                      <small className="server-log-ts">{entry.ts}</small>
                      <small className="mono-text muted-text">/{entry.source}</small>
                      <span className={`log-level ${entry.level}`}>{entry.level}</span>
                      <span>{entry.message}</span>
                    </div>
                  ))
                ) : (
                  <div className="server-log-line">
                    <span className="server-log-placeholder">No log lines yet.</span>
                  </div>
                )}
              </div>
              {!serverLogAtBottom && serverLogEntries.length > 0 ? (
                <button
                  className="server-log-jump"
                  type="button"
                  onClick={scrollServerLogToBottom}
                >
                  Latest
                </button>
              ) : null}
            </div>
          </div>
        </div>
      </Panel>

      {showRemoteTest ? (
        <div className="modal-overlay" onClick={() => onShowRemoteTestChange(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>API Test Commands</h3>
              <p>Copy these commands to test the server from a terminal.{testModelId ? ` Pre-filled for ${testModelId}.` : ""}</p>
            </div>
            <div className="modal-body">
              <div className="server-command-list">
                <div className="server-command">
                  <div className="server-command-header">
                    <strong>Health check</strong>
                    <button className="secondary-button" type="button" onClick={() => copyText(localHealthCurl)}>Copy</button>
                  </div>
                  <p className="mono-text">{localHealthCurl}</p>
                </div>
                <div className="server-command">
                  <div className="server-command-header">
                    <strong>List models</strong>
                    <button className="secondary-button" type="button" onClick={() => copyText(localModelsCurl)}>Copy</button>
                  </div>
                  <p className="mono-text">{localModelsCurl}</p>
                </div>
                {testModelId ? (() => {
                  const authFlag = apiToken
                    ? ` -H 'Authorization: Bearer ${apiToken}'`
                    : " -H 'Authorization: Bearer <chaosengine-api-token>'";
                  const cmd = `curl -sS ${localServerUrl}/chat/completions -H 'Content-Type: application/json'${authFlag} -d '{"model":"${testModelId}","messages":[{"role":"user","content":"Hello"}]}'`;
                  return (
                    <div className="server-command">
                      <div className="server-command-header">
                        <strong>Chat completion ({testModelId})</strong>
                        <button className="secondary-button" type="button" onClick={() => copyText(cmd)}>Copy</button>
                      </div>
                      <p className="mono-text">{cmd}</p>
                    </div>
                  );
                })() : null}
                {remoteAccessActive && remoteHealthCurl ? (
                  <>
                    <div className="server-command">
                      <div className="server-command-header">
                        <strong>Remote health</strong>
                        <button className="secondary-button" type="button" onClick={() => copyText(remoteHealthCurl)}>Copy</button>
                      </div>
                      <p className="mono-text">{remoteHealthCurl}</p>
                    </div>
                    {remoteModelsCurl ? (
                      <div className="server-command">
                        <div className="server-command-header">
                          <strong>Remote models</strong>
                          <button className="secondary-button" type="button" onClick={() => copyText(remoteModelsCurl)}>Copy</button>
                        </div>
                        <p className="mono-text">{remoteModelsCurl}</p>
                      </div>
                    ) : null}
                  </>
                ) : null}
              </div>
            </div>
            <div className="modal-footer">
              <button className="secondary-button" type="button" onClick={() => onShowRemoteTestChange(false)}>Close</button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
