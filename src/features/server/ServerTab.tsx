import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation("common");
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
        title={t("tabs.server")}
        subtitle={t("tabCaptions.server")}
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
                  {t("serverTab.loadModel", { defaultValue: "Load Model" })}
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => void onRestartServer()}
                  disabled={busy || !backendOnline}
                >
                  {t("serverTab.restart", { defaultValue: "Restart" })}
                </button>
                <button
                  className="secondary-button danger-button"
                  type="button"
                  onClick={() => void onStopServer()}
                  disabled={busy || !backendOnline}
                >
                  {t("serverTab.stop", { defaultValue: "Stop" })}
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => { onTestModelIdChange(null); onShowRemoteTestChange(true); }}
                >
                  {t("serverTab.test", { defaultValue: "Test" })}
                </button>
              </div>
            </div>

            <div className="server-api-token-row">
              <div className="server-api-token-label">
                <strong>{t("serverTab.apiKey", { defaultValue: "API key" })}</strong>
                <small className="muted-text">{t("serverTab.apiKeyHint", { defaultValue: "Required as Authorization: Bearer … on every /v1 and /api call." })}</small>
              </div>
              <div className="server-api-token-value">
                <span className="mono-text">
                  {apiToken
                    ? (showApiToken ? apiToken : maskedToken)
                    : t("serverTab.waitingForBackend", { defaultValue: "(waiting for backend…)" })}
                </span>
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={!apiToken}
                    onClick={() => setShowApiToken((v) => !v)}
                  >
                    {showApiToken
                      ? t("serverTab.hide", { defaultValue: "Hide" })
                      : t("serverTab.reveal", { defaultValue: "Reveal" })}
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={!apiToken}
                    onClick={() => apiToken && copyText(apiToken)}
                  >
                    {t("serverTab.copy", { defaultValue: "Copy" })}
                  </button>
                </div>
              </div>
            </div>

            {recentOrphanedWorkers.length > 0 && !orphansDismissed ? (
              <div className="callout warning server-warning-callout">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <strong>{t("serverTab.orphans.title", { defaultValue: "Orphaned backend workers were cleaned up" })}</strong>
                  <button
                    type="button"
                    className="callout-dismiss-btn"
                    onClick={() => setOrphansDismissed(true)}
                    title={t("serverTab.orphans.dismiss", { defaultValue: "Dismiss" })}
                    aria-label={t("serverTab.orphans.dismissAria", { defaultValue: "Dismiss orphan worker notification" })}
                  >
                    &times;
                  </button>
                </div>
                <p>
                  {t("serverTab.orphans.body", {
                    count: recentOrphanedWorkers.length,
                    defaultValue: "{count, plural, one {ChaosEngineAI recently found and terminated # untracked worker left behind by an older load or crash.} other {ChaosEngineAI recently found and terminated # untracked workers left behind by an older load or crash.}}",
                  })}
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
                    const badgeLabel = w.active
                      ? t("serverTab.badgeActive", { defaultValue: "ACTIVE" })
                      : isLoading
                        ? t("serverTab.badgeLoading", { defaultValue: "LOADING" })
                        : t("serverTab.badgeWarm", { defaultValue: "WARM" });
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
                            <p className="mono-text muted-text">{t("serverTab.modelIdLabel", { ref: w.ref, defaultValue: "model id: {ref}" })}</p>
                            <button
                              className="secondary-button"
                              type="button"
                              onClick={() => copyText(endpoint)}
                            >
                              {t("serverTab.copy", { defaultValue: "Copy" })}
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
                            {t("serverTab.activate", { defaultValue: "Activate" })}
                          </button>
                          <button
                            className="secondary-button"
                            type="button"
                            disabled={busy || !backendOnline}
                            onClick={() => void onUnloadWarmModel(w.ref)}
                          >
                            {t("serverTab.unload", { defaultValue: "Unload" })}
                          </button>
                          <button
                            className="secondary-button"
                            type="button"
                            onClick={() => { onTestModelIdChange(w.ref); onShowRemoteTestChange(true); }}
                          >
                            {t("serverTab.test", { defaultValue: "Test" })}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })()}

            <div className="stat-grid server-stat-grid">
              <StatCard
                label={t("serverTab.stat.port", { defaultValue: "Port" })}
                value={String(serverPort)}
                hint={preferredPortUnavailable
                  ? t("serverTab.stat.portBusy", { defaultValue: "Preferred port is busy" })
                  : t("serverTab.stat.portActive", { defaultValue: "Active" })}
              />
              <StatCard
                label={t("serverTab.stat.active", { defaultValue: "Active" })}
                value={warmModels.find((m) => m.active)?.name ?? t("serverTab.stat.none", { defaultValue: "None" })}
                hint={engineLabel}
              />
              <StatCard
                label={t("serverTab.stat.warmPool", { defaultValue: "Warm pool" })}
                value={String(warmModels.length)}
                hint={t("serverTab.stat.warmCount", { count: warmModels.filter((m) => m.warm).length, defaultValue: "{count} warm" })}
              />
              <StatCard
                label={t("serverTab.stat.lan", { defaultValue: "LAN" })}
                value={remoteAccessActive
                  ? t("serverTab.stat.lanEnabled", { defaultValue: "Enabled" })
                  : t("serverTab.stat.lanLocalOnly", { defaultValue: "Local only" })}
                hint={
                  remoteAccessRequested && !remoteAccessActive
                    ? t("serverTab.stat.restartToEnableLan", { defaultValue: "Restart to enable LAN" })
                    : remoteAccessActive
                      ? primaryLanOrigin ?? "0.0.0.0"
                      : t("serverTab.stat.localhost", { defaultValue: "Localhost" })
                }
              />
              <StatCard
                label={t("serverTab.stat.requests", { defaultValue: "Requests" })}
                value={String(requestsServed)}
                hint={t("serverTab.stat.activeConnections", { count: activeConnections, defaultValue: "{count} active" })}
              />
            </div>

            <div className="server-compact-settings">
              <div className="server-settings-row">
                <label>
                  {t("serverTab.settings.port", { defaultValue: "Port" })}
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
                  {t("serverTab.settings.lanAccess", { defaultValue: "LAN access" })}
                </label>
                <label
                  className="check-row"
                  title={t("serverTab.settings.requireApiAuthTooltip", { defaultValue: "Disable to let external clients (OpenWebUI, curl, other apps) hit /api and /v1 without a bearer token. Leave on for local-only use." })}
                >
                  <input
                    type="checkbox"
                    checked={settingsDraft.requireApiAuth}
                    disabled={busy}
                    onChange={(event) => onSettingsDraftChange((c) => ({ ...c, requireApiAuth: event.target.checked }))}
                  />
                  {t("serverTab.settings.requireApiToken", { defaultValue: "Require API token" })}
                </label>
                <label className="check-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.autoStartServer}
                    disabled={busy}
                    onChange={(event) => onSettingsDraftChange((c) => ({ ...c, autoStartServer: event.target.checked }))}
                  />
                  {t("serverTab.settings.autoStart", { defaultValue: "Auto-start" })}
                </label>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => void onSaveSettings()}
                  disabled={busy || !backendOnline}
                >
                  {t("serverTab.settings.save", { defaultValue: "Save" })}
                </button>
              </div>
            </div>

            <div className="server-log-panel">
              <span className="eyebrow">{t("serverTab.devLog", { defaultValue: "Dev Log" })}</span>
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
                    <span className="server-log-placeholder">{t("serverTab.noLogLines", { defaultValue: "No log lines yet." })}</span>
                  </div>
                )}
              </div>
              {!serverLogAtBottom && serverLogEntries.length > 0 ? (
                <button
                  className="server-log-jump"
                  type="button"
                  onClick={scrollServerLogToBottom}
                >
                  {t("serverTab.latest", { defaultValue: "Latest" })}
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
              <h3>{t("serverTab.testModal.title", { defaultValue: "API Test Commands" })}</h3>
              <p>
                {t("serverTab.testModal.subtitle", { defaultValue: "Copy these commands to test the server from a terminal." })}
                {testModelId ? ` ${t("serverTab.testModal.preFilledFor", { id: testModelId, defaultValue: "Pre-filled for {id}." })}` : ""}
              </p>
            </div>
            <div className="modal-body">
              <div className="server-command-list">
                <div className="server-command">
                  <div className="server-command-header">
                    <strong>{t("serverTab.testModal.healthCheck", { defaultValue: "Health check" })}</strong>
                    <button className="secondary-button" type="button" onClick={() => copyText(localHealthCurl)}>{t("serverTab.copy", { defaultValue: "Copy" })}</button>
                  </div>
                  <p className="mono-text">{localHealthCurl}</p>
                </div>
                <div className="server-command">
                  <div className="server-command-header">
                    <strong>{t("serverTab.testModal.listModels", { defaultValue: "List models" })}</strong>
                    <button className="secondary-button" type="button" onClick={() => copyText(localModelsCurl)}>{t("serverTab.copy", { defaultValue: "Copy" })}</button>
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
                        <strong>{t("serverTab.testModal.chatCompletion", { id: testModelId, defaultValue: "Chat completion ({id})" })}</strong>
                        <button className="secondary-button" type="button" onClick={() => copyText(cmd)}>{t("serverTab.copy", { defaultValue: "Copy" })}</button>
                      </div>
                      <p className="mono-text">{cmd}</p>
                    </div>
                  );
                })() : null}
                {remoteAccessActive && remoteHealthCurl ? (
                  <>
                    <div className="server-command">
                      <div className="server-command-header">
                        <strong>{t("serverTab.testModal.remoteHealth", { defaultValue: "Remote health" })}</strong>
                        <button className="secondary-button" type="button" onClick={() => copyText(remoteHealthCurl)}>{t("serverTab.copy", { defaultValue: "Copy" })}</button>
                      </div>
                      <p className="mono-text">{remoteHealthCurl}</p>
                    </div>
                    {remoteModelsCurl ? (
                      <div className="server-command">
                        <div className="server-command-header">
                          <strong>{t("serverTab.testModal.remoteModels", { defaultValue: "Remote models" })}</strong>
                          <button className="secondary-button" type="button" onClick={() => copyText(remoteModelsCurl)}>{t("serverTab.copy", { defaultValue: "Copy" })}</button>
                        </div>
                        <p className="mono-text">{remoteModelsCurl}</p>
                      </div>
                    ) : null}
                  </>
                ) : null}
              </div>
            </div>
            <div className="modal-footer">
              <button className="secondary-button" type="button" onClick={() => onShowRemoteTestChange(false)}>{t("serverTab.testModal.close", { defaultValue: "Close" })}</button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
