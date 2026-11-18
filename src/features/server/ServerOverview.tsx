import { useEffect, useState } from "react";
import { maskSecret } from "../../utils/secretMask";
import { useTranslation } from "react-i18next";
import { StatCard } from "../../components/StatCard";
import { ModelLoadingProgress } from "../../components/ModelLoadingProgress";
import { busyEngineLines } from "../../utils/busyEngine";
import { ConnectPanel } from "./ConnectPanel";
import { RecentActivity } from "./RecentActivity";
import type { ModelEvent } from "./modelEvents";
import type { ServerTabProps } from "./ServerTab";

export interface ServerOverviewProps extends Omit<ServerTabProps, "showRemoteTest" | "testModelId" | "localHealthCurl" | "localModelsCurl" | "remoteHealthCurl" | "remoteModelsCurl" | "logEntries"> {
  recentEvents: ModelEvent[];
  onOpenActivity: () => void;
  copyText: (text: string) => void;
}

/** The Server page's Overview: status, API key, the connect line, warm models, stats and settings. */
export function ServerOverview({
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
  activeGenerations,
  settingsDraft,
  apiToken,
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
  recentEvents,
  onOpenActivity,
  copyText,
}: ServerOverviewProps) {
  const { t } = useTranslation("common");
  const [orphansDismissed, setOrphansDismissed] = useState(false);
  const [showApiToken, setShowApiToken] = useState(false);

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

  const maskedToken = apiToken ? maskSecret(apiToken) : null;
  const loadingRef = serverLoading?.modelRef ?? null;
  const connectModel =
    warmModels.find((m) => m.active)?.ref ?? warmModels[0]?.ref ?? "your-model-id";

  return (
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
          {busyEngineLines(t, activeGenerations).map((line) => (
            <p key={line} className="busy-indicator" role="status"><span className="busy-dot" />{line}</p>
          ))}
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

      <ConnectPanel
        localServerUrl={localServerUrl}
        apiToken={apiToken}
        connectModel={connectModel}
        onCopy={copyText}
      />

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
      <RecentActivity events={recentEvents} onOpenActivity={onOpenActivity} />
    </div>
  );
}
