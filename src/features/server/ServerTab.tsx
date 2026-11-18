import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import type { ActiveGeneration, LogEntry, ModelLoadingState, OrphanedWorker, WarmModel } from "../../types";
import type { SettingsDraft } from "../../types/chat";
import { useModelEvents } from "../../hooks/useModelEvents";
import { ServerActivity } from "./ServerActivity";
import { ServerOverview } from "./ServerOverview";
import { useServerView, type ServerViewId } from "./serverView";

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
  /** Image, video or music runs in flight: the API reads "idle" while one holds the GPU. */
  activeGenerations?: ActiveGeneration[];
  settingsDraft: SettingsDraft;
  /** Plain log lines, newest first (the old Logs page); shown under Activity. */
  logEntries: LogEntry[];
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

const VIEWS: ServerViewId[] = ["overview", "activity"];

/** The Server page: Overview (status, connect, warm models, settings) and Activity (model events and log lines). */
export function ServerTab(props: ServerTabProps) {
  const {
    backendOnline,
    localServerUrl,
    apiToken,
    logEntries,
    showRemoteTest,
    testModelId,
    localHealthCurl,
    localModelsCurl,
    remoteHealthCurl,
    remoteModelsCurl,
    remoteAccessActive,
    onShowRemoteTestChange,
  } = props;
  const { t } = useTranslation("common");
  const [view, setView] = useServerView();
  const feed = useModelEvents(backendOnline);

  function copyText(text: string) {
    void navigator.clipboard.writeText(text);
  }

  const viewLabel = (id: ServerViewId) =>
    id === "overview"
      ? t("serverActivity.views.overview", { defaultValue: "Overview" })
      : t("serverActivity.views.activity", { defaultValue: "Activity" });

  return (
    <div className="content-grid">
      <Panel
        title={t("tabs.server")}
        subtitle={t("tabCaptions.server")}
        className="span-2"
        actions={
          <div className="segmented server-view-switch" role="tablist" aria-label={t("serverActivity.views.label", { defaultValue: "Server views" })}>
            {VIEWS.map((id) => (
              <button
                key={id}
                type="button"
                role="tab"
                id={`server-view-${id}`}
                aria-selected={view === id}
                aria-controls="server-view-panel"
                className={`segment${view === id ? " active" : ""}`}
                onClick={() => setView(id)}
              >
                {viewLabel(id)}
              </button>
            ))}
          </div>
        }
      >
        <div className="server-layout" id="server-view-panel" role="tabpanel" aria-labelledby={`server-view-${view}`}>
          {view === "activity" ? (
            <ServerActivity feed={feed} logEntries={logEntries} onCopy={copyText} />
          ) : (
            <ServerOverview
              {...props}
              recentEvents={feed.events}
              onOpenActivity={() => setView("activity")}
              copyText={copyText}
            />
          )}
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
