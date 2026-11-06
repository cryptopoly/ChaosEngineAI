import { useTranslation } from "react-i18next";
import { checkBackend } from "../api";
import type { useSettings, useWorkspace } from "../hooks";

type WorkspaceHook = ReturnType<typeof useWorkspace>;

/** "Restart required" prompt shown after the data directory changes. */
export function DataDirRestartModal({
  dataDirRestartPrompt,
  setDataDirRestartPrompt,
  setBusyAction,
  setBackendOnline,
  setError,
  refreshWorkspace,
  activeChatId,
}: {
  dataDirRestartPrompt: ReturnType<typeof useSettings>["dataDirRestartPrompt"];
  setDataDirRestartPrompt: ReturnType<typeof useSettings>["setDataDirRestartPrompt"];
  setBusyAction: WorkspaceHook["setBusyAction"];
  setBackendOnline: WorkspaceHook["setBackendOnline"];
  setError: (msg: string | null) => void;
  refreshWorkspace: WorkspaceHook["refreshWorkspace"];
  activeChatId: string;
}) {
  const { t: tCommon } = useTranslation("common");
  if (!dataDirRestartPrompt) return null;
  return (
    <div className="modal-overlay" onClick={() => setDataDirRestartPrompt(null)}>
      <div className="modal-content" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header"><h3>{tCommon("dataDirRestart.title", { defaultValue: "Restart required" })}</h3></div>
        <div className="modal-body">
          <p>{tCommon("dataDirRestart.body", { defaultValue: "The data directory has been changed. ChaosEngineAI needs to restart the backend to start reading and writing files in the new location." })}</p>
          {dataDirRestartPrompt.migration ? (
            <div className="control-stack">
              <p className="mono-text">{tCommon("dataDirRestart.from", { path: dataDirRestartPrompt.migration.from, defaultValue: "From: {path}" })}</p>
              <p className="mono-text">{tCommon("dataDirRestart.to", { path: dataDirRestartPrompt.migration.to, defaultValue: "To: {path}" })}</p>
              {dataDirRestartPrompt.migration.copied.length > 0 ? <p>{tCommon("dataDirRestart.copied", { items: dataDirRestartPrompt.migration.copied.join(", "), defaultValue: "Copied: {items}" })}</p> : null}
              {dataDirRestartPrompt.migration.skipped.length > 0 ? <p>{tCommon("dataDirRestart.skipped", { items: dataDirRestartPrompt.migration.skipped.join(", "), defaultValue: "Skipped (already present at destination): {items}" })}</p> : null}
            </div>
          ) : null}
        </div>
        <div className="modal-footer">
          <button className="secondary-button" type="button" onClick={() => setDataDirRestartPrompt(null)}>{tCommon("dataDirRestart.notNow", { defaultValue: "Not now" })}</button>
          <button className="primary-button" type="button" onClick={async () => {
            setDataDirRestartPrompt(null);
            setBusyAction(tCommon("dataDirRestart.restarting", { defaultValue: "Restarting backend..." }));
            try {
              const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
              await tauriInvoke("restart_backend_sidecar");
              // Backend may still be starting — retry until healthy.
              let online = await checkBackend();
              if (!online) {
                for (let i = 0; i < 5; i++) {
                  await new Promise((r) => setTimeout(r, 2000));
                  online = await checkBackend();
                  if (online) break;
                }
              }
              if (online) {
                setBackendOnline(true);
                await refreshWorkspace(activeChatId || undefined);
              } else {
                setError(tCommon("dataDirRestart.notBackOnline", { defaultValue: "Backend did not come back online after restart." }));
              }
            } catch (err) {
              setError(err instanceof Error ? err.message : tCommon("dataDirRestart.failed", { defaultValue: "Failed to restart backend." }));
            } finally {
              setBusyAction(null);
            }
          }}>{tCommon("dataDirRestart.confirm", { defaultValue: "OK, restart now" })}</button>
        </div>
      </div>
    </div>
  );
}
