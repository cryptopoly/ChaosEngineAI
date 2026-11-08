import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getAgentAccess, updateAgentAccess } from "../../api";
import type { AgentAccess } from "../../api";
import type { AgentAccessPatch } from "../../api/admin";
import { Panel } from "../../components/Panel";

interface AgentAccessPanelProps {
  backendOnline: boolean;
  onCopyText: (text: string) => void;
}

/** Quote a header value for a POSIX shell; the token is URL-safe so this is belt and braces. */
function shellQuote(value: string): string {
  return `"${value.replace(/(["\\$`])/g, "\\$1")}"`;
}

export function agentConfigLine(access: AgentAccess, token: string = access.token): string {
  return `claude mcp add --transport http chaosengine ${access.mcpUrl} --header ${shellQuote(`Authorization: Bearer ${token}`)}`;
}

export function agentStdioLine(access: AgentAccess): string | null {
  if (!access.stdioCommand) return null;
  return `claude mcp add chaosengine -- ${access.stdioCommand.map(shellQuote).join(" ")}`;
}

export function agentConfigJson(access: AgentAccess): string {
  return JSON.stringify(
    {
      mcpServers: {
        chaosengine: {
          type: "http",
          url: access.mcpUrl,
          headers: { Authorization: `Bearer ${access.token}` },
        },
      },
    },
    null,
    2,
  );
}

export function AgentAccessPanel({ backendOnline, onCopyText }: AgentAccessPanelProps) {
  const { t } = useTranslation("common");
  const [access, setAccess] = useState<AgentAccess | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!backendOnline) return;
    let cancelled = false;
    getAgentAccess()
      .then((value) => {
        if (!cancelled) setAccess(value);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [backendOnline]);

  const apply = useCallback(async (patch: AgentAccessPatch) => {
    setBusy(true);
    setError(null);
    try {
      setAccess(await updateAgentAccess(patch));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <Panel
      title={t("settingsTab.agentAccess.title", { defaultValue: "Agent access" })}
      subtitle={t("settingsTab.agentAccess.subtitle", {
        defaultValue: "Let an AI agent (Claude Code, Cursor and others) make images, video, speech and transcripts here.",
      })}
      className="settings-integrations-panel"
    >
      <div className="control-stack">
        <label className="check-row">
          <input
            type="checkbox"
            checked={access?.enabled ?? false}
            disabled={busy || !backendOnline || access === null}
            onChange={(event) => void apply({ enabled: event.target.checked })}
          />
          {t("settingsTab.agentAccess.switch", { defaultValue: "Allow agents to use ChaosEngineAI" })}
        </label>
        <p className="muted-text">
          {t("settingsTab.agentAccess.hint", {
            defaultValue:
              "Off by default. Agents run one job at a time, only on this computer, and need the token below. They can only use models you have already downloaded.",
          })}
        </p>
        {error ? <p className="error-text">{error}</p> : null}
        {access ? (
          <>
            <div className="integration-card">
              <div className="integration-card-header">
                <strong>{t("settingsTab.agentAccess.claudeCode", { defaultValue: "Claude Code" })}</strong>
                <button className="secondary-button" type="button" onClick={() => onCopyText(agentConfigLine(access))}>
                  {t("settingsTab.integrations.copyButton", { defaultValue: "Copy" })}
                </button>
              </div>
              <pre className="integration-snippet">{agentConfigLine(access)}</pre>
            </div>
            <div className="integration-card">
              <div className="integration-card-header">
                <strong>{t("settingsTab.agentAccess.otherClients", { defaultValue: "Other MCP clients (JSON)" })}</strong>
                <button className="secondary-button" type="button" onClick={() => onCopyText(agentConfigJson(access))}>
                  {t("settingsTab.integrations.copyButton", { defaultValue: "Copy" })}
                </button>
              </div>
              <pre className="integration-snippet">{agentConfigJson(access)}</pre>
            </div>
            {agentStdioLine(access) ? (
              <div className="integration-card">
                <div className="integration-card-header">
                  <strong>{t("settingsTab.agentAccess.stdio", { defaultValue: "Clients that only start local programs (stdio)" })}</strong>
                  <button className="secondary-button" type="button" onClick={() => onCopyText(agentStdioLine(access) ?? "")}>
                    {t("settingsTab.integrations.copyButton", { defaultValue: "Copy" })}
                  </button>
                </div>
                <pre className="integration-snippet">{agentStdioLine(access)}</pre>
              </div>
            ) : null}
            <details className="integration-card">
              <summary>{t("settingsTab.agentAccess.permissions", { defaultValue: "Permissions" })}</summary>
              <p className="muted-text">
                {t("settingsTab.agentAccess.permissionsHint", {
                  defaultValue: "The main token can read, make things and move files. Nothing else unless you tick it.",
                })}
              </p>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={access.scopes.includes("clone")}
                  disabled={busy}
                  onChange={(event) =>
                    void apply({
                      scopes: event.target.checked ? [...access.scopes, "clone"] : access.scopes.filter((s) => s !== "clone"),
                    })
                  }
                />
                {t("settingsTab.agentAccess.scopeClone", { defaultValue: "Allow voice cloning" })}
              </label>
              {access.scopes.includes("clone") ? (
                <label className="check-row">
                  <input
                    type="checkbox"
                    checked={access.remoteClone}
                    disabled={busy}
                    onChange={(event) => void apply({ remoteClone: event.target.checked })}
                  />
                  {t("settingsTab.agentAccess.remoteClone", { defaultValue: "Also allow cloning from other computers" })}
                </label>
              ) : null}
              <p className="muted-text">
                {t("settingsTab.agentAccess.readTokenHint", {
                  defaultValue: "For a dashboard or watcher: it can look at status and jobs but cannot make or change anything.",
                })}
              </p>
              {access.readToken ? (
                <>
                  <div className="integration-card-header">
                    <strong>{t("settingsTab.agentAccess.readTokenLabel", { defaultValue: "Read-only token (Claude Code)" })}</strong>
                    <button className="secondary-button" type="button" onClick={() => onCopyText(agentConfigLine(access, access.readToken ?? ""))}>
                      {t("settingsTab.integrations.copyButton", { defaultValue: "Copy" })}
                    </button>
                  </div>
                  <pre className="integration-snippet">{agentConfigLine(access, access.readToken)}</pre>
                  <button className="secondary-button" type="button" disabled={busy} onClick={() => void apply({ readToken: false })}>
                    {t("settingsTab.agentAccess.readTokenRemove", { defaultValue: "Remove the read-only token" })}
                  </button>
                </>
              ) : (
                <button className="secondary-button" type="button" disabled={busy} onClick={() => void apply({ readToken: true })}>
                  {t("settingsTab.agentAccess.readTokenMake", { defaultValue: "Make a read-only token" })}
                </button>
              )}
            </details>
            <div>
              <button className="secondary-button" type="button" disabled={busy} onClick={() => void apply({ rotateToken: true })}>
                {t("settingsTab.agentAccess.rotate", { defaultValue: "Make a new token" })}
              </button>
              <p className="muted-text">
                {t("settingsTab.agentAccess.rotateHint", {
                  defaultValue: "The old token stops working at once. Agents need the new line.",
                })}
              </p>
            </div>
          </>
        ) : null}
      </div>
    </Panel>
  );
}
