import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getAgentAccess, updateAgentAccess } from "../../api";
import type { AgentAccess } from "../../api";
import { Panel } from "../../components/Panel";

interface AgentAccessPanelProps {
  backendOnline: boolean;
  onCopyText: (text: string) => void;
}

/** Quote a header value for a POSIX shell; the token is URL-safe so this is belt and braces. */
function shellQuote(value: string): string {
  return `"${value.replace(/(["\\$`])/g, "\\$1")}"`;
}

export function agentConfigLine(access: AgentAccess): string {
  return `claude mcp add --transport http chaosengine ${access.mcpUrl} --header ${shellQuote(`Authorization: Bearer ${access.token}`)}`;
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

  const apply = useCallback(async (patch: { enabled?: boolean; rotateToken?: boolean }) => {
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
