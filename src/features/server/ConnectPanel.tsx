import { useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDownIcon } from "../../components/icons/ActionIcons";

export const CONNECT_OPEN_KEY = "chaosengine.server.connectOpen";

export function readConnectOpen(): boolean {
  try {
    return window.localStorage.getItem(CONNECT_OPEN_KEY) === "1";
  } catch {
    return false;
  }
}

function writeConnectOpen(open: boolean): void {
  try {
    window.localStorage.setItem(CONNECT_OPEN_KEY, open ? "1" : "0");
  } catch {
    // not remembered
  }
}

export interface ConnectPanelProps {
  localServerUrl: string;
  apiToken: string | null;
  /** Model id the snippets name: the active or first warm model. */
  connectModel: string;
  onCopy: (text: string) => void;
}

/** "Connect your app": one line with the base URL and Copy; opens to the snippets. Closed by default, remembered. */
export function ConnectPanel({ localServerUrl, apiToken, connectModel, onCopy }: ConnectPanelProps) {
  const { t } = useTranslation("common");
  const [open, setOpen] = useState<boolean>(() => readConnectOpen());
  const toggle = () => {
    setOpen((current) => {
      writeConnectOpen(!current);
      return !current;
    });
  };

  const connectKey = apiToken ?? "<chaosengine-api-token>";
  const ollamaBase = localServerUrl.replace(/\/v1\/?$/, "");
  const pythonSnippet = `from openai import OpenAI

client = OpenAI(base_url="${localServerUrl}", api_key="${connectKey}")
resp = client.chat.completions.create(
    model="${connectModel}",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)`;
  const jsSnippet = `import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${localServerUrl}", apiKey: "${connectKey}" });
const resp = await client.chat.completions.create({
  model: "${connectModel}",
  messages: [{ role: "user", content: "Hello" }],
});
console.log(resp.choices[0].message.content);`;
  const continueSnippet = `{
  "models": [
    {
      "title": "ChaosEngineAI",
      "provider": "openai",
      "model": "${connectModel}",
      "apiBase": "${localServerUrl}",
      "apiKey": "${connectKey}"
    }
  ]
}`;
  const copyLabel = t("serverTab.copy", { defaultValue: "Copy" });

  return (
    <div className={`server-connect-panel${open ? " is-open" : ""}`} data-testid="server-connect">
      <div className="server-connect-line">
        <button
          type="button"
          className="server-connect-toggle"
          aria-expanded={open}
          aria-controls="server-connect-body"
          onClick={toggle}
        >
          <ChevronDownIcon className={`server-connect-chevron${open ? " is-open" : ""}`} aria-hidden="true" />
          <span>{t("serverTab.connect.title", { defaultValue: "Connect your app" })}</span>
        </button>
        <span className="mono-text server-connect-url" title={localServerUrl}>{localServerUrl}</span>
        <button className="secondary-button" type="button" onClick={() => onCopy(localServerUrl)}>
          {copyLabel}
        </button>
      </div>
      {open ? (
        <div className="server-connect-body" id="server-connect-body">
          <p className="muted-text server-connect-sub">
            {t("serverTab.connect.subtitle", { defaultValue: "OpenAI-compatible endpoint. Point any OpenAI SDK or compatible app at the base URL below with the API key above." })}
          </p>
          <div className="server-command">
            <div className="server-command-header">
              <strong>{t("serverTab.connect.python", { defaultValue: "Python · openai" })}</strong>
              <button className="secondary-button" type="button" onClick={() => onCopy(pythonSnippet)}>{copyLabel}</button>
            </div>
            <pre className="mono-text server-code-block">{pythonSnippet}</pre>
          </div>
          <div className="server-command">
            <div className="server-command-header">
              <strong>{t("serverTab.connect.javascript", { defaultValue: "JavaScript · openai" })}</strong>
              <button className="secondary-button" type="button" onClick={() => onCopy(jsSnippet)}>{copyLabel}</button>
            </div>
            <pre className="mono-text server-code-block">{jsSnippet}</pre>
          </div>
          <div className="server-command">
            <div className="server-command-header">
              <strong>{t("serverTab.connect.continueDev", { defaultValue: "Continue.dev (config)" })}</strong>
              <button className="secondary-button" type="button" onClick={() => onCopy(continueSnippet)}>{copyLabel}</button>
            </div>
            <pre className="mono-text server-code-block">{continueSnippet}</pre>
          </div>
          <div className="server-command">
            <div className="server-command-header">
              <strong>{t("serverTab.connect.openWebui", { defaultValue: "Open WebUI" })}</strong>
            </div>
            <p className="muted-text">
              {t("serverTab.connect.openWebuiHint", { base: localServerUrl, defaultValue: "Admin → Settings → Connections → OpenAI API. Base URL: {base} — use the API key above." })}
            </p>
          </div>
          <div className="server-command">
            <div className="server-command-header">
              <strong>{t("serverTab.connect.ollamaApps", { defaultValue: "Ollama-compatible apps" })}</strong>
              <button className="secondary-button" type="button" onClick={() => onCopy(ollamaBase)}>{copyLabel}</button>
            </div>
            <p className="muted-text">
              {t("serverTab.connect.ollamaHint", { defaultValue: "Set the app's Ollama host / base URL to the address below — native /api endpoints are served alongside /v1." })}
            </p>
            <p className="mono-text">{ollamaBase}</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
