import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { VoiceModelRow } from "../../../types";
import { useCostText } from "./costText";
import { costLabel } from "./speakLogic";

const DISMISS_KEY = "chaos.voice.installCardDismissed";

function readDismissed(): boolean {
  try {
    return window.sessionStorage.getItem(DISMISS_KEY) === "1";
  } catch {
    return false;
  }
}

function writeDismissed(): void {
  try {
    window.sessionStorage.setItem(DISMISS_KEY, "1");
  } catch {
    // Storage blocked: the card simply stays hidden until the window reloads.
  }
}

/**
 * "Install a better voice": shown on Speak while no natural (CustomVoice) model is downloaded. It names what the model
 * adds, its download size and its measured memory and speed, and downloads it in place. "Not now" hides it for the
 * session.
 */
export function InstallVoiceCard({ model, onDownload }: { model: VoiceModelRow; onDownload: () => void }) {
  const { t } = useTranslation("studio");
  const costs = useCostText();
  const [dismissed, setDismissed] = useState(readDismissed);
  if (dismissed) return null;
  const downloading = model.download.state === "downloading";

  return (
    <section className="voice-install-card" aria-labelledby="voice-install-card-title">
      <h4 id="voice-install-card-title" className="voice-install-card__title">
        {t("voice.install.title", { defaultValue: "Install a better voice" })}
      </h4>
      <p className="voice-install-card__body">
        {t("voice.install.body", {
          defaultValue:
            "The quick voices are fast but flat. {name} adds natural prosody and a written style instruction (angry, whisper, slow, an accent).",
          name: model.name,
        })}
      </p>
      <p className="muted-text voice-install-card__cost">
        {t("voice.install.cost", { defaultValue: "Download {gb} GB; {detail}.", gb: model.sizeGb, detail: costs.detail(costLabel(model)) })}
        {model.licence ? ` ${model.licence.name}.` : ""}
      </p>
      <div className="button-row">
        {downloading ? (
          <span className="muted-text" aria-live="polite">
            {model.download.phase === "watermark"
              ? t("voice.clone.installingWatermark", { defaultValue: "Installing the watermark component…" })
              : t("voice.speak.modelDownloading", { defaultValue: "Downloading {name}…", name: model.name })}
          </span>
        ) : (
          <button type="button" className="primary-button" onClick={onDownload}>
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M8 2.5v8M4.5 7 8 10.5 11.5 7M3 13.5h10" />
            </svg>{" "}
            {t("voice.cost.download", { defaultValue: "Download {gb} GB", gb: model.sizeGb })}
          </button>
        )}
        {!downloading ? (
          <button
            type="button"
            className="secondary-button"
            onClick={() => {
              writeDismissed();
              setDismissed(true);
            }}
          >
            {t("voice.install.notNow", { defaultValue: "Not now" })}
          </button>
        ) : null}
      </div>
      {model.download.error ? (
        <p className="voice-error" role="alert">
          {model.download.error}
        </p>
      ) : null}
    </section>
  );
}
