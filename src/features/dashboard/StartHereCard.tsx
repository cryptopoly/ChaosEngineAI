import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { DownloadStatus } from "../../api";
import { Panel } from "../../components/Panel";
import { FitBadge } from "../../components/models/FitBadge";
import type { ChatPick } from "../../types";
import { number, sizeLabel } from "../../utils";
import { pickSummary } from "./chatPick";

export interface StartHereCardProps {
  /** ``Recommendation.bestFit``; without it the card points at Chat Models. */
  pick: ChatPick | null | undefined;
  /** What the card says this computer is ("NVIDIA GPU, 24 GB VRAM"). */
  hardwareLabel: string;
  gpu: boolean;
  download?: DownloadStatus | null;
  onDownload?: (repo: string) => void;
  onOpenChatModels?: () => void;
  onOpenChat?: () => void;
}

/** First run (no chat model on disk): the two steps to a first chat, with the recommended model named (D-105 E). */
export function StartHereCard({ pick, hardwareLabel, gpu, download, onDownload, onOpenChatModels, onOpenChat }: StartHereCardProps) {
  const { t } = useTranslation("dashboard");
  const state = download?.state ?? null;
  let downloadButton: ReactNode = null;
  if (pick && onDownload) {
    if (state === "downloading") {
      downloadButton = (
        <button className="secondary-button" type="button" disabled>
          {t("startHere.downloading", {
            defaultValue: "Downloading, {percent}%",
            percent: number(Math.max(0, Math.min(1, download?.progress ?? 0)) * 100, 0),
          })}
        </button>
      );
    } else if (state === "completed" || pick.availableLocally) {
      downloadButton = (
        <button className="secondary-button" type="button" disabled>
          {t("startHere.downloaded", { defaultValue: "Downloaded" })}
        </button>
      );
    } else {
      downloadButton = (
        <button className="primary-button" type="button" onClick={() => onDownload(pick.repo)}>
          {pick.sizeGb
            ? t("startHere.downloadSize", { defaultValue: "Download ({size})", size: sizeLabel(pick.sizeGb) })
            : t("startHere.download", { defaultValue: "Download" })}
        </button>
      );
    }
  }

  return (
    <Panel
      title={t("startHere.title", { defaultValue: "Start here" })}
      subtitle={t("startHere.subtitle", {
        defaultValue: "No chat model is downloaded yet. Two steps to a first chat on this computer ({hardware}).",
        hardware: hardwareLabel,
      })}
      className="span-2 start-here-card"
    >
      <ol className="start-here-steps">
        <li>
          <strong>{t("startHere.step1", { defaultValue: "Download a model that fits this computer." })}</strong>
          {pick ? (
            <div className="start-here-pick">
              <div className="start-here-pick__text">
                <span className="badge accent">{t("startHere.recommended", { defaultValue: "Recommended" })}</span>{" "}
                <strong>{pick.name}</strong> <FitBadge fit={pick.fit} />
                <p className="muted-text">{pickSummary(pick, gpu, t)}</p>
              </div>
              <div className="button-row">
                {downloadButton}
                {onOpenChatModels ? (
                  <button className="secondary-button" type="button" onClick={onOpenChatModels}>
                    {t("startHere.otherModels", { defaultValue: "Other recommended models" })}
                  </button>
                ) : null}
              </div>
              {state === "failed" && download?.error ? <p className="callout error">{download.error}</p> : null}
            </div>
          ) : (
            <div className="start-here-pick">
              <p className="muted-text">
                {t("startHere.noPick", {
                  defaultValue: "No catalogue model fits this computer comfortably. Chat Models lists every model with a fit verdict.",
                })}
              </p>
              {onOpenChatModels ? (
                <div className="button-row">
                  <button className="primary-button" type="button" onClick={onOpenChatModels}>
                    {t("startHere.openModels", { defaultValue: "Open Chat Models" })}
                  </button>
                </div>
              ) : null}
            </div>
          )}
        </li>
        <li>
          <strong>{t("startHere.step2", { defaultValue: "Open Chat and pick the model at the top of the thread." })}</strong>
          {onOpenChat ? (
            <div className="button-row">
              <button className="secondary-button" type="button" onClick={onOpenChat}>
                {t("startHere.openChat", { defaultValue: "Open Chat" })}
              </button>
            </div>
          ) : null}
        </li>
      </ol>
    </Panel>
  );
}
