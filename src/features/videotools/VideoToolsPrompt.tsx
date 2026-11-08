/**
 * The first-use prompt: "Video tools are needed" with one Install button, a failure message, and an offline
 * "choose an ffmpeg file" field. It shows in the Mixer when ffmpeg is missing and goes away once it is installed.
 */
import { useTranslation } from "react-i18next";
import { PathField } from "../storyboard/PathField";
import { promptPhase, type PromptPhase } from "./videoToolsLogic";
import type { FfmpegStatus } from "../../api/ffmpeg";

export interface VideoToolsPromptViewProps {
  phase: PromptPhase;
  error: string | null;
  percent: number | null;
  onInstall: () => void;
  onChooseFile: (path: string) => void;
  onDismissError?: () => void;
}

export function VideoToolsPromptView({ phase, error, percent, onInstall, onChooseFile, onDismissError }: VideoToolsPromptViewProps) {
  const { t } = useTranslation("studio");
  if (phase === "hidden") return null;
  const installing = phase === "installing";
  return (
    <div className={phase === "failed" ? "callout warning video-tools-prompt" : "callout video-tools-prompt"} role={phase === "failed" ? "alert" : "status"}>
      <p>
        <strong>{t("videoTools.title", { defaultValue: "Video tools are needed" })}</strong>
      </p>
      <p className="muted-text">
        {t("videoTools.body", {
          defaultValue: "Joining and exporting video uses FFmpeg, a free helper program. It installs as a package for this app only.",
        })}
      </p>
      {phase === "failed" && error ? (
        <p className="error-text">{t("videoTools.failed", { defaultValue: "Video tools did not install: {error}", error })}</p>
      ) : null}
      <div className="button-row">
        <button type="button" className="primary-button" disabled={installing} onClick={onInstall}>
          {installing
            ? t("videoTools.installing", { defaultValue: "Installing video tools..." })
            : t("videoTools.install", { defaultValue: "Install video tools" })}
        </button>
        {phase === "failed" && onDismissError ? (
          <button type="button" className="secondary-button" onClick={onDismissError}>
            {t("storyboard.dismiss", { defaultValue: "Dismiss" })}
          </button>
        ) : null}
      </div>
      {installing && percent !== null ? (
        <progress max={100} value={percent} aria-label={t("videoTools.installing", { defaultValue: "Installing video tools..." })} />
      ) : null}
      <PathField
        label={t("videoTools.chooseLabel", { defaultValue: "Offline? Use an ffmpeg file you already have" })}
        buttonLabel={t("videoTools.use", { defaultValue: "Use this file" })}
        extensions={["*"]}
        disabled={installing}
        onSubmit={onChooseFile}
      />
    </div>
  );
}

export interface VideoToolsPromptProps {
  status: FfmpegStatus | null;
  busy: boolean;
  percent: number | null;
  error: string | null;
  /** Show the prompt even when status says installed (a render just failed for want of ffmpeg). */
  forced?: boolean;
  onInstall: () => void;
  onChooseFile: (path: string) => void;
  onDismissError?: () => void;
}

export function VideoToolsPrompt({ status, busy, percent, error, forced, onInstall, onChooseFile, onDismissError }: VideoToolsPromptProps) {
  return (
    <VideoToolsPromptView
      phase={promptPhase(status, busy, error, forced)}
      error={error}
      percent={percent}
      onInstall={onInstall}
      onChooseFile={onChooseFile}
      onDismissError={onDismissError}
    />
  );
}
