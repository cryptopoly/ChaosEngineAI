/**
 * Setup page row for the video tools (ffmpeg): what is in use and where it came from, the install button,
 * the optional pinned LGPL download (Windows and Linux), and a remove button for a download or chosen file.
 */
import { useTranslation } from "react-i18next";
import type { FfmpegStatus } from "../../api/ffmpeg";
import { useVideoTools } from "./useVideoTools";
import { canRemove, formatMb, sourceLabel } from "./videoToolsLogic";

export interface VideoToolsRowViewProps {
  status: FfmpegStatus | null;
  busy: boolean;
  percent: number | null;
  error: string | null;
  onInstall: () => void;
  onDownload: () => void;
  onCancel: () => void;
  onRemove: () => void;
}

export function VideoToolsRowView({ status, busy, percent, error, onInstall, onDownload, onCancel, onRemove }: VideoToolsRowViewProps) {
  const { t } = useTranslation("studio");
  const source = sourceLabel(status?.source ?? null);
  const pin = status?.pin ?? null;
  return (
    <div className="video-tools-row">
      <div>
        <strong>{t("videoTools.row.label", { defaultValue: "Video tools (FFmpeg)" })}</strong>
        <p className="muted-text">
          {status === null
            ? t("videoTools.row.checking", { defaultValue: "Checking..." })
            : status.installed
              ? t("videoTools.row.installed", {
                  defaultValue: "{version} from {source}{licence}",
                  version: status.version ?? "",
                  source: t(`videoTools.row.source.${source.key}`, { defaultValue: source.fallback }),
                  licence: status.licence ? ` (${status.licence})` : "",
                })
              : t("videoTools.row.notInstalled", { defaultValue: "Not installed" })}
        </p>
        {status?.installed && status.encoder ? (
          <p className="muted-text">{t("videoTools.row.encoder", { defaultValue: "H.264 encoder: {encoder}", encoder: status.encoder })}</p>
        ) : null}
        {error ? <p className="error-text">{t("videoTools.failed", { defaultValue: "Video tools did not install: {error}", error })}</p> : null}
        {pin ? (
          <p className="muted-text">
            <a href={pin.licenceUrl} target="_blank" rel="noreferrer">
              {t("videoTools.row.licence", { defaultValue: "Licence" })}
            </a>
            {" / "}
            <a href={pin.sourceUrl} target="_blank" rel="noreferrer">
              {t("videoTools.row.source.link", { defaultValue: "Source" })}
            </a>
          </p>
        ) : null}
      </div>
      <div className="button-row">
        {!status?.installed ? (
          <button type="button" className="primary-button" disabled={busy || status === null} onClick={onInstall}>
            {busy ? t("videoTools.installing", { defaultValue: "Installing video tools..." }) : t("videoTools.install", { defaultValue: "Install video tools" })}
          </button>
        ) : null}
        {status?.canDownload && pin && status.source !== "download" ? (
          <button type="button" className="secondary-button" disabled={busy} onClick={onDownload}>
            {t("videoTools.row.download", { defaultValue: "Download LGPL build ({size} MB)", size: formatMb(pin.size) })}
          </button>
        ) : null}
        {busy && percent !== null ? (
          <button type="button" className="secondary-button" onClick={onCancel}>
            {t("videoTools.row.cancel", { defaultValue: "Cancel" })}
          </button>
        ) : null}
        {canRemove(status) ? (
          <button type="button" className="secondary-button" disabled={busy} onClick={onRemove}>
            {t("videoTools.row.remove", { defaultValue: "Remove" })}
          </button>
        ) : null}
      </div>
      {busy && percent !== null ? (
        <progress max={100} value={percent} aria-label={t("videoTools.installing", { defaultValue: "Installing video tools..." })} />
      ) : null}
    </div>
  );
}

export function VideoToolsRow({ backendOnline }: { backendOnline: boolean }) {
  const tools = useVideoTools(backendOnline);
  return (
    <VideoToolsRowView
      status={tools.status}
      busy={tools.busy}
      percent={tools.percent}
      error={tools.error}
      onInstall={() => void tools.install()}
      onDownload={() => void tools.download()}
      onCancel={() => void tools.cancel()}
      onRemove={() => void tools.remove()}
    />
  );
}
