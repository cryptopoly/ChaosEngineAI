/**
 * "Copy now" for the music model files (D-100, FU-276). Shown under a failed load when the backend says the
 * weight files could not be linked into the music folder and must be copied: why it happens, the two ways to
 * avoid it, then a button that runs the copy as a backend job with a progress bar and Cancel. When the copy is
 * done the panel calls `onDone` once so the load that stopped is tried again by itself.
 */
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { cancelMusicWeightCopy, getMusicWeightCopyStatus, startMusicWeightCopy } from "../../api";
import type { MusicWeightCopyInfo, MusicWeightCopyStatus } from "../../types";
import { copyMinutes, formatCopyGb } from "./musicLogic";

const POLL_MS = 1000;

function fallbackError(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function MusicWeightCopyPanel({ info, onDone }: { info: MusicWeightCopyInfo; onDone: () => void }) {
  const [status, setStatus] = useState<MusicWeightCopyStatus | null>(null);
  const [started, setStarted] = useState(false);
  const [refusal, setRefusal] = useState<string | null>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;
  const doneFired = useRef(false);

  const phase = status?.phase ?? "idle";
  const running = started && phase !== "done" && phase !== "error" && phase !== "cancelled";
  const done = started && phase === "done";

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      getMusicWeightCopyStatus().then(setStatus).catch(() => {
        // A missed poll changes nothing; the next one answers.
      });
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => {
    if (done && !doneFired.current) {
      doneFired.current = true;
      onDoneRef.current();
    }
  }, [done]);

  async function copyNow() {
    setRefusal(null);
    doneFired.current = false;
    try {
      const next = await startMusicWeightCopy();
      setStarted(true);
      setStatus(next);
    } catch (err) {
      setRefusal(fallbackError(err));
    }
  }

  async function cancel() {
    try {
      setStatus(await cancelMusicWeightCopy());
    } catch (err) {
      setRefusal(fallbackError(err));
    }
  }

  return (
    <MusicWeightCopyView
      info={info}
      status={status}
      started={started}
      refusal={refusal}
      onCopy={() => void copyNow()}
      onCancel={() => void cancel()}
    />
  );
}

export interface MusicWeightCopyViewProps {
  info: MusicWeightCopyInfo;
  status: MusicWeightCopyStatus | null;
  /** The Copy now button was pressed in this panel (a stale finished job from earlier is not shown). */
  started: boolean;
  refusal: string | null;
  onCopy: () => void;
  onCancel: () => void;
}

/** The panel's states (offer, copying, cancelled, failed, done) as plain markup from its inputs. */
export function MusicWeightCopyView({ info, status, started, refusal, onCopy, onCancel }: MusicWeightCopyViewProps) {
  const { t } = useTranslation("studio");
  const phase = status?.phase ?? "idle";
  const running = started && phase !== "done" && phase !== "error" && phase !== "cancelled";
  const done = started && phase === "done";
  const size = formatCopyGb(info.totalBytes);
  const minutes = copyMinutes(info.totalBytes);
  const total = status?.totalBytes || info.totalBytes;
  const percent = total > 0 ? Math.min(100, Math.round(((status?.bytesDone ?? 0) / total) * 100)) : 0;
  const eta = status?.etaSeconds ? Math.max(1, Math.round(status.etaSeconds / 60)) : null;

  return (
    <div className="music-weight-copy" role="group" aria-label={t("musicStudio.copy.title", { defaultValue: "The music model's files need to be copied" })}>
      <strong>{t("musicStudio.copy.title", { defaultValue: "The music model's files need to be copied" })}</strong>
      <p className="muted-text">
        {t("musicStudio.copy.why", {
          defaultValue:
            "This PC did not allow a link from the downloaded model files to the music folder, so {count} files ({size} GB) have to be copied. It takes extra disk space and happens once.",
          count: info.fileCount,
          size,
        })}
      </p>
      <ul className="muted-text music-weight-copy__ways">
        <li>{t("musicStudio.copy.wayDeveloper", { defaultValue: "Turn on Developer Mode (Windows Settings, System, For developers) so links are allowed." })}</li>
        <li>{t("musicStudio.copy.waySameDrive", { defaultValue: "Or keep the model cache on the same drive as the app's data ({targetDrive})." , targetDrive: info.targetDrive || info.target })}</li>
      </ul>
      {running ? (
        <div className="music-weight-copy__progress">
          <progress className="music-pack-banner__bar" max={100} value={percent} aria-label={t("musicStudio.copy.progress", { defaultValue: "Copy progress" })} />
          <span className="muted-text">
            {t("musicStudio.copy.status", {
              defaultValue: "{done} of {size} GB · {percent}%",
              done: formatCopyGb(status?.bytesDone ?? 0),
              size: formatCopyGb(total),
              percent,
            })}
            {status?.currentFile ? ` · ${status.currentFile}` : ""}
            {eta ? ` · ${t("musicStudio.copy.eta", { defaultValue: "about {minutes} min left", minutes: eta })}` : ""}
          </span>
          <div className="button-row">
            <button type="button" className="secondary-button" onClick={onCancel}>
              {t("musicStudio.copy.cancel", { defaultValue: "Cancel" })}
            </button>
          </div>
        </div>
      ) : done ? (
        <p role="status">{t("musicStudio.copy.done", { defaultValue: "Done, loading…" })}</p>
      ) : (
        <div className="button-row">
          <button type="button" className="primary-button" onClick={onCopy}>
            {t(phase === "cancelled" || phase === "error" ? "musicStudio.copy.resume" : "musicStudio.copy.button", {
              defaultValue: phase === "cancelled" || phase === "error" ? "Copy again ({size} GB, about {minutes} min)" : "Copy now ({size} GB, about {minutes} min)",
              size,
              minutes,
            })}
          </button>
        </div>
      )}
      {phase === "cancelled" ? <p className="muted-text" role="status">{t("musicStudio.copy.cancelled", { defaultValue: "Copy cancelled. Finished files are kept, so copying again continues from there." })}</p> : null}
      {phase === "error" && status?.error ? <p className="muted-text" role="alert">{status.error}</p> : null}
      {refusal ? <p className="muted-text" role="alert">{refusal}</p> : null}
    </div>
  );
}
