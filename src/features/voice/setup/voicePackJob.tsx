/**
 * The voice install job (runtime, update, add-ons): the hook that starts, polls and cancels it, its progress row and
 * its log. Shared by the Voice setup card and the Mandarin card on Speak. A failed start is shown through
 * ``describeVoiceError``: one plain sentence, the raw text under Details.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { cancelVoiceInstall, getVoiceInstallStatus, startVoiceInstall } from "../../../api";
import type { VoiceInstallJob, VoiceRuntime } from "../../../types";
import { describeVoiceError } from "../voiceErrorText";

export function useVoicePackJob(runtime: VoiceRuntime, onChanged: () => void): VoicePackJob & { errorDetails: string; start: (add?: string[]) => Promise<boolean> } {
  const [job, setJob] = useState<VoiceInstallJob>(runtime.install);
  const [error, setErrorText] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState("");

  useEffect(() => {
    if (!job.running) return;
    const timer = window.setInterval(() => {
      void getVoiceInstallStatus()
        .then((next) => {
          setJob(next);
          // The runtime status (installed, extras, voices) changed under us: fetch it again.
          if (!next.running) onChanged();
        })
        .catch(() => {});
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job.running, onChanged]);

  /** Show a failure (anything thrown, or a plain string) as its sentence and details; null clears it. */
  const setError = (err: unknown) => {
    if (err == null) {
      setErrorText(null);
      setErrorDetails("");
      return;
    }
    const view = describeVoiceError(err);
    setErrorText(view.sentence);
    setErrorDetails(view.details);
  };

  // Extras already in the runtime are kept by the backend; `add` names the ones to add. True when the job started.
  const start = async (add: string[] = []): Promise<boolean> => {
    setError(null);
    try {
      const installedExtras = runtime.extras.filter((e) => e.installed).map((e) => e.id);
      setJob(await startVoiceInstall([...new Set([...installedExtras, ...add])]));
      return true;
    } catch (err) {
      setError(err);
      return false;
    }
  };
  const cancel = () => void cancelVoiceInstall().then(setJob);
  return { job, error, errorDetails, setError, start, cancel };
}

/** What the setup card and the runtime panel need from the job; ``start`` resolves false when it did not start. */
export interface VoicePackJob {
  job: VoiceInstallJob;
  error: string | null;
  errorDetails?: string;
  setError: (err: unknown) => void;
  start: (add?: string[]) => Promise<boolean | void>;
  cancel: () => void;
}

export function InstallProgress({ pack }: { pack: Pick<VoicePackJob, "job" | "cancel"> }) {
  const { t } = useTranslation("studio");
  const { job } = pack;
  return (
    <div className="progress-row">
      <div className="progress-meta">
        <span>
          {job.phaseLabel} ({job.phaseIndex}/{job.phaseTotal})
        </span>
        <button type="button" className="secondary-button" onClick={pack.cancel}>
          {t("voice.pack.cancel", { defaultValue: "Cancel" })}
        </button>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${Math.max(4, job.percent)}%` }} />
      </div>
    </div>
  );
}

/** A failure as one sentence with the raw text under Details. */
export function SetupError({ sentence, details }: { sentence: string; details?: string }) {
  const { t } = useTranslation("studio");
  return (
    <div className="voice-error" role="alert">
      <p>{sentence}</p>
      {details ? (
        <details>
          <summary>{t("voice.setup.details",{ defaultValue: "Details" })}</summary>
          <pre className="install-log-terminal">{details}</pre>
        </details>
      ) : null}
    </div>
  );
}

export function InstallLog({ pack }: { pack: Pick<VoicePackJob, "job" | "error" | "errorDetails"> }) {
  const { t } = useTranslation("studio");
  const { job, error } = pack;
  const failed = job.phase === "error" || job.phase === "cancelled";
  const jobError = failed && job.error ? describeVoiceError(job.error) : null;
  return (
    <>
      {error ? (
        <SetupError sentence={error} details={pack.errorDetails} />
      ) : jobError ? (
        <SetupError sentence={jobError.sentence} details={jobError.details} />
      ) : null}
      {job.log.length > 0 ? (
        <details className="install-log-panel">
          <summary className="install-log-summary">{t("voice.pack.log", { defaultValue: "Install log" })}</summary>
          <pre className="install-log-terminal">{job.log.join("\n")}</pre>
        </details>
      ) : null}
    </>
  );
}
