/**
 * The voice pack: install, update, repair and cancel, with progress and the install log. The job hook and the
 * progress and log pieces are shared with the Mandarin add-on buttons (Models page and the Speak picker).
 */
import { useEffect, useState, type ReactNode } from "react";
import i18next from "i18next";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { cancelVoiceInstall, getVoiceInstallStatus, startVoiceInstall } from "../../api";
import type { VoiceInstallJob, VoiceRuntime } from "../../types";
import { voicePackMode } from "./voicePackState";

function laneDescription(runtime: VoiceRuntime): string {
  if (runtime.lane === "mlx") {
    return i18next.t("studio:voice.lane.mlx", { defaultValue: "Apple Silicon: Parakeet and Whisper on MLX, Kokoro voices" });
  }
  if (runtime.cuda) return i18next.t("studio:voice.lane.cuda", { defaultValue: "NVIDIA GPU: Whisper on CUDA, Kokoro voices" });
  if (runtime.nvidiaGpu) {
    return i18next.t("studio:voice.lane.nvidia", {
      defaultValue: "Whisper (CUDA libraries are added by the install), Kokoro voices",
    });
  }
  return i18next.t("studio:voice.lane.cpu", { defaultValue: "Whisper on the CPU, Kokoro voices" });
}

export function useVoicePackJob(runtime: VoiceRuntime, onChanged: () => void) {
  const [job, setJob] = useState<VoiceInstallJob>(runtime.install);
  const [error, setError] = useState<string | null>(null);

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

  // Extras already in the pack are kept by the backend; `add` names the ones to add.
  const start = async (add: string[] = []) => {
    setError(null);
    try {
      const installedExtras = runtime.extras.filter((e) => e.installed).map((e) => e.id);
      setJob(await startVoiceInstall([...new Set([...installedExtras, ...add])]));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  const cancel = () => void cancelVoiceInstall().then(setJob);
  return { job, error, start, cancel };
}

export type VoicePackJob = ReturnType<typeof useVoicePackJob>;

function formatInstalledAt(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString(i18next.language);
}

export function InstallProgress({ pack }: { pack: VoicePackJob }) {
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

export function InstallLog({ pack }: { pack: VoicePackJob }) {
  const { t } = useTranslation("studio");
  const { job, error } = pack;
  return (
    <>
      {job.phase === "error" || job.phase === "cancelled" || error ? (
        <p className="voice-error" role="alert">
          {error ?? job.error}
        </p>
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

export function VoicePackPanel({ runtime, pack, actions }: { runtime: VoiceRuntime; pack: VoicePackJob; actions?: ReactNode }) {
  const { t } = useTranslation("studio");
  const { job, start } = pack;
  const mode = voicePackMode({
    installed: runtime.installed,
    updateAvailable: runtime.updateAvailable,
    vadAvailable: runtime.vadAvailable,
    ttsModelInstalled: runtime.ttsModelInstalled,
    jobPhase: job.phase,
  });
  const installedAt = formatInstalledAt(runtime.installedAt);

  return (
    <Panel
      actions={actions}
      title={t("voice.pack.title", { defaultValue: "Voice pack" })}
      subtitle={laneDescription(runtime)}
      className="span-2"
    >
      <div className="voice-pack">
        <div className="voice-pack__status">
          {mode === "healthy" ? (
            <span className="badge success">
              {installedAt
                ? t("voice.pack.checkedOkAt", { defaultValue: "Installed, checked OK ({time})", time: installedAt })
                : t("voice.pack.checkedOk", { defaultValue: "Installed, checked OK" })}
            </span>
          ) : runtime.installed ? (
            <span className="badge success">{t("voice.pack.installed", { defaultValue: "Installed" })}</span>
          ) : (
            <span className="badge muted">{t("voice.pack.notInstalled", { defaultValue: "Not installed" })}</span>
          )}
          <span className="muted-text">
            {t("voice.pack.about", {
              defaultValue:
                "Speech-to-text and text-to-speech run in their own environment on this computer (about 2 GB, plus the models you download). Nothing is sent online.",
            })}
          </span>
        </div>

        {job.running ? (
          <InstallProgress pack={pack} />
        ) : mode !== "healthy" ? (
          <div className="button-row">
            <button type="button" className="primary-button" onClick={() => void start()}>
              {mode === "install"
                ? t("voice.pack.install", { defaultValue: "Install voice pack" })
                : mode === "update"
                  ? t("voice.pack.update", { defaultValue: "Update voice pack" })
                  : t("voice.pack.repair", { defaultValue: "Repair voice pack" })}
            </button>
          </div>
        ) : (
          <details className="voice-pack__troubleshoot">
            <summary className="muted-text">{t("voice.pack.troubleshoot", { defaultValue: "Troubleshooting" })}</summary>
            <p className="muted-text">
              {t("voice.pack.troubleshootBody", {
                defaultValue: "Only needed if voice features stop working. This runs the whole install again.",
              })}
            </p>
            <button type="button" className="secondary-button" onClick={() => void start()}>
              {t("voice.pack.reinstallSmall", { defaultValue: "Reinstall" })}
            </button>
          </details>
        )}

        {mode === "update" && !job.running ? (
          <p className="muted-text">
            {t("voice.pack.updateFeatures", {
              defaultValue:
                "Update the voice pack for voice calls in Chat and speaker labels on transcripts (adds about 40 MB of models; a minute or two).",
            })}
          </p>
        ) : null}
        <InstallLog pack={pack} />
      </div>
    </Panel>
  );
}
