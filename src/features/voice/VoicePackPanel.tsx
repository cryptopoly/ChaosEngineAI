/**
 * The voice runtime (D-73 M4, M5; called the voice pack before): install, update, repair and cancel with progress and
 * the install log; the four parts with their size; what is in memory now with Unload and the idle minutes; the
 * add-ons as tick boxes (tick installs, untick removes); Troubleshooting with Reinstall and Remove voice runtime. The
 * job hook and the progress piece are shared with the Mandarin card on Speak and Voices.
 */
import { useEffect, useState, type ReactNode } from "react";
import i18next from "i18next";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import {
  cancelVoiceInstall,
  getVoiceInstallStatus,
  removeVoiceExtra,
  startVoiceInstall,
  stopVoiceWorker,
  uninstallVoiceRuntime,
} from "../../api";
import type { VoiceExtra, VoiceInstallJob, VoiceRuntime } from "../../types";
import { useVoice } from "./VoiceContext";
import { runtimeRemoveConfirmText } from "./models/voiceModelsLogic";
import { runtimePartsSummary, voicePackMode } from "./voicePackState";

function laneDescription(runtime: VoiceRuntime): string {
  if (runtime.lane === "mlx") return i18next.t("studio:voice.lane.mlxRuntime", { defaultValue: "Apple Silicon" });
  if (runtime.cuda || runtime.nvidiaGpu) return i18next.t("studio:voice.lane.nvidiaRuntime", { defaultValue: "NVIDIA GPU" });
  return i18next.t("studio:voice.lane.cpuRuntime", { defaultValue: "CPU" });
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

  // Extras already in the runtime are kept by the backend; `add` names the ones to add.
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
  return { job, error, setError, start, cancel };
}

export type VoicePackJob = ReturnType<typeof useVoicePackJob>;

function formatInstalledAt(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString(i18next.language, { day: "numeric", month: "short", year: "numeric" });
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

// i18n-extract: voice.runtime.addon.chinese
// i18n-extract: voice.runtime.addon.speech-gpu
const ADDON_LABELS: Record<string, string> = {
  chinese: "Mandarin Chinese quick voices (8 voices, about 60 MB)",
  "speech-gpu": "Speech GPU add-on: Chatterbox, VoxCPM2 and Orpheus on the NVIDIA GPU (several GB)",
};

/** The add-ons as tick boxes. Before install they choose what the install adds; after it, ticking installs the add-on
 * and unticking removes it. */
function AddOns({
  extras,
  chosen,
  onToggle,
  disabled,
  installed,
}: {
  extras: VoiceExtra[];
  chosen: (extra: VoiceExtra) => boolean;
  onToggle: (extra: VoiceExtra, on: boolean) => void;
  disabled: boolean;
  installed: boolean;
}) {
  const { t } = useTranslation("studio");
  if (extras.length === 0) return null;
  return (
    <fieldset className="voice-runtime__addons" disabled={disabled}>
      <legend>{t("voice.runtime.addons", { defaultValue: "Add-ons" })}</legend>
      {extras.map((extra) => (
        <label key={extra.id} className="checkbox-label">
          <input type="checkbox" checked={chosen(extra)} onChange={(e) => onToggle(extra, e.target.checked)} />
          {t(`voice.runtime.addon.${extra.id}`, { defaultValue: ADDON_LABELS[extra.id] ?? extra.label })}
        </label>
      ))}
      <small className="muted-text">
        {installed
          ? t("voice.runtime.addonsHint", { defaultValue: "Installs when ticked; untick to remove." })
          : t("voice.runtime.addonsHintInstall", { defaultValue: "Ticked add-ons install with the runtime; you can change them later." })}
      </small>
    </fieldset>
  );
}

function InMemory({ runtime }: { runtime: VoiceRuntime }) {
  const { t } = useTranslation("studio");
  const { settings, saveSettings, refresh } = useVoice();
  const [unloading, setUnloading] = useState(false);
  const loaded = runtime.inMemory ?? [];
  const names = loaded
    .map((m) => (m.peakGb != null ? t("voice.runtime.modelWithGb", { defaultValue: "{name}, {gb} GB", name: m.name, gb: m.peakGb.toFixed(1) }) : m.name))
    .join(" · ");
  return (
    <div className="voice-runtime__memory">
      <span>
        <strong>{t("voice.runtime.inMemory", { defaultValue: "In memory now:" })}</strong>{" "}
        {runtime.workerPid
          ? names || t("voice.runtime.workerOnly", { defaultValue: "the voice worker, no model yet" })
          : t("voice.runtime.nothingLoaded", { defaultValue: "nothing" })}
        {runtime.workerPid ? (
          <>
            {" "}
            <button
              type="button"
              className="secondary-button voice-runtime__unload"
              disabled={unloading}
              onClick={() => {
                setUnloading(true);
                void stopVoiceWorker().finally(() => {
                  setUnloading(false);
                  refresh();
                });
              }}
            >
              {t("voice.settings.unload", { defaultValue: "Unload" })}
            </button>
          </>
        ) : null}
      </span>
      <label className="voice-runtime__idle">
        {t("voice.runtime.idleBefore", { defaultValue: "Free memory after" })}
        <input
          type="number"
          className="text-input"
          min={0}
          max={240}
          value={settings.idleUnloadMinutes}
          aria-label={t("voice.settings.idle", { defaultValue: "Free voice memory after (minutes idle, 0 = never)" })}
          onChange={(e) => void saveSettings({ idleUnloadMinutes: Math.max(0, Math.min(240, Number(e.target.value) || 0)) })}
        />
        {t("voice.runtime.idleAfter", { defaultValue: "minutes idle (0 = never)" })}
      </label>
    </div>
  );
}

export function VoiceRuntimePanel({ runtime, pack, actions }: { runtime: VoiceRuntime; pack: VoicePackJob; actions?: ReactNode }) {
  const { t } = useTranslation("studio");
  const { refresh } = useVoice();
  const { job, start } = pack;
  const [wanted, setWanted] = useState<Set<string>>(() => new Set());
  const [busyExtra, setBusyExtra] = useState<string | null>(null);
  const [alsoModels, setAlsoModels] = useState(false);
  const [removing, setRemoving] = useState(false);
  const mode = voicePackMode({
    installed: runtime.installed,
    updateAvailable: runtime.updateAvailable,
    vadAvailable: runtime.vadAvailable,
    ttsModelInstalled: runtime.ttsModelInstalled,
    jobPhase: job.phase,
  });
  const installedAt = formatInstalledAt(runtime.installedAt);
  const parts = runtimePartsSummary(runtime.parts ?? []);
  const partsText = parts
    ? t("voice.runtime.parts", {
        defaultValue: "{total} GB on disk: speech packages {packages} GB, Kokoro quick voices {kokoro} GB, turn detection and speaker labels {small} GB. Nothing is sent online.",
        ...parts,
      })
    : t("voice.runtime.about", {
        defaultValue:
          "About 2.3 GB: the speech packages, the Kokoro quick voices, turn detection and speaker labels. Dictation and speech models are separate downloads on this page. Nothing is sent online.",
      });

  const toggleExtra = async (extra: VoiceExtra, on: boolean) => {
    if (!runtime.installed) {
      setWanted((previous) => {
        const next = new Set(previous);
        if (on) next.add(extra.id);
        else next.delete(extra.id);
        return next;
      });
      return;
    }
    if (on) {
      await start([extra.id]);
      return;
    }
    pack.setError(null);
    setBusyExtra(extra.id);
    try {
      await removeVoiceExtra(extra.id);
      refresh();
    } catch (err) {
      pack.setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyExtra(null);
    }
  };

  const removeRuntime = async () => {
    if (!window.confirm(runtimeRemoveConfirmText(parts ? Number(parts.total) : null))) return;
    pack.setError(null);
    setRemoving(true);
    try {
      await uninstallVoiceRuntime(alsoModels);
    } catch (err) {
      pack.setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRemoving(false);
      refresh();
    }
  };

  return (
    <Panel
      actions={actions}
      title={t("voice.runtime.title", { defaultValue: "Voice runtime" })}
      subtitle={t("voice.runtime.subtitle", {
        defaultValue: "{lane}. Everything voice needs that is not a model you choose.",
        lane: laneDescription(runtime),
      })}
      className="span-2 voice-runtime"
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
          <span className="muted-text">{partsText}</span>
        </div>

        {runtime.installed ? <InMemory runtime={runtime} /> : null}

        <AddOns
          extras={runtime.extras}
          installed={runtime.installed}
          disabled={job.running || busyExtra !== null || removing}
          chosen={(extra) =>
            runtime.installed
              ? // Ticked while its install runs; unticked while its removal runs.
                (extra.installed || (job.running && job.extras.includes(extra.id))) && busyExtra !== extra.id
              : wanted.has(extra.id)
          }
          onToggle={(extra, on) => void toggleExtra(extra, on)}
        />

        {job.running ? (
          <InstallProgress pack={pack} />
        ) : mode !== "healthy" ? (
          <div className="button-row">
            <button type="button" className="primary-button" onClick={() => void start(runtime.installed ? [] : [...wanted])}>
              {mode === "install"
                ? t("voice.runtime.install", { defaultValue: "Install voice runtime" })
                : mode === "update"
                  ? t("voice.runtime.update", { defaultValue: "Update voice runtime" })
                  : t("voice.runtime.repair", { defaultValue: "Repair voice runtime" })}
            </button>
          </div>
        ) : null}

        {mode === "update" && !job.running ? (
          <p className="muted-text">
            {t("voice.runtime.updateFeatures", {
              defaultValue:
                "Update the voice runtime for voice calls in Chat and speaker labels on transcripts (adds about 40 MB of models; a minute or two).",
            })}
          </p>
        ) : null}

        {runtime.installed && !job.running ? (
          <details className="voice-pack__troubleshoot">
            <summary className="muted-text">{t("voice.pack.troubleshoot", { defaultValue: "Troubleshooting" })}</summary>
            <p className="muted-text">
              {t("voice.runtime.troubleshootBody", {
                defaultValue: "Reinstall runs the whole install again. Remove frees the disk space; dictation, read-aloud, calls and Voice Studio stop until it is installed again.",
              })}
            </p>
            <div className="button-row">
              <button type="button" className="secondary-button" disabled={removing} onClick={() => void start()}>
                {t("voice.pack.reinstallSmall", { defaultValue: "Reinstall" })}
              </button>
              <button type="button" className="secondary-button danger-text" disabled={removing} onClick={() => void removeRuntime()}>
                {removing
                  ? t("voice.runtime.removing", { defaultValue: "Removing…" })
                  : t("voice.runtime.remove", { defaultValue: "Remove voice runtime" })}
              </button>
            </div>
            <label className="checkbox-label">
              <input type="checkbox" checked={alsoModels} onChange={(e) => setAlsoModels(e.target.checked)} />
              {t("voice.runtime.alsoModels", {
                defaultValue: "Also delete the Kokoro quick voices, turn detection and speaker labels. Downloaded dictation and speech models always stay.",
              })}
            </label>
          </details>
        ) : null}
        <InstallLog pack={pack} />
      </div>
    </Panel>
  );
}
