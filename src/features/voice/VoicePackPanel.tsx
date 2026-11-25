/**
 * Voice setup on Voice > Models (the voice runtime, D-73 M4, M5; voice studio UX audit, change 1): the setup card
 * (tasks, presets, Install what is missing, Choose parts), the runtime's parts with their size, what is in memory now
 * with Unload and the idle minutes, and Troubleshooting with Reinstall and Remove. The install job hook and its
 * progress row live in ``setup/voicePackJob.tsx`` and are re-exported here for the Mandarin card.
 */
import { useState, type ReactNode } from "react";
import i18next from "i18next";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { stopVoiceWorker, uninstallVoiceRuntime } from "../../api";
import type { VoiceRuntime } from "../../types";
import { useVoice } from "./VoiceContext";
import { runtimeRemoveConfirmText } from "./models/voiceModelsLogic";
import { VoiceSetupCard, laneDescription, type SetupCatalog } from "./setup/VoiceSetupCard";
import type { VoicePackJob } from "./setup/voicePackJob";
import { kokoroInstalled, runtimePartsSummary, voicePackMode } from "./voicePackState";

export { InstallLog, InstallProgress, useVoicePackJob, type VoicePackJob } from "./setup/voicePackJob";

function formatInstalledAt(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString(i18next.language, { day: "numeric", month: "short", year: "numeric" });
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

const NO_CATALOG: SetupCatalog = { models: [], loaded: false, refresh: async () => {} };

export function VoiceRuntimePanel({
  runtime,
  pack,
  actions,
  catalog = NO_CATALOG,
  onOpenClone,
}: {
  runtime: VoiceRuntime;
  pack: VoicePackJob;
  actions?: ReactNode;
  /** The speech models, for the natural voice and cloning rows and the Full preset's model. */
  catalog?: SetupCatalog;
  onOpenClone?: () => void;
}) {
  const { t } = useTranslation("studio");
  const { refresh } = useVoice();
  const { job } = pack;
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
  const withKokoro = kokoroInstalled(runtime.extras, runtime.ttsModelInstalled);
  const partsText = parts
    ? withKokoro
      ? t("voice.setup.parts", {
          defaultValue: "{total} GB on disk: speech packages {packages} GB, Kokoro quick voices {kokoro} GB, turn detection and speaker labels {small} GB. Nothing is sent online.",
          ...parts,
        })
      : t("voice.setup.partsNoKokoro", {
          defaultValue: "{total} GB on disk: speech packages {packages} GB, turn detection and speaker labels {small} GB; the Kokoro quick voices are not installed. Nothing is sent online.",
          ...parts,
        })
    : t("voice.setup.beforeInstall", {
        defaultValue: "Everything runs on this computer and nothing is sent online. Speech-to-text and speech models are separate downloads on this page.",
      });

  const removeRuntime = async () => {
    if (!window.confirm(runtimeRemoveConfirmText(parts ? Number(parts.total) : null))) return;
    pack.setError(null);
    setRemoving(true);
    try {
      await uninstallVoiceRuntime(alsoModels);
    } catch (err) {
      pack.setError(err);
    } finally {
      setRemoving(false);
      refresh();
    }
  };

  return (
    <Panel
      actions={actions}
      title={t("voice.setup.title", { defaultValue: "Voice setup" })}
      subtitle={t("voice.setup.subtitle", {
        defaultValue: "{lane}. What voice can do on this computer, and what it still needs.",
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

        <VoiceSetupCard runtime={runtime} pack={pack} catalog={catalog} onRuntimeChanged={refresh} onOpenClone={onOpenClone} busy={removing} />

        {runtime.installed ? <InMemory runtime={runtime} /> : null}

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
              <button type="button" className="secondary-button" disabled={removing} onClick={() => void pack.start()}>
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
              {t("voice.setup.alsoModels", {
                defaultValue: "Also delete the Kokoro quick voices, turn detection and speaker labels. Downloaded speech-to-text and speech models always stay.",
              })}
            </label>
          </details>
        ) : null}
      </div>
    </Panel>
  );
}
