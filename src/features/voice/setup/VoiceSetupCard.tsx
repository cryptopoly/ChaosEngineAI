/**
 * The Voice setup card (voice studio UX audit, change 1): each task (Quick speech, Natural voices and style, Clone or
 * design, Speech to text on GPU, Calls) as Ready, Missing (what, how big) or Not on this machine (why), and one button
 * that runs the install job and then downloads the natural voice model. Before the first install it offers the two
 * presets, Basic and Full for this GPU (D0141); the optional parts sit under "Choose parts".
 */
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import i18next from "i18next";
import { useTranslation } from "react-i18next";
import { removeVoiceExtra } from "../../../api";
import type { VoiceExtra, VoiceModelRow, VoiceRuntime } from "../../../types";
import { downloadCloneModel } from "../clone/cloneApi";
import { describeVoiceError } from "../voiceErrorText";
import { voicePackMode } from "../voicePackState";
import { ChooseParts } from "./ChooseParts";
import { InstallLog, InstallProgress, SetupError, type VoicePackJob } from "./voicePackJob";
import {
  gbText,
  installGb,
  installPlan,
  setupPresets,
  setupTasks,
  type SetupInput,
  type SetupPart,
  type SetupPreset,
  type SetupTaskId,
  type SetupTaskState,
  type UnavailableReason,
} from "./voiceSetupLogic";

export function laneDescription(runtime: Pick<VoiceRuntime, "lane" | "cuda" | "nvidiaGpu">): string {
  if (runtime.lane === "mlx") return i18next.t("studio:voice.lane.mlxRuntime", { defaultValue: "Apple Silicon" });
  if (runtime.cuda || runtime.nvidiaGpu) return i18next.t("studio:voice.lane.nvidiaRuntime", { defaultValue: "NVIDIA GPU" });
  return i18next.t("studio:voice.lane.cpuRuntime", { defaultValue: "CPU" });
}

/** The speech model list the card reads (Studio and Models share ``useVoiceCatalog``). */
export interface SetupCatalog {
  models: VoiceModelRow[];
  loaded: boolean;
  refresh: () => Promise<void>;
}

export function setupInput(runtime: VoiceRuntime, catalog: Pick<SetupCatalog, "models" | "loaded">): SetupInput {
  return {
    lane: runtime.lane,
    installed: runtime.installed,
    needsUpdate: runtime.installed && (runtime.updateAvailable || !runtime.vadAvailable),
    extras: runtime.extras,
    ttsModelInstalled: runtime.ttsModelInstalled,
    models: catalog.models,
    modelsLoaded: catalog.loaded,
  };
}

function taskName(task: SetupTaskId): string {
  const t = (key: string, defaultValue: string) => i18next.t(`studio:voice.setup.task.${key}`, { defaultValue });
  switch (task) {
    case "quick":
      return t("quick", "Quick speech");
    case "natural":
      return t("natural", "Natural voices and style");
    case "clone":
      return t("clone", "Clone or design a voice");
    case "stt":
      return t("stt", "Speech to text on GPU");
    case "calls":
      return t("calls", "Voice calls");
  }
}

function partText(part: SetupPart): string {
  const t = (key: string, defaultValue: string, values: Record<string, unknown> = {}) =>
    i18next.t(`studio:voice.setup.missing.${key}`, { defaultValue, ...values });
  const gb = part.gb != null ? gbText(part.gb) : null;
  if (part.downloading !== undefined) {
    return part.downloading != null
      ? t("downloadingPct", "{name}, downloading ({pct}%)", { name: part.name ?? "", pct: Math.round(part.downloading * 100) })
      : t("downloading", "{name}, downloading", { name: part.name ?? "" });
  }
  switch (part.id) {
    case "runtime":
      return t("runtime", "voice runtime {gb} GB", { gb });
    case "update":
      return t("update", "a voice runtime update");
    case "kokoro":
      return t("kokoro", "Kokoro quick voices {gb} GB", { gb });
    case "speech-gpu":
      return t("speechGpu", "GPU speech part {gb} GB", { gb });
    case "stt-gpu":
      return t("sttGpu", "GPU speech-to-text part {gb} GB", { gb });
    case "natural-model":
      return part.name && gb ? t("naturalModel", "{name} {gb} GB", { name: part.name, gb }) : t("naturalModelUnknown", "a natural voice model");
    case "clone-model":
      return part.name && gb
        ? t("cloneModel", "a cloning model, such as {name} {gb} GB", { name: part.name, gb })
        : t("cloneModelUnknown", "a cloning model");
  }
}

function reasonText(reason: UnavailableReason, lane: VoiceRuntime["lane"]): string {
  const t = (key: string, defaultValue: string, values: Record<string, unknown> = {}) =>
    i18next.t(`studio:voice.setup.unavailable.${key}`, { defaultValue, ...values });
  switch (reason.kind) {
    case "needs-gpu":
      return lane === "portable"
        ? t("needsGpu", "needs an NVIDIA GPU that this computer does not have")
        : t("needsGpuMac", "needs a GPU this computer does not have");
    case "memory":
      return t("memory", "needs about {need} GB of GPU memory; this GPU has {have} GB", { need: reason.need, have: reason.have });
    case "compute":
      return t("compute", "needs an NVIDIA GPU of compute capability {need} or newer; this one is {have}", { need: reason.need, have: reason.have });
    case "no-model":
      return t("noModel", "no model for this runs here yet");
  }
}

/** The status cell of one task row, in words. */
export function taskStatusText(task: SetupTaskState, lane: VoiceRuntime["lane"]): string {
  const t = (key: string, defaultValue: string, values: Record<string, unknown> = {}) =>
    i18next.t(`studio:voice.setup.status.${key}`, { defaultValue, ...values });
  if (task.state === "ready") return task.task === "calls" ? t("readyInChat", "Ready, happens in Chat") : t("ready", "Ready");
  if (task.state === "unavailable") return t("unavailable", "Not on this machine: {why}", { why: reasonText(task.reason, lane) });
  const what = task.parts.map(partText).join(" + ");
  return task.task === "calls" ? t("missingInChat", "Missing: {what}. Happens in Chat", { what }) : t("missing", "Missing: {what}", { what });
}

type ChoiceSet = ReadonlySet<string>;
const MODEL_CHOICE = "model";

export interface VoiceSetupCardProps {
  runtime: VoiceRuntime;
  pack: VoicePackJob;
  catalog: SetupCatalog;
  /** Refresh the runtime status (after a part is removed). */
  onRuntimeChanged: () => void;
  /** Calls happen in Chat: the Calls row's button. */
  onOpenChat?: () => void;
  /** Where the cloning model is picked (Studio's Clone a voice). */
  onOpenClone?: () => void;
  /** Keeps the parts from changing while something else runs (the runtime is being removed). */
  busy?: boolean;
  /** Shown under the task list (Models: the Open Studio button, Studio: nothing). */
  footer?: ReactNode;
}

export function VoiceSetupCard({ runtime, pack, catalog, onRuntimeChanged, onOpenChat, onOpenClone, busy = false, footer }: VoiceSetupCardProps) {
  const { t } = useTranslation("studio");
  const { job } = pack;
  const { models, loaded } = catalog;
  const input = useMemo(() => setupInput(runtime, { models, loaded }), [runtime, models, loaded]);
  const tasks = useMemo(() => setupTasks(input), [input]);
  const plan = useMemo(() => installPlan(tasks), [tasks]);
  const presets = useMemo(() => setupPresets(input), [input]);
  const mode = voicePackMode({
    installed: runtime.installed,
    updateAvailable: runtime.updateAvailable,
    vadAvailable: runtime.vadAvailable,
    ttsModelInstalled: runtime.ttsModelInstalled,
    jobPhase: job.phase,
  });

  // Before install: null follows the preferred preset until the person picks a preset or ticks a part.
  const [choice, setChoice] = useState<ChoiceSet | null>(null);
  const presetSet = (preset: SetupPreset): ChoiceSet => {
    const picked = preset === "full" && presets.full ? presets.full : presets.basic;
    return new Set([...picked.addons, ...(picked.modelKey ? [MODEL_CHOICE] : [])]);
  };
  const chosen = choice ?? presetSet(presets.preferred);
  const sameAs = (preset: SetupPreset) => {
    const other = presetSet(preset);
    return other.size === chosen.size && [...other].every((id) => chosen.has(id));
  };
  const presetList: SetupPreset[] = presets.full ? ["full", "basic"] : ["basic"];
  const activePreset = presetList.find(sameAs) ?? null;
  const fullModelKey = presets.full?.modelKey ?? null;
  const fullModel = fullModelKey ? catalog.models.find((m) => m.key === fullModelKey) ?? null : null;

  const [busyExtra, setBusyExtra] = useState<string | null>(null);
  const [queuedModel, setQueuedModel] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<{ sentence: string; details: string } | null>(null);

  const refreshModels = catalog.refresh;
  const downloadModel = useCallback(
    async (key: string) => {
      setDownloadError(null);
      try {
        await downloadCloneModel(key);
      } catch (err) {
        const view = describeVoiceError(err);
        setDownloadError({ sentence: view.sentence, details: view.details });
      }
      await refreshModels();
    },
    [refreshModels],
  );

  // The model download waits for the install job: it starts once the job is done, and is dropped if the job fails.
  useEffect(() => {
    if (!queuedModel || job.running) return;
    const key = queuedModel;
    setQueuedModel(null);
    if (job.phase === "done") void downloadModel(key);
  }, [queuedModel, job.running, job.phase, downloadModel]);

  const install = async (addons: string[], modelKey: string | null, runJob: boolean) => {
    if (!runJob) {
      if (modelKey) await downloadModel(modelKey);
      return;
    }
    const started = await pack.start(addons);
    if (started !== false && modelKey) setQueuedModel(modelKey);
  };

  const toggleExtra = async (extra: VoiceExtra, on: boolean) => {
    if (!runtime.installed) {
      const next = new Set(chosen);
      if (on) next.add(extra.id);
      else next.delete(extra.id);
      setChoice(next);
      return;
    }
    if (on) {
      await pack.start([extra.id]);
      return;
    }
    pack.setError(null);
    setBusyExtra(extra.id);
    try {
      await removeVoiceExtra(extra.id);
      onRuntimeChanged();
    } catch (err) {
      pack.setError(err);
    } finally {
      setBusyExtra(null);
    }
  };

  const toggleModel = (on: boolean) => {
    const next = new Set(chosen);
    if (on) next.add(MODEL_CHOICE);
    else next.delete(MODEL_CHOICE);
    setChoice(next);
  };

  const chosenAddons = runtime.extras.filter((extra) => chosen.has(extra.id)).map((extra) => extra.id);
  const chosenModel = chosen.has(MODEL_CHOICE) ? fullModelKey : null;
  const chosenGb = installGb(chosenAddons, chosenModel && fullModel ? fullModel.downloadGb ?? fullModel.sizeGb : null);
  const working = job.running || queuedModel !== null;

  const presetLabel = (preset: SetupPreset) => {
    if (preset === "basic") {
      return t("voice.setup.presetBasic", { defaultValue: "Basic: quick voices, about {gb} GB", gb: gbText(presets.basic.gb) });
    }
    const full = presets.full!;
    return full.partial
      ? t("voice.setup.presetFullPartial", { defaultValue: "Full for this GPU: the GPU parts, about {gb} GB; the natural voice model is offered once the list of models loads", gb: gbText(full.gb) })
      : t("voice.setup.presetFull", { defaultValue: "Full for this GPU: natural voices, cloning and speech to text on the GPU, about {gb} GB", gb: gbText(full.gb) });
  };

  return (
    <div className="voice-setup">
      <table className="voice-setup__tasks">
        <tbody>
          {tasks.map((task) => (
            <tr key={task.task} data-task={task.task} data-state={task.state}>
              <th scope="row">{taskName(task.task)}</th>
              <td>
                <span className={task.state === "ready" ? "badge success" : "muted-text"}>{taskStatusText(task, runtime.lane)}</span>
              </td>
              <td>
                {task.task === "calls" && onOpenChat ? (
                  <button type="button" className="secondary-button" onClick={onOpenChat}>
                    {t("voice.setup.openChat", { defaultValue: "Open Chat" })}
                  </button>
                ) : task.task === "clone" && onOpenClone && task.state === "missing" && task.parts.every((p) => p.id === "clone-model") ? (
                  <button type="button" className="secondary-button" onClick={onOpenClone}>
                    {t("voice.setup.chooseCloneModel", { defaultValue: "Choose a cloning model" })}
                  </button>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {!runtime.installed && !working ? (
        <fieldset className="voice-setup__presets">
          <legend>{t("voice.setup.presets", { defaultValue: "What to install" })}</legend>
          {presetList.map((preset) => (
            <label key={preset} className="checkbox-label">
              <input type="radio" name="voice-setup-preset" checked={activePreset === preset} onChange={() => setChoice(presetSet(preset))} />
              {presetLabel(preset)}
            </label>
          ))}
        </fieldset>
      ) : null}

      <ChooseParts
        extras={runtime.extras}
        installed={runtime.installed}
        disabled={working || busyExtra !== null || busy}
        chosen={(extra) =>
          runtime.installed
            ? // Ticked while its install runs; unticked while its removal runs.
              (extra.installed || (job.running && job.extras.includes(extra.id))) && busyExtra !== extra.id
            : chosen.has(extra.id)
        }
        onToggle={(extra, on) => void toggleExtra(extra, on)}
        model={!runtime.installed && fullModel ? { name: fullModel.name, gb: fullModel.downloadGb ?? fullModel.sizeGb } : null}
        modelChosen={chosen.has(MODEL_CHOICE)}
        onToggleModel={toggleModel}
      />

      {job.running ? (
        <InstallProgress pack={pack} />
      ) : queuedModel ? null : !runtime.installed ? (
        <div className="button-row">
          <button type="button" className="primary-button" disabled={busy} onClick={() => void install(chosenAddons, chosenModel, true)}>
            {t("voice.setup.installChosen", { defaultValue: "Install, about {gb} GB", gb: gbText(chosenGb) })}
          </button>
        </div>
      ) : plan.runJob || plan.modelKey ? (
        <div className="button-row">
          <button type="button" className="primary-button" disabled={busy} onClick={() => void install(plan.addons, plan.modelKey, plan.runJob)}>
            {plan.gb > 0
              ? plan.partial
                ? t("voice.setup.installMissingOver", { defaultValue: "Install what is missing, over {gb} GB", gb: gbText(plan.gb) })
                : t("voice.setup.installMissingGb", { defaultValue: "Install what is missing, {gb} GB", gb: gbText(plan.gb) })
              : t("voice.setup.installMissing", { defaultValue: "Install what is missing" })}
          </button>
        </div>
      ) : mode === "repair" ? (
        <div className="button-row">
          <button type="button" className="primary-button" disabled={busy} onClick={() => void pack.start()}>
            {t("voice.setup.repair", { defaultValue: "Repair Voice setup" })}
          </button>
        </div>
      ) : null}

      {footer}
      {downloadError ? <SetupError sentence={downloadError.sentence} details={downloadError.details} /> : null}
      <InstallLog pack={pack} />
    </div>
  );
}

/** True when the card has something to install here (a task Missing a part the button or the Clone screen gets). */
export function setupHasMissing(runtime: VoiceRuntime, catalog: Pick<SetupCatalog, "models" | "loaded">): boolean {
  return setupTasks(setupInput(runtime, catalog)).some((task) => task.state === "missing");
}
