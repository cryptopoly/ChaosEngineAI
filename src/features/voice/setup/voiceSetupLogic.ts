/**
 * Pure rules for the Voice setup card (voice studio UX audit, change 1; D0141 presets, D0143 names): for each task
 * (Quick speech, Natural voices and style, Clone or design, Speech to text on GPU, Calls) whether it is ready, what it
 * lacks with sizes, or why this machine cannot do it; what "Install what is missing" installs; the Basic and Full
 * presets offered before the first install. Sizes come from ``runtimeSize.ts`` and the speech model catalog only.
 */
import { readFit } from "../../../components/models/fit";
import type { VoiceModelRow } from "../../../types";
import { ADDON_GB, RUNTIME_BASE_GB } from "../runtimeSize";

export type SetupTaskId = "quick" | "natural" | "clone" | "stt" | "calls";

export const SETUP_TASKS: SetupTaskId[] = ["quick", "natural", "clone", "stt", "calls"];

/** One missing piece: the runtime, its update, an add-on or a speech model. ``gb`` is null when no size is known. */
export interface SetupPart {
  id: "runtime" | "update" | "kokoro" | "speech-gpu" | "stt-gpu" | "natural-model" | "clone-model";
  gb: number | null;
  /** Speech models: the catalog key and name. */
  modelKey?: string;
  name?: string;
  /** Speech models: 0..1 while the download runs (null when it runs without a figure); absent when not downloading. */
  downloading?: number | null;
}

export type UnavailableReason =
  | { kind: "needs-gpu" }
  | { kind: "memory" | "compute"; need: number; have: number }
  | { kind: "no-model" };

export type SetupTaskState =
  | { task: SetupTaskId; state: "ready" }
  | { task: SetupTaskId; state: "missing"; parts: SetupPart[] }
  | { task: SetupTaskId; state: "unavailable"; reason: UnavailableReason };

export type SetupModel = Pick<
  VoiceModelRow,
  "key" | "name" | "kind" | "lanes" | "capabilities" | "installed" | "sizeGb" | "downloadGb" | "unsupportedFacts" | "fit" | "download"
>;

export interface SetupInput {
  lane: "mlx" | "portable";
  installed: boolean;
  /** The installed runtime lacks something this version provides (an update, or no turn detection for calls). */
  needsUpdate: boolean;
  /** The add-ons the backend offers here; it lists the GPU ones only where this GPU can install them. */
  extras: Array<{ id: string; installed: boolean }>;
  /** Kokoro's files are on disk (an older backend lists no Kokoro add-on). */
  ttsModelInstalled: boolean;
  models: SetupModel[];
  /** The speech model list has been fetched; until then a model's size is unknown rather than absent. */
  modelsLoaded: boolean;
}

const offered = (input: SetupInput, id: string) => input.extras.find((extra) => extra.id === id);

function addonPart(input: SetupInput, id: "kokoro" | "speech-gpu" | "stt-gpu"): SetupPart[] {
  const extra = offered(input, id);
  return extra && !extra.installed ? [{ id, gb: ADDON_GB[id] ?? null }] : [];
}

function basePart(input: SetupInput): SetupPart[] {
  if (!input.installed) return [{ id: "runtime", gb: RUNTIME_BASE_GB }];
  return input.needsUpdate ? [{ id: "update", gb: null }] : [];
}

/** The catalog lane a speech model must list to run here: MLX on a Mac, the speech GPU add-on's CUDA on a PC. */
function onLane(model: SetupModel, lane: SetupInput["lane"]): boolean {
  return !model.lanes || model.lanes.includes(lane === "mlx" ? "mlx" : "cuda");
}

const downloadGb = (model: SetupModel) => model.downloadGb ?? model.sizeGb;

/** The memory or age problem the backend reported, or "wont-fit" from the measured peak; null when it may run. */
function blockedBy(model: SetupModel): UnavailableReason | null {
  const facts = model.unsupportedFacts;
  if (facts) return { kind: facts.kind, need: facts.need, have: facts.have };
  const fit = readFit(model.fit);
  if (fit?.status === "wont-fit" && fit.needGb != null) return { kind: "memory", need: fit.needGb, have: fit.usableGb };
  return null;
}

/**
 * The model a task needs: one already on disk that runs here, else the one to download (natural: the style-capable
 * CustomVoice row, as Speak's install card offers; clone: the smallest clone or design row), else why none can run.
 */
function pickModel(
  input: SetupInput,
  wants: (model: SetupModel) => boolean,
  prefer: (a: SetupModel, b: SetupModel) => number,
): { have: true } | { download: SetupModel } | { reason: UnavailableReason } | { unknown: true } {
  const rows = input.models.filter((model) => onLane(model, input.lane) && wants(model));
  if (rows.some((model) => model.installed && !blockedBy(model))) return { have: true };
  const runnable = rows.filter((model) => !blockedBy(model)).sort(prefer);
  if (runnable.length > 0) return { download: runnable[0] };
  if (rows.length > 0) return { reason: blockedBy(rows[0]) ?? { kind: "no-model" } };
  return input.modelsLoaded ? { reason: { kind: "no-model" } } : { unknown: true };
}

function modelPart(id: "natural-model" | "clone-model", model: SetupModel): SetupPart {
  const part: SetupPart = { id, gb: downloadGb(model), modelKey: model.key, name: model.name };
  if (model.download?.state === "downloading") part.downloading = model.download.progress ?? null;
  return part;
}

export function naturalModelPick(input: SetupInput) {
  return pickModel(
    input,
    (model) => model.capabilities.style,
    (a, b) => Number(b.kind === "custom") - Number(a.kind === "custom") || downloadGb(a) - downloadGb(b),
  );
}

export function cloneModelPick(input: SetupInput) {
  return pickModel(input, (model) => model.capabilities.clone || model.capabilities.design, (a, b) => downloadGb(a) - downloadGb(b));
}

function result(task: SetupTaskId, parts: SetupPart[]): SetupTaskState {
  return parts.length === 0 ? { task, state: "ready" } : { task, state: "missing", parts };
}

/** The speech-model tasks (natural, clone) on this machine. */
function speechModelTask(task: "natural" | "clone", input: SetupInput): SetupTaskState {
  // On a PC the speech models run only with the speech GPU add-on, which the backend offers only for a usable GPU.
  if (input.lane === "portable" && !offered(input, "speech-gpu")) return { task, state: "unavailable", reason: { kind: "needs-gpu" } };
  const pick = task === "natural" ? naturalModelPick(input) : cloneModelPick(input);
  if ("reason" in pick) return { task, state: "unavailable", reason: pick.reason };
  const parts = [...basePart(input).filter((p) => p.id === "runtime"), ...addonPart(input, "speech-gpu")];
  if ("download" in pick) parts.push(modelPart(task === "natural" ? "natural-model" : "clone-model", pick.download));
  // Size unknown until the list arrives: the model still counts as missing.
  if ("unknown" in pick) parts.push({ id: task === "natural" ? "natural-model" : "clone-model", gb: null });
  return result(task, parts);
}

/** Every task's state, in the card's order. */
export function setupTasks(input: SetupInput): SetupTaskState[] {
  const runtimeOnly = basePart(input).filter((p) => p.id === "runtime");
  const kokoroExtra = offered(input, "kokoro");
  const quickParts = [...runtimeOnly, ...(kokoroExtra ? addonPart(input, "kokoro") : [])];
  // An older backend without the Kokoro add-on: the quick voices came with the runtime.
  if (!kokoroExtra && input.installed && !input.ttsModelInstalled) quickParts.push({ id: "kokoro", gb: ADDON_GB.kokoro });

  let stt: SetupTaskState;
  if (input.lane === "mlx") stt = result("stt", runtimeOnly);
  else if (!offered(input, "stt-gpu")) stt = { task: "stt", state: "unavailable", reason: { kind: "needs-gpu" } };
  else stt = result("stt", [...runtimeOnly, ...addonPart(input, "stt-gpu")]);

  return [
    result("quick", quickParts),
    speechModelTask("natural", input),
    speechModelTask("clone", input),
    stt,
    // Calls need the runtime with turn detection; an older runtime needs its update.
    result("calls", basePart(input)),
  ];
}

/** What one "Install what is missing" click does: one install job (runtime, update, add-ons), then one download. */
export interface InstallPlan {
  /** The install job runs (runtime missing or out of date, or add-ons to add). */
  runJob: boolean;
  /** Add-ons the job adds. */
  addons: string[];
  /** The speech model downloaded once the job is done; null for none. */
  modelKey: string | null;
  /** Known size in GB, one decimal; ``partial`` when a part's size is unknown (the update, an unlisted model). */
  gb: number;
  partial: boolean;
}

/**
 * The parts every task lacks, once each. The cloning model is left to Clone a voice, where the person picks it, so
 * the button installs what the Full preset does (D0141): add-ons and the recommended natural voice model.
 */
export function installPlan(tasks: SetupTaskState[]): InstallPlan {
  const parts = new Map<string, SetupPart>();
  for (const task of tasks) {
    if (task.state !== "missing") continue;
    for (const part of task.parts) if (part.id !== "clone-model" && part.downloading === undefined) parts.set(part.id, part);
  }
  const list = [...parts.values()];
  const addons = list.filter((p) => p.id === "kokoro" || p.id === "speech-gpu" || p.id === "stt-gpu").map((p) => p.id);
  const gb = list.reduce((sum, p) => sum + (p.gb ?? 0), 0);
  return {
    runJob: list.some((p) => p.id === "runtime" || p.id === "update") || addons.length > 0,
    addons,
    modelKey: parts.get("natural-model")?.modelKey ?? null,
    gb: Math.round(gb * 10) / 10,
    partial: list.some((p) => p.gb == null),
  };
}

export type SetupPreset = "basic" | "full";

export interface PresetChoice {
  addons: string[];
  modelKey: string | null;
  gb: number;
  /** A model is part of the preset but its size is not known yet. */
  partial: boolean;
}

/**
 * The two first-install presets (D0141): Basic is the runtime with the Kokoro quick voices; Full for this GPU adds
 * every GPU add-on offered here and the recommended natural voice model. ``full`` is null where it would add nothing
 * (a CPU-only PC).
 */
export function setupPresets(input: SetupInput): { basic: PresetChoice; full: PresetChoice | null; preferred: SetupPreset } {
  const ids = (list: string[]) => list.filter((id) => offered(input, id));
  const basicAddons = ids(["kokoro"]);
  const basic: PresetChoice = { addons: basicAddons, modelKey: null, gb: installGb(basicAddons, null), partial: false };

  const natural = speechModelTask("natural", input);
  const fullAddons = ids(["kokoro", "speech-gpu", "stt-gpu"]);
  const modelPartFound = natural.state === "missing" ? natural.parts.find((p) => p.id === "natural-model") : undefined;
  // A model counts once the list names it; before that Full is its add-ons, marked as more to come.
  const model = modelPartFound?.modelKey ? modelPartFound : undefined;
  if (fullAddons.length === basicAddons.length && !model) return { basic, full: null, preferred: "basic" };
  const full: PresetChoice = {
    addons: fullAddons,
    modelKey: model?.modelKey ?? null,
    gb: installGb(fullAddons, model?.gb ?? null),
    partial: Boolean(modelPartFound && !model),
  };
  // Full is preselected when this GPU can run the natural voices; a GPU too small or too old starts on Basic.
  return { basic, full, preferred: natural.state === "unavailable" ? "basic" : "full" };
}

/** A first install's size: the runtime, these add-ons and the model, rounded once to one decimal. */
export function installGb(addons: string[], modelGb: number | null): number {
  const total = RUNTIME_BASE_GB + addons.reduce((sum, id) => sum + (ADDON_GB[id] ?? 0), 0) + (modelGb ?? 0);
  return Math.round(total * 10) / 10;
}

/** The GB figure as the card prints it: one decimal, "0.3" rather than "0". */
export function gbText(gb: number): string {
  return gb > 0 && gb < 0.1 ? gb.toFixed(2) : gb.toFixed(1);
}
