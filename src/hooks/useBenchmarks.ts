import { useEffect, useRef, useState } from "react";
import { cancelBenchmark, deleteBenchmark, runBenchmark } from "../api";
import { emptyWorkspace } from "../defaults";
import { BENCHMARK_PROMPTS } from "../constants";
import { syncRuntime } from "../utils";
import type {
  BenchmarkCancelOutcome,
  BenchmarkRunPayload,
  LaunchPreferences,
  TabId,
  WorkspaceData,
} from "../types";
import type { ChatModelOption } from "../types/chat";

export function useBenchmarks(
  workspace: WorkspaceData,
  setWorkspace: React.Dispatch<React.SetStateAction<WorkspaceData>>,
  launchSettings: LaunchPreferences,
  activeTab: TabId,
  setError: (msg: string | null) => void,
  setBusyAction: (action: string | null) => void,
) {
  const [benchmarkDraft, setBenchmarkDraft] = useState<BenchmarkRunPayload>({
    cacheBits: emptyWorkspace.settings.launchPreferences.cacheBits,
    fp16Layers: emptyWorkspace.settings.launchPreferences.fp16Layers,
    fusedAttention: emptyWorkspace.settings.launchPreferences.fusedAttention,
    cacheStrategy: emptyWorkspace.settings.launchPreferences.cacheStrategy,
    fitModelInMemory: emptyWorkspace.settings.launchPreferences.fitModelInMemory,
    speculativeDecoding: emptyWorkspace.settings.launchPreferences.speculativeDecoding,
    treeBudget: emptyWorkspace.settings.launchPreferences.treeBudget,
    kvBudget: emptyWorkspace.settings.launchPreferences.kvBudget,
    contextTokens: emptyWorkspace.settings.launchPreferences.contextTokens,
    maxTokens: 4096,
    temperature: 0.2,
  });
  const [benchmarkModelKey, setBenchmarkModelKey] = useState("");
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState("");
  const [compareBenchmarkId, setCompareBenchmarkId] = useState("");
  const [benchmarkModelFilter, setBenchmarkModelFilter] = useState<string | null>(null);
  const [benchmarkViewMode, setBenchmarkViewMode] = useState<"table" | "chart" | "both">("both");
  const [benchmarkPromptId, setBenchmarkPromptId] = useState(BENCHMARK_PROMPTS[0]?.id ?? "balanced");
  const [benchmarkStartedAt, setBenchmarkStartedAt] = useState<number | null>(null);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
  // Cancel was pressed and the backend has not answered the run request yet.
  const [benchmarkCancelling, setBenchmarkCancelling] = useState(false);
  const [benchmarkCancelOutcome, setBenchmarkCancelOutcome] = useState<BenchmarkCancelOutcome | null>(null);
  const [showBenchmarkPicker, setShowBenchmarkPicker] = useState(false);
  const [showBenchmarkModal, setShowBenchmarkModal] = useState(false);

  // Sync benchmarkDraft from launchSettings
  useEffect(() => {
    setBenchmarkDraft((current) => {
      if (
        current.cacheBits === launchSettings.cacheBits &&
        current.fp16Layers === launchSettings.fp16Layers &&
        current.fusedAttention === launchSettings.fusedAttention &&
        current.cacheStrategy === launchSettings.cacheStrategy &&
        current.fitModelInMemory === launchSettings.fitModelInMemory &&
        current.contextTokens === launchSettings.contextTokens
      ) return current;
      return {
        ...current,
        cacheBits: launchSettings.cacheBits,
        fp16Layers: launchSettings.fp16Layers,
        fusedAttention: launchSettings.fusedAttention,
        cacheStrategy: launchSettings.cacheStrategy,
        fitModelInMemory: launchSettings.fitModelInMemory,
        contextTokens: launchSettings.contextTokens,
      };
    });
  }, [
    launchSettings.contextTokens,
    launchSettings.fitModelInMemory,
    launchSettings.fp16Layers,
    launchSettings.fusedAttention,
    launchSettings.cacheBits,
    launchSettings.cacheStrategy,
  ]);

  // Benchmark selection sync
  useEffect(() => {
    if (!workspace.benchmarks.length) {
      setSelectedBenchmarkId("");
      setCompareBenchmarkId("");
      return;
    }
    setSelectedBenchmarkId((current) =>
      workspace.benchmarks.some((benchmark) => benchmark.id === current) ? current : workspace.benchmarks[0]?.id ?? "",
    );
    setCompareBenchmarkId((current) =>
      workspace.benchmarks.some((benchmark) => benchmark.id === current)
        ? current
        : workspace.benchmarks[1]?.id ?? workspace.benchmarks[0]?.id ?? "",
    );
  }, [workspace.benchmarks]);

  function updateBenchmarkDraft<K extends keyof BenchmarkRunPayload>(key: K, value: BenchmarkRunPayload[K]) {
    setBenchmarkDraft((current) => ({
      ...current,
      [key]: value,
    }));
  }

  // Only used when the backend cannot be reached to cancel: it stops the wait, nothing more.
  const benchmarkAbortRef = useRef<AbortController | null>(null);

  async function handleCancelBenchmark() {
    if (!benchmarkAbortRef.current || benchmarkCancelling) return;
    setBenchmarkCancelling(true);
    try {
      // The run request itself answers once the backend has stopped: that answer ends the dialog.
      await cancelBenchmark();
    } catch {
      benchmarkAbortRef.current?.abort();
    }
  }

  async function handleRunBenchmark(benchmarkOption: ChatModelOption | null) {
    const promptPreset = BENCHMARK_PROMPTS.find((item) => item.id === benchmarkPromptId) ?? BENCHMARK_PROMPTS[0];
    if (!benchmarkOption) {
      setError("Choose a model before running a benchmark.");
      return;
    }

    setBusyAction("Running benchmark...");
    setBenchmarkStartedAt(Date.now());
    setShowBenchmarkModal(true);
    setBenchmarkError(null);
    setBenchmarkCancelling(false);
    setBenchmarkCancelOutcome(null);
    const controller = new AbortController();
    benchmarkAbortRef.current = controller;

    try {
      const response = await runBenchmark(
        {
          ...benchmarkDraft,
          modelRef: benchmarkOption.modelRef,
          modelName: benchmarkOption.model,
          source: benchmarkOption.source,
          backend: benchmarkOption.backend,
          path: benchmarkOption.path ?? undefined,
          prompt: promptPreset.prompt,
          label: `${benchmarkOption.model} / ${benchmarkDraft.cacheStrategy === "native" ? "Native f16" : `${benchmarkDraft.cacheStrategy} ${benchmarkDraft.cacheBits}-bit ${benchmarkDraft.fp16Layers}+${benchmarkDraft.fp16Layers}`} / ${Math.round(benchmarkDraft.contextTokens / 1024)}K ctx`,
        },
        { signal: controller.signal },
      );
      setWorkspace((current) =>
        syncRuntime(
          {
            ...current,
            benchmarks: response.benchmarks,
          },
          response.runtime,
        ),
      );
      const result = response.result;
      if (response.cancelled || !result) {
        // The backend stopped the run and saved nothing; the runtime above already reflects an unload.
        setBenchmarkCancelOutcome({ confirmed: true, modelUnloaded: Boolean(response.modelUnloaded) });
        return;
      }
      setSelectedBenchmarkId(result.id);
      setCompareBenchmarkId((current) => (current === result.id ? selectedBenchmarkId : current));
    } catch (actionError) {
      const stoppedWaiting =
        controller.signal.aborted ||
        (actionError instanceof Error && actionError.message === "Request cancelled");
      if (stoppedWaiting) {
        // The cancel never reached the backend, so the run may still be going there.
        setBenchmarkCancelOutcome({ confirmed: false, modelUnloaded: false });
      } else {
        const message = actionError instanceof Error ? actionError.message : "Failed to run benchmark.";
        setError(message);
        setBenchmarkError(message);
      }
    } finally {
      if (benchmarkAbortRef.current === controller) benchmarkAbortRef.current = null;
      setBusyAction(null);
      setBenchmarkStartedAt(null);
      setBenchmarkCancelling(false);
    }
  }

  async function handleDeleteBenchmark(runId: string) {
    try {
      const response = await deleteBenchmark(runId);
      setWorkspace((current) => ({ ...current, benchmarks: response.benchmarks }));
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Failed to delete the benchmark run.");
    }
  }

  // Computed
  const selectedBenchmark = workspace.benchmarks.find((item) => item.id === selectedBenchmarkId) ?? workspace.benchmarks[0] ?? null;
  const compareBenchmark =
    workspace.benchmarks.find((item) => item.id === compareBenchmarkId && item.id !== selectedBenchmark?.id) ??
    workspace.benchmarks.find((item) => item.id !== selectedBenchmark?.id) ??
    null;
  const benchmarkSpeedDelta = selectedBenchmark && compareBenchmark ? selectedBenchmark.tokS - compareBenchmark.tokS : 0;
  const benchmarkCacheDelta =
    selectedBenchmark && compareBenchmark ? selectedBenchmark.cacheGb - compareBenchmark.cacheGb : 0;
  const benchmarkLatencyDelta =
    selectedBenchmark && compareBenchmark ? selectedBenchmark.responseSeconds - compareBenchmark.responseSeconds : 0;
  const benchmarkMaxTokS = Math.max(1, ...workspace.benchmarks.map((item) => item.tokS));
  const benchmarkMaxCacheGb = Math.max(1, ...workspace.benchmarks.map((item) => item.baselineCacheGb || item.cacheGb));

  return {
    benchmarkDraft,
    setBenchmarkDraft,
    benchmarkModelKey,
    setBenchmarkModelKey,
    selectedBenchmarkId,
    setSelectedBenchmarkId,
    compareBenchmarkId,
    setCompareBenchmarkId,
    benchmarkModelFilter,
    setBenchmarkModelFilter,
    benchmarkViewMode,
    setBenchmarkViewMode,
    benchmarkPromptId,
    setBenchmarkPromptId,
    benchmarkStartedAt,
    benchmarkError,
    benchmarkCancelling,
    benchmarkCancelOutcome,
    showBenchmarkPicker,
    setShowBenchmarkPicker,
    showBenchmarkModal,
    setShowBenchmarkModal,
    updateBenchmarkDraft,
    handleRunBenchmark,
    handleCancelBenchmark,
    handleDeleteBenchmark,
    selectedBenchmark,
    compareBenchmark,
    benchmarkSpeedDelta,
    benchmarkCacheDelta,
    benchmarkLatencyDelta,
    benchmarkMaxTokS,
    benchmarkMaxCacheGb,
  };
}
