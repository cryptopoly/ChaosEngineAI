import { useEffect, useState } from "react";
import { runBenchmark } from "../api";
import { emptyWorkspace } from "../defaults";
import { BENCHMARK_PROMPTS } from "../constants";
import { syncRuntime } from "../utils";
import type {
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

    try {
      const response = await runBenchmark({
        ...benchmarkDraft,
        modelRef: benchmarkOption.modelRef,
        modelName: benchmarkOption.model,
        source: benchmarkOption.source,
        backend: benchmarkOption.backend,
        path: benchmarkOption.path ?? undefined,
        prompt: promptPreset.prompt,
        label: `${benchmarkOption.model} / ${benchmarkDraft.cacheStrategy === "native" ? "Native f16" : `${benchmarkDraft.cacheStrategy} ${benchmarkDraft.cacheBits}-bit ${benchmarkDraft.fp16Layers}+${benchmarkDraft.fp16Layers}`} / ${Math.round(benchmarkDraft.contextTokens / 1024)}K ctx`,
      });
      setWorkspace((current) =>
        syncRuntime(
          {
            ...current,
            benchmarks: response.benchmarks,
          },
          response.runtime,
        ),
      );
      setSelectedBenchmarkId(response.result.id);
      setCompareBenchmarkId((current) => (current === response.result.id ? selectedBenchmarkId : current));
    } catch (actionError) {
      const message = actionError instanceof Error ? actionError.message : "Failed to run benchmark.";
      setError(message);
      setBenchmarkError(message);
    } finally {
      setBusyAction(null);
      setBenchmarkStartedAt(null);
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
    showBenchmarkPicker,
    setShowBenchmarkPicker,
    showBenchmarkModal,
    setShowBenchmarkModal,
    updateBenchmarkDraft,
    handleRunBenchmark,
    selectedBenchmark,
    compareBenchmark,
    benchmarkSpeedDelta,
    benchmarkCacheDelta,
    benchmarkLatencyDelta,
    benchmarkMaxTokS,
    benchmarkMaxCacheGb,
  };
}
