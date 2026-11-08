import { BenchmarkRunTab } from "../features/benchmarks/BenchmarkRunTab";
import { BenchmarkHistoryTab } from "../features/benchmarks/BenchmarkHistoryTab";
import type { TabId } from "../types";
import type { AppTabContext } from "./context";

export function BenchmarkRunView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    chatLibrary,
    threadModelOptions,
    benchmarkOption,
    benchmarks,
    settings,
    busy,
    busyAction,
    handleInstallMtplx,
    installingMtplx,
    mtplxJob,
    tensorfoldControls,
    isAppleSilicon,
    setActiveTab,
  } = ctx;
  const {
    benchmarkDraft,
    benchmarkPromptId,
    benchmarkStartedAt,
    benchmarkError,
    showBenchmarkPicker,
    showBenchmarkModal,
    updateBenchmarkDraft,
    setBenchmarkPromptId,
    setBenchmarkModelKey,
    setBenchmarkDraft,
    setShowBenchmarkPicker,
    setShowBenchmarkModal,
    setSelectedBenchmarkId,
    setCompareBenchmarkId,
  } = benchmarks;
  const {
    preview,
    installingPackage,
    installLogs,
    handleInstallPackage,
  } = settings;
  return (
    <BenchmarkRunTab
      workspace={{
        benchmarks: workspace.benchmarks,
        library: chatLibrary,
        system: {
          availableMemoryGb: workspace.system.availableMemoryGb,
          totalMemoryGb: workspace.system.totalMemoryGb,
          availableCacheStrategies: workspace.system.availableCacheStrategies,
          llamaServerTurboPath: workspace.system.llamaServerTurboPath,
          dflash: workspace.system.dflash,
          mtplx: workspace.system.mtplx,
        },
      }}
      threadModelOptions={threadModelOptions}
      benchmarkDraft={benchmarkDraft}
      benchmarkOption={benchmarkOption}
      benchmarkPromptId={benchmarkPromptId}
      preview={preview}
      busy={busy}
      busyAction={busyAction}
      benchmarkStartedAt={benchmarkStartedAt}
      benchmarkError={benchmarkError}
      showBenchmarkPicker={showBenchmarkPicker}
      showBenchmarkModal={showBenchmarkModal}
      installingPackage={installingPackage}
      installLogs={installLogs}
      onInstallMtplx={() => void handleInstallMtplx()}
      installingMtplx={installingMtplx}
      mtplxJob={mtplxJob}
      tensorfold={tensorfoldControls}
      isAppleSilicon={isAppleSilicon}
      onBenchmarkDraftChange={updateBenchmarkDraft}
      onBenchmarkPromptIdChange={setBenchmarkPromptId}
      onBenchmarkModelKeyChange={setBenchmarkModelKey}
      onBenchmarkDraftUpdate={setBenchmarkDraft}
      onRunBenchmark={() => void benchmarks.handleRunBenchmark(benchmarkOption)}
      onCancelBenchmark={benchmarks.handleCancelBenchmark}
      onShowBenchmarkPickerChange={setShowBenchmarkPicker}
      onShowBenchmarkModalChange={setShowBenchmarkModal}
      onSelectedBenchmarkIdChange={setSelectedBenchmarkId}
      onCompareBenchmarkIdChange={setCompareBenchmarkId}
      onActiveTabChange={(tab) => setActiveTab(tab as TabId)}
      onInstallPackage={handleInstallPackage}
    />
  );
}

export function BenchmarkHistoryView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    benchmarks,
  } = ctx;
  const {
    benchmarkModelFilter,
    benchmarkViewMode,
    selectedBenchmarkId,
    compareBenchmarkId,
    setBenchmarkModelFilter,
    setBenchmarkViewMode,
    setSelectedBenchmarkId,
    setCompareBenchmarkId,
  } = benchmarks;
  return (
    <BenchmarkHistoryTab
      benchmarks={workspace.benchmarks}
      benchmarkModelFilter={benchmarkModelFilter}
      benchmarkViewMode={benchmarkViewMode}
      selectedBenchmarkId={selectedBenchmarkId}
      compareBenchmarkId={compareBenchmarkId}
      onBenchmarkModelFilterChange={setBenchmarkModelFilter}
      onBenchmarkViewModeChange={setBenchmarkViewMode}
      onSelectedBenchmarkIdChange={setSelectedBenchmarkId}
      onCompareBenchmarkIdChange={setCompareBenchmarkId}
    />
  );
}
