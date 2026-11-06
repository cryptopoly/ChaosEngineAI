import { updateSettings as updateSettingsApi } from "../api";
import { ChatModelsPage } from "../features/models/ChatModelsPage";
import { ConversionTab } from "../features/conversion/ConversionTab";
import type { AppTabContext } from "./context";

export function ChatModelsView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    chatLibrary,
    models,
    modelsAvailability,
    setAreaAvailability,
    isAppleSilicon,
    enabledDirectoryCount,
    fileRevealLabel,
    refreshWorkspace,
    handleDeleteModel,
    prepareCatalogConversion,
    openModelSelector,
    handleOpenExternalUrl,
    handleRevealPath,
  } = ctx;
  const {
    searchInput,
    setSearchInput,
    hubResults,
    searchError,
    expandedHubId,
    hubFileCache,
    hubFileLoading,
    hubFileError,
    setDetailFamilyId,
    activeDownloads,
    discoverCapFilter,
    setDiscoverCapFilter,
    discoverFormatFilter,
    setDiscoverFormatFilter,
    discoverAccelFilter,
    setDiscoverAccelFilter,
    chatModelsBackendFilter,
    setChatModelsBackendFilter,
    chatModelsSort,
    setChatModelsSort,
    chatModelsSortDir,
    setChatModelsSortDir,
    handleDownloadModel,
    handleCancelModelDownload,
    handleDeleteModelDownload,
    toggleHubExpand,
  } = models;
  return (
    <ChatModelsPage
      families={workspace.featuredModels}
      library={chatLibrary}
      activeDownloads={activeDownloads}
      searchInput={searchInput}
      onSearchInputChange={setSearchInput}
      searchError={searchError}
      hubResults={hubResults}
      expandedHubId={expandedHubId}
      onToggleHubExpand={(repo) => void toggleHubExpand(repo)}
      hubFileCache={hubFileCache}
      hubFileLoading={hubFileLoading}
      hubFileError={hubFileError}
      capFilter={discoverCapFilter}
      onCapFilterChange={setDiscoverCapFilter}
      formatFilter={discoverFormatFilter}
      onFormatFilterChange={setDiscoverFormatFilter}
      accelFilter={discoverAccelFilter}
      onAccelFilterChange={setDiscoverAccelFilter}
      backendFilter={chatModelsBackendFilter}
      onBackendFilterChange={setChatModelsBackendFilter}
      sort={chatModelsSort}
      sortDir={chatModelsSortDir}
      onSortChange={(sort, dir) => {
        setChatModelsSort(sort);
        setChatModelsSortDir(dir);
      }}
      availability={modelsAvailability.chat}
      onAvailabilityChange={(value) => setAreaAvailability("chat", value)}
      accelCompat={{
        dflashModels: workspace.system.dflash?.supportedModels ?? [],
        mtplxModels: workspace.system.mtplx?.supportedModels ?? [],
        tensorfoldModels: workspace.system.tensorfold?.supportedModels ?? [],
        turboInstalled: Boolean(workspace.system.llamaServerTurboPath),
        turboquantMlxAvailable: workspace.system.availableCacheStrategies?.some((s) => s.id === "turboquant" && s.available) ?? false,
        appleSilicon: isAppleSilicon,
      }}
      hostSystem={workspace.system}
      loadedModel={workspace.runtime.loadedModel}
      availableMemoryGb={workspace.system.availableMemoryGb}
      enabledDirectoryCount={enabledDirectoryCount}
      fileRevealLabel={fileRevealLabel}
      favoriteModelRefs={workspace.settings?.favoriteModelRefs ?? []}
      onToggleFavoriteModel={(ref) => {
        const current = workspace.settings?.favoriteModelRefs ?? [];
        const next = current.includes(ref)
          ? current.filter((r) => r !== ref)
          : [...current, ref];
        void updateSettingsApi({ favoriteModelRefs: next }).then(() => {
          void refreshWorkspace();
        });
      }}
      onDownloadModel={(repo) => void handleDownloadModel(repo)}
      onCancelModelDownload={(repo) => void handleCancelModelDownload(repo)}
      onDeleteModelDownload={(repo) => void handleDeleteModelDownload(repo)}
      onDeleteModel={(item) => void handleDeleteModel(item)}
      onPrepareCatalogConversion={prepareCatalogConversion}
      onOpenModelSelector={openModelSelector}
      onOpenFamilyDetails={setDetailFamilyId}
      onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
      onRevealPath={(path) => void handleRevealPath(path)}
    />
  );
}

export function ConversionView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    chatLibrary,
    settings,
    conversionSource,
    conversionVariant,
    conversionReady,
    convertibleLibrary,
    nativeBackends,
    launchCacheLabel,
    busy,
    busyAction,
    conversionStartedAt,
    conversionError,
    showConversionPicker,
    showConversionModal,
    fileRevealLabel,
    setShowConversionPicker,
    setShowConversionModal,
    handleConvertModel,
    handlePickConversionOutputDir,
    prepareLibraryConversion,
    handleRevealPath,
  } = ctx;
  const {
    preview,
    conversionDraft,
    lastConversion,
    setConversionDraft,
    updateConversionDraft,
  } = settings;
  return (
    <ConversionTab
      conversionSource={conversionSource}
      conversionVariant={conversionVariant}
      conversionDraft={conversionDraft}
      lastConversion={lastConversion}
      conversionReady={conversionReady}
      convertibleLibrary={convertibleLibrary}
      nativeBackends={nativeBackends}
      preview={preview}
      workspace={{ system: workspace.system, library: chatLibrary }}
      launchCacheLabel={launchCacheLabel}
      busy={busy}
      busyAction={busyAction}
      conversionStartedAt={conversionStartedAt}
      conversionError={conversionError}
      showConversionPicker={showConversionPicker}
      showConversionModal={showConversionModal}
      fileRevealLabel={fileRevealLabel}
      onConversionDraftChange={updateConversionDraft}
      onConversionDraftReset={() =>
        setConversionDraft({
          modelRef: "", path: "", hfRepo: "", outputPath: "",
          quantize: true, qBits: 4, qGroupSize: 64, dtype: "float16",
        })
      }
      onConvertModel={() => void handleConvertModel()}
      onPickConversionOutputDir={() => void handlePickConversionOutputDir()}
      onShowConversionPickerChange={setShowConversionPicker}
      onShowConversionModalChange={setShowConversionModal}
      onPrepareLibraryConversion={prepareLibraryConversion}
      onRevealPath={(path) => void handleRevealPath(path)}
    />
  );
}
