import type { Dispatch, SetStateAction } from "react";
import type {
  LibraryItem,
  LoadModelActionResult,
  ModelVariant,
  TabId,
  TauriBackendInfo,
  VoiceRuntime,
  WorkspaceData,
} from "../types";
import type { ChatModelOption } from "../types/chat";
import type { TabRequest } from "../constants";
import type { ModelAvailabilityFilter, ModelsArea } from "../components/models/modelEntry";
import type { TensorfoldLaunchControls } from "../components/tensorfoldSupport";
import type { OpenRequest as VoiceOpenRequest } from "../features/voice/VoiceStudioTab";
import type {
  useBenchmarks,
  useChat,
  useCudaTorchInstall,
  useDetailsWindowResize,
  useFileActions,
  useImageState,
  useModels,
  useMusicState,
  useSettings,
  useSidebarPrefs,
  useUiScale,
  useVideoState,
  useWorkspace,
} from "../hooks";
import type { UseMtplxInstallReturn } from "../hooks/useMtplxInstall";

export type LoadModelPayload = {
  modelRef: string;
  modelName?: string;
  canonicalRepo?: string | null;
  source?: string;
  backend?: string;
  path?: string;
  nextTab?: TabId;
  busyLabel?: string;
  cacheBits?: number;
  fp16Layers?: number;
  fusedAttention?: boolean;
  cacheStrategy?: string;
  fitModelInMemory?: boolean;
  contextTokens?: number;
  speculativeDecoding?: boolean;
  treeBudget?: number;
  kvBudget?: number;
};

type FileActions = ReturnType<typeof useFileActions>;
type CudaTorch = ReturnType<typeof useCudaTorchInstall>;
type WorkspaceHook = ReturnType<typeof useWorkspace>;

/**
 * Everything the tab views need from `App`. `App` keeps every hook call and
 * all state; it builds this object once per render and hands it to the views
 * under `src/app/`, which render exactly what `App` used to render inline.
 */
export interface AppTabContext {
  // Core workspace state
  workspace: WorkspaceData;
  backendOnline: boolean;
  tauriBackend: TauriBackendInfo | null;
  busy: boolean;
  busyAction: string | null;
  setError: (msg: string | null) => void;
  refreshWorkspace: WorkspaceHook["refreshWorkspace"];
  setActiveTab: (request: TabRequest) => void;

  // Hook result objects
  imgState: ReturnType<typeof useImageState>;
  videoState: ReturnType<typeof useVideoState>;
  musicState: ReturnType<typeof useMusicState>;
  models: ReturnType<typeof useModels>;
  chat: ReturnType<typeof useChat>;
  benchmarks: ReturnType<typeof useBenchmarks>;
  settings: ReturnType<typeof useSettings>;
  sidebarPrefs: ReturnType<typeof useSidebarPrefs>;
  uiScalePrefs: ReturnType<typeof useUiScale>;

  // Installs
  mtplxJob: UseMtplxInstallReturn["mtplxJob"];
  installingMtplx: boolean;
  handleInstallMtplx: UseMtplxInstallReturn["handleInstallMtplx"];
  tensorfoldControls: TensorfoldLaunchControls;
  installingCudaTorch: CudaTorch["installingCudaTorch"];
  cudaTorchRawResult: CudaTorch["cudaTorchRawResult"];
  handleInstallCudaTorch: CudaTorch["handleInstallCudaTorch"];

  // Derived model values
  chatLibrary: LibraryItem[];
  libraryChatOptions: ChatModelOption[];
  threadModelOptions: ChatModelOption[];
  activeThreadOption: ChatModelOption | null;
  selectedServerOption: ChatModelOption | null;
  benchmarkOption: ChatModelOption | null;
  convertibleLibrary: LibraryItem[];
  conversionSource: LibraryItem | null;
  conversionVariant: ModelVariant | null;
  modelsAvailability: Record<ModelsArea, ModelAvailabilityFilter>;
  setAreaAvailability: (area: ModelsArea, value: ModelAvailabilityFilter) => void;
  nativeBackends: WorkspaceData["runtime"]["nativeBackends"];
  isAppleSilicon: boolean;
  enabledDirectoryCount: number;
  fileRevealLabel: string;
  conversionReady: boolean;
  launchCacheLabel: string;

  // Handlers
  handleRevealPath: FileActions["handleRevealPath"];
  handleOpenFilePath: FileActions["handleOpenFilePath"];
  handleOpenExternalUrl: FileActions["handleOpenExternalUrl"];
  handleLoadModel: (payload: LoadModelPayload) => Promise<LoadModelActionResult>;
  handleUnloadModel: (ref?: string) => Promise<unknown>;
  handleUnloadWarmModel: (ref: string) => Promise<void>;
  handleDeleteModel: (item: LibraryItem) => Promise<unknown>;
  prepareCatalogConversion: (model: ModelVariant) => void;
  prepareLibraryConversion: (item: LibraryItem, resolvedPath?: string) => void;
  handleConvertModel: () => Promise<unknown>;
  handlePickConversionOutputDir: () => Promise<unknown>;
  openModelSelector: (action: "chat" | "server" | "thread", preselectedKey?: string) => void;
  handleDetailsToggle: ReturnType<typeof useDetailsWindowResize>["handleDetailsToggle"];
  copyText: (text: string) => void;

  // Conversion state
  conversionStartedAt: number | null;
  conversionError: string | null;
  showConversionPicker: boolean;
  setShowConversionPicker: Dispatch<SetStateAction<boolean>>;
  showConversionModal: boolean;
  setShowConversionModal: Dispatch<SetStateAction<boolean>>;

  // Server / logs state
  apiToken: string | null;
  showRemoteTest: boolean;
  setShowRemoteTest: Dispatch<SetStateAction<boolean>>;
  testModelId: string | null;
  setTestModelId: Dispatch<SetStateAction<string | null>>;
  logQuery: string;
  setLogQuery: Dispatch<SetStateAction<string>>;

  // Voice state
  voiceRuntime: VoiceRuntime | null;
  voiceOpenRequest: VoiceOpenRequest | null;
  setVoiceOpenRequest: Dispatch<SetStateAction<VoiceOpenRequest | null>>;
  refreshVoiceRuntime: () => void;
}
