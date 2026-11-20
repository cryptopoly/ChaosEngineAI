// The old Discover / My Models ids ("online-models", "my-models",
// "image-discover", "video-discover") are no longer tabs: one Models page
// per area replaced them, and the Storyboard's Projects, Board and Mixer ids
// ("story-projects", "story-board", "story-mixer") became one "storyboard"
// entry (D-65). The Prompts page moved into Chat and Plugins was removed
// (D-87): "prompt-library" opens Chat and "plugins" the Dashboard.
// ``constants/tabRedirects.ts`` maps them.
export type TabId =
  | "dashboard"
  | "chat-models"
  | "image-models"
  | "image-studio"
  | "image-gallery"
  | "video-models"
  | "video-studio"
  | "video-gallery"
  | "voice-studio"
  | "voice-models"
  | "voice-gallery"
  | "music-models"
  | "music-studio"
  | "music-library"
  | "storyboard"
  | "conversion"
  | "chat"
  | "chat-compare"
  | "html-challenge"
  | "server"
  | "benchmarks"
  | "benchmark-history"
  | "finetuning"
  | "settings";

export type SidebarGroupId =
  | "chat"
  | "models"
  | "images"
  | "video"
  | "voice"
  | "music"
  | "benchmarks"
  | "tools";

export type SidebarMode = "collapsible" | "tabs";

/** An image, video or music run in flight, read from the shared progress trackers. */
export interface ActiveGeneration {
  kind: "image" | "video" | "music";
  runLabel: string | null;
  phase: string;
  message: string;
  step: number;
  totalSteps: number;
  elapsedSeconds: number;
}

export interface SystemStats {
  platform: string;
  arch: string;
  hardwareSummary: string;
  backendLabel: string;
  appVersion: string;
  availableCacheStrategies: Array<{
    id: string;
    name: string;
    available: boolean;
    bitRange: number[] | null;
    defaultBits: number | null;
    supportsFp16Layers: boolean;
    availabilityBadge?: string | null;
    availabilityTone?: string | null;
    availabilityReason?: string | null;
    requiredLlamaBinary?: string | null;
    appliesTo?: string[];
  }>;
  llamaServerTurboPath?: string | null;
  mlxAvailable: boolean;
  mlxLmAvailable: boolean;
  totalMemoryGb: number;
  /** Discrete GPU VRAM in GB (CUDA cards on Windows / Linux). Null on
   * Apple Silicon (unified memory is already in totalMemoryGb), and on
   * hosts with no detected discrete GPU. The chat cache-fit warning uses
   * this to surface "60 GB cache > 24 GB GPU VRAM" instead of comparing
   * against system RAM only -- llama.cpp places the KV cache on GPU when
   * full-offload is on, so the GPU is the binding constraint there. */
  gpuVramTotalGb?: number | null;
  /** ``false`` on a graphics card with its own memory, ``true`` on shared memory (Apple Silicon, some iGPUs). */
  gpuUnifiedMemory?: boolean | null;
  availableMemoryGb: number;
  usedMemoryGb: number;
  swapUsedGb: number;
  cpuUtilizationPercent: number;
  gpuUtilizationPercent: number | null;
  spareHeadroomGb: number;
  dflash?: {
    available: boolean;
    mlxAvailable: boolean;
    vllmAvailable: boolean;
    ddtreeAvailable?: boolean;
    supportedModels: string[];
  };
  mtplx?: {
    available: boolean;
    /** A venv from before MTPLX 2.12 exists: the Install button rebuilds it. */
    outdated?: boolean;
    supportedModels: string[];
  };
  // TensorFold engine (exact speculative decoding, Apple Silicon). The repo
  // lists come from the backend registry so the launch settings and the
  // engine agree on which checkpoints are served and which are exclusive.
  tensorfold?: {
    available: boolean;
    version?: string | null;
    /** Optional extras installed in the TensorFold venv ("vision", "grammar"). */
    extras?: string[];
    supportedModels: string[];
    exclusiveModels: string[];
    /** Checkpoints whose images TensorFold can read once the vision extra is installed. */
    visionModels?: string[];
    minMemoryGb?: Record<string, number>;
    /** "native" on CUDA hosts (TensorFold 1.x binary); absent for the Mac Python engine. */
    engine?: "native";
  };
  runningLlmProcesses: Array<{
    pid: number;
    name: string;
    owner?: string;
    modelName?: string | null;
    modelStatus?: "active" | "warm" | null;
    kind?: "mlx_worker" | "llama_server" | "backend" | "other";
    memoryGb: number;
    cpuPercent: number;
  }>;
  compressedMemoryGb?: number;
  memoryPressurePercent?: number;
  swapTotalGb?: number;
  diskFreeGb?: number;
  diskUsedGb?: number;
  diskTotalGb?: number;
  diskPath?: string;
  /** Runs in flight in a subprocess or worker engine; the chat runtime reports no loaded model for them. */
  activeGenerations?: ActiveGeneration[];
  battery?: {
    percent: number;
    powerSource: "AC" | "Battery";
    charging: boolean;
  } | null;
  uptimeMinutes: number;
}

/** A catalogue chat row recommended for this machine (``helpers/chat_picks.py``), judged by the same
 * ``fit`` the Models page shows. */
export interface ChatPick {
  variantId: string;
  familyId: string;
  familyName?: string | null;
  name: string;
  repo: string;
  sizeGb: number | null;
  paramsB: number | null;
  format?: string | null;
  quantization?: string | null;
  releaseDate?: string | null;
  availableLocally: boolean;
  fit: import("./models").ModelFit;
  /** Expected decode speed: this machine's own measured median, else the bandwidth estimate. */
  tokS: number | null;
  tokSBasis: "measured" | "estimated" | null;
  /** "NVIDIA GPU, 24 GB VRAM", "Apple Silicon, 64 GB unified memory"... */
  hardwareLabel?: string | null;
}

export interface Recommendation {
  title: string;
  detail: string;
  /** The best-fit row's name when there is one, else a size class ("14B"). */
  targetModel: string;
  cacheLabel: string;
  /** The best-fit row's own ``fit.headroomPct``; ``null`` when no catalogue row fits comfortably. */
  headroomPercent: number | null;
  /** The catalogue's best-fit chat row for this machine (D-105 B); ``null`` when none fits comfortably. */
  bestFit?: ChatPick | null;
  /** Up to five recommended rows, best first (``bestFit`` is the first). */
  chatPicks?: ChatPick[];
  /** Optional i18n key under `dashboard` namespace for the title. */
  titleKey?: string;
  /** Optional i18n key under `dashboard` namespace for the detail. */
  detailKey?: string;
  /** ICU MessageFormat variables for the keyed strings. */
  payload?: Record<string, unknown>;
}
