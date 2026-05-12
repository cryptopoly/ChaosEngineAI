export type TabId =
  | "dashboard"
  | "online-models"
  | "my-models"
  | "image-discover"
  | "image-models"
  | "image-studio"
  | "image-gallery"
  | "video-models"
  | "video-discover"
  | "video-studio"
  | "video-gallery"
  | "conversion"
  | "chat"
  | "chat-compare"
  | "html-challenge"
  | "server"
  | "benchmarks"
  | "benchmark-history"
  | "finetuning"
  | "prompt-library"
  | "plugins"
  | "logs"
  | "settings";

export type SidebarGroupId =
  | "chat"
  | "models"
  | "images"
  | "video"
  | "benchmarks"
  | "tools";

export type SidebarMode = "collapsible" | "tabs";

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
  battery?: {
    percent: number;
    powerSource: "AC" | "Battery";
    charging: boolean;
  } | null;
  uptimeMinutes: number;
}

export interface Recommendation {
  title: string;
  detail: string;
  targetModel: string;
  cacheLabel: string;
  headroomPercent: number;
  /** Optional i18n key under `dashboard` namespace for the title. */
  titleKey?: string;
  /** Optional i18n key under `dashboard` namespace for the detail. */
  detailKey?: string;
  /** ICU MessageFormat variables for the keyed strings. */
  payload?: Record<string, unknown>;
}
