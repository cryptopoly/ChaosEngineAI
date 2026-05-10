import type { LaunchPreferences, ModelDirectorySetting } from "./models";


export interface StrategyInstallLogStep {
  id: string;
  label: string;
  command: string;
  status: "running" | "success" | "failed";
  output: string;
}

export interface StrategyInstallLog {
  strategyId: string;
  label: string;
  status: "running" | "success" | "failed";
  startedAt: string;
  finishedAt?: string | null;
  steps: StrategyInstallLogStep[];
}

export interface RemoteProvider {
  id: string;
  label: string;
  apiBase: string;
  model: string;
  hasApiKey?: boolean;
  apiKeyMasked?: string;
  apiKey?: string;
}

export interface AppSettings {
  modelDirectories: ModelDirectorySetting[];
  preferredServerPort: number;
  allowRemoteConnections: boolean;
  // When false, the backend disables bearer-token enforcement so external
  // clients (OpenWebUI, curl, another desktop app) can hit /api and /v1
  // endpoints without a token. Default true.
  requireApiAuth: boolean;
  autoStartServer: boolean;
  launchPreferences: LaunchPreferences;
  remoteProviders?: RemoteProvider[];
  huggingFaceToken?: string | null;
  hasHuggingFaceToken?: boolean;
  dataDirectory?: string;
  // Empty string means "use the default under dataDirectory". A non-empty
  // value redirects new image / video artifacts to a custom folder (e.g. an
  // external SSD or a cloud-synced delivery folder).
  imageOutputsDirectory?: string;
  videoOutputsDirectory?: string;
  /**
   * Phase 3.3: when true, the chat composer adds `logprobs: 5` to
   * every send so llama-server returns top-k per-token confidence
   * info. Off by default — bandwidth + render cost is non-trivial.
   */
  advancedLogprobs?: boolean;
}

export interface SettingsUpdateResponse {
  settings: AppSettings;
  restartRequired?: boolean;
  migrationSummary?: {
    copied: string[];
    skipped: string[];
    from: string;
    to: string;
  };
}

export interface UpdateSettingsPayload {
  modelDirectories?: ModelDirectorySetting[];
  preferredServerPort?: number;
  allowRemoteConnections?: boolean;
  requireApiAuth?: boolean;
  autoStartServer?: boolean;
  launchPreferences?: LaunchPreferences;
  remoteProviders?: Array<{ id: string; label: string; apiBase: string; apiKey: string; model: string }>;
  huggingFaceToken?: string | null;
  dataDirectory?: string | null;
  imageOutputsDirectory?: string | null;
  videoOutputsDirectory?: string | null;
}
