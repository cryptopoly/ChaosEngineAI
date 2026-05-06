import type { AppSettings } from "../types";

export interface ChatModelOption {
  key: string;
  label: string;
  detail: string;
  group: string;
  model: string;
  modelRef: string;
  canonicalRepo?: string | null;
  source: string;
  path?: string | null;
  backend: string;
  paramsB?: number;
  sizeGb?: number;
  contextWindow?: string;
  format?: string;
  quantization?: string;
  maxContext?: number | null;
}

export interface DataDirRestartPrompt {
  migration: {
    copied: string[];
    skipped: string[];
    from: string;
    to: string;
  } | null;
}

export type SettingsDraft = {
  modelDirectories: AppSettings["modelDirectories"];
  preferredServerPort: number;
  allowRemoteConnections: boolean;
  requireApiAuth: boolean;
  autoStartServer: boolean;
  remoteProviders: AppSettings["remoteProviders"];
  huggingFaceToken: string;
  hasHuggingFaceToken: boolean;
  huggingFaceTokenMasked: string;
  dataDirectory: string;
  imageOutputsDirectory: string;
  videoOutputsDirectory: string;
};
