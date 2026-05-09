export interface HubModel {
  id: string;
  repo: string;
  name: string;
  provider: string;
  link: string;
  format: string;
  tags: string[];
  downloads: number;
  likes: number;
  downloadsLabel: string;
  likesLabel: string;
  lastModified?: string | null;
  updatedLabel?: string | null;
  createdAt?: string | null;
  releaseLabel?: string | null;
  availableLocally: boolean;
  launchMode: string;
  backend: string;
}

export interface HubFile {
  path: string;
  sizeBytes: number;
  sizeGb: number;
  kind: "weight" | "vision_projector" | "config" | "tokenizer" | "readme" | "template" | "other";
}

export interface HubFileListResponse {
  repo: string;
  files: HubFile[];
  totalSizeBytes: number;
  totalSizeGb: number;
  license: string | null;
  tags: string[];
  pipelineTag: string | null;
  lastModified: string | null;
  warning?: string | null;
}
