export interface GenerationProgressSnapshot {
  kind: "image" | "video" | "music";
  active: boolean;
  phase: "idle" | "loading" | "encoding" | "diffusing" | "decoding" | "saving" | "load" | "encode" | "compose" | "decode" | "save";
  message: string;
  step: number;
  totalSteps: number;
  startedAt: number;
  updatedAt: number;
  elapsedSeconds: number;
  runLabel: string | null;
  // FU-018 part 2: live denoise thumbnail. Base64-encoded PNG the runtime
  // emits from inside callback_on_step_end after decoding the current
  // latent through the swapped TAESD/TAEHV preview VAE. ``null`` when
  // previewVae is off, when the swap didn't apply, or before the first
  // decoded step. Capped at 192 px on the long edge backend-side.
  thumbnail?: string | null;
  cancelRequested?: boolean;
  /** A notice code for the run in flight: the job runs on the CPU, or fell back to it after the GPU failed. */
  warning?: "cpu" | "cpu-fallback" | null;
  /** Shared job queue: set while this request waits behind another heavy job. */
  queue?: { waiting: boolean; position: number; agentAhead: boolean };
}
