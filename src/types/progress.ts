export interface GenerationProgressSnapshot {
  kind: "image" | "video" | "music" | "benchmark" | "conversion";
  active: boolean;
  phase:
    | "idle" | "loading" | "encoding" | "diffusing" | "decoding" | "saving" | "load" | "encode" | "compose" | "decode" | "save"
    // Benchmarks and conversion (text-model jobs).
    | "preparing" | "dataset" | "evaluating" | "generating" | "downloading" | "quantizing";
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
  /**
   * Benchmarks and conversion: measured numbers of the current phase besides ``step`` / ``totalSteps``
   * (``correct`` answers so far, ``bytesWritten``, ``bitsPerWeight``). ``expectedBytes`` is the one
   * estimate in here and is shown as one.
   */
  detail?: Record<string, number>;
  /** Shared job queue: set while this request waits behind another heavy job. */
  queue?: { waiting: boolean; position: number; agentAhead: boolean };
}
