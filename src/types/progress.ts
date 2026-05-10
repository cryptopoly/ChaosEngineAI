export interface GenerationProgressSnapshot {
  kind: "image" | "video";
  active: boolean;
  phase: "idle" | "loading" | "encoding" | "diffusing" | "decoding" | "saving";
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
}
