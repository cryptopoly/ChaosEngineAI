/**
 * Chat-domain API endpoints.
 *
 * Sessions CRUD, message variants (Phase 2.5), Delve critique
 * (Phase 3.6), thread fork (Phase 2.4), generate + cancel +
 * generateChatStream (with phase / panic / thermal / logprobs
 * callbacks), session document upload / list / delete.
 *
 * Re-exported from ``./index`` so existing
 * ``import { generateChatStream } from "../api"`` paths keep working.
 *
 * Extracted from ``api.ts`` as part of the v0.8.0 refactor.
 */

import { apiFetch, fetchJson, patchJson, postJson, readErrorDetail } from "./index";
import type {
  BusyAdvice,
  ChatSession,
  CreateSessionResponse,
  GeneratePayload,
  GenerateResponse,
  StudioJob,
  StudioPlan,
  ToolCallInfo,
  UpdateSessionPayload,
} from "../types";

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

export async function createSession(title?: string): Promise<ChatSession> {
  const result = await postJson<CreateSessionResponse>("/api/chat/sessions", { title });
  return result.session;
}

/**
 * Phase 2.5: generate a sibling variant for an assistant message
 * using a different (currently-loaded) model. Returns the updated
 * session payload with `messages[messageIndex].variants` populated.
 */
export async function addMessageVariant(
  sessionId: string,
  payload: {
    messageIndex: number;
    modelRef: string;
    modelName: string;
    canonicalRepo?: string | null;
    source?: string;
    path?: string;
    backend?: string;
    maxTokens?: number;
    temperature?: number;
  },
): Promise<ChatSession> {
  const result = await postJson<CreateSessionResponse>(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/variants`,
    payload,
    300000,
  );
  return result.session;
}

/**
 * Phase 3.6: ask the loaded model to re-read an assistant message
 * with a critic's framing and produce a Critique / Revised answer
 * pair. Result attaches as a "Delve critique" variant on the
 * message so the frontend's existing variant card surfaces it.
 */
export async function delveMessage(
  sessionId: string,
  messageIndex: number,
): Promise<ChatSession> {
  const result = await postJson<CreateSessionResponse>(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/delve/${messageIndex}`,
    {},
    300000,
  );
  return result.session;
}

/**
 * Phase 2.4: fork an existing thread at a specific message index.
 * Returns the new session, which the caller swaps active to so the
 * user can continue divergently. Parent linkage is preserved on
 * `parentSessionId` + `forkedAtMessageIndex`.
 */
export async function forkChatSession(
  sourceSessionId: string,
  forkAtMessageIndex: number,
  title?: string,
): Promise<ChatSession> {
  const result = await postJson<CreateSessionResponse>(
    `/api/chat/sessions/${encodeURIComponent(sourceSessionId)}/fork`,
    { forkAtMessageIndex, title },
  );
  return result.session;
}

export async function updateSession(sessionId: string, payload: UpdateSessionPayload): Promise<ChatSession> {
  const result = await patchJson<CreateSessionResponse>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, payload);
  return result.session;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await apiFetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `Delete failed with status ${response.status}`));
  }
}

// ---------------------------------------------------------------------------
// Generate / cancel / streaming
// ---------------------------------------------------------------------------

export async function generateChat(payload: GeneratePayload): Promise<GenerateResponse> {
  return await postJson<GenerateResponse>("/api/chat/generate", payload, 300000);
}

export type ChatStreamPhase = "prompt_eval" | "generating";

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onReasoning?: (reasoning: string) => void;
  onReasoningDone?: () => void;
  onCancelled?: () => void;
  /**
   * Phase transition signal (Phase 2.0). Backend emits `prompt_eval`
   * immediately when generation begins, then `generating` (with a
   * `ttftSeconds` measurement) the moment the model produces its first
   * token or reasoning fragment. Use this to render an explicit
   * "Processing prompt..." indicator instead of a blank flashing cursor.
   */
  onPhase?: (phase: ChatStreamPhase, ttftSeconds?: number) => void;
  /**
   * Phase 2.0.5-G: mid-stream panic signal. Backend emits at most once
   * per turn when memory crosses critical floors (free < 0.5 GB OR
   * pressure > 96%). Stream continues; user decides whether to cancel.
   */
  onPanic?: (signal: { message: string; availableGb?: number; pressurePercent?: number }) => void;
  /**
   * Phase 2.0.5-I: mid-stream thermal warning. Backend emits when host
   * is actively thermally throttling. Stream continues.
   */
  onThermalWarning?: (signal: { state: "moderate" | "critical"; message: string }) => void;
  /**
   * Phase 3.3: per-token logprob batches. The backend forwards
   * llama-server's `logprobs.content` shape verbatim — each entry has
   * the chosen token + top-k alternatives. Only fires when the request
   * had `logprobs: N` set.
   */
  onTokenLogprobs?: (entries: Array<{
    token: string | null;
    logprob: number | null;
    alternatives: Array<{ token: string | null; logprob: number | null }>;
  }>) => void;
  /** The machine is busy or tight on memory: emitted once, before the first token, only when not "ok". */
  onAdvice?: (advice: BusyAdvice) => void;
  /** A tool finished (streamed as it happens, so studio job cards show before the turn ends). */
  onToolCallResult?: (toolCall: ToolCallInfo) => void;
  /** D-60: the model asked for renders; the plan card waits for the user's click. */
  onStudioPlan?: (plan: StudioPlan) => void;
  onStudioPlanResolved?: (plan: StudioPlan) => void;
  onDone: (response: GenerateResponse) => void;
  onError: (error: string) => void;
}

/** Busy-machine advice for loading a model of this size (or starting a Studio job); null when unreadable. */
export async function fetchBusyAdvice(footprintGb?: number | null): Promise<BusyAdvice | null> {
  const query = footprintGb != null && footprintGb > 0 ? `?footprintGb=${encodeURIComponent(String(footprintGb))}` : "";
  try {
    const result = await fetchJson<{ advice: BusyAdvice | null }>(`/api/busy-advice${query}`, 5000);
    return result.advice ?? null;
  } catch {
    return null;
  }
}

/**
 * Ask the backend to cancel an in-flight chat generation. The streaming loop
 * checks this flag between events and stops within ~one tick, persisting
 * whatever output has accumulated. Safe to call when no generation is active.
 */
export async function cancelChatGeneration(sessionId: string): Promise<{ sessionId: string; cancelled: boolean; wasActive: boolean }> {
  return await postJson<{ sessionId: string; cancelled: boolean; wasActive: boolean }>(
    `/api/chat/generate/${encodeURIComponent(sessionId)}/cancel`,
    {},
    10000,
  );
}

export async function generateChatStream(
  payload: GeneratePayload,
  callbacks: StreamCallbacks,
  abortSignal?: AbortController,
): Promise<void> {
  const controller = abortSignal ?? new AbortController();
  let timer = setTimeout(() => controller.abort(), 300000);
  // With studio tools a turn can wait on the plan card and on long jobs, so the 5 minutes count from the
  // last thing the backend sent (it sends a keepalive while a plan waits), not from the start.
  const bumpTimer = () => {
    if (!payload.studioTools) return;
    clearTimeout(timer);
    timer = setTimeout(() => controller.abort(), 300000);
  };

  try {
    const response = await apiFetch("/api/chat/generate/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const errorBody = await response.json();
        if (errorBody?.detail) {
          detail = typeof errorBody.detail === "string"
            ? errorBody.detail
            : JSON.stringify(errorBody.detail);
        }
      } catch { /* ignore */ }
      callbacks.onError(detail);
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      callbacks.onError("Streaming not supported");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bumpTimer();

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);
          if (event.error) {
            const errDetail = typeof event.error === "string"
              ? event.error
              : event.error?.detail ?? event.error?.message ?? JSON.stringify(event.error);
            callbacks.onError(errDetail);
            return;
          }
          if (event.token) {
            callbacks.onToken(event.token);
          }
          if (event.reasoning) {
            callbacks.onReasoning?.(event.reasoning);
          }
          if (event.reasoningDone) {
            callbacks.onReasoningDone?.();
          }
          if (event.advice && typeof event.advice === "object" && typeof event.advice.message === "string") {
            callbacks.onAdvice?.(event.advice as BusyAdvice);
          }
          if (event.cancelled) {
            callbacks.onCancelled?.();
          }
          if (event.toolCallResult && typeof event.toolCallResult === "object") {
            const tc = event.toolCallResult;
            callbacks.onToolCallResult?.({
              id: String(tc.id ?? ""),
              name: String(tc.name ?? "tool"),
              arguments: tc.arguments && typeof tc.arguments === "object" ? tc.arguments : {},
              result: String(tc.result ?? ""),
              elapsed: typeof tc.elapsed === "number" ? tc.elapsed : 0,
              renderAs: tc.renderAs ?? null,
              data: tc.data ?? null,
            });
          }
          if (event.studioPlan && Array.isArray(event.studioPlan.items)) {
            callbacks.onStudioPlan?.(event.studioPlan as StudioPlan);
          }
          if (event.studioPlanResolved && Array.isArray(event.studioPlanResolved.items)) {
            callbacks.onStudioPlanResolved?.(event.studioPlanResolved as StudioPlan);
          }
          if (event.phase === "prompt_eval" || event.phase === "generating") {
            const ttft = typeof event.ttftSeconds === "number" ? event.ttftSeconds : undefined;
            callbacks.onPhase?.(event.phase, ttft);
          }
          if (event.panic === true && typeof event.message === "string") {
            callbacks.onPanic?.({
              message: event.message,
              availableGb: typeof event.availableGb === "number" ? event.availableGb : undefined,
              pressurePercent: typeof event.pressurePercent === "number" ? event.pressurePercent : undefined,
            });
          }
          if (event.thermalWarning === true && typeof event.message === "string"
              && (event.state === "moderate" || event.state === "critical")) {
            callbacks.onThermalWarning?.({
              state: event.state,
              message: event.message,
            });
          }
          if (Array.isArray(event.tokenLogprobs) && event.tokenLogprobs.length > 0) {
            callbacks.onTokenLogprobs?.(event.tokenLogprobs);
          }
          if (event.done) {
            callbacks.onDone({
              session: event.session,
              assistant: event.assistant,
              runtime: event.runtime,
            });
          }
        } catch {
          // Malformed JSON chunk, skip
        }
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      callbacks.onError("Streaming timed out");
    } else {
      callbacks.onError(err instanceof Error ? err.message : "Unknown streaming error");
    }
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// D-60: studio tools in Chat (plan decisions, job cards)
// ---------------------------------------------------------------------------

export interface StudioPlanDecisionBody {
  decisions: Record<string, "approve" | "deny">;
  edit?: boolean;
  alwaysAllow?: boolean;
}

export async function decideStudioPlan(planId: string, body: StudioPlanDecisionBody): Promise<void> {
  await postJson(`/api/chat/studio/plans/${encodeURIComponent(planId)}/decision`, body, 10000);
}

export async function fetchStudioJob(jobId: string): Promise<StudioJob> {
  const result = await fetchJson<{ job: StudioJob }>(`/api/chat/studio/jobs/${encodeURIComponent(jobId)}`, 10000);
  return result.job;
}

export async function cancelStudioJob(jobId: string): Promise<StudioJob> {
  const result = await postJson<{ job: StudioJob }>(`/api/chat/studio/jobs/${encodeURIComponent(jobId)}/cancel`, {}, 10000);
  return result.job;
}

/** The file a finished chat job made, as a Blob (the route needs the bearer token). */
export async function fetchStudioJobMedia(jobId: string, track = 0): Promise<Blob> {
  const response = await apiFetch(`/api/chat/studio/jobs/${encodeURIComponent(jobId)}/media?track=${track}`);
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `Loading the file failed with status ${response.status}`));
  }
  return await response.blob();
}

// ---------------------------------------------------------------------------
// Session documents (RAG attachments)
// ---------------------------------------------------------------------------

export interface SessionDocument {
  id: string;
  filename: string;
  originalName: string;
  sizeBytes: number;
  chunkCount: number;
  uploadedAt: string;
}

export async function uploadSessionDocument(sessionId: string, file: File): Promise<SessionDocument> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await apiFetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}/documents`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `Upload failed with status ${response.status}`));
  }
  const result = await response.json();
  return result.document;
}

export async function listSessionDocuments(sessionId: string): Promise<SessionDocument[]> {
  const result = await fetchJson<{ documents: SessionDocument[] }>(`/api/chat/sessions/${encodeURIComponent(sessionId)}/documents`);
  return result.documents;
}

export async function deleteSessionDocument(sessionId: string, docId: string): Promise<void> {
  const response = await apiFetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `Delete failed with status ${response.status}`));
  }
}
