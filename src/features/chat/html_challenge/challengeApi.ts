/**
 * Thin async wrappers around the ``/api/chat/html-challenges`` endpoints.
 *
 * Each helper returns parsed data (or null / a default on failure) — no
 * React state, no UI side-effects. The tab decides how to react to each
 * outcome.
 */

import { apiFetch } from "../../../api";
import type { CompareTarget } from "../CompareView";
import type {
  HtmlChallengeManifest,
  HtmlValidation,
} from "../htmlChallengeHelpers";
import { validationMessage } from "../htmlChallengeHelpers";

export async function readResponseDetail(response: Response, fallback: string) {
  try {
    const payload = await response.json();
    if (payload?.detail) return String(payload.detail);
  } catch {
    try {
      const text = await response.text();
      if (text.trim()) return text.trim();
    } catch {
      // Ignore unreadable error bodies.
    }
  }
  return fallback;
}

export async function fetchChallengeList(): Promise<HtmlChallengeManifest[]> {
  try {
    const response = await apiFetch("/api/chat/html-challenges");
    if (!response.ok) return [];
    const payload = await response.json() as { challenges?: HtmlChallengeManifest[] };
    return payload.challenges ?? [];
  } catch {
    return [];
  }
}

export async function fetchChallenge(challengeId: string): Promise<HtmlChallengeManifest | null> {
  const response = await apiFetch(`/api/chat/html-challenges/${encodeURIComponent(challengeId)}`);
  if (!response.ok) return null;
  const payload = await response.json() as { challenge?: HtmlChallengeManifest };
  return payload.challenge ?? null;
}

export interface ChallengeFileResult {
  status: "ok" | "deleted" | "error";
  html?: string;
  error?: string;
}

export async function fetchChallengeFile(challengeId: string, slotId: CompareTarget): Promise<ChallengeFileResult> {
  const response = await apiFetch(
    `/api/chat/html-challenges/${encodeURIComponent(challengeId)}/files/${encodeURIComponent(slotId)}`,
  );
  if (response.ok) {
    return { status: "ok", html: await response.text() };
  }
  if (response.status === 404 || response.status === 410) {
    return { status: "deleted" };
  }
  return { status: "error", error: await readResponseDetail(response, "Could not load saved HTML.") };
}

export async function deleteChallenge(challengeId: string): Promise<{ ok: boolean; error?: string }> {
  const response = await apiFetch(`/api/chat/html-challenges/${encodeURIComponent(challengeId)}`, {
    method: "DELETE",
  });
  if (response.ok) return { ok: true };
  return { ok: false, error: await readResponseDetail(response, "Delete challenge failed.") };
}

export async function patchSlotValidation(
  challengeId: string,
  target: CompareTarget,
  validation: HtmlValidation,
): Promise<HtmlChallengeManifest | null> {
  try {
    const response = await apiFetch(
      `/api/chat/html-challenges/${encodeURIComponent(challengeId)}/slots/${encodeURIComponent(target)}/validation`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          status: validation.status,
          message: validationMessage(validation),
          issues: validation.issues ?? [],
          source: validation.source ?? "runtime",
        }),
      },
    );
    if (!response.ok) return null;
    const payload = await response.json() as { challenge?: HtmlChallengeManifest };
    return payload.challenge ?? null;
  } catch {
    // Runtime preview validation is best-effort; the local card already shows it.
    return null;
  }
}
