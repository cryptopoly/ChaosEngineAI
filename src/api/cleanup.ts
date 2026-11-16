/**
 * Clean-up API client (`/api/cleanup`): what the backend can list for deleting, and the delete call.
 * The delete call sends ids from that list only; the backend never takes a path from a request.
 */

import { fetchJson, postJson } from "./index";

export interface CleanupProject {
  id: string;
  name: string;
  shots: number;
  bytes: number;
  /** Seconds since the epoch (the project file's last write). */
  lastEdited: number;
}

export interface CleanupThread {
  id: string;
  title: string;
  messages: number;
  bytes: number;
  /** Seconds since the epoch; null when the thread has no usable date. */
  lastUsed: number | null;
}

export interface CleanupOutput {
  id: string;
  kind: "image" | "video";
  name: string;
  bytes: number;
  /** ISO date text as the gallery saved it ("" when unknown). */
  createdAt: string;
}

export interface CleanupItems {
  projects: CleanupProject[];
  threads: CleanupThread[];
  outputs: CleanupOutput[];
}

export interface CleanupDeleteBody {
  projects: string[];
  threads: string[];
  outputs: string[];
}

export interface CleanupResultRow {
  kind: "project" | "thread" | "output";
  id: string;
  name: string | null;
  ok: boolean;
  bytes: number;
  reason: string | null;
}

export interface CleanupDeleteResult {
  results: CleanupResultRow[];
  deleted: number;
  refused: number;
  freedBytes: number;
}

export async function getCleanupItems(): Promise<CleanupItems> {
  return await fetchJson<CleanupItems>("/api/cleanup/items", 60000);
}

export async function deleteCleanupItems(body: CleanupDeleteBody): Promise<CleanupDeleteResult> {
  return await postJson<CleanupDeleteResult>("/api/cleanup/delete", body, 600000);
}
