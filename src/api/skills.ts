/**
 * D-62: skills API client (`/api/skills`): list, import from a folder or .zip path, switch on or off,
 * delete. The backend validates the path; errors keep its plain `detail` sentence as the message.
 * D-68: every skill carries a rule-based security scan; a skill rated high switches on only with
 * `acknowledgeFindings: true`.
 */

import { deleteJson, fetchJson, postJson } from "./index";

export interface SkillFile {
  path: string;
  bytes: number;
}

export interface SkillSkippedFile {
  path: string;
  reason: string;
}

export type SkillScanRating = "low" | "medium" | "high";

export interface SkillScanFinding {
  file: string;
  /** 0 when the finding is about the whole file (a bundled script, the size). */
  line: number;
  rule: string;
  severity: SkillScanRating;
  message: string;
  excerpt: string;
}

export interface SkillScan {
  rating: SkillScanRating;
  findings: SkillScanFinding[];
  truncated?: boolean;
  scannedAt: string;
  rulesVersion: number;
}

export interface Skill {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  importedAt: string;
  source: string;
  files: SkillFile[];
  skipped: SkillSkippedFile[];
  scan: SkillScan;
}

export async function listSkills(): Promise<Skill[]> {
  const result = await fetchJson<{ skills: Skill[] }>("/api/skills", 10000);
  return result.skills;
}

export async function importSkill(path: string): Promise<Skill> {
  const result = await postJson<{ skill: Skill }>("/api/skills/import", { path }, 60000);
  return result.skill;
}

export async function setSkillEnabled(id: string, enabled: boolean, acknowledgeFindings = false): Promise<Skill> {
  const body = acknowledgeFindings ? { enabled, acknowledgeFindings } : { enabled };
  const result = await postJson<{ skill: Skill }>(`/api/skills/${encodeURIComponent(id)}/enabled`, body, 10000);
  return result.skill;
}

export async function deleteSkill(id: string): Promise<void> {
  await deleteJson(`/api/skills/${encodeURIComponent(id)}`, undefined, 10000);
}
