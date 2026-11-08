/**
 * D-62: skills API client (`/api/skills`): list, import from a folder or .zip path, switch on or off,
 * delete. The backend validates the path; errors keep its plain `detail` sentence as the message.
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

export interface Skill {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  importedAt: string;
  source: string;
  files: SkillFile[];
  skipped: SkillSkippedFile[];
}

export async function listSkills(): Promise<Skill[]> {
  const result = await fetchJson<{ skills: Skill[] }>("/api/skills", 10000);
  return result.skills;
}

export async function importSkill(path: string): Promise<Skill> {
  const result = await postJson<{ skill: Skill }>("/api/skills/import", { path }, 60000);
  return result.skill;
}

export async function setSkillEnabled(id: string, enabled: boolean): Promise<Skill> {
  const result = await postJson<{ skill: Skill }>(`/api/skills/${encodeURIComponent(id)}/enabled`, { enabled }, 10000);
  return result.skill;
}

export async function deleteSkill(id: string): Promise<void> {
  await deleteJson(`/api/skills/${encodeURIComponent(id)}`, undefined, 10000);
}
