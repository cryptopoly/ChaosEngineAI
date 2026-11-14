/**
 * Territory statement (D-76): which territory-limited models exist, whether the user confirmed, confirm and
 * withdraw. Confirm and withdraw are accepted only from this window; the backend refuses agents.
 */

import { fetchJson, postJson } from "./index";

export interface TerritoryModel {
  key: string;
  name: string;
  kinds: string[];
  excludedTerritories: string[];
  licenceName: string;
  licenceUrl: string;
  summaryLines: string[];
  confirmSentence: string;
  note: string;
  confirmed: boolean;
  confirmedAt: string | null;
}

export interface LicenceGateStatus {
  models: TerritoryModel[];
}

export async function getLicenceGate(): Promise<LicenceGateStatus> {
  return await fetchJson<LicenceGateStatus>("/api/licence-gate");
}

export async function confirmLicenceGate(key: string, sentence: string): Promise<LicenceGateStatus> {
  return await postJson<LicenceGateStatus>("/api/licence-gate/confirm", { key, confirmed: true, sentence });
}

export async function withdrawLicenceGate(key: string): Promise<LicenceGateStatus> {
  return await postJson<LicenceGateStatus>("/api/licence-gate/withdraw", { key });
}
