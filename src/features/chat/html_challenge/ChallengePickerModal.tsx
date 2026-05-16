/**
 * Wraps the shared ``ModelLaunchModal`` for the HTML Challenge tab and
 * owns the picker draft state — the in-flight key, search, and launch
 * settings the user is editing before they confirm.
 *
 * The composition root opens / closes the picker via ``target`` +
 * ``autoRetry`` and reads back the chosen option through ``onConfirm``.
 */

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { ModelLaunchModal } from "../../../components/ModelLaunchModal";
import type { LaunchPreferences, StrategyInstallLog, SystemStats } from "../../../types";
import type { ChatModelOption } from "../../../types/chat";
import type { MtplxJobState } from "../../../api";
import { compareTargetLabels, type CompareTarget, cloneLaunchSettings, useLaunchPreview } from "../CompareView";

interface ChallengePickerModalProps {
  target: CompareTarget | null;
  initialKey: string;
  initialSettings: LaunchPreferences;
  textModelOptions: ChatModelOption[];
  availableMemoryGb: number;
  totalMemoryGb: number;
  gpuVramTotalGb?: number | null;
  availableCacheStrategies?: SystemStats["availableCacheStrategies"];
  dflashInfo?: SystemStats["dflash"];
  installingPackage: string | null;
  installLogs?: Record<string, StrategyInstallLog>;
  turboInstalled?: boolean;
  mtplxSystemInfo?: SystemStats["mtplx"];
  onInstallMtplx?: () => void;
  installingMtplx?: boolean;
  mtplxJob?: MtplxJobState | null;
  onConfirm: (selectedKey: string, settings: LaunchPreferences) => void;
  onClose: () => void;
  onInstallPackage: (strategyId: string) => void;
}

export function ChallengePickerModal({
  target,
  initialKey,
  initialSettings,
  textModelOptions,
  availableMemoryGb,
  totalMemoryGb,
  gpuVramTotalGb,
  availableCacheStrategies,
  dflashInfo,
  installingPackage,
  installLogs,
  turboInstalled,
  mtplxSystemInfo,
  onInstallMtplx,
  installingMtplx,
  mtplxJob,
  onConfirm,
  onClose,
  onInstallPackage,
}: ChallengePickerModalProps) {
  const { t } = useTranslation("chat");
  const [search, setSearch] = useState("");
  const [draftKey, setDraftKey] = useState(initialKey);
  const [draftSettings, setDraftSettings] = useState<LaunchPreferences>(() => cloneLaunchSettings(initialSettings));

  useEffect(() => {
    if (target) {
      setSearch("");
      setDraftKey(initialKey);
      setDraftSettings(cloneLaunchSettings(initialSettings));
    }
    // The picker is opened by the parent flipping `target` from null →
    // a target. Reset the draft only on that transition so typing into
    // the search box doesn't reset itself on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target]);

  const draftOption =
    textModelOptions.find((option) => option.key === draftKey)
    ?? (target ? textModelOptions[0] ?? null : null);
  const draftPreview = useLaunchPreview(draftOption, draftSettings);

  return (
    <ModelLaunchModal
      open={target != null}
      title={target
        ? t("htmlChallenge.picker.titleForSlot", { defaultValue: "Select {slot}", slot: compareTargetLabels[target] })
        : t("htmlChallenge.picker.title", { defaultValue: "Select Model" })}
      confirmLabel={target
        ? t("htmlChallenge.picker.confirmForSlot", { defaultValue: "Use for {slot}", slot: compareTargetLabels[target] })
        : t("htmlChallenge.picker.confirm", { defaultValue: "Use model" })}
      selectedKey={draftKey}
      collapseOnOpen={Boolean(draftKey)}
      search={search}
      options={textModelOptions}
      settings={draftSettings}
      preview={draftPreview}
      availableMemoryGb={availableMemoryGb}
      totalMemoryGb={totalMemoryGb}
      gpuVramTotalGb={gpuVramTotalGb}
      availableCacheStrategies={availableCacheStrategies}
      dflashInfo={dflashInfo}
      installingPackage={installingPackage}
      installLogs={installLogs}
      turboInstalled={turboInstalled}
      mtplxSystemInfo={mtplxSystemInfo}
      onInstallMtplx={onInstallMtplx}
      installingMtplx={installingMtplx}
      mtplxJob={mtplxJob}
      onSelectedKeyChange={setDraftKey}
      onSearchChange={setSearch}
      onSettingChange={(key, value) => {
        setDraftSettings((current) => ({ ...current, [key]: value }));
      }}
      onConfirm={(selectedKey) => {
        const newSettings = cloneLaunchSettings(draftSettings);
        onConfirm(selectedKey, newSettings);
        setSearch("");
      }}
      onClose={() => {
        setSearch("");
        onClose();
      }}
      onInstallPackage={onInstallPackage}
    />
  );
}
