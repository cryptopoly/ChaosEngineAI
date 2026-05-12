/**
 * The compact model card shown above each slot before / between runs.
 * Lists the selected model + sampler row (thinking, temperature, seed)
 * and the "Select" / "Change" + "Remove" affordances.
 *
 * Pure leaf — owns no state. The composition root passes in callbacks
 * for the real state mutations.
 */

import { useTranslation } from "react-i18next";
import { compareTargetLabels, type CompareTarget } from "../CompareView";
import {
  type ChallengeSlot,
  type HtmlChallengeManifestSlot,
  type HtmlChallengeReasoningEffort,
  type HtmlChallengeThinkingMode,
  randomChallengeSeed,
} from "../htmlChallengeHelpers";
import { sizeLabel } from "../../../utils";
import type { ChatModelOption } from "../../../types/chat";

interface ChallengeModelCardProps {
  slot: ChallengeSlot;
  option: ChatModelOption | null;
  manifestSlot?: HtmlChallengeManifestSlot;
  busy: boolean;
  completedChallenge: boolean;
  isLastSlot: boolean;
  canRemove: boolean;
  summary: string;
  onUpdateThinking: (
    slotId: CompareTarget,
    mode: HtmlChallengeThinkingMode,
    effort?: HtmlChallengeReasoningEffort,
  ) => void;
  onUpdateTemperature: (slotId: CompareTarget, value: number) => void;
  onUpdateSeed: (slotId: CompareTarget, value: number | null) => void;
  onRemoveLastSlot: () => void;
  onOpenPicker: (slotId: CompareTarget) => void;
}

export function ChallengeModelCard({
  slot,
  option,
  manifestSlot,
  busy,
  completedChallenge,
  isLastSlot,
  canRemove,
  summary,
  onUpdateThinking,
  onUpdateTemperature,
  onUpdateSeed,
  onRemoveLastSlot,
  onOpenPicker,
}: ChallengeModelCardProps) {
  const { t } = useTranslation("chat");
  const label = option?.label ?? manifestSlot?.displayLabel ?? manifestSlot?.modelName ?? t("htmlChallenge.modelCard.selectAModel", { defaultValue: "Select a model" });
  const format = option?.format ?? manifestSlot?.format ?? "";
  const quantization = option?.quantization ?? manifestSlot?.quantization ?? "";
  const sizeGb = typeof option?.sizeGb === "number"
    ? option.sizeGb
    : typeof manifestSlot?.sizeGb === "number" ? manifestSlot.sizeGb : null;
  const contextWindow = option?.contextWindow ?? manifestSlot?.contextWindow ?? "";
  const thinkingValue = slot.thinkingMode === "auto" ? slot.reasoningEffort : "off";

  return (
    <div key={slot.id}>
      <span className="eyebrow">{compareTargetLabels[slot.id]}</span>
      <div className="model-selected-card model-selected-card--compact">
        <div className="model-selected-info">
          <div className="html-challenge-slot-headline">
            <strong className="html-challenge-slot-name" title={label}>{label}</strong>
            <div className="model-selected-meta html-challenge-slot-badges">
              {format ? <span className="badge muted">{format}</span> : null}
              {quantization ? <span className="badge muted">{quantization}</span> : null}
              {sizeGb ? <span className="badge muted">{sizeLabel(sizeGb)}</span> : null}
              {contextWindow ? <span className="badge muted">{contextWindow}</span> : null}
            </div>
          </div>
          <small className="muted-text">{summary}</small>
          {!completedChallenge ? (
            <div className="html-challenge-slot-sampler-row">
              <label>
                <span>{t("htmlChallenge.modelCard.thinking", { defaultValue: "Thinking" })}</span>
                <select
                  className="text-input"
                  value={thinkingValue}
                  disabled={busy}
                  onChange={(event) => {
                    const next = event.target.value;
                    if (next === "off") onUpdateThinking(slot.id, "off");
                    else onUpdateThinking(slot.id, "auto", next as HtmlChallengeReasoningEffort);
                  }}
                >
                  <option value="off">{t("htmlChallenge.modelCard.thinkingOff", { defaultValue: "Off" })}</option>
                  <option value="low">{t("htmlChallenge.modelCard.thinkingLow", { defaultValue: "Low" })}</option>
                  <option value="medium">{t("htmlChallenge.modelCard.thinkingMed", { defaultValue: "Med" })}</option>
                  <option value="high">{t("htmlChallenge.modelCard.thinkingHigh", { defaultValue: "High" })}</option>
                </select>
              </label>
              <label>
                <span>{t("htmlChallenge.modelCard.temp", { defaultValue: "Temp" })}</span>
                <input
                  className="text-input"
                  type="number"
                  min={0}
                  max={2}
                  step={0.05}
                  value={slot.settings.temperature}
                  disabled={busy}
                  onChange={(event) => {
                    const parsed = parseFloat(event.target.value);
                    if (Number.isFinite(parsed)) onUpdateTemperature(slot.id, parsed);
                  }}
                />
              </label>
              <label className="html-challenge-seed-field">
                <span>{t("htmlChallenge.modelCard.seed", { defaultValue: "Seed" })}</span>
                <div className="html-challenge-seed-field-controls">
                  <input
                    className="text-input"
                    type="number"
                    min={0}
                    max={2147483647}
                    step={1}
                    placeholder={t("htmlChallenge.modelCard.seedPlaceholder", { defaultValue: "random" })}
                    value={slot.seed ?? ""}
                    disabled={busy}
                    onChange={(event) => {
                      const raw = event.target.value;
                      if (!raw) {
                        onUpdateSeed(slot.id, null);
                        return;
                      }
                      const parsed = parseInt(raw, 10);
                      if (Number.isFinite(parsed)) onUpdateSeed(slot.id, parsed);
                    }}
                  />
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={busy}
                    onClick={() => onUpdateSeed(slot.id, randomChallengeSeed())}
                  >
                    {t("htmlChallenge.modelCard.randomize", { defaultValue: "Randomize" })}
                  </button>
                </div>
              </label>
            </div>
          ) : null}
        </div>
        {!completedChallenge ? (
          <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
            {canRemove && isLastSlot ? (
              <button className="secondary-button" type="button" disabled={busy} onClick={onRemoveLastSlot}>
                {t("htmlChallenge.modelCard.remove", { defaultValue: "Remove" })}
              </button>
            ) : null}
            <button className="secondary-button" type="button" disabled={busy} onClick={() => onOpenPicker(slot.id)}>
              {option || manifestSlot
                ? t("htmlChallenge.modelCard.change", { defaultValue: "Change" })
                : t("htmlChallenge.modelCard.select", { defaultValue: "Select" })}
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
