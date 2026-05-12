/**
 * FU-022: Prompt enhancer button for the Image / Video Studio prompt
 * fields. Click → POST /api/prompt/enhance with the current prompt +
 * the selected variant's repo id; on success, replace the prompt
 * textarea via the parent's setter and surface a 1-line note as a
 * tooltip on the button (so the user knows which model rewrote it).
 *
 * Apple Silicon path uses the small LLM rewrite. Other platforms use
 * the backend's deterministic template fallback so the button still
 * changes short prompts without adding runtime cost.
 */
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { enhancePromptViaLLM } from "../api";

export interface PromptEnhanceButtonProps {
  prompt: string;
  repo: string;
  onEnhanced: (next: string) => void;
}

export function PromptEnhanceButton({
  prompt,
  repo,
  onEnhanced,
}: PromptEnhanceButtonProps) {
  const { t } = useTranslation("studio");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const trimmed = prompt.trim();
  const disabled = busy || !trimmed || !repo;

  const handleClick = async () => {
    if (disabled) return;
    setBusy(true);
    setNote(null);
    try {
      const result = await enhancePromptViaLLM({ prompt: trimmed, repo });
      // Only replace when the model actually changed the prompt — when
      // the helper falls back (no Apple Silicon, mlx_lm missing, model
      // not cached), enhanced === original and we just surface the
      // note instead of clobbering the textarea.
      if (result.enhanced && result.enhanced !== trimmed) {
        onEnhanced(result.enhanced);
      }
      setNote(result.note);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setNote(t("enhance.error", { message, defaultValue: `Enhancer error: ${message}` }));
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      type="button"
      className="prompt-enhance-button"
      onClick={() => void handleClick()}
      disabled={disabled}
      title={note ?? t("enhance.tooltip", { defaultValue: "Enhance this prompt locally" })}
    >
      {busy
        ? t("enhance.busy", { defaultValue: "Enhancing..." })
        : t("enhance.label", { defaultValue: "Enhance" })}
    </button>
  );
}
