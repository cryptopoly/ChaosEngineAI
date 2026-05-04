/**
 * FU-022: Prompt enhancer button for the Image / Video Studio prompt
 * fields. Click → POST /api/prompt/enhance with the current prompt +
 * the selected variant's repo id; on success, replace the prompt
 * textarea via the parent's setter and surface a 1-line note as a
 * tooltip on the button (so the user knows which model rewrote it).
 *
 * Apple Silicon path actually rewrites the prompt; other platforms
 * get a no-op + the runtimeNote ("enhancer requires mlx_lm"), and we
 * leave the original prompt in place so the user isn't blocked.
 */
import { useState } from "react";
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
      setNote(`Enhancer error: ${message}`);
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
      title={note ?? "Rewrite the prompt via the local LLM enhancer (Apple Silicon)"}
    >
      {busy ? "Enhancing..." : "Enhance"}
    </button>
  );
}
