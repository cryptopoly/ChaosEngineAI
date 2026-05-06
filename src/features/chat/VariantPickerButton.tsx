import { useEffect, useRef, useState } from "react";
import type { ModelCapabilities, WarmModel } from "../../types";
import { resolveCapabilities } from "../../utils";

const CAPABILITY_HINT_FLAGS: Array<{
  flag: keyof Omit<ModelCapabilities, "tags">;
  label: string;
}> = [
  { flag: "supportsVision", label: "Vision" },
  { flag: "supportsTools", label: "Tools" },
  { flag: "supportsReasoning", label: "Reasoning" },
  { flag: "supportsCoding", label: "Code" },
];

/**
 * Phase 2.5: dropdown that triggers in-thread compare. Picking a warm
 * model schedules a sibling response from that model for the same
 * prompt. Cards render under the assistant bubble; primary text is
 * unchanged. Only warm models are offered so the alt response is
 * available without a model load.
 */
export interface VariantPickerButtonProps {
  warmModels: WarmModel[];
  /** The model that produced the primary text — excluded from the list. */
  currentModelRef: string | null;
  onPick: (warm: WarmModel) => void;
  disabled?: boolean;
}

export function VariantPickerButton({
  warmModels,
  currentModelRef,
  onPick,
  disabled,
}: VariantPickerButtonProps) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (event: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const candidates = warmModels.filter((warm) => warm.ref !== currentModelRef);
  if (candidates.length === 0) return null;

  return (
    <div className="variant-picker" ref={wrapRef}>
      <button
        type="button"
        className="message-action-btn"
        title="Compare with another warm model"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <rect x="3" y="3" width="7" height="18" rx="1" />
          <rect x="14" y="3" width="7" height="18" rx="1" />
        </svg>
      </button>
      {open ? (
        <div className="variant-picker__popover" role="dialog" aria-label="Pick a model to compare">
          <div className="variant-picker__heading">
            <strong>Compare with</strong>
            <small>Adds a sibling response from another warm model.</small>
          </div>
          {candidates.map((warm) => {
            const caps = resolveCapabilities(warm.ref, null);
            const hints = CAPABILITY_HINT_FLAGS.filter((entry) => caps[entry.flag]);
            return (
              <button
                key={warm.ref}
                type="button"
                className="variant-picker__item"
                onClick={() => {
                  onPick(warm);
                  setOpen(false);
                }}
              >
                <div className="variant-picker__item-main">
                  <span className="variant-picker__item-name">{warm.name}</span>
                  <span className="variant-picker__item-engine">{warm.engine}</span>
                </div>
                {hints.length ? (
                  <span className="variant-picker__item-hints">
                    {hints.map((entry) => (
                      <span key={entry.flag} className="capability-badge">
                        {entry.label}
                      </span>
                    ))}
                  </span>
                ) : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
