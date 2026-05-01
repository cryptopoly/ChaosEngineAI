import { useEffect, useRef, useState } from "react";
import type { WarmModel } from "../../types";

/**
 * Phase 2.12: dropdown that lets the user send the next message through
 * a different warm model without changing the thread's default. Picking
 * an entry sets a one-turn override; the override clears after the
 * stream finishes (parent owns that lifecycle). Picking "Clear override"
 * (or unloading the chosen model) reverts to the session default.
 *
 * The menu only surfaces *warm* models (already resident) so the swap
 * is instantaneous — switching to a cold model would force a load and
 * defeat the "quick alt for one turn" framing.
 */
export interface MidThreadSwapMenuProps {
  warmModels: WarmModel[];
  /** The session's current default model ref (excluded from the list). */
  sessionModelRef: string | undefined;
  /** Currently-selected one-turn override, or null when none. */
  overrideRef: string | null;
  onSelect: (warm: WarmModel | null) => void;
  disabled?: boolean;
}

export function MidThreadSwapMenu({
  warmModels,
  sessionModelRef,
  overrideRef,
  onSelect,
  disabled,
}: MidThreadSwapMenuProps) {
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

  const candidates = warmModels.filter(
    (warm) => warm.ref !== sessionModelRef,
  );

  const selectedWarm = overrideRef
    ? warmModels.find((warm) => warm.ref === overrideRef) ?? null
    : null;

  if (candidates.length === 0 && !selectedWarm) return null;

  const triggerLabel = selectedWarm
    ? `Next: ${truncateName(selectedWarm.name)}`
    : "Send via...";

  return (
    <div className="swap-menu" ref={wrapRef}>
      <button
        type="button"
        className={`secondary-button swap-menu__trigger${selectedWarm ? " swap-menu__trigger--active" : ""}`}
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        title={
          selectedWarm
            ? `Next message will go to ${selectedWarm.name} (one-turn override)`
            : "Send the next message through a different warm model"
        }
      >
        {triggerLabel}
        {selectedWarm ? (
          <span
            className="swap-menu__clear"
            role="button"
            tabIndex={0}
            aria-label="Clear override"
            title="Clear override"
            onClick={(e) => {
              e.stopPropagation();
              onSelect(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.stopPropagation();
                onSelect(null);
              }
            }}
          >
            ×
          </span>
        ) : null}
      </button>
      {open ? (
        <div className="swap-menu__popover" role="dialog" aria-label="Pick a model for the next turn">
          <div className="swap-menu__heading">
            <strong>Send next via</strong>
            <small>Override applies for one turn only.</small>
          </div>
          {candidates.length === 0 ? (
            <p className="muted-text" style={{ margin: "8px 0", fontSize: 11 }}>
              No other warm models available. Load a second model from My Models to enable quick swaps.
            </p>
          ) : (
            candidates.map((warm) => (
              <button
                key={warm.ref}
                type="button"
                className={`swap-menu__item${overrideRef === warm.ref ? " swap-menu__item--active" : ""}`}
                onClick={() => {
                  onSelect(warm);
                  setOpen(false);
                }}
              >
                <span className="swap-menu__item-name">{warm.name}</span>
                <span className="swap-menu__item-engine">{warm.engine}</span>
              </button>
            ))
          )}
          {selectedWarm ? (
            <button
              type="button"
              className="swap-menu__reset"
              onClick={() => {
                onSelect(null);
                setOpen(false);
              }}
            >
              Clear override (use thread default)
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function truncateName(name: string): string {
  if (name.length <= 18) return name;
  return `${name.slice(0, 16)}…`;
}
