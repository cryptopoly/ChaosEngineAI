/**
 * One screen anatomy for every Studio: header, inputs, preset row,
 * estimate, Options and Expert disclosures, results. The shell only lays
 * the slots out and remembers which disclosures the user left open
 * (per `studioId`); what goes in each slot belongs to the Studio.
 */
import { useCallback, useMemo, useRef, type ReactNode } from "react";
import {
  DisclosureMemoryContext,
  readDisclosures,
  writeDisclosures,
  type DisclosureMemory,
  type DisclosureState,
} from "./studioDisclosureMemory";

export interface StudioShellProps {
  studioId: string;
  header: ReactNode;
  inputs: ReactNode;
  presets?: ReactNode;
  estimate?: ReactNode;
  options?: ReactNode;
  expert?: ReactNode;
  results?: ReactNode;
}

export function StudioShell({ studioId, header, inputs, presets, estimate, options, expert, results }: StudioShellProps) {
  const stored = useRef<{ id: string; state: DisclosureState } | null>(null);
  const read = useCallback((): DisclosureState => {
    if (!stored.current || stored.current.id !== studioId) {
      stored.current = { id: studioId, state: readDisclosures(studioId) };
    }
    return stored.current.state;
  }, [studioId]);

  const memory = useMemo<DisclosureMemory>(
    () => ({
      isOpen: (id) => read()[id],
      setOpen: (id, open) => {
        const next = { ...read(), [id]: open };
        stored.current = { id: studioId, state: next };
        writeDisclosures(studioId, next);
      },
    }),
    [read, studioId],
  );

  return (
    <DisclosureMemoryContext.Provider value={memory}>
      <div className="studio-shell" data-studio={studioId}>
        <header className="studio-shell__header">{header}</header>
        <div className="studio-shell__inputs">{inputs}</div>
        {presets ? <div className="studio-shell__presets">{presets}</div> : null}
        {estimate ? <div className="studio-shell__estimate">{estimate}</div> : null}
        {options || expert ? (
          <div className="studio-shell__options">
            {options}
            {expert}
          </div>
        ) : null}
        {results ? <div className="studio-shell__results">{results}</div> : null}
      </div>
    </DisclosureMemoryContext.Provider>
  );
}
