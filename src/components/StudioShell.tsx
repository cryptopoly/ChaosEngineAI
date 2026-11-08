/**
 * One screen anatomy for every Studio: header, inputs, preset row,
 * estimate, Options and Expert disclosures, results. The shell only lays
 * the slots out and remembers which disclosures the user left open
 * (per `studioId`); what goes in each slot belongs to the Studio.
 */
import type { ReactNode } from "react";
import { DisclosureMemoryContext, useDisclosureMemory } from "./studioDisclosureMemory";

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
  const memory = useDisclosureMemory(studioId);

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
