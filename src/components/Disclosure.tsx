/**
 * A native `<details>` group: the summary line shows the title and a muted
 * summary of the current values, so a collapsed group still tells the user
 * what is set. Keyboard and screen reader support come from the element.
 *
 * Open state: pass `open` + `onToggle` to control it; otherwise it starts
 * at `defaultOpen`, or at what the surrounding `StudioShell` remembered.
 */
import { useContext, useState, type ReactNode } from "react";
import { DisclosureMemoryContext } from "./studioDisclosureMemory";

export interface DisclosureProps {
  id: string;
  title: string;
  /** Muted values shown next to the title, e.g. "BPM 120 · Key A minor". */
  summary?: string;
  defaultOpen?: boolean;
  open?: boolean;
  onToggle?: (open: boolean) => void;
  children: ReactNode;
}

export function Disclosure({ id, title, summary, defaultOpen, open, onToggle, children }: DisclosureProps) {
  const memory = useContext(DisclosureMemoryContext);
  const controlled = open !== undefined;
  const [inner, setInner] = useState<boolean>(() => memory?.isOpen(id) ?? defaultOpen ?? false);
  const isOpen = controlled ? open : inner;

  return (
    <details
      className="disclosure"
      id={`disclosure-${id}`}
      open={isOpen}
      onToggle={(event) => {
        const next = event.currentTarget.open;
        if (next === isOpen) return;
        if (!controlled) setInner(next);
        memory?.setOpen(id, next);
        onToggle?.(next);
      }}
    >
      <summary className="disclosure__summary">
        <svg
          className={`models-card__chevron${isOpen ? " is-open" : ""}`}
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        <span className="disclosure__title">{title}</span>
        {summary ? <span className="disclosure__values">{summary}</span> : null}
      </summary>
      <div className="disclosure__body">{children}</div>
    </details>
  );
}
