/**
 * A small "more actions" menu: an icon button that opens a list of buttons. Closes on Escape, an
 * outside click or a pick. Items are plain buttons so the keyboard path is Tab and Enter.
 */
import { useEffect, useRef, useState, type ReactNode } from "react";
import { MoreIcon } from "../../components/icons/ActionIcons";

export interface ActionMenuItem {
  id: string;
  label: string;
  onSelect: () => void;
  disabled?: boolean;
  danger?: boolean;
  icon?: ReactNode;
}

export function ActionMenu({ label, items }: { label: string; items: ActionMenuItem[] }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="action-menu" ref={root}>
      <button
        type="button"
        className="secondary-button icon-button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
        title={label}
        onClick={() => setOpen((value) => !value)}
      >
        <MoreIcon />
      </button>
      {open ? (
        <div className="action-menu__list" role="menu">
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              role="menuitem"
              className={`action-menu__item${item.danger ? " action-menu__item--danger" : ""}`}
              disabled={item.disabled}
              onClick={() => {
                setOpen(false);
                item.onSelect();
              }}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
