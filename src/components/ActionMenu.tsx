/**
 * A small "more actions" menu: an icon button that opens a list of buttons.
 *
 * Keyboard: Enter / Space / Arrow Down on the button opens the list and focuses
 * the first enabled item; Arrow Up / Down, Home and End move between items;
 * Escape closes and returns focus to the button; Tab closes. An outside click
 * or a pick closes it too (a pick returns focus to the button).
 */
import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { MoreIcon } from "./icons/ActionIcons";

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
  const trigger = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLDivElement>(null);

  const enabledItems = (): HTMLButtonElement[] =>
    Array.from(list.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)') ?? []);

  function close(returnFocus: boolean) {
    setOpen(false);
    if (returnFocus) trigger.current?.focus();
  }

  useEffect(() => {
    if (!open) return;
    enabledItems()[0]?.focus();
    const onPointer = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onPointer);
    return () => document.removeEventListener("mousedown", onPointer);
  }, [open]);

  function onTriggerKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "ArrowDown" && !open) {
      event.preventDefault();
      setOpen(true);
    }
  }

  function onListKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const buttons = enabledItems();
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    switch (event.key) {
      case "Escape":
        event.preventDefault();
        close(true);
        break;
      case "Tab":
        setOpen(false);
        break;
      case "ArrowDown":
        event.preventDefault();
        buttons[(index + 1) % buttons.length]?.focus();
        break;
      case "ArrowUp":
        event.preventDefault();
        buttons[(index - 1 + buttons.length) % buttons.length]?.focus();
        break;
      case "Home":
        event.preventDefault();
        buttons[0]?.focus();
        break;
      case "End":
        event.preventDefault();
        buttons[buttons.length - 1]?.focus();
        break;
    }
  }

  return (
    <div className="action-menu" ref={root}>
      <button
        ref={trigger}
        type="button"
        className="secondary-button icon-button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
        title={label}
        onClick={() => setOpen((value) => !value)}
        onKeyDown={onTriggerKeyDown}
      >
        <MoreIcon />
      </button>
      {open ? (
        <div className="action-menu__list" role="menu" ref={list} onKeyDown={onListKeyDown}>
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              role="menuitem"
              className={`action-menu__item${item.danger ? " action-menu__item--danger" : ""}`}
              disabled={item.disabled}
              onClick={() => {
                close(true);
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
