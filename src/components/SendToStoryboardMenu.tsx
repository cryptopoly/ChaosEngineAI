/**
 * "Use in storyboard" / "Send to storyboard": a menu button that lists the most recent projects, then (for a
 * picture or a clip) where in the project it goes. It is the keyboard path for dragging a thing onto the
 * storyboard. Not rendered at all unless the Music and Storyboard areas are on. Used by the Image and Video
 * Galleries, Voice Studio and the Voice Library, and later by the Music Library (kinds `music` and `sfx`).
 */
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { HOME_STUDIO_ENABLED } from "../constants/features";
import {
  hasSingleTarget,
  menuKeyAction,
  sendMenuVisible,
  targetsFor,
  type SendItem,
  type SendProject,
  type SendTarget,
} from "../features/storyboard/sendToStoryboardLogic";
import { SendIcon } from "./icons/ActionIcons";

export interface SendToStoryboardMenuProps {
  item: SendItem | null;
  projects: SendProject[];
  /** True once the project list was read; before that the button keeps its normal label. */
  loaded?: boolean;
  busy?: boolean;
  notice?: string | null;
  /** Defaults to the app-wide switch; a test or a caller can pass its own. */
  enabled?: boolean;
  /** Called when the menu opens, to refresh the project list. */
  onOpen?: () => void;
  /** `projectId` null means "a new project". */
  onSend: (projectId: string | null, target: SendTarget) => void | Promise<unknown>;
  onDismissNotice?: () => void;
  /** Render the list already open (on a project's targets when `initialProjectId` is set); for tests. */
  initialOpen?: boolean;
  initialProjectId?: string | null;
}

/** One row of the open menu; `key` keeps the rows apart for focus. */
export interface MenuRow {
  key: string;
  label: string;
  title?: string;
  run: () => void;
}

export function SendToStoryboardMenu({
  item,
  projects,
  loaded = true,
  busy = false,
  notice = null,
  enabled = HOME_STUDIO_ENABLED,
  onOpen,
  onSend,
  onDismissNotice,
  initialOpen = false,
  initialProjectId = null,
}: SendToStoryboardMenuProps) {
  const { t } = useTranslation("studio");
  const [open, setOpen] = useState(initialOpen);
  const [projectId, setProjectId] = useState<string | null>(initialProjectId);
  // The notice belongs to the menu whose send failed, not to every card on the page.
  const [failed, setFailed] = useState(false);
  const [mine, setMine] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const rows = useRef<(HTMLButtonElement | null)[]>([]);

  const visible = sendMenuVisible(enabled, item);
  const noProjects = loaded && projects.length === 0;

  function close(refocus: boolean) {
    setOpen(false);
    setProjectId(null);
    if (refocus) button.current?.focus();
  }

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) close(false);
    };
    document.addEventListener("mousedown", onPointer);
    return () => document.removeEventListener("mousedown", onPointer);
  }, [open]);

  // Focus the first row when the menu opens and when it moves on to the targets of a project.
  useEffect(() => {
    if (open) rows.current[0]?.focus({ preventScroll: true });
  }, [open, projectId]);

  // A menu near the bottom of the window opens upward, and a long one scrolls inside itself.
  const list = useRef<HTMLDivElement>(null);
  const [up, setUp] = useState(false);
  useEffect(() => {
    if (!open || !list.current || !button.current) return;
    const below = window.innerHeight - button.current.getBoundingClientRect().bottom;
    const above = button.current.getBoundingClientRect().top;
    setUp(list.current.scrollHeight + 8 > below && above > below);
  }, [open, projectId]);

  if (!visible || !item) return null;

  const single = hasSingleTarget(item);
  const label = noProjects
    ? t("sendToStoryboard.start", { defaultValue: "Start a storyboard with this" })
    : single
      ? t("sendToStoryboard.send", { defaultValue: "Send to storyboard" })
      : t("sendToStoryboard.use", { defaultValue: "Use in storyboard" });

  const run = async (id: string | null, target: SendTarget) => {
    setFailed(false);
    setMine(true);
    const result = await onSend(id, target);
    setMine(false);
    setFailed(result === false);
  };
  const pick = (id: string | null, target: SendTarget) => {
    close(true);
    void run(id, target);
  };
  const newProjectTarget: SendTarget = single ? { type: "audio" } : { type: "new-shot" };

  const chosen = projectId ? projects.find((project) => project.id === projectId) ?? null : null;
  let menuRows: MenuRow[] = [];
  if (chosen) {
    menuRows = [
      {
        key: "back",
        label: t("sendToStoryboard.back", { defaultValue: "Back to projects" }),
        run: () => setProjectId(null),
      },
      ...targetsFor(item, chosen).map((target): MenuRow => {
        if (target.type === "new-take") {
          const shot = chosen.shots.find((entry) => entry.id === target.shotId);
          return {
            key: `take-${target.shotId}`,
            label: shot?.title
              ? t("sendToStoryboard.newTakeNamed", { defaultValue: "{n} · {title} (new take)", n: shot.number, title: shot.title })
              : t("sendToStoryboard.newTake", { defaultValue: "Shot {n} (new take)", n: shot?.number ?? 0 }),
            title: shot?.title,
            run: () => pick(chosen.id, target),
          };
        }
        return {
          key: "new-shot",
          label: t("sendToStoryboard.newShot", { defaultValue: "New shot" }),
          run: () => pick(chosen.id, target),
        };
      }),
    ];
  } else {
    menuRows = [
      ...projects.map(
        (project): MenuRow => ({
          key: `project-${project.id}`,
          label: project.name,
          run: () => (single ? pick(project.id, { type: "audio" }) : setProjectId(project.id)),
        }),
      ),
      {
        key: "new-project",
        label: t("sendToStoryboard.newProject", { defaultValue: "New project…" }),
        run: () => pick(null, newProjectTarget),
      },
    ];
  }

  function onButtonClick() {
    if (noProjects) {
      void run(null, newProjectTarget);
      return;
    }
    if (open) {
      close(false);
      return;
    }
    onOpen?.();
    setOpen(true);
  }

  function onListKey(event: KeyboardEvent<HTMLDivElement>) {
    const index = rows.current.findIndex((row) => row === document.activeElement);
    const action = menuKeyAction(event.key, index, menuRows.length);
    if (!action) return;
    if (action.type === "focus") {
      event.preventDefault();
      rows.current[action.index]?.focus();
    } else {
      if (event.key === "Escape") event.preventDefault();
      close(event.key === "Escape");
    }
  }

  return (
    <div className="send-to-menu" ref={root}>
      <button
        type="button"
        ref={button}
        className="secondary-button send-to-menu__button"
        aria-haspopup={noProjects ? undefined : "menu"}
        aria-expanded={noProjects ? undefined : open}
        disabled={busy}
        onClick={onButtonClick}
      >
        <SendIcon />
        {busy && mine ? t("sendToStoryboard.sending", { defaultValue: "Sending…" }) : label}
      </button>
      {open ? (
        <div ref={list} className={`send-to-menu__list${up ? " send-to-menu__list--up" : ""}`} role="menu" aria-label={label} onKeyDown={onListKey}>
          {chosen ? <p className="send-to-menu__heading muted-text">{chosen.name}</p> : null}
          {menuRows.map((row, index) => (
            <button
              key={row.key}
              type="button"
              role="menuitem"
              className="send-to-menu__item"
              title={row.title}
              ref={(node) => {
                rows.current[index] = node;
              }}
              onClick={row.run}
            >
              {row.label}
            </button>
          ))}
        </div>
      ) : null}
      {notice && failed ? (
        <p className="send-to-menu__notice" role="status">
          {notice}
          {onDismissNotice ? (
            <button
              type="button"
              className="secondary-button"
              onClick={() => {
                setFailed(false);
                onDismissNotice();
              }}
            >
              {t("sendToStoryboard.dismiss", { defaultValue: "Dismiss" })}
            </button>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}
