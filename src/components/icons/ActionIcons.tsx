/**
 * Small monochrome action icons for the Music and Storyboard areas. Same
 * contract as SidebarIcons: stroke is `currentColor`, no fill colours.
 */
import type { SVGProps } from "react";

const baseProps: SVGProps<SVGSVGElement> = {
  xmlns: "http://www.w3.org/2000/svg",
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.9,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

type IconProps = Omit<SVGProps<SVGSVGElement>, "children">;

export function LockIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <rect x="4" y="11" width="16" height="10" rx="2" />
      <path d="M8 11V7a4 4 0 0 1 8 0v4" />
    </svg>
  );
}

export function RedoIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <polyline points="21 4 21 10 15 10" />
      <path d="M20 14a8 8 0 1 1-2.3-6.4L21 10" />
    </svg>
  );
}

export function PlayIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <polygon points="7 4 20 12 7 20 7 4" />
    </svg>
  );
}

export function StopIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
    </svg>
  );
}

export function LoopIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <polyline points="17 2 21 6 17 10" />
      <path d="M3 11V9a3 3 0 0 1 3-3h15" />
      <polyline points="7 22 3 18 7 14" />
      <path d="M21 13v2a3 3 0 0 1-3 3H3" />
    </svg>
  );
}

/** Lowers one lane under another: a tall bar, then a short one. */
export function DuckIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <line x1="5" y1="5" x2="5" y2="19" />
      <line x1="10" y1="9" x2="10" y2="19" />
      <line x1="15" y1="13" x2="15" y2="19" />
      <line x1="20" y1="16" x2="20" y2="19" />
    </svg>
  );
}

export function UnlockIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <rect x="4" y="11" width="16" height="10" rx="2" />
      <path d="M8 11V7a4 4 0 0 1 7.5-2" />
    </svg>
  );
}

export function UndoIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <polyline points="3 4 3 10 9 10" />
      <path d="M4 14a8 8 0 1 0 2.3-6.4L3 10" />
    </svg>
  );
}

export function PlusIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

/** Three dots in a row: a menu of more actions. */
export function MoreIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <circle cx="5" cy="12" r="1.2" />
      <circle cx="12" cy="12" r="1.2" />
      <circle cx="19" cy="12" r="1.2" />
    </svg>
  );
}

/** Six dots: a drag handle. */
export function GripIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <circle cx="9" cy="6" r="1.1" />
      <circle cx="15" cy="6" r="1.1" />
      <circle cx="9" cy="12" r="1.1" />
      <circle cx="15" cy="12" r="1.1" />
      <circle cx="9" cy="18" r="1.1" />
      <circle cx="15" cy="18" r="1.1" />
    </svg>
  );
}

export function ChevronUpIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <polyline points="6 15 12 9 18 15" />
    </svg>
  );
}

export function ChevronDownIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

export function TrashIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6M9 6V4h6v2" />
    </svg>
  );
}

export function CloseIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <line x1="6" y1="6" x2="18" y2="18" />
      <line x1="18" y1="6" x2="6" y2="18" />
    </svg>
  );
}

export function PauseIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <line x1="8" y1="5" x2="8" y2="19" />
      <line x1="16" y1="5" x2="16" y2="19" />
    </svg>
  );
}

export function MinusIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}
