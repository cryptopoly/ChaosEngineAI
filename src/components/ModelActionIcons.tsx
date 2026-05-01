import type { ButtonHTMLAttributes, ReactNode } from "react";

export type ActionIconName =
  | "cancel"
  | "chat"
  | "convert"
  | "delete"
  | "download"
  | "generate"
  | "huggingFace"
  | "install"
  | "modelCard"
  | "pause"
  | "resume"
  | "retry"
  | "reveal"
  | "server";

export type ModelStatusKind =
  | "downloaded"
  | "downloading"
  | "failed"
  | "incomplete"
  | "installed"
  | "loaded"
  | "paused";

type IconProps = {
  name: ActionIconName | ModelStatusKind;
  className?: string;
};

function Svg({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <svg
      className={className}
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

export function ModelActionIcon({ name, className }: IconProps) {
  switch (name) {
    case "cancel":
    case "failed":
      return (
        <Svg className={className}>
          <circle cx="12" cy="12" r="9" />
          <path d="M9 9l6 6" />
          <path d="M15 9l-6 6" />
        </Svg>
      );
    case "chat":
      return (
        <Svg className={className}>
          <path d="M5 6.5A3.5 3.5 0 0 1 8.5 3h7A3.5 3.5 0 0 1 19 6.5v4A3.5 3.5 0 0 1 15.5 14H10l-5 4v-4.8A3.5 3.5 0 0 1 5 12.5z" />
          <path d="M8.5 8h7" />
          <path d="M8.5 11h4" />
        </Svg>
      );
    case "convert":
      return (
        <Svg className={className}>
          <path d="M7 7h9.5A3.5 3.5 0 0 1 20 10.5v0" />
          <path d="M10 4 7 7l3 3" />
          <path d="M17 17H7.5A3.5 3.5 0 0 1 4 13.5v0" />
          <path d="m14 14 3 3-3 3" />
        </Svg>
      );
    case "delete":
      return (
        <Svg className={className}>
          <path d="M4 6h16" />
          <path d="M9 6V4h6v2" />
          <path d="m6 6 1 14h10l1-14" />
          <path d="M10 11v5" />
          <path d="M14 11v5" />
        </Svg>
      );
    case "download":
    case "downloading":
      return (
        <Svg className={className}>
          <path d="M12 3v11" />
          <path d="m7 10 5 5 5-5" />
          <path d="M5 19h14" />
        </Svg>
      );
    case "generate":
      return (
        <Svg className={className}>
          <path d="m12 3 1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6z" />
          <path d="m19 14 .9 2.1L22 17l-2.1.9L19 20l-.9-2.1L16 17l2.1-.9z" />
          <path d="M5 15v5" />
          <path d="M2.5 17.5h5" />
        </Svg>
      );
    case "huggingFace":
      return (
        <Svg className={className}>
          <rect x="4" y="5" width="13" height="14" rx="2" />
          <path d="M8 9h5" />
          <path d="M8 13h3" />
          <path d="M16 4h4v4" />
          <path d="M13 11 20 4" />
          <path d="M7.5 17h1.5v-3" />
          <path d="M9 15.5H7.5" />
          <path d="M11 17v-3h2" />
          <path d="M11 15.5h1.5" />
        </Svg>
      );
    case "install":
      return (
        <Svg className={className}>
          <path d="M12 3v10" />
          <path d="m8 9 4 4 4-4" />
          <path d="M5 17h14v4H5z" />
        </Svg>
      );
    case "modelCard":
      return (
        <Svg className={className}>
          <rect x="4" y="5" width="13" height="14" rx="2" />
          <path d="M8 9h5" />
          <path d="M8 13h5" />
          <path d="M8 17h3" />
          <path d="M16 4h4v4" />
          <path d="M13 11 20 4" />
        </Svg>
      );
    case "pause":
    case "paused":
      return (
        <Svg className={className}>
          <circle cx="12" cy="12" r="9" />
          <path d="M9.5 8.5v7" />
          <path d="M14.5 8.5v7" />
        </Svg>
      );
    case "resume":
      return (
        <Svg className={className}>
          <path d="M8 5v14l11-7z" />
        </Svg>
      );
    case "retry":
      return (
        <Svg className={className}>
          <path d="M20 12a8 8 0 1 1-2.3-5.7" />
          <path d="M20 5v6h-6" />
        </Svg>
      );
    case "reveal":
      return (
        <Svg className={className}>
          <path d="M3.5 7.5h6l2 2H20a1.5 1.5 0 0 1 1.5 1.5v6A2.5 2.5 0 0 1 19 19.5H5A2.5 2.5 0 0 1 2.5 17V9A1.5 1.5 0 0 1 4 7.5z" />
          <path d="M13 15h6" />
          <path d="m16.5 12 2.5 3-2.5 3" />
        </Svg>
      );
    case "server":
      return (
        <Svg className={className}>
          <rect x="4" y="4" width="16" height="6" rx="2" />
          <rect x="4" y="14" width="16" height="6" rx="2" />
          <path d="M8 7h.01" />
          <path d="M8 17h.01" />
          <path d="M12 7h4" />
          <path d="M12 17h4" />
        </Svg>
      );
    case "downloaded":
    case "installed":
      return (
        <Svg className={className}>
          <circle cx="12" cy="12" r="9" />
          <path d="m8 12 2.6 2.6L16.5 9" />
        </Svg>
      );
    case "loaded":
      return (
        <Svg className={className}>
          <circle cx="12" cy="12" r="9" />
          <path d="m13 3-4 10h4l-2 8 5-11h-4z" />
        </Svg>
      );
    case "incomplete":
    default:
      return (
        <Svg className={className}>
          <circle cx="12" cy="12" r="9" />
          <path d="M8 12h8" />
        </Svg>
      );
  }
}

type IconActionButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  icon: ActionIconName;
  label: string;
  buttonStyle?: "primary" | "secondary";
  danger?: boolean;
};

export function IconActionButton({
  icon,
  label,
  buttonStyle = "secondary",
  danger = false,
  className = "",
  title,
  type = "button",
  ...props
}: IconActionButtonProps) {
  return (
    <button
      {...props}
      className={`${buttonStyle}-button icon-button action-icon-button${danger ? " danger-button" : ""}${className ? ` ${className}` : ""}`}
      type={type}
      title={title ?? label}
      aria-label={label}
    >
      <ModelActionIcon name={icon} />
      <span className="sr-only">{label}</span>
    </button>
  );
}

export function StatusIcon({
  status,
  label,
  detail,
}: {
  status: ModelStatusKind;
  label: string;
  detail?: string | null;
}) {
  const title = detail ? `${label}: ${detail}` : label;
  return (
    <span className={`status-icon-pill status-icon-pill--${status}`} title={title} aria-label={title}>
      <ModelActionIcon name={status} />
      <span className="sr-only">{title}</span>
    </span>
  );
}
