import type { ReactNode } from "react";

interface InfoTooltipProps {
  /** Body of the tooltip. Plain text preferred; pass JSX only when a line
   * break or emphasis is load-bearing for comprehension. */
  text: ReactNode;
  /** Override the default ℹ︎ icon when a different affordance (warning,
   * help, etc.) fits the surrounding context better. */
  icon?: string;
}

// Small hover-triggered tooltip used to explain settings fields without
// inflating the label itself. Uses CSS :hover + :focus-within so the
// bubble is keyboard-accessible (tab onto the icon span → it shows) and
// doesn't require any JS state. Native `title` was considered but its
// 500-1500ms delay and un-styleable body make it a bad fit for dense
// forms where the user wants a quick glance at what a slider does.
export function InfoTooltip({ text, icon = "ℹ" }: InfoTooltipProps) {
  return (
    <span className="info-tooltip" tabIndex={0} aria-label="More info">
      <span className="info-tooltip-icon" aria-hidden="true">{icon}</span>
      <span className="info-tooltip-body" role="tooltip">{text}</span>
    </span>
  );
}
