import {
  type CSSProperties,
  type ReactNode,
  useCallback,
  useId,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";

interface InfoTooltipProps {
  /** Body of the tooltip. Plain text preferred; pass JSX only when a line
   * break or emphasis is load-bearing for comprehension. */
  text: ReactNode;
  /** Override the default ℹ︎ icon when a different affordance (warning,
   * help, etc.) fits the surrounding context better. */
  icon?: string;
}

interface TooltipPosition {
  left: number;
  maxWidth: number;
  placement: "above" | "below";
  top?: number;
  bottom?: number;
}

const TOOLTIP_MARGIN = 12;
const TOOLTIP_GAP = 8;
const TOOLTIP_MAX_WIDTH = 320;

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

// Small hover-triggered tooltip used to explain settings fields without
// inflating the label itself. The bubble is portalled to document.body and
// clamped to the viewport so Studio panels/sidebar overflow cannot crop it.
// Native `title` was considered but its 500-1500ms delay and un-styleable
// body make it a bad fit for dense forms where users want quick context.
export function InfoTooltip({ text, icon = "ℹ" }: InfoTooltipProps) {
  const tooltipId = useId();
  const triggerRef = useRef<HTMLSpanElement>(null);
  const [position, setPosition] = useState<TooltipPosition | null>(null);

  const showTooltip = useCallback(() => {
    if (typeof window === "undefined") return;
    const trigger = triggerRef.current;
    if (!trigger) return;

    const rect = trigger.getBoundingClientRect();
    const availableWidth = Math.max(160, window.innerWidth - TOOLTIP_MARGIN * 2);
    const maxWidth = Math.min(TOOLTIP_MAX_WIDTH, availableWidth);
    const maxLeft = Math.max(TOOLTIP_MARGIN, window.innerWidth - maxWidth - TOOLTIP_MARGIN);
    const left = clamp(
      rect.left + rect.width / 2 - maxWidth / 2,
      TOOLTIP_MARGIN,
      maxLeft,
    );

    if (rect.top > 120) {
      setPosition({
        left,
        maxWidth,
        placement: "above",
        bottom: window.innerHeight - rect.top + TOOLTIP_GAP,
      });
      return;
    }

    setPosition({
      left,
      maxWidth,
      placement: "below",
      top: rect.bottom + TOOLTIP_GAP,
    });
  }, []);

  const hideTooltip = useCallback(() => {
    setPosition(null);
  }, []);

  const tooltipStyle: CSSProperties | undefined = position
    ? {
        left: position.left,
        maxWidth: position.maxWidth,
        ...(position.placement === "above"
          ? { bottom: position.bottom }
          : { top: position.top }),
      }
    : undefined;

  return (
    <>
      <span
        ref={triggerRef}
        className="info-tooltip"
        tabIndex={0}
        aria-label="More info"
        aria-describedby={position ? tooltipId : undefined}
        onMouseEnter={showTooltip}
        onMouseLeave={hideTooltip}
        onFocus={showTooltip}
        onBlur={hideTooltip}
      >
        <span className="info-tooltip-icon" aria-hidden="true">{icon}</span>
      </span>
      {position && typeof document !== "undefined"
        ? createPortal(
            <span
              id={tooltipId}
              className="info-tooltip-body info-tooltip-body--portal"
              role="tooltip"
              style={tooltipStyle}
            >
              {text}
            </span>,
            document.body,
          )
        : null}
    </>
  );
}
