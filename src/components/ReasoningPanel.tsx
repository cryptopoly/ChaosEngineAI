import { useEffect, useRef, useState } from "react";
import { RichMarkdown } from "./RichMarkdown";

interface ReasoningPanelProps {
  text?: string | null;
  streaming?: boolean;
}

function lastLine(text: string): string {
  const lines = text.split("\n").filter(Boolean);
  return lines.length > 0 ? lines[lines.length - 1] : "";
}

export function ReasoningPanel({ text, streaming = false }: ReasoningPanelProps) {
  const content = text?.trim() ?? "";
  const [open, setOpen] = useState(Boolean(content && streaming));
  const prevStreamingRef = useRef(streaming);
  const userCollapsedRef = useRef(false);

  // Auto-open when streaming starts (new reasoning content appears),
  // but only if the user hasn't manually collapsed it.
  useEffect(() => {
    if (streaming && content && !userCollapsedRef.current) {
      setOpen(true);
    }
  }, [streaming, content]);

  // Auto-collapse when streaming ends.  Reset the user-collapsed
  // flag so the next message auto-opens fresh.
  useEffect(() => {
    if (prevStreamingRef.current && !streaming && content) {
      setOpen(false);
      userCollapsedRef.current = false;
    }
    prevStreamingRef.current = streaming;
  }, [streaming, content]);

  if (!content) return null;

  const handleToggle = () => {
    setOpen((current) => {
      const next = !current;
      // Track that the user explicitly collapsed so auto-open
      // doesn't fight with them during streaming.
      if (!next) {
        userCollapsedRef.current = true;
      } else {
        userCollapsedRef.current = false;
      }
      return next;
    });
  };

  return (
    <div className={`reasoning-panel${open ? " reasoning-panel--open" : ""}`}>
      <button
        type="button"
        className="reasoning-panel__toggle"
        onClick={handleToggle}
        aria-expanded={open}
      >
        <span className={`reasoning-panel__chevron${open ? " reasoning-panel__chevron--open" : ""}`}>›</span>
        <span>{streaming ? "Thinking..." : "Thinking"}</span>
        {!open && streaming ? (
          <span className="reasoning-panel__preview">{lastLine(content)}</span>
        ) : null}
      </button>
      {open ? (
        <div className="reasoning-panel__body">
          <div className="markdown-content reasoning-panel__content">
            <RichMarkdown>{content}</RichMarkdown>
          </div>
        </div>
      ) : null}
    </div>
  );
}
