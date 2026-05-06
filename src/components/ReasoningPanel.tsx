import { useEffect, useRef, useState } from "react";
import { RichMarkdown } from "./RichMarkdown";

interface ReasoningPanelProps {
  text?: string | null;
  streaming?: boolean;
}

/**
 * Phase 2.5+ post-fix: take the last N non-empty lines from the
 * cumulative reasoning text. The streaming preview shows these so
 * the user sees something meaningful even when collapsed mid-stream.
 * Older revisions returned a single line, which made the preview
 * jump abruptly when the model emitted short tokens.
 */
export function lastLines(text: string, count: number): string {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return "";
  return lines.slice(-count).join(" · ");
}

/**
 * Models often emit a leading newline after `<think>` and an extra
 * blank line between the first thought and the rest, which renders
 * as a tall visual gap inside the reasoning panel. Trim leading
 * whitespace and collapse the very first paragraph break so the
 * panel reads as one continuous thought stream.
 */
export function tidyReasoningForDisplay(text: string): string {
  const trimmed = text.replace(/^[\s\n]+/, "");
  // Collapse the *first* `\n\n` (or longer) to a single newline so the
  // first paragraph sits flush against subsequent content. Mid-stream
  // paragraph breaks are preserved.
  return trimmed.replace(/^([^\n]+)\n{2,}/, "$1\n");
}

export function ReasoningPanel({ text, streaming = false }: ReasoningPanelProps) {
  const rawContent = text?.trim() ?? "";
  const content = tidyReasoningForDisplay(rawContent);
  // Default to *collapsed* during streaming so the user sees a compact
  // running preview instead of a wall of streaming thought. The user
  // can still expand explicitly; once expanded the choice sticks until
  // streaming ends. Pre-fix this auto-opened, which clashed with the
  // request for a 1-2 line streaming preview.
  const [open, setOpen] = useState(false);
  const prevStreamingRef = useRef(streaming);
  const userExpandedRef = useRef(false);

  // Reset auto-expand state whenever streaming starts again so the
  // next message starts collapsed.
  useEffect(() => {
    if (streaming && !prevStreamingRef.current) {
      userExpandedRef.current = false;
      setOpen(false);
    }
    prevStreamingRef.current = streaming;
  }, [streaming]);

  // Auto-collapse when streaming ends if the user never expanded —
  // matches the previous behaviour for the "thought trace landed"
  // moment where the user typically wants the answer, not the full
  // chain of thought, in front of them.
  useEffect(() => {
    if (!streaming && !userExpandedRef.current) {
      setOpen(false);
    }
  }, [streaming]);

  if (!content) return null;

  const handleToggle = () => {
    setOpen((current) => {
      const next = !current;
      if (next) userExpandedRef.current = true;
      return next;
    });
  };

  // Two-line preview when collapsed during streaming — gives the user
  // a real glimpse of the model's current train of thought without
  // committing the whole panel to display.
  const preview = !open && streaming ? lastLines(content, 2) : null;

  return (
    <div className={`reasoning-panel${open ? " reasoning-panel--open" : ""}${streaming ? " reasoning-panel--streaming" : ""}`}>
      <button
        type="button"
        className="reasoning-panel__toggle"
        onClick={handleToggle}
        aria-expanded={open}
      >
        <span className={`reasoning-panel__chevron${open ? " reasoning-panel__chevron--open" : ""}`}>›</span>
        <span className="reasoning-panel__label">{streaming ? "Thinking..." : "Thinking"}</span>
        {preview ? (
          <span className="reasoning-panel__preview" title={preview}>
            {preview}
          </span>
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
