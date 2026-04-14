import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";

interface ReasoningPanelProps {
  text?: string | null;
  streaming?: boolean;
}

export function ReasoningPanel({ text, streaming = false }: ReasoningPanelProps) {
  const content = text?.trim() ?? "";
  const [open, setOpen] = useState(Boolean(content && streaming));
  const prevStreamingRef = useRef(streaming);
  const prevContentLengthRef = useRef(content.length);

  useEffect(() => {
    if (content.length > prevContentLengthRef.current) {
      setOpen(true);
    }
    prevContentLengthRef.current = content.length;
  }, [content.length]);

  useEffect(() => {
    if (prevStreamingRef.current && !streaming && content) {
      setOpen(false);
    }
    prevStreamingRef.current = streaming;
  }, [streaming, content]);

  if (!content) return null;

  return (
    <div className={`reasoning-panel${open ? " reasoning-panel--open" : ""}`}>
      <button
        type="button"
        className="reasoning-panel__toggle"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
      >
        <span className={`reasoning-panel__chevron${open ? " reasoning-panel__chevron--open" : ""}`}>›</span>
        <span>{streaming ? "Thinking..." : "Thinking"}</span>
      </button>
      {open ? (
        <div className="reasoning-panel__body">
          <div className="markdown-content reasoning-panel__content">
            <Markdown>{content}</Markdown>
          </div>
        </div>
      ) : null}
    </div>
  );
}
