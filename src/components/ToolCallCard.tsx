import { useState } from "react";
import type { ToolCallInfo } from "../types";

interface ToolCallCardProps {
  toolCall: ToolCallInfo;
}

const TOOL_ICONS: Record<string, string> = {
  web_search: "search",
  calculator: "calc",
  code_executor: "code",
  file_reader: "file",
};

export function ToolCallCard({ toolCall }: ToolCallCardProps) {
  const [expanded, setExpanded] = useState(false);

  const icon = TOOL_ICONS[toolCall.name] ?? "tool";
  const argSummary = Object.entries(toolCall.arguments)
    .map(([k, v]) => {
      const str = typeof v === "string" ? v : JSON.stringify(v);
      return `${k}: ${str.length > 60 ? str.slice(0, 60) + "..." : str}`;
    })
    .join(", ");

  return (
    <div
      style={{
        margin: "8px 0",
        border: "1px solid #27303a",
        borderRadius: 8,
        background: "#1a1f26",
        overflow: "hidden",
      }}
    >
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          width: "100%",
          padding: "8px 12px",
          border: "none",
          background: "transparent",
          color: "#c8d0da",
          cursor: "pointer",
          textAlign: "left",
          fontSize: 13,
        }}
      >
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 24,
            height: 24,
            borderRadius: 6,
            background: "#2a3442",
            color: "#8fb4ff",
            fontSize: 10,
            fontWeight: 700,
            textTransform: "uppercase",
            flexShrink: 0,
          }}
        >
          {icon.slice(0, 4)}
        </span>
        <span style={{ fontWeight: 600, color: "#8fb4ff" }}>{toolCall.name}</span>
        <span style={{ color: "#7a8594", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {argSummary}
        </span>
        <span style={{ color: "#5a6574", fontSize: 11, flexShrink: 0 }}>
          {toolCall.elapsed.toFixed(1)}s
        </span>
        <span style={{ color: "#5a6574", fontSize: 14, flexShrink: 0 }}>
          {expanded ? "\u25B2" : "\u25BC"}
        </span>
      </button>

      {expanded && (
        <div style={{ padding: "0 12px 12px", fontSize: 12 }}>
          <div style={{ marginBottom: 8 }}>
            <div style={{ color: "#7a8594", marginBottom: 4, fontWeight: 600 }}>Input</div>
            <pre
              style={{
                background: "#0f1215",
                borderRadius: 6,
                padding: 8,
                margin: 0,
                color: "#c8d0da",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                maxHeight: 120,
                overflow: "auto",
              }}
            >
              {JSON.stringify(toolCall.arguments, null, 2)}
            </pre>
          </div>
          <div>
            <div style={{ color: "#7a8594", marginBottom: 4, fontWeight: 600 }}>Result</div>
            <pre
              style={{
                background: "#0f1215",
                borderRadius: 6,
                padding: 8,
                margin: 0,
                color: "#c8d0da",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                maxHeight: 200,
                overflow: "auto",
              }}
            >
              {toolCall.result}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
