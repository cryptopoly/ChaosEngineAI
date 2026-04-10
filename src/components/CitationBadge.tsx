import { useState } from "react";
import type { CitationInfo } from "../types";

interface CitationBadgeProps {
  citations: CitationInfo[];
}

export function CitationBadge({ citations }: CitationBadgeProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (!citations.length) return null;

  return (
    <div style={{ margin: "6px 0 2px", display: "flex", flexWrap: "wrap", gap: 4, alignItems: "flex-start" }}>
      <span style={{ color: "#5a6574", fontSize: 11, lineHeight: "22px" }}>Sources:</span>
      {citations.map((c, i) => (
        <span key={`${c.docId}-${c.chunkIndex}`} style={{ position: "relative" }}>
          <button
            type="button"
            onClick={() => setExpandedIndex(expandedIndex === i ? null : i)}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 4,
              padding: "2px 8px",
              borderRadius: 4,
              border: "1px solid #27303a",
              background: expandedIndex === i ? "#1e3a5f" : "#1a1f26",
              color: "#8fb4ff",
              cursor: "pointer",
              fontSize: 11,
              fontWeight: 600,
            }}
          >
            [{i + 1}] {c.docName}
          </button>
          {expandedIndex === i && (
            <div
              style={{
                position: "absolute",
                top: "100%",
                left: 0,
                zIndex: 20,
                marginTop: 4,
                padding: 10,
                borderRadius: 8,
                border: "1px solid #27303a",
                background: "#15191e",
                boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
                width: 320,
                maxHeight: 200,
                overflow: "auto",
                fontSize: 12,
                color: "#c8d0da",
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 4, color: "#8fb4ff" }}>
                {c.docName} {c.page != null ? `(page ${c.page})` : ""} - Chunk {c.chunkIndex + 1}
              </div>
              <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, color: "#9ea8b4" }}>
                {c.preview}
              </div>
            </div>
          )}
        </span>
      ))}
    </div>
  );
}
