import { useEffect, useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface CodeBlockProps {
  code: string;
  language?: string;
}

const COPY_RESET_MS = 1500;

export function CodeBlock({ code, language }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const lang = (language ?? "").toLowerCase().trim();
  const displayLang = lang || "text";

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), COPY_RESET_MS);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
    } catch {
      // Clipboard unavailable; silently no-op
    }
  };

  return (
    <div className="code-block">
      <div className="code-block__toolbar">
        <span className="code-block__lang">{displayLang}</span>
        <button
          type="button"
          className="code-block__copy"
          onClick={handleCopy}
          aria-label={copied ? "Copied" : "Copy code"}
        >
          {copied ? (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          ) : (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
            </svg>
          )}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <SyntaxHighlighter
        language={lang || "text"}
        style={oneDark}
        customStyle={{
          margin: 0,
          padding: "12px 14px",
          background: "#0a0d11",
          fontSize: "0.82rem",
          lineHeight: 1.5,
          borderBottomLeftRadius: 8,
          borderBottomRightRadius: 8,
          borderTopLeftRadius: 0,
          borderTopRightRadius: 0,
        }}
        codeTagProps={{
          style: {
            fontFamily: "SF Mono, SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, monospace",
          },
        }}
        PreTag="div"
      >
        {code.replace(/\n$/, "")}
      </SyntaxHighlighter>
    </div>
  );
}
