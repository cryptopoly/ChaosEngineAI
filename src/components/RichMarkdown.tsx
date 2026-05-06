import type { ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { CodeBlock } from "./CodeBlock";

interface RichMarkdownProps {
  children: string;
}

interface MarkdownCodeProps {
  inline?: boolean;
  className?: string;
  children?: ReactNode;
}

function extractLanguage(className?: string): string | undefined {
  if (!className) return undefined;
  const match = /language-([\w+-]+)/i.exec(className);
  return match?.[1];
}

function flattenChildren(children: ReactNode): string {
  if (children == null) return "";
  if (typeof children === "string") return children;
  if (typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(flattenChildren).join("");
  if (typeof children === "object") {
    const maybeElement = children as unknown as { props?: { children?: ReactNode } };
    if (maybeElement.props?.children !== undefined) {
      return flattenChildren(maybeElement.props.children);
    }
  }
  return "";
}

export function RichMarkdown({ children }: RichMarkdownProps) {
  return (
    <Markdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        code: ({ inline, className, children: codeChildren }: MarkdownCodeProps) => {
          const language = extractLanguage(className);
          const raw = flattenChildren(codeChildren);
          // react-markdown reports `inline` for backtick spans; absence of newline is also a strong hint
          const isInline = inline === true || (!language && !raw.includes("\n"));
          if (isInline) {
            return <code className={className}>{codeChildren}</code>;
          }
          return <CodeBlock code={raw} language={language} />;
        },
        // Avoid wrapping the CodeBlock in a default <pre> — CodeBlock owns its own container
        pre: ({ children: preChildren }: { children?: ReactNode }) => <>{preChildren}</>,
      }}
    >
      {children}
    </Markdown>
  );
}
