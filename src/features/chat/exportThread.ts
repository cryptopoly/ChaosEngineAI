import type { ChatSession, ChatMessage } from "../../types";

export type ExportFormat = "md" | "json" | "txt";

const SAFE_FILENAME = /[^a-zA-Z0-9._-]+/g;

function safeFilename(title: string, format: ExportFormat): string {
  const base = (title || "chat").trim().replace(SAFE_FILENAME, "_").slice(0, 64) || "chat";
  return `${base}.${format}`;
}

function stamp(): string {
  return new Date().toISOString();
}

function renderMessageMarkdown(message: ChatMessage): string {
  const role = message.role === "user" ? "User" : "Assistant";
  const parts: string[] = [`## ${role}`, ""];
  if (message.reasoning && message.role === "assistant") {
    parts.push("<details><summary>Reasoning</summary>", "", message.reasoning, "", "</details>", "");
  }
  parts.push(message.text || "_(empty)_", "");
  if (message.toolCalls?.length) {
    parts.push("**Tool calls:**", "");
    for (const tc of message.toolCalls) {
      parts.push(`- \`${tc.name}\` — ${typeof tc.arguments === "string" ? tc.arguments : JSON.stringify(tc.arguments)}`);
    }
    parts.push("");
  }
  if (message.citations?.length) {
    parts.push("**Citations:**", "");
    for (const cit of message.citations) {
      const pageRef = cit.page != null ? ` p.${cit.page}` : "";
      parts.push(`- ${cit.docName}${pageRef} (chunk ${cit.chunkIndex})`);
    }
    parts.push("");
  }
  return parts.join("\n");
}

function renderMessageTxt(message: ChatMessage): string {
  const role = message.role === "user" ? "USER" : "ASSISTANT";
  const lines: string[] = [`--- ${role} ---`];
  if (message.reasoning && message.role === "assistant") {
    lines.push("[reasoning]", message.reasoning, "[/reasoning]");
  }
  lines.push(message.text || "");
  return lines.join("\n");
}

export function buildMarkdown(session: ChatSession): string {
  const header = [
    `# ${session.title || "Untitled chat"}`,
    "",
    `- **Model:** ${session.model || "Unknown"}`,
    `- **Updated:** ${session.updatedAt || ""}`,
    `- **Exported:** ${stamp()}`,
    "",
    "---",
    "",
  ];
  const body = session.messages.map(renderMessageMarkdown).join("\n");
  return header.join("\n") + body;
}

export function buildJson(session: ChatSession): string {
  const payload = {
    exportedAt: stamp(),
    session,
  };
  return JSON.stringify(payload, null, 2);
}

export function buildTxt(session: ChatSession): string {
  const header = [
    `${session.title || "Untitled chat"}`,
    `Model: ${session.model || "Unknown"}`,
    `Updated: ${session.updatedAt || ""}`,
    `Exported: ${stamp()}`,
    "",
  ];
  const body = session.messages.map(renderMessageTxt).join("\n\n");
  return header.join("\n") + body;
}

export function buildExportContent(session: ChatSession, format: ExportFormat): { content: string; filename: string; mime: string } {
  switch (format) {
    case "md":
      return {
        content: buildMarkdown(session),
        filename: safeFilename(session.title, "md"),
        mime: "text/markdown;charset=utf-8",
      };
    case "json":
      return {
        content: buildJson(session),
        filename: safeFilename(session.title, "json"),
        mime: "application/json;charset=utf-8",
      };
    case "txt":
    default:
      return {
        content: buildTxt(session),
        filename: safeFilename(session.title, "txt"),
        mime: "text/plain;charset=utf-8",
      };
  }
}

export function downloadExport(session: ChatSession, format: ExportFormat): void {
  const { content, filename, mime } = buildExportContent(session, format);
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  // Defer revoke so the browser has time to start the download
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
