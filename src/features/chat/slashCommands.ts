import type { ChatSession, ChatThinkingMode } from "../../types";
import { downloadExport, type ExportFormat } from "./exportThread";

export interface SlashCommand {
  /** Primary command string, e.g. "/clear" or "/think on" */
  command: string;
  /** Short description shown in the menu */
  description: string;
  /** Returns true when this command can run with the given args + context */
  isAvailable: (ctx: SlashCommandContext) => boolean;
  /** Execute the command. Returns true if the draft text should be cleared after running. */
  run: (ctx: SlashCommandContext) => boolean;
}

export interface SlashCommandContext {
  args: string;
  activeChat: ChatSession | undefined;
  loadedModelRef: string | undefined;
  enableTools: boolean;
  chatBusySessionId: string | null;
  onClearDraft: () => void;
  onThinkingModeChange: (mode: ChatThinkingMode) => void;
  onToggleTools: (enabled: boolean) => void;
  onOpenModelSelector: (action: "chat" | "server" | "thread", preselectedKey?: string) => void;
  onCancelGeneration: () => void;
  activeThreadOptionKey?: string;
}

export const SLASH_COMMANDS: SlashCommand[] = [
  {
    command: "/clear",
    description: "Clear the draft message and any pending images",
    isAvailable: () => true,
    run: (ctx) => {
      ctx.onClearDraft();
      return false;
    },
  },
  {
    command: "/think on",
    description: "Use the model's default reasoning behavior",
    isAvailable: () => true,
    run: (ctx) => {
      ctx.onThinkingModeChange("auto");
      return true;
    },
  },
  {
    command: "/think off",
    description: "Bias the thread toward direct answers (no thinking)",
    isAvailable: () => true,
    run: (ctx) => {
      ctx.onThinkingModeChange("off");
      return true;
    },
  },
  {
    command: "/tools on",
    description: "Enable agent tools (web search, code, calculator, file reader)",
    isAvailable: () => true,
    run: (ctx) => {
      ctx.onToggleTools(true);
      return true;
    },
  },
  {
    command: "/tools off",
    description: "Disable agent tools for this thread",
    isAvailable: () => true,
    run: (ctx) => {
      ctx.onToggleTools(false);
      return true;
    },
  },
  {
    command: "/model",
    description: "Open the model selector",
    isAvailable: () => true,
    run: (ctx) => {
      ctx.onOpenModelSelector("chat", ctx.activeThreadOptionKey);
      return true;
    },
  },
  {
    command: "/cancel",
    description: "Stop the current generation",
    isAvailable: (ctx) => ctx.chatBusySessionId !== null,
    run: (ctx) => {
      ctx.onCancelGeneration();
      return true;
    },
  },
  {
    command: "/export md",
    description: "Export this thread as Markdown",
    isAvailable: (ctx) => Boolean(ctx.activeChat && ctx.activeChat.messages.length > 0),
    run: (ctx) => {
      if (ctx.activeChat) downloadExport(ctx.activeChat, "md");
      return true;
    },
  },
  {
    command: "/export json",
    description: "Export this thread as JSON",
    isAvailable: (ctx) => Boolean(ctx.activeChat && ctx.activeChat.messages.length > 0),
    run: (ctx) => {
      if (ctx.activeChat) downloadExport(ctx.activeChat, "json");
      return true;
    },
  },
  {
    command: "/export txt",
    description: "Export this thread as plain text",
    isAvailable: (ctx) => Boolean(ctx.activeChat && ctx.activeChat.messages.length > 0),
    run: (ctx) => {
      if (ctx.activeChat) downloadExport(ctx.activeChat, "txt");
      return true;
    },
  },
];

/**
 * If the draft is a slash command (starts with "/" and contains no newline),
 * return the matching commands ranked by prefix match.  Returns an empty
 * array when the draft is not a slash command.
 */
export function matchSlashCommands(draft: string, ctx: SlashCommandContext): SlashCommand[] {
  if (!draft.startsWith("/")) return [];
  if (draft.includes("\n")) return [];
  const lower = draft.toLowerCase().trim();
  return SLASH_COMMANDS.filter((cmd) => {
    if (!cmd.isAvailable(ctx)) return false;
    return cmd.command.startsWith(lower) || lower.startsWith(cmd.command);
  });
}

/** Find an exact command match for a draft, ignoring trailing whitespace. */
export function findExactCommand(draft: string): SlashCommand | undefined {
  const trimmed = draft.trim().toLowerCase();
  return SLASH_COMMANDS.find((cmd) => cmd.command === trimmed);
}

export type { ExportFormat };
