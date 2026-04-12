export const CAPABILITY_META: Record<string, { shortLabel: string; title: string; icon: string; color: string }> = {
  agents: { shortLabel: "Agents", title: "Agent workflows", icon: "\uD83E\uDD16", color: "#c084fc" },
  chat: { shortLabel: "Chat", title: "General chat", icon: "\uD83D\uDCAC", color: "#8fb4ff" },
  coding: { shortLabel: "Code", title: "Coding support", icon: "\uD83D\uDCBB", color: "#34d399" },
  multilingual: { shortLabel: "Multi", title: "Multilingual support", icon: "\uD83C\uDF10", color: "#fbbf24" },
  reasoning: { shortLabel: "Reason", title: "Reasoning-focused", icon: "\uD83E\uDDE0", color: "#f472b6" },
  thinking: { shortLabel: "Think", title: "Thinking / deliberate reasoning", icon: "\uD83D\uDCA1", color: "#facc15" },
  "tool-use": { shortLabel: "Tools", title: "Tool use / function calling", icon: "\uD83D\uDD27", color: "#fb923c" },
  video: { shortLabel: "Video", title: "Video understanding", icon: "\uD83C\uDFA5", color: "#f87171" },
  vision: { shortLabel: "Vision", title: "Image / vision support", icon: "\uD83D\uDC41\uFE0F", color: "#22d3ee" },
};
