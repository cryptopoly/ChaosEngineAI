import type { SidebarGroupId, TabId } from "../types";

export interface TabConfig {
  id: TabId;
  label: string;
  caption: string;
  group?: SidebarGroupId;
  shortLabel?: string;
}

export const tabs: TabConfig[] = [
  { id: "dashboard", label: "Dashboard", caption: "System overview" },
  { id: "chat", label: "Chat", caption: "Local AI chat" },
  { id: "online-models", label: "Discover", caption: "Browse and download AI models", group: "models", shortLabel: "Discover" },
  { id: "my-models", label: "My Models", caption: "Models on this machine", group: "models", shortLabel: "My Models" },
  { id: "image-discover", label: "Image Discover", caption: "Browse image models", group: "images", shortLabel: "Discover" },
  { id: "image-models", label: "Image Models", caption: "Installed image generators", group: "images", shortLabel: "My Models" },
  { id: "image-studio", label: "Image Studio", caption: "Prompt, generate, and iterate", group: "images", shortLabel: "Studio" },
  { id: "image-gallery", label: "Image Gallery", caption: "Saved outputs and filters", group: "images", shortLabel: "Gallery" },
  { id: "server", label: "Server", caption: "OpenAI-compatible local API" },
  { id: "benchmarks", label: "Benchmarks", caption: "Run a new benchmark", group: "benchmarks", shortLabel: "Run" },
  { id: "benchmark-history", label: "History", caption: "Compare saved runs", group: "benchmarks", shortLabel: "History" },
  { id: "conversion", label: "Conversion", caption: "Convert models to MLX format", group: "tools", shortLabel: "Conversion" },
  { id: "finetuning", label: "Fine-Tuning", caption: "LoRA adapters and training", group: "tools", shortLabel: "Fine-Tuning" },
  { id: "prompt-library", label: "Prompts", caption: "Reusable prompt templates", group: "tools", shortLabel: "Prompts" },
  { id: "plugins", label: "Plugins", caption: "Extensions and plugin system", group: "tools", shortLabel: "Plugins" },
  { id: "logs", label: "Logs", caption: "Runtime events" },
  { id: "settings", label: "Settings", caption: "Directories and defaults" },
];
