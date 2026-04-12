import type { TabId } from "../types";

export const tabs: Array<{ id: TabId; label: string; caption: string }> = [
  { id: "dashboard", label: "Dashboard", caption: "System overview" },
  { id: "chat", label: "Chat", caption: "Local AI chat" },
  { id: "online-models", label: "Discover", caption: "Browse and download AI models" },
  { id: "my-models", label: "My Models", caption: "Models on this machine" },
  { id: "image-discover", label: "Image Discover", caption: "Browse image models" },
  { id: "image-models", label: "Image Models", caption: "Installed image generators" },
  { id: "image-studio", label: "Image Studio", caption: "Prompt, generate, and iterate" },
  { id: "image-gallery", label: "Image Gallery", caption: "Saved outputs and filters" },
  { id: "server", label: "Server", caption: "OpenAI-compatible local API" },
  { id: "benchmarks", label: "Benchmarks", caption: "Run a new benchmark" },
  { id: "benchmark-history", label: "History", caption: "Compare saved runs" },
  { id: "conversion", label: "Conversion", caption: "Convert models to MLX format" },
  { id: "logs", label: "Logs", caption: "Runtime events" },
  { id: "settings", label: "Settings", caption: "Directories and defaults" },
];
