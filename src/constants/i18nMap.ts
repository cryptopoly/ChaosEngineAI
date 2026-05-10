// Map tab IDs to their translation keys in zh-CN.ts
// Keys follow the pattern: tab.<id>, group.<id>

export const TAB_I18N_KEY: Record<string, string> = {
  dashboard: "tab.dashboard",
  chat: "tab.chat",
  "chat-compare": "tab.compare",
  "html-challenge": "tab.htmlChallenge",
  "my-models": "tab.myModels",
  "online-models": "tab.discover",
  "image-models": "tab.imageModels",
  "image-discover": "tab.imageDiscover",
  "image-studio": "tab.imageStudio",
  "image-gallery": "tab.imageGallery",
  "video-models": "tab.videoModels",
  "video-discover": "tab.videoDiscover",
  "video-studio": "tab.videoStudio",
  "video-gallery": "tab.videoGallery",
  server: "tab.server",
  benchmarks: "tab.benchmarks",
  "benchmark-history": "tab.benchmarkHistory",
  conversion: "tab.conversion",
  finetuning: "tab.fineTuning",
  "prompt-library": "tab.promptLibrary",
  plugins: "tab.plugins",
  logs: "tab.logs",
  settings: "tab.settings",
};

export const GROUP_I18N_KEY: Record<string, string> = {
  chat: "group.chat",
  images: "group.images",
  video: "group.video",
  benchmarks: "group.benchmarks",
  tools: "group.tools",
};
