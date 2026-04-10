import { check } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { ask, message } from "@tauri-apps/plugin-dialog";

// Run an update check against the GitHub Releases endpoint configured in
// tauri.conf.json. We prompt the user before downloading and again before
// relaunching, since the app holds local model state in memory.
export async function checkForUpdates(opts: { silent?: boolean } = {}): Promise<void> {
  try {
    const update = await check();
    if (!update?.available) {
      if (!opts.silent) {
        await message("You're on the latest version of ChaosEngineAI.", {
          title: "No updates",
          kind: "info",
        });
      }
      return;
    }

    const proceed = await ask(
      `Version ${update.version} is available (you're on ${update.currentVersion}).\n\nDownload and install now?`,
      { title: "ChaosEngineAI update available", kind: "info" },
    );
    if (!proceed) return;

    await update.downloadAndInstall();
    await relaunch();
  } catch (error) {
    if (!opts.silent) {
      await message(`Update check failed: ${error}`, {
        title: "Update error",
        kind: "error",
      });
    } else {
      // eslint-disable-next-line no-console
      console.warn("[updater] silent check failed", error);
    }
  }
}
