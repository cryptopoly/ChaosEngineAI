import { beforeEach, describe, expect, it, vi } from "vitest";

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));

vi.mock("@tauri-apps/api/core", () => ({
  invoke,
  isTauri: vi.fn(() => true),
}));

import type { TauriBackendInfo } from "../types";

function info(overrides: Partial<TauriBackendInfo> = {}): TauriBackendInfo {
  return {
    apiBase: "http://127.0.0.1:8876",
    port: 8876,
    managedByTauri: true,
    processRunning: false,
    started: false,
    startupError: null,
    ...overrides,
  };
}

// The module keeps its replies in module-level promises, so each test loads a
// fresh copy.
async function loadApi() {
  vi.resetModules();
  return import("./index");
}

describe("getTauriBackendInfo", () => {
  beforeEach(() => {
    invoke.mockReset();
  });

  it("reuses the first reply until a fresh read is forced", async () => {
    invoke.mockResolvedValue(info());
    const { getTauriBackendInfo } = await loadApi();

    await getTauriBackendInfo();
    await getTauriBackendInfo();

    expect(invoke).toHaveBeenCalledTimes(1);
  });

  it("reads the shell again when forced, so a dead sidecar shows up", async () => {
    invoke
      .mockResolvedValueOnce(info())
      .mockResolvedValueOnce(
        info({ startupError: "The backend sidecar exited with status signal: 9 (SIGKILL)." }),
      );
    const { getTauriBackendInfo } = await loadApi();

    expect((await getTauriBackendInfo())?.startupError).toBeNull();
    expect((await getTauriBackendInfo(true))?.startupError).toContain("exited with status");
    expect(invoke).toHaveBeenCalledTimes(2);
  });

  it("follows the sidecar to another port after a forced read", async () => {
    invoke
      .mockResolvedValueOnce(info())
      .mockResolvedValueOnce(info({ apiBase: "http://127.0.0.1:54321", port: 54321, processRunning: true }));
    const { getTauriBackendInfo, resolveApiBase } = await loadApi();

    expect(await resolveApiBase()).toBe("http://127.0.0.1:8876");
    await getTauriBackendInfo(true);

    expect(await resolveApiBase()).toBe("http://127.0.0.1:54321");
  });
});
