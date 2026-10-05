import { describe, expect, it } from "vitest";

import type { TauriBackendInfo } from "../../types";
import { startupFailure, startupNote } from "../StartupProgressPanel";

function makeInfo(overrides: Partial<TauriBackendInfo> = {}): TauriBackendInfo {
  return {
    apiBase: "http://127.0.0.1:8876",
    port: 8876,
    managedByTauri: true,
    processRunning: true,
    started: false,
    startupError: null,
    ...overrides,
  };
}

describe("startupFailure / startupNote", () => {
  it("reports nothing before the shell has anything to say", () => {
    expect(startupFailure(null)).toBeNull();
    expect(startupNote(null)).toBeNull();
    expect(startupFailure(makeInfo())).toBeNull();
    expect(startupNote(makeInfo())).toBeNull();
  });

  it("calls a message on a dead sidecar a failure", () => {
    const info = makeInfo({
      processRunning: false,
      startupError: "The backend sidecar exited with status signal: 9 (SIGKILL).",
    });
    expect(startupFailure(info)).toBe("The backend sidecar exited with status signal: 9 (SIGKILL).");
    expect(startupNote(info)).toBeNull();
  });

  it("keeps waiting while the sidecar still runs", () => {
    const info = makeInfo({
      processRunning: true,
      startupError: "The backend sidecar has not answered yet and may still be starting.",
    });
    expect(startupFailure(info)).toBeNull();
    expect(startupNote(info)).toBe("The backend sidecar has not answered yet and may still be starting.");
  });

  it("treats a missing processRunning flag as a stopped sidecar", () => {
    const info = makeInfo({ processRunning: undefined, startupError: "Failed to start the backend sidecar: denied" });
    expect(startupFailure(info)).toBe("Failed to start the backend sidecar: denied");
  });
});
