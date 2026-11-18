import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_k: string, o?: { defaultValue?: string }) => o?.defaultValue ?? _k }),
}));

import { StatusBanners } from "../StatusBanners";

type Props = Parameters<typeof StatusBanners>[0];

function render(extra: Partial<Props>): string {
  const props = {
    gpuStatus: { showBanner: false, status: null, dismiss: () => undefined },
    cudaTorchResult: null,
    installingCudaTorch: false,
    handleInstallCudaTorch: async () => undefined,
    error: null,
    rebuildingLlama: false,
    handleRebuildLlamaCpp: async () => undefined,
    rebuildOutput: null,
    setRebuildOutput: () => undefined,
    ...extra,
  } as unknown as Props;
  return renderToStaticMarkup(<StatusBanners {...props} />);
}

describe("StatusBanners", () => {
  it("says the backend is not answering while it is offline", () => {
    const html = render({ backendOffline: true });
    expect(html).toContain("backend-offline-banner");
    expect(html).toContain('role="status"');
    expect(html).toContain("The backend is not answering.");
    expect(html).toContain("The app reconnects by itself.");
  });

  it("shows nothing about the backend while it is online", () => {
    expect(render({})).not.toContain("backend-offline-banner");
    expect(render({ backendOffline: false })).not.toContain("backend-offline-banner");
  });

  it("keeps the error banner beside the offline notice", () => {
    const html = render({ backendOffline: true, error: "Could not save." });
    expect(html).toContain("Could not save.");
    expect(html).toContain("backend-offline-banner");
  });
});
