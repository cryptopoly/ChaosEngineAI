import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { TERMS_VERSION, formatAcceptedDate, termsNeedAcceptance } from "../termsLogic";
import { TermsDialog, TermsGate, TermsOfUsePanel } from "../TermsOfUse";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

const noop = () => {};

describe("termsNeedAcceptance", () => {
  it("waits for settings to load", () => {
    expect(termsNeedAcceptance(null)).toBe(false);
    expect(termsNeedAcceptance(undefined)).toBe(false);
  });

  it("asks when never accepted or when an older version was accepted", () => {
    expect(termsNeedAcceptance({})).toBe(true);
    expect(termsNeedAcceptance({ termsAcceptedVersion: 0 })).toBe(true);
    expect(termsNeedAcceptance({ termsAcceptedVersion: TERMS_VERSION - 1 })).toBe(true);
  });

  it("stays quiet once the current version is accepted", () => {
    expect(termsNeedAcceptance({ termsAcceptedVersion: TERMS_VERSION })).toBe(false);
    expect(termsNeedAcceptance({ termsAcceptedVersion: TERMS_VERSION + 1 })).toBe(false);
  });
});

describe("formatAcceptedDate", () => {
  it("formats an ISO time and ignores junk", () => {
    expect(formatAcceptedDate("2026-11-07T10:00:00Z", "en-GB")).toMatch(/2026/);
    expect(formatAcceptedDate("", "en")).toBe("");
    expect(formatAcceptedDate(undefined, "en")).toBe("");
    expect(formatAcceptedDate("not a date", "en")).toBe("");
  });
});

describe("terms screens", () => {
  it("shows the first-run dialog with the points, I agree and Quit", () => {
    const html = renderToStaticMarkup(<TermsGate settings={{ termsAcceptedVersion: 0 }} onAgree={noop} onQuit={noop} />);
    expect(html).toContain("Terms of use");
    expect(html).toContain("I agree");
    expect(html).toContain("Quit");
    expect(html).toContain("EU AI Act");
    expect(html).toContain("consent");
    expect(html).toContain("licence");
  });

  it("renders nothing when the current version is accepted", () => {
    expect(renderToStaticMarkup(<TermsGate settings={{ termsAcceptedVersion: TERMS_VERSION }} onAgree={noop} onQuit={noop} />)).toBe("");
  });

  it("the read-only dialog has Close and no agree button", () => {
    const html = renderToStaticMarkup(<TermsDialog mode="read" onClose={noop} />);
    expect(html).toContain("Close");
    expect(html).not.toContain("I agree");
  });

  it("the Settings row has a read button and shows when it was accepted", () => {
    const html = renderToStaticMarkup(
      <TermsOfUsePanel settings={{ termsAcceptedVersion: 1, termsAcceptedAt: "2026-11-07T10:00:00Z" }} onRead={noop} />,
    );
    expect(html).toContain("Accepted on");
    expect(html).toContain("Read the terms");
    expect(renderToStaticMarkup(<TermsOfUsePanel settings={{}} onRead={noop} />)).toContain("Not accepted yet");
  });
});
