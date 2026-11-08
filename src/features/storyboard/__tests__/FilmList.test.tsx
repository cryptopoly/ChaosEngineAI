import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { FilmListView, formatFilmSize } from "../FilmList";

describe("FilmListView", () => {
  it("lists films with a Reveal button under a Film heading", () => {
    const html = renderToStaticMarkup(
      <FilmListView
        films={[{ file: "exports/harbour.mp4", name: "harbour.mp4", bytes: 5 * 1024 * 1024, modifiedAt: 1 }]}
        revealLabel="Show in folder"
        onReveal={() => {}}
      />,
    );
    expect(html).toContain("Film");
    expect(html).toContain("harbour.mp4");
    expect(html).toContain("5.0 MB");
    expect(html).toContain("Show in folder");
  });

  it("renders nothing without films", () => {
    expect(renderToStaticMarkup(<FilmListView films={[]} revealLabel="x" onReveal={() => {}} />)).toBe("");
    // The Export tab asks for the empty state.
    expect(renderToStaticMarkup(<FilmListView films={[]} revealLabel="x" onReveal={() => {}} showEmpty />)).toContain("No films yet.");
  });

  it("formats sizes", () => {
    expect(formatFilmSize(10)).toBe("1 KB");
    expect(formatFilmSize(2048)).toBe("2 KB");
    expect(formatFilmSize(3 * 1024 ** 3)).toBe("3.0 GB");
  });
});
