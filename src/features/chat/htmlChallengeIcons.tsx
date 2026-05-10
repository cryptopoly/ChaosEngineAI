/**
 * Inline SVG icons for the HTML Challenge tab toolbar / chrome.
 *
 * Each icon is monochrome, ``viewBox="0 0 24 24"``, with no stroke / fill
 * baked in — the parent CSS sets ``fill: currentColor`` /
 * ``stroke: currentColor`` so the icon picks up the surrounding text colour.
 */

export function OpenFileIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M4 6.5A2.5 2.5 0 0 1 6.5 4H10l2 2h5.5A2.5 2.5 0 0 1 20 8.5v9A2.5 2.5 0 0 1 17.5 20h-11A2.5 2.5 0 0 1 4 17.5z" />
      <path d="M8 13h8" />
      <path d="m13 10 3 3-3 3" />
    </svg>
  );
}

export function BrowserIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="9" />
      <path d="M3.6 9h16.8" />
      <path d="M3.6 15h16.8" />
      <path d="M12 3a13.5 13.5 0 0 1 0 18" />
      <path d="M12 3a13.5 13.5 0 0 0 0 18" />
    </svg>
  );
}

export function ExpandIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M8 3H3v5" />
      <path d="M3 3l7 7" />
      <path d="M16 3h5v5" />
      <path d="M21 3l-7 7" />
      <path d="M8 21H3v-5" />
      <path d="M3 21l7-7" />
      <path d="M16 21h5v-5" />
      <path d="M21 21l-7-7" />
    </svg>
  );
}

export function CollapseIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M10 3v7H3" />
      <path d="M3 10l7-7" />
      <path d="M14 3v7h7" />
      <path d="M21 10l-7-7" />
      <path d="M10 21v-7H3" />
      <path d="M3 14l7 7" />
      <path d="M14 21v-7h7" />
      <path d="M21 14l-7 7" />
    </svg>
  );
}
