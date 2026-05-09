/**
 * HTML markup helpers for the HTML Challenge tab — preview-doc assembly +
 * syntax-highlighting renderer for the code-view side panel.
 *
 * Pure string functions; no React state, no ``ChallengeSlotState``
 * dependencies. The matching iframe asset constants live in
 * ``htmlChallengePreviewAssets.ts``.
 */

import {
  htmlChallengePreviewBaseStyle,
  htmlChallengePreviewFitScript,
  htmlChallengePreviewKeyBridge,
  htmlChallengePreviewStorageShim,
  htmlChallengePreviewValidationBridge,
} from "./htmlChallengePreviewAssets";
import type { CompareTarget } from "./CompareView";


export function previewSrcDoc(html: string, slotId: CompareTarget) {
  const csp = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline';">`;
  // Storage shim must come BEFORE any model script so the very first access
  // to localStorage hits the stub instead of the throwing native binding.
  const injection = `${csp}${htmlChallengePreviewStorageShim}${htmlChallengePreviewBaseStyle}${htmlChallengePreviewKeyBridge}${htmlChallengePreviewValidationBridge(slotId)}${htmlChallengePreviewFitScript}`;
  if (/<head[^>]*>/i.test(html)) {
    return html.replace(/<head([^>]*)>/i, `<head$1>${injection}`);
  }
  return `${injection}${html}`;
}

export function escapeHtmlCode(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function highlightHtmlTag(token: string) {
  if (token.startsWith("&lt;!--")) {
    return `<span class="html-code-comment">${token}</span>`;
  }
  if (/^&lt;!doctype/i.test(token)) {
    return `<span class="html-code-doctype">${token}</span>`;
  }
  const match = token.match(/^(&lt;\/?)([A-Za-z][A-Za-z0-9:-]*)([\s\S]*?)(&gt;)$/);
  if (!match) return `<span class="html-code-tag">${token}</span>`;
  const [, open, name, attrs, close] = match;
  const highlightedAttrs = attrs.replace(
    /(\s+)([A-Za-z_:][A-Za-z0-9_.:-]*)(=)(&quot;.*?&quot;|&#39;.*?&#39;|[^\s&]+)?/g,
    (_attr, space, attrName, equals, attrValue = "") => (
      `${space}<span class="html-code-attr">${attrName}</span>`
      + `<span class="html-code-punct">${equals}</span>`
      + (attrValue ? `<span class="html-code-string">${attrValue}</span>` : "")
    ),
  );
  return (
    `<span class="html-code-punct">${open}</span>`
    + `<span class="html-code-tag-name">${name}</span>`
    + highlightedAttrs
    + `<span class="html-code-punct">${close}</span>`
  );
}

export function highlightHtmlCode(value: string) {
  const escaped = escapeHtmlCode(value);
  return escaped.replace(
    /(&lt;!--[\s\S]*?--&gt;|&lt;!doctype[\s\S]*?&gt;|&lt;\/?[A-Za-z][\s\S]*?&gt;)/gi,
    (token) => highlightHtmlTag(token),
  );
}
