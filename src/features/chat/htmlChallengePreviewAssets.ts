/**
 * HTML Challenge preview iframe assets — multi-line ``<script>`` / ``<style>``
 * payloads injected into the sandboxed preview iframe.
 *
 * Extracted from ``HtmlChallengeTab.tsx`` to keep the React component focused
 * on render logic. These four constants + one helper sum to ~390 LOC of
 * inert template strings + a ``CompareTarget``-parameterised validation
 * bridge — pure assets, no React state.
 */

import type { CompareTarget } from "./CompareView";


// Sandboxed iframes (sandbox="allow-scripts" without allow-same-origin) run
// in an opaque origin, so any access to localStorage / sessionStorage / cookies
// throws SecurityError and aborts the page's init script. Many LLM-generated
// pages use localStorage for high-scores or settings, which silently kills the
// rest of the script (event listeners never get attached). Install in-memory
// stubs as the very first thing in the document so model code finds working
// APIs and never throws.
export const htmlChallengePreviewStorageShim = `
<script>
(function () {
  function makeStorage() {
    var data = {};
    return {
      getItem: function (key) { return Object.prototype.hasOwnProperty.call(data, String(key)) ? data[String(key)] : null; },
      setItem: function (key, value) { data[String(key)] = String(value); },
      removeItem: function (key) { delete data[String(key)]; },
      clear: function () { data = {}; },
      key: function (index) { var keys = Object.keys(data); return index >= 0 && index < keys.length ? keys[index] : null; },
      get length() { return Object.keys(data).length; }
    };
  }
  function safeDefine(name) {
    try {
      var current = window[name];
      // If access already throws, descriptor lookup fails silently. Always
      // define our own to be safe.
      void current;
    } catch (_) { /* fall through to define */ }
    try {
      Object.defineProperty(window, name, {
        configurable: true,
        get: function () { return store; }
      });
      var store = makeStorage();
    } catch (_) {
      try { window[name] = makeStorage(); } catch (__) {}
    }
  }
  safeDefine("localStorage");
  safeDefine("sessionStorage");
})();
</script>`;

// Minimal preview-frame styling. Uses :where() (zero specificity) so any
// rule the model writes wins. Goal: when the model sets a page background,
// it covers the iframe; when it doesn't, the iframe falls through to the
// frame-shell's neutral colour rather than showing browser-default white.
// Scrollbar styles use the model-overridable `:where()` form too so the
// page can swap them out, but default to a transparent track that matches
// the frame chrome instead of opaque WebKit white.
export const htmlChallengePreviewBaseStyle = `
<style id="chaosengine-preview-base">
  :where(html, body) { margin: 0; padding: 0; background: transparent; }
  :where(body) { min-height: 100vh; }
  :where(html) {
    scrollbar-width: thin;
    scrollbar-color: rgba(255, 255, 255, 0.18) transparent;
  }
  :where(*::-webkit-scrollbar) { width: 8px; height: 8px; }
  :where(*::-webkit-scrollbar-track) { background: transparent; }
  :where(*::-webkit-scrollbar-thumb) {
    background: rgba(255, 255, 255, 0.18);
    border-radius: 4px;
  }
  :where(*::-webkit-scrollbar-thumb:hover) { background: rgba(255, 255, 255, 0.32); }
</style>`;

// After the page loads, measure its natural extent and zoom-to-fit the
// iframe viewport. Mirrors how a desktop browser would let you "reset
// zoom" on a page that's larger than the window — content shrinks to fit
// while preserving aspect ratio, no scrollbars, no clipping.
//
// Skipped when:
//   * The page already fits within ~5% of the viewport (avoids futile
//     fractional zooms that smear text).
//   * Scaling would invert (page smaller than viewport — leave at 1x so
//     fonts stay crisp).
export const htmlChallengePreviewFitScript = `
<script>
(function () {
  var lastZoom = 1;
  // Walk visible descendants to find the true rendered bounds. Pages that
  // use \`body { overflow: hidden }\` with an oversized canvas/svg inside
  // report \`scrollWidth\` == viewport even though content extends past
  // the viewport — relying on scrollWidth alone leaves them cropped.
  function measureContentBounds() {
    var maxRight = 0;
    var maxBottom = 0;
    var TAG_SKIP = { SCRIPT: 1, STYLE: 1, NOSCRIPT: 1, TEMPLATE: 1, LINK: 1, META: 1, HEAD: 1 };
    function walk(node, depth) {
      if (!node || node.nodeType !== 1) return;
      if (TAG_SKIP[node.tagName]) return;
      try {
        var style = window.getComputedStyle(node);
        if (style.display === "none" || style.visibility === "hidden") return;
        var rect = node.getBoundingClientRect();
        if (rect.right > maxRight) maxRight = rect.right;
        if (rect.bottom > maxBottom) maxBottom = rect.bottom;
      } catch (_) {}
      // Cap traversal depth to keep cost bounded on huge DOMs.
      if (depth > 6) return;
      for (var i = 0; i < node.children.length; i += 1) {
        walk(node.children[i], depth + 1);
      }
    }
    walk(document.body, 0);
    return { width: maxRight, height: maxBottom };
  }
  function fit() {
    if (!document.documentElement || !document.body) return;
    document.documentElement.style.zoom = "";
    var bounds = measureContentBounds();
    var contentWidth = Math.max(
      bounds.width,
      document.documentElement.scrollWidth,
      document.body.scrollWidth,
      document.body.offsetWidth
    );
    var contentHeight = Math.max(
      bounds.height,
      document.documentElement.scrollHeight,
      document.body.scrollHeight,
      document.body.offsetHeight
    );
    var vw = window.innerWidth;
    var vh = window.innerHeight;
    if (contentWidth <= 0 || contentHeight <= 0 || vw <= 0 || vh <= 0) return;
    var scale = Math.min(vw / contentWidth, vh / contentHeight);
    if (!isFinite(scale) || scale <= 0) return;
    // Clamp range. Don't shrink more than 4x (illegible) or grow more than 2x.
    if (scale < 0.25) scale = 0.25;
    if (scale > 2) scale = 2;
    if (Math.abs(scale - 1) < 0.04) {
      document.documentElement.style.zoom = "";
      lastZoom = 1;
      return;
    }
    if (Math.abs(scale - lastZoom) < 0.01) return;
    document.documentElement.style.zoom = String(scale);
    lastZoom = scale;
  }
  function schedule() {
    window.requestAnimationFrame(function () { window.requestAnimationFrame(fit); });
  }
  window.addEventListener("load", function () {
    schedule();
    window.setTimeout(schedule, 200);
    window.setTimeout(schedule, 800);
  });
  window.addEventListener("resize", schedule);
  if (typeof ResizeObserver !== "undefined") {
    try {
      var ro = new ResizeObserver(schedule);
      ro.observe(document.documentElement);
      ro.observe(document.body);
    } catch (_) {}
  }
})();
</script>`;

export const htmlChallengePreviewKeyBridge = `
<script>
(function () {
  function isStartKey(data) {
    var key = String(data.key || "").toLowerCase();
    var code = String(data.code || "").toLowerCase();
    return key === " " || key === "spacebar" || key === "enter" || code === "space" || code === "enter";
  }

  function isVisible(element) {
    if (!element || typeof element.getBoundingClientRect !== "function") return false;
    var style = window.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity || "1") === 0) return false;
    var rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function hasStartPrompt() {
    var text = ((document.body && (document.body.innerText || document.body.textContent)) || "").toLowerCase();
    return text.indexOf("press space") !== -1 ||
      text.indexOf("press enter") !== -1 ||
      text.indexOf("space to start") !== -1 ||
      text.indexOf("enter to start") !== -1 ||
      text.indexOf("space or enter") !== -1;
  }

  function runStartFallback(data) {
    if ((data.type || "keydown") !== "keydown" || !isStartKey(data) || !hasStartPrompt()) return;
    window.setTimeout(function () {
      var names = ["startGame", "beginGame", "playGame", "restartGame"];
      for (var i = 0; i < names.length; i += 1) {
        var candidate = window[names[i]];
        if (typeof candidate !== "function") continue;
        try {
          candidate();
          return;
        } catch (_) {}
      }

      var controls = Array.prototype.slice.call(document.querySelectorAll("button, [role='button'], input[type='button'], input[type='submit']"));
      for (var j = 0; j < controls.length; j += 1) {
        var control = controls[j];
        var label = String(control.innerText || control.textContent || control.value || "").toLowerCase();
        if (!isVisible(control) || !/(start|play|resume|restart)/.test(label)) continue;
        try {
          control.click();
          return;
        } catch (_) {}
      }
    }, 0);
  }

  window.addEventListener("message", function (event) {
    var data = event.data;
    if (!data || data.__htmlChallengePreviewKey !== true) return;
    var init = {
      key: data.key,
      code: data.code,
      keyCode: data.keyCode,
      which: data.which,
      bubbles: true,
      cancelable: true,
      repeat: Boolean(data.repeat),
      altKey: Boolean(data.altKey),
      ctrlKey: Boolean(data.ctrlKey),
      metaKey: Boolean(data.metaKey),
      shiftKey: Boolean(data.shiftKey)
    };
    var targets = [window, document, document.activeElement || document.body, document.body];
    var cancelled = false;
    targets.forEach(function (target) {
      if (!target || typeof target.dispatchEvent !== "function") return;
      try {
        var keyEvent = new KeyboardEvent(data.type || "keydown", init);
        if (target.dispatchEvent(keyEvent) === false || keyEvent.defaultPrevented) cancelled = true;
      } catch (_) {}
    });
    if (!cancelled) runStartFallback(data);
  });
})();
</script>`;

export function htmlChallengePreviewValidationBridge(slotId: CompareTarget) {
  return `
<script>
(function () {
  var slotId = ${JSON.stringify(slotId)};
  var lastStatus = "";

  function post(status, message) {
    if (status === lastStatus && status !== "script-error") return;
    lastStatus = status;
    window.parent.postMessage({
      __htmlChallengePreviewValidation: true,
      slotId: slotId,
      status: status,
      message: message || ""
    }, "*");
  }

  function errorMessage(event) {
    if (event && event.message) return event.message;
    if (event && event.error && event.error.message) return event.error.message;
    return "Script error";
  }

  window.addEventListener("error", function (event) {
    post("script-error", errorMessage(event));
  });
  window.addEventListener("unhandledrejection", function (event) {
    var reason = event.reason;
    post("script-error", reason && reason.message ? reason.message : String(reason || "Unhandled promise rejection"));
  });

  function canvasHasSignal(canvas) {
    if (!canvas || canvas.width <= 0 || canvas.height <= 0) return false;
    var context = null;
    try {
      context = canvas.getContext("2d", { willReadFrequently: true });
    } catch (_) {
      return true;
    }
    if (!context) return true;
    try {
      var sampleWidth = Math.min(32, canvas.width);
      var sampleHeight = Math.min(32, canvas.height);
      var scratch = document.createElement("canvas");
      scratch.width = sampleWidth;
      scratch.height = sampleHeight;
      var scratchContext = scratch.getContext("2d", { willReadFrequently: true });
      if (!scratchContext) return true;
      scratchContext.drawImage(canvas, 0, 0, sampleWidth, sampleHeight);
      var pixels = scratchContext.getImageData(0, 0, sampleWidth, sampleHeight).data;
      if (pixels.length < 4) return false;
      var firstR = pixels[0];
      var firstG = pixels[1];
      var firstB = pixels[2];
      var firstA = pixels[3];
      var hasOpaquePixel = firstA > 0;
      for (var index = 4; index < pixels.length; index += 4) {
        if (pixels[index + 3] > 0) hasOpaquePixel = true;
        if (
          Math.abs(pixels[index] - firstR) > 3 ||
          Math.abs(pixels[index + 1] - firstG) > 3 ||
          Math.abs(pixels[index + 2] - firstB) > 3 ||
          Math.abs(pixels[index + 3] - firstA) > 3
        ) {
          return true;
        }
      }
      return hasOpaquePixel && firstA > 0 && (firstR > 8 || firstG > 8 || firstB > 8);
    } catch (_) {
      return true;
    }
  }

  function hasVisibleElement() {
    if (!document.body) return false;
    var selector = "svg,img,video,button,input,textarea,select,a,p,h1,h2,h3,h4,h5,h6,main,section,article,div,span";
    var nodes = Array.prototype.slice.call(document.body.querySelectorAll(selector));
    return nodes.some(function (node) {
      var style = window.getComputedStyle(node);
      if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity || "1") === 0) return false;
      var rect = node.getBoundingClientRect();
      if (rect.width < 4 || rect.height < 4) return false;
      var text = (node.innerText || node.textContent || "").trim();
      if (text) return true;
      return style.backgroundColor && style.backgroundColor !== "rgba(0, 0, 0, 0)";
    });
  }

  function scan() {
    if (lastStatus === "script-error") return;
    var body = document.body;
    if (!body) {
      post("blank-render", "No document body rendered.");
      return;
    }
    var text = (body.innerText || body.textContent || "").trim();
    var canvases = Array.prototype.slice.call(document.querySelectorAll("canvas"));
    var canvasSignal = canvases.some(canvasHasSignal);
    if (!text && !canvasSignal && !hasVisibleElement()) {
      post("blank-render", "Preview rendered without visible content.");
      return;
    }
    post("valid-runtime", "");
  }

  // Detect the page's actual background colour and report it to the parent
  // so the frame-shell margin matches the content. Walks body -> html ->
  // body's largest top-level child until we find a non-transparent colour.
  function isOpaqueColor(color) {
    if (!color) return false;
    if (color === "transparent") return false;
    var match = /rgba?\(([^)]+)\)/i.exec(color);
    if (!match) return color !== "transparent";
    var parts = match[1].split(",").map(function (part) { return parseFloat(part.trim()); });
    if (parts.length < 4) return true;
    return parts[3] > 0.05;
  }
  function detectBackground() {
    if (!document.body || !document.documentElement) return;
    var bodyBg = window.getComputedStyle(document.body).backgroundColor;
    var htmlBg = window.getComputedStyle(document.documentElement).backgroundColor;
    var color = isOpaqueColor(bodyBg) ? bodyBg : (isOpaqueColor(htmlBg) ? htmlBg : "");
    if (!color) {
      // Fall back to the largest visible top-level child's bg.
      var children = Array.prototype.slice.call(document.body.children || []);
      for (var i = 0; i < children.length; i += 1) {
        var rect = children[i].getBoundingClientRect();
        if (rect.width < 200 || rect.height < 200) continue;
        var bg = window.getComputedStyle(children[i]).backgroundColor;
        if (isOpaqueColor(bg)) { color = bg; break; }
      }
    }
    if (!color) return;
    window.parent.postMessage({
      __htmlChallengePreviewBackground: true,
      slotId: slotId,
      color: color
    }, "*");
  }

  window.addEventListener("load", function () {
    window.setTimeout(scan, 300);
    window.setTimeout(scan, 1200);
    window.setTimeout(detectBackground, 350);
    window.setTimeout(detectBackground, 1300);
  });
  window.setTimeout(scan, 1600);
  window.setTimeout(detectBackground, 1700);
})();
</script>`;
}
