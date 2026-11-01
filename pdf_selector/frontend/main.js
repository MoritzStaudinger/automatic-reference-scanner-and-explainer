// PDF viewer — vanilla JS using PDF.js, with native scroll across all pages.
//
// Behaviour:
//   1. On first PDF load, render every page (canvas + text layer + annotation
//      layer) inside a scrollable wrap.  Pages stay mounted from then on so
//      scrolling is buttery-smooth.
//   2. An IntersectionObserver watches each page; whichever page covers the
//      most of the viewport is the "current" page.  When it changes we emit
//      a {kind:"page"} event back to Python so the ref-list on the left
//      filters to that page.  Debounced 200 ms so a fast scroll doesn't
//      flood Streamlit with reruns.
//   3. When only annotations change (e.g. user adopts an orphan ref), we
//      do NOT re-render canvases.  We just clear and re-paint annotation
//      layers — visually instant.
//   4. Text-layer painting is manual (PDF.js v4 TextLayer class was
//      flaky); spans are transparent but selectable.


// Vendored copy (see vendor/README.md) — no third-party CDN at runtime.
async function loadPdfJs() {
  if (window._pdfjs) return window._pdfjs;
  const base = new URL("./vendor/", import.meta.url);
  const mod = await import(new URL("pdf.min.js", base).href);
  mod.GlobalWorkerOptions.workerSrc = new URL("pdf.worker.min.js", base).href;
  window._pdfjs = mod;
  return mod;
}

const state = {
  pdfDoc: null,
  pdfHash: null,
  annotations: [],
  pageNum: 1,
  totalPages: 0,
  pageElements: [],          // { el, annotLayer, viewport }
  pageObserver: null,
  pendingScrollToPage: null,
  selectionCounter: 0,
};

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function quickHash(bytes) {
  const len = bytes.length;
  let h = len * 2654435761;
  const step = Math.max(1, Math.floor(len / 256));
  for (let i = 0; i < len; i += step) {
    h = (h ^ bytes[i]) * 16777619;
    h |= 0;
  }
  return `${len}:${h >>> 0}`;
}

function selectionPayload() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return null;
  const text = sel.toString().trim();
  if (text.length < 5) return null;

  let node = sel.anchorNode;
  while (node && (!node.dataset || !node.dataset.pageNum)) {
    node = node.parentNode;
  }
  const page = node ? parseInt(node.dataset.pageNum, 10) : state.pageNum;

  // Convert client (browser CSS) rects → PDF native coords so the
  // annotation overlay can paint a highlight rectangle for this
  // selection.  Without this, the manual-ref's `rect` would be in
  // window-relative pixels and `annotViewportRect` would produce
  // nonsense.
  //
  // We take the union bounding box of all client rects (multi-line
  // selections span several) — one tight rect per manual ref is what
  // the rest of the pipeline expects.
  let pageInfo = state.pageElements.find(pi => pi.pageNum === page);
  const clientRects = Array.from(sel.getRangeAt(0).getClientRects());

  // Fallback: if the anchor-node walk didn't land us on a known page
  // (can happen when the anchor is inside whitespace, etc.), find the
  // page by which one contains the selection's centre.  Without this
  // fallback `pageInfo` is undefined and the highlight rect goes
  // missing, which is exactly the "added but not highlighted" symptom.
  if (!pageInfo && clientRects.length && state.pageElements.length) {
    const all = clientRects[0];
    const cx = (all.left + all.right) / 2;
    const cy = (all.top  + all.bottom) / 2;
    pageInfo = state.pageElements.find(pi => {
      const r = pi.el.getBoundingClientRect();
      return cx >= r.left && cx <= r.right && cy >= r.top && cy <= r.bottom;
    }) || null;
  }

  let pdfRect = null;
  let resolvedPage = page;
  if (pageInfo && clientRects.length) {
    resolvedPage = pageInfo.pageNum;   // trust the geometric lookup
    const pageElRect = pageInfo.el.getBoundingClientRect();
    let cssL = Infinity, cssT = Infinity, cssR = -Infinity, cssB = -Infinity;
    for (const r of clientRects) {
      cssL = Math.min(cssL, r.left - pageElRect.left);
      cssT = Math.min(cssT, r.top  - pageElRect.top);
      cssR = Math.max(cssR, r.right - pageElRect.left);
      cssB = Math.max(cssB, r.bottom - pageElRect.top);
    }
    // viewport CSS (top-down y) → PDF native (bottom-up y).
    // Pass the two diagonal corners; annotViewportRect will normalise.
    const [px0, py0] = pageInfo.viewport.convertToPdfPoint(cssL, cssB);
    const [px1, py1] = pageInfo.viewport.convertToPdfPoint(cssR, cssT);
    pdfRect = [px0, py0, px1, py1];
  }

  state.selectionCounter += 1;
  return {
    kind: "selection",
    text: text,
    page: resolvedPage,
    rect: pdfRect,    // PDF native coords; consumed by _attach_manual_rects
    rects: clientRects.map(r => [r.left, r.top, r.right, r.bottom]),
    id: `sel-${Date.now()}-${state.selectionCounter}`,
  };
}

function emitPageChange() {
  // Only emit page-change events back to Python if the host has
  // explicitly opted in.  Each emission costs a full Streamlit rerun
  // (the entire left panel re-renders), so by default we stay silent
  // and let the user scroll without consequences.
  if (!state.emitPageChanges) return;
  Streamlit.setComponentValue({
    kind: "page",
    page: state.pageNum,
    id: `page-${Date.now()}-${state.pageNum}`,
  });
}

async function paintTextLayer(textContent, viewport, container, pdfjs) {
  // Use PDF.js's official TextLayer class.  Earlier versions of v4 had
  // bugs that made us write a manual painter, but those are fixed in
  // v4.0.379 (the version we pin).  The official painter does what our
  // manual version couldn't:
  //   - Sizes each <span> to the actual canvas-rendered width via a
  //     per-span `transform: scaleX(...)`, so spans never overlap
  //     neighbouring words.
  //   - Picks a font that matches the PDF's character widths rather than
  //     a fixed `sans-serif` (which was much wider than the typical
  //     serif PDF fonts and made selection feel "off by a word").
  //   - Handles rotation, RTL text, and ligatures correctly.
  if (pdfjs.TextLayer) {
    // v4.x API: construct + render.  textContentSource accepts either
    // the resolved textContent object or a Promise.
    const tl = new pdfjs.TextLayer({
      textContentSource: textContent,
      container:         container,
      viewport:          viewport,
    });
    await tl.render();
    return;
  }
  // Last-resort fallback for older PDF.js versions — keeps the
  // component working but with the imperfect manual painter.  Should
  // never fire on pinned v4.0.379.
  const items = textContent.items || [];
  for (const item of items) {
    if (!item.str || item.str === "") continue;
    const tx = pdfjs.Util.transform(viewport.transform, item.transform);
    const fontHeight = Math.hypot(tx[2], tx[3]);
    if (fontHeight < 0.5) continue;
    const span = document.createElement("span");
    span.textContent       = item.str;
    span.style.position    = "absolute";
    span.style.left        = `${tx[4]}px`;
    span.style.top         = `${tx[5] - fontHeight}px`;
    span.style.fontSize    = `${fontHeight}px`;
    span.style.fontFamily  = "serif";
    container.appendChild(span);
  }
}

// Convert a Python-side annotation (in PDF coords) to a CSS rect on the
// rendered page.  Used both by drawAnnotation (to position the overlay)
// and by the page-level click handler (to hit-test mouse coords).
function annotViewportRect(annot, viewport) {
  const [pdfX0, pdfY0, pdfX1, pdfY1] = annot.rect;
  const [vx0, vy0] = viewport.convertToViewportPoint(pdfX0, pdfY0);
  const [vx1, vy1] = viewport.convertToViewportPoint(pdfX1, pdfY1);
  return {
    left: Math.min(vx0, vx1),
    top:  Math.min(vy0, vy1),
    w:    Math.abs(vx1 - vx0),
    h:    Math.abs(vy1 - vy0),
  };
}

function drawAnnotation(layer, annot, viewport) {
  const r = annotViewportRect(annot, viewport);
  const el = document.createElement("div");
  el.style.position = "absolute";
  el.style.left   = `${r.left}px`;
  el.style.top    = `${r.top}px`;
  el.style.width  = `${r.w}px`;
  el.style.height = `${r.h}px`;
  // ALWAYS non-interactive — the layer is below the text layer, and
  // clicks are handled at the page level via hit-testing the
  // annotation rectangles.  This way text selection through the
  // colored highlight still works normally.
  el.style.pointerEvents = "none";
  if (annot.kind === "outline") {
    el.style.border  = `1.5px dashed ${annot.color || "#88a"}`;
    el.style.opacity = annot.opacity || 0.85;
  } else {
    el.style.background   = annot.color || "rgba(255, 215, 50, 0.3)";
    el.style.opacity      = annot.opacity || 0.30;
    el.style.mixBlendMode = "multiply";
  }
  if (annot.label) el.title = annot.label;
  if (typeof annot.ref_index === "number") {
    // Indicate clickability via cursor on hover — the click event itself
    // is handled at the page level.
    el.style.cursor = "pointer";
  }
  layer.appendChild(el);
}

function repaintAnnotationsOnly() {
  // Clear and redraw annotation overlays for every page.  PDF.js doesn't
  // render PDF annotation objects on its canvas, so we paint matched-ref
  // highlights here as DOM overlay divs.
  for (const pi of state.pageElements) {
    pi.annotLayer.innerHTML = "";
    for (const a of (state.annotations || [])) {
      if (a.page === pi.pageNum) drawAnnotation(pi.annotLayer, a, pi.viewport);
    }
  }
}

async function renderAllPages() {
  const pdfjs = await loadPdfJs();
  const pages = document.getElementById("pages");
  pages.innerHTML = "";
  state.pageElements = [];

  if (!state.pdfDoc) return;
  state.totalPages = state.pdfDoc.numPages;

  const wrap = document.getElementById("pages-wrap");
  const targetWidth = Math.max(360, Math.min(wrap.clientWidth - 28, 980));

  // Render pages sequentially.  ~50 ms per page is typical; for a 20-page
  // paper that's ~1 s upfront, then zero per-scroll cost.
  for (let n = 1; n <= state.totalPages; n++) {
    const page = await state.pdfDoc.getPage(n);
    const baseVp = page.getViewport({ scale: 1 });
    const scale = targetWidth / baseVp.width;
    const viewport = page.getViewport({ scale });

    const pageEl = document.createElement("div");
    pageEl.className = "pdf-page";
    pageEl.dataset.pageNum = n;
    pageEl.style.position   = "relative";
    pageEl.style.margin     = "12px auto";
    pageEl.style.width      = `${viewport.width}px`;
    pageEl.style.height     = `${viewport.height}px`;
    pageEl.style.background = "white";
    pageEl.style.boxShadow  = "0 2px 12px rgba(0,0,0,0.15)";

    const canvas = document.createElement("canvas");
    canvas.width  = viewport.width;
    canvas.height = viewport.height;
    canvas.style.position = "absolute";
    canvas.style.left = "0";
    canvas.style.top  = "0";
    pageEl.appendChild(canvas);
    await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;

    // ORDER MATTERS:
    //   canvas (visual)  →  annotLayer (visible but non-interactive)
    //                     →  textLayer (on top — captures selection)
    // The annotation layer never captures pointer events; everything
    // passes through to the text layer.  To still get "click annotation
    // → focus ref" linking, we attach a page-level CLICK handler that
    // hit-tests the click coordinate against the page's annotation
    // rectangles — `click` only fires on a real click (no drag), so
    // text selection still works normally.
    const annotLayer = document.createElement("div");
    annotLayer.style.position = "absolute";
    annotLayer.style.left = "0";
    annotLayer.style.top  = "0";
    annotLayer.style.width  = `${viewport.width}px`;
    annotLayer.style.height = `${viewport.height}px`;
    annotLayer.style.pointerEvents = "none";
    pageEl.appendChild(annotLayer);

    const textLayer = document.createElement("div");
    textLayer.className = "text-layer";
    textLayer.style.position = "absolute";
    textLayer.style.left = "0";
    textLayer.style.top  = "0";
    textLayer.style.width  = `${viewport.width}px`;
    textLayer.style.height = `${viewport.height}px`;
    pageEl.appendChild(textLayer);

    const textContent = await page.getTextContent();
    await paintTextLayer(textContent, viewport, textLayer, pdfjs);

    // Click hit-test against the page's annotation rectangles.
    // Some browsers fire `click` even after a small mouse movement,
    // and `mouseup` on the text layer can confuse this with a drag-
    // select that happens to start near a highlight.  Track mousedown
    // → mouseup distance and only treat it as a click if the user
    // didn't move (< 4 px).  This makes selection within a highlighted
    // ref reliable without losing the click-to-focus affordance.
    let _mouseDownPos = null;
    pageEl.addEventListener("mousedown", (e) => {
      _mouseDownPos = { x: e.clientX, y: e.clientY };
    });
    pageEl.addEventListener("click", (e) => {
      if (_mouseDownPos) {
        const dx = e.clientX - _mouseDownPos.x;
        const dy = e.clientY - _mouseDownPos.y;
        if (dx * dx + dy * dy > 16) {   // moved > 4 px → it was a drag
          _mouseDownPos = null;
          return;
        }
      }
      _mouseDownPos = null;
      const pageRect = pageEl.getBoundingClientRect();
      const x = e.clientX - pageRect.left;
      const y = e.clientY - pageRect.top;
      for (const a of (state.annotations || [])) {
        if (a.page !== n || typeof a.ref_index !== "number") continue;
        const r = annotViewportRect(a, viewport);
        if (x >= r.left && x <= r.left + r.w && y >= r.top && y <= r.top + r.h) {
          if (list.items) {
            focusCard(a.ref_index);
          } else {
            Streamlit.setComponentValue({
              kind: "ref_click",
              ref_index: a.ref_index,
              id: `ref-${Date.now()}-${a.ref_index}`,
            });
          }
          e.stopPropagation();
          return;
        }
      }
    });

    pages.appendChild(pageEl);
    state.pageElements.push({
      el: pageEl, annotLayer, viewport, pageNum: n,
    });
  }

  // Initial annotation paint
  repaintAnnotationsOnly();

  // Scroll to requested page (if any) — passed in via state.pendingScrollToPage
  if (state.pendingScrollToPage) {
    const idx = state.pendingScrollToPage - 1;
    if (state.pageElements[idx]) {
      // small delay to let layout settle
      requestAnimationFrame(() => scrollWrapTo(state.pageElements[idx].el, "auto"));
    }
    state.pendingScrollToPage = null;
  }

  setupPageObserver();
  updatePageBar();
  Streamlit.setFrameHeight();
}

function setupPageObserver() {
  // Stable, deterministic page detection: on scroll, find the page whose
  // CENTER is closest to the viewport's center.  IntersectionObserver was
  // flickering between adjacent pages because their ratios are both >0
  // when the user is in-between — a max-ratio tiebreak isn't smooth.
  const wrap = document.getElementById("pages-wrap");
  if (state._scrollHandler) wrap.removeEventListener("scroll", state._scrollHandler);

  function detectVisiblePage() {
    if (!state.pageElements.length) return;
    const wrapRect = wrap.getBoundingClientRect();
    const targetY = wrapRect.top + wrapRect.height / 2;   // viewport vertical centre

    let best = state.pageElements[0];
    let bestDist = Infinity;
    for (const pi of state.pageElements) {
      const r = pi.el.getBoundingClientRect();
      const centre = r.top + r.height / 2;
      const dist = Math.abs(centre - targetY);
      if (dist < bestDist) {
        bestDist = dist;
        best = pi;
      }
    }

    if (best.pageNum !== state.pageNum) {
      state.pageNum = best.pageNum;
      updatePageBar();
      if (list.items && list.pageOnly) renderList();
      emitPageChange();
    }
  }

  // Debounced scroll handler — wait for the user to pause before
  // emitting page-change to Python.  rAF for smoothness, setTimeout for
  // settle.
  let scrollRafId = null;
  let scrollSettleTimer = null;
  state._scrollHandler = () => {
    if (scrollRafId) cancelAnimationFrame(scrollRafId);
    if (scrollSettleTimer) clearTimeout(scrollSettleTimer);
    scrollSettleTimer = setTimeout(() => {
      scrollRafId = requestAnimationFrame(detectVisiblePage);
    }, 180);
  };
  wrap.addEventListener("scroll", state._scrollHandler, { passive: true });

  // Detect once on setup
  detectVisiblePage();
}

function updatePageBar() {
  const label = document.getElementById("page-label");
  if (label) label.textContent = `${state.pageNum} / ${state.totalPages}`;
}

// Scroll only the viewer's own container.  scrollIntoView() also scrolls
// every ancestor — including the Streamlit page around this iframe — so
// "Show on p. N" used to yank the whole app down.
function scrollWrapTo(el, behavior) {
  const wrap = document.getElementById("pages-wrap");
  if (!wrap || !el) return;
  const top = el.getBoundingClientRect().top - wrap.getBoundingClientRect().top + wrap.scrollTop;
  wrap.scrollTo({ top: Math.max(0, top - 8), behavior });
}

// Scroll a reference into view and flash an outline around it — the
// card → PDF backlink.  Works when the reference is on the current page
// too (the page-only jump did nothing then).
let _lastFocusNonce = null;
function focusOn(focus) {
  if (!focus || focus.nonce === _lastFocusNonce) return;
  const pi = state.pageElements[(focus.page || 1) - 1];
  if (!pi) return;
  _lastFocusNonce = focus.nonce;
  const wrap = document.getElementById("pages-wrap");
  if (!focus.rect) { scrollWrapTo(pi.el, "smooth"); return; }
  const r = annotViewportRect({ rect: focus.rect }, pi.viewport);
  const pageTop = pi.el.getBoundingClientRect().top - wrap.getBoundingClientRect().top + wrap.scrollTop;
  wrap.scrollTo({ top: Math.max(0, pageTop + r.top - wrap.clientHeight * 0.12), behavior: "smooth" });
  const flash = document.createElement("div");
  flash.className = "focus-flash";
  Object.assign(flash.style, { left: `${r.left - 4}px`, top: `${r.top - 4}px`,
                               width: `${r.w + 8}px`, height: `${r.h + 8}px` });
  pi.annotLayer.appendChild(flash);
  setTimeout(() => flash.remove(), 2400);
}

function scrollToPage(n) {
  if (n < 1 || n > state.totalPages) return;
  const pi = state.pageElements[n - 1];
  if (pi) scrollWrapTo(pi.el, "smooth");
}

async function loadPdf(pdfBytes, annotations, scrollToTarget) {
  const newHash = quickHash(pdfBytes);
  const annotsChanged =
    JSON.stringify(state.annotations || []) !== JSON.stringify(annotations || []);
  state.annotations = annotations || [];

  // Fast path: same PDF — just update annotations (and maybe scroll).
  if (state.pdfDoc && state.pdfHash === newHash) {
    if (annotsChanged) repaintAnnotationsOnly();
    if (scrollToTarget && scrollToTarget !== state.pageNum) {
      scrollToPage(scrollToTarget);
    }
    return;
  }

  // Slow path: new PDF.  Re-rendering all pages naturally resets scroll
  // to the top.  When the caller didn't ask for an explicit scroll
  // target, that reset is a visible jolt for the user — they were
  // reading at page N, the script silently re-ran (e.g. they clicked
  // an annotation, which causes a Python rerun that recomputes the
  // annotated PDF bytes), and the view snaps back to page 1.
  //
  // Capture the scroll position *before* the slow re-render, then
  // restore it after layout settles when no explicit target was given.
  const wrap = document.getElementById("pages-wrap");
  const prevScrollTop = wrap ? wrap.scrollTop : 0;
  const prevPageNum   = state.pageNum || 1;

  const pdfjs = await loadPdfJs();
  state.pdfHash         = newHash;
  state.pdfDoc          = await pdfjs.getDocument({ data: pdfBytes }).promise;
  state.totalPages      = state.pdfDoc.numPages;
  state.pendingScrollToPage = scrollToTarget || null;
  state.pageNum         = scrollToTarget || prevPageNum || 1;
  await renderAllPages();

  // Restore scroll position when no explicit target was requested.
  // `renderAllPages` already handles `pendingScrollToPage`, so we only
  // step in when the caller is silent about scrolling.
  if (!scrollToTarget && wrap && prevScrollTop > 0) {
    requestAnimationFrame(() => {
      wrap.scrollTop = prevScrollTop;
    });
  }
}

function setupNavButtons() {
  const prev = document.getElementById("page-prev");
  const next = document.getElementById("page-next");
  if (prev) prev.onclick = () => scrollToPage(state.pageNum - 1);
  if (next) next.onclick = () => scrollToPage(state.pageNum + 1);
}

function setupSelectionWatcher() {
  const addBtn = document.getElementById("add-btn");
  const statusText = document.getElementById("status-text");
  let lastSelection = null;

  // Selection state shown in the toolbar.  The Add button is always
  // visible but only ENABLED when there's a real selection — clarifies
  // to the user that the button exists and what it does, vs floating
  // appear-on-selection which can be hidden/missed.
  function updateState() {
    const payload = selectionPayload();
    if (payload) {
      lastSelection = payload;
      addBtn.disabled = false;
      const preview = payload.text.length > 90 ? payload.text.slice(0, 90) + "…" : payload.text;
      statusText.innerHTML = `Selected: <b>${preview.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</b>`;
    } else {
      addBtn.disabled = true;
      // Don't clear lastSelection on every mousemove — only when we know
      // there's no active selection at this moment.
      lastSelection = null;
      statusText.textContent = "Tip: drag-select an unhighlighted reference in the PDF, then click the button.";
    }
  }

  // Native browser selection lifecycle.  selectionchange fires whenever
  // the user changes the selection (drag, click-collapse, etc.).
  document.addEventListener("selectionchange", () => {
    // Debounce: settle for a brief moment before reading the selection
    clearTimeout(state._selectionTimer);
    state._selectionTimer = setTimeout(updateState, 50);
  });

  // Capture on mousedown so we don't lose the selection to focus changes
  addBtn.addEventListener("mousedown", (e) => {
    e.preventDefault();   // keep the selection alive
    const payload = selectionPayload();
    if (payload) lastSelection = payload;
  });

  addBtn.addEventListener("click", (e) => {
    e.preventDefault();
    // Last-ditch: re-read selection in case state went stale
    const fresh = selectionPayload();
    const payload = fresh || lastSelection;
    if (!payload) {
      statusText.textContent = "Couldn't read your selection — try selecting again.";
      return;
    }
    Streamlit.setComponentValue(payload);
    statusText.innerHTML = `<b>Sent to backend.</b>  Processing…`;
    addBtn.disabled = true;
    lastSelection = null;
    window.getSelection().removeAllRanges();
  });
}

let _lastPdfKey       = null;     // PDF-bytes identity (length)
let _lastAnnotKey     = null;     // serialised annotations
let _deferredRender   = null;     // args from a render deferred during selection
let _selectingTimeout = null;     // safety reset for stuck isSelecting

// `isSelecting` gates re-renders while the user is dragging a text
// selection — re-rendering mid-drag rebuilds the text-layer spans and
// destroys the selection range.  Three ways the flag gets cleared, so
// it can't stay stuck even if the user drags out of the window:
//
//   1. mouseup anywhere on document  (normal case)
//   2. selectionchange → collapsed   (selection was released; covers
//                                     the "released outside the window"
//                                     edge case where mouseup never
//                                     fires on document)
//   3. 1.5s safety timeout           (last-resort heal — drops the flag
//                                     even if both 1 and 2 are missed)
//
// Used to be just (1), which is why renders would silently stop
// updating after a drag-out-of-window.

function _clearSelecting() {
  state.isSelecting = false;
  if (_selectingTimeout) {
    clearTimeout(_selectingTimeout);
    _selectingTimeout = null;
  }
  // Flush any render that was queued while the flag was set
  if (_deferredRender) {
    const ev = _deferredRender;
    _deferredRender = null;
    onRender(ev);
  }
}

(function () {
  const wrap = document.getElementById("pages-wrap");
  if (!wrap) return;
  wrap.addEventListener("mousedown", () => {
    state.isSelecting = true;
    if (_selectingTimeout) clearTimeout(_selectingTimeout);
    // Heal-after-1.5s: if we somehow miss both mouseup AND the
    // selectionchange→collapsed signal, drop the flag anyway so the
    // viewer doesn't go silent.  Real drags rarely take longer than
    // half a second; 1.5s is a comfortable cushion.
    _selectingTimeout = setTimeout(_clearSelecting, 1500);
  });
  document.addEventListener("mouseup", _clearSelecting);
  document.addEventListener("selectionchange", () => {
    // If the selection has collapsed (user clicked away / released
    // outside the window), the drag is over even if we never saw the
    // mouseup event.
    const sel = window.getSelection();
    if (sel && sel.isCollapsed && state.isSelecting) {
      _clearSelecting();
    }
  });
})();

// ─────────  Reference list (client-side)  ─────────
// Items come from Python with pre-rendered, escaped card HTML.  Filtering,
// search, the page filter, opening cards and "Show in PDF" all happen here
// without a Streamlit rerun; only server actions are sent back.
const list = {
  items: null, labels: {}, order: [], filter: "all", query: "", pageOnly: false,
  open: new Set(), keyHtml: "",
};

function itemVisible(it) {
  if (list.filter !== "all" && it.status !== list.filter) return false;
  if (list.query && !it.text.includes(list.query)) return false;
  if (list.pageOnly && it.page !== state.pageNum) return false;
  return true;
}

function renderFilterButtons() {
  const counts = { all: list.items.length };
  for (const it of list.items) counts[it.status] = (counts[it.status] || 0) + 1;
  const box = document.getElementById("status-filter");
  box.innerHTML = "";
  for (const key of ["all", ...list.order]) {
    if (key !== "all" && !counts[key]) continue;
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = `${key === "all" ? "All" : list.labels[key]} ${counts[key] || 0}`;
    b.setAttribute("aria-pressed", String(list.filter === key));
    b.onclick = () => { list.filter = key; renderFilterButtons(); renderList(); };
    box.appendChild(b);
  }
}

function renderList() {
  const el = document.getElementById("list");
  const scroll = el.scrollTop;
  const vis = list.items.filter(itemVisible);
  document.getElementById("list-count").textContent =
    `${vis.length} of ${list.items.length} references`;
  if (!vis.length) {
    el.innerHTML = '<div class="empty">No references match these filters.</div>';
    return;
  }
  el.innerHTML = vis.map((it) => {
    const acts = [];
    if (it.page) acts.push(`<button data-act="show" data-i="${it.i}">◎ Show in PDF (p. ${it.page})</button>`);
    for (const a of it.actions) {
      acts.push(`<button data-act="${a.act}" data-i="${it.i}"${a.disabled ? " disabled" : ""}>${a.label}</button>`);
    }
    return `<div class="item" id="item-${it.i}">${it.html}<div class="actions">${acts.join("")}</div></div>`;
  }).join("");
  for (const d of el.querySelectorAll("details.refcard")) {
    const i = Number(d.closest(".item").id.slice(5));
    if (list.open.has(i)) d.open = true;
    d.addEventListener("toggle", () => { d.open ? list.open.add(i) : list.open.delete(i); });
  }
  el.scrollTop = scroll;
}

function focusCard(i) {
  // PDF highlight → card: make sure it is visible, open it, scroll to it.
  const it = (list.items || []).find((x) => x.i === i);
  if (!it) return;
  if (!itemVisible(it)) {
    list.filter = "all"; list.query = ""; list.pageOnly = false;
    document.getElementById("search").value = "";
    document.getElementById("page-only").checked = false;
    renderFilterButtons();
  }
  list.open.add(i);
  renderList();
  const node = document.getElementById(`item-${i}`);
  if (!node) return;
  // Scroll #list itself — scrollIntoView would also scroll the app page.
  const box = document.getElementById("list");
  box.scrollTo({ top: node.offsetTop - 8, behavior: "smooth" });
  node.classList.add("flash");
  setTimeout(() => node.classList.remove("flash"), 1600);
}

function setupList() {
  const search = document.getElementById("search");
  let t = null;
  search.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(() => { list.query = search.value.trim().toLowerCase(); renderList(); }, 120);
  });
  document.getElementById("page-only").addEventListener("change", (e) => {
    list.pageOnly = e.target.checked; renderList();
  });
  document.getElementById("list").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-act]");
    if (!b || b.disabled) return;
    const i = Number(b.dataset.i);
    const it = list.items.find((x) => x.i === i);
    if (b.dataset.act === "show") {
      focusOn({ page: it.page, rect: it.rect, nonce: `show-${Date.now()}` });
      return;
    }
    b.disabled = true;     // server action: one click, then wait for the rerun
    Streamlit.setComponentValue({ kind: "action", action: b.dataset.act, ref_index: i,
                                  id: `act-${Date.now()}-${i}` });
  });
}

function updateList(args) {
  const pane = document.getElementById("list-pane");
  if (!args.items) { pane.hidden = true; return; }
  pane.hidden = false;
  if (args.cards_css && document.getElementById("cards-css").textContent !== args.cards_css) {
    document.getElementById("cards-css").textContent = args.cards_css;
  }
  const key = JSON.stringify(args.items);
  if (key === list.keyHtml) return;          // nothing changed → keep DOM, scroll, focus
  list.keyHtml = key;
  list.items = args.items;
  list.labels = args.status_labels || {};
  list.order = args.status_order || [];
  if (list.filter !== "all" && !list.items.some((x) => x.status === list.filter)) list.filter = "all";
  renderFilterButtons();
  renderList();
}

function applyTheme(theme) {
  if (!theme) return;
  const root = document.documentElement;
  root.classList.toggle("dark", theme.base === "dark");
  if (theme.backgroundColor) root.style.setProperty("--bg", theme.backgroundColor);
  if (theme.textColor) root.style.setProperty("--fg", theme.textColor);
  if (theme.font) root.style.setProperty("font-family", theme.font);
  if (theme.primaryColor) {
    root.style.setProperty("--accent", theme.primaryColor);
    root.style.setProperty("--accent-hover", theme.primaryColor);
  }
}

function onRender(event) {
  const args = event.detail.args || {};
  applyTheme(event.detail.theme);
  updateList(args);
  if (args.height) Streamlit.setFrameHeight(args.height);
  // Always update the emit flag — it controls JS-side behaviour for
  // page-change emissions and doesn't itself require a re-render.
  state.emitPageChanges = !!args.emit_page_changes;

  // Defer renders only when the user is actively dragging.  When they
  // release, the queued render fires.
  if (state.isSelecting) {
    _deferredRender = event;
    return;
  }

  // SPLIT the dedup logic.  PDF-bytes identity and annotation content
  // are tracked independently:
  //
  //   pdfKey   — if unchanged, we can take the JS fast path (no
  //              renderAllPages, just repaint annotations + maybe scroll)
  //   annotKey — if unchanged, we can skip even repaintAnnotationsOnly
  //
  // The previous version OR'd both into one hash and a single skip,
  // which meant Streamlit reruns that fired with identical annotations
  // would silently skip — fine in the common case, but if the JS
  // already had stale annotations drawn (e.g. from a deferred render
  // that got lost), the skip prevented recovery.  Splitting lets the
  // annotation-only path always run when there's a change, while the
  // expensive PDF reload still gets the skip optimisation.
  // Length plus content samples: length alone treated two different
  // PDFs of the same encoded size as unchanged.
  const b64      = args.pdf_b64 || "";
  const mid      = b64.length >> 1;
  const pdfKey   = `${b64.length}:${b64.slice(-96)}:${b64.slice(mid, mid + 96)}`;
  const annotKey = JSON.stringify(args.annotations || []);
  const pdfUnchanged   = pdfKey   === _lastPdfKey;
  const annotUnchanged = annotKey === _lastAnnotKey;
  const focus = args.focus || null;
  const focusIsNew = !!focus && focus.nonce !== _lastFocusNonce;
  if (pdfUnchanged && annotUnchanged && !args.scroll_to_page) {
    // Nothing to re-render; maybe a backlink jump.
    if (focusIsNew) focusOn(focus);
    return;
  }
  _lastPdfKey   = pdfKey;
  _lastAnnotKey = annotKey;

  const pdfBytes = b64ToBytes(args.pdf_b64 || "");
  const annotations = args.annotations || [];
  const scrollTarget = args.scroll_to_page || null;
  loadPdf(pdfBytes, annotations, scrollTarget).then(() => {
    if (focusIsNew) requestAnimationFrame(() => focusOn(focus));
  }).catch((err) => {
    document.getElementById("pages").innerHTML =
      `<pre style="color:red; padding:1em;">PDF render error: ${err.message}\n${err.stack}</pre>`;
    Streamlit.setFrameHeight();
  });
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
setupNavButtons();
setupSelectionWatcher();
setupList();
Streamlit.setFrameHeight(900);
