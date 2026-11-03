"""
Locate extracted references inside a PDF, add highlight annotations,
and return the annotated PDF bytes together with per-reference metadata.
"""

from __future__ import annotations
import re
from typing import Any

import fitz  # PyMuPDF

from verdict import STATUS_RGB


Color = tuple[float, float, float]

# Status → RGB highlight colour
#   match    = green   → confirmed by an academic source
#   mismatch = red     → wrong paper at this title (likely fabrication)
#   missing  = yellow  → couldn't find in any database (NOT proof of wrong;
#                        common for blogs/tech-reports/uncommon venues)
#   pending  = pale yellow → lookup not yet completed
COLORS: dict[str, Color] = dict(STATUS_RGB)


def _clean_for_search(text: str) -> str:
    """Normalise a reference string so it can be matched against PDF text."""
    # Remove markdown italic underscores and bold/italic asterisks
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"_+", " ", text)
    # Strip leading markdown list bullet: "- " or "* "
    text = re.sub(r"^\s*[-*]\s+", "", text)
    # Strip leading numbered marker: [1], (1), 1., 1)
    text = re.sub(r"^\s*[\[\(]?\d+[\]\)\.]\s*", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _search_snippets(raw: str, title: str = "") -> list[str]:
    """
    Return candidate search snippets in **longest-first** order so the highlighter
    covers the full reference whenever the raw text matches the PDF cleanly.

    Order:
      1. Full raw (capped at 500 chars — PyMuPDF handles multi-line / hyphenated
         line breaks for searches up to a few hundred chars).
      2. Progressively shorter raw prefixes — used as fallbacks when GROBID has
         reformatted the raw (e.g. initials' periods stripped) and the full
         string doesn't match byte-for-byte.
      3. Title.  Strong final fallback because titles usually appear verbatim
         in the PDF even when the surrounding author / venue text differs.
    """
    snippets: list[str] = []

    cleaned = _clean_for_search(raw)

    # 1. Full raw first — gives the whole-reference bounding box when matched
    if 30 <= len(cleaned) <= 500:
        snippets.append(cleaned)
    elif len(cleaned) > 500:
        cut = cleaned[:500].rsplit(" ", 1)[0].strip(" .,;")
        if cut:
            snippets.append(cut)

    # 2. Progressively shorter raw prefixes — fragment fallbacks
    for target_len in (180, 120, 70, 35):
        if len(cleaned) <= target_len:
            s = cleaned
        else:
            cut = cleaned[:target_len].rsplit(" ", 1)[0]
            s = cut.strip(" .,;")
        if len(s) >= 15 and s not in snippets:
            snippets.append(s)

    # 3. Title fallback
    title_clean = _clean_for_search(title)
    if title_clean and title_clean not in snippets and len(title_clean) >= 15:
        snippets.append(title_clean)
        if len(title_clean) > 40:
            cut = title_clean[:40].rsplit(" ", 1)[0].strip(" .,;")
            if cut and cut not in snippets and len(cut) >= 15:
                snippets.append(cut)

    return snippets


_ORPHAN_AUTHOR_START_RE = re.compile(
    r"^[A-Z][a-zA-Z\-']{1,}"          # capital surname
    r"(?:,?\s+[A-Z]\.?"               # comma+initial OR
    r"|\s+[A-Z][a-zA-Z\-']+)",        # space+capital first name
)
_ORPHAN_DOI_RE   = re.compile(r"10\.\d{4,}/\S+")
_ORPHAN_ARXIV_RE = re.compile(r"arXiv:\s*\d{4}\.\d+|arxiv\.org/", re.IGNORECASE)
_ORPHAN_URL_RE   = re.compile(r"https?://", re.IGNORECASE)
_ORPHAN_ETAL_RE  = re.compile(r"\bet\s*al\.?\b", re.IGNORECASE)

# Phrases that mark text as prose / not a citation.  Conservative: each
# of these is something we'd basically never see at the START of a
# bibliography entry but commonly see in body text inside the bib
# region (e.g. an editor's note, an embedded paragraph).
_ORPHAN_BODY_MARKERS = (
    "we propose", "we present", "we describe", "in this paper",
    "in this section", "in this work", "in conclusion",
    "table ", "figure ", "section ", "appendix ",
    "as shown in", "for example", "however,",
)


def _orphan_confidence(text: str) -> str:
    """
    Classify an orphan text block as 'high' or 'low' confidence reference.

    'high' means the block looks unambiguously like a bibliography
    entry (author surname start + ref-shaped punctuation + at least one
    strong signal: DOI / arXiv ID / URL / "et al" / ≥3 commas).  Callers
    can auto-add these as references without user review.

    'low' means we surface it as a "possible missed reference" but
    don't act on it automatically.  Stays in the manual-triage panel.
    """
    text = text.strip()
    if len(text) < 60:
        return "low"

    if not _ORPHAN_AUTHOR_START_RE.match(text):
        return "low"

    low = text.lower()
    if any(marker in low for marker in _ORPHAN_BODY_MARKERS):
        return "low"

    has_doi          = bool(_ORPHAN_DOI_RE.search(text))
    has_arxiv        = bool(_ORPHAN_ARXIV_RE.search(text))
    has_url          = bool(_ORPHAN_URL_RE.search(text))
    has_etal         = bool(_ORPHAN_ETAL_RE.search(text))
    has_many_commas  = text.count(",") >= 3

    if has_doi or has_arxiv or has_url or has_etal or has_many_commas:
        return "high"
    return "low"


# ---------------------------------------------------------------------------
# Line-level locating
# ---------------------------------------------------------------------------
# PyMuPDF often splits one reference over several text blocks (at a
# hyperlinked DOI / URL, a font change, a column or page break), so
# highlighting the single best-matching block left references partially
# boxed or not boxed at all.  Instead: find the line where the reference
# starts, follow it line by line in reading order (across columns and
# pages), and draw one rectangle per contiguous piece.

def _collect_lines(doc) -> list[tuple[int, "fitz.Rect", str]]:
    lines = []
    for page in doc:
        for b in page.get_text("dict").get("blocks", []):
            if b.get("type") != 0:
                continue
            for line in b.get("lines") or []:
                t = "".join(sp.get("text", "") for sp in line.get("spans") or [])
                n = _clean_for_search(t).lower()
                if n:
                    lines.append((page.number, fitz.Rect(line["bbox"]), n))
    return lines


def _locate_by_lines(ref_norm: str, lines, used: list[bool]) -> list[int] | None:
    """Indices of the lines that make up ``ref_norm`` (best start wins)."""
    from rapidfuzz import fuzz
    if len(ref_norm) < 25:
        return None
    head = ref_norm[:60]
    budget = len(ref_norm) * 1.15
    best: tuple[float, list[int]] | None = None
    for i, (_pn, _r, n) in enumerate(lines):
        if used[i] or len(n) < 8:
            continue
        # The reference must *start* on this line.
        if fuzz.partial_ratio(n[:30], head) < 88:
            continue
        if fuzz.ratio(n[:40], ref_norm[:min(len(n), 40)]) < 75:
            continue
        run, acc, j, skipped = [i], len(n), i + 1, False
        while j < len(lines) and acc < budget and not used[j]:
            nj = lines[j][2]
            if fuzz.partial_ratio(nj, ref_norm) >= 88:
                run.append(j)
                acc += len(nj) + 1
                skipped = False
            elif len(nj) < 15 and not skipped:
                skipped = True       # page number / running header between pieces
            else:
                break
            j += 1
        text = " ".join(lines[k][2] for k in run)
        score = fuzz.ratio(text, ref_norm)
        if best is None or score > best[0]:
            best = (score, run)
    return best[1] if best and best[0] >= 75 else None


def _group_rects(run: list[int], lines) -> list[tuple[int, "fitz.Rect"]]:
    """Merge consecutive lines into one rectangle per column / page piece."""
    groups: list[tuple[int, fitz.Rect]] = []
    for k in run:
        pn, r, _ = lines[k]
        if groups:
            gpn, g = groups[-1]
            if (gpn == pn and abs(r.x0 - g.x0) < 40
                    and -3 <= r.y0 - g.y1 < 8):
                groups[-1] = (gpn, fitz.Rect(g) | r)   # `g |= r` doesn't update the list
                continue
        groups.append((pn, fitz.Rect(r)))
    return groups


def _pdf_native(rect: "fitz.Rect", page_h: float) -> list[float]:
    """PyMuPDF top-down y → PDF-native bottom-up y (what PDF.js expects)."""
    return [rect.x0, page_h - rect.y1, rect.x1, page_h - rect.y0]


def highlight_references(
    pdf_bytes: bytes,
    references: list[dict[str, Any]],
    statuses: list[str] | None = None,
    bake_color: bool = True,
) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Add highlights for each reference found in the PDF.

    Parameters
    ----------
    pdf_bytes : bytes
        Original PDF.
    references : list[dict]
        Extracted reference dicts (must have ``raw`` key).
    statuses : list[str] | None
        Per-reference status strings (``"match"``, ``"mismatch"``,
        ``"missing"``, ``"pending"``).  Defaults to ``"pending"`` for all.
    bake_color : bool
        Whether to draw colored rectangles into the PDF page content.

        ``True`` (default) — used by the legacy ``streamlit_pdf_viewer``
        viewer which has no overlay capability; the colour has to live in
        the PDF bytes themselves.

        ``False`` — used by our custom ``pdf_selector`` component, which
        paints its own DOM overlay rectangles on top of PDF.js.  Baking
        the rect AS WELL doubles the visual opacity for parsed refs
        (baked 30% + overlay ~14%), while manually-added refs would only
        get the overlay — producing the "different shade of yellow"
        symptom the user sees.  We still record ``rect`` / ``page`` on
        each entry so the overlay knows where to draw.
    """
    if statuses is None:
        statuses = ["pending"] * len(references)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    enriched: list[dict[str, Any]] = []

    # Block-based highlighting:  PyMuPDF gives us a paragraph-level list of
    # text "blocks" per page, each with a bounding box.  A reference is
    # almost always exactly one block, so we find the block whose text
    # contains the ref (fuzzy match), then highlight the whole block rect.
    # This produces clean "one rectangle per ref" highlights instead of
    # the fragmented per-line search-for results.
    #
    # Each block: dict gathered up front so we can fuzzy-match each ref
    # against all of them once.
    from rapidfuzz import fuzz as _fuzz

    # ACM- and Springer-style bibliographies pack refs tightly enough
    # that PyMuPDF's block detector often merges several citations
    # into a single block.  When that happens, highlighting the whole
    # block paints a giant rectangle that covers refs [3]-[7] for the
    # cost of one match — the user can't visually tell them apart.
    #
    # Detect merged blocks by scanning their lines for citation markers
    # via `get_text("dict")` (which carries per-line bboxes).  When a
    # block contains ≥2 marker lines, split it into sub-blocks at each
    # marker boundary, using the spanning line bboxes to derive a tight
    # rect for each individual ref.
    #
    # Marker forms recognised:
    #   "[1] Author..."   ACM / NeurIPS / IEEE
    #   "(1) Author..."   uncommon but used
    #   "1. Author..."    Springer LNCS / many books
    #   "1) Author..."    occasional
    # The trailing capital letter requirement (`[A-Z]`) excludes things
    # like inline "Section 3. discusses" from accidentally splitting a
    # body-text block — refs invariably start with an author's surname.
    _REF_MARKER_RE = re.compile(
        r"^\s*(?:\[\d+\]|\(\d+\)|\d+[.)])\s+[A-Z]"
    )

    blocks: list[tuple[int, fitz.Rect, str, str]] = []   # (page_idx, rect, raw_text, norm_text)
    for page in doc:
        text_dict = page.get_text("dict")
        for b in text_dict.get("blocks", []):
            if b.get("type") != 0:   # 0 = text, 1 = image
                continue
            lines = b.get("lines") or []
            if not lines:
                continue

            # Find lines that start a new numbered reference
            marker_idxs: list[int] = []
            line_texts: list[str] = []
            for li, line in enumerate(lines):
                spans   = line.get("spans") or []
                line_t  = "".join(s.get("text", "") for s in spans)
                line_texts.append(line_t)
                if _REF_MARKER_RE.match(line_t):
                    marker_idxs.append(li)

            if len(marker_idxs) >= 2:
                # Merged block — split into per-ref sub-blocks.  Each
                # sub-block spans lines [marker_idxs[k], marker_idxs[k+1]).
                marker_idxs.append(len(lines))   # sentinel for last segment
                for k in range(len(marker_idxs) - 1):
                    start, end = marker_idxs[k], marker_idxs[k + 1]
                    sub_lines = lines[start:end]
                    sub_text  = "\n".join(line_texts[start:end]).strip()
                    if len(sub_text) < 30:
                        continue
                    # Union of all line bboxes in this segment
                    x0 = min(li["bbox"][0] for li in sub_lines)
                    y0 = min(li["bbox"][1] for li in sub_lines)
                    x1 = max(li["bbox"][2] for li in sub_lines)
                    y1 = max(li["bbox"][3] for li in sub_lines)
                    rect = fitz.Rect(x0, y0, x1, y1)
                    norm = _clean_for_search(sub_text).lower()
                    blocks.append((page.number, rect, sub_text, norm))
                continue

            # Default path: one block = one ref (ACL / NeurIPS / etc.)
            bbox = b.get("bbox", (0, 0, 0, 0))
            text = "\n".join(line_texts).strip()
            if len(text) < 30:
                continue
            rect = fitz.Rect(*bbox)
            norm = _clean_for_search(text).lower()
            blocks.append((page.number, rect, text, norm))

    used_block_idx: set[int] = set()  # blocks already claimed by an earlier ref
    lines = _collect_lines(doc)
    line_used = [False] * len(lines)

    for ref, status in zip(references, statuses):
        entry = dict(ref)
        entry["found"] = False
        entry["page"] = None
        color = COLORS.get(status, COLORS["pending"])

        ref_raw  = _clean_for_search(ref.get("raw", "")).lower()
        ref_title = _clean_for_search(ref.get("title", "")).lower()
        if not ref_raw and not ref_title:
            enriched.append(entry)
            continue

        run = _locate_by_lines(ref_raw, lines, line_used) if ref_raw else None
        if run:
            for k in run:
                line_used[k] = True
            pieces = _group_rects(run, lines)
            rects = []
            for pn, rect in pieces:
                page = doc[pn]
                if bake_color:
                    page.draw_rect(rect, color=None, fill=color, fill_opacity=0.30,
                                   width=0, overlay=True)
                rects.append({"page": pn + 1, "rect": _pdf_native(rect, page.rect.height)})
            entry.update(found=True, page=rects[0]["page"], rect=rects[0]["rect"],
                         rects=rects, status=status)
            enriched.append(entry)
            continue

        # Fallback: best-matching block.  Score every block by best fuzzy
        # match on raw OR title.
        # Use partial_ratio so a short ref title still scores high against
        # a longer block (block can contain title + venue etc.).
        best_score = 0
        best_idx = -1
        for i, (pn, rect, text, norm) in enumerate(blocks):
            if i in used_block_idx:
                continue
            s = 0
            if ref_raw:
                s = max(s, _fuzz.partial_ratio(ref_raw, norm))
            if ref_title and len(ref_title) >= 15:
                s = max(s, _fuzz.partial_ratio(ref_title, norm))
            if s > best_score:
                best_score = s
                best_idx = i

        if best_idx >= 0 and best_score >= 80:
            pn, rect, _, _ = blocks[best_idx]
            page = doc[pn]
            if bake_color:
                # Draw the highlight as page CONTENT, not as a PDF annotation.
                # add_rect_annot would create an interactive widget that PDF.js
                # renders in its AnnotationLayer with pointer-events:auto —
                # which intercepts clicks on any PDF link annotation underneath
                # (DOIs, URLs, etc.) and makes them unclickable.  draw_rect
                # bakes the rectangle into the page's content stream, so it's
                # visually identical but completely non-interactive — links
                # underneath stay clickable.
                page.draw_rect(rect, color=None, fill=color, fill_opacity=0.30,
                               width=0, overlay=True)
            # When `bake_color` is False we still record rect + page so the
            # JS overlay knows where to draw — it just owns the colour.
            used_block_idx.add(best_idx)
            entry["found"] = True
            entry["page"] = pn + 1
            # Expose the rect so the JS PDF component can overlay the
            # highlight on its own canvas (PDF.js doesn't render PDF
            # annotation objects by default).
            #
            # COORDINATE FLIP: PyMuPDF uses top-down y (y=0 at top of page);
            # PDF.js's viewport.convertToViewportPoint expects PDF-native
            # bottom-up y (y=0 at bottom).  We translate here so the
            # component can pass the rect straight into the converter.
            page_h = page.rect.height
            entry["rect"] = [
                rect.x0,
                page_h - rect.y1,   # bottom of pymupdf box → y0 in PDF native
                rect.x1,
                page_h - rect.y0,   # top of pymupdf box → y1 in PDF native
            ]
            entry["rects"] = [{"page": entry["page"], "rect": entry["rect"]}]
            entry["status"] = status

        enriched.append(entry)

    # Identify orphan blocks — text paragraphs in the bibliography region
    # that no reference claimed.  These are the most likely "missed refs"
    # the user might want to add (or the system might want to auto-add).
    # Restricted to pages where at least one parsed ref was located
    # (so we don't surface body text from the rest of the doc).
    #
    # Each orphan carries a `confidence` field — "high" means we're
    # confident enough to auto-add it as a ref (callers can do this
    # without user review); "low" means it goes to the "possible missed
    # references" panel for manual triage.
    lines_by_page: dict[int, list[int]] = {}
    for k, (lpn, _lr, _) in enumerate(lines):
        lines_by_page.setdefault(lpn, []).append(k)
    for bi, (pn, rect, _t, _n) in enumerate(blocks):
        inside = [k for k in lines_by_page.get(pn, [])
                  if rect.contains(fitz.Point((lines[k][1].x0 + lines[k][1].x1) / 2,
                                              (lines[k][1].y0 + lines[k][1].y1) / 2))]
        if inside and sum(line_used[k] for k in inside) >= 0.5 * len(inside):
            used_block_idx.add(bi)

    pages_with_refs = {e["page"] for e in enriched if e.get("page")}
    orphans: list[dict[str, Any]] = []
    import re as _re_h
    for i, (pn, rect, text, _norm) in enumerate(blocks):
        if i in used_block_idx:
            continue
        if (pn + 1) not in pages_with_refs:
            continue
        # Baseline orphan filter: must look like a citation
        # (has year + reasonable length).
        if len(text) < 40:
            continue
        if not _re_h.search(r"\b(?:19|20)\d{2}\b", text):
            continue
        # PDF-native rect coords (bottom-up y) — same convention as
        # parsed-ref rects above so the JS overlay can render
        # auto-added orphans consistently.
        page_h_orph = doc[pn].rect.height
        pdf_rect = [
            rect.x0,
            page_h_orph - rect.y1,
            rect.x1,
            page_h_orph - rect.y0,
        ]
        orphans.append({
            "page":       pn + 1,
            "rect":       pdf_rect,
            "text":       text,
            "confidence": _orphan_confidence(text),
        })
        # NOTE: we deliberately do NOT draw the orphan rectangle as a PyMuPDF
        # annotation here.  The PDF selector component receives `orphans` via
        # its `annotations` arg and paints the dashed outline as a DOM overlay
        # on top of the rendered canvas.  Drawing both produces overlapping
        # boxes.

    annotated_bytes = doc.tobytes(garbage=2, deflate=True)
    doc.close()
    return annotated_bytes, enriched, orphans
