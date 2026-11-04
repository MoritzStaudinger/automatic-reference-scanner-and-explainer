"""
Remove line numbers from draft / review-mode PDFs.

Submissions in review mode (ACM "manuscript, review", ACL/EMNLP review,
NeurIPS/ICLR drafts, ...) print a running line number in the margin next to
every text line.  Text extraction then interleaves those numbers with the
prose and the references: "Language models are 1561 few-shot learners",
"recovered 161 search date", "168 2 Background".  That corrupts citation
strings, splits sentences and confuses GROBID.

Line numbers have a tell no other text has: bare integers stacked in one
narrow column whose values increase steadily from top to bottom.  This
module finds such columns on each page and removes the numbers from the
text layer (redaction), so every later step — GROBID, the section parser,
highlighting, citation extraction — reads a clean PDF.  Coordinates of all
other text are unchanged.

Reference markers ("[12]", "12.") and page numbers (one per page) never
qualify: the former aren't bare integers, the latter don't form a column.
"""

from __future__ import annotations

import re
from collections import defaultdict

_NUM_RE = re.compile(r"\d{1,5}")

MIN_RUN = 15          # numbers in a column before it counts on its own
MIN_INCREASING = 0.8  # share of consecutive pairs that go up by a small step
MAX_STEP = 10         # +1 per line; some styles label every 5th or 10th line
MARGIN = 0.15         # line numbers sit in the outer 15 % of the page width
MIN_SPAN = 0.5        # ... and accompany at least half the page height
EDGE_TOL = 3.0        # pt: aligned numbers share an edge within this


def _increasing(nums: list[int]) -> bool:
    if len(nums) < 3:
        return False
    steps = [b - a for a, b in zip(nums, nums[1:])]
    return sum(1 for s in steps if 0 < s <= MAX_STEP) / len(steps) >= MIN_INCREASING


def _columns(words: list[tuple], edge: int) -> list[list[tuple]]:
    """Group words whose left (edge=0) or right (edge=2) x lies within
    EDGE_TOL of the previous one — no chaining across a wide range."""
    out: list[list[tuple]] = []
    for w in sorted(words, key=lambda w: w[edge]):
        if out and w[edge] - out[-1][0][edge] <= EDGE_TOL:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def _in_margin(col: list[tuple], width: float) -> bool:
    x = sum((w[0] + w[2]) / 2 for w in col) / len(col)
    return x < MARGIN * width or x > (1 - MARGIN) * width


def find_line_numbers(doc) -> dict[int, list[tuple]]:
    """{page index: line-number words} for the whole document.

    A column qualifies on its own when it is in the margin, increasing,
    has MIN_RUN+ numbers and spans MIN_SPAN of the page height.  Columns at
    the x position of a qualifying column then also count on pages with
    few lines (a draft's last page), as long as they increase."""
    pages = []
    strong_edges: list[tuple[int, float]] = []
    for page in doc:
        words = [w for w in page.get_text("words") if _NUM_RE.fullmatch(w[4])]
        cols = []
        for edge in (0, 2):
            for col in _columns(words, edge):
                col = sorted(col, key=lambda w: (w[1], w[0]))
                if not _in_margin(col, page.rect.width) or not _increasing([int(w[4]) for w in col]):
                    continue
                span = (col[-1][3] - col[0][1]) / page.rect.height
                strong = len(col) >= MIN_RUN and span >= MIN_SPAN
                cols.append((edge, col, strong))
                if strong:
                    strong_edges.append((edge, sum(w[edge] for w in col) / len(col)))
        pages.append(cols)
    found: dict[int, list[tuple]] = {}
    for i, cols in enumerate(pages):
        hits: dict[tuple, tuple] = {}
        for edge, col, strong in cols:
            x = sum(w[edge] for w in col) / len(col)
            if strong or any(e == edge and abs(x - sx) <= EDGE_TOL for e, sx in strong_edges):
                for w in col:
                    hits[w[:4]] = w
        if hits:
            found[i] = list(hits.values())
    return found


def find_line_number_words(page) -> list[tuple]:
    """Line numbers on a single page (no cross-page evidence)."""
    return find_line_numbers([page]).get(0, [])


def strip_line_numbers(pdf_bytes: bytes) -> tuple[bytes, int]:
    """Return ``(clean_pdf_bytes, n_removed)``.  The input is returned
    unchanged (same object) when the PDF has no line numbering or can't be
    processed — callers can compare with ``is``."""
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return pdf_bytes, 0
    removed = 0
    try:
        per_page = find_line_numbers(doc)
        for i, page in enumerate(doc):
            hits = per_page.get(i)
            if not hits:
                continue
            for w in hits:
                r = fitz.Rect(w[:4])
                # A hair inside the word box so neighbouring text is untouched.
                r = fitz.Rect(r.x0 + 0.3, r.y0 + 0.3, r.x1 - 0.3, r.y1 - 0.3)
                page.add_redact_annot(r)
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                                  graphics=fitz.PDF_REDACT_LINE_ART_NONE)
            removed += len(hits)
        if not removed:
            return pdf_bytes, 0
        return doc.tobytes(deflate=True), removed
    except Exception:
        return pdf_bytes, 0
    finally:
        doc.close()
