"""
Document processing for the UI: parse → (LLM repair) → dedupe → locate in
PDF → auto-detect missed refs.  Every expensive step is ``st.cache_data``'d
(bounded, so a public instance doesn't grow without limit) and keyed on
content, never on file names.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import streamlit as st

from highlighter import highlight_references
from llm_parser import active_model, llm_available
from parsers import (
    ParserError,
    _grobid_process_citation_list,
    dedupe_references,
    extract_references_grobid,
    extract_references_hybrid,
    extract_references_pymupdf4llm,
    grobid_is_available,
)

_CACHE = dict(show_spinner=False, max_entries=64, ttl=6 * 3600)

LLM_BACKEND, LLM_MODEL = active_model()
LLM_READY = llm_available()

try:
    import docling  # noqa: F401
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

PARSER_HELP = {
    "hybrid":      "Best quality. GROBID's full-document parse merged with the "
                   "bibliography section structured by GROBID — catches references "
                   "either method misses alone.",
    "grobid":      "GROBID full-document parse only. Fast and clean; may miss "
                   "references in unusual bibliographies.",
    "pymupdf4llm": "Fastest, no GROBID needed. Rule-based field extraction; "
                   "brittle on complex bibliographies.",
    "docling":     "Slow (minutes). Rule-based extraction with docling's layout "
                   "analysis.",
}


def parser_options() -> list[str]:
    opts = (["hybrid", "grobid"] if grobid_is_available() else []) + ["pymupdf4llm"]
    if DOCLING_AVAILABLE:
        opts.append("docling")
    return opts


# ---------------------------------------------------------------------------
# Cached building blocks
# ---------------------------------------------------------------------------

@st.cache_data(**_CACHE)
def run_parser(pdf_bytes: bytes, parser: str) -> tuple[list[dict], str]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name
    try:
        if parser == "hybrid":
            return extract_references_hybrid(path, return_markdown=True)
        if parser == "grobid":
            return extract_references_grobid(path, return_markdown=True)
        if parser == "docling":
            from parsers import extract_references_docling
            return extract_references_docling(path, return_markdown=True)
        return extract_references_pymupdf4llm(path, return_markdown=True)
    finally:
        os.unlink(path)


def parse(pdf_bytes: bytes, parser: str) -> tuple[list[dict], str, str | None]:
    """run_parser with a fallback when GROBID is down.  ParserError isn't
    cached, so the next run retries GROBID."""
    try:
        refs, dbg = run_parser(pdf_bytes, parser)
        return refs, dbg, None
    except ParserError as e:
        refs, dbg = run_parser(pdf_bytes, "pymupdf4llm")
        return refs, dbg, (f"{e}. Used the simpler pymupdf4llm parser instead — "
                           f"try again in a few minutes for the full parse.")


class LLMRepairIncomplete(Exception):
    def __init__(self, refs_json: str, stats: dict[str, Any]):
        super().__init__(stats.get("first_error") or "LLM repair incomplete")
        self.refs_json, self.stats = refs_json, stats


@st.cache_data(**_CACHE)
def run_llm_repair(refs_json: str, backend: str, model: str) -> str:
    from llm_parser import parse_references_with_llm
    stats: dict[str, Any] = {}
    out = parse_references_with_llm(json.loads(refs_json), stats=stats,
                                    backend=backend, model=model)
    if stats.get("n_failed_refs") or (stats.get("first_error") and not stats.get("n_sent")):
        # Raising keeps a failed run out of the cache.
        raise LLMRepairIncomplete(json.dumps(out), stats)
    return json.dumps(out)


def llm_repair(refs: list[dict]) -> tuple[list[dict], str | None]:
    """Cached LLM repair → (refs, warning).  Incomplete runs are memoised
    per session so reruns don't re-bill the LLM; a new session retries."""
    refs_json = json.dumps(refs)
    memo = ("llm_fail", hash((refs_json, LLM_BACKEND, LLM_MODEL)))
    if memo in st.session_state:
        return st.session_state[memo]
    try:
        return json.loads(run_llm_repair(refs_json, LLM_BACKEND, LLM_MODEL)), None
    except LLMRepairIncomplete as e:
        s = e.stats
        msg = (f"LLM repair failed for {s['n_failed_refs']} of {s['n_sent']} references "
               f"({s.get('first_error')}); those keep the parser's fields."
               if s.get("n_sent") else f"LLM repair unavailable: {s.get('first_error')}")
        st.session_state[memo] = (json.loads(e.refs_json), msg)
        return st.session_state[memo]


@st.cache_data(**_CACHE)
def run_highlighter(pdf_bytes: bytes, refs_json: str, statuses_json: str = "[]",
                    bake_color: bool = False):
    return highlight_references(pdf_bytes, json.loads(refs_json),
                                json.loads(statuses_json) or None, bake_color=bake_color)


def bibliography_orphans(located: list[dict], orphans: list[dict]) -> list[dict]:
    """Keep orphan blocks from the bibliography onwards.  The bibliography
    starts at the first page holding >= 3 located references; a single
    reference "located" earlier is usually a match inside body text."""
    per_page: dict[int, int] = {}
    for r in located:
        if r.get("found") and r.get("page"):
            per_page[r["page"]] = per_page.get(r["page"], 0) + 1
    starts = [p for p, n in sorted(per_page.items()) if n >= 3]
    if not starts:
        return orphans
    return [o for o in orphans if (o or {}).get("page", 0) >= starts[0]]


@st.cache_data(**_CACHE)
def auto_add_orphans(orphans_json: str) -> str:
    """Structure high-confidence orphan blocks (bibliography text no parsed
    ref claimed) with GROBID and return them as auto-detected refs."""
    high = [o for o in json.loads(orphans_json) if (o or {}).get("confidence") == "high"]
    if not high:
        return "[]"
    try:
        parsed = _grobid_process_citation_list([o["text"] for o in high])
    except Exception:
        return "[]"      # GROBID down: orphans stay in the suggestions list
    out = []
    for orph, p in zip(high, parsed):
        if p:
            out.append({**p, "_auto_added": True, "raw": orph["text"], "page": orph["page"],
                        "rect": orph["rect"], "found": True, "status": "pending"})
    return json.dumps(out)


@st.cache_data(**_CACHE)
def find_text_rect(pdf_bytes: bytes, raw: str, page_hint: int | None):
    """Locate a manual ref in the PDF by fuzzy-matching its text against
    text blocks → ``(rect_pdf_coords, page) | (None, None)``.  Keyed on the
    PDF content too (the old ``_pdf_bytes`` parameter was excluded from the
    cache key, so another user's PDF could supply the rectangle)."""
    import fitz
    from rapidfuzz import fuzz
    from highlighter import _clean_for_search

    needle = _clean_for_search(raw or "").lower()
    if len(needle) < 15:
        return None, None
    best, best_page, best_score = None, None, 0
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page_hint - 1] if page_hint else range(len(doc))
        for pn in pages:
            if not 0 <= pn < len(doc):
                continue
            page = doc[pn]
            for b in page.get_text("blocks"):
                text = (b[4] or "").strip() if len(b) >= 5 else ""
                if len(text) < 30:
                    continue
                score = fuzz.partial_ratio(needle, _clean_for_search(text).lower())
                if score > best_score:
                    best_score = score
                    if score >= 75:
                        h = page.rect.height        # top-down → PDF bottom-up
                        best, best_page = [b[0], h - b[3], b[2], h - b[1]], pn + 1
    return best, best_page


def structure_citation(text: str) -> dict | None:
    """Structure one pasted / selected citation with GROBID.  Raises
    ParserError when GROBID is unreachable."""
    parsed = _grobid_process_citation_list([text.strip()])
    return {**parsed[0], "raw": text.strip()} if parsed and parsed[0] else None


# ---------------------------------------------------------------------------
# The whole document
# ---------------------------------------------------------------------------

@dataclass
class Document:
    refs: list[dict]                 # parsed + auto + manual, located in the PDF
    n_base: int                      # parsed + auto (manual refs start here)
    pdf: bytes                       # PDF bytes for the viewer
    orphans: list[dict]              # possible missed refs (low confidence)
    debug: str
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


def process(pdf_bytes: bytes, parser: str, use_llm: bool,
            log: Callable[[str], None] | None = None) -> Document:
    """Run (or replay from cache) the processing pipeline for the parsed
    and auto-detected refs; add manual refs with ``with_manual``.  ``log``
    gets one line per stage when the caller shows progress."""
    say = log or (lambda _m: None)
    t0 = time.time()
    warnings: list[str] = []
    timings: dict[str, float] = {}

    say(f"Parsing references with **{parser}** …")
    refs, debug, warn = parse(pdf_bytes, parser)
    if warn:
        warnings.append(warn)
    timings["parse"] = time.time() - t0
    say(f"Found **{len(refs)}** references ({timings['parse']:.1f}s)")

    if use_llm and refs:
        t = time.time()
        say(f"Repairing fields with **{LLM_MODEL}** …")
        refs, warn = llm_repair(refs)
        if warn:
            warnings.append(warn)
        timings["repair"] = time.time() - t

    refs, n_merged = dedupe_references(refs)
    if n_merged:
        say(f"Merged **{n_merged}** duplicate reference{'s' if n_merged != 1 else ''}")

    t = time.time()
    say("Locating references in the PDF …")
    pdf, located, orphans = run_highlighter(pdf_bytes, json.dumps(refs))
    # Missed references can only sit on the bibliography pages.  Body-text
    # paragraphs earlier in the paper (which cite papers and so "look like"
    # references) were being auto-added as references.
    orphans = bibliography_orphans(located, orphans)
    auto = json.loads(auto_add_orphans(json.dumps(orphans)))
    orphans = [o for o in orphans if (o or {}).get("confidence") != "high"]
    if auto:
        say(f"Auto-detected **{len(auto)}** reference{'s' if len(auto) != 1 else ''} "
            f"the parser missed")
    timings["locate"] = time.time() - t

    base = list(located) + auto
    timings["total"] = time.time() - t0
    return Document(refs=base, n_base=len(base), pdf=pdf, orphans=orphans,
                    debug=debug, warnings=warnings, timings=timings)


def with_manual(doc: Document, pdf_bytes: bytes, manual: list[dict]) -> Document:
    """``doc`` with the user's manual refs appended (always at the tail)."""
    refs = doc.refs[:doc.n_base] + [_locate_manual(pdf_bytes, m) for m in manual]
    return Document(refs=refs, n_base=doc.n_base, pdf=doc.pdf, orphans=doc.orphans,
                    debug=doc.debug, warnings=doc.warnings, timings=doc.timings)


def _locate_manual(pdf_bytes: bytes, mr: dict) -> dict:
    """Give a manual ref the highlighter's shape: rect/page from the PDF
    selection that created it, else from fuzzy text search."""
    entry = dict(mr)
    rect, page = mr.get("rect"), mr.get("page")
    if not rect:
        rect, found_page = find_text_rect(pdf_bytes, mr.get("raw") or "", page)
        page = found_page or page
    entry.update(found=bool(rect), page=page, rect=rect or None)
    return entry


@st.cache_data(**_CACHE)
def annotated_pdf(pdf_bytes: bytes, refs_json: str, statuses_json: str) -> bytes:
    """PDF with status colours baked in — for the download button."""
    return highlight_references(pdf_bytes, json.loads(refs_json),
                                json.loads(statuses_json), bake_color=True)[0]
