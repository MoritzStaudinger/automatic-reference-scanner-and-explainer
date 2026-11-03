"""
PDF parsing backends.

Each function receives a path to a PDF file and returns a list of reference
dicts with (at least) these optional keys:
    raw     – the raw reference string as it appeared in the PDF
    title   – extracted title (if parseable)
    authors – extracted author string (if parseable)
    year    – extracted year (if parseable)
    venue   – journal / conference name (if parseable)
    doi     – DOI string (if parseable)
    url     – URL (if parseable)
"""

from __future__ import annotations

import arse_env  # noqa: F401  (accepts the old ARES_* settings)
import os
import time
import threading
import re
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from typing import Any

import requests as _requests

GROBID_URL: str = os.getenv("GROBID_URL", "http://localhost:8070")
GROBID_TIMEOUT: float = float(os.getenv("GROBID_TIMEOUT", "90"))
_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# GROBID runs with a 2 GB heap, but its native PDF layer (pdfalto) uses
# memory outside the heap: one 3 MB paper peaked at ~3.8 GB.  Two parallel
# full-document parses got the container OOM-killed at mem_limit 4g, which
# surfaced as random parsing failures.  Full-document parses are therefore
# serialised per process; the light citation-list calls get two slots.
_GROBID_SLOTS = threading.BoundedSemaphore(int(os.getenv("ARSE_GROBID_CONCURRENCY", "2")))
_GROBID_FULLTEXT_SLOTS = threading.BoundedSemaphore(
    int(os.getenv("ARSE_GROBID_FULLTEXT_CONCURRENCY", "1")))


class ParserError(RuntimeError):
    """A parser backend failed (e.g. GROBID down) — callers should tell
    the user and fall back instead of showing "no references found"."""


def _grobid_post(endpoint: str, **kwargs) -> _requests.Response:
    """POST to GROBID with bounded concurrency and one retry on 503 /
    connection failure.  Raises ParserError when GROBID can't answer."""
    url = f"{GROBID_URL}/api/{endpoint}"
    last: Exception | None = None
    for attempt in range(2):
        files = kwargs.get("files")
        if files and attempt:
            for fh in files.values():
                fh.seek(0)
        heavy = _GROBID_FULLTEXT_SLOTS if endpoint == "processFulltextDocument" else nullcontext()
        with heavy, _GROBID_SLOTS:
            try:
                r = _requests.post(url, timeout=GROBID_TIMEOUT, **kwargs)
            except (_requests.ConnectionError, _requests.Timeout) as e:
                last = e
            else:
                if r.status_code == 503 and attempt == 0:
                    last = RuntimeError("GROBID busy (503)")
                elif r.ok:
                    return r
                else:
                    raise ParserError(f"GROBID returned HTTP {r.status_code} for {endpoint}")
        if attempt == 0:
            time.sleep(2)
    _grobid_alive_cache.update(ts=0.0)
    raise ParserError(f"GROBID unavailable ({type(last).__name__}: {last})"[:300])


# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_DOI_RE = re.compile(r"10\.\d{4,}/\S+")
_URL_RE = re.compile(r"https?://\S+")

# Matches a reference entry marker at the start of a line:
# - [1], * [1], [1], (1), 1., 1)
_REF_MARKER_RE = re.compile(r"^\s*(?:[-*]\s+)?[\[\(]?\d+[\]\)\.]\s+")

# Heading patterns for the reference section (markdown or plain)
_REF_SECTION_RE = re.compile(
    r"(?:^|\n)"                          # start of string or after newline
    r"\s*#{0,4}\s*"                      # optional markdown heading hashes
    r"\*{0,2}"                           # optional bold markers
    r"(?:References?|Bibliography|Works\s+Cited|Literature)"
    r"\*{0,2}"
    r"\s*\n",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _locate_reference_section(full_text: str) -> str | None:
    """Return the portion of *full_text* that starts after the reference heading.

    Takes the last heading that is followed by a reasonable amount of text:
    a "References" line can also appear earlier (table of contents, a
    figure, a related-work paragraph)."""
    matches = list(_REF_SECTION_RE.finditer(full_text))
    if not matches:
        return None

    def looks_like_bibliography(m: re.Match) -> bool:
        # A figure label "Literature" or a TOC line isn't followed by a
        # run of publication years; a bibliography is.
        window = full_text[m.end(): m.end() + 1500]
        return len(re.findall(r"\b(?:19|20)\d{2}\b", window)) >= 4

    for match in reversed(matches):
        if looks_like_bibliography(match):
            return full_text[match.end():]
    return full_text[matches[0].end():]


_NUM_MARKER_RE = re.compile(r"^\s*(?:[-*]\s+)?(?:\[(\d{1,3})\]|\((\d{1,3})\)|(\d{1,3})[.)])\s+")


def _split_numbered(text: str) -> list[str]:
    """Split a numbered bibliography ([1] …, 1. …, (1) …) at its markers.

    A marker only counts when its number continues the sequence (n+1), so
    wrapped lines that happen to start with "2011. ACM." or "16:1–16:20"
    don't open a new entry.  Returns [] unless at least three markers form
    a sequence covering most of the candidates — the caller then falls back
    to another strategy."""
    lines = text.splitlines()
    groups: list[list[str]] = []
    expected = None
    hits = 0
    for line in lines:
        m = _NUM_MARKER_RE.match(line)
        n = int(next(g for g in m.groups() if g)) if m else None
        if n is not None and (expected is None and n <= 2 or n == expected):
            groups.append([line.strip()])
            expected = n + 1
            hits += 1
        elif groups:
            stripped = line.strip()
            if stripped:
                groups[-1].append(stripped)
    if hits < 3:
        return []
    out = []
    for g in groups:
        # Re-join hyphenated line breaks ("Longeval-\nretrieval" stays,
        # "evalu-\nation" → "evaluation") and collapse the rest.
        joined = g[0]
        for nxt in g[1:]:
            if re.search(r"[a-z]-$", joined) and nxt[:1].islower():
                joined = joined[:-1] + nxt
            else:
                joined += " " + nxt
        out.append(joined.strip())
    return out


def _split_into_references(text: str) -> list[str]:
    """
    Split a block of text into individual reference strings.

    Strategy:
    1. Try numbered markers ([1], 1., etc.) — join continuation lines first.
    2. Fall back to blank-line separation.
    """
    # --- Strategy 1: numbered markers ---
    numbered = _split_numbered(text)
    if numbered:
        return numbered

    # --- Strategy 2: blank-line separated ---
    paragraphs = re.split(r"\n\s*\n", text)
    refs = []
    for p in paragraphs:
        # Collapse internal newlines (continuation lines).
        collapsed = " ".join(line.strip() for line in p.splitlines() if line.strip())
        if collapsed:
            refs.append(collapsed)
    return refs


def _parse_raw_reference(raw: str) -> dict[str, Any]:
    """
    Best-effort field extraction from a raw reference string.

    Handles the most common academic citation styles:
      IEEE:  [1] Authors, "Title," Venue, year.
      APA:   Authors (year). Title. Venue.
      ACM:   Authors. year. Title. Venue.
      plain: Authors. Title. Venue, year.  ← year at end
    """
    ref: dict[str, Any] = {"raw": raw}

    # Strip leading marker (- [1], [1], 1., etc.) and markdown bold/italic
    body = _REF_MARKER_RE.sub("", raw).strip()
    body = re.sub(r"\*+", "", body)
    body = re.sub(r"_+", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    # --- Year ---
    year_match = _YEAR_RE.search(body)
    if year_match:
        ref["year"] = year_match.group(0)

    # --- DOI ---
    doi_match = _DOI_RE.search(body)
    if doi_match:
        ref["doi"] = doi_match.group(0).rstrip(".,)")

    # --- URL ---
    url_match = _URL_RE.search(body)
    if url_match:
        ref["url"] = url_match.group(0).rstrip(".,)")

    # --- IEEE style: title in double quotes ---
    ieee_title = re.search(r'"([^"]{10,})"', body)
    if ieee_title:
        ref["title"] = ieee_title.group(1).strip()
        before_quote = body[: ieee_title.start()].strip().rstrip(",")
        if before_quote:
            ref["authors"] = before_quote
        after_quote = body[ieee_title.end():].strip().lstrip(",").strip()
        if after_quote:
            venue_candidate = _YEAR_RE.split(after_quote)[0].strip().strip(".,")
            if venue_candidate:
                ref["venue"] = venue_candidate
        return ref

    # --- APA style: Authors (year). Title. Venue. ---
    # Distinguishing feature: year is in parentheses right after authors.
    apa_match = re.search(r"\((" + _YEAR_RE.pattern + r")\)\s*\.", body)
    if apa_match:
        ref["authors"] = body[: apa_match.start()].strip().rstrip(".,")
        after_year = body[apa_match.end():].strip()
        parts = re.split(r"\.\s+", after_year, maxsplit=2)
        if parts:
            ref["title"] = parts[0].strip().rstrip(".,")
        if len(parts) > 1:
            ref["venue"] = parts[1].strip().rstrip(".,")
        return ref

    # --- Numbered / plain style: Authors. Title. Venue, year. ---
    # Year is NOT in parens (it appears at the end). Split on ". " first.
    sentences = re.split(r"\.\s+", body)
    # Drop trailing empty/very-short segments and URL/DOI fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 4]

    if len(sentences) >= 3:
        ref["authors"] = sentences[0]
        ref["title"] = sentences[1]
        # Venue is the rest up to (but not including) a trailing year
        venue_raw = ". ".join(sentences[2:])
        venue_clean = _YEAR_RE.sub("", venue_raw).strip(" .,;:")
        if venue_clean:
            ref["venue"] = venue_clean
    elif len(sentences) == 2:
        ref["authors"] = sentences[0]
        ref["title"] = sentences[1].rstrip(".,")
    elif sentences:
        ref["title"] = sentences[0].rstrip(".,")

    return ref


# Body-text opener patterns — paragraphs from the paper that the PDF parser
# sometimes grabs as "references".  Conservative list: only fires when one
# of these patterns is at the very start of `raw`.
_BODY_TEXT_OPENER = re.compile(
    r"^(?:we |our |this |these |those |another |similarly|additionally|"
    r"however|furthermore|moreover|notably|representing |encoding |"
    r"to encode|to represent|to generalize|using |for example|"
    r"in particular|notation\b|figure \d|table \d|equation |"
    r"as shown|as the|consider |suppose |let \w|recall that|"
    r"vsas? |hrrs? |cot )",
    re.IGNORECASE,
)


_OPENER_NOISE_RE = re.compile(r"^[\s\-\[\]\*_~`0-9.]+")


def _looks_like_body_text(raw: str) -> bool:
    """
    Heuristic: does this string look like a paragraph from the paper's body
    rather than a bibliographic entry?

    Used as a fallback when LLM repair is unavailable.  Conservative — only
    fires when the opener pattern matches AND the raw has none of the
    classic citation anchors (year, DOI, URL).
    """
    head = _OPENER_NOISE_RE.sub("", raw)
    if not _BODY_TEXT_OPENER.match(head):
        return False
    has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", raw))
    has_doi  = "10." in raw and "/" in raw
    has_url  = "http" in raw
    if has_year or has_doi or has_url:
        return False
    return True


# Regex parser sometimes drops a name fragment into `title` when the field
# split mis-fires on comma-separated author lists.  Examples from real PDFs:
# "Li, P", "Saha, S", "Kabatiansky and V.I", "Zhang, and H".  These are
# unrecoverable without LLM repair; flag them so they're excluded from the
# match-rate denominator instead of poisoning it.
_NAME_FRAGMENT_TITLE_RE = re.compile(
    r"^(?:[A-Z][a-zA-Z\-']+(?:,\s*[A-Z]\.?)+|"           # "Li, P" / "Smith, J., K."
    r"[A-Z][a-zA-Z\-']+\s+and\s+[A-Z]\.?[A-Z]?\.?|"      # "Kabatiansky and V.I"
    r"[A-Z][a-zA-Z\-']+,\s*and\s+[A-Z]\.?)$"             # "Zhang, and H"
)


def _looks_like_name_fragment_title(title: str) -> bool:
    """True if `title` is almost certainly a leaked author fragment."""
    if not title:
        return False
    t = title.strip().rstrip(".,;:")
    if len(t.split()) > 4:
        return False
    return bool(_NAME_FRAGMENT_TITLE_RE.match(t))


def _is_plausible_reference(ref: dict[str, Any]) -> bool:
    """Filter out obvious noise: too short, body-text paragraphs, no anchors."""
    raw = ref.get("raw", "")
    if len(raw) < 30:
        return False
    if _looks_like_body_text(raw):
        return False
    # The last entry sometimes swallows the appendix that follows the
    # bibliography (figure captions, tables, prompts).
    if len(raw) > 1500 and re.search(r"(?:^|\s)(?:Figure|Table|Algorithm)\s+\d+\s*:", raw):
        return False
    if _looks_like_name_fragment_title(ref.get("title", "")):
        # Real ref but with a corrupted title field; mark so lookup skips
        # AND it's excluded from the match-rate denominator.
        ref["_not_a_citation"] = True
        ref["_corruption"] = "name_fragment_title"
        return True   # keep for visibility in the report

    # Backfill year from raw if GROBID didn't structure it.
    year_in_raw = re.search(r"\b(?:19|20)\d{2}\b", raw)
    if year_in_raw and not ref.get("year"):
        ref["year"] = year_in_raw.group(0)

    # A reference needs at least one anchor: an explicit year/doi/url field,
    # OR a year in the raw text, OR a comma+period combo (typical citation
    # punctuation).  Without any of these, it's probably noise.
    has_anchor = (
        ref.get("year") or ref.get("doi") or ref.get("url")
        or year_in_raw
        or ("," in raw and "." in raw)
    )
    if not has_anchor:
        return False
    return True


# ---------------------------------------------------------------------------
# docling backend
# ---------------------------------------------------------------------------

def extract_references_docling(
    pdf_path: str, return_markdown: bool = False
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    md_text = doc.export_to_markdown()
    ref_section = _locate_reference_section(md_text)
    source = ref_section if ref_section else md_text

    raw_refs = _split_into_references(source)
    references = [_parse_raw_reference(r) for r in raw_refs]
    filtered = [r for r in references if _is_plausible_reference(r)]

    if return_markdown:
        return filtered, md_text
    return filtered


# ---------------------------------------------------------------------------
# pymupdf4llm backend
# ---------------------------------------------------------------------------

def extract_references_pymupdf4llm(
    pdf_path: str, return_markdown: bool = False
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    import pymupdf4llm

    md_text: str = pymupdf4llm.to_markdown(pdf_path, use_ocr=False)
    ref_section = _locate_reference_section(md_text)
    source = ref_section if ref_section else md_text

    raw_refs = _split_into_references(source)
    references = [_parse_raw_reference(r) for r in raw_refs]
    filtered = [r for r in references if _is_plausible_reference(r)]

    if return_markdown:
        return filtered, md_text
    return filtered


# ---------------------------------------------------------------------------
# GROBID backend
# ---------------------------------------------------------------------------

_grobid_alive_cache: dict[str, Any] = {"ts": 0.0, "ok": False}


def grobid_is_available(max_age: float = 30.0) -> bool:
    """Return True if the GROBID service is reachable (cached ``max_age`` s,
    so UI reruns don't each pay a network round trip)."""
    now = time.monotonic()
    if now - _grobid_alive_cache["ts"] < max_age:
        return _grobid_alive_cache["ok"]
    try:
        ok = _requests.get(f"{GROBID_URL}/api/isalive", timeout=3).status_code == 200
    except Exception:
        ok = False
    _grobid_alive_cache.update(ts=now, ok=ok)
    return ok


def _clean_doi(value: str) -> str:
    """Bare DOI from GROBID's idno, which can hold a whole URL
    ("…/article?id=10.1371/journal.pone.0163309") or LaTeX escapes
    ("10.1007/978-3-319-06028-6\\_97")."""
    value = value.replace("\\", "")
    m = re.search(r"10\.\d{4,9}/\S+", value)
    return m.group(0).rstrip(".,;)]") if m else ""


_ORG_WORD_RE = re.compile(
    r"\b(?:Committee|Group|Organi[sz]ation|Institute|Consortium|Association|"
    r"Alliance|Observatory|Foundation|Council|Agency|Society|Working|Standardi[sz]ation|"
    r"Team|Contributors|Project|Initiative|Lab|Laboratory|Inc|Ltd|Corporation|Office)\b")
_NAME_LIST_WORDS = {"and", "of", "for", "the", "on", "&", "others", "et", "al", "al."}
_VENUE_TITLE_RE = re.compile(
    r"^(?:In\s+)?(?:Proceedings\b|Proc\.|Advances in\b|arXiv\b|Transactions\b|Journal of\b|"
    r"International Conference\b|Conference on\b|Findings of\b|Workshop\b)", re.I)


def _looks_like_venue(text: str) -> bool:
    return bool(_VENUE_TITLE_RE.match(text.strip()))


def _is_name_list(text: str) -> bool:
    """"Alan Schelten, Alex Vaughan, and others", "Tonja, and 1 others" — a
    title field that GROBID filled with author names."""
    words = re.findall(r"[^\s,()]+", text)
    listy = "," in text or re.search(r"\b(?:and (?:\d+ )?others|et al)\b", text)
    return (bool(listy) and len(words) >= 3
            and all(w.lower() in _NAME_LIST_WORDS or w[:1].isupper() or w.isdigit()
                    for w in words))


def _strip_name_prefix(title: str) -> str:
    """Drop an author / organisation list that GROBID glued to the front of
    the title, e.g. "Mateusz Lango, and Ondřej Dušek. Leak, cheat, repeat: …"
    or "ESIP Data … Committee. Data Citation Guideline …".

    Only when the part before the first ". " consists solely of capitalised
    words (plus and/of/for/the), contains a comma-separated name list or an
    organisation word, and what follows is still a title of 3+ words."""
    # Leftovers of a mis-split author list: "others. 2021. Learning …",
    # "and 1 others. Title", "2019a. Sentencebert: …".
    title = re.sub(r"^(?:and\s+)?(?:\d+\s+)?others\.\s+", "", title)
    title = re.sub(r"^\(?(?:19|20)\d{2}[a-z]?\)?\.\s+(?=\S)", "", title)
    head, sep, rest = title.partition(". ")
    if not sep or len(rest.split()) < 3:
        return title
    words = re.findall(r"[^\s,()]+", head)
    if not words or any(w.lower() not in _NAME_LIST_WORDS and not w[:1].isupper()
                        and not w.isdigit() for w in words):
        return title
    has_list = (", and " in head or (", " in head and len(words) >= 4)
                or re.search(r"\b(?:and others|et al)\b", head))
    if not (has_list or _ORG_WORD_RE.search(head)):
        return title
    # ACL / author-year style also leaves the year behind: "2024. The llama 3 …"
    rest = re.sub(r"^\(?(?:19|20)\d{2}[a-z]?\)?\.\s+", "", rest.strip())
    return rest.strip()


def _parse_tei_references(tei_xml: str) -> list[dict[str, Any]]:
    """Parse GROBID TEI XML and return a list of reference dicts."""
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return []

    refs = []
    for bib in root.findall(".//tei:listBibl/tei:biblStruct", _TEI_NS):
        ref: dict[str, Any] = {}

        # Raw reference string (GROBID preserves this)
        note = bib.find("tei:note[@type='raw_reference']", _TEI_NS)
        if note is not None and note.text:
            ref["raw"] = note.text.strip()

        # Title — analytic (article) takes priority over monogr (book/chapter)
        for xpath in (
            "tei:analytic/tei:title[@type='main']",
            "tei:analytic/tei:title",
            "tei:monogr/tei:title[@level='m']",
            "tei:monogr/tei:title",
        ):
            el = bib.find(xpath, _TEI_NS)
            if el is not None and el.text:
                ref["title"] = el.text.strip()
                break

        if ref.get("title"):
            ref["title"] = _strip_name_prefix(ref["title"])

        # Authors
        authors = []
        for author in bib.findall(".//tei:author", _TEI_NS):
            forename = author.findtext("tei:persName/tei:forename", "", _TEI_NS).strip()
            surname  = author.findtext("tei:persName/tei:surname",  "", _TEI_NS).strip()
            name = f"{forename} {surname}".strip() if forename else surname
            if name:
                authors.append(name)
        if authors:
            ref["authors"] = ", ".join(authors)

        # Year
        date_el = bib.find(".//tei:imprint/tei:date[@type='published']", _TEI_NS)
        if date_el is not None:
            when = date_el.get("when", "")
            m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", when)
            if m:
                ref["year"] = m.group(0)

        # Venue — journal title preferred over series/book title.  For a
        # standalone preprint, GROBID stores the title in monogr/title[level='m']
        # AND that's the only "venue-like" element — we'd otherwise duplicate
        # the title into venue.  Skip when venue text equals title.
        title_text = (ref.get("title") or "").strip()
        for xpath in (
            "tei:monogr/tei:title[@level='j']",   # journal
            "tei:monogr/tei:title[@level='m']",   # book / proceedings
            "tei:series/tei:title",
        ):
            el = bib.find(xpath, _TEI_NS)
            if el is not None and el.text:
                venue_text = el.text.strip()
                if venue_text and venue_text != title_text:
                    ref["venue"] = venue_text
                    break

        # DOI
        # Identifiers sit under <analytic> or <monogr>, not on biblStruct.
        doi_el = bib.find(".//tei:idno[@type='DOI']", _TEI_NS)
        if doi_el is not None and doi_el.text:
            doi = _clean_doi(doi_el.text)
            if doi:
                ref["doi"] = doi

        # arXiv → URL
        arxiv_el = bib.find(".//tei:idno[@type='arXiv']", _TEI_NS)
        if arxiv_el is not None and arxiv_el.text:
            ref["url"] = f"https://arxiv.org/abs/{arxiv_el.text.strip()}"

        # Web pointer fallback
        if not ref.get("url"):
            ptr = bib.find(".//tei:ptr[@type='web']", _TEI_NS)
            if ptr is None:
                ptr = bib.find(".//tei:ptr", _TEI_NS)
            if ptr is not None:
                ref["url"] = ptr.get("target", "")

        # Reconstruct raw if GROBID didn't provide one
        if "raw" not in ref:
            parts = [p for p in (ref.get("authors"), ref.get("title"),
                                  ref.get("venue"), ref.get("year")) if p]
            ref["raw"] = ". ".join(parts)

        if ref.get("raw") or ref.get("title"):
            refs.append(ref)

    return refs


def extract_references_grobid(
    pdf_path: str, return_markdown: bool = False
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    """
    Extract references using the GROBID service (must be running on GROBID_URL).
    ``return_markdown`` returns the raw TEI XML for debugging (not actual markdown).
    """
    with open(pdf_path, "rb") as fh:
        response = _grobid_post(
            "processFulltextDocument",
            files={"input": fh},
            # GROBID's parameter names: the old "includeRawReferences" /
            # "consolidateReferences" were silently ignored, so every raw
            # string was reconstructed from parsed fields — dropping DOIs,
            # arXiv IDs and URLs and breaking highlight matching.
            data={"consolidateCitations": "0",
                  "includeRawCitations": "1"},
        )
    tei_xml = response.text

    references = _parse_tei_references(tei_xml)
    filtered   = [ref for ref in references if _is_plausible_reference(ref)]

    if return_markdown:
        return filtered, tei_xml
    return filtered


# ---------------------------------------------------------------------------
# Hybrid backend — pymupdf isolates the References section, GROBID structures
# the individual citation strings via /api/processCitationList
# ---------------------------------------------------------------------------

# Patterns that mark the END of the References section (start of appendix etc).
_REF_SECTION_END_RE = re.compile(
    r"\n\s*(?:"
    r"#{1,6}\s*"                                      # markdown heading
    r"(?:A(?:ppendix)?\b|Supplementary|Limitations?|Ethics?|"
    r"Broader\s+Impact|Acknowledg|Author\s+Contributions)|"
    r"[A-Z]\.?\s+(?:Appendix|Supplementary|Limitations?|Ethics?)|"
    # "*A. Implementation*", "Appendix B".  The letter after "A" must be a
    # real capital: under IGNORECASE "[A-Z]" also matched "A replicability
    # study" — a wrapped title line — and cut the bibliography in half.
    r"\*{0,2}(?:A\.?|Appendix)[ \t]+(?:(?-i:[A-Z])(?-i:[a-z]*)\b(?![ \t]+(?-i:[a-z]))|\d)|"
    r"\d+\s+Appendix|"
    # ACL style: the appendix letter alone on its line, title on the next
    # ("A\nSystem Screen Shot", "A.1\nPrompts").
    r"(?-i:A)(?:\.\d+)?\.?[ \t]*\n[ \t]*(?-i:[A-Z][a-z]+)|"
    # A figure / table caption can't be part of a bibliography.
    r"(?-i:Figure|Table)\s+\d+\s*[:.]\s"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _trim_to_section_end(section_text: str) -> str:
    """Return the prefix of *section_text* before the next major section heading."""
    m = _REF_SECTION_END_RE.search(section_text)
    return section_text[: m.start()] if m else section_text


def _looks_like_ref_candidate(raw: str, require_year: bool = True) -> bool:
    """Quick filter: must look like a citation, not body text from an appendix."""
    if len(raw) < 30:
        return False
    if _looks_like_body_text(raw):
        return False
    # Needs a 4-digit year — every modern citation has one
    if require_year and not re.search(r"\b(?:19|20)\d{2}\b", raw):
        return False
    return True


def _grobid_process_citation_list(citations: list[str]) -> list[dict[str, Any]]:
    """Structure raw citation strings via GROBID's /api/processCitationList."""
    if not citations:
        return []
    # form-encoded with repeated `citations` parameter; ask for TEI XML
    payload = [("citations", c) for c in citations]
    response = _grobid_post(
        "processCitationList",
        data=payload,
        headers={"Accept": "application/xml"},
    )
    tei = response.text
    refs = _parse_tei_references(tei)
    # processCitationList preserves input order.  Force raw to the input
    # string — GROBID sometimes reconstructs a partial raw from its structured
    # fields, losing the arXiv ID / "preprint arXiv:NNNN.NNNN" suffix that
    # `_extract_arxiv_id` relies on for the arXiv batch prewarm.
    for r, c in zip(refs, citations):
        r["raw"] = c
    return refs


def _split_plaintext_bibliography(section: str) -> list[str]:
    """
    Split a PDF-extracted plain-text references section into individual ref
    strings.  The native PyMuPDF text retains every hard-wrap newline and
    has no blank lines between refs, so neither numbered-marker nor
    blank-line splitters work.

    Strategy: anchor on each YYYY. (publication year) — there's exactly one
    per ref.  Between two consecutive years, the LAST `". [A-Z]"` is the
    boundary where ref-N's venue ends and ref-(N+1)'s authors begin.  All
    earlier `". [A-Z]"` candidates in that span are typically inside the
    title or venue (subtitles, "vol. N", "arXiv:NNNN.").
    """
    # Un-hyphenate: word- continued on next line ("evalu-\nation"), but
    # keep real hyphens before a capital ("Callison-\nBurch").
    text = re.sub(r"-\s*\n\s*(?=[a-zà-ÿ])", "", section)
    text = re.sub(r"-\s*\n\s*", "-", text)
    # Drop page footers (a line that is just digits, often a page number)
    text = re.sub(r"\n\s*\d{1,5}\s*\n", "\n", text)
    # Collapse all whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # A year, not the end of a page range ("33:1877–1901.") or an ID.
    year_positions = [m.end() for m in re.finditer(
        r"(?<![–\-:/.\d])\b(?:19|20)\d{2}[a-z]?\.", text)]
    if len(year_positions) < 2:
        return [text] if len(text) >= 30 else []

    boundaries: list[int] = [0]
    for i in range(len(year_positions) - 1):
        # Search the span between this year and the next for the boundary
        span_start = year_positions[i]
        span_end   = year_positions[i + 1]
        span       = text[span_start:span_end]
        # Find ALL ". X" with X a capital starting a name-like token.  We
        # want the LAST one — that's where the next ref's authors begin.
        # Tighten the lookahead so we don't fire on ". The" (title words):
        # an author start is typically followed by lowercase letters (a
        # name continues), comma, period (initial), or "and".
        # Negative lookbehind `(?<![\s.][A-Z])` prevents matching where the
        # period is part of an initial token — either a lone initial
        # (" Q." → space + capital) or chained initials ("G.A." → period +
        # capital).  Without this we'd split between an initial and its
        # surname (e.g. ". Wang" instead of "Q. Wang. 2023.").
        # We still need to allow legitimate splits after an all-caps
        # acronym like "OpenAI." — there the I is preceded by another
        # capital (not [\s.]), so the lookbehind correctly allows the split.
        cand = list(re.finditer(
            r"(?<![\s.][A-Z])\.\s(?=[A-Z](?:[a-z]+|\.|\s+(?:and|\d+\s+others)))",
            span,
        ))
        if not cand:
            continue
        last = cand[-1]
        # boundary is just AFTER the ". " preceding the new ref's first author
        boundaries.append(span_start + last.end())
    boundaries.append(len(text))

    refs: list[str] = []
    for i in range(len(boundaries) - 1):
        chunk = text[boundaries[i] : boundaries[i + 1]].strip()
        if len(chunk) >= 30:
            refs.append(chunk)
    return refs


def _extract_via_section_pipeline(pdf_path: str) -> list[dict[str, Any]]:
    """
    Slice the References section out of the PDF's raw text (via PyMuPDF
    direct text extraction — no OCR, no markdown), split it into candidate
    citation strings, and have GROBID's /api/processCitationList structure
    each one.  Returns [] if the section can't be located or no candidate
    looks like a citation.

    We use raw `page.get_text()` instead of `pymupdf4llm.to_markdown` because
    pymupdf4llm/PyMuPDF auto-triggers Tesseract OCR on appendix pages with
    figures/equations, adding 10-30 s per page — a 1-3 minute freeze on
    papers with extensive appendices.
    """
    full_text = _plain_text(pdf_path)
    section = _locate_reference_section(full_text)
    if not section:
        return []
    section = _trim_to_section_end(section)
    # Numbered bibliographies split exactly at their markers; only
    # author-year lists need the year-anchor heuristic.
    candidates_raw = _split_numbered(section)
    numbered = bool(candidates_raw)
    if not numbered:
        candidates_raw = _split_plaintext_bibliography(section)
    if len(candidates_raw) < 3:
        candidates_raw = _split_into_references(section)
    # A numbered entry is delimited by its markers, so it needn't carry a
    # year to count (standards, software, "n.d." web pages).
    candidates = [c for c in candidates_raw
                  if _looks_like_ref_candidate(c, require_year=not numbered)]
    if not candidates:
        return []
    refs = _grobid_process_citation_list(candidates)   # ParserError propagates
    return [r for r in refs if _is_plausible_reference(r)]


def _plain_text(pdf_path: str) -> str:
    """PyMuPDF text of every page (no OCR, no markdown conversion), with
    page furniture removed: line numbers of review manuscripts, page
    numbers, and running headers / footers that repeat on most pages.
    Left in, they land in the middle of references that cross a page."""
    import fitz
    with fitz.open(pdf_path) as doc:
        pages = [page.get_text().splitlines() for page in doc]
    return "\n".join("\n".join(lines) for lines in _strip_page_furniture(pages))


def _furniture_key(line: str) -> str:
    """Line with digits removed — "Manuscript … 31" and "… 32" compare equal."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _strip_page_furniture(pages: list[list[str]]) -> list[list[str]]:
    from collections import Counter
    n_pages = len(pages)
    # Line / page numbers first, so the header is "first line" again.
    pages = [[l for l in lines if not re.fullmatch(r"\d{1,5}", l.strip())] for lines in pages]
    counts: Counter[str] = Counter()
    for lines in pages:
        # Headers / footers sit in the first or last few lines of a page.
        edge = lines[:4] + lines[-4:]
        counts.update({_furniture_key(l) for l in edge if l.strip()})
    min_pages = max(3, n_pages // 3)
    repeated = {k for k, c in counts.items() if c >= min_pages and len(k) <= 120}
    out = []
    for lines in pages:
        kept = []
        for i, line in enumerate(lines):
            at_edge = i < 4 or i >= len(lines) - 4
            if at_edge and _furniture_key(line) in repeated:
                continue
            kept.append(line)
        out.append(kept)
    return out


def _norm_for_dedup(text: str) -> str:
    """Lowercase + strip + collapse whitespace + drop punctuation, for fuzzy dedup."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def _distinct_refs(a: dict[str, Any], b: dict[str, Any], check_year: bool = True) -> bool:
    """True when two similar-titled refs are evidently different works:
    the titles carry different numbers ("CLEF 2017 …" vs "CLEF 2018 …",
    "Part 1" vs "Part 2"), or both have a year / DOI and they differ."""
    ta, tb = (r.get("title") or "" for r in (a, b))
    garbage = any(_looks_like_venue(t) or _is_name_list(t) for t in (ta, tb))
    na, nb = (set(re.findall(r"\d+", t)) for t in (ta, tb))
    # "2017" vs "2018" separates works; {1, 32} vs {1} is a lost "-32b"
    # from hyphenation, not a different work.
    if na and nb and not (na <= nb or nb <= na) and not garbage:
        return True
    ya, yb = (a.get("year") or "").strip(), (b.get("year") or "").strip()
    if check_year and ya and yb and ya != yb:
        return True
    da, db = (a.get("doi") or "").lower().strip(), (b.get("doi") or "").lower().strip()
    return bool(da and db and da != db)


def dedupe_references(
    refs: list[dict[str, Any]],
    title_threshold: int = 88,
) -> tuple[list[dict[str, Any]], int]:
    """
    Collapse near-duplicate references in-place.

    Two refs are considered duplicates when their normalised titles match
    at `token_sort_ratio >= title_threshold` (default 88) — strict enough
    to avoid false-merges of "X" vs "X: A Survey", loose enough to catch
    "Foo bar" / "Foo Bar." / "Foo  bar".

    Returns (deduped_list, n_merged).  Field-level merge picks the LONGEST
    non-empty value for each field across duplicates (titles, authors,
    venues sometimes get truncated in one source but full in another).
    Each kept ref gains a `_merged_from` list of duplicate raws so the
    user can see what was folded together.

    Each ref also gets ``_orig_index`` (its position in the parser output)
    and merged refs collect ``_merged_orig_indices``, so numeric citation
    markers ("[12]") still resolve to the right entry after merging.
    """
    from rapidfuzz import fuzz as _fuzz
    if not refs:
        return [], 0
    for i, r in enumerate(refs):
        if not r.get("_user_added") and not r.get("_auto_added"):
            r.setdefault("_orig_index", i)

    kept: list[dict[str, Any]] = []
    kept_titles_norm: list[str] = []
    merged_count = 0

    def _merge_into(canonical: dict[str, Any], dup: dict[str, Any]) -> None:
        # Pick the longer non-empty value for each field
        for field in ("title", "authors", "year", "venue", "doi", "url"):
            cur = (canonical.get(field) or "").strip()
            new = (dup.get(field) or "").strip()
            if new and (not cur or len(new) > len(cur)):
                canonical[field] = new
        # Track what we merged for the audit card
        canonical.setdefault("_merged_from", []).append(
            (dup.get("raw") or "")[:200]
        )
        if "_orig_index" in dup:
            canonical.setdefault("_merged_orig_indices", []).append(dup["_orig_index"])

    for ref in refs:
        # User-added refs are NEVER auto-merged — they're explicit user input.
        if ref.get("_user_added"):
            kept.append(ref)
            kept_titles_norm.append(_norm_for_dedup(ref.get("title") or ""))
            continue

        t = _norm_for_dedup(ref.get("title") or "")
        if not t:
            # No title to compare on — keep as-is (junk filter handles this)
            kept.append(ref)
            kept_titles_norm.append("")
            continue

        # Find a fuzzy title match among already-kept refs
        match_idx = -1
        for i, kt in enumerate(kept_titles_norm):
            if (kt and _fuzz.token_sort_ratio(t, kt) >= title_threshold
                    and not kept[i].get("_user_added")
                    and not _distinct_refs(ref, kept[i])):
                match_idx = i
                break

        if match_idx >= 0:
            _merge_into(kept[match_idx], ref)
            merged_count += 1
        else:
            kept.append(ref)
            kept_titles_norm.append(t)

    return kept, merged_count


def _merge_ref_lists(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    title_threshold: int = 85,
) -> list[dict[str, Any]]:
    """
    Union two ref lists, fuzzy-deduping by title.

    `primary` is kept in full and in order; entries from `secondary` are
    added only if no `primary` ref already matches their title (fuzzy
    token_sort_ratio ≥ threshold).  When a secondary entry has no title
    but `primary` has one of similar raw text, we still drop the dupe.
    """
    from rapidfuzz import fuzz as _fuzz
    primary_titles = [_norm_for_dedup(r.get("title") or "") for r in primary]
    primary_raws   = [_norm_for_dedup(r.get("raw")   or "") for r in primary]

    merged = list(primary)
    claimed: set[int] = set()
    for s in secondary:
        st = _norm_for_dedup(s.get("title") or "")
        sr = _norm_for_dedup(s.get("raw") or "")
        if not st and not sr:
            continue

        # Best match, not the first one over the threshold: "Kernel
        # principal component analysis" must not land on an earlier
        # "Principal component analysis".  Each primary entry absorbs at
        # most one secondary entry.
        dup_idx, best = None, 0.0
        for i, (pt, pr) in enumerate(zip(primary_titles, primary_raws)):
            if i in claimed:
                continue
            # Two parses of the same citation can disagree on the year,
            # so only numbers in the title and DOIs separate them here.
            if _distinct_refs(s, primary[i], check_year=False):
                continue
            score = 0.0
            # Title match is the strong signal
            if st and pt:
                t = _fuzz.token_sort_ratio(st, pt)
                if t >= title_threshold:
                    score = t
            # Fallback: high partial overlap on raw text (one side got a
            # fragmentary or garbage title, but it is the same citation).
            # Not when the primary raw is much longer and its title is
            # nowhere in this entry: then the primary holds two citations
            # glued together and this is the one it swallowed.
            if sr and pr and min(len(sr), len(pr)) >= 50:
                r = _fuzz.partial_ratio(sr, pr)
                glued = (len(pr) >= 1.4 * len(sr) and st and pt
                         and not _is_name_list(primary[i].get("title") or "")
                         and _fuzz.partial_ratio(pt, sr) < 80)
                if r >= 90 and not glued:
                    score = max(score, r - 0.5)   # a title match wins ties
            if score > best:
                dup_idx, best = i, score
        if dup_idx is not None:
            claimed.add(dup_idx)

        if dup_idx is None:
            merged.append(s)
            continue

        # Duplicate: the secondary entry's `raw` came from pymupdf's actual
        # PDF text, while the primary entry's `raw` may be GROBID's
        # reconstructed string (initials' periods stripped, "and" dropped).
        # Prefer the PDF-extracted raw so the highlighter can locate this
        # ref via strict text search.  Keep primary's structured fields.
        s_raw = s.get("raw") or ""
        if s_raw and merged[dup_idx].get("raw") != s_raw:
            merged[dup_idx] = {**merged[dup_idx], "raw": s_raw}
        better = _better_title(merged[dup_idx].get("title") or "", s.get("title") or "")
        if better:
            merged[dup_idx] = {**merged[dup_idx], "title": better}
    return merged


# ---------------------------------------------------------------------------
# Deterministic repair against the printed citation text
# ---------------------------------------------------------------------------
# GROBID's CRF splits "Peter G. Anick and Rex A. Flynn" into forename /
# surname pairs that drift across names ("G Peter, Rex Anick, Flynn") and
# sometimes stops a title at a comma.  With the real citation text in
# ``raw`` both can be checked and fixed without an LLM.

_NAME_TOKEN_RE = re.compile(r"^(?:[A-ZÀ-ÖØ-Þ][\w'’\-]*\.?|[A-Z]\.(?:-?[A-Z]\.)*|"
                            r"de|van|von|der|den|da|di|du|la|le|dos|del|bin|al|el|ter|ten)$")


def _dehyphenate(text: str) -> str:
    """Undo line-break hyphenation: "Sori- cut" → "Soricut"; keep real
    hyphens before a capital ("Klaus- Robert" → "Klaus-Robert")."""
    text = re.sub(r"(\w)- (?=[a-zà-ÿ])", r"\1", text)
    return re.sub(r"(\w)- (?=[A-ZÀ-Þ])", r"\1-", text)


def _norm_loose(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def _title_start(body: str, title: str) -> int:
    """Index of ``title`` in ``body`` (loose match on its first words), or -1."""
    words = re.findall(r"\w+", title)[:6]
    if len(words) < 2:
        return -1
    pat = r"[\W_]+".join(re.escape(w) for w in words)
    m = re.search(pat, body, re.I)
    return m.start() if m else -1


def _names_from_raw(raw: str, title: str) -> list[str] | None:
    """Author names as printed before the title, when they are written
    "First Middle Last" (ACM, ACL, IEEE).  None when the segment can't be
    read that way — "Last, F." styles, editors, no title found."""
    body = _dehyphenate(re.sub(r"^\s*(?:\[\d+\]|\(\d+\)|\d+\.)\s*", "", raw))
    i = _title_start(body, title)
    if i <= 0:
        return None
    seg = body[:i].strip()
    seg = re.sub(r"[“\"'‘(]+$", "", seg).strip()
    # Trailing year ("2024b.", "(2020).") of author-year styles
    seg = re.sub(r"[.,]?\s*\(?(?:19|20)\d{2}[a-z]?\)?[.,:]?$", "", seg).strip()
    seg = seg.rstrip(".,:; ")
    if not seg or len(seg) > 2000:
        return None
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+|&\s+)?|\s+and\s+|\s+&\s+", seg) if p.strip()]
    names = [p for p in parts if not re.fullmatch(r"(?:\d+\s+)?others|et\s+al\.?", p, re.I)]
    if not names:
        return None
    for n in names:
        toks = n.split()
        if not 2 <= len(toks) <= 6:
            return None             # "Smith", "J." → a "Last, F." style
        if not all(_NAME_TOKEN_RE.match(t) for t in toks):
            return None
        last = toks[-1]
        if len(last.rstrip(".")) < 2 or last.endswith("."):
            return None             # ends in an initial → "Last, F. M." style
    return names


def _surname_set(names: list[str]) -> set[str]:
    return {(_norm_loose(n).split() or [""])[-1] for n in names if _norm_loose(n)}


def _extend_title(raw: str, title: str) -> str:
    """Complete a title GROBID ended at a comma: "… High-Impact Pediatrics"
    → "… High-Impact Pediatrics, Cardiology and Surgery Journals: A
    Cross-Sectional Study".  Only within the same sentence, and only when
    the continuation doesn't look like a venue, pages or a year."""
    body = _dehyphenate(" ".join(raw.split()))
    words = re.findall(r"\w+", title)
    if len(words) < 2:
        return title
    whole = re.search(r"[\W_]+".join(re.escape(w) for w in words), body, re.I)
    if not whole:
        return title
    rest = body[whole.end():]
    if not rest.startswith(", "):
        return title
    stop = re.search(r"[.?!](?:\s|$)", rest)
    extra = rest[: stop.start()] if stop else ""
    if (not extra or len(extra.split()) > 15 or re.search(r"\d", extra)
            or re.search(r"\b(?:In|Proceedings|Proc|Journal|Conference|vol|pp|pages|"
                         r"Springer|ACM|IEEE|arXiv|Press|University)\b", extra)):
        return title
    return (title.rstrip(".") + extra).strip()


def _org_author_from_raw(raw: str, title: str) -> str:
    """"International Organization for Standardization (ISO)" — a corporate
    author GROBID left empty.  Only short segments naming an organisation."""
    body = re.sub(r"^\s*(?:\[\d+\]|\(\d+\)|\d+\.)\s*", "", " ".join(raw.split()))
    i = _title_start(body, title)
    if i <= 0:
        return ""
    seg = re.sub(r"[.,]?\s*\(?(?:19|20)\d{2}[a-z]?\)?[.,:]?$", "", body[:i].strip()).rstrip(".,:; ")
    if 3 <= len(seg) <= 160 and (_ORG_WORD_RE.search(seg) or re.search(r"\([A-Z]{2,}\)", seg)):
        return seg
    return ""


def _repair_from_raw(ref: dict[str, Any]) -> dict[str, Any]:
    raw, title = ref.get("raw") or "", (ref.get("title") or "").strip()
    if not raw or not title:
        return ref
    out = ref
    # Publisher / organisation appended to the title:
    # "Data Citation of Evolving Data. Research Data Alliance"
    head, sep, tail = title.rpartition(". ")
    if sep and len(head.split()) >= 3 and len(tail.split()) <= 5 and (
            _ORG_WORD_RE.search(tail) or _looks_like_venue(tail)):
        title = head
        out = {**out, "title": title}
    names = _names_from_raw(raw, title)
    if names:
        grobid = [a for a in (ref.get("authors") or "").split(", ") if a]
        # A scramble re-assigns the same name tokens; a different token set
        # means raw holds a neighbouring citation — leave GROBID's names.
        g_tokens = set(_norm_loose(" ".join(grobid)).split())
        r_tokens = set(_norm_loose(" ".join(names)).split())
        same_people = not g_tokens or len(g_tokens & r_tokens) >= 0.6 * len(g_tokens)
        if same_people and _surname_set(names) != _surname_set(grobid):
            out = {**out, "authors": ", ".join(names), "_authors_from_raw": True}
    elif not (ref.get("authors") or "").strip():
        org = _org_author_from_raw(raw, title)
        if org:
            out = {**out, "authors": org}
    longer = _extend_title(raw, title)
    if longer != title:
        out = {**out, "title": longer}
    return out


def _line_break_hyphen_fixer(doc_text: str):
    """Restore real hyphens that a line break made ambiguous.

    "few-⏎shot" in a PDF is either a hyphenated word ("evalu-⏎ation") or a
    compound broken at its hyphen ("few-shot").  Parsers (GROBID too) join
    both, and "Language models are fewshot learners" then matches nothing
    in the indexes.  The document decides: if the joined form never occurs
    elsewhere in it but the second part does as a word, the hyphen was real."""
    breaks = {(m.group(1), m.group(2))
              for m in re.finditer(r"\b([A-Za-z]{2,})-[ \t]*\n\s*([a-z]{2,})\b", doc_text)}
    if not breaks:
        return lambda s: s
    clean = re.sub(r"[A-Za-z]+-[ \t]*\n\s*[A-Za-z]+", " ", doc_text)
    from collections import Counter
    tokens = Counter(re.findall(r"[a-z]+", clean.lower()))
    compounds: dict[str, int] = {}
    for a, b in breaks:
        joined = (a + b).lower()
        # The second part is the telling one: hyphenation leaves fragments
        # ("-ation", "-hoods", "-cessing") that never stand alone, while a
        # compound's second word ("shot", "training", "based") does.
        # Lowercase only (names such as "Bren-⏎nan" stay joined), and a
        # second part of 4+ letters ("lever-⏎age" is hyphenation).
        if a[0].islower() and not tokens[joined] and len(b) >= 4 and tokens[b.lower()]:
            compounds[joined] = len(a)
    if not compounds:
        return lambda s: s
    rx = re.compile(r"\b(" + "|".join(map(re.escape, sorted(compounds, key=len, reverse=True)))
                    + r")\b", re.I)

    def fix(s: str) -> str:
        return rx.sub(lambda m: m.group(0)[:compounds[m.group(0).lower()]] + "-"
                      + m.group(0)[compounds[m.group(0).lower()]:], s)
    return fix


def _better_title(primary: str, secondary: str) -> str:
    """The citation-list title when it fixes a typical full-document error:
    the venue glued on ("Title. Advances in Neural …") or a leading
    "name:" lost ("A Toolkit …" for "pybool_ir: A Toolkit …").  "" = keep."""
    p, q = primary.strip().rstrip("."), secondary.strip().rstrip(".")
    if not q or p == q:
        return ""
    if not p or (_looks_like_venue(p) and not _looks_like_venue(q)):
        return q
    if p.startswith(q + ". ") and len(q.split()) >= 3:
        return q
    if _is_name_list(p) and not _is_name_list(q) and len(q.split()) >= 2:
        return q
    if q.endswith(": " + p) or (q.endswith(p) and re.match(r"^[^\s]+:\s", q)):
        return q
    return ""


def extract_references_hybrid(
    pdf_path: str, return_markdown: bool = False
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    """
    Merge two independent extractions, taking the **union of refs**:

      A) GROBID's full-document layout parse — high-precision structure but
         occasionally drops refs (long author lists, awkward page breaks).
      B) pymupdf4llm-located References section → GROBID /processCitationList
         — recovers refs A missed because B sees every line of the section.

    The two lists are merged with `_merge_ref_lists`: every ref from A is
    kept, refs from B are appended only when fuzzy title (or raw-text)
    matching shows they're not already present in A.

    This is a strict superset of GROBID-only: paper #60 (where the section
    pipeline alone collapsed to 1 ref) is unaffected because A=52 is
    preserved; paper #178 (where A misses 5 refs that B catches) gets
    those extras back.
    """
    # Run GROBID full-doc parse and the pymupdf-section pipeline in
    # parallel — they share no state and hit different HTTP endpoints.
    from concurrent.futures import ThreadPoolExecutor
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_grobid  = pool.submit(extract_references_grobid, pdf_path)
        f_section = pool.submit(_extract_via_section_pipeline, pdf_path)
        try:
            grobid_refs = f_grobid.result()
        except Exception as e:
            errors.append(str(e))
            grobid_refs = []
        try:
            section_refs = f_section.result()
        except Exception as e:
            errors.append(str(e))
            section_refs = []
    merged = _merge_ref_lists(grobid_refs, section_refs)
    merged = [_repair_from_raw(r) for r in merged if _is_plausible_reference(r)]
    try:
        fix = _line_break_hyphen_fixer(_plain_text(pdf_path))
        # Titles only: they drive the lookups.  Raw text stays as printed
        # (highlighting and the author repair compare against the PDF).
        merged = [{**r, "title": fix(r["title"])} if r.get("title") else r for r in merged]
    except Exception:
        pass
    if errors and not merged:
        # Both halves need GROBID; silently returning [] used to show
        # "no references found" when the real problem was GROBID.
        raise ParserError(errors[0])

    if return_markdown:
        # Debug view of what the section splitter saw.  Plain page text,
        # not pymupdf4llm: to_markdown can trigger minutes of OCR on
        # appendix pages and this ran on every hybrid parse.
        try:
            md = _locate_reference_section(_plain_text(pdf_path)) or ""
        except Exception:
            md = ""
        return merged, md
    return merged
