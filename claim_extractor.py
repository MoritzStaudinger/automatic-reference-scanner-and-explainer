"""
Extract per-citation claims from a paper's body text.

Given the citing paper's markdown body (everything before the reference
section) and its parsed bibliography, this module walks the body,
identifies inline citation markers, and emits one :class:`Claim` per
(sentence, citation) pair — including cases where a single sentence
carries multiple citations, which each get their own claim.

Two citation-marker conventions are recognised:

  Numeric  ::  [3]  [3, 5]  [3-7]  [3,5,7]
  Author-year ::  (Smith, 2020)  (Smith et al., 2020)
                 (Smith and Jones, 2020)  (Smith 2020; Jones 2021)
                  Smith et al. (2020)

Mapping a marker back to a bibliography entry:
  - Numeric  →  1-based index into the parsed reference list.
  - Author-year → fuzzy match of (first author's surname, 4-digit year)
    against every ref's parsed `authors` + `year`.

Sentence boundary detection: naive split on ". " with heuristics to
avoid splitting on abbreviations ("e.g.", "et al.", initials).  Good
enough for citation-context extraction; not intended as a general NLP
sentence splitter.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# Numeric citation markers: [3], [3, 5], [3-7], [3,5,7], [3, 5-7]
_NUMERIC_MARKER_RE = re.compile(r"\[\s*(\d[\d\s,\-–]*)\s*\]")

# Author-year INSIDE parens: (Smith 2020), (Smith et al., 2020),
# (Smith and Jones 2020), (Smith 2020; Jones 2021).
# We capture the whole inside-parens payload then split on ; for
# multi-cite groups.
_PAREN_MARKER_RE = re.compile(
    r"\(([A-Z][A-Za-z\-']+"                       # first author surname
    r"(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-z\-']+)?"  # optional "et al." or "and Jones"
    r"(?:,?\s+\d{4}[a-z]?"                        # comma+year OR space+year
    r"(?:\s*;\s*[A-Z][A-Za-z\-']+"                # additional cites separated by ;
    r"(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-z\-']+)?"
    r",?\s+\d{4}[a-z]?)*)"
    r")\)"
)

# Narrative form: "Smith et al. (2020)" or "Smith (2020)"
_NARRATIVE_MARKER_RE = re.compile(
    r"([A-Z][A-Za-z\-']+"
    r"(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-z\-']+)?)"
    r"\s*\((\d{4}[a-z]?)\)"
)

# Split multi-cites inside parens ("Smith, 2020; Jones, 2021" → 2 cites).
_MULTICITE_SPLIT_RE = re.compile(r"\s*;\s*")

# One (surname, year) pair inside an in-paren citation.
_AUTHOR_YEAR_ONE_RE = re.compile(
    r"([A-Z][A-Za-z\-']+)"                        # surname
    r"(?:\s+(et\s+al\.?)|\s+and\s+([A-Z][A-Za-z\-']+))?"   # "et al." / "and Jones"
    r",?\s+(\d{4}[a-z]?)"                        # year, with its a/b/c letter
)

# Sentence-boundary heuristics: split on ". " but not immediately after
# common abbreviations that carry an internal period.
_SENT_END_RE = re.compile(r"(?<![A-Z])(?<!\bal)(?<!\be\.g)(?<!\bi\.e)(?<!\bcf)(?<!\bvs)\.(?=\s+[A-Z\d\[\(])")

# Markdown-specific noise we don't want in a claim text: headings, table
# separators, image refs.
_MD_HEADING_RE = re.compile(r"^\s*#+\s+", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """One (sentence, citation) pair worth verifying."""
    text:             str                # the sentence containing the citation
    citation_marker:  str                # the raw marker as it appears in text
    cited_ref_index:  int                # index into the parsed refs list
    style:            str                # numeric | paren | narrative

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Reference lookup helpers
# ---------------------------------------------------------------------------

def _first_surname(authors: str) -> str:
    """
    Return the surname of the first author, lowercased.  Handles common
    forms: "Smith, J.", "John Smith", "Smith et al.", "Smith and Jones".
    """
    if not authors:
        return ""
    # Strip leading markdown bullets, "and " prefixes, whitespace.
    s = authors.lstrip("- \t").strip()
    # "Smith, J." — surname is before the comma
    if "," in s:
        first = s.split(",", 1)[0]
    else:
        first = s.split(" and ", 1)[0].split(";", 1)[0]
    # "John Smith" — surname is the last token
    toks = [t for t in first.split() if t]
    if not toks:
        return ""
    last = toks[-1]
    # Drop trailing punctuation
    return re.sub(r"[^\w\-']", "", last).lower()


def _year_of(ref: dict[str, Any]) -> str:
    """Year with its disambiguation letter when the reference has one
    ("2024b"): "Zhao and Lian, 2024a" and "2024b" are different papers."""
    raw = ref.get("raw", "") or ""
    y = str(ref.get("year") or "").strip()
    m = re.search(r"\b((?:19|20)\d{2})([a-z])\b", raw)      # "… Lian. 2024b. …"
    if re.fullmatch(r"\d{4}[a-z]?", y):
        if len(y) == 4 and m and m.group(1) == y:
            return y + m.group(2)
        return y
    if m:
        return m.group(0)
    m = re.search(r"\b(19|20)\d{2}\b", raw)
    return m.group(0) if m else ""


def _build_author_year_index(refs: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    """Map (first_author_surname_lower, year) → ref indices, in order.
    A reference with a suffixed year ("2024b") is indexed under that and,
    as a fallback for citations written without the letter, under "2024"."""
    idx: dict[tuple[str, str], list[int]] = {}
    for i, ref in enumerate(refs):
        surname = _first_surname(ref.get("authors", ""))
        year    = _year_of(ref)
        if not surname or not year:
            continue
        idx.setdefault((surname, year), []).append(i)
        if year[:4] != year:
            idx.setdefault((surname, year[:4]), []).append(i)
    return idx


def _n_authors(authors: str) -> int:
    return len([a for a in re.split(r",\s*(?:and\s+)?|\s+and\s+", authors or "") if a.strip()])


def _lookup_ay(idx: dict[tuple[str, str], list[int]], surname: str, year: str,
               refs: list[dict[str, Any]] | None = None,
               second: str | None = None, et_al: bool = False) -> int | None:
    """The reference a "(Surname[ and Second | et al.], Year)" citation
    means.  Several Zhangs can share a year: the co-author ("Zhang and
    Lian") or "et al." (3+ authors) picks the right one."""
    cands = idx.get((surname.lower(), year))
    if not cands:
        if len(year) == 5:              # "2024c" with no matching suffixed entry
            return None
        cands = idx.get((surname.lower(), year[:4]))
    if not cands:
        return None
    if len(cands) == 1 or refs is None:
        return cands[0]
    def fits(i: int) -> bool:
        a = refs[i].get("authors") or ""
        if second:
            return second.lower() in a.lower() and _n_authors(a) == 2
        if et_al:
            return _n_authors(a) >= 3
        return _n_authors(a) == 1
    return next((i for i in cands if fits(i)), cands[0])


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _sentences_around(text: str, span_start: int, span_end: int) -> str:
    """
    Return the sentence that contains the character range [start:end).
    We look outward for sentence-ending punctuation.
    """
    # Look backwards for a sentence start.
    left = text.rfind(". ", 0, span_start)
    if left == -1:
        left = 0
    else:
        left += 2   # step past ". "
    # Also stop at paragraph breaks / headings.
    para_left = max(
        text.rfind("\n\n", 0, span_start),
        text.rfind("\n#", 0, span_start),
    )
    if para_left > left:
        left = para_left + 1
    # Forward: first sentence terminator after the span.
    right_match = _SENT_END_RE.search(text, span_end)
    right_para = text.find("\n\n", span_end)
    candidates = []
    if right_match is not None:
        candidates.append(right_match.end())
    if right_para != -1:
        candidates.append(right_para)
    right = min(candidates) if candidates else len(text)
    sentence = text[left:right].strip()
    # Collapse whitespace + strip markdown emphasis noise.
    sentence = re.sub(r"\s+", " ", sentence)
    # pymupdf4llm body text is markdown: drop emphasis markers so they
    # don't reach the LLM prompt or the UI.
    sentence = re.sub(r"(\*\*|__|(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w))", "", sentence)
    return sentence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(body_text: str, refs: list[dict[str, Any]]) -> list[Claim]:
    """
    Extract claims from the citing paper's body text.

    Parameters
    ----------
    body_text : str
        Everything BEFORE the References/Bibliography section.  Callers
        should slice this off with the same heuristic parsers.py uses
        (``_locate_reference_section``).
    refs : list[dict]
        The parsed bibliography, indexed in the order they appear.
    """
    if not body_text or not refs:
        return []

    # Strip markdown headings so they don't leak into the sentence text.
    body = _MD_HEADING_RE.sub("", body_text)

    ay_index = _build_author_year_index(refs)
    num_index = build_numeric_index(refs)
    claims:  list[Claim] = []

    # 1. Author-year in parens (may contain multiple cites via `;`)
    for m in _PAREN_MARKER_RE.finditer(body):
        payload = m.group(1)
        sentence = _sentences_around(body, m.start(), m.end())
        if not sentence:
            continue
        for surname, year, second, et_al in _parse_paren_payload(payload):
            idx = _lookup_ay(ay_index, surname, year, refs, second, et_al)
            if idx is None:
                continue
            claims.append(Claim(
                text=sentence,
                citation_marker=m.group(0),
                cited_ref_index=idx,
                style="paren",
            ))

    # 2. Narrative form "Smith et al. (2020)"
    for m in _NARRATIVE_MARKER_RE.finditer(body):
        # Grab just the first surname from the group (before "et al." / "and")
        surname_part = m.group(1)
        surname = surname_part.split()[0].strip(",.")
        sm = re.search(r"\band\s+([A-Z][A-Za-z\-']+)", surname_part)
        idx = _lookup_ay(ay_index, surname, m.group(2), refs,
                         sm.group(1) if sm else None, "et al" in surname_part)
        if idx is None:
            continue
        sentence = _sentences_around(body, m.start(), m.end())
        if not sentence:
            continue
        claims.append(Claim(
            text=sentence,
            citation_marker=m.group(0),
            cited_ref_index=idx,
            style="narrative",
        ))

    # 3. Numeric markers — unless the paper cites author-year and its
    #    bibliography carries no "[n]" labels.  Then a "[4]" is an
    #    affiliation or footnote mark, and mapping it to the 4th entry by
    #    position attached title-page author lists to random references.
    if _numeric_labels_dominant(refs) or len(claims) < 3:
        for m in _NUMERIC_MARKER_RE.finditer(body):
            payload = m.group(1)
            numbers = _parse_numeric_payload(payload, 10_000)
            indices = list(dict.fromkeys(num_index[n + 1] for n in numbers if n + 1 in num_index))
            if not indices:
                continue
            sentence = _sentences_around(body, m.start(), m.end())
            if not sentence:
                continue
            for idx in indices:
                claims.append(Claim(
                    text=sentence,
                    citation_marker=m.group(0),
                    cited_ref_index=idx,
                    style="numeric",
                ))

    return _dedupe(claims)


_REF_LABEL_RE = re.compile(r"^\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)]\s)")


def _numeric_labels_dominant(refs: list[dict[str, Any]]) -> bool:
    """True when most parsed references carry a "[12]" / "12." label."""
    cands = [r for r in refs if not r.get("_user_added") and not r.get("_auto_added")]
    if not cands:
        return False
    labelled = sum(1 for r in cands if _REF_LABEL_RE.match(r.get("raw") or ""))
    return labelled >= 0.6 * len(cands)


def build_numeric_index(refs: list[dict[str, Any]]) -> dict[int, int]:
    """
    Map a numeric citation label (the 12 in "[12]") to a position in
    ``refs``.  Using ``n - 1`` directly broke whenever the parser skipped
    an entry, dedupe merged two, or manual / auto-detected refs were
    appended.

    1. If most refs carry their own label in the raw text ("[12] …" or
       "12. …"), trust those labels.
    2. Otherwise use the parser's original position (``_orig_index``,
       recorded by ``dedupe_references``), including merged duplicates.
    3. Refs the user or the orphan detector added are never targets.
    """
    candidates = [(p, r) for p, r in enumerate(refs)
                  if not r.get("_user_added") and not r.get("_auto_added")]
    labelled: dict[int, int] = {}
    for p, r in candidates:
        m = _REF_LABEL_RE.match(r.get("raw") or "")
        if m:
            labelled.setdefault(int(m.group(1) or m.group(2)), p)
    if candidates and len(labelled) >= 0.6 * len(candidates):
        return labelled

    by_pos: dict[int, int] = {}
    for p, r in candidates:
        by_pos.setdefault(r.get("_orig_index", p) + 1, p)
        for oi in r.get("_merged_orig_indices", []):
            by_pos.setdefault(oi + 1, p)
    return by_pos


def _parse_numeric_payload(payload: str, n_refs: int) -> list[int]:
    """
    Parse the inside of a numeric marker: "3", "3, 5", "3-7", "3,5-8".
    Returns 0-based indices into the refs list, clamped to valid range.
    Hyphen accepted as either ASCII '-' or en-dash '–'.
    """
    parts = re.split(r"[,\s]+", payload)
    out: list[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.fullmatch(r"\d+[\-–]\d+", p):
            a, b = re.split(r"[\-–]", p)
            try:
                start, end = int(a), int(b)
            except ValueError:
                continue
            if end < start or (end - start) > 30:
                # sanity: nobody cites a 30-ref span at once — likely
                # noise ("Table 3-45" mis-detected)
                continue
            out.extend(range(start - 1, end))
        else:
            try:
                out.append(int(p) - 1)
            except ValueError:
                continue
    # Filter to valid indices, preserve order + dedupe.
    seen: set[int] = set()
    valid: list[int] = []
    for i in out:
        if 0 <= i < n_refs and i not in seen:
            seen.add(i)
            valid.append(i)
    return valid


def _parse_paren_payload(payload: str) -> Iterable[tuple[str, str, str | None, bool]]:
    """
    Yield (surname, year, second_surname, et_al) from an in-paren citation payload.
    Handles ; separator for grouped cites.
    """
    for chunk in _MULTICITE_SPLIT_RE.split(payload):
        m = _AUTHOR_YEAR_ONE_RE.match(chunk.strip())
        if not m:
            continue
        yield m.group(1), m.group(4), m.group(3), bool(m.group(2))


def _dedupe(claims: list[Claim]) -> list[Claim]:
    """
    Same (sentence, ref_index) pair may be picked up twice (numeric +
    narrative, paren + narrative).  Collapse duplicates, preserving
    first-seen order.
    """
    seen: set[tuple[str, int]] = set()
    out: list[Claim] = []
    for c in claims:
        key = (c.text, c.cited_ref_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
