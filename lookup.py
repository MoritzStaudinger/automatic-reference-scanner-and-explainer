"""
Reference lookup against Semantic Scholar, DBLP, OpenAlex, arXiv, ACL Anthology, and Crossref.

Pipeline
--------
1. **S2 batch pre-warm** — for every ref that carries a DOI or arXiv ID,
   collapse those into a single ``POST /paper/batch`` request to S2 and
   pre-populate the per-source cache.  Subsequent per-ref lookups become
   cache hits, eliminating dozens of sequential round trips.
2. **Router** — per-ref classification picks the most likely source(s):
   arXiv ID → arxiv only, DOI → Crossref only, CS-conf venue → DBLP only,
   book signal → OpenLibrary only.  Identifier-based routing is treated
   as authoritative — if the routed source returns ``not_found`` we do
   NOT escalate (a fabricated DOI/arXiv ID stays fabricated).
3. **Escalation** — refs that the router couldn't classify, or that the
   routed (non-identifier) source missed, fan out to the remaining APIs.
4. **ACL backfill** — derive aclanthology.org URLs from any source that
   returned an ACL DOI / S2 external ACL ID.
5. **OpenLibrary** — runs only when ``_VENUE_BOOK_RE`` matches the raw or
   venue text.  The previous "fall through if all academic sources
   missed" rule was burning ~100 calls/paper for zero hits.

All HTTP traffic goes through a module-level ``requests.Session()`` with
a 20-connection keep-alive pool, eliminating per-request TCP+TLS setup.

Validation sources (title + venue → card colour):
    semantic_scholar, acl_anthology, dblp, openalex, crossref

Identifier source (link only):
    arxiv
"""

from __future__ import annotations

import arse_env  # noqa: F401  (accepts the old ARES_* settings)

import logging
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rapidfuzz import fuzz as _fuzz

import requests as _requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv

from cache import cache_key as _cache_key, cached as _cached, get_cache as _get_cache

load_dotenv()

log = logging.getLogger(__name__)

# One contact address for every polite-pool API (Crossref, OpenAlex,
# doi.org, Open Library) and for URL fetches.  Override with
# ARSE_CONTACT_EMAIL; the per-source OPENALEX_MAILTO / DOI_ORG_MAILTO
# variables still win when set.
CONTACT_EMAIL = os.getenv("ARSE_CONTACT_EMAIL", "moritz.staudinger@tuwien.ac.at")
USER_AGENT    = f"ARSE/1.0 (+https://reference-cleaner.ds-ifs.tuwien.ac.at; mailto:{CONTACT_EMAIL})"

# Connection-pooled HTTP session — every source uses this so TCP+TLS
# handshakes amortise across the ~200 calls per paper.
_SESSION = _requests.Session()
# Pool sized for the actual fanout: 8 refs in parallel × up to 5 sources
# each = 40 simultaneous connections.  With pool_maxsize<40, late
# requests block waiting for a free slot, which serialises what should
# be parallel work.  pool_connections=40 keeps that many DNS-resolved
# host slots warm; pool_maxsize=80 lets short bursts overshoot without
# spawning fresh TCP+TLS handshakes.
_adapter = HTTPAdapter(pool_connections=40, pool_maxsize=80, max_retries=0)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)

# Lightweight HTTP-call counter for cost/throughput analysis.  Increments
# every time `_http_get_retry` or any `_SESSION.{get,post}` is invoked.
# Categorised by host so the benchmark can show "X S2 calls, Y Crossref
# calls, ..." per paper.  Process-wide: totals mix concurrent users.
HTTP_CALL_COUNTS: dict[str, int] = {}
_HTTP_COUNT_LOCK = threading.Lock()


def _bump_http_count(url: str) -> None:
    try:
        host = url.split("//", 1)[1].split("/", 1)[0]
    except Exception:
        host = "unknown"
    with _HTTP_COUNT_LOCK:
        HTTP_CALL_COUNTS[host] = HTTP_CALL_COUNTS.get(host, 0) + 1


def reset_http_counts() -> dict[str, int]:
    """Snapshot + zero the HTTP call counter.  Returns the previous totals
    so the caller can attribute calls to a unit of work (e.g. a paper)."""
    with _HTTP_COUNT_LOCK:
        prev = dict(HTTP_CALL_COUNTS)
        HTTP_CALL_COUNTS.clear()
    return prev

# Transient HTTP statuses worth retrying — public APIs hit these under load.
_RETRY_STATUSES = {429, 500, 502, 503, 504}


_MAX_RETRY_AFTER = 5.0   # never sleep longer than this between retries


def _http_get_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 10,
    attempts: int = 3,
) -> _requests.Response:
    """
    GET with bounded exponential backoff on 429/5xx + transient network errors.

    Sleep schedule (seconds): 0.5, 1.5 between attempts (max 2 sleeps).
    `Retry-After` is honoured but CAPPED at _MAX_RETRY_AFTER seconds — some
    APIs send 30+ s values that would freeze the pipeline.  When a server is
    that overloaded, giving up and reporting `error` is better than hanging.
    """
    last_exc: Exception | None = None
    _bump_http_count(url)
    for i in range(attempts):
        try:
            r = _SESSION.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in _RETRY_STATUSES and i < attempts - 1:
                ra = r.headers.get("Retry-After")
                retry_after = 0.0
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        retry_after = 0.0
                sleep_for = min(_MAX_RETRY_AFTER, max(retry_after, 0.5 + i))
                time.sleep(sleep_for)
                continue
            return r
        except (_requests.ConnectionError, _requests.Timeout) as exc:
            last_exc = exc
            if i == attempts - 1:
                raise
            time.sleep(0.5 + i)
    assert last_exc is not None
    raise last_exc

S2_API_KEY: str | None = os.getenv("S2_API_KEY") or None

_S2_BASE   = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "title,authors,year,externalIds,venue,openAccessPdf,abstract"

_CR_BASE    = "https://api.crossref.org/works"
_CR_HEADERS = {"User-Agent": USER_AGENT}

_DBLP_BASE  = "https://dblp.org/search/publ/api"
_OA_BASE    = "https://api.openalex.org/works"
_CORE_BASE  = "https://api.core.ac.uk/v3"
_CORE_API_KEY = os.getenv("CORE_API_KEY", "")
# CORE's free tier: 10 req/min, 1000/day with an API key.  Caller-side
# pacing is the responsibility of whoever calls _lookup_core — this is
# intentionally NOT wired into the default lookup_all() escalation,
# so it only fires when something explicitly calls it.
# OpenAlex grants ~10× higher rate limits ("polite pool") to identifiable
# users via the mailto query parameter.  Set OPENALEX_MAILTO in .env to
# your real address; we fall back to a generic one only if not set.
_OA_MAILTO  = os.getenv("OPENALEX_MAILTO") or CONTACT_EMAIL
_OA_HEADERS = {"User-Agent": USER_AGENT}

# Title similarity thresholds
MATCH_THRESHOLD          = 0.90
FUZZY_THRESHOLD          = 0.70
AUTHOR_MATCH_THRESHOLD   = 0.35   # last-name Jaccard to confirm identity
VENUE_MISMATCH_THRESHOLD = 0.40

# wrong_authors / wrong_venue gates.
#
# `wrong_authors` is meant to catch the precise failure mode:
#   "a real paper with essentially THIS title exists, the cited authors
#    are not its authors — likely a fabricated / mis-attributed citation."
#
# That requires the titles to genuinely match, not just share a few
# words.  Originally 0.50 — too low: a 0.55-0.65 tsim is usually the
# source returning the nearest-titled paper in its index, NOT actually
# the cited paper.  Flagging wrong_authors in that case produces false
# positives where the right paper just isn't indexed by that source
# (Anthropic publications on transformer-circuits.pub, NeurIPS workshop
# papers, project pages, blog posts, ...).
#
# 0.85 = "titles are essentially identical, modulo punctuation and word
# order".  Below that, the result falls through to the existing fuzzy /
# missing classification, which is the honest answer.
#
# wrong_doi / wrong_arxiv_id still use the 0.50 floor independently:
# those gates are inverted (tsim < 0.50 → flag) because the identifier
# itself proves the source resolved a specific paper, so a wildly
# different title means the IDENTIFIER is wrong.
#
# Raised from 0.85 to 0.93: genuine same-paper pairs score ~1.0, while a
# *different* paper with a similar title lands at 0.82-0.86 ("Paraphrasing
# user stories with LLMs" vs "Paraphrasing with LLMs" = 0.86).  At 0.85 such
# near-titles were declared "right paper, wrong authors" → false mismatch.
WRONG_FLAG_TITLE_FLOOR = 0.93
WRONG_VENUE_THRESHOLD  = 0.30
# Inverted gate for identifier routes: below this the cited DOI / arXiv ID
# resolved to a clearly different paper.
WRONG_ID_TITLE_FLOOR   = 0.50

_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})"
    r"|arXiv:(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)
# DOI embedded in a doi.org / dx.doi.org URL
_DOI_FROM_URL_RE = re.compile(
    r"(?:(?:dx\.)?doi\.org/|doi:)\s*(10\.\d{4,}/[^\s,>\"'<]+)",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"10\.\d{4,}/\S+")

# ACL Anthology DOI prefix → used to derive aclanthology.org links without S2
_ACL_DOI_RE = re.compile(r"10\.18653/v1/(.+)", re.IGNORECASE)


def _normalize_acl_id(acl_id: str) -> str:
    """
    Canonicalise an ACL anthology paper id.

    ACL has two id formats:
      old style — ``P18-1234`` (leading letter is venue, always UPPERCASE
                  on aclanthology.org).  Crossref/OpenAlex frequently
                  store the DOI suffix lowercased (``p18-1234``), which
                  is a 301 redirect on the website but breaks
                  downstream pattern matching that expects the canonical
                  uppercase form.
      new style — ``2024.acl-long.123`` (lowercase, leading digit).  Must
                  NOT be uppercased — the venue-tag and rank-tag are
                  case-sensitive on aclanthology.org.

    Distinguish by the first character: a leading letter means old
    style → uppercase it.  A leading digit means new style → leave alone.
    """
    if acl_id and acl_id[0].isalpha():
        return acl_id[0].upper() + acl_id[1:]
    return acl_id

# --- Source-routing patterns ---
# DBLP indexes most CS conferences cleanly — when we see one of these venue
# names, query DBLP only instead of fanning out to all five sources.
_VENUE_DBLP_RE = re.compile(
    r"\b(?:NeurIPS|NIPS|ICML|ICLR|AAAI|IJCAI|CVPR|ICCV|ECCV|"
    r"KDD|SIGIR|UAI|AISTATS|COLT|FOCS|STOC|SODA|"
    r"VLDB|SIGMOD|OSDI|SOSP|ASPLOS|ISCA|MICRO|"
    r"USENIX\s+Security|CCS|NDSS|"
    r"IROS|ICRA|RSS|"
    r"InterSpeech|ICASSP|WWW|WSDM|RecSys|CIKM|ICDM|ECML|"
    r"CHI|UIST|CSCW)\b",
    re.IGNORECASE,
)
# Strong book/monograph signals — use Open Library only.
_VENUE_BOOK_RE = re.compile(
    r"\bISBN[- ]?(?:13|10)?:?\s*[\d\-Xx]{10,17}|"
    r"\b(?:O'Reilly|MIT Press|Cambridge University Press|"
    r"Springer-Verlag|Wiley|McGraw-Hill|Pearson|Manning Publications|"
    r"Oxford University Press|Princeton University Press)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Normalisation & similarity
# ---------------------------------------------------------------------------

_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+")


def _ascii_fold(text: str) -> str:
    """Strip diacritics: Müller → Muller, naïve → naive."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _normalize(text: Any) -> str:
    # Defensive: an upstream parser / API sometimes hands us a list or
    # dict instead of a string.  Coerce to a single string before any
    # str-only operations so callers don't crash on "list has no .lower".
    if not isinstance(text, str):
        if isinstance(text, list):
            text = " ".join(t for t in text if isinstance(t, str))
        elif isinstance(text, dict):
            text = str(text.get("name") or text.get("title") or "")
        else:
            text = str(text or "")
    text = _ascii_fold(text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _LEADING_ARTICLE_RE.sub("", text)
    return text


def title_similarity(t1: str, t2: str) -> float:
    """
    Robust title similarity.

    Combines three scorers and adds a subtitle-aware pass so common parser
    failure modes ("Foo" vs "Foo: A Subtitle", "The Foo" vs "Foo") don't
    score below threshold for the same paper.
    """
    if not t1 or not t2:
        return 0.0
    n1, n2 = _normalize(t1), _normalize(t2)

    # Use token_sort_ratio only.  partial_ratio and WRatio look generous
    # in isolation but they over-match plausible-sounding fabrications by
    # treating shared substrings like "large language models" or "survey
    # of X" as near-identity (verified: "A survey of causal discovery and
    # causal inference" vs "A Survey on Causal Inference" scores 0.86 on
    # partial_ratio and 0.85 on WRatio — both above our fuzzy floor —
    # whereas token_sort_ratio correctly says 0.68).
    score = _fuzz.token_sort_ratio(n1, n2) / 100.0

    # Subtitle-aware pass: if either title has a colon, also score the
    # pre-colon prefix.  Catches "Foo" vs "Foo: A Detailed Survey" without
    # the false-positive risk of plain partial_ratio.
    p1 = t1.split(":", 1)[0]
    p2 = t2.split(":", 1)[0]
    if (p1 != t1 or p2 != t2) and len(p1.strip()) >= 4 and len(p2.strip()) >= 4:
        np1, np2 = _normalize(p1), _normalize(p2)
        score = max(score, _fuzz.token_sort_ratio(np1, np2) / 100.0)

    # Existing prefix-substring boost (kept for full-prefix matches like
    # "Foo" inside "Foo extended").
    shorter, longer = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
    if longer.startswith(shorter) and len(shorter) / len(longer) >= 0.6:
        score = max(score, 1.0)

    # Containment boost: when the shorter normalized title appears as a
    # substring of the longer, treat it as a match.  Common when the
    # parser left author names / year glued onto the title field
    # ("Aaron Grattafiori and 1 others. 2024. The llama 3 herd of models"
    # vs "The Llama 3 Herd of Models" — substring-contained at 100%, but
    # token_sort_ratio sees the extra prefix words and scores ~50%).
    # Guarded by a 15-char minimum on the shorter side so single-word
    # titles can't false-match (e.g. "models").
    if len(shorter) >= 15 and shorter in longer:
        score = max(score, 1.0)
    return score


def venue_similarity(v1: str, v2: str) -> float:
    """Fuzzy venue match. Handles 'EMNLP' vs 'Proceedings of EMNLP 2020' etc."""
    if not v1 or not v2:
        return 0.0
    n1, n2 = _normalize(v1), _normalize(v2)
    return max(
        _fuzz.token_sort_ratio(n1, n2) / 100.0,
        _fuzz.partial_ratio(n1, n2)    / 100.0,
    )


# Words that aren't real surnames but show up at the tail of a free-form
# author string — placeholder for "we didn't list everyone".  We drop these
# so they don't poison the Jaccard / coverage author-similarity score.
_AUTHOR_PLACEHOLDER_TOKENS = {
    "", "al", "et", "others", "other", "etal", "and",
    "colleagues", "coauthors", "co", "auth",
}


def _extract_lastnames(authors: str) -> set[str]:
    """
    Extract normalised, ASCII-folded last names from a free-form author string.

    Drops obvious placeholder tokens ("et al.", "and 1 others", "and others")
    so the resulting set contains only real surnames.

    Handles:
      "J. Smith"              → {smith}
      "Yan Wang"              → {wang}      (was buggy: previously returned {yan})
      "Smith, J."             → {smith}     (the " J." sub-part is dropped as noise)
      "Smith and Jones"       → {smith, jones}
      "Müller"                → {muller}    (ASCII-folded via _normalize)
      "Foo and 1 others"      → {foo}       (placeholder dropped)
      "Foo, Bar et al."       → {foo, bar}  (placeholder dropped)
    """
    # Remove "et al." in any form before splitting
    authors = re.sub(r"\bet\.?\s*al\.?", " ", authors, flags=re.IGNORECASE)
    # Drop standalone digits ("1 others" → "others") so the digit doesn't
    # become its own "lastname"
    authors = re.sub(r"\b\d+\b", " ", authors)

    parts = re.split(r"[,;]|\band\b", authors, flags=re.IGNORECASE)
    lastnames: set[str] = set()
    for part in parts:
        tokens = part.strip().split()
        if not tokens:
            continue
        candidate = _normalize(tokens[-1] if len(tokens) >= 2 else tokens[0])
        if candidate and candidate not in _AUTHOR_PLACEHOLDER_TOKENS:
            lastnames.add(candidate)
    return lastnames - {""}


def _grounded_cited_authors(extracted_authors: str, raw: str) -> str:
    """
    Drop "cited" author names whose surname doesn't appear in the raw
    citation text — these are typically LLM-repair hallucinations.

    Real example: raw says only "C Raffel. Exploring the limits of
    transfer learning..." but the LLM repair pass filled `authors` with
    8 names — most of which are actually the authors of "Attention is
    All You Need", not T5.  Comparing those 8 against the actual paper's
    9 authors gives a 13% Jaccard score (only Raffel/Shazeer in common)
    and the ref shows as red, even though the citation is just
    underspecified, not wrong.

    By dropping hallucinated names we let the coverage path
    (intersection-over-smaller-set) handle truncated citations
    correctly: 1 grounded surname matching 1 of 9 found authors → 100%.
    """
    if not extracted_authors or not raw:
        return extracted_authors
    raw_l = raw.lower()
    pieces = re.split(r"[,;]|\band\b", extracted_authors, flags=re.IGNORECASE)
    kept: list[str] = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        words = re.findall(r"[A-Za-zÀ-ÿ'\-]{2,}", p)
        if not words:
            continue
        surname = words[-1].lower()
        if surname in raw_l:
            kept.append(p)
    if not kept:
        # Nothing grounded — return original so we don't silently drop
        # the entire author list (e.g. raw is corrupted / non-Latin).
        return extracted_authors
    return ", ".join(kept)


def author_similarity(a1: str, a2: str) -> float | None:
    """
    Author similarity tolerant of truncated lists.

    Citations frequently truncate author lists with "et al." or "and N
    others", so the smaller set may genuinely contain a strict subset of
    the larger set's authors.  Plain Jaccard (intersection / union)
    would unfairly penalise this: 2 cited names in a 10-author paper
    scores 2/10 = 0.2 even when ALL cited names match.

    We use coverage-of-smaller when the size ratio is large (≥ 2×),
    falling back to Jaccard otherwise.  Lastnames are matched fuzzily
    (`fuzz.ratio` ≥ 85) so Müller/Mueller, OCR typos, etc. still pair.

    Returns None when the available signal is too weak for a confident
    judgement — specifically when one side has only a single surname
    token AND no substring overlap exists with the other side.  This
    catches the "cited as just 'OpenAI'" / "cited as just 'Google'"
    corporate-author case where Jaccard would collapse to 0 and trigger
    a spurious wrong_authors flag.  Callers and _composite_label treat
    None as "no author signal" and skip author-based gating.
    """
    if not a1 or not a2:
        return 0.0
    ln1 = {n for n in _extract_lastnames(a1) if n}
    ln2 = {n for n in _extract_lastnames(a2) if n}
    if not ln1 or not ln2:
        return 0.0

    # Single-token-cited / corporate-author guard.
    # When ONE side has just one surname token (e.g. "OpenAI", "Google",
    # a single last name) and the OTHER side is a non-trivial personal-
    # name list, the comparison is brittle:
    #   (a) "OpenAI" cited / "OpenAI Josh Achiam, Steven Adler, ..." found
    #       → substring matches → strong evidence of consistency → 1.0
    #   (b) "OpenAI" cited / "J Achiam, S Adler, ..." found
    #       → no substring → we genuinely don't know whether the
    #         citation is right or wrong (only one cited token to compare
    #         against) → return None so downstream treats it as
    #         "no author signal" instead of "authors disagree".
    # Whole-word / surname match only: a raw substring test let a cited
    # "Li" match "Oliver" or "Lin" in the found list.
    def _single_token_hit(tok: str, other_names: set[str], other_raw: str) -> bool:
        if tok in other_names:
            return True
        return re.search(rf"(?<![a-z]){re.escape(tok)}(?![a-z])",
                         _ascii_fold(other_raw.lower())) is not None

    if len(ln1) == 1 and len(ln2) >= 3:
        return 1.0 if _single_token_hit(next(iter(ln1)), ln2, a2) else None
    if len(ln2) == 1 and len(ln1) >= 3:
        return 1.0 if _single_token_hit(next(iter(ln2)), ln1, a1) else None

    smaller, larger = (ln1, ln2) if len(ln1) <= len(ln2) else (ln2, ln1)
    consumed: set[str] = set()
    intersection = 0
    for name in smaller:
        # Exact match (after _normalize) — cheap path
        if name in larger and name not in consumed:
            consumed.add(name)
            intersection += 1
            continue
        # Fuzzy match against unmatched names in the larger set
        best = None
        best_score = 0.0
        for other in larger:
            if other in consumed:
                continue
            s = _fuzz.ratio(name, other) / 100.0
            if s > best_score:
                best, best_score = other, s
        if best is not None and best_score >= 0.85:
            consumed.add(best)
            intersection += 1

    # Coverage path: when the larger set is at least 2× the smaller, treat
    # the smaller as a truncation of the larger.  How much of the cited
    # (smaller) set actually appears in the found (larger) set?
    if len(larger) >= 2 * len(smaller):
        return intersection / len(smaller)

    union = len(ln1) + len(ln2) - intersection
    return intersection / union if union else 0.0


def similarity_label(sim: float | None) -> str:
    if sim is None:
        return "unknown"
    if sim >= MATCH_THRESHOLD:
        return "match"
    if sim >= FUZZY_THRESHOLD:
        return "fuzzy"
    return "mismatch"


def _venue_label(vsim: float | None, found_venue: str = "") -> str:
    """
    Venue label.  arXiv is treated as a prepublisher — a found venue of 'arXiv'
    never counts as a mismatch against a conference/journal name.
    """
    if found_venue and "arxiv" in found_venue.lower():
        return "unknown"
    if vsim is None:
        return "unknown"
    if vsim >= 0.70:
        return "match"
    if vsim >= VENUE_MISMATCH_THRESHOLD:
        return "fuzzy"
    return "mismatch"


# Citation venue markers that mean "this is a preprint" — when the
# cited venue matches, we don't compare against the canonical published
# venue (which is naturally different) because the user explicitly cited
# the preprint version.
_PREPRINT_VENUE_RE = re.compile(
    r"\b(?:arxiv|preprint|biorxiv|medrxiv|chemrxiv|psyarxiv|techrxiv|"
    r"ssrn|researchgate|research\s*square|hal\b|osf\.io|ssrn\.com)\b",
    re.IGNORECASE,
)


def _is_preprint_venue(extracted_venue: str | None) -> bool:
    return bool(extracted_venue) and bool(_PREPRINT_VENUE_RE.search(extracted_venue))


def _refresh_label(result: dict[str, Any]) -> dict[str, Any]:
    """
    Recompute the composite label from a result's stored similarity fields.
    Used as a postprocess hook on cache reads so changes to
    `_composite_label` (and wrong_authors / wrong_venue gating) take
    effect without invalidating the cache.

    Note: this only refreshes the LABEL and the wrong_authors / wrong_venue
    flags — the similarity scores remain whatever was stored.  For a full
    recompute (authors_sim and venue_sim) use
    ``refresh_result_against_ref(result, ref)`` which can be called from
    app.py where the original ref data is available.
    """
    if not isinstance(result, dict) or result.get("status") != "found":
        return result
    tsim = result.get("similarity")
    asim = result.get("authors_sim")
    vsim = result.get("venue_sim")
    if tsim is None:
        return result

    finalised = _finalize_match(
        tsim, asim, vsim,
        # Cached results don't retain the original extracted_venue.
        # Without it, _finalize_match can't run the wrong_venue check
        # (it skips when extracted_venue is empty), which is exactly
        # what we want here: a label refresh shouldn't introduce a new
        # flag based on partial information.
        extracted_venue="",
        found_venue=result.get("found_venue", "") or "",
    )

    out = dict(result)
    out["label"]       = finalised["label"]
    out["venue_label"] = finalised["venue_label"]
    if finalised.get("wrong_authors"):
        out["wrong_authors"] = True
    # wrong_venue can't be recomputed without the extracted venue, so a
    # stored flag stays authoritative (refresh_result_against_ref, which
    # has the ref, re-derives it properly).
    if (result.get("wrong_doi") or result.get("wrong_arxiv_id")
            or result.get("wrong_venue")):
        out["label"] = "mismatch"
    return out


def refresh_result_against_ref(
    result: dict[str, Any],
    ref: dict[str, Any],
) -> dict[str, Any]:
    """
    Recompute authors_sim, venue_sim, and the composite label using the
    CURRENT similarity logic against the ref's currently-extracted fields.

    This is the heavyweight refresh — needed when the cached result's
    sims were computed before a logic change (e.g. before the coverage
    path in author_similarity).  Called from app.py on each rerun so the
    user doesn't have to clear the SQLite cache.

    Also honours the preprint exception: if the cited venue is a known
    preprint marker (arXiv, ResearchGate, HAL, etc.) we don't compare
    against the canonical published venue — venue_sim is set to None and
    the venue label becomes "unknown".
    """
    if not isinstance(result, dict) or result.get("status") != "found":
        return result
    tsim = result.get("similarity")
    if tsim is None:
        return result

    extracted_title   = (ref or {}).get("title", "") or ""
    extracted_authors = (ref or {}).get("authors", "") or ""
    extracted_venue   = (ref or {}).get("venue", "") or ""
    found_title       = result.get("found_title", "") or ""
    found_authors     = result.get("found_authors", "") or ""
    found_venue       = result.get("found_venue", "") or ""

    # Re-score title with the CURRENT title_similarity (it may have
    # gained the containment boost or other improvements since the
    # cached score was computed).  Otherwise a parser-poisoned title
    # like "Authors. Year. The actual title" would stay yellow forever.
    if extracted_title and found_title:
        tsim = title_similarity(extracted_title, found_title)

    # Ground the cited authors against the raw text — drops LLM-repair
    # hallucinations that aren't actually in the citation, so the
    # coverage path can correctly score a 1-grounded-name citation as
    # a match against a multi-author paper.
    grounded_authors = _grounded_cited_authors(
        extracted_authors, (ref or {}).get("raw", "") or "")
    asim = (author_similarity(grounded_authors, found_authors)
            if (grounded_authors and found_authors) else None)

    if _is_preprint_venue(extracted_venue):
        vsim = None
    else:
        vsim = (venue_similarity(extracted_venue, found_venue)
                if (extracted_venue and found_venue) else None)

    finalised = _finalize_match(
        tsim, asim, vsim, extracted_venue, found_venue,
    )
    new_label       = finalised["label"]
    new_venue_label = finalised["venue_label"]

    # Retroactive wrong-arxiv-id detection for cache entries that predate
    # the flag.  Two guards:
    #   1. The result was returned via an arxiv-ID route (carries
    #      `arxiv_id`) and title sim is below the noise floor.
    #   2. The ID also appears in the citation's *raw* text (so the
    #      user really cited it).  LLM repair sometimes injects a
    #      fabricated arxiv URL into `ref["url"]`; without this guard
    #      we'd accuse the user of a citation error they never made.
    wrong_arxiv_id = result.get("wrong_arxiv_id")
    if (not wrong_arxiv_id and not result.get("wrong_doi")
            and result.get("arxiv_id")
            and tsim is not None and tsim < 0.50
            and (ref or {}).get("title")):
        cited_id = _cited_arxiv_id(ref or {})
        if cited_id:
            wrong_arxiv_id = cited_id

    # Same rule for DOIs: only blame the citation for a DOI it prints.
    # A DOI the LLM repair added, or one GROBID took from a hidden PDF
    # link, still routes the lookup but can't produce "wrong DOI".
    if result.get("wrong_doi") and not _doi_in_citation(ref or {}, result["wrong_doi"]):
        result = {k: v for k, v in result.items() if k != "wrong_doi"}

    # wrong_doi / wrong_arxiv_id / wrong_authors / wrong_venue all force
    # mismatch — when a cited identifier resolves to a different paper,
    # or when authors / venue clearly disagree at a high-enough title sim
    # to be confident, that's stronger than any composite label.
    if (result.get("wrong_doi") or wrong_arxiv_id
            or finalised.get("wrong_authors")
            or finalised.get("wrong_venue")):
        new_label = "mismatch"

    out = {
        **result,
        "similarity":  tsim,        # may be recomputed above
        "authors_sim": asim,
        "venue_sim":   vsim,
        "label":       new_label,
        "venue_label": new_venue_label,
    }
    if wrong_arxiv_id:
        out["wrong_arxiv_id"] = wrong_arxiv_id
    if finalised.get("wrong_authors"):
        out["wrong_authors"] = True
    else:
        out.pop("wrong_authors", None)
    if finalised.get("wrong_venue"):
        out["wrong_venue"] = True
    else:
        out.pop("wrong_venue", None)
    # A one- or two-word "title" (usually a parser fragment such as an author
    # surname) can match some paper exactly; that is no evidence the citation
    # names the wrong authors or venue.  Identifier flags stay authoritative.
    if len(extracted_title.split()) < 3:
        flagged = [out.pop(k, None) for k in ("wrong_authors", "wrong_venue")]
        if any(flagged) and not (out.get("wrong_doi") or out.get("wrong_arxiv_id")):
            out["label"] = "fuzzy"
    return out


def refresh_results(results: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """Apply ``refresh_result_against_ref`` to every source in a
    ``lookup_reference`` result, in place.  Returns ``results``.

    The cache key ignores authors/venue, so a cached entry may carry sims
    computed against someone else's version of the reference — re-scoring
    against *this* ref makes UI, API and report agree."""
    for src, r in results.items():
        if not src.startswith("_") and isinstance(r, dict):
            results[src] = refresh_result_against_ref(r, ref)
    return results


def _composite_label(
    tsim: float | None,
    asim: float | None,
    found_venue: str = "",
    vsim: float | None = None,
) -> str:
    """
    Composite per-source label.

    Three colours, three meanings:

    Green ("match") — confidently the cited paper
        – title ≥ 0.90 AND (authors ≥ 0.35 OR no author data)
        – OR title ∈ [0.70, 0.90) AND authors ≥ 0.35

    Red ("mismatch") — confidently the WRONG paper at the *same* title
        – title ≥ 0.90 (essentially identical) AND authors clearly
          disagree (asim known, < 0.35).  This is the classic
          fabrication / wrong-attribution tell: a paper with this exact
          title exists, just not the one cited.
        – ALSO red when a cited identifier (DOI / arXiv ID) resolves to
          a different paper.  That case is set externally via
          `wrong_doi` / `wrong_arxiv_id` flags and forces "mismatch".

    Yellow ("fuzzy") — couldn't confidently find it
        – Tier 2 (title 0.70 – 0.90) where authors don't confirm.  A
          70-90% title overlap is usually word-overlap, not the same
          paper — we can't *prove* it's wrong, just that we didn't find
          it.  Examples: a TCT blog post returning a "language models
          in biology" academic paper at 73% title.
        – Tier 3 (title < 0.70).  The source returned the nearest-titled
          paper in its index — not evidence of a wrong paper, only of
          insufficient signal.
    """
    if tsim is None:
        return "fuzzy"

    # Tier 1: strong title (≥ 0.90) — title is essentially identical.
    if tsim >= MATCH_THRESHOLD:
        if asim is not None and asim < AUTHOR_MATCH_THRESHOLD:
            # Wrong paper at the same title — only when the titles are
            # practically identical (same floor as wrong_authors); between
            # 0.90 and the floor it is more often a different paper with a
            # similar title, i.e. "not found", not "wrong".
            return "mismatch" if tsim >= WRONG_FLAG_TITLE_FLOOR else "fuzzy"
        return "match"

    # Tier 2: medium title (0.70 – 0.90).  Likely word overlap; only
    # call it a match when authors corroborate.  Otherwise fall through
    # to "fuzzy" — we don't have enough to claim *wrong*, only that we
    # didn't confidently find it.
    if tsim >= FUZZY_THRESHOLD:
        if asim is not None and asim >= AUTHOR_MATCH_THRESHOLD:
            return "match"
        return "fuzzy"

    # Tier 3: weak title (< 0.70) — couldn't find it.
    return "fuzzy"


def _finalize_match(
    tsim: float | None,
    asim: float | None,
    vsim: float | None,
    extracted_venue: str = "",
    found_venue: str = "",
) -> dict[str, Any]:
    """
    Centralised label + wrong_authors / wrong_venue determination.

    Every source result builder should splat this into the returned dict
    in place of an inline ``_composite_label`` call.  Keeping the logic
    in one place ensures every source — DOI fast path, title fallback,
    URL verification — treats author and venue disagreement the same way.

    Returns a dict with::

        label:         "match" | "fuzzy" | "mismatch"
        venue_label:   "match" | "fuzzy" | "mismatch" | "unknown"
        wrong_authors: True (only present when set)
        wrong_venue:   True (only present when set)

    A flag is set when the title is high enough that the candidate is
    plausibly the cited paper (``tsim ≥ WRONG_FLAG_TITLE_FLOOR``) AND
    the disagreement is unambiguous.  Either flag forces ``mismatch``.

    Venue is excluded for preprints — the canonical published venue
    (e.g. ACL) doesn't match the arXiv preprint marker, but the paper
    is still correctly identified.
    """
    out: dict[str, Any] = {
        "venue_label": _venue_label(vsim, found_venue),
    }

    title_high_enough = tsim is not None and tsim >= WRONG_FLAG_TITLE_FLOOR

    if (title_high_enough
            and asim is not None
            and asim < AUTHOR_MATCH_THRESHOLD):
        out["wrong_authors"] = True

    if (title_high_enough
            and not _is_preprint_venue(extracted_venue)
            and extracted_venue
            and found_venue
            and vsim is not None
            and vsim < WRONG_VENUE_THRESHOLD):
        out["wrong_venue"] = True

    if out.get("wrong_authors") or out.get("wrong_venue"):
        out["label"] = "mismatch"
    else:
        out["label"] = _composite_label(tsim, asim, found_venue, vsim)
    return out


# ---------------------------------------------------------------------------
# Identifier extraction helpers
# ---------------------------------------------------------------------------

# Recognise the markers used to start a numbered citation entry, in
# either ACM-style ([1]) or Springer-style (39.) bibliographies.
#
# Guards against false matches on inline numbers ("Section 3.",
# "in 1990.", "1.5 is the value"):
#   - look-behind: must be at string start OR preceded by whitespace
#   - the digit run is bounded to 1-3 digits (no 4-digit years like "1990.")
#   - look-ahead: must be followed by whitespace + uppercase letter
#     (real citations start with an author surname, not a lowercase word)
_ACM_MARKER_FIND_RE = re.compile(
    r"(?:(?<=\s)|^)"               # start-of-string OR after whitespace
    r"(?:\[\d{1,3}\]|\d{1,3}\.)"   # [N] or N. (N up to 3 digits)
    r"(?=\s+[A-ZÀ-Ý])"             # followed by whitespace + uppercase
)


def _ref_raw_segment(ref: dict[str, Any]) -> str:
    """
    Return the slice of ``ref["raw"]`` that actually belongs to *this*
    reference, even when the parser left adjacent refs glued on.

    ACM-style bibliographies pack citations as ``... doi:X [N] authors
    ... title ... doi:Y [N+1] ...`` with no blank-line break.  When the
    parser doesn't split on the ``[N]`` markers cleanly, ref [N]'s raw
    can include the tail of ref [N-1] (carrying ITS DOI) plus the head
    of ref [N+1].  Routing by the first DOI found would then pick the
    PREVIOUS ref's DOI and send us to the wrong paper — exactly the
    "DOI lookup doesn't work" symptom.

    Strategy: split raw on ``[\\d+]`` markers, pick the segment whose
    text best overlaps with ``ref["title"]`` (by word-set intersection).
    Falls back to the full raw when there's nothing to split on.
    """
    raw = (ref.get("raw", "") or "").strip()
    if not raw:
        return ""

    marker_spans = list(_ACM_MARKER_FIND_RE.finditer(raw))
    if len(marker_spans) < 1:
        return raw   # no [N] markers — nothing to split

    # Build candidate segments.  The text BEFORE the first marker is the
    # tail of the previous ref; we never want that.  Segments are then
    # [marker_k.start : marker_{k+1}.start) for k in 0..len-1, plus a
    # final segment from the last marker to end.
    boundaries = [m.start() for m in marker_spans] + [len(raw)]
    segments   = [raw[boundaries[k]:boundaries[k + 1]].strip()
                  for k in range(len(boundaries) - 1)]
    if not segments:
        return raw

    # Score each segment by title-word overlap.  Cheap, dependency-free.
    title  = (ref.get("title") or "").lower()
    if not title:
        # No title to disambiguate with — prefer the LAST segment (ACM
        # places the DOI at the end of each citation, so the last one
        # is most likely the current ref's own segment).
        return segments[-1]

    title_words = {w for w in re.findall(r"[a-z0-9]{3,}", title)}
    if not title_words:
        return segments[-1]

    best_seg, best_score = segments[0], -1
    for seg in segments:
        seg_words = set(re.findall(r"[a-z0-9]{3,}", seg.lower()))
        score = len(title_words & seg_words)
        if score > best_score:
            best_score = score
            best_seg   = seg
    return best_seg


def _extract_doi(ref: dict[str, Any]) -> str | None:
    """Return DOI from the doi field or from a doi.org URL in url/raw."""
    doi = ref.get("doi", "") or ""
    if doi:
        return doi.rstrip(".,)")
    # url first (LLM-enriched, clean), then the ref-segment of raw —
    # NOT the full raw, which can include the previous ref's DOI.
    url = ref.get("url", "") or ""
    m   = _DOI_FROM_URL_RE.search(url)
    if m:
        return m.group(1).rstrip(".,)")
    m = _DOI_FROM_URL_RE.search(_ref_raw_segment(ref))
    if m:
        return m.group(1).rstrip(".,)")
    return None


def _extract_arxiv_id(ref: dict[str, Any]) -> str | None:
    # Check url first (LLM-enriched), then this ref's segment of raw,
    # then venue (Springer / ACM sometimes embed "arXiv:NNNN.NNNNN" in
    # the venue field), then doi.  Scanning the full raw risks picking
    # up adjacent refs' arxiv IDs in merged blocks.
    for field, val in (
        ("url",   ref.get("url", "") or ""),
        ("raw",   _ref_raw_segment(ref)),
        ("venue", ref.get("venue", "") or ""),
        ("doi",   ref.get("doi", "") or ""),
    ):
        m = _ARXIV_ID_RE.search(val)
        if m:
            return m.group(1) or m.group(2)
    return None


def _doi_in_citation(ref: dict[str, Any], doi: str) -> bool:
    """True when ``doi`` is printed in this reference's own raw text
    (line breaks and case ignored)."""
    squash = lambda s: re.sub(r"\s+", "", s or "").lower()
    d = squash(doi).rstrip(".,;)")
    return bool(d) and d in squash(_ref_raw_segment(ref))


def _cited_arxiv_id(ref: dict[str, Any]) -> str | None:
    """
    Return an arXiv ID ONLY when it actually appears in the citation's
    user-authored text — `raw` (original PDF text) or `venue` (parsed
    from raw text by GROBID).  These are reliable signals the user
    really cited this ID.

    NOT checked: `url`, `doi`.  The LLM repair pass occasionally injects
    a fabricated arxiv URL with a hallucinated ID into `url`; without
    this guard we'd accuse the user of citing an ID they never wrote.
    We use the broader `_extract_arxiv_id` for routing (worth a shot at
    arxiv if any field has an ID), but `_cited_arxiv_id` for "wrong
    arXiv ID" claims (only blame the user for an ID they actually cited).
    """
    # Scope `raw` to this ref's segment so we don't pick up an arxiv ID
    # from the previous/next citation in an ACM-style merged block.
    for val in (_ref_raw_segment(ref), ref.get("venue", "") or ""):
        m = _ARXIV_ID_RE.search(val)
        if m:
            return m.group(1) or m.group(2)
    return None


_URL_FOLLOW_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept":     "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
}

# Short-lived per-process memo of page fetches.  The DOI-resolution
# pre-step and the url_verify source both need the same page; this makes
# that one request instead of two.  Values are FetchResult or FetchError.
_PAGE_MEMO: dict[str, tuple[float, Any]] = {}
_PAGE_MEMO_LOCK = threading.Lock()
_PAGE_MEMO_TTL = 600
_PAGE_MEMO_MAX = 512


def _fetch_page(url: str):
    """SSRF-safe GET via netguard, memoised for ``_PAGE_MEMO_TTL`` seconds.
    Returns a ``netguard.FetchResult`` or a ``netguard.FetchError``."""
    import netguard
    now = time.time()
    with _PAGE_MEMO_LOCK:
        hit = _PAGE_MEMO.get(url)
        if hit and now - hit[0] < _PAGE_MEMO_TTL:
            return hit[1]
    _bump_http_count(url)
    try:
        res: Any = netguard.safe_get(url, headers=_URL_FOLLOW_HEADERS,
                                     timeout=10, max_bytes=512 * 1024)
    except netguard.FetchError as e:
        res = e
    with _PAGE_MEMO_LOCK:
        if len(_PAGE_MEMO) >= _PAGE_MEMO_MAX:
            for k, _ in sorted(_PAGE_MEMO.items(), key=lambda kv: kv[1][0])[:_PAGE_MEMO_MAX // 4]:
                _PAGE_MEMO.pop(k, None)
        _PAGE_MEMO[url] = (now, res)
    return res


def _resolve_url_doi(url: str) -> str | None:
    """
    Follow a URL and try to extract a DOI from:
      1. Any redirect hop or the final URL (e.g. publisher → doi.org/…)
      2. HTML <meta> citation tags (citation_doi, dc.identifier)
    Returns a DOI string or None.  Never raises.
    """
    import netguard
    r = _fetch_page(url)
    if isinstance(r, netguard.FetchError):
        return None
    for hop in [*r.history, r.url]:
        m = _DOI_FROM_URL_RE.search(hop)
        if m:
            return m.group(1).rstrip(".,)")
    if r.ok and "html" in r.content_type:
        meta = _parse_html_metadata(r.text[:60_000])
        doi_val = (meta.get("doi") or "").strip()
        doi_val = re.sub(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)", "", doi_val, flags=re.I)
        if re.match(r"10\.\d{4,}/", doi_val):
            return doi_val
    return None


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def _s2_headers(api_key: str | None) -> dict:
    return {"x-api-key": api_key} if api_key else {}


def _s2_get(path: str, params: dict, api_key: str | None) -> dict | None:
    url = f"{_S2_BASE}/{path}"
    for attempt in range(3):
        r = _SESSION.get(url, params=params, headers=_s2_headers(api_key), timeout=8)
        if r.status_code == 403:
            raise PermissionError("S2 403 Forbidden")
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("S2 rate limit persists after retries")


def _s2_paper_to_result(
    data: dict,
    extracted_title: str,
    extracted_venue: str,
    extracted_authors: str,
) -> dict[str, Any]:
    found_title   = data.get("title") or ""
    found_venue   = data.get("venue") or ""
    found_authors = ", ".join(
        a.get("name", "") for a in (data.get("authors") or [])
    )
    tsim = title_similarity(extracted_title, found_title)           if extracted_title  else None
    asim = author_similarity(extracted_authors, found_authors)      if (extracted_authors and found_authors) else None
    vsim = venue_similarity(extracted_venue, found_venue)           if (extracted_venue and found_venue) else None
    ext  = data.get("externalIds") or {}

    finalised = _finalize_match(tsim, asim, vsim, extracted_venue, found_venue)
    result: dict[str, Any] = {
        "status":        "found",
        "found_title":   found_title,
        "found_authors": found_authors,
        "found_venue":   found_venue,
        "similarity":    tsim,
        "authors_sim":   asim,
        "label":         finalised["label"],
        "venue_sim":     vsim,
        "venue_label":   finalised["venue_label"],
        "year":          data.get("year"),
        "url":           f"https://www.semanticscholar.org/paper/{data['paperId']}",
    }
    if finalised.get("wrong_authors"):
        result["wrong_authors"] = True
    if finalised.get("wrong_venue"):
        result["wrong_venue"] = True
    if ext.get("ACL"):
        acl_id = _normalize_acl_id(ext["ACL"])
        result["acl_url"] = f"https://aclanthology.org/{acl_id}"
        result["acl_id"]  = acl_id
    if ext.get("ArXiv"):
        result["arxiv_id"]  = ext["ArXiv"]
        result["arxiv_url"] = f"https://arxiv.org/abs/{ext['ArXiv']}"
    if ext.get("DOI"):
        result["doi"] = ext["DOI"]
    # Full-text pointers + abstract — consumed by the claim verifier so
    # it can compare cited claims against the actual paper content.
    # openAccessPdf is a {"url": "...", "status": "..."} dict.
    oa_pdf = (data.get("openAccessPdf") or {}).get("url") or ""
    if oa_pdf:
        result["open_access_pdf"] = oa_pdf
    abstract = data.get("abstract") or ""
    if abstract:
        result["abstract"] = abstract
    return result


@_cached("semantic_scholar", postprocess=_refresh_label)
def _lookup_semantic_scholar(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title   = ref.get("title", "")
    extracted_venue   = ref.get("venue", "")
    extracted_authors = ref.get("authors", "")
    doi = _extract_doi(ref)
    keys_to_try: list[str | None] = [S2_API_KEY, None] if S2_API_KEY else [None]

    for api_key in keys_to_try:
        try:
            data           = None
            wrong_doi      = None
            wrong_arxiv_id = None

            if doi:
                doi_data = _s2_get(f"paper/DOI:{doi}", {"fields": _S2_FIELDS}, api_key)
                if doi_data:
                    if not extracted_title or title_similarity(extracted_title, doi_data.get("title", "")) >= 0.5:
                        data = doi_data
                    else:
                        wrong_doi = doi

            if data is None:
                arxiv_id = _extract_arxiv_id(ref)
                if arxiv_id:
                    aid_data = _s2_get(f"paper/ARXIV:{arxiv_id}", {"fields": _S2_FIELDS}, api_key)
                    if aid_data:
                        # Wrong-arXiv-ID detection: same precision tell as
                        # wrong_doi — cited ID resolves to a different
                        # paper.  Only blame the user if the ID actually
                        # appears in the citation's raw text — otherwise
                        # the LLM may have fabricated it in `url` and we
                        # shouldn't surface it as a citation error.
                        if not extracted_title or title_similarity(extracted_title, aid_data.get("title", "")) >= 0.5:
                            data = aid_data
                        elif _cited_arxiv_id(ref):
                            wrong_arxiv_id = _cited_arxiv_id(ref)

            if data is None and extracted_title:
                resp   = _s2_get("paper/search", {"query": extracted_title, "limit": 3, "fields": _S2_FIELDS}, api_key)
                papers = (resp or {}).get("data", [])
                data   = papers[0] if papers else None

            if data is None:
                # If we routed by an identifier that resolved to a clearly
                # wrong paper, surface that fact instead of "not_found".
                if wrong_doi or wrong_arxiv_id:
                    r = {"status": "found", "label": "mismatch",
                         "similarity": 0.0, "found_title": "",
                         "found_authors": "", "found_venue": "",
                         "authors_sim": None, "venue_sim": None,
                         "venue_label": "unknown"}
                    if wrong_doi:      r["wrong_doi"]      = wrong_doi
                    if wrong_arxiv_id: r["wrong_arxiv_id"] = wrong_arxiv_id
                    return r
                return {"status": "not_found"}

            result = _s2_paper_to_result(data, extracted_title, extracted_venue, extracted_authors)
            if wrong_doi:      result["wrong_doi"]      = wrong_doi
            if wrong_arxiv_id: result["wrong_arxiv_id"] = wrong_arxiv_id
            return result

        except PermissionError:
            continue
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    return {"status": "error", "error": "S2 API key rejected (403)"}


# --- Batch S2 lookup for refs carrying a DOI or arXiv ID -------------------

_S2_BATCH_SIZE = 500   # S2 batch endpoint cap


def _s2_batch_get(ids: list[str], api_key: str | None) -> list[dict | None]:
    """
    POST {ids: [...]} to /paper/batch. Returns a list aligned with the
    input where each entry is either a paper dict or None (not found).
    Raises on transport errors (caller decides whether to fall back).
    """
    if not ids:
        return []
    url = f"{_S2_BASE}/paper/batch"
    headers = {**_s2_headers(api_key), "Content-Type": "application/json"}
    _bump_http_count(url)
    for attempt in range(3):
        r = _SESSION.post(
            url,
            params={"fields": _S2_FIELDS},
            headers=headers,
            json={"ids": ids},
            timeout=15,
        )
        if r.status_code == 403:
            raise PermissionError("S2 403 Forbidden")
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("S2 batch rate limit persists after retries")


def _s2_batch_prewarm(refs: list[dict[str, Any]]) -> int:
    """
    Pre-populate the S2 cache for every ref that carries a DOI or arXiv ID.
    One HTTP request replaces up to 500 sequential lookups.

    Returns the number of refs pre-warmed.  Silently no-ops on transport
    errors — the per-ref pipeline will retry normally.
    """
    if not refs:
        return 0

    cache = _get_cache()
    pending: list[tuple[str, str, dict[str, Any]]] = []
    seen_keys: set[str] = set()

    for ref in refs:
        key = _cache_key(ref)
        if not key or key in seen_keys:
            continue
        if cache.get("semantic_scholar", key) is not None:
            continue
        doi   = _extract_doi(ref)
        arxiv = _extract_arxiv_id(ref)
        if doi:
            pending.append((key, f"DOI:{doi}", ref))
            seen_keys.add(key)
        elif arxiv:
            pending.append((key, f"ARXIV:{arxiv}", ref))
            seen_keys.add(key)

    if not pending:
        return 0

    keys_to_try: list[str | None] = [S2_API_KEY, None] if S2_API_KEY else [None]
    papers: list[dict | None] | None = None
    for api_key in keys_to_try:
        try:
            papers = _s2_batch_get([s2id for _, s2id, _ in pending], api_key)
            break
        except PermissionError:
            continue
        except Exception:
            return 0   # fall back silently

    if papers is None:
        return 0

    for (key, _s2id, ref), paper in zip(pending, papers):
        if paper is None:
            cache.set("semantic_scholar", key, {"status": "not_found"})
        else:
            result = _s2_paper_to_result(
                paper,
                ref.get("title", ""),
                ref.get("venue", ""),
                ref.get("authors", ""),
            )
            cache.set("semantic_scholar", key, result)

    return len(pending)


# ---------------------------------------------------------------------------
# DBLP
# ---------------------------------------------------------------------------

@_cached("dblp", postprocess=_refresh_label)
def _lookup_dblp(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title   = ref.get("title", "")
    extracted_venue   = ref.get("venue", "")
    extracted_authors = ref.get("authors", "")
    if not extracted_title:
        return {"status": "skipped"}

    try:
        r = _http_get_retry(
            _DBLP_BASE,
            params={"q": extracted_title, "format": "json", "h": 10},
            timeout=8,
        )
        r.raise_for_status()
        hits = r.json().get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return {"status": "not_found"}

        best = max(
            hits,
            key=lambda h: title_similarity(extracted_title, h.get("info", {}).get("title", "")),
        )
        info        = best.get("info", {})
        found_title = info.get("title", "")
        if title_similarity(extracted_title, found_title) < FUZZY_THRESHOLD:
            return {"status": "not_found"}

        found_venue = info.get("venue", "")
        raw_authors = info.get("authors", {}).get("author", [])
        if isinstance(raw_authors, dict):
            raw_authors = [raw_authors]
        found_authors = ", ".join(
            a.get("text", "") if isinstance(a, dict) else str(a)
            for a in raw_authors
        )

        tsim = title_similarity(extracted_title, found_title)
        asim = author_similarity(extracted_authors, found_authors) if (extracted_authors and found_authors) else None
        vsim = venue_similarity(extracted_venue, found_venue)      if (extracted_venue and found_venue) else None
        doi  = info.get("doi", "")
        url  = info.get("url", "") or (f"https://doi.org/{doi}" if doi else "")

        finalised = _finalize_match(tsim, asim, vsim, extracted_venue, found_venue)
        out: dict[str, Any] = {
            "status":        "found",
            "found_title":   found_title,
            "found_authors": found_authors,
            "found_venue":   found_venue,
            "similarity":    tsim,
            "authors_sim":   asim,
            "label":         finalised["label"],
            "venue_sim":     vsim,
            "venue_label":   finalised["venue_label"],
            "year":          info.get("year"),
            "doi":           doi,
            "url":           url,
        }
        if finalised.get("wrong_authors"):
            out["wrong_authors"] = True
        if finalised.get("wrong_venue"):
            out["wrong_venue"] = True
        return out

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

@_cached("openalex", postprocess=_refresh_label)
def _lookup_openalex(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title   = ref.get("title", "")
    extracted_venue   = ref.get("venue", "")
    extracted_authors = ref.get("authors", "")
    doi = _extract_doi(ref)

    try:
        data = None

        if doi:
            r = _http_get_retry(
                f"{_OA_BASE}/https://doi.org/{doi}",
                params={"mailto": _OA_MAILTO},
                headers=_OA_HEADERS,
                timeout=8,
            )
            if r.status_code == 200:
                data = r.json()

        if data is None and extracted_title:
            r = _http_get_retry(
                _OA_BASE,
                params={
                    "search":   extracted_title,
                    "per_page": 3,
                    "select":   "id,title,authorships,publication_year,primary_location,doi,open_access,abstract_inverted_index",
                    "mailto":   _OA_MAILTO,
                },
                headers=_OA_HEADERS,
                timeout=8,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            if results:
                data = max(
                    results,
                    key=lambda w: title_similarity(extracted_title, w.get("title", "") or ""),
                )

        if data is None:
            return {"status": "not_found"}

        return _oa_work_to_result(
            data, extracted_title, extracted_venue, extracted_authors,
            cited_doi=doi,
        )

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _oa_work_to_result(
    data: dict,
    extracted_title: str,
    extracted_venue: str,
    extracted_authors: str,
    cited_doi: str = "",
) -> dict[str, Any]:
    found_title   = data.get("title") or ""
    doi_found     = (data.get("doi", "") or "").replace("https://doi.org/", "")
    loc           = data.get("primary_location") or {}
    source        = loc.get("source") or {}
    found_venue   = source.get("display_name") or ""
    found_authors = ", ".join(
        (a.get("author") or {}).get("display_name", "")
        for a in (data.get("authorships") or [])
    )
    oa_url = (data.get("open_access") or {}).get("oa_url") or ""
    url    = oa_url or (f"https://doi.org/{doi_found}" if doi_found else data.get("id", ""))

    tsim = title_similarity(extracted_title, found_title) if extracted_title else None
    # Title-search path: drop low-tsim hits early — the search just
    # surfaced the nearest title in the index, which isn't evidence
    # of a wrong paper.  DOI-route hits stay so we can flag wrong_doi.
    if (not cited_doi
            and extracted_title
            and tsim is not None
            and tsim < FUZZY_THRESHOLD):
        return {"status": "not_found"}
    asim = author_similarity(extracted_authors, found_authors) if (extracted_authors and found_authors) else None
    vsim = venue_similarity(extracted_venue, found_venue)      if (extracted_venue and found_venue) else None

    finalised = _finalize_match(tsim, asim, vsim, extracted_venue, found_venue)
    out: dict[str, Any] = {
        "status":        "found",
        "found_title":   found_title,
        "found_authors": found_authors,
        "found_venue":   found_venue,
        "similarity":    tsim,
        "authors_sim":   asim,
        "label":         finalised["label"],
        "venue_sim":     vsim,
        "venue_label":   finalised["venue_label"],
        "year":          data.get("publication_year"),
        "doi":           doi_found,
        "url":           url,
    }
    # DOI route: when the resolved paper's title is wildly off, the
    # cited DOI itself is wrong — surface that as the dominant signal.
    if (cited_doi
            and extracted_title
            and tsim is not None
            and tsim < WRONG_ID_TITLE_FLOOR):
        out["wrong_doi"] = cited_doi
        out["label"]     = "mismatch"
    if finalised.get("wrong_authors"):
        out["wrong_authors"] = True
    if finalised.get("wrong_venue"):
        out["wrong_venue"] = True
    # Full-text pointers + abstract — consumed by the claim verifier.
    if oa_url:
        out["open_access_pdf"] = oa_url
    abstract_text = _oa_reconstruct_abstract(data.get("abstract_inverted_index"))
    if abstract_text:
        out["abstract"] = abstract_text
    return out


def _oa_reconstruct_abstract(inv_index: dict[str, list[int]] | None) -> str:
    """
    OpenAlex returns abstracts as an inverted index
    ``{"word": [pos1, pos2, ...]}`` to sidestep publisher licensing on
    plaintext.  Rebuild the plain text by placing each word at its
    positions and joining.  Returns "" for missing / malformed input.
    """
    if not isinstance(inv_index, dict) or not inv_index:
        return ""
    positioned: dict[int, str] = {}
    for word, positions in inv_index.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            if isinstance(p, int):
                positioned[p] = word
    if not positioned:
        return ""
    return " ".join(positioned[i] for i in sorted(positioned.keys()))


def _oa_batch_prewarm(refs: list[dict[str, Any]]) -> int:
    """
    Pre-populate the OpenAlex cache by querying its filter API once per
    chunk of up to 50 DOIs.  Mirrors `_s2_batch_prewarm` and lets a DOI
    ref check two independent sources without two sequential round trips.
    """
    if not refs:
        return 0
    cache = _get_cache()

    pending: list[tuple[str, str, dict[str, Any]]] = []   # (cache_key, doi_lower, ref)
    seen: set[str] = set()
    for ref in refs:
        key = _cache_key(ref)
        if not key or key in seen:
            continue
        if cache.get("openalex", key) is not None:
            continue
        doi = _extract_doi(ref)
        if not doi:
            continue
        pending.append((key, doi.lower(), ref))
        seen.add(key)
    if not pending:
        return 0

    works_by_doi: dict[str, dict] = {}
    # DOIs whose chunk request failed.  Absence from the response only
    # means "not indexed" when the request itself succeeded — writing
    # not_found for a failed chunk would poison the cache permanently.
    failed: set[str] = set()
    for i in range(0, len(pending), 50):
        chunk = pending[i : i + 50]
        doi_filter = "|".join(d for _, d, _ in chunk)
        params = {
            "filter":   f"doi:{doi_filter}",
            "per_page": 50,
            "select":   "id,title,authorships,publication_year,primary_location,doi,open_access,abstract_inverted_index",
            "mailto":   _OA_MAILTO,
        }
        try:
            r = _http_get_retry(_OA_BASE, params=params, headers=_OA_HEADERS, timeout=15)
            r.raise_for_status()
            for work in r.json().get("results", []):
                wdoi = (work.get("doi") or "").replace("https://doi.org/", "").lower()
                if wdoi:
                    works_by_doi[wdoi] = work
        except Exception:
            failed.update(d for _, d, _ in chunk)
            continue   # leave this chunk to the per-ref fallback

    for key, doi, ref in pending:
        if doi in failed:
            continue
        work = works_by_doi.get(doi)
        if work is None:
            cache.set("openalex", key, {"status": "not_found"})
        else:
            result = _oa_work_to_result(
                work,
                ref.get("title", ""),
                ref.get("venue", ""),
                ref.get("authors", ""),
                cited_doi=doi,
            )
            cache.set("openalex", key, result)
    return len(pending)


# ---------------------------------------------------------------------------
# arXiv  (identifier only — link + ID, no title/venue validation)
# ---------------------------------------------------------------------------

class _TimeoutSession(_requests.Session):
    """requests.Session with a default timeout — the arxiv library calls
    ``session.get(url, headers=...)`` with none, so a stalled export.arxiv.org
    connection used to hang a lookup worker forever."""
    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", 15)
        return super().request(*args, **kwargs)


_ARXIV_CLIENTS: dict[int, Any] = {}
_ARXIV_CLIENTS_LOCK = threading.Lock()


def _arxiv_client(page_size: int = 100):
    """Shared arxiv.Client per page size.  One client means the library's
    built-in 3 s inter-request pacing actually applies across threads
    (a fresh client per call bypassed it).  The client isn't thread-safe
    around its pacing timestamp, so callers hold ``_ARXIV_CLIENTS_LOCK``
    while iterating results."""
    import arxiv
    with _ARXIV_CLIENTS_LOCK:
        c = _ARXIV_CLIENTS.get(page_size)
        if c is None:
            c = arxiv.Client(num_retries=3, delay_seconds=3, page_size=page_size)
            c._session = _TimeoutSession()
            _ARXIV_CLIENTS[page_size] = c
        return c


_ARXIV_CALL_LOCK = threading.Lock()


def _arxiv_batch_prewarm(refs: list[dict[str, Any]]) -> int:
    """
    Pre-populate the arXiv cache by querying its API with `id_list=`
    (max 100 IDs per request).  arXiv-bearing refs become cache hits
    instead of N sequential calls — and we hit the flaky export API
    only once per chunk.
    """
    if not refs:
        return 0
    cache = _get_cache()

    pending: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for ref in refs:
        key = _cache_key(ref)
        if not key or key in seen:
            continue
        if cache.get("arxiv", key) is not None:
            continue
        arxiv_id = _extract_arxiv_id(ref)
        if not arxiv_id:
            continue
        pending.append((key, arxiv_id, ref))
        seen.add(key)
    if not pending:
        return 0

    import arxiv
    papers_by_id: dict[str, Any] = {}
    try:
        client = _arxiv_client(100)
        for i in range(0, len(pending), 100):
            chunk = pending[i : i + 100]
            search = arxiv.Search(id_list=[p[1] for p in chunk])
            with _ARXIV_CALL_LOCK:
                chunk_papers = list(client.results(search))
            for paper in chunk_papers:
                full = paper.entry_id.split("/abs/")[-1]
                base = full.split("v")[0]     # strip version (1234.5678v3 -> 1234.5678)
                papers_by_id[base] = paper
    except Exception:
        return 0

    for key, arxiv_id, ref in pending:
        paper = papers_by_id.get(arxiv_id)
        if paper is None:
            cache.set("arxiv", key, {"status": "not_found"})
        else:
            full_id        = paper.entry_id.split("/abs/")[-1]
            found_title    = (paper.title or "").strip()
            found_authors  = ", ".join(a.name for a in (paper.authors or []) if a.name)
            extracted_title   = ref.get("title", "")
            extracted_authors = ref.get("authors", "")
            tsim = title_similarity(extracted_title, found_title) if extracted_title else None
            asim = (author_similarity(extracted_authors, found_authors)
                    if (extracted_authors and found_authors) else None)

            # Wrong-arXiv-ID detection: only blame the user if the ID
            # they cited (i.e. appears in raw text) is the one we just
            # resolved.  The LLM repair pass occasionally fabricates an
            # arxiv URL with a hallucinated ID — that ID makes it into
            # `ref["url"]` but never into `ref["raw"]`, so we don't
            # confuse the user with a "wrong ID" claim they never made.
            wrong_arxiv_id = None
            cited_id = _cited_arxiv_id(ref)
            if (tsim is not None and tsim < 0.50 and extracted_title
                    and cited_id):
                wrong_arxiv_id = cited_id

            # arXiv carries no venue field — pass vsim=None and skip the
            # venue gate.  Author gate still applies.
            finalised = _finalize_match(tsim, asim, None, "", "")
            label = finalised["label"] if tsim is not None else None
            if wrong_arxiv_id:
                label = "mismatch"

            entry = {
                "status":        "found",
                "found_title":   found_title,
                "found_authors": found_authors,
                "similarity":    tsim,
                "authors_sim":   asim,
                "label":         label,
                "venue_sim":     None,
                "venue_label":   "unknown",
                "arxiv_id":      full_id,
                "url":           paper.entry_id,
                "abstract":      (paper.summary or "").strip()[:8000],
            }
            if wrong_arxiv_id:
                entry["wrong_arxiv_id"] = wrong_arxiv_id
            if finalised.get("wrong_authors"):
                entry["wrong_authors"] = True
            cache.set("arxiv", key, entry)
    return len(pending)


@_cached("arxiv", postprocess=_refresh_label)
def _lookup_arxiv(ref: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve an arXiv ID and compare the actual paper's title against the
    cited title.  This catches the "valid arXiv ID but pointing to a
    different paper" failure mode — a common pattern in fabricated
    citations that pick a plausible-looking ID at random.

    Without title comparison, `arxiv: found` would falsely confirm a
    fabricated ref whose ID happens to resolve.
    """
    import arxiv

    arxiv_id = _extract_arxiv_id(ref)
    if not arxiv_id:
        return {"status": "skipped"}

    try:
        # Usually a cache hit thanks to _arxiv_batch_prewarm; the shared
        # client + lock keep stragglers within arXiv's 1 req / 3 s policy.
        search = arxiv.Search(id_list=[arxiv_id])
        with _ARXIV_CALL_LOCK:
            results = list(_arxiv_client(1).results(search))
        if not results:
            return {"status": "not_found"}

        paper          = results[0]
        arxiv_id_found = paper.entry_id.split("/abs/")[-1]
        found_title    = (paper.title or "").strip()
        found_authors  = ", ".join(a.name for a in (paper.authors or []) if a.name)

        extracted_title   = ref.get("title", "")
        extracted_authors = ref.get("authors", "")

        tsim = title_similarity(extracted_title, found_title) if extracted_title else None
        asim = (author_similarity(extracted_authors, found_authors)
                if (extracted_authors and found_authors) else None)

        # Wrong-arXiv-ID detection: only flag when the cited raw text
        # actually mentions an arXiv ID.  The LLM repair step sometimes
        # injects a hallucinated ID into ref["url"]; without this guard
        # we'd accuse the user of citing an ID they never wrote.
        wrong_arxiv_id = None
        cited_id = _cited_arxiv_id(ref)
        if (tsim is not None and tsim < 0.50 and extracted_title
                and cited_id):
            wrong_arxiv_id = cited_id

        # arXiv carries no venue field — pass vsim=None and skip the
        # venue gate.  Author gate still applies.
        finalised = _finalize_match(tsim, asim, None, "", "")
        label = finalised["label"] if tsim is not None else None
        if wrong_arxiv_id:
            label = "mismatch"

        result = {
            "status":        "found",
            "found_title":   found_title,
            "found_authors": found_authors,
            "similarity":    tsim,
            "authors_sim":   asim,
            "label":         label,
            "venue_sim":     None,
            "venue_label":   "unknown",
            "arxiv_id":      arxiv_id_found,
            "url":           paper.entry_id,
            "abstract":      (paper.summary or "").strip()[:8000],
        }
        if wrong_arxiv_id:
            result["wrong_arxiv_id"] = wrong_arxiv_id
        if finalised.get("wrong_authors"):
            result["wrong_authors"] = True
        return result

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Crossref  (full validation when DOI present; title-search fallback)
# ---------------------------------------------------------------------------

def _first_str(value: Any) -> str:
    """Return the first non-empty string found in `value`, unwrapping
    nested lists if Crossref hands us something like ``[["Title"]]``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            s = _first_str(item)
            if s:
                return s
    if isinstance(value, dict):
        # Crossref `event` is a dict; sometimes container-title comes as one too
        for k in ("name", "title"):
            s = _first_str(value.get(k))
            if s:
                return s
    return ""


def _crossref_msg_to_result(
    msg: dict, doi_val: str,
    extracted_title: str = "",
    extracted_venue: str = "",
    extracted_authors: str = "",
) -> dict[str, Any]:
    """Convert a Crossref `message` dict to our result schema."""
    found_title = _first_str(msg.get("title"))
    found_venue = _first_str(msg.get("container-title")) \
                  or _first_str(msg.get("event"))
    found_authors = "; ".join(
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in (msg.get("author") or [])
        if isinstance(a, dict)
    )
    # Defensive date-parts extraction.  Crossref's `published` is usually
    # a dict {"date-parts": [[YYYY, MM, DD]]}, but it can also come back
    # as a list, a string, or be missing entirely on weird records.
    year = ""
    pub = msg.get("published")
    if isinstance(pub, dict):
        parts = pub.get("date-parts")
        if isinstance(parts, list) and parts:
            first = parts[0] if isinstance(parts[0], list) else None
            if first and first[0]:
                year = str(first[0])

    tsim = title_similarity(extracted_title, found_title)      if (extracted_title and found_title) else None
    asim = author_similarity(extracted_authors, found_authors)  if (extracted_authors and found_authors) else None
    vsim = venue_similarity(extracted_venue, found_venue)       if (extracted_venue and found_venue) else None

    finalised = _finalize_match(tsim, asim, vsim, extracted_venue, found_venue)
    out: dict[str, Any] = {
        "status":        "found",
        "found_title":   found_title,
        "found_authors": found_authors,
        "found_venue":   found_venue,
        "similarity":    tsim,
        "authors_sim":   asim,
        "label":         finalised["label"],
        "venue_sim":     vsim,
        "venue_label":   finalised["venue_label"],
        "year":          year,
        "doi":           doi_val,
        "url":           f"https://doi.org/{doi_val}",
    }
    if msg.get("abstract"):
        out["abstract"] = str(msg["abstract"])[:8000]   # JATS; stripped by fulltext
    if finalised.get("wrong_authors"):
        out["wrong_authors"] = True
    if finalised.get("wrong_venue"):
        out["wrong_venue"] = True
    return out


def _crossref_batch_prewarm(refs: list[dict[str, Any]]) -> int:
    """
    Pre-populate the Crossref cache via the filter API (~25 DOIs per call).

    Crossref doesn't have a true batch endpoint, but
        GET /works?filter=doi:X,doi:Y&rows=N
    returns up to N records in one round trip.  Collapsing the per-ref
    Crossref calls is the biggest remaining wall-time win because the
    S2/OpenAlex short-circuit only helps when those two have the paper;
    real DOIs that only Crossref indexes still hit per-ref otherwise.
    """
    if not refs:
        return 0
    cache = _get_cache()

    pending: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for ref in refs:
        key = _cache_key(ref)
        if not key or key in seen:
            continue
        if cache.get("crossref", key) is not None:
            continue
        doi = _extract_doi(ref)
        if not doi:
            continue
        pending.append((key, doi, ref))
        seen.add(key)
    if not pending:
        return 0

    msgs_by_doi: dict[str, dict] = {}
    failed: set[str] = set()   # see _oa_batch_prewarm
    # 25 DOIs per query — Crossref accepts up to ~50 in the filter but the
    # URL gets long; 25 keeps us well under any limit and parallelises better.
    for i in range(0, len(pending), 25):
        chunk = pending[i : i + 25]
        doi_filter = ",".join(f"doi:{d}" for _, d, _ in chunk)
        params = {"filter": doi_filter, "rows": 25}
        try:
            r = _http_get_retry(_CR_BASE, params=params, headers=_CR_HEADERS, timeout=15)
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            for msg in items:
                d = (msg.get("DOI") or "").lower()
                if d:
                    msgs_by_doi[d] = msg
        except Exception:
            failed.update(d.lower() for _, d, _ in chunk)
            continue   # this chunk falls back to per-ref lookups

    for key, doi, ref in pending:
        if doi.lower() in failed:
            continue
        msg = msgs_by_doi.get(doi.lower())
        if msg is None:
            cache.set("crossref", key, {"status": "not_found"})
        else:
            result = _crossref_msg_to_result(
                msg, msg.get("DOI", doi),
                ref.get("title", ""), ref.get("venue", ""), ref.get("authors", ""),
            )
            cache.set("crossref", key, result)
    return len(pending)


@_cached("crossref", postprocess=_refresh_label)
def _lookup_crossref(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title   = ref.get("title", "")
    extracted_venue   = ref.get("venue", "")
    extracted_authors = ref.get("authors", "")
    doi = _extract_doi(ref)

    def _build(msg: dict, doi_val: str) -> dict[str, Any]:
        return _crossref_msg_to_result(
            msg, doi_val, extracted_title, extracted_venue, extracted_authors,
        )

    try:
        if doi:
            r = _http_get_retry(f"{_CR_BASE}/{doi}", headers=_CR_HEADERS, timeout=8)
            if r.status_code == 200:
                msg = r.json().get("message", {})
                result = _build(msg, msg.get("DOI", doi))
                # Wrong-DOI detection: when the resolved paper's title is
                # wildly off from the cited title, the user's DOI itself
                # is wrong — surface that as the dominant signal.  Mirrors
                # the same gate in _lookup_doi_org / _lookup_semantic_scholar.
                if (extracted_title and result.get("found_title")
                        and result.get("similarity") is not None
                        and result["similarity"] < WRONG_ID_TITLE_FLOOR):
                    result["wrong_doi"] = doi
                    result["label"]     = "mismatch"
                return result
            if r.status_code == 404:
                # Crossref says this DOI doesn't exist → likely fabricated.
                # Don't fall back to title search; a title-search hit on a
                # DIFFERENT paper would mask the bad DOI.
                return {"status": "not_found"}

        if extracted_title:
            r = _http_get_retry(
                _CR_BASE,
                params={"query.title": extracted_title, "rows": 1},
                headers=_CR_HEADERS,
                timeout=8,
            )
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            if items:
                result = _build(items[0], items[0].get("DOI", ""))
                if result.get("similarity") is not None and result["similarity"] < FUZZY_THRESHOLD:
                    return {"status": "not_found"}
                return result

        return {"status": "not_found"}

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Open Library  (fallback for books, proceedings, and government reports)
# ---------------------------------------------------------------------------

_OL_BASE    = "https://openlibrary.org/search.json"
_OL_HEADERS = {"User-Agent": USER_AGENT}


@_cached("openlibrary", postprocess=_refresh_label)
def _lookup_openlibrary(ref: dict[str, Any]) -> dict[str, Any]:
    """
    Search Open Library by title (+ first-author hint when available).
    Used only as a fallback when all academic databases fail — it covers
    books, proceedings volumes, government reports, and edited collections
    that are absent from S2 / DBLP / OpenAlex / Crossref.
    Venue comparison is skipped: OL returns publishers, not conference names.
    """
    extracted_title   = ref.get("title", "")
    extracted_authors = ref.get("authors", "")
    if not extracted_title:
        return {"status": "skipped"}

    try:
        params: dict[str, Any] = {
            "title":  extracted_title,
            "limit":  5,
            "fields": "key,title,author_name,first_publish_year,publisher",
        }
        # Add first author's last name as an optional refinement
        lastnames = _extract_lastnames(extracted_authors) if extracted_authors else set()
        if lastnames:
            params["author"] = next(iter(lastnames))

        r = _SESSION.get(_OL_BASE, params=params, headers=_OL_HEADERS, timeout=8)
        r.raise_for_status()
        docs = r.json().get("docs", [])

        if not docs:
            # Retry without author constraint if the first attempt was empty
            if "author" in params:
                del params["author"]
                r = _SESSION.get(_OL_BASE, params=params, headers=_OL_HEADERS, timeout=8)
                r.raise_for_status()
                docs = r.json().get("docs", [])

        if not docs:
            return {"status": "not_found"}

        best = max(
            docs,
            key=lambda d: title_similarity(extracted_title, d.get("title", "") or ""),
        )
        found_title = best.get("title", "")
        tsim = title_similarity(extracted_title, found_title)

        if tsim < FUZZY_THRESHOLD:
            return {"status": "not_found"}

        found_authors = ", ".join(best.get("author_name") or [])
        asim = author_similarity(extracted_authors, found_authors) if (extracted_authors and found_authors) else None
        year = str(best.get("first_publish_year")) if best.get("first_publish_year") else ""
        key  = best.get("key", "")
        url  = f"https://openlibrary.org{key}" if key else ""

        # OL has no venue field — pass vsim=None and skip the venue gate.
        finalised = _finalize_match(tsim, asim, None, "", "")
        out: dict[str, Any] = {
            "status":        "found",
            "found_title":   found_title,
            "found_authors": found_authors,
            "found_venue":   "",          # OL has publishers, not venues
            "similarity":    tsim,
            "authors_sim":   asim,
            "label":         finalised["label"],
            "venue_sim":     None,
            "venue_label":   "unknown",   # don't compare publisher to conference
            "year":          year,
            "url":           url,
        }
        if finalised.get("wrong_authors"):
            out["wrong_authors"] = True
        return out

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Google Scholar  (via the `scholarly` library — last-resort title verifier)
# ---------------------------------------------------------------------------
#
# Scholar has by far the broadest coverage (theses, workshop papers, books,
# non-English work) but no API, aggressive CAPTCHA, and no batch endpoint.
# We therefore:
#   - call it only as Phase 5, when no academic source matched
#   - rate-limit to one request every _SCHOLARLY_MIN_DELAY seconds
#   - back off for _SCHOLARLY_BLOCK_COOLDOWN the moment Google
#     CAPTCHAs us, so we don't waste minutes hammering a wall

_SCHOLARLY_LAST_CALL: float = 0.0
_SCHOLARLY_LOCK = threading.Lock()
# Monotonic timestamp until which Scholar is considered blocked.  A CAPTCHA
# used to disable Scholar for the life of the process; now it backs off.
_SCHOLARLY_BLOCKED_UNTIL: float = 0.0
_SCHOLARLY_BLOCK_COOLDOWN = 30 * 60
_SCHOLARLY_TIMEOUT = 20
_SCHOLARLY_MIN_DELAY = 2.5
_SCHOLARLY_BLOCK_HINTS = ("captcha", "blocked", "429", "too many requests",
                          "maxtriesexceeded")

# Per-lookup_all() budget: Scholar is rate-limited and serialised, so on
# papers with many unmatched refs it would otherwise dominate wall time.
# Each lookup_all() call owns its own budget object, so concurrent users
# (one Streamlit process, many sessions) can't drain or reset each other's.
_SCHOLARLY_BUDGET_DEFAULT  = 30
_SCHOLARLY_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scholar")


class ScholarBudget:
    def __init__(self, n: int = _SCHOLARLY_BUDGET_DEFAULT):
        self._remaining = max(0, n)
        self._lock = threading.Lock()

    def take(self) -> bool:
        """Atomically claim one Scholar call.  Returns False once depleted."""
        with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            return True


@_cached("scholarly", postprocess=_refresh_label)
def _lookup_scholarly(ref: dict[str, Any],
                      budget: ScholarBudget | None = None) -> dict[str, Any]:
    global _SCHOLARLY_LAST_CALL, _SCHOLARLY_BLOCKED_UNTIL

    if budget is None or time.monotonic() < _SCHOLARLY_BLOCKED_UNTIL:
        return {"status": "skipped"}

    extracted_title   = (ref.get("title") or "").strip()
    extracted_venue   = ref.get("venue", "") or ""
    extracted_authors = ref.get("authors", "") or ""
    if len(extracted_title) < 10:
        return {"status": "skipped"}

    # Per-lookup_all() budget: ensures Scholar doesn't dominate wall time
    # on papers with many unmatched refs.  Refuses extra calls quietly.
    if not budget.take():
        return {"status": "skipped"}

    try:
        from scholarly import scholarly as _sch
    except Exception:
        return {"status": "skipped"}

    # Pace at ~one call every 2.5 s.  The lock only covers the pacing
    # bookkeeping; the request itself runs on a single-worker pool with a
    # timeout so one hung Scholar call can't wedge every waiting thread.
    with _SCHOLARLY_LOCK:
        wait = _SCHOLARLY_MIN_DELAY - (time.time() - _SCHOLARLY_LAST_CALL)
        _SCHOLARLY_LAST_CALL = time.time() + max(wait, 0)
    if wait > 0:
        time.sleep(wait)

    try:
        pub = _SCHOLARLY_POOL.submit(_sch.search_single_pub, extracted_title) \
                             .result(timeout=_SCHOLARLY_TIMEOUT)
    except Exception as exc:
        msg = (str(exc) or type(exc).__name__).lower()
        if any(h in msg for h in _SCHOLARLY_BLOCK_HINTS):
            _SCHOLARLY_BLOCKED_UNTIL = time.monotonic() + _SCHOLARLY_BLOCK_COOLDOWN
        return {"status": "error", "error": msg[:120]}

    if not pub:
        return {"status": "not_found"}

    bib            = pub.get("bib", {}) or {}
    found_title    = (bib.get("title") or "").strip()
    if not found_title:
        return {"status": "not_found"}

    raw_authors    = bib.get("author") or []
    found_authors  = ", ".join(raw_authors) if isinstance(raw_authors, list) else str(raw_authors)
    found_venue    = (bib.get("venue") or "").rstrip("…").strip()
    year           = bib.get("pub_year") or ""
    pub_url        = pub.get("pub_url") or ""

    tsim = title_similarity(extracted_title, found_title)
    if tsim < FUZZY_THRESHOLD:
        return {"status": "not_found"}

    asim = author_similarity(extracted_authors, found_authors) if (extracted_authors and found_authors) else None
    vsim = venue_similarity(extracted_venue, found_venue)      if (extracted_venue and found_venue) else None

    finalised = _finalize_match(tsim, asim, vsim, extracted_venue, found_venue)
    out: dict[str, Any] = {
        "status":        "found",
        "found_title":   found_title,
        "found_authors": found_authors,
        "found_venue":   found_venue,
        "similarity":    tsim,
        "authors_sim":   asim,
        "label":         finalised["label"],
        "venue_sim":     vsim,
        "venue_label":   finalised["venue_label"],
        "year":          year,
        "url":           pub_url,
    }
    if finalised.get("wrong_authors"):
        out["wrong_authors"] = True
    if finalised.get("wrong_venue"):
        out["wrong_venue"] = True
    return out


# ---------------------------------------------------------------------------
# Source routing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# doi.org  (content negotiation against the canonical DOI handle system)
# ---------------------------------------------------------------------------
#
# doi.org redirects DOI resolution to the registering registry (Crossref,
# DataCite, MEDRA, JaLC, ...) and returns the metadata in CSL-JSON when
# asked via `Accept: application/vnd.citationstyles.csl+json`.  No API
# key needed — they just want a User-Agent identifying the requester
# (a `mailto:` is the conventional courtesy).
#
# Advantages over hitting Crossref/S2/OpenAlex directly:
#   - One canonical source per DOI (whichever registry minted it).
#   - Covers DataCite (datasets, software), MEDRA, JaLC — DOIs that
#     Crossref alone doesn't index.
#   - CSL-JSON shape is identical to Crossref's `message`, so we reuse
#     `_crossref_msg_to_result` for the mapping.
#
# Cached in SQLite via `@_cached("doi_org")`.

_DOI_ORG_MAILTO  = os.getenv("DOI_ORG_MAILTO") or _OA_MAILTO
_DOI_ORG_HEADERS = {
    "User-Agent": f"ARSE/1.0 (mailto:{_DOI_ORG_MAILTO})",
    "Accept":     "application/vnd.citationstyles.csl+json",
}


@_cached("doi_org", postprocess=_refresh_label)
def _lookup_doi_org(ref: dict[str, Any]) -> dict[str, Any]:
    """
    Primary DOI resolver: content-negotiates against doi.org and parses
    the returned CSL-JSON.  Falls back through the rest of the DOI vote
    (Crossref / S2 / OpenAlex) only when this fails — covered by the
    escalation phase in `lookup_reference`.
    """
    doi = _extract_doi(ref)
    if not doi:
        return {"status": "skipped"}

    extracted_title   = ref.get("title", "")
    extracted_venue   = ref.get("venue", "")
    extracted_authors = ref.get("authors", "")

    try:
        r = _http_get_retry(
            f"https://doi.org/{doi}",
            headers=_DOI_ORG_HEADERS,
            timeout=10,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:120]}

    # 404 / 410 → DOI doesn't resolve.  Anything else non-2xx is a
    # transient transport error; let the cache stay clean and let the
    # escalation phase try other registries.
    if r.status_code in (404, 410):
        return {"status": "not_found"}
    if not r.ok:
        return {"status": "error", "error": f"HTTP {r.status_code}"}

    try:
        data = r.json()
    except Exception:
        return {"status": "error", "error": "non-JSON response"}
    if not isinstance(data, dict):
        return {"status": "error", "error": "unexpected response shape"}

    # CSL-JSON returned by doi.org matches Crossref's `message` shape
    # closely enough to reuse the mapper.  Wrong-DOI detection: if the
    # resolved title is clearly different from the cited title, flag it
    # so downstream sees a confident "wrong paper" — same precision tell
    # as the S2 / arxiv routes.
    result = _crossref_msg_to_result(
        data, doi, extracted_title, extracted_venue, extracted_authors)
    if (extracted_title and result.get("found_title")
            and result.get("similarity", 1.0) < 0.50):
        result["wrong_doi"] = doi
        result["label"]     = "mismatch"
    return result


# ---------------------------------------------------------------------------
# CORE  (https://core.ac.uk) — open-access aggregator covering preprints,
# institutional repositories, theses, and grey literature that S2 / Crossref
# / OpenAlex skip.  Intentionally NOT in the default router or escalation
# — wired only as a callable for opt-in "rescue" studies on refs the
# main pipeline marked `missing`.  Activate by importing _lookup_core
# explicitly; pace at the caller (free tier is 10 req/min).
#
# Requires CORE_API_KEY in env.  Register at https://core.ac.uk/services/api
# for a free key.

@_cached("core", postprocess=_refresh_label)
def _lookup_core(ref: dict[str, Any]) -> dict[str, Any]:
    """
    Title-search CORE for refs the main pipeline flagged as `missing`.
    Returns the standard result schema so it can be merged into a ref's
    `lookup_results` dict and re-classified by `_overall_status`.

    Status semantics same as other sources:
      - `skipped`   : no title to search with, or no CORE_API_KEY in env
      - `not_found` : title search returned nothing usable
      - `found`     : best match populated (label may be match/fuzzy/mismatch)
      - `error`     : transport/parse problem
    """
    extracted_title   = ref.get("title", "")
    extracted_venue   = ref.get("venue", "")
    extracted_authors = ref.get("authors", "")
    if not extracted_title or not _CORE_API_KEY:
        return {"status": "skipped"}

    headers = {"Authorization": f"Bearer {_CORE_API_KEY}"}
    try:
        # CORE's /search/works endpoint accepts a `q` query and Lucene-
        # style filters.  Limit to 5 results — we re-rank by our own
        # title_similarity rather than trusting CORE's relevance score.
        r = _http_get_retry(
            f"{_CORE_BASE}/search/works",
            params={"q": extracted_title, "limit": 5},
            headers=headers,
            timeout=15,
        )
        if r.status_code in (401, 403):
            return {"status": "error", "error": f"CORE auth failed ({r.status_code})"}
        r.raise_for_status()
        hits = r.json().get("results", []) or []
        if not hits:
            return {"status": "not_found"}

        # Re-rank by our own title_similarity.  CORE's relevance ranking
        # can put unrelated papers at the top when the title has common
        # words ("transformer", "survey").
        ranked = sorted(
            hits,
            key=lambda h: title_similarity(extracted_title, h.get("title", "") or ""),
            reverse=True,
        )
        best = ranked[0]
        found_title = (best.get("title") or "").strip()
        if not found_title:
            return {"status": "not_found"}

        # Authors come as a list of {"name": "..."} dicts.
        authors_list = best.get("authors") or []
        found_authors = ", ".join(
            (a.get("name") or "").strip()
            for a in authors_list if isinstance(a, dict)
        ).strip(", ")

        # Venue / journal — CORE returns several possibilities.
        found_venue = (
            (best.get("publisher") or "")
            or (best.get("journals") or [{}])[0].get("title", "")
            or ""
        )

        # Identifiers
        doi      = (best.get("doi") or "").strip()
        oa_url   = best.get("downloadUrl") or best.get("sourceFulltextUrls") or ""
        if isinstance(oa_url, list):
            oa_url = oa_url[0] if oa_url else ""
        url = oa_url or (f"https://doi.org/{doi}" if doi else "") \
              or f"https://core.ac.uk/works/{best.get('id', '')}"

        tsim = title_similarity(extracted_title, found_title)
        asim = (author_similarity(extracted_authors, found_authors)
                if (extracted_authors and found_authors) else None)
        vsim = (venue_similarity(extracted_venue, found_venue)
                if (extracted_venue and found_venue) else None)

        finalised = _finalize_match(tsim, asim, vsim, extracted_venue, found_venue)
        out: dict[str, Any] = {
            "status":        "found",
            "found_title":   found_title,
            "found_authors": found_authors,
            "found_venue":   found_venue,
            "similarity":    tsim,
            "authors_sim":   asim,
            "label":         finalised["label"],
            "venue_sim":     vsim,
            "venue_label":   finalised["venue_label"],
            "year":          (best.get("yearPublished") or
                              (best.get("publishedDate") or "")[:4] or ""),
            "doi":           doi,
            "url":           url,
        }
        if finalised.get("wrong_authors"):
            out["wrong_authors"] = True
        if finalised.get("wrong_venue"):
            out["wrong_venue"] = True
        return out

    except Exception as exc:
        return {"status": "error", "error": str(exc)[:120]}


# ---------------------------------------------------------------------------
# URL verification  (any http(s) URL in the citation that isn't already
# routed by a more authoritative source)
# ---------------------------------------------------------------------------
#
# Many refs carry a URL that bypasses our identifier routes — openreview
# pages, papers.nips.cc, proceedings.mlr.press, IEEE Xplore, ACM Digital
# Library, project homepages, GitHub READMEs, blog posts.  Without a
# DOI / arxiv ID those URLs aren't reached by any of the existing sources;
# the LLM repair pass dutifully copies them into ref["url"] and they sit
# there unverified.
#
# This source fetches the URL, extracts the page's authoritative metadata
# (Google Scholar's `citation_*` meta tags — the de-facto cross-publisher
# standard, used by openreview, ACM, IEEE, MLR proceedings, NeurIPS, most
# institutional repositories), falls back to `og:title` and `<title>`, and
# scores the result against the cited title / authors / venue exactly like
# the other sources.
#
# Skipped when:
#   - the URL is doi.org / arxiv / one of the indexed registries (those
#     routes are already authoritative)
#   - the URL fails to resolve (4xx, 5xx, timeout) — surfaced as `error`
#     so the orchestrator can decide whether to escalate elsewhere
#   - the response isn't HTML (PDF / image / JSON) — we'd have to OCR or
#     parse, well outside the scope of a verification check

_URL_ROUTE_COVERED_RE = re.compile(
    r"(?:^|//|\.)(?:doi\.org|dx\.doi\.org|arxiv\.org|"
    r"semanticscholar\.org|openalex\.org|dblp\.org|aclanthology\.org)/",
    re.IGNORECASE,
)
_HTTP_URL_RE = re.compile(r'https?://[^\s<>"\)\]\}\\,]+', re.IGNORECASE)

# Google Scholar's de-facto citation_* meta tag schema.  Captures
# repeated meta tags (citation_author appears once per author on most
# publisher pages).  Tolerant of attribute order — content= can come
# before or after name=, and there can be extra whitespace.
_HTML_META_TAG_RE = re.compile(
    r'<meta\b[^>]*>',
    re.IGNORECASE,
)
_META_NAME_RE    = re.compile(r'\bname\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_META_PROP_RE    = re.compile(r'\bproperty\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_META_CONTENT_RE = re.compile(r'\bcontent\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)
_HTML_TITLE_RE   = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
# Strip common " | Site Name" / " — Site Name" suffixes from <title>.
_TITLE_SUFFIX_RE = re.compile(r"\s*[|–—\-:]\s*[^|–—\-]{3,}\s*$")


def _extract_ref_url(ref: dict[str, Any]) -> str | None:
    """Return the best candidate http(s) URL for this ref, or None."""
    url = (ref.get("url", "") or "").strip()
    if url.startswith(("http://", "https://")):
        return url
    raw = _ref_raw_segment(ref)
    if not raw:
        raw = ref.get("raw", "") or ""
    m = _HTTP_URL_RE.search(raw)
    if m:
        return m.group(0).rstrip(".,);]'\"")
    return None


def _parse_html_metadata(html_text: str) -> dict[str, Any]:
    """
    Extract title / authors / venue / year / doi from HTML.

    Order of preference:
      1. ``<meta name="citation_*">`` (Google Scholar tags — the most
         reliable cross-publisher schema; openreview, ACM, IEEE, MLR
         proceedings, NeurIPS, most institutional repositories all
         expose them).
      2. ``<meta property="og:title">`` — Open Graph fallback for sites
         without citation tags (most blog posts, project pages).
      3. ``<title>`` — last resort.  Aggressively trimmed: most CMSes
         suffix `" | Site Name"` which would tank the title similarity.
    """
    meta: dict[str, Any] = {}
    authors: list[str] = []

    for tag in _HTML_META_TAG_RE.findall(html_text):
        name_m = _META_NAME_RE.search(tag) or _META_PROP_RE.search(tag)
        cont_m = _META_CONTENT_RE.search(tag)
        if not name_m or not cont_m:
            continue
        key = name_m.group(1).strip().lower()
        val = _html_unescape(cont_m.group(1)).strip()
        if not val:
            continue

        if key == "citation_title":
            meta.setdefault("title", val)
        elif key == "citation_author":
            authors.append(val)
        elif key in ("citation_journal_title", "citation_conference_title",
                     "citation_inbook_title", "citation_book_title"):
            meta.setdefault("venue", val)
        elif key == "citation_doi":
            meta.setdefault("doi", val)
        elif key in ("dc.identifier", "dc.identifier.doi", "prism.doi"):
            m_doi = re.search(r"10\.\d{4,}/\S+", val)
            if m_doi:
                meta.setdefault("doi", m_doi.group(0))
        elif key == "citation_publication_date":
            year = val[:4]
            if year.isdigit():
                meta.setdefault("year", year)
        elif key in ("og:title", "twitter:title") and "title" not in meta:
            # Fallback only — don't overwrite a citation_title.  Trim
            # site-suffix here too in case the OG tag carries one.
            t = _TITLE_SUFFIX_RE.sub("", val).strip()
            if t:
                meta.setdefault("title", t)

    if authors:
        meta["authors"] = "; ".join(authors)

    if "title" not in meta:
        m = _HTML_TITLE_RE.search(html_text)
        if m:
            t = _html_unescape(m.group(1)).strip()
            t = re.sub(r"\s+", " ", t)
            t = _TITLE_SUFFIX_RE.sub("", t).strip()
            if t:
                meta["title"] = t

    return meta


# Lazy import to avoid pulling html at module-import time for callers
# (like the benchmark scripts) that never touch URL verification.
def _html_unescape(s: str) -> str:
    import html as _h
    return _h.unescape(s)


_GITHUB_REPO_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/([A-Za-z0-9][\w.-]*)/([\w.-]+?)(?:\.git)?(?:[/#?].*)?$",
    re.IGNORECASE,
)
_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
# Host-name decorations that sites put around the real title.
_TITLE_DECOR_RE = re.compile(r"^(?:GitHub|GitLab|Hugging Face|PyPI)\s*[-–—:]\s*", re.IGNORECASE)

# Statuses that mean "the URL itself is gone" — distinct from 403/429/5xx,
# which are usually bot-blocking or a temporary outage, not a dead link.
_DEAD_HTTP_STATUSES = {404, 410}


def _url_cache_key(ref: dict[str, Any]) -> str | None:
    """url_verify results depend on the URL, not just title/year."""
    url = _extract_ref_url(ref)
    if not url:
        return None
    return f"url={url.strip().rstrip('/')}|{_cache_key(ref) or ''}"


def _score_page(ref: dict[str, Any], meta: dict[str, Any],
                alt_titles: list[str] | None = None) -> dict[str, Any]:
    """Score page metadata against the cited fields like any other source."""
    extracted_title   = ref.get("title", "") or ""
    extracted_venue   = ref.get("venue", "") or ""
    extracted_authors = ref.get("authors", "") or ""
    found_title   = meta.get("title", "")
    found_authors = meta.get("authors", "")
    found_venue   = meta.get("venue", "")

    tsim = None
    if extracted_title:
        cands = [t for t in [found_title, *(alt_titles or [])] if t]
        if cands:
            tsim = max(title_similarity(extracted_title, t) for t in cands)
    asim = author_similarity(extracted_authors, found_authors) if (extracted_authors and found_authors) else None
    vsim = venue_similarity(extracted_venue, found_venue)      if (extracted_venue and found_venue) else None

    finalised = _finalize_match(tsim, asim, vsim, extracted_venue, found_venue)
    out: dict[str, Any] = {
        "status":        "found",
        "found_title":   found_title,
        "found_authors": found_authors,
        "found_venue":   found_venue,
        "similarity":    tsim,
        "authors_sim":   asim,
        "label":         finalised["label"],
        "venue_sim":     vsim,
        "venue_label":   finalised["venue_label"],
        "year":          meta.get("year", ""),
        "doi":           meta.get("doi", ""),
    }
    if finalised.get("wrong_authors"):
        out["wrong_authors"] = True
    if finalised.get("wrong_venue"):
        out["wrong_venue"] = True
    return out


def _github_repo(url: str) -> dict[str, Any] | None:
    """Repo metadata from the GitHub API, ``{"_dead": True}`` for a 404,
    or None when the API is unavailable (rate limit, network)."""
    m = _GITHUB_REPO_RE.match(url)
    if not m or m.group(1).lower() in ("orgs", "topics", "features", "about", "sponsors"):
        return None
    owner, repo = m.group(1), m.group(2)
    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    if _GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {_GITHUB_TOKEN}"
    try:
        _bump_http_count("https://api.github.com/")
        r = _SESSION.get(f"https://api.github.com/repos/{owner}/{repo}",
                         headers=headers, timeout=8)
    except Exception:
        return None
    if r.status_code == 404:
        return {"_dead": True}
    if not r.ok:
        return None
    return r.json()


def _wayback_snapshot(url: str) -> dict[str, str] | None:
    """Closest Internet Archive snapshot for a dead URL, or None."""
    try:
        _bump_http_count("https://archive.org/")
        r = _SESSION.get("https://archive.org/wayback/available",
                         params={"url": url}, headers={"User-Agent": USER_AGENT},
                         timeout=10)
        snap = (r.json().get("archived_snapshots") or {}).get("closest") or {}
    except Exception:
        return None
    if snap.get("available") and snap.get("url"):
        ts = snap.get("timestamp", "")
        return {"url": snap["url"].replace("http://", "https://", 1), "date": ts[:4]}
    return None


def _dead_link_result(ref: dict[str, Any], url: str, reason: str) -> dict[str, Any]:
    """URL is gone.  If the Wayback Machine has it, score the archived page
    (a real-but-moved resource then still verifies); otherwise report a
    dead link, which the UI surfaces as a warning — not as a mismatch."""
    import netguard
    snap = _wayback_snapshot(url)
    if snap:
        page = _fetch_page(snap["url"])
        if isinstance(page, netguard.FetchResult) and page.ok and "html" in page.content_type:
            meta = _parse_html_metadata(page.text)
            if meta.get("title"):
                out = _score_page(ref, meta)
                out.update(url=snap["url"], archived=True, dead_url=url,
                           note=f"cited URL is dead ({reason}); archived copy from {snap['date']}")
                return out
        return {"status": "dead_link", "url": url, "archived_url": snap["url"],
                "note": f"cited URL is dead ({reason}); archived copy from {snap['date']} exists"}
    return {"status": "dead_link", "url": url,
            "note": f"cited URL is dead ({reason}); no archived copy"}


@_cached("url_verify", postprocess=_refresh_label, key_fn=_url_cache_key)
def _lookup_url(ref: dict[str, Any]) -> dict[str, Any]:
    """
    Verify a web resource cited by URL (project pages, GitHub repos, blog
    posts, tech reports, workshop pages …).

    Returns ``status`` in:
      found      page (or its archived copy) parsed; standard similarity /
                 label fields, plus ``archived``/``dead_url`` when it came
                 from the Wayback Machine
      dead_link  404 / 410 / host does not resolve, no usable archive
      not_found  page exists but carries no usable title (PDF, image …)
      error      blocked by the SSRF guard, timeout, 403/429/5xx
      skipped    no URL, or a URL an authoritative source already covers
    """
    import netguard

    url = _extract_ref_url(ref)
    if not url:
        return {"status": "skipped"}
    if _URL_ROUTE_COVERED_RE.search(url):
        return {"status": "skipped"}

    # GitHub: the API gives a clean repo name + description, where the HTML
    # <title> is "GitHub - owner/repo: description" and scores badly.
    gh = _github_repo(url)
    if gh is not None:
        if gh.get("_dead"):
            return _dead_link_result(ref, url, "GitHub repository not found")
        name, desc = gh.get("name", ""), gh.get("description") or ""
        full = gh.get("full_name", name)
        meta = {
            "title":   f"{full}: {desc}" if desc else full,
            "authors": (gh.get("owner") or {}).get("login", ""),
            "venue":   "GitHub",
            "year":    (gh.get("created_at") or "")[:4],
        }
        spaced = re.sub(r"[_\-.]+", " ", name)
        out = _score_page(ref, meta, alt_titles=[name, spaced, desc, full,
                                                 f"{spaced}: {desc}" if desc else ""])
        out["url"] = gh.get("html_url", url)
        # Owner logins rarely equal cited author names; don't flag on that.
        out.pop("wrong_authors", None)
        out["authors_sim"] = None
        if out["label"] == "mismatch" and not out.get("wrong_venue"):
            out["label"] = "fuzzy"
        return out

    page = _fetch_page(url)
    if isinstance(page, netguard.FetchError):
        if page.kind == "dns":
            return _dead_link_result(ref, url, "host does not resolve")
        return {"status": "error", "error": f"{page.kind}: {page}"[:160], "url": url}
    if page.status_code in _DEAD_HTTP_STATUSES:
        return _dead_link_result(ref, url, f"HTTP {page.status_code}")
    if not page.ok:
        # 401/403/429/5xx — paywall, bot protection or outage, not proof
        # the resource is gone.
        return {"status": "error", "error": f"HTTP {page.status_code}", "url": page.url}

    if "html" not in page.content_type and "xml" not in page.content_type:
        # PDF / image / JSON — the resource exists but has no title to compare.
        return {"status": "not_found", "url": page.url, "note": "resource exists (not an HTML page)"}

    meta = _parse_html_metadata(page.text)
    if meta.get("title"):
        meta["title"] = _TITLE_DECOR_RE.sub("", meta["title"]).strip()
    if not meta.get("title"):
        return {"status": "not_found", "url": page.url, "note": "page has no title"}
    out = _score_page(ref, meta)
    out["url"] = page.url    # final URL after redirects — useful for the UI
    return out


_LOOKUP_FNS = {
    "doi_org":          _lookup_doi_org,
    "semantic_scholar": _lookup_semantic_scholar,
    "dblp":             _lookup_dblp,
    "openalex":         _lookup_openalex,
    "arxiv":            _lookup_arxiv,
    "crossref":         _lookup_crossref,
    "url_verify":       _lookup_url,
}
# Note: `core` is intentionally absent from _LOOKUP_FNS so the default
# router / escalation never picks it up.  It IS reachable via direct
# import (`from lookup import _lookup_core`) for opt-in rescue studies.
_ALL_API_SOURCES = list(_LOOKUP_FNS.keys())


def _classify_ref(ref: dict[str, Any]) -> list[str]:
    """
    Pick which API source(s) to query based on identifiers and venue cues.

    Identifier-bearing refs query multiple sources in parallel and vote —
    any single 503/missing entry should not mark a real paper as fabricated.
    Non-identifier refs return a single best-guess source and escalate
    if it misses.
    """
    if _extract_arxiv_id(ref):
        # arXiv only.  arXiv is authoritative for arxiv-IDs (the ID
        # routes directly to the paper), and querying other indices
        # adds latency without precision: if S2/OpenAlex disagree, it's
        # because they don't have the preprint indexed yet, not because
        # arXiv is wrong.  Skipping them collapses arxiv refs to a
        # single fast lookup.
        return ["arxiv"]
    if _extract_doi(ref):
        # doi.org first — content negotiation hits the canonical
        # registry for the DOI (Crossref / DataCite / MEDRA / JaLC).
        # The other registries are kept as fallback votes so a single
        # 503 or "not yet indexed" doesn't sink a real paper.
        return ["doi_org", "crossref", "semantic_scholar", "openalex"]

    text = " ".join([
        ref.get("raw", "") or "",
        ref.get("venue", "") or "",
    ])
    if _VENUE_DBLP_RE.search(text):
        return ["dblp"]
    if _VENUE_BOOK_RE.search(text):
        return ["openlibrary"]

    return []   # unclear — fan out


def _run_sources(
    ref: dict[str, Any],
    sources: list[str],
    early_exit_on_match: bool = True,
) -> dict[str, Any]:
    """
    Run named sources in parallel and return their results.

    With ``early_exit_on_match=True`` (the default), as soon as one source
    returns label="match" we stop waiting on the others and return.  The
    still-running threads finish in the background (they keep populating
    the cache for the next paper), but the caller no longer pays their
    tail latency.

    The slowest source on a title-only fanout would otherwise gate the
    whole ref — early-exit collapses that to "fastest match wins".
    """
    if not sources:
        return {}
    out: dict[str, Any] = {}
    fns = {s: _LOOKUP_FNS[s] for s in sources if s in _LOOKUP_FNS}
    if not fns:
        return out

    pool = ThreadPoolExecutor(max_workers=min(5, len(fns)))
    try:
        futures = {pool.submit(fn, ref): name for name, fn in fns.items()}
        for f in as_completed(futures):
            name = futures[f]
            try:
                out[name] = f.result()
            except Exception as exc:
                out[name] = {"status": "error", "error": str(exc)}

            if early_exit_on_match and out[name].get("label") == "match":
                # Mark remaining sources as skipped in the result so the
                # audit shows the short-circuit; the background threads
                # finish on their own.
                for other_name in fns:
                    out.setdefault(other_name, {"status": "skipped"})
                break
    finally:
        # wait=False returns immediately; in-flight threads continue but
        # don't block this ref's return path.
        pool.shutdown(wait=False, cancel_futures=True)
    return out


def _has_match(results: dict[str, Any], sources: list[str]) -> bool:
    return any(results.get(s, {}).get("label") == "match" for s in sources)


# ---------------------------------------------------------------------------
# Junk filter — skip parser garbage before any lookup
# ---------------------------------------------------------------------------

# Title is "2024", "2024b", possibly with trailing punctuation
_YEAR_TITLE_RE = re.compile(r"^\d{4}[a-z]?[\.\,]?$")
# Title is only digits / punctuation / whitespace
_NONALPHA_TITLE_RE = re.compile(r"^[\d\s\.\,\-\:;\(\)]+$")
# Common parser-junk strings that aren't real titles
_JUNK_TITLE_TOKENS = {"et al.", "et al", "ibid.", "ibid", "op. cit.", "n.d.", "n/a"}


def _is_junk_ref(ref: dict[str, Any]) -> bool:
    """
    Detect parser-garbage refs that aren't worth looking up.

    Catches:
      - LLM repair flagging the raw text as body-text / not a citation
      - The common pymupdf4llm failure where a year token like "2024b"
        becomes the title
      - Empty / numeric-only / near-empty titles
    These would otherwise burn a full 5-source escalation each AND poison
    the match-rate denominator.
    """
    if ref.get("_not_a_citation"):
        return True
    title = (ref.get("title") or "").strip()
    if not title:
        # No title at all — only useful if we have a DOI or arXiv ID
        return not (_extract_doi(ref) or _extract_arxiv_id(ref))

    low = title.lower().strip(".,;:")
    if low in _JUNK_TITLE_TOKENS:
        return True
    if _YEAR_TITLE_RE.match(title):
        return True
    if _NONALPHA_TITLE_RE.match(title):
        return True
    # Very short title with no authors is almost always parser noise
    if len(title) < 5 and not (ref.get("authors") or "").strip():
        return True
    return False


def _junk_result() -> dict[str, Any]:
    """Stub lookup result for a ref that was filtered as parser garbage."""
    out: dict[str, Any] = {
        src: {"status": "skipped"}
        for src in _ALL_API_SOURCES + ["openlibrary", "acl_anthology", "scholarly"]
    }
    out["_router_path"] = "junk_filter"
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _book_text(ref: dict[str, Any]) -> str:
    return " ".join([ref.get("raw") or "", ref.get("venue") or ""])


def _all_not_found(results: dict[str, Any], sources: list[str]) -> bool:
    """True only if every named source returned a clean not_found (no errors)."""
    return bool(sources) and all(
        results.get(s, {}).get("status") == "not_found" for s in sources
    )


def _has_mismatch(results: dict[str, Any], sources: list[str]) -> bool:
    """True if any chosen source returned a confident wrong-paper mismatch
    (cited identifier resolved to something with a clearly different title).
    This is an authoritative answer for id-refs — no point in escalating
    to title-only search when the cited DOI/arXiv-ID itself is wrong."""
    return any(
        results.get(s, {}).get("status") == "found"
        and results.get(s, {}).get("label") == "mismatch"
        for s in sources
    )


def lookup_reference(ref: dict[str, Any],
                     scholar_budget: ScholarBudget | None = None,
                     ) -> dict[str, Any]:
    """
    Look up a single reference (see ``_lookup_reference_core``), then make
    sure a cited non-academic URL is always checked — even when a DOI or
    title route already confirmed the paper, the reader should learn that
    the link in the citation is dead or points somewhere else.
    """
    results = _lookup_reference_core(ref, scholar_budget)
    if results.get("_router_path") == "junk_filter":
        return results
    url = _extract_ref_url(ref)
    if (url and not _URL_ROUTE_COVERED_RE.search(url)
            and (results.get("url_verify") or {}).get("status") in (None, "skipped")):
        try:
            results["url_verify"] = _lookup_url(ref)
        except Exception as exc:
            results["url_verify"] = {"status": "error", "error": str(exc)[:160]}
    return results


def _lookup_reference_core(ref: dict[str, Any],
                           scholar_budget: ScholarBudget | None = None,
                           ) -> dict[str, Any]:
    """
    Look up a single reference through a tiered pipeline:

      Phase 1  Source router — pick 1-2 APIs based on identifiers/venue.
               Identifier-based routes (arxiv / crossref) are treated as
               authoritative: a `not_found` answer there stops the pipeline.
      Phase 2  Escalate to the remaining APIs when the router missed.
      Phase 3  ACL anthology derivation from S2 externalIds / ACL DOIs.
      Phase 4  Open Library — only when `_VENUE_BOOK_RE` matches.  Earlier
               implementations ran it as a universal fallback and burned
               ~100 calls/paper for zero hits.
    """
    if _is_junk_ref(ref):
        return _junk_result()

    results: dict[str, Any] = {}

    # If the reference has a URL that isn't doi.org / arXiv, try to resolve it
    # to a DOI by following redirects and checking HTML meta tags.
    ref_url = ref.get("url", "") or ""
    if (ref_url
            and not _DOI_FROM_URL_RE.search(ref_url)
            and not _ARXIV_ID_RE.search(ref_url)
            and not _extract_doi(ref)):
        fetched_doi = _resolve_url_doi(ref_url)
        if fetched_doi:
            ref = {**ref, "doi": fetched_doi}

    if ref_url:
        results["_source_url"] = ref_url

    # --- Phase 1: Source router (with multi-source vote for ID refs) ---
    chosen = _classify_ref(ref)
    if chosen:
        is_id_ref = bool(_extract_arxiv_id(ref) or _extract_doi(ref))

        # Fast path: for ID refs, query the most authoritative sources
        # one at a time and stop early on a confident match.  Order:
        #   1. doi_org   — canonical for DOI refs
        #   2. semantic_scholar / openalex — batch-prewarmed, ms lookups
        # The remaining sources only corroborate; skip their per-ref
        # API calls once we have a match.
        chosen_to_run: list[str] = list(chosen)
        if is_id_ref:
            for fast_src in ("doi_org", "semantic_scholar", "openalex"):
                if fast_src in chosen_to_run:
                    results[fast_src] = _LOOKUP_FNS[fast_src](ref)
                    chosen_to_run = [s for s in chosen_to_run if s != fast_src]
                    if results[fast_src].get("label") == "match":
                        chosen_to_run = []
                        break

        if chosen_to_run:
            results.update(_run_sources(ref, chosen_to_run))
        # Verified: any chosen source returned a title match.
        # Fabricated (id_ref): every chosen source confirmed not_found.
        # Wrong-id (id_ref): the cited identifier resolved to a clearly
        # different paper — authoritative, no need to escalate.
        if (
            _has_match(results, chosen)
            or (is_id_ref and _all_not_found(results, chosen))
            or (is_id_ref and _has_mismatch(results, chosen))
        ):
            results["_router_path"] = f"routed:{','.join(chosen)}"
            for src in _ALL_API_SOURCES:
                results.setdefault(src, {"status": "skipped"})
            _backfill_acl_from_doi(results)
            if "acl_anthology" not in results:
                results["acl_anthology"] = {"status": "not_in_anthology"}
            results["openlibrary"] = {"status": "skipped"}
            results["scholarly"]   = {"status": "skipped"}
            return results

    # --- Phase 2: Escalation — query whatever the router didn't ---
    leftover = [s for s in _ALL_API_SOURCES if s not in results]
    if leftover:
        results.update(_run_sources(ref, leftover))
    results["_router_path"] = "escalated" if chosen else "fanout"

    # --- Phase 3: ACL anthology backfill (S2 externalIds path) ---
    if results.get("acl_anthology", {}).get("status") != "found":
        s2 = results.get("semantic_scholar", {})
        if s2.get("acl_url"):
            results["acl_anthology"] = {
                "status":      "found",
                "found_title": s2["found_title"],
                "found_venue": s2.get("found_venue", ""),
                "similarity":  s2["similarity"],
                "label":       s2["label"],
                "venue_sim":   s2.get("venue_sim"),
                "venue_label": s2.get("venue_label", "unknown"),
                "url":         s2["acl_url"],
                "acl_id":      s2.get("acl_id"),
            }
        else:
            _backfill_acl_from_doi(results)
            if "acl_anthology" not in results:
                results["acl_anthology"] = {"status": "not_in_anthology"}

    # --- Phase 4: Open Library — only if the ref looks like a book ---
    academic = ("semantic_scholar", "dblp", "openalex", "crossref", "acl_anthology")
    no_academic_match = not any(
        results.get(s, {}).get("status") == "found"
        and results.get(s, {}).get("label") == "match"
        for s in academic
    )
    if _VENUE_BOOK_RE.search(_book_text(ref)) and no_academic_match:
        results["openlibrary"] = _lookup_openlibrary(ref)
    else:
        results["openlibrary"] = {"status": "skipped"}

    # --- Phase 5: Google Scholar — last-resort title verifier ---
    # Heavy rate-limit + CAPTCHA risk → call it only when nothing else
    # confirmed the paper.  Skip when the ref is so short or junk-like that
    # a Scholar query would just burn quota.
    if no_academic_match and results["openlibrary"].get("label") != "match":
        results["scholarly"] = _lookup_scholarly(ref, scholar_budget)
    else:
        results["scholarly"] = {"status": "skipped"}

    return results


def _backfill_acl_from_doi(results: dict[str, Any]) -> None:
    """If any source returned an ACL DOI, populate results['acl_anthology']."""
    if results.get("acl_anthology", {}).get("status") == "found":
        return
    for src in ("crossref", "openalex", "dblp"):
        d = (results.get(src) or {}).get("doi", "") or ""
        m = _ACL_DOI_RE.match(d)
        if m:
            r = results[src]
            acl_id = _normalize_acl_id(m.group(1))
            results["acl_anthology"] = {
                "status":      "found",
                "found_title": r.get("found_title", ""),
                "found_venue": r.get("found_venue", ""),
                "similarity":  r.get("similarity"),
                "label":       r.get("label", "match"),
                "venue_sim":   r.get("venue_sim"),
                "venue_label": r.get("venue_label", "unknown"),
                "url":         f"https://aclanthology.org/{acl_id}",
                "acl_id":      acl_id,
            }
            return


def count_cached_refs(references: list[dict[str, Any]]) -> tuple[int, int]:
    """
    Return ``(cached, total)`` — how many of the given refs already have
    a result in the SQLite cache for the **primary validation source**
    (Semantic Scholar).

    Used by the UI to show honest expectations before kicking off a
    lookup pass: if you've processed the paper before, every ref's S2
    entry is cached and the run completes in seconds; if it's a fresh
    paper, none are cached and you'll pay full API latency.

    Uses S2 as the proxy because S2 is queried for every ref regardless
    of router path, so its hit-rate is a tight upper bound on the
    overall lookup cost.
    """
    cache = _get_cache()
    n_cached = 0
    for ref in references:
        key = _cache_key(ref)
        if key and cache.get("semantic_scholar", key) is not None:
            n_cached += 1
    return n_cached, len(references)


def fetch_abstract(doi: str) -> str:
    """Abstract for a DOI from OpenAlex, on demand — used by citation
    checks when the lookup route never queried an abstract-bearing source
    (e.g. a reference confirmed by doi.org or arXiv alone)."""
    try:
        _bump_http_count(_OA_BASE)
        r = _http_get_retry(f"{_OA_BASE}/https://doi.org/{doi}",
                            params={"mailto": _OA_MAILTO,
                                    "select": "abstract_inverted_index"},
                            headers=_OA_HEADERS, timeout=10)
        if r.status_code != 200:
            return ""
        return _oa_reconstruct_abstract(r.json().get("abstract_inverted_index"))
    except Exception:
        return ""


def forget_reference(ref: dict[str, Any]) -> int:
    """Drop every cached source result for ``ref`` so the next lookup
    re-queries live (the UI's per-reference "Re-check")."""
    return _get_cache().forget([_cache_key(ref), _url_cache_key(ref)])


def lookup_all(
    references: list[dict[str, Any]],
    delay: float = 0.0,
    progress_cb=None,
    max_concurrent: int = 8,
    scholarly_budget: int = _SCHOLARLY_BUDGET_DEFAULT,
) -> list[dict[str, Any]]:
    """
    Look up every reference with limited concurrency.

    Begins with a single S2 batch pre-warm for every ref carrying a DOI
    or arXiv ID — that collapses up to 500 sequential S2 lookups into
    one HTTP request and the per-ref S2 calls become cache hits.

    Each ref's internal API fanout uses up to 5 workers, so peak parallel
    HTTP requests = max_concurrent * 5.  With the rate-limit caps (max
    5 s sleep, 3 retries) and the S2/OpenAlex short-circuit (most DOI
    refs never hit live Crossref), 8 concurrent refs is now safe — the
    earlier cap at 5 was paired with longer 4-retry exponential backoff.

    Set ``scholarly_budget=0`` to disable Phase 5 Google Scholar lookups
    entirely (recommended for batch / benchmark runs — Scholar's 2.5 s
    rate-limit lock dominates wall time).  The UI default is off.

    A failure while looking up one reference never aborts the run: that
    reference gets ``{"_error": ...}`` and every source marked ``error``.
    """
    # Scholar budget — capped per lookup_all() call so a single paper with
    # many unmatched refs doesn't burn minutes on serialised Scholar queries.
    budget = ScholarBudget(scholarly_budget) if scholarly_budget > 0 else None

    # Identifier-batch pre-warm — one round trip per source, populates the
    # cache so per-ref S2 / OpenAlex / Crossref / arXiv calls become local
    # lookups.  Four sources run concurrently against four independent APIs.
    _batch_fns = (
        _s2_batch_prewarm,
        _oa_batch_prewarm,
        _crossref_batch_prewarm,
        _arxiv_batch_prewarm,
    )
    with ThreadPoolExecutor(max_workers=len(_batch_fns)) as _pool:
        for _fut in [_pool.submit(_fn, references) for _fn in _batch_fns]:
            # Pre-warm is an optimisation; a failure just means per-ref calls.
            if _fut.exception() is not None:
                log.warning("pre-warm failed: %r", _fut.exception())

    results: list[Any] = [None] * len(references)
    _lock = threading.Lock()

    def _worker(i: int, ref: dict) -> int:
        try:
            res = refresh_results(lookup_reference(ref, budget), ref)
        except Exception as exc:
            log.exception("lookup failed for ref %d", i)
            res = {src: {"status": "error", "error": str(exc)[:200]}
                   for src in _ALL_API_SOURCES}
            res["_error"] = f"{type(exc).__name__}: {exc}"[:300]
        with _lock:
            results[i] = res
        return i

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_worker, i, ref): i for i, ref in enumerate(references)}
        done = 0
        for f in as_completed(futures):
            f.result()  # propagate exceptions
            done += 1
            if progress_cb:
                # Called on the main thread — safe for Streamlit UI updates
                progress_cb(done, len(references))

    return results
