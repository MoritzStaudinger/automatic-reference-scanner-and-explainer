"""
Best-effort full-text acquisition for a cited paper.

Consumed by the claim verifier: given a reference (plus the lookup
results we already gathered for it), try — in priority order — to
download a PDF, parse it, and hand back plain text.  Fall back to the
paper's abstract when no PDF is reachable.  Return None only when the
paper has neither an OA PDF nor an abstract anywhere.

Cached in the same SQLite table the rest of the pipeline uses; the
cache key is the source URL, so re-runs (batch reprocess, iterating on
prompts, etc.) don't refetch or re-parse.

Priority chain (first success wins):
  1. arXiv PDF (via arxiv_url from any source)
  2. Semantic Scholar `openAccessPdf`
  3. OpenAlex `open_access.oa_url`
  4. CORE `downloadUrl` (opt-in — only when the CORE result is present)
  5. Abstract fallback (S2 abstract → OpenAlex reconstructed abstract)
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from cache import get_cache
from lookup import USER_AGENT, _SESSION      # connection-pooled session


# Lazy CORE support: only try CORE for refs whose OA chain came up empty,
# and only when the user has configured CORE_API_KEY.  CORE is a broad OA
# aggregator (millions of institutional-repository papers not on arXiv /
# S2 / OpenAlex) — especially strong for non-CS/ML domains — but its
# free tier is rate-limited (10 req/min).  Deferring calls to the
# fallback path keeps us well inside quota on typical papers.
import os as _os
_CORE_API_KEY = _os.getenv("CORE_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# 30 s per URL is generous — arxiv can be slow, OA hosts on the free tier
# can be slower.  10 MB cap protects the pipeline from a massive PDF
# blowing memory (unusual for a paper — most are <2 MB).
_FETCH_TIMEOUT_S = 30
_FETCH_MAX_BYTES = 10 * 1024 * 1024

# Cap the parsed text at ~120 K chars.  A typical academic paper is
# 30-60 K chars; 120 K accommodates long survey papers without blowing
# the LLM context (Sonnet 4.6 has 200 K, so ~120 K leaves room for the
# claim + prompt + response).
_MAX_TEXT_CHARS = 120_000

_UA = USER_AGENT


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class FullText:
    """Result of a fetch attempt."""
    text:                 str
    source:               str          # arxiv | s2_oa | openalex_oa | core_oa | abstract_s2 | abstract_openalex
    is_abstract_fallback: bool
    url:                  Optional[str]      # None for abstract-only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_SOURCE = "fulltext"


def _cached_get(key: str) -> Optional[FullText]:
    raw = get_cache().get(_CACHE_SOURCE, key)
    if not raw:
        return None
    try:
        return FullText(**raw)
    except (TypeError, ValueError):
        return None


def _cached_set(key: str, ft: FullText) -> None:
    # Even short abstracts are worth caching so re-runs don't hit OpenAlex.
    get_cache().set(_CACHE_SOURCE, key, ft.to_dict())


# ---------------------------------------------------------------------------
# PDF fetch + parse
# ---------------------------------------------------------------------------

def _http_get_pdf(url: str) -> Optional[bytes]:
    """
    Fetch a URL that we expect to return a PDF.  Returns bytes on
    success, None on any failure (network error, non-2xx, wrong
    content-type, oversize).  Silent — the caller cascades to the next
    source in the priority chain.
    """
    try:
        r = _SESSION.get(
            url,
            headers={"User-Agent": _UA, "Accept": "application/pdf"},
            timeout=_FETCH_TIMEOUT_S,
            allow_redirects=True,
            stream=True,
        )
    except Exception:
        return None
    if not r.ok:
        return None
    ctype = (r.headers.get("Content-Type") or "").lower()
    # Some OA hosts serve PDFs with generic octet-stream content type.
    if "pdf" not in ctype and "octet-stream" not in ctype:
        return None
    try:
        # Read in chunks to enforce the size cap without buffering
        # arbitrarily large responses.
        buf = bytearray()
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > _FETCH_MAX_BYTES:
                return None
        return bytes(buf) if buf else None
    finally:
        r.close()


def _pdf_bytes_to_text(pdf_bytes: bytes) -> Optional[str]:
    """
    Extract plain text from PDF bytes via pymupdf4llm.
    Returns None on any parse failure — caller cascades.
    """
    if not pdf_bytes.startswith(b"%PDF"):
        return None
    import pymupdf4llm
    # pymupdf4llm needs a file path, not bytes.  Write to a tmpfile,
    # unlink in finally so a parse crash doesn't leak.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name
    try:
        text = pymupdf4llm.to_markdown(path, use_ocr=False)
    except Exception:
        return None
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass
    text = (text or "").strip()
    if not text:
        return None
    return text[:_MAX_TEXT_CHARS]


# ---------------------------------------------------------------------------
# URL discovery from lookup results
# ---------------------------------------------------------------------------

def _extract_urls(lookup_sources: dict[str, Any]) -> list[tuple[str, str]]:
    """
    Walk a ref's lookup_sources dict and yield (source_tag, url) pairs
    in the priority order the fetcher should try.
    """
    urls: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(tag: str, url: str) -> None:
        if not url or not url.startswith("http"):
            return
        # Deduplicate — same PDF can appear on multiple sources.
        if url in seen:
            return
        seen.add(url)
        urls.append((tag, url))

    # Priority 1: arXiv (any source that recognised an arxiv id)
    for src in ("arxiv", "semantic_scholar", "openalex", "crossref"):
        r = lookup_sources.get(src) or {}
        arxiv_url = r.get("arxiv_url") or ""
        arxiv_id = r.get("arxiv_id") or ""
        if arxiv_url:
            # Convert /abs/{id} to /pdf/{id}.pdf
            if "/abs/" in arxiv_url:
                pdf_url = arxiv_url.replace("/abs/", "/pdf/").rstrip("/") + ".pdf"
                _add("arxiv", pdf_url)
            else:
                _add("arxiv", arxiv_url)
        elif arxiv_id:
            _add("arxiv", f"https://arxiv.org/pdf/{arxiv_id}.pdf")

    # Priority 2: S2 openAccessPdf
    s2 = lookup_sources.get("semantic_scholar") or {}
    _add("s2_oa", s2.get("open_access_pdf", ""))

    # Priority 3: OpenAlex OA URL
    oa = lookup_sources.get("openalex") or {}
    _add("openalex_oa", oa.get("open_access_pdf", ""))

    # Priority 4: CORE (only present when the ref was routed through
    # the opt-in CORE lookup)
    core = lookup_sources.get("core") or {}
    core_url = core.get("open_access_pdf", "") or core.get("url", "")
    _add("core_oa", core_url)

    return urls


def _extract_abstract(lookup_sources: dict[str, Any]) -> tuple[str, str]:
    """
    Prefer S2 abstract (usually longer + cleaner); fall back to OpenAlex.
    Returns (text, source_tag) — empty text signals no abstract available.
    """
    s2 = (lookup_sources.get("semantic_scholar") or {}).get("abstract") or ""
    if s2:
        return s2, "abstract_s2"
    oa = (lookup_sources.get("openalex") or {}).get("abstract") or ""
    if oa:
        return oa, "abstract_openalex"
    return "", ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _try_urls(urls: list[tuple[str, str]]) -> Optional[FullText]:
    """
    Try each (tag, url) in order; return the first FullText we can
    build.  Common helper for the primary + fallback URL lists.
    """
    for src_tag, url in urls:
        cached = _cached_get(url)
        if cached is not None:
            return cached
        pdf_bytes = _http_get_pdf(url)
        if not pdf_bytes:
            continue
        text = _pdf_bytes_to_text(pdf_bytes)
        if not text:
            continue
        ft = FullText(text=text, source=src_tag,
                      is_abstract_fallback=False, url=url)
        _cached_set(url, ft)
        return ft
    return None


def _core_lookup_urls(ref: dict[str, Any]) -> list[tuple[str, str]]:
    """
    Lazy CORE lookup: only fires when the primary chain empty AND
    CORE_API_KEY is configured.  Returns (tag, url) pairs the same
    shape ``_try_urls`` consumes.
    """
    if not _CORE_API_KEY or not ref:
        return []
    try:
        from lookup import _lookup_core
        result = _lookup_core(ref) or {}
    except Exception:
        return []
    if result.get("status") != "found":
        return []
    url = result.get("open_access_pdf") or result.get("url") or ""
    if not url or not url.startswith("http"):
        return []
    return [("core_oa", url)]


def _fetch_abstract_only(lookup_sources: dict[str, Any]) -> Optional[FullText]:
    """
    Return the cited paper's abstract without hitting the OA PDF chain.
    Called by the fast verification path — abstracts are already in
    lookup_sources (S2 + OpenAlex populate them during the standard
    lookup), so this stage is effectively free.
    """
    abs_text, abs_src = _extract_abstract(lookup_sources)
    if not abs_text:
        return None
    abs_text = abs_text[:_MAX_TEXT_CHARS]
    ft = FullText(text=abs_text, source=abs_src,
                  is_abstract_fallback=True, url=None)
    import hashlib
    key = f"abstract:{hashlib.sha256(abs_text.encode('utf-8')).hexdigest()[:16]}"
    _cached_set(key, ft)
    return ft


def fetch(
    lookup_sources: dict[str, Any],
    ref:            Optional[dict[str, Any]] = None,
    depth:          str = "fulltext",
) -> Optional[FullText]:
    """
    Try to acquire the cited paper's full text.  Falls back to abstract
    when no OA PDF works.  Returns None only when neither is available.

    Parameters
    ----------
    lookup_sources : dict
        Per-ref dict of source results already gathered by
        ``lookup_reference`` (the ``lookup_sources`` field on
        ``RefReport`` in pipeline.py — pass it in directly).
    ref : dict, optional
        The parsed reference itself (title, authors, year, doi, ...).
        Used to fire a lazy CORE lookup when the primary chain came up
        empty AND ``CORE_API_KEY`` is set.  Pass it whenever you have it.
    depth : "fulltext" | "abstract"
        ``fulltext`` (default): try the OA PDF chain first; fall back to
        the abstract when no PDF is reachable.  ``abstract``: skip the
        PDF chain entirely and return the abstract directly — used by
        the fast verification path so a paper's worth of claims can be
        verified in seconds instead of minutes.
    """
    if depth == "abstract":
        return _fetch_abstract_only(lookup_sources)

    # Priority chain from the sources we already queried.
    urls = _extract_urls(lookup_sources)
    ft = _try_urls(urls)
    if ft is not None:
        return ft

    # Lazy CORE fallback — only pays the rate-limited cost for refs the
    # standard chain missed.  Silently skipped when CORE_API_KEY is
    # unset, so unconfigured deployments don't pay any latency here.
    if ref is not None:
        ft = _try_urls(_core_lookup_urls(ref))
        if ft is not None:
            return ft

    # Abstract fallback — cheap and always worth trying.
    return _fetch_abstract_only(lookup_sources)
