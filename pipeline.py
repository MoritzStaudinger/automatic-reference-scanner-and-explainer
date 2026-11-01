"""
Headless ARES pipeline — parse → (LLM repair) → dedupe → lookup → report.

Used by both:
  - the Streamlit UI (app.py) for interactive single-PDF runs, and
  - the HTTP API (api.py) for sync single-PDF calls and async batch jobs.

No Streamlit or web-framework dependencies — pure functions that take
bytes/options in and return a serialisable report out.  The function
``process_pdf`` is the only public entry point; everything else is
internal plumbing.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

from parsers import (
    _locate_reference_section,
    dedupe_references,
    extract_references_grobid,
    extract_references_hybrid,
    extract_references_pymupdf4llm,
    ParserError,
)
from lookup import lookup_all
from report import _best_found, _categorise_suspicious
from verdict import overall_status as _overall_status, status_reason


# ---------------------------------------------------------------------------
# Options + report shapes
# ---------------------------------------------------------------------------

@dataclass
class PipelineOptions:
    """User-tunable knobs for one pipeline run."""
    parser: str = "grobid"                  # grobid | hybrid | pymupdf4llm
    use_llm: bool = False                   # run LLM-based field repair
    llm_backend: Optional[str] = None       # anthropic | aqueduct (None = env default)
    llm_model: Optional[str] = None         # specific model id (None = backend default)
    verify_claims: bool = False             # run per-citation claim verification
                                            # (slow: extracts inline citations,
                                            # fetches cited papers, LLM verdicts)
    verify_scope: str = "matched"           # matched | all
                                            # "matched": only verify claims that
                                            #   point at refs the lookup phase
                                            #   found in a database (green).
                                            #   Wasting LLM budget on
                                            #   fabricated / paywalled refs
                                            #   just yields no_source verdicts.
                                            # "all": verify every claim, treat
                                            #   no-source cases as best-effort.
    verify_depth: str = "abstract"          # abstract | fulltext
                                            # "abstract" (default, fast):
                                            #   verify claims against the cited
                                            #   paper's ABSTRACT only.  ~30-60s
                                            #   for a typical paper on Anthropic,
                                            #   ~1-3 min on Aqueduct.  Abstracts
                                            #   don't cover every cited detail
                                            #   so some claims will come back
                                            #   `not_addressed` that a fulltext
                                            #   pass could have supported.
                                            # "fulltext" (slow, high-recall):
                                            #   fetch each cited paper's OA PDF
                                            #   and verify against the whole
                                            #   thing.  Minutes-to-hours per
                                            #   paper depending on LLM backend.
    verify_refs:  Optional[list[int]] = None  # None → verify all in-scope refs;
                                              # list of ref indices → only those.
                                              # Used by the per-ref "escalate to
                                              # fulltext" flow in the UI.


@dataclass
class RefReport:
    """Per-reference verdict + the data needed to act on it."""
    index: int
    title: str
    authors: str
    venue: str
    year: str
    doi: str
    url: str
    raw: str
    page: Optional[int]
    overall_status: str                     # match | mismatch | missing | pending
    reason: str                             # one-line explanation, as shown in the UI
    category: Optional[str]                 # wrong_doi / wrong_arxiv_id / wrong_authors / ...
    best_source: Optional[str]              # the source that returned the best hit
    best_url: Optional[str]                 # URL of the best candidate (clickable)
    best_title: Optional[str]               # found_title from best candidate
    best_authors: Optional[str]
    best_venue: Optional[str]
    best_similarity: Optional[float]
    best_authors_sim: Optional[float]
    best_venue_sim: Optional[float]
    wrong_doi: Optional[str]                # cited DOI flagged as resolving elsewhere
    wrong_arxiv_id: Optional[str]
    wrong_authors: bool
    wrong_venue: bool
    lookup_sources: dict[str, Any]          # raw per-source results for power users


@dataclass
class ClaimVerificationReport:
    """One (claim, cited-ref) verdict — mirrors claim_verifier.ClaimVerification."""
    claim_index:          int
    cited_ref_index:      int
    claim_text:           str
    verdict:              str          # supported | contradicted | not_addressed | ambiguous | error
    evidence_passage:     str
    confidence:           str          # high | medium | low
    explanation:          str
    source:               str          # arxiv | s2_oa | openalex_oa | core_oa | abstract_s2 | abstract_openalex
    is_abstract_fallback: bool
    model:                str
    citation_marker:      str          # the raw marker as it appears in text
    style:                str          # numeric | paren | narrative
    error:                str = ""     # why the check failed (verdict == "error")


@dataclass
class PipelineReport:
    """Top-level result for one PDF."""
    n_refs: int
    n_match: int
    n_mismatch: int
    n_missing: int
    refs: list[RefReport]
    parse_seconds: float
    repair_seconds: float
    lookup_seconds: float
    parser: str
    used_llm: bool
    # LLM repair outcome — {backend, model, n_sent, n_failed_refs, first_error, ...}
    llm_stats: dict[str, Any] = field(default_factory=dict)
    # Set when the requested parser failed and pymupdf4llm was used instead.
    parser_fallback: str = ""
    # Claim verification block — empty when opts.verify_claims is False.
    verify_claims_seconds: float = 0.0
    n_claims:           int = 0
    n_supported:        int = 0
    n_contradicted:     int = 0
    n_not_addressed:    int = 0
    n_ambiguous:        int = 0
    n_verify_errors:    int = 0
    n_verify_no_source: int = 0        # claims we couldn't verify (no full text and no abstract)
    claim_verifications: list["ClaimVerificationReport"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "n_refs":     self.n_refs,
            "n_match":    self.n_match,
            "n_mismatch": self.n_mismatch,
            "n_missing":  self.n_missing,
            "parser":     self.parser,
            "used_llm":   self.used_llm,
            **({"llm_repair": self.llm_stats} if self.llm_stats else {}),
            **({"parser_fallback": self.parser_fallback} if self.parser_fallback else {}),
            "timings_seconds": {
                "parse":  round(self.parse_seconds, 2),
                "repair": round(self.repair_seconds, 2),
                "lookup": round(self.lookup_seconds, 2),
            },
        }
        if self.claim_verifications or self.verify_claims_seconds:
            summary["claim_verification"] = {
                "n_claims":           self.n_claims,
                "n_supported":        self.n_supported,
                "n_contradicted":     self.n_contradicted,
                "n_not_addressed":    self.n_not_addressed,
                "n_ambiguous":        self.n_ambiguous,
                "n_verify_errors":    self.n_verify_errors,
                "n_verify_no_source": self.n_verify_no_source,
            }
            summary["timings_seconds"]["verify_claims"] = round(
                self.verify_claims_seconds, 2)
        return {
            "summary": summary,
            "refs":    [asdict(r) for r in self.refs],
            "claim_verifications": [asdict(cv) for cv in self.claim_verifications],
        }


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _run_parser(pdf_path: str, parser: str) -> tuple[list[dict[str, Any]], str]:
    """Dispatch to the chosen parser backend."""
    if parser == "hybrid":
        return extract_references_hybrid(pdf_path, return_markdown=True)
    if parser == "grobid":
        return extract_references_grobid(pdf_path, return_markdown=True)
    if parser == "pymupdf4llm":
        return extract_references_pymupdf4llm(pdf_path, return_markdown=True)
    raise ValueError(
        f"unknown parser {parser!r} — expected one of: "
        f"grobid, hybrid, pymupdf4llm"
    )


def _maybe_run_llm(refs: list[dict[str, Any]], opts: PipelineOptions,
                   stats: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not opts.use_llm:
        return refs
    # Import lazily — `llm_parser` imports anthropic / openai which are
    # only needed when the user opted in.  Saves ~150ms startup on the API.
    from llm_parser import parse_references_with_llm
    # Per-call backend/model: set_active_model mutates module state and
    # would race between concurrent API requests.
    return parse_references_with_llm(refs, stats=stats, backend=opts.llm_backend,
                                     model=opts.llm_model)


def _ref_to_report(
    i: int,
    ref: dict[str, Any],
    lr: Optional[dict[str, Any]],
) -> RefReport:
    """Convert one ref + its lookup-result dict into a RefReport."""
    status = _overall_status(lr) if lr is not None else "pending"
    category = _categorise_suspicious(lr) if (lr is not None and status not in ("match", "pending")) else None
    best = _best_found(lr) if lr else None

    # Find the source name that produced `best` so the consumer can
    # route the user to the right URL (S2 result page, Crossref work
    # page, openreview, etc.).
    best_source = None
    if best is not None and lr is not None:
        for src, r in lr.items():
            if r is best:
                best_source = src
                break

    wrong_doi      = None
    wrong_arxiv_id = None
    wrong_authors  = False
    wrong_venue    = False
    if lr is not None:
        for src, r in lr.items():
            if not isinstance(r, dict):
                continue
            wrong_doi      = wrong_doi      or r.get("wrong_doi")
            wrong_arxiv_id = wrong_arxiv_id or r.get("wrong_arxiv_id")
            wrong_authors  = wrong_authors  or bool(r.get("wrong_authors"))
            wrong_venue    = wrong_venue    or bool(r.get("wrong_venue"))

    return RefReport(
        reason            = status_reason(lr),
        index             = i,
        title             = ref.get("title", "") or "",
        authors           = ref.get("authors", "") or "",
        venue             = ref.get("venue", "") or "",
        year              = str(ref.get("year") or ""),
        doi               = ref.get("doi", "") or "",
        url               = ref.get("url", "") or "",
        raw               = ref.get("raw", "") or "",
        page              = ref.get("page"),
        overall_status    = status,
        category          = category,
        best_source       = best_source,
        best_url          = (best or {}).get("url"),
        best_title        = (best or {}).get("found_title"),
        best_authors      = (best or {}).get("found_authors"),
        best_venue        = (best or {}).get("found_venue"),
        best_similarity   = (best or {}).get("similarity"),
        best_authors_sim  = (best or {}).get("authors_sim"),
        best_venue_sim    = (best or {}).get("venue_sim"),
        wrong_doi         = wrong_doi,
        wrong_arxiv_id    = wrong_arxiv_id,
        wrong_authors     = wrong_authors,
        wrong_venue       = wrong_venue,
        lookup_sources    = lr or {},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_bytes: bytes,
    options: Optional[PipelineOptions] = None,
) -> PipelineReport:
    """
    Parse a PDF's bibliography, optionally repair with an LLM, dedupe, and
    look up every reference against the configured external sources.

    Returns a fully-realised :class:`PipelineReport`.  Safe to call from
    threads — the underlying lookup pipeline uses a connection-pooled
    requests Session and the SQLite cache is thread-safe.
    """
    opts = options or PipelineOptions()

    # 1. Parse — needs an on-disk path (parsers use fitz/pymupdf which
    #    open files, not bytes).  We write to a tmpfile and unlink in a
    #    `finally` so a parser crash doesn't leak the file.
    t0 = time.time()
    parser_fallback = ""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        try:
            refs, _debug_md = _run_parser(tmp_path, opts.parser)
        except ParserError as exc:
            # GROBID down: degrade to the GROBID-free parser and say so.
            parser_fallback = f"{exc}; used pymupdf4llm"
            refs, _debug_md = _run_parser(tmp_path, "pymupdf4llm")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    parse_s = time.time() - t0

    # 2. LLM repair (optional)
    t = time.time()
    llm_stats: dict[str, Any] = {}
    refs = _maybe_run_llm(refs, opts, llm_stats)
    repair_s = time.time() - t

    # 3. Dedupe near-duplicates introduced by the parser (e.g. one ref
    #    parsed once from the bibliography and once from a body footnote).
    refs, _n_merged = dedupe_references(refs)

    # 4. Lookup — fan out to the configured sources per ref.
    t = time.time()
    # Scholar scraping from a server IP gets CAPTCHA-blocked fast; never in API/batch.
    lookup_results = lookup_all(refs, scholarly_budget=0)
    lookup_s = time.time() - t

    # 5. Tabulate
    ref_reports: list[RefReport] = []
    n_match = n_mismatch = n_missing = 0
    for i, ref in enumerate(refs):
        lr = lookup_results[i] if i < len(lookup_results) else None
        rr = _ref_to_report(i, ref, lr)
        if   rr.overall_status == "match":    n_match    += 1
        elif rr.overall_status == "mismatch": n_mismatch += 1
        elif rr.overall_status == "missing":  n_missing  += 1
        ref_reports.append(rr)

    # 6. Claim verification (optional, slow, per-citation LLM)
    verify_s = 0.0
    claim_verdicts: list[ClaimVerificationReport] = []
    if opts.verify_claims:
        t = time.time()
        claim_verdicts = _verify_claims_stage(pdf_bytes, refs, lookup_results, opts)
        verify_s = time.time() - t

    counts = _count_verdicts(claim_verdicts)

    return PipelineReport(
        n_refs          = len(refs),
        n_match         = n_match,
        n_mismatch      = n_mismatch,
        n_missing       = n_missing,
        refs            = ref_reports,
        parse_seconds   = parse_s,
        repair_seconds  = repair_s,
        lookup_seconds  = lookup_s,
        parser          = opts.parser,
        used_llm        = opts.use_llm and not (
            llm_stats.get("n_sent", 0) == 0 or
            llm_stats.get("n_failed_refs") == llm_stats.get("n_sent")),
        llm_stats       = llm_stats,
        parser_fallback = parser_fallback,
        verify_claims_seconds = verify_s,
        n_claims           = counts["n_claims"],
        n_supported        = counts["n_supported"],
        n_contradicted     = counts["n_contradicted"],
        n_not_addressed    = counts["n_not_addressed"],
        n_ambiguous        = counts["n_ambiguous"],
        n_verify_errors    = counts["n_verify_errors"],
        n_verify_no_source = counts["n_verify_no_source"],
        claim_verifications = claim_verdicts,
    )


# ---------------------------------------------------------------------------
# Claim verification stage
# ---------------------------------------------------------------------------

def _count_verdicts(verdicts: list["ClaimVerificationReport"]) -> dict[str, int]:
    out = {
        "n_claims":           len(verdicts),
        "n_supported":        0,
        "n_contradicted":     0,
        "n_not_addressed":    0,
        "n_ambiguous":        0,
        "n_verify_errors":    0,
        "n_verify_no_source": 0,
    }
    for v in verdicts:
        k = "n_" + v.verdict if v.verdict in ("supported", "contradicted",
                                              "not_addressed", "ambiguous") else None
        if k:
            out[k] += 1
        elif v.verdict == "error":
            out["n_verify_errors"] += 1
        elif v.verdict == "no_source":
            out["n_verify_no_source"] += 1
    return out


def extract_claims_from_pdf(
    pdf_bytes:  bytes,
    refs:       list[dict[str, Any]],
) -> list[Any]:
    """
    Re-parse the citing PDF's body (everything BEFORE the References
    section) and pull out per-citation claims.  Fast — no network calls,
    no LLM.  Returns ``list[claim_extractor.Claim]``.
    """
    from claim_extractor import extract as extract_claims
    import pymupdf4llm

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        try:
            md_text = pymupdf4llm.to_markdown(tmp_path, use_ocr=False)
        except Exception:
            return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    ref_section = _locate_reference_section(md_text)
    body_text = md_text if ref_section is None else md_text[: md_text.find(ref_section)]
    return extract_claims(body_text, refs)


def _verify_claims_stage(
    pdf_bytes: bytes,
    refs:            list[dict[str, Any]],
    lookup_results:  list[Optional[dict[str, Any]]],
    opts:            PipelineOptions,
) -> list["ClaimVerificationReport"]:
    """
    Extract inline citations from the citing PDF's body, filter to
    ref indices whose lookup landed on a 'match' (unless
    ``opts.verify_scope=="all"``), fetch each cited paper's full text
    (or abstract fallback), and batch-verify all sentences citing a
    given ref in ONE LLM call.

    Batching cuts LLM cost dramatically: a paper that cites the same
    ref 5 times used to be 5 calls; now it's 1 call with 5 sentences.
    Full-text fetches are already de-duped per ref.
    """
    from claim_verifier   import verify_batch
    from fulltext         import fetch  as fetch_fulltext

    claims = extract_claims_from_pdf(pdf_bytes, refs)
    if not claims:
        return []

    # Restrict to matched refs unless the caller explicitly opted into
    # "all".  Verifying fabricated/paywalled refs just yields no_source
    # verdicts that burn user time and (with commercial LLMs) money.
    if opts.verify_scope != "all":
        matched_indices = {
            i for i, lr in enumerate(lookup_results)
            if lr and _overall_status(lr) == "match"
        }
        claims = [c for c in claims if c.cited_ref_index in matched_indices]
        if not claims:
            return []

    # Per-ref escalation: when opts.verify_refs is set, only verify
    # claims that cite those specific refs.  Used by the UI's "Check
    # with full text" button — the fast-path run already produced
    # verdicts for every claim; this second run re-verifies a handful
    # of them against the full paper and the caller merges results.
    if opts.verify_refs is not None:
        wanted = set(opts.verify_refs)
        claims = [c for c in claims if c.cited_ref_index in wanted]
        if not claims:
            return []

    # Group claims by cited_ref_index so each ref's fetch happens once
    # and each ref's LLM call gets ALL its citing sentences.
    from collections import defaultdict
    claims_by_ref: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for pipeline_idx, c in enumerate(claims):
        claims_by_ref[c.cited_ref_index].append((pipeline_idx, c))

    # Full-text fetch: 6 workers (CPU-bound PDF parse benefits little
    # from higher parallelism due to the GIL).
    from concurrent.futures import ThreadPoolExecutor as _TPE
    fulltexts: dict[int, Any] = {}
    def _fetch_one(idx: int) -> tuple[int, Any]:
        lr = lookup_results[idx] if idx < len(lookup_results) else None
        ref = refs[idx] if idx < len(refs) else {}
        try:
            return idx, fetch_fulltext(lr or {}, ref=ref,
                                       depth=opts.verify_depth)
        except Exception:
            return idx, None
    with _TPE(max_workers=6, thread_name_prefix="ares-fetch") as pool:
        for fut in as_completed(pool.submit(_fetch_one, i) for i in claims_by_ref):
            idx, ft = fut.result()
            fulltexts[idx] = ft

    # LLM verify: one batched call per ref, parallelised over refs.
    llm_parallel = max(1, min(int(os.getenv("LLM_PARALLEL", "8")), 16))
    verdicts: list[Optional[ClaimVerificationReport]] = [None] * len(claims)

    def _verify_one_ref(ref_idx: int) -> list[tuple[int, ClaimVerificationReport]]:
        entries      = claims_by_ref[ref_idx]
        ft           = fulltexts.get(ref_idx)
        pipeline_ixs = [e[0] for e in entries]
        raw_claims   = [e[1] for e in entries]

        if ft is None:
            out = []
            for pi, c in zip(pipeline_ixs, raw_claims):
                out.append((pi, ClaimVerificationReport(
                    claim_index          = pi,
                    cited_ref_index      = c.cited_ref_index,
                    claim_text           = c.text,
                    verdict              = "no_source",
                    evidence_passage     = "",
                    confidence           = "low",
                    explanation          = "no open-access full text or abstract available",
                    source               = "",
                    is_abstract_fallback = False,
                    model                = "",
                    citation_marker      = c.citation_marker,
                    style                = c.style,
                )))
            return out

        raw_verdicts = verify_batch(raw_claims, ft, pipeline_ixs)
        out = []
        for c, raw in zip(raw_claims, raw_verdicts):
            out.append((raw.claim_index, ClaimVerificationReport(
                claim_index          = raw.claim_index,
                cited_ref_index      = raw.cited_ref_index,
                claim_text           = raw.claim_text,
                verdict              = raw.verdict,
                evidence_passage     = raw.evidence_passage,
                confidence           = raw.confidence,
                explanation          = raw.explanation,
                source               = raw.source,
                is_abstract_fallback = raw.is_abstract_fallback,
                model                = raw.model,
                citation_marker      = c.citation_marker,
                style                = c.style,
                error                = raw.error or "",
            )))
        return out

    with _TPE(max_workers=llm_parallel, thread_name_prefix="ares-verify") as pool:
        for fut in as_completed(pool.submit(_verify_one_ref, ri) for ri in claims_by_ref):
            for pi, v in fut.result():
                verdicts[pi] = v

    return [v for v in verdicts if v is not None]
