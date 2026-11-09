"""
Overall verdict for one reference — the single source of truth shared by
the Streamlit UI (app.py), the Markdown report (report.py), the headless
pipeline / API (pipeline.py) and the PDF highlighter.

Statuses
--------
match     (green)  – at least one source returned a confirmed match
mismatch  (red)    – a source found a paper at this title/identifier but
                     the details clearly disagree (wrong DOI / arXiv ID /
                     authors / venue) — the citation errors the user
                     wants surfaced
missing   (amber)  – not found anywhere, or only weak/fuzzy results
pending   (amber)  – lookup has not run yet for this reference

Per-source `label` values ("match" / "fuzzy" / "mismatch") are decided in
lookup._finalize_match; mismatch is only set on positive evidence of a
wrong paper, never from a low-similarity closest match, so lifting it to
the overall verdict here is safe.
"""

from __future__ import annotations

from typing import Any

VALIDATION_SOURCES: tuple[str, ...] = (
    "doi_org", "semantic_scholar", "acl_anthology", "dblp", "openalex",
    "crossref", "openlibrary", "scholarly", "arxiv", "url_verify",
)

SOURCE_NAMES: dict[str, str] = {
    "doi_org":          "doi.org",
    "semantic_scholar": "Semantic Scholar",
    "crossref":         "Crossref",
    "openalex":         "OpenAlex",
    "acl_anthology":    "ACL Anthology",
    "dblp":             "DBLP",
    "arxiv":            "arXiv",
    "openlibrary":      "Open Library",
    "scholarly":        "Google Scholar",
    "url_verify":       "Cited URL",
}

STATUS_ICON  = {"match": "🟢", "mismatch": "🔴", "missing": "🟡", "pending": "⚪"}
STATUS_LABEL = {"match": "Verified", "mismatch": "Mismatch",
                "missing": "Not found", "pending": "Pending"}

# One palette for the PDF overlay (rgba, drawn over the page canvas) and
# the baked export highlights (rgb floats for PyMuPDF).  Pending shares the
# missing colour: until a verdict arrives they mean the same to a reader.
STATUS_RGB: dict[str, tuple[float, float, float]] = {
    "match":    (0.18, 0.78, 0.35),
    "mismatch": (0.93, 0.27, 0.27),
    "missing":  (1.00, 0.85, 0.20),
    "pending":  (1.00, 0.85, 0.20),
}
STATUS_ALPHA = {"match": 0.40, "mismatch": 0.40, "missing": 0.45, "pending": 0.45}


def status_rgba(status: str) -> str:
    r, g, b = STATUS_RGB.get(status, STATUS_RGB["pending"])
    a = STATUS_ALPHA.get(status, STATUS_ALPHA["pending"])
    return f"rgba({round(r * 255)}, {round(g * 255)}, {round(b * 255)}, {a})"


def found_results(lr: dict | None) -> list[tuple[str, dict[str, Any]]]:
    """(source, result) pairs for every validation source that found a paper."""
    if not lr:
        return []
    return [(src, lr[src]) for src in VALIDATION_SOURCES
            if isinstance(lr.get(src), dict) and lr[src].get("status") == "found"]


def overall_status(lr: dict | None) -> str:
    """'match' | 'mismatch' | 'missing' | 'pending'"""
    if lr is None:
        return "pending"
    labels = [r.get("label") for _, r in found_results(lr)]
    if "match" in labels:
        return "match"
    if "mismatch" in labels:
        return "mismatch"
    return "missing"


def status_reason(lr: dict | None) -> str:
    """One-line plain-English reason for the overall status, shown next to
    the ref summary so the user knows *why* without expanding the card."""
    if lr is None:
        return ""
    if lr.get("_error"):
        return "lookup failed for this reference — try re-checking"

    found = found_results(lr)
    if not found:
        url = lr.get("url_verify") or {}
        if url.get("status") == "dead_link":
            return "not in any database, and the cited URL is unreachable"
        return "not found in any database"

    matches = [src for src, r in found if r.get("label") == "match"]
    if matches:
        names = [SOURCE_NAMES.get(s, s) for s in matches[:2]]
        return f"confirmed by {', '.join(names)}"

    mismatches = [r for _, r in found if r.get("label") == "mismatch"]
    if mismatches:
        # Most specific reason first, across all mismatching sources.
        for flag, reason in (
            ("wrong_arxiv_id", "cited arXiv ID points to a different paper"),
            ("wrong_doi",      "cited DOI points to a different paper"),
            ("wrong_authors",  "title matches but the cited authors don't"),
            ("wrong_venue",    "title matches but the cited venue doesn't"),
        ):
            if any(r.get(flag) for r in mismatches):
                return reason
        return "found a paper at this title but details disagree"

    return "no confident match found"


# ---------------------------------------------------------------------------
# Citation-context (claim check) wording
# ---------------------------------------------------------------------------
# The verdicts are pointers for a human reader, so the labels describe what
# the comparison with the source found rather than passing judgement on the
# citing authors ("possible conflict", not "contradicted").  The internal
# values (keys) are unchanged — the API returns them as before.
CLAIM_LABEL = {
    "supported":           "matches source",
    "partially_supported": "partly matches",
    "contradicted":        "possible conflict",
    "ambiguous":           "unclear",
    "not_addressed":       "not found in source",
    "evidence_only":       "passages only",
    "no_source":           "no text available",
    "error":               "check failed",
}


def claim_label(verdict: str) -> str:
    return CLAIM_LABEL.get(verdict, verdict.replace("_", " "))
