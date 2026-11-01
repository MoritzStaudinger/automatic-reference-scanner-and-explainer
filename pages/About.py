"""About: what ARES checks, how, and where it falls short.  Static page."""

import streamlit as st

st.title("About ARES")
st.markdown(
    "The **Automatic Reference Explainer and Scanner** extracts the bibliography "
    "from an academic PDF, checks every reference against scholarly indexes and the "
    "web pages it cites, and points out the ones that don't add up: DOIs and arXiv IDs "
    "that lead to a different paper, real titles with the wrong authors, dead links, "
    "and citations no index has ever seen."
)
st.info("**Every result needs human review.** ARES reports what databases and web pages "
        "say; it does not decide whether a citation is correct.",
        icon=":material/person_search:")

st.header("What each verdict means")
c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.markdown("#### :green[:material/check_circle:] Verified")
    st.markdown("At least one source confirmed the title (and, where available, the "
                "authors) with high similarity.")
with c2.container(border=True):
    st.markdown("#### :red[:material/cancel:] Mismatch")
    st.markdown("A source found a paper at this title, DOI or arXiv ID, but the details "
                "clearly disagree — e.g. the DOI belongs to another paper, or the authors "
                "don't match. The strongest signal of a wrong or invented citation.")
with c3.container(border=True):
    st.markdown("#### :orange[:material/help:] Not found")
    st.markdown("No source matched confidently. Normal for books, theses, software, "
                "datasets, blogs and very recent preprints — worth a manual check for "
                "journal and conference papers.")
st.markdown("Each card also says *why* — for example *cited arXiv ID points to a "
            "different paper* — and cited URLs that no longer resolve get a "
            "**dead link** badge.")

st.header("How it works")
st.markdown("""
1. **Extract.** The `hybrid` parser merges GROBID's full-document parse with the
   bibliography section structured citation by citation, then looks for reference-like
   text no extracted entry covers. `grobid`, `pymupdf4llm` and (if installed) `docling`
   are available under **Options**.
2. **Repair (optional, on by default when configured).** An LLM fixes swapped or truncated
   fields and drops body text that was mistaken for a reference. Identifier checks only
   trust identifiers present in the original citation text, never ones the LLM added.
3. **Verify.** Each reference is routed by what it carries: DOIs go to doi.org first
   (with Crossref, Semantic Scholar and OpenAlex as a vote), arXiv IDs to arXiv, CS venues
   to DBLP, books to Open Library, and title-only references fan out to several indexes.
   Cited web pages are fetched and compared too: GitHub repositories through the GitHub
   API, dead links through the Internet Archive.
4. **Review.** Results appear as cards and as coloured highlights in the PDF. Click a
   highlight to open its card; use **Show on p. N** to jump the other way.
5. **Citation context (optional).** For verified references, an LLM checks whether the
   cited paper (its abstract, or its open-access full text) supports the sentences that
   cite it.
""")

st.header("How matching is scored")
from ui.explain import MATCHING_MD  # noqa: E402
st.markdown(MATCHING_MD)

st.header("Sources")
st.table({
    "Source": ["doi.org", "Semantic Scholar", "Crossref", "OpenAlex", "arXiv", "DBLP",
               "ACL Anthology", "Open Library", "Cited web pages", "Google Scholar (opt-in)"],
    "Used for": [
        "Canonical metadata for any registered DOI",
        "Broad index; DOI, arXiv and title lookups (batched)",
        "DOI registry; title search fallback",
        "Broad open index; DOI lookups (batched), abstracts",
        "Authoritative for arXiv IDs",
        "Computer-science venues",
        "NLP papers, derived from Semantic Scholar / ACL DOIs",
        "Books",
        "Project pages, repositories, reports — page metadata, GitHub API, Wayback Machine",
        "Last resort for grey literature; slow and rate-limited",
    ],
})
st.caption("Results are cached, so a second run on the same paper takes seconds. "
           "\"Not found\" results are re-checked after two weeks, since indexes catch up.")

st.header("Limitations")
st.markdown("""
- **Verified can still be wrong.** A generic title can match the wrong paper, and author
  checks are skipped when either side has no usable author list.
- **Not found is not proof of fabrication.** Coverage is best for computer science and
  weaker for other fields, older work, non-English venues and non-paper outputs.
- **Mismatch needs a look, too.** If the extracted authors are wrong (parser or LLM error),
  a real citation can look mis-attributed — compare against the PDF.
- **Scanned PDFs** without a text layer extract poorly.
- **Citation context** verdicts from abstracts are partial by nature; *not addressed*
  often just means the abstract is silent.

ARES surfaces candidates for review. It is not a citation auditor of record, and a clean
result does not replace reading what you cite.
""")

st.caption("Built at TU Wien.")
