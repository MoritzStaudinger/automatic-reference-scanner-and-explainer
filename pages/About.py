"""
About / introduction page for the Automatic Reference Explainer and Scanner.

Streamlit auto-loads any file in `pages/` and lists it in the sidebar
navigation.  This page is a static explainer — it has no side effects
on the main workflow's session state.
"""
import streamlit as st

st.set_page_config(
    page_title="About — Automatic Reference Explainer and Scanner",
    layout="wide",
)

# Reuse the same typography polish from the main app
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter",
                 "Helvetica Neue", Arial, sans-serif;
}
h1, h2, h3, h4 {
    letter-spacing: -0.01em;
    color: #1f2933;
}
.stApp h1 { font-weight: 600; font-size: 1.95rem; }
.stApp h2 { font-weight: 600; font-size: 1.35rem; margin-top: 2rem; }
.stApp h3 { font-weight: 600; font-size: 1.05rem; margin-top: 1.2rem; }

.about-lead {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #343a40;
    max-width: 780px;
    margin-bottom: 1.5rem;
}

.about-card {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.93em;
    line-height: 1.55;
    color: #343a40;
}
.about-card.green  { border-left: 3px solid #2f9e44; background: #f4faf5; }
.about-card.red    { border-left: 3px solid #c92a2a; background: #fdf4f4; }
.about-card.yellow { border-left: 3px solid #f08c00; background: #fff9ef; }
.about-card.grey   { border-left: 3px solid #adb5bd; background: #fafbfc; }

.about-card h4 {
    margin: 0 0 6px 0;
    font-size: 1rem;
    font-weight: 600;
}
.about-card p { margin: 0; }

.kv-table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.92em;
}
.kv-table th, .kv-table td {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid #e9ecef;
    vertical-align: top;
}
.kv-table th {
    color: #6c757d;
    font-weight: 500;
    font-size: 0.82em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: #f6f7f9;
}
.kv-table tr:last-child td { border-bottom: none; }

.section-divider {
    border: none;
    border-top: 1px solid #e9ecef;
    margin: 2.5rem 0 1.5rem 0;
}

/* Hide Streamlit's default filename-derived nav (matches main app) */
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Custom sidebar nav, same as the main page
with st.sidebar:
    st.page_link("app.py",          label="Validate a PDF")
    st.page_link("pages/API.py",    label="API")
    st.page_link("pages/About.py",  label="About")
    st.divider()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("About")
st.markdown(
    '<p class="about-lead">'
    "The <strong>Automatic Reference Explainer and Scanner</strong> "
    "extracts the bibliography from an academic PDF, "
    "audits every reference against multiple authoritative databases, "
    "and surfaces the ones that don't add up — wrong-DOI citations, "
    "fabricated arXiv IDs, mis-attributed papers, and citations that "
    "can't be found in academic databases.  Every flagged reference "
    "is a candidate for human review, not a verdict."
    '</p>',
    unsafe_allow_html=True,
)

# Prominent warning at the top of the About page too
st.warning(
    "**Human verification is required for every result.** "
    "The tool reports what academic databases say — it does not decide "
    "whether a citation is correct.  See *Limitations and required "
    "human review* below for the full list of failure modes.",
    icon=None,
)


# ---------------------------------------------------------------------------
# Why
# ---------------------------------------------------------------------------
st.header("What problem does it solve?")
st.markdown(
    "Academic papers — especially those written with LLM assistance — "
    "increasingly contain references to papers that don't quite exist. "
    "Common patterns:"
)

st.markdown(
    """
- **Plausible-looking arXiv ID, wrong paper.** Author copies a citation that says `arXiv:2305.00471` but the ID actually resolves to a completely different paper.
- **DOI from a neighbouring reference.** A reference's title and authors are real, but the DOI was accidentally pasted from the citation next to it.
- **Title-author mismatch.** A real paper title is paired with authors who didn't write it (often Vaswani et al. or similar high-frequency author lists hallucinated in).
- **Title that exists nowhere.** A confident-sounding citation that no academic database has ever heard of.
- **Venue mismatch.** Title and authors check out, but the venue listed is wrong (e.g. arXiv preprint cited as a conference paper).

Reviewers can't catch most of these in a manual pass — there's no shortcut for opening a DOI and checking it resolves to what the bibliography says. This tool does that for every reference automatically.
""")


# ---------------------------------------------------------------------------
# Verdict colors
# ---------------------------------------------------------------------------
st.header("Three verdicts per reference")
st.markdown(
    "Every reference is classified as one of three states, shown next to "
    "the reference in the audit panel and as a colored highlight on the PDF:"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        '<div class="about-card green">'
        '<h4>🟢 Verified</h4>'
        '<p>At least one academic source returned a confident, high-similarity '
        'match. The reference is what it claims to be.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<div class="about-card yellow">'
        '<h4>🟡 Not found</h4>'
        '<p>No source confidently matched the citation. This is normal for '
        'books, theses, technical reports, blog posts, and very recent '
        'preprints — but warrants a manual check for conference and journal '
        'papers.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="about-card red">'
        '<h4>🔴 Wrong paper</h4>'
        '<p>A source confidently found a paper at this title or identifier, '
        'but the metadata doesn\'t match — e.g. cited DOI resolves to a '
        'different paper, or title matches but authors clearly disagree.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    "Each card in the audit also tells you **why** it's that color — the "
    "specific reason ('cited arXiv ID points to a different paper', "
    "'title matches but the cited authors don't', etc.) is shown directly "
    "in the reference summary so you don't have to expand the card to "
    "understand the verdict."
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.header("How it works")

st.markdown("Three stages run when you upload a PDF:")

st.markdown(
    """
### 1. Parse references from the PDF

Four parser backends are available:

- **hybrid** *(recommended)* — combines GROBID's structured TEI output with a separate section-pipeline that uses PyMuPDF to locate the References section and sends each candidate citation through GROBID's `/api/processCitationList` endpoint. Recovers references each method misses alone.
- **grobid** — GROBID's full-document layout parser only.
- **pymupdf4llm** — fast regex-based extraction; brittle on irregular bibliographies.
- **docling** — same regex extractor backed by docling's layout analysis.

After parsing, each reference becomes a structured record with `title`, `authors`, `year`, `venue`, `doi`, and `url` fields plus the original raw citation text.

### 2. LLM repair *(optional)*

Backed by either the Anthropic API (Claude) or a TU Wien Aqueduct endpoint (Qwen, Gemma, etc). The LLM:

- Repairs garbled fields (e.g. when the regex parser mis-split authors and title).
- Flags entries that aren't actually citations (body-text fragments mis-classified by the parser).
- Recovers identifiers from the raw text that the structured extractor missed.

The LLM is opt-in. Disabling it makes runs ~30% faster but characterizes failure modes less precisely.

### 3. Validate against academic databases

Each reference is routed to the most relevant sources based on what identifiers it has. The router:

- arXiv-cited refs → arXiv lookup only (it's authoritative for arXiv IDs)
- DOI-cited refs → doi.org first (canonical metadata via content negotiation), then Crossref / Semantic Scholar / OpenAlex as a vote
- CS-conference venues → DBLP
- Book-shaped venues → Open Library
- Title-only refs → parallel fanout across S2, OpenAlex, Crossref, DBLP

Identifier-routed lookups are authoritative — if the cited arXiv ID resolves to a paper with a clearly different title, that's a confident `wrong_arxiv_id` flag, and we don't go shopping at other databases looking for a confirmation.
"""
)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
st.header("Sources consulted")

st.markdown(
    """
<table class="kv-table">
  <thead><tr><th>Source</th><th>Coverage</th><th>Role</th></tr></thead>
  <tbody>
    <tr><td><b>doi.org</b></td><td>Any registered DOI (Crossref / DataCite / MEDRA / JaLC)</td><td>Primary DOI resolver via content negotiation. One canonical metadata record per DOI.</td></tr>
    <tr><td><b>Semantic Scholar</b></td><td>Broad academic index</td><td>Per-DOI and per-arXiv lookup, plus title search. Batched up to 500 IDs per request.</td></tr>
    <tr><td><b>Crossref</b></td><td>DOI registry</td><td>Vote for DOI-cited refs. Title search fallback.</td></tr>
    <tr><td><b>OpenAlex</b></td><td>Broad open-access index</td><td>Vote for DOI-cited refs. Batched up to 50 DOIs per request.</td></tr>
    <tr><td><b>arXiv</b></td><td>Preprint server</td><td>Authoritative for arXiv-ID lookups (resolves the cited ID directly to its abstract record).</td></tr>
    <tr><td><b>DBLP</b></td><td>Computer-science conferences and journals</td><td>Routed to when the venue smells like CS-conference (EMNLP, ACL, NeurIPS, etc.).</td></tr>
    <tr><td><b>ACL Anthology</b></td><td>NLP-conference papers</td><td>Derived from S2's externalIds and ACL DOIs — no separate API call.</td></tr>
    <tr><td><b>Open Library</b></td><td>Books and editions, with ISBN matching</td><td>Routed to when the venue text contains a book-shape signal.</td></tr>
    <tr><td><b>Google Scholar</b> <i>(opt-in)</i></td><td>Broadest (blog posts, theses, books, non-Western venues)</td><td>Last-resort title verifier. Slow (2.5 s per request, CAPTCHA-prone). Disabled by default.</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True
)

st.markdown(
    "All HTTP calls share a connection-pooled session with bounded retries. "
    "Results are cached indefinitely in a local SQLite database — so the "
    "second time you process a paper, almost everything is a cache hit and "
    "the run takes seconds rather than minutes."
)


# ---------------------------------------------------------------------------
# Wrong-id detection
# ---------------------------------------------------------------------------
st.header("Wrong-identifier detection")

st.markdown(
    """
The strongest precision signal in the pipeline is **identifier mismatch**: when a reference includes a DOI or arXiv ID that resolves to a paper with a clearly different title than what's cited.

For example, a citation says:

> Chen, J., Li, R. & Wang, Q. (2023). *Evaluating the logical consistency of GPT models.* arXiv preprint arXiv:2305.00471.

When we resolve `arXiv:2305.00471` directly, we get back a paper titled *"Classification, α-Inner Derivations and α-Centroids of Finite-Dimensional Complex Hom-Trialgebras"* — a completely unrelated math paper. The arXiv ID was plausible-looking but fabricated.

This pattern is caught explicitly: when an identifier resolves to a title at < 50% similarity to the cited title, the reference is flagged with an inline warning explaining exactly what's wrong. The same logic applies to DOIs that resolve to neighbouring-reference papers (a classic copy-paste error).

To avoid blaming the user for identifiers the LLM repair pass might have injected itself, the wrong-identifier check looks only at the raw citation text and the parsed venue field — never at LLM-edited URL fields.
""")


# ---------------------------------------------------------------------------
# Workflow walkthrough
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.header("Workflow")

st.markdown(
    """
1. **Upload a PDF** on the main page. Parsing, LLM repair (if enabled), and PDF highlighting run automatically.
2. **Click "Look up all references"** to fire the validation pipeline. Progress is shown live; a persistent cache means re-runs of the same paper are nearly instant.
3. **Browse the audit cards on the left.** Each card has a colored verdict, a one-line reason, and (when expanded) per-source results showing exactly what each database returned. Sources that returned no useful result are hidden — only sources that contributed to the verdict are shown.
4. **Filter the list** by verdict (Verified / Wrong / Not found / Pending) or by PDF page using the toggles at the top of the audit panel.
5. **Click a highlighted ref in the PDF** to jump to its audit card on the left. Click "Go to p. N" on an audit card to scroll the PDF to where that reference lives.
6. **Download the markdown report** when you're done — it groups every reference by verdict, includes the per-source breakdown, and is ready to send to a co-author.
"""
)


# ---------------------------------------------------------------------------
# Manual editing
# ---------------------------------------------------------------------------
st.header("Adding missed references")

st.markdown(
    """
Sometimes the parser misses a citation entirely (especially for unusual layouts or merged-block bibliographies). Two ways to add one manually:

**From the PDF.** Drag-select the citation text in the PDF viewer, then click the **"Add selected text as reference"** button in the viewer's toolbar. The selection is sent to GROBID for structuring, then appears as a new reference in the audit panel with a pending verdict.

**From the sidebar.** Open the **"Add a missed reference"** expander, paste the citation text, and click **"Add to references"**. Same flow — GROBID structures it, the reference enters the audit panel as pending.

Either way, the new reference shows up immediately without re-running the lookup pipeline. Click **"Look up N new ref(s)"** when you're ready to validate the newly-added entries. Refs that were already looked up keep their cached results.

You can also remove manually-added references from the sidebar list (click the **×** next to any entry).
""")

# Possible missed references panel
st.markdown(
    """
The system also surfaces **possible missed references** — text blocks on the same pages as your parsed references that didn't get claimed by any extracted reference and look citation-shaped (have a year, ≥40 chars). They appear in a panel below the PDF after parsing. Click **+** next to one to add it to the reference list.
"""
)


# ---------------------------------------------------------------------------
# Limitations and required human review
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.header("Limitations and required human review")

st.markdown(
    """
This tool reports what academic databases say.  It does not decide whether a
citation is correct.  Every result — green, yellow, or red — is a
**starting point for human verification**, not a final verdict.  The
specific failure modes you need to be aware of:
"""
)

st.markdown(
    """
- **"Verified" (green) can still be wrong.**  A reference is marked green when at least one source returns a high-similarity match.  But database matching is imperfect — the database can have a typo, the reference can list authors who didn't actually write the paper, or the title can be so generic that a near-match isn't the right paper.  Trust green only after spot-checking that the linked database entry really corresponds to the bibliographic record.
- **"Not found" (yellow) does not mean fabricated.**  Many real references aren't in academic databases at all — books, theses, technical reports, lab blog posts, very recent preprints, conference workshop papers, non-English venues, and grey literature.  When a reference is marked yellow, the only thing you've learned is that the system couldn't find it where it looked.  Open the original source and confirm it exists.
- **"Wrong paper" (red) is the most reliable signal — but still requires checking.**  Red is fired when a cited DOI or arXiv ID resolves to a paper with a clearly different title, or when a paper with the cited title has clearly different authors.  These are strong fabrication-or-error tells.  But: the system can mis-attribute a red flag (e.g. the LLM-extracted authors might be wrong, making a real citation look mis-attributed).  Open the cited identifier in your browser and compare against the bibliography entry before concluding the citation is bad.
- **LLM repair occasionally invents authors.**  The optional LLM repair pass cleans messy parsed fields, but it can also confidently hallucinate plausible-looking authors for incomplete citations.  We mitigate by only attributing identifier claims to text the user actually wrote, but author-list-level hallucinations can still leak through.  If a red flag is "title matches but cited authors don't", verify the authors in the original PDF — the system's view of who the authors are might itself be wrong.
- **Identifier extraction is best-effort.**  GROBID and the parser don't always recover the DOI or arXiv ID from the raw text — meaning the lookup falls back to title search, which is noisier.  When the tool reports "not found" for a reference that obviously has a DOI in the PDF, it's worth manually pasting that DOI into doi.org to confirm.
- **Database coverage is uneven.**  We query Semantic Scholar, OpenAlex, Crossref, arXiv, DBLP, the ACL Anthology, Open Library, and (optionally) Google Scholar and CORE.  Coverage is good for mainstream CS / NLP / ML papers and shrinks for other fields, older work, non-Western venues, and non-paper outputs (datasets, software).  A red or yellow flag on a citation from outside the well-covered area is much more likely to be a false alarm.
- **Manually-added references inherit the same limitations.**  Adding a reference via PDF text selection runs the same lookup pipeline against the same databases.  All of the caveats above apply equally.

In short: **the tool surfaces candidates for review**.  It is not a fact-checker.  It is not a citation auditor of record.  A reference passing through this tool clean does not absolve the author of the responsibility to have actually read what they cite, and a flagged reference is an invitation to look at the source — not evidence to act on by itself.
"""
)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.header("Operational notes")

st.markdown(
    """
- **First-time runs are slow; repeat runs are fast.** The persistent cache means that once a reference has been looked up, every future paper that cites it reads from disk.  A 50-reference paper takes ~1 minute cold, ~5 seconds warm.
- **GROBID needs to be running.** The `hybrid` and `grobid` parsers depend on a GROBID service at `localhost:8070`.  Start it with `docker compose up -d grobid`.
- **LLM repair is opt-in.**  Useful for messy PDFs where the regex extractor stuffed years or author fragments into the title field.  For clean publications it makes little difference and costs ~25 seconds per paper.
- **Google Scholar is disabled by default.**  Its broader coverage is valuable but it has no API and aggressively CAPTCHA-blocks scrapers.  Enable in the sidebar if you're processing references that lean toward grey literature.
- **Status colors on the PDF match the audit cards.**  The colored rectangles use the same green / yellow / red scheme as the audit panel.
"""
)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.caption("Built at TU Wien.")
