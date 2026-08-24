# Automatic Reference Explainer and Scanner

A pipeline + Streamlit UI for extracting references from academic PDFs and
auditing them against multiple authoritative databases. The tool surfaces
hallucinated citations — wrong paper at the cited DOI, fabricated arXiv
IDs, mis-attributed authors, titles that exist nowhere — before they
reach reviewers.

> **Human verification is required for every result.** The tool reports
> what academic databases say. It does not decide whether a citation is
> correct. Treat its verdicts as candidates for review, not as ground
> truth.

---

## What the tool does

Upload a PDF → parse the bibliography → optionally repair structured
fields with an LLM → look up each reference across academic sources in
parallel (Semantic Scholar, Crossref, OpenAlex, arXiv, DBLP, ACL
Anthology, doi.org, Open Library) → classify each result and explain why.

The UI shows:

- A side-by-side PDF view with each reference highlighted by status
  (**green** = match, **yellow** = couldn't find / suspicious, **red** =
  wrong paper at this title).
- Per-reference audit cards with the cited metadata, the best match
  found, and which sources agreed or disagreed.
- A Markdown report you can export for a paper author or reviewer.

A SQLite cache makes any second pass on the same paper near-instant.

## Quick start (Docker)

You need Docker, Docker Compose, and ~10 GB of free RAM (GROBID alone
asks for 6–8 GB).

```bash
git clone https://github.com/<your-org>/reference-explainer.git
cd reference-explainer
cp .env.example .env        # fill in any API keys you have
docker compose up -d --build
```

The Streamlit UI lives behind nginx at port 443 in the shipped compose.
For local development without TLS, comment out the `nginx` service and
add `ports: ["8501:8501"]` to the `ares` service, then visit
`http://localhost:8501`.

Production deployment with an institutional certificate is documented in
the comments at the top of `docker-compose.yml`.

## Quick start (local Python)

If you'd rather run it on your host without Docker:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in API keys
# Start a GROBID instance separately (the `grobid` parser needs one):
docker run -d --rm --name grobid -p 8070:8070 lfoppiano/grobid:0.8.1
streamlit run app.py
```

## Configuration

All configuration is via environment variables (typically loaded from
`.env`). See `.env.example` for the complete list with explanations. The
short version:

| Variable | Purpose |
|---|---|
| `S2_API_KEY` | Semantic Scholar — higher rate limits |
| `CORE_API_KEY` | CORE rescue lookup |
| `OPENALEX_MAILTO` | Polite-pool identifier for OpenAlex |
| `ANTHROPIC_API_KEY` | LLM repair via Anthropic |
| `AQUEDUCT_API_KEY` | LLM repair via TU Wien DataLab |
| `LLM_BACKEND` | `anthropic` or `aqueduct` |
| `LLM_MODEL` | Model name for the chosen backend |
| `GROBID_URL` | GROBID HTTP endpoint (set automatically in Docker) |
| `ARES_CACHE` | SQLite cache path |

All keys are optional — the tool degrades gracefully when a source is
unavailable. More keys → fewer rate-limit walls.

## Parser backends

Several parser backends ship in the box. The right one depends on the
trade-off you want between speed and recall.

| Backend | Notes |
|---|---|
| `hybrid` | **Recommended.** Merges GROBID's full-document parse with `pymupdf4llm`'s section-located refs structured by GROBID's `/api/processCitationList`. Catches references each method misses alone. Requires GROBID running. |
| `grobid` | GROBID full-document layout parse only. Fast, clean; may miss refs in oddly-formatted bibliographies. Requires GROBID running. |
| `pymupdf4llm` | Fastest (~1 s/paper). Regex-based field extraction; brittle on complex bibliographies. No external dependencies. |
| `docling` *(optional)* | Slow (minutes), same regex extractor as `pymupdf4llm` but with docling's layout analysis. Requires `pip install "docling>=2.15.0"` — not included by default because it pulls in PyTorch (~5 GB image bloat). The UI auto-hides this option when docling isn't installed. |

`hybrid` with LLM repair gives the best precision/recall trade-off in our
evaluation. See *Limitations* below for the caveats.

## Architecture

```
   PDF
    │
    ├── parse references ──── pymupdf4llm  ──┐
    │                         grobid         │── parsers.py
    │                         hybrid         │
    │                         docling       ─┘
    │
    ├── (optional) LLM repair ─────────────── llm_parser.py
    │
    ├── identifier routing ──────────────┐
    │     • DOI → doi.org content-neg    │
    │     • arXiv ID → arXiv API         ├── lookup.py
    │     • Title only → fanout to all    │
    │                                    │
    ├── parallel source queries           │
    │     S2, Crossref, OpenAlex,         │
    │     DBLP, ACL Anthology, …          │
    │
    ├── per-source verdict ──── _composite_label
    │     (title + author + venue similarity)
    │
    ├── overall classification ── _overall_status / _categorise_suspicious
    │     match / missing / mismatch  → report.py
    │
    └── PDF annotation ──── highlight_references (highlighter.py)
                            colored rectangles per status
```

The SQLite cache (`cache.py`) memoizes every external request keyed by
the canonicalized reference. Re-runs on the same paper hit the cache and
return in seconds.

## Limitations and required human review

The tool catches a meaningful fraction of fabricated citations but is
**not a replacement for human review**. In our 13-paper manually-validated
sample (460 references), the pipeline reached ~94% accuracy with ~83%
recall on true fabrications, but precision was ~57% — meaning about 4 of
every 10 flags were false alarms, often legitimate non-academic references
like Github repos, tools, and tech reports.

Failure modes you should expect:

1. **Title-perfect, author-wrong fabrications.** A real paper title
   paired with hallucinated authors will sometimes pass — the title
   similarity threshold (≥0.90) accepts a match even when the cited
   authors don't agree with the found paper. The author check defaults
   to "match" when either side lacks parsed authors.
2. **Real but non-academic references.** Github repos, blog posts,
   datasets, and tech reports will often be flagged "missing" because
   they don't live in academic indexes. That's a false alarm, not a
   fabrication.
3. **Recently-published papers.** Source indexes lag a few weeks behind
   publication. Genuine new papers may be flagged "missing" until the
   indexes catch up.
4. **OCR'd or scanned PDFs.** Reference extraction degrades when the PDF
   layer is OCR text rather than digital text.
5. **Non-English bibliographies.** Title/author similarity uses Latin
   normalization. Non-Latin scripts will produce lower confidence scores.

Treat this tool as a **first-pass filter that surfaces candidates for
human review**, not as an oracle.

## Repository layout

```
app.py                  Streamlit UI
parsers.py              PDF → reference extraction (4 backends)
llm_parser.py           Optional LLM-based reference repair
lookup.py               Parallel multi-source verification
cache.py                SQLite memoization layer
highlighter.py          PyMuPDF-based PDF annotation
report.py               Markdown audit report generation
pdf_selector/           Custom Streamlit component (PDF.js-based viewer
                        with native text selection for adding missed refs)
pages/About.py          About / methodology page (Streamlit auto-loads)
nginx/                  Reverse-proxy config for the deployed instance
benchmark/              Evaluation harness against HalluCitation
                        (kept lightweight — the 295-PDF corpus and
                        per-run intermediates are gitignored)
```

## Citing the tool

If you use the Automatic Reference Explainer and Scanner in academic
work, please cite it as:

```bibtex
@software{referenceexplainer2026,
  author = {Staudinger, Moritz},
  title  = {Automatic Reference Explainer and Scanner},
  year   = {2026},
  url    = {https://github.com/<your-org>/reference-explainer}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built at TU Wien. Uses GROBID, Semantic Scholar, Crossref, OpenAlex,
arXiv, DBLP, ACL Anthology, doi.org, Open Library, and Open CORE — none
of which would be possible without the work of the people who maintain
those services.
