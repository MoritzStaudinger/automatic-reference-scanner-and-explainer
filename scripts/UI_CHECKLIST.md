# ARES manual UI checklist

`pytest tests/` and `python scripts/smoke_test.py` cover the pipeline, API and
page rendering. These checks need a browser: run them after changing anything
in `ui/`, `pdf_selector/` or the highlighter.

    GROBID_URL=http://localhost:8070 streamlit run app.py
    # or the full stack: docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

Use any paper from `benchmark/pdfs/`. One you haven't processed before exercises
the uncached path.

## 1. Upload and automatic check
- [ ] Top navigation shows **Check a PDF · API · About**. There is no sidebar.
- [ ] The landing page shows three verdict explanations and the human-review notice.
- [ ] After uploading, the "Reading the paper …" status lists the stages: parse,
      LLM repair, dedupe, locate, auto-detected references. It then collapses.
- [ ] Verification starts **automatically**. A progress bar shows "n / N".
- [ ] While it runs, a filter click, **Show on p. N** or opening a card does
      **not** cancel it. The tiles fill in when it finishes.
- [ ] When it finishes, the PDF **keeps its scroll position** and the highlights
      recolour in place.

## 2. Results (no page reruns)
- [ ] Status filter, search, page filter, opening cards, **Show in PDF** and
      clicking a highlight do **not** rerun the page (no "Running…" indicator,
      the PDF keeps its position). They are handled inside the component.
- [ ] Every highlight covers the whole reference, including references that
      continue in the next column or on the next page.
- [ ] Tiles show References / Verified / Mismatch / Not found / Dead links.
- [ ] Status pills and search narrow the list ("n of N references"). The
      Pending pill is hidden when nothing is pending.
- [ ] A card opens to show the raw citation, fields, sources (with links that
      open in a new tab), warnings (wrong DOI / arXiv ID, authors, venue) and
      the reason line.
- [ ] **Show on p. N** scrolls only the PDF, not the whole page.
- [ ] Clicking a highlight in the PDF shows only that card, expanded.
      **Show all** returns to the full list.
- [ ] "Only references on the visible PDF page" follows PDF scrolling.

## 3. Adding and removing references
- [ ] In **Add reference → Paste a citation**, paste a citation and press Add.
      A toast appears, the count increases, and **Check 1 new** appears.
- [ ] **Check 1 new** looks up only that reference. It gets an "added by you" badge.
- [ ] **Suggestions** tab and "Possibly missed references" under the PDF:
      **Add** turns a block into a reference.
- [ ] Selecting text in the PDF and pressing **Add selected text as reference**
      adds it with its highlight.
- [ ] **Remove** on a manual card, or in the dialog's "Added by you" tab, removes
      it without shifting other cards' results.
- [ ] **Re-check** on a flagged card re-queries the sources, bypassing the cache.

## 4. Citation context (LLM configured)
- [ ] **Check citation context → Quick** runs in the background and adds a
      "Claims contradicted" tile and the summary line.
- [ ] Cards show the citing sentences with their verdicts, without markdown
      asterisks.
- [ ] **Full text (n)** on a card, and "Re-check n inconclusive with full texts",
      re-run only those references.

## 5. Export
- [ ] The Export menu offers Report (Markdown), Annotated PDF, Table (CSV) and
      Full results (JSON). Each one downloads.
- [ ] The report includes the citation-context section once claims were checked.

## 6. Robustness
- [ ] Stop GROBID (`docker compose stop grobid`) and upload: a warning says the
      simple parser was used, and **Options** offers only pymupdf4llm.
- [ ] Upload a second, different PDF: no cards, results or claims carry over.
- [ ] Switch Light / Dark in the ⋮ menu: cards, chips and the PDF viewer
      toolbar follow the theme.
- [ ] At phone width (< 640 px) the columns stack.
- [ ] The browser console shows no errors. pdf.js loads from
      `…/vendor/pdf.min.js`, not a CDN.
