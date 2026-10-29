# ARES manual UI checklist

The headless smoke test (`python scripts/smoke_test.py`) covers the
pipeline, API, and CLI.  These checks need the browser — go through them
when you've changed anything that touches Streamlit, the highlighter,
or the PDF.js selector component.

Run the app:

    streamlit run app.py

Open it in a browser, upload `benchmark/pdfs/2024.acl-long.560.pdf` (any
paper in `benchmark/pdfs/` will do).  Walk through each section.

---

## 1. Upload + initial render

- [ ] Upload a PDF; the status block shows three stages (parse, locate, ready).
- [ ] After "Ready", the PDF appears on the right with coloured rectangles.
- [ ] Refs that were located in the PDF are highlighted yellow (no lookup
      run yet — yellow = pending).
- [ ] Re-uploading a different file resets `manual_refs` and `lookup_results`
      cleanly (no stale data carries over).

## 2. Lookup run

- [ ] Click "Look up references".  A progress bar appears, fills, and the
      ref cards turn green / yellow / red as their results arrive.
- [ ] Status icon on the PDF overlay updates to match (rectangles change colour).
- [ ] At least one ref shows the new author / venue evidence rows when
      the source disagrees — look for the "Authors: ..." or "Venue: ..."
      lines below the main "found_title" row.

## 3. Sidebar "Add a missed reference"

This is bug-prone because the sidebar runs before the main pipeline.

- [ ] Open the "Add a missed reference" expander in the sidebar.
- [ ] Paste a citation string (e.g.
      `John Hewitt and Christopher D. Manning. 2019. A structural probe
       for finding syntax in word representations. NAACL.`).
- [ ] Click **Add to references**.  Expect a green toast ("Added: …"),
      the input clears (after rerun), and the manual-ref count increments.
- [ ] The new ref appears at the BOTTOM of the audit list with status =
      "pending" / yellow.
- [ ] Click "Look up references" again — only the pending slot is
      re-queried (parsed refs return from cache instantly).  The new
      ref's status changes from yellow to its actual verdict.
- [ ] **(Bug-watch)** The colour of the manually-added ref's rectangle
      on the PDF should change with the verdict (was stuck on yellow
      before the fix in feature/api-mode-and-author-matching).
- [ ] Click × next to the manual ref to remove it; lookup_results
      shrinks correctly (no index drift on the remaining refs).

## 4. In-PDF text selection → add ref

- [ ] In the PDF panel, select a citation block with the mouse.
- [ ] A floating "+ Add" button appears next to the selection.
- [ ] Click it; the ref is added as a manual ref, with its rect coming
      from the JS selection (no fuzzy fallback needed → fast).
- [ ] The status rectangle on the PDF appears immediately (yellow).
- [ ] Re-run lookup; same colour-update behaviour as #3.

## 5. Orphan blocks

- [ ] Below the PDF, "Possible missed references" lists text blocks the
      parser missed.
- [ ] Click + next to one; it's added as a manual ref.
- [ ] The orphan row disappears (matched against `manual_refs`'s raw text).

## 6. Page-filter toggle

- [ ] "Filter references to current PDF page" toggle is OFF by default.
- [ ] Scrolling the PDF with the filter OFF should NOT trigger Streamlit
      reruns (smooth scrolling — no UI flicker, no app reload).
- [ ] Turning it ON starts emitting page changes; the ref list filters
      to whichever page is in view.

## 7. Quality Report

- [ ] After lookup completes, "Quality Report" expander shows a Markdown
      report.
- [ ] The new `wrong_authors` / `wrong_venue` categories appear in the
      breakdown when the test paper hits them (or the report explains
      cleanly that nothing was flagged).
- [ ] "Download report (.md)" button works.

## 8. Soft failure: GROBID down

- [ ] Stop the GROBID container (`docker compose stop grobid`).
- [ ] Try a parser that needs GROBID (`grobid` or `hybrid`).
- [ ] The status block surfaces the error rather than spinning forever.
- [ ] The sidebar "Add to references" button surfaces a clear error
      (GROBID is what structures the pasted citation).

## 9. Browser console / network

Open browser devtools.

- [ ] No red errors in the console after initial render.
- [ ] Re-runs go through Streamlit's `/_stcore/stream` WebSocket (not
      full page reloads).
- [ ] The custom PDF selector component (`pdf_selector`) doesn't refetch
      the PDF on every rerun — the iframe `src` shouldn't change unless
      the file id changes.

---

## When something is off

- **Highlight colours wrong** → `_STATUS_TO_COLOR` + the
  `component_annots` build loop in `app.py`.  Manual refs get their
  status from `lookup_results` (live) rather than the baked
  `er["status"]` (frozen).
- **Sidebar add does nothing** → `_append_manual_ref` must be defined
  ABOVE the `with st.sidebar:` block.  If you reorganise the file and
  push it back down, the button will silently fail with `NameError`.
- **Manual ref index drift after remove** → the `manual_offset` formula
  in the × button handler must stay in sync with how the ref list is
  assembled (`parsed_refs + manual_refs`, manual always at the tail).
