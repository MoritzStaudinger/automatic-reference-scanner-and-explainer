"""
"Check a PDF" — upload a paper, extract its bibliography, verify every
reference against academic indexes and cited web pages, and (optionally)
check that the cited papers support the sentences citing them.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import streamlit as st

from lookup import forget_reference, lookup_all
from ui import cards, explain, jobs, state, viewer
from ui.processing import (
    LLM_BACKEND, LLM_MODEL, LLM_READY, PARSER_HELP, Document, annotated_pdf,
    parser_options, process, with_manual,
)
from verdict import SOURCE_NAMES, STATUS_LABEL, overall_status, status_reason

st.html(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>")


# ---------------------------------------------------------------------------
# Header + upload
# ---------------------------------------------------------------------------

st.html("""
<div class="ares-hero">
  <h1>Check the references in a paper</h1>
  <p>Upload a PDF. ARES extracts its bibliography, looks every reference up in
  Semantic Scholar, Crossref, OpenAlex, DBLP, arXiv, ACL Anthology and doi.org,
  checks cited web pages and repositories, and highlights the results in the PDF.</p>
  <div class="ares-steps"><span>1 · Extract references</span><span>2 · Verify each one</span>
  <span>3 · Review flagged items</span><span>4 · Export a report</span></div>
</div>
""")

def _landing() -> None:
    c1, c2, c3 = st.columns(3)
    c1.markdown("**:material/verified: Verified**  \nA database or the cited page "
                "confirms title and authors.")
    c2.markdown("**:material/report: Mismatch**  \nA paper exists at that title, DOI or "
                "arXiv ID, but the details disagree — often a wrong or invented citation.")
    c3.markdown("**:material/help: Not found**  \nNo source knows it. Common for "
                "software, datasets, blogs and very new papers.")
    st.info("**Human review is required.** ARES points you to references worth checking; "
            "it does not decide whether a citation is correct. Databases are incomplete "
            "and matching has limits.", icon=":material/person_search:")


up_col, opt_col = st.columns([5, 1], vertical_alignment="center")
with up_col:
    uploaded = st.file_uploader("Paper (PDF, up to 25 MB)", type=["pdf"],
                                label_visibility="collapsed")
with opt_col:
    with st.popover("Options", icon=":material/tune:", width="stretch"):
        options = parser_options()
        if st.session_state.get("opt_parser") not in options:
            st.session_state.opt_parser = options[0]
        parser = st.segmented_control(
            "Parser", options, key="opt_parser", required=True,
            help="\n\n".join(f"**{o}** — {PARSER_HELP[o]}" for o in options))
        if "hybrid" not in options:
            st.caption(":material/cloud_off: GROBID is unavailable — only the simple "
                       "parser can run right now.")
        use_llm = st.toggle(
            "Repair fields with an LLM", value=LLM_READY, disabled=not LLM_READY,
            key="opt_llm",
            help=(f"Fixes swapped / truncated fields and drops body text that was "
                  f"mistaken for a reference. Model: {LLM_MODEL} ({LLM_BACKEND})."
                  if LLM_READY else "No LLM is configured on this server."))
        use_scholar = st.toggle(
            "Google Scholar fallback (slow)", value=False, key="opt_scholar",
            help="Last-resort lookup for references no index knows. Rate-limited "
                 "to one query every few seconds.")
        show_debug = st.toggle("Show parser output", value=False, key="opt_debug")

# One slot shared by the landing content and (after upload) the processing
# log.  Streamlit keeps an element from the previous run on screen until
# something new is written at its position, so with separate elements the
# greyed-out landing page lingered under the progress log and the page
# jumped when processing finished.  Writing into the same slot replaces the
# landing page the moment the upload run starts.
slot = st.empty()

if uploaded is None:
    st.session_state.pop("doc_id", None)
    # A single child inside the slot: the first notice written after an
    # upload (the processing status) then replaces all of it at once.
    with slot.container(), st.container():
        _landing()
    st.stop()

pdf_bytes = uploaded.getvalue()
doc_id = state.file_hash(pdf_bytes)
state.bind_document(doc_id)



# ---------------------------------------------------------------------------
# Processing (status block the first time, memoised afterwards)
# ---------------------------------------------------------------------------

# Everything that appears only sometimes (processing log, warnings, job
# progress, errors) lives in this one container, created on every run.  A
# custom component is remounted when its position in the page changes, so
# a status block present on one run and gone on the next used to reload
# the PDF viewer and lose the reader's scroll position.
notices = slot.container()

proc_key = (doc_id, parser, bool(use_llm))
cached = st.session_state.get("doc_cache")
if cached and cached[0] == proc_key:
    base_doc: Document = cached[1]
    # Same collapsed status line the processing run ends with, so the page
    # doesn't shift up when later reruns no longer process anything.
    if base_doc.refs:
        with notices, st.status(f"Found {len(base_doc.refs)} references in "
                                f"{base_doc.timings['total']:.0f}s",
                                state="complete", expanded=False):
            t = base_doc.timings
            st.write(f"Parsed with **{parser}** in {t.get('parse', 0):.1f}s"
                     + (f" · LLM repair {t['repair']:.1f}s" if "repair" in t else "")
                     + f" · located in PDF {t.get('locate', 0):.1f}s")
else:
    with notices, st.status("Reading the paper …", expanded=True) as status:
        base_doc = process(pdf_bytes, parser, bool(use_llm), log=st.write)
        if not base_doc.refs:
            status.update(label="No references found", state="error")
        else:
            status.update(label=f"Found {len(base_doc.refs)} references in "
                                f"{base_doc.timings['total']:.0f}s", state="complete",
                          expanded=False)
    st.session_state.doc_cache = (proc_key, base_doc)
    # (Re-)processed: results from other parser / LLM settings no longer
    # line up with this reference list.
    for k in ("lookup_results", "claim_verifications", "auto_lookup_done", "focused_ref"):
        st.session_state.pop(k, None)

for w in base_doc.warnings:
    notices.warning(w, icon=":material/warning:")
if not base_doc.refs:
    st.error("No references could be extracted from this PDF. If it is a scanned "
             "document, the text layer may be missing. Try another parser under **Options**.")
    if show_debug:
        st.code(base_doc.debug[:20000] or "(empty)")
    st.stop()

doc = with_manual(base_doc, pdf_bytes, state.manual_refs())
refs = doc.refs
st.session_state.n_base = doc.n_base
lookup_results = state.lookup_results()
if lookup_results is not None and len(lookup_results) != len(refs):
    # Out of sync (e.g. a manual ref was removed elsewhere) — pad / trim.
    lookup_results = (lookup_results + [None] * len(refs))[:len(refs)]
    st.session_state.lookup_results = lookup_results


# ---------------------------------------------------------------------------
# Background jobs: lookups and claim checks
# ---------------------------------------------------------------------------

def _lookup(subset: list[dict], progress_cb=None) -> list[dict]:
    return lookup_all(subset, progress_cb=progress_cb,
                      scholarly_budget=30 if use_scholar else 0)


def start_lookup(indices: list[int] | None = None, label: str | None = None) -> None:
    idx = list(range(len(refs))) if indices is None else indices
    jobs.start("lookup", label or f"Checking {len(idx)} references",
               _lookup, [refs[i] for i in idx], payload={"indices": idx, "n": len(refs)})


def _lookup_done(job: jobs.Job) -> None:
    idx, n = job.payload["indices"], job.payload["n"]
    current = state.lookup_results()
    if current is None or len(current) != n or len(idx) == n:
        current = [None] * n
    for slot, res in zip(idx, job.result):
        current[slot] = res
    st.session_state.lookup_results = current
    if len(idx) == n:
        st.session_state.pop("claim_verifications", None)


def _verify(pdf: bytes, refs_: list[dict], lrs: list, depth: str, ref_indices,
            progress_cb=None) -> list[dict]:
    from dataclasses import asdict
    from pipeline import PipelineOptions, _verify_claims_stage
    out = _verify_claims_stage(pdf, refs_, lrs, PipelineOptions(
        verify_claims=True, verify_depth=depth, verify_refs=ref_indices))
    return [asdict(v) for v in out]


def start_claims(depth: str, ref_indices: list[int] | None = None) -> None:
    what = "against full texts" if depth == "fulltext" else "against abstracts"
    jobs.start("claims", f"Checking citation context {what}", _verify,
               pdf_bytes, refs, lookup_results, depth, ref_indices,
               payload={"ref_indices": ref_indices})


def _claims_done(job: jobs.Job) -> None:
    only = job.payload.get("ref_indices")
    old = st.session_state.get("claim_verifications") or []
    keep = [v for v in old if only is not None and v.get("cited_ref_index") not in set(only)]
    st.session_state.claim_verifications = keep + job.result


ON_DONE = {"lookup": _lookup_done, "claims": _claims_done}

# Verify automatically right after extraction — one upload, one result.
if lookup_results is None and not jobs.busy() and not st.session_state.get("auto_lookup_done"):
    st.session_state.auto_lookup_done = True
    start_lookup()

with notices:
    if err := st.session_state.pop("job_error", None):
        st.error(err, icon=":material/error:")
    if jobs.current():
        jobs.progress_panel(ON_DONE)


# ---------------------------------------------------------------------------
# Summary tiles + toolbar
# ---------------------------------------------------------------------------

c = cards.counts(lookup_results, len(refs))
verdicts = st.session_state.get("claim_verifications") or []
m = st.columns(5 if not verdicts else 6)
m[0].metric("References", len(refs), border=True,
            help=f"{sum(1 for r in refs if r.get('found'))} located in the PDF")
m[1].metric("Verified", c["match"], border=True)
m[2].metric("Mismatch", c["mismatch"], border=True,
            help="A paper exists at this title / DOI / arXiv ID but details disagree.")
m[3].metric("Not found", c["missing"], border=True,
            help="No index knows it — common for software, datasets and new papers.")
m[4].metric("Dead links", c["dead"], border=True,
            help="Cited URLs that no longer resolve (archived copies are checked).")
if verdicts:
    n_con = sum(1 for v in verdicts if v.get("verdict") == "contradicted")
    m[5].metric("Claims contradicted", n_con, border=True,
                help=f"of {len(verdicts)} citing sentences checked")


@st.dialog("Add a reference", width="large")
def add_dialog() -> None:
    st.caption("Add references the parser missed. You can also select text in the PDF "
               "and press **Add selected text as reference** in the viewer.")
    tab_paste, tab_sugg, tab_mine = st.tabs([
        "Paste a citation", f"Suggestions ({len(viewer.visible_orphans(doc))})",
        f"Added by you ({len(state.manual_refs())})"])
    with tab_paste:
        text = st.text_area("Citation", height=110, label_visibility="collapsed",
                            placeholder="John Hewitt and Christopher D. Manning. 2019. A "
                                        "structural probe for finding syntax in word "
                                        "representations. NAACL.")
        if st.button("Add", type="primary", disabled=not text.strip(), icon=":material/add:"):
            if viewer.add_citation(text):
                st.rerun()
    with tab_sugg:
        st.caption("Text blocks on the bibliography pages that no extracted reference "
                   "covers.")
        viewer.orphan_list(doc, "dlg")
    with tab_mine:
        mine = state.manual_refs()
        if not mine:
            st.caption("Nothing added yet.")
        for i, mr in enumerate(mine):
            a, b = st.columns([12, 2], vertical_alignment="center")
            a.markdown(f"**{doc.n_base + i + 1}.** {(mr.get('title') or mr.get('raw') or '')[:100]}")
            if b.button("Remove", key=f"rm_{i}", icon=":material/delete:", width="stretch"):
                state.remove_manual_ref(i)
                st.rerun()


pending = [i for i, lr in enumerate(lookup_results or []) if lr is None]
with st.container(horizontal=True, gap="small"):
    if st.button("Add reference", icon=":material/playlist_add:"):
        add_dialog()
    if lookup_results is not None and pending and not jobs.busy():
        if st.button(f"Check {len(pending)} new", type="primary", icon=":material/search:"):
            start_lookup(pending, f"Checking {len(pending)} new reference(s)")
            st.rerun()
    if LLM_READY and lookup_results is not None and c["match"] and not pending:
        with st.popover("Check citation context", icon=":material/fact_check:",
                        disabled=jobs.busy()):
            st.markdown("Does each cited paper support the sentence citing it? "
                        f"Uses **{LLM_MODEL}** on the {c['match']} verified references.")
            if st.button("Quick — abstracts (about a minute)", type="primary",
                         icon=":material/bolt:", width="stretch"):
                start_claims("abstract")
                st.rerun()
            if st.button("Thorough — open-access full texts (slow)",
                         icon=":material/menu_book:", width="stretch"):
                start_claims("fulltext")
                st.rerun()
    if lookup_results is not None and not jobs.busy():
        if st.button("Re-check all", icon=":material/refresh:", type="tertiary",
                     help="Looks everything up again (cached results are reused)."):
            start_lookup()
            st.rerun()

    if lookup_results is not None:
        from report import generate_report
        report_md = generate_report(refs, lookup_results, pdf_name=uploaded.name,
                                    parser=parser, claim_verifications=verdicts)
        stem = uploaded.name.rsplit(".", 1)[0]
        with st.popover("Export", icon=":material/download:"):
            st.download_button("Report (Markdown)", report_md, f"{stem}_references.md",
                               "text/markdown", icon=":material/description:", width="stretch")
            statuses = [overall_status(lr) for lr in lookup_results]
            st.download_button(
                "Annotated PDF",
                lambda: annotated_pdf(pdf_bytes, json.dumps(doc.refs[:doc.n_base]),
                                      json.dumps(statuses[:doc.n_base])),
                f"{stem}_annotated.pdf", "application/pdf",
                icon=":material/picture_as_pdf:", width="stretch",
                on_click="ignore")
            rows = [{"n": i + 1, "status": statuses[i], "reason": status_reason(lr),
                     "title": r.get("title", ""), "authors": r.get("authors", ""),
                     "year": r.get("year", ""), "venue": r.get("venue", ""),
                     "doi": r.get("doi", ""), "url": r.get("url", ""),
                     "page": r.get("page") or "",
                     "confirmed_by": ", ".join(SOURCE_NAMES.get(s, s) for s, v in (lr or {}).items()
                                               if isinstance(v, dict) and v.get("label") == "match"),
                     "raw": r.get("raw", "")}
                    for i, (r, lr) in enumerate(zip(refs, lookup_results))]
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
            st.download_button("Table (CSV)", buf.getvalue(), f"{stem}_references.csv",
                               "text/csv", icon=":material/table:", width="stretch")
            st.download_button(
                "Full results (JSON)",
                json.dumps({"file": uploaded.name, "parser": parser, "references": rows,
                            "lookups": lookup_results, "claims": verdicts}, default=str),
                f"{stem}_references.json", "application/json",
                icon=":material/data_object:", width="stretch")

claims_bar = st.container()     # fixed slot, see `notices`
if verdicts:
    vc = {k: sum(1 for v in verdicts if v.get("verdict") == k) for k in cards.CLAIM_LABEL}
    summary = " · ".join(f"{n} {cards.CLAIM_LABEL[k]}" for k, n in vc.items() if n)
    esc_refs = sorted({v["cited_ref_index"] for v in cards.escalatable(verdicts)})
    with claims_bar, st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f":material/fact_check: **Citation context:** {summary}")
        if esc_refs and not jobs.busy():
            if st.button(f"Re-check {len(esc_refs)} inconclusive with full texts",
                         type="tertiary", icon=":material/menu_book:"):
                start_claims("fulltext", esc_refs)
                st.rerun()


# ---------------------------------------------------------------------------
# Workspace: reference list + PDF.
#
# Not a fragment on purpose: a custom component inside a fragment is
# remounted on every full-page rerun (e.g. when a job finishes), which
# reloads the PDF and loses the reader's scroll position.  Full reruns are
# cheap here — the processed document is memoised in session state.
# ---------------------------------------------------------------------------

def workspace() -> None:
    lrs = state.lookup_results()
    claims_by_ref = cards.by_ref(st.session_state.get("claim_verifications"))
    statuses = cards.counts(lrs, len(refs))["statuses"]
    left, right = st.columns([5, 6], gap="medium")

    with left:
        f1, f2 = st.columns([3, 2], vertical_alignment="bottom")
        with f1:
            opts = ["all"] + [k for k in ("mismatch", "missing", "match", "pending")
                              if k != "pending" or "pending" in statuses]
            if st.session_state.get("status_filter") not in opts:
                st.session_state.status_filter = "all"
            chosen = st.segmented_control(
                "Filter by status", opts, key="status_filter", required=True,
                format_func=lambda s: (f"All {len(statuses)}" if s == "all"
                                       else f"{STATUS_LABEL[s]} {statuses.count(s)}"))
        with f2:
            query = st.text_input("Search", key="search",
                                  placeholder="Title, author, venue …",
                                  icon=":material/search:")
        with st.container(horizontal=True, vertical_alignment="center"):
            page_only = st.toggle("Only references on the visible PDF page", key="page_only")
            with st.popover("How is this decided?", icon=":material/help:", type="tertiary"):
                st.markdown(explain.MATCHING_MD)

        visible = list(range(len(refs)))
        if chosen and chosen != "all":
            visible = [i for i in visible if statuses[i] == chosen]
        if query:
            q = query.lower()
            visible = [i for i in visible
                       if q in " ".join(str(refs[i].get(k) or "") for k in
                                        ("title", "authors", "raw", "venue")).lower()]
        if page_only:
            page = st.session_state.get("pdf_current_page") or 1
            visible = [i for i in visible if refs[i].get("page") == page]

        focused = st.session_state.get("focused_ref")
        if isinstance(focused, int) and 0 <= focused < len(refs):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"Showing reference **[{focused + 1}]** from the PDF")
                if st.button("Show all", type="tertiary", icon=":material/close:"):
                    st.session_state.focused_ref = None
                    st.rerun()
            visible = [focused]
        else:
            focused = None
            st.caption(f"{len(visible)} of {len(refs)} references")

        with st.container(height=760, border=False):
            if not visible:
                st.caption("No references match these filters.")
            for i in visible:
                ref = refs[i]
                lr = lrs[i] if lrs and i < len(lrs) else None
                mine = claims_by_ref.get(i, [])
                # st.markdown, not st.html: the latter's sanitiser strips
                # target="_blank", so source links would leave the app.
                st.markdown(cards.card_html(i, ref, lr, mine, expanded=(i == focused)),
                            unsafe_allow_html=True)
                with st.container(horizontal=True, gap="small"):
                    if ref.get("found") and ref.get("page"):
                        if st.button(f"Show in PDF (p. {ref['page']})", key=f"go_{i}",
                                     type="tertiary", icon=":material/my_location:"):
                            st.session_state.selected_ref = i
                            st.session_state.focus_nonce = st.session_state.get("focus_nonce", 0) + 1
                    if lr is not None and overall_status(lr) != "match" and not jobs.busy():
                        if st.button("Re-check", key=f"re_{i}", type="tertiary",
                                     icon=":material/refresh:",
                                     help="Query every source again, bypassing the cache."):
                            forget_reference(ref)
                            start_lookup([i], f"Re-checking reference [{i + 1}]")
                            st.rerun()
                    esc = cards.escalatable(mine)
                    if esc and not jobs.busy():
                        if st.button(f"Full text ({len(esc)})", key=f"ft_{i}", type="tertiary",
                                     icon=":material/menu_book:",
                                     help="Re-check these citing sentences against the "
                                          "cited paper's open-access full text."):
                            start_claims("fulltext", [i])
                            st.rerun()
                    if ref.get("_user_added"):
                        if st.button("Remove", key=f"del_{i}", type="tertiary",
                                     icon=":material/delete:"):
                            state.remove_manual_ref(i - doc.n_base)
                            st.rerun()

    with right:
        viewer.render(doc, lrs, doc_id)
        n_orph = len(viewer.visible_orphans(doc))
        if n_orph:
            with st.expander(f"Possibly missed references ({n_orph})",
                             icon=":material/lightbulb:"):
                viewer.orphan_list(doc, "ws")


workspace()

if show_debug:
    with st.expander("Parser output", icon=":material/code:"):
        st.code(doc.debug[:50000] or "(empty)", language="xml" if parser == "grobid" else None)

if lookup_results is not None:
    with st.expander("Full report", icon=":material/description:"):
        st.markdown(report_md)

st.caption("ARES surfaces candidates for human review — always confirm a flagged "
           "reference against the original source before acting on it.")
