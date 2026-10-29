"""PDF viewer column: the pdf_selector component plus its events."""

from __future__ import annotations

import html

import streamlit as st

from parsers import ParserError
from pdf_selector import pdf_selector
from ui import state
from ui.processing import Document, structure_citation
from verdict import overall_status, status_rgba

GROBID_DOWN = ("GROBID (the citation parser service) is unreachable right now — "
               "try again in a minute.")


def add_citation(text: str, page: int | None = None, rect: list | None = None) -> bool:
    """Structure ``text`` with GROBID and append it as a manual ref.
    Shows the outcome as a toast; returns True when added."""
    try:
        ref = structure_citation(text)
    except ParserError:
        st.toast(GROBID_DOWN, icon=":material/cloud_off:")
        return False
    if not ref:
        st.toast("That doesn't look like a citation — include authors, year and title.",
                 icon=":material/help:")
        return False
    if page:
        ref["page"] = page
    if rect:
        ref["rect"] = rect
    state.append_manual_ref(ref)
    st.toast(f"Added: {(ref.get('title') or text)[:80]}", icon=":material/add_circle:")
    return True


def render(doc: Document, lookup_results: list | None, doc_id: str) -> None:
    refs = doc.refs
    # Consume a "Show in PDF" request once, so later reruns don't snap back.
    scroll_page = None
    sel = st.session_state.get("selected_ref")
    if sel is not None and sel != st.session_state.get("last_scrolled_ref") and sel < len(refs):
        scroll_page = refs[sel].get("page")
        st.session_state.last_scrolled_ref = sel

    annots = []
    for i, r in enumerate(refs):
        if not r.get("rect") or not r.get("page"):
            continue
        lr = lookup_results[i] if lookup_results and i < len(lookup_results) else None
        status = overall_status(lr)
        annots.append({
            "page": r["page"], "rect": r["rect"], "color": status_rgba(status),
            "opacity": 1.0, "kind": "filled", "ref_index": i,
            "label": f"[{i + 1}] {(r.get('title') or '')[:80]} · {status} — click for details",
        })

    event = pdf_selector(
        pdf_bytes=doc.pdf, annotations=annots, height=860, scroll_to_page=scroll_page,
        emit_page_changes=bool(st.session_state.get("page_only")),
        key=f"pdf_{doc_id}",     # stable per document: remounting re-sends the PDF
    )
    if not (isinstance(event, dict) and event.get("id")
            and event["id"] != st.session_state.get("last_pdf_event_id")):
        return
    st.session_state.last_pdf_event_id = event["id"]
    kind = event.get("kind")
    if kind == "page":
        if event.get("page") and event["page"] != st.session_state.get("pdf_current_page"):
            st.session_state.pdf_current_page = event["page"]
            st.rerun()
    elif kind == "ref_click":
        idx = event.get("ref_index")
        if isinstance(idx, int) and 0 <= idx < len(refs):
            st.session_state.focused_ref = idx
            st.rerun()
    elif kind == "selection":
        text = (event.get("text") or "").strip()
        if text and add_citation(text, event.get("page"), event.get("rect")):
            st.rerun()


def visible_orphans(doc: Document) -> list[dict]:
    """Suggestions not yet added by the user."""
    added = {(m.get("raw") or "").strip().lower() for m in state.manual_refs()}
    return [o for o in doc.orphans if (o.get("text") or "").strip().lower() not in added]


def orphan_list(doc: Document, key_prefix: str) -> None:
    orphans = visible_orphans(doc)
    if not orphans:
        st.caption("No further suggestions — every reference-like block on the "
                   "bibliography pages is already in the list.")
        return
    for j, o in enumerate(orphans):
        c1, c2 = st.columns([12, 2], vertical_alignment="center")
        text = o["text"]
        c1.markdown(f'<div class="orphan"><span class="pg">p. {o["page"]}</span> '
                    f'{html.escape(text[:260])}{"…" if len(text) > 260 else ""}</div>',
                    unsafe_allow_html=True)
        if c2.button("Add", key=f"{key_prefix}_orph_{j}", icon=":material/add:",
                     width="stretch"):
            if add_citation(text.strip(), o.get("page"), o.get("rect")):
                st.rerun()
