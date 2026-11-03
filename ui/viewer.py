"""PDF viewer column: the pdf_selector component plus its events."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from parsers import ParserError
from pdf_selector import pdf_selector
from ui import cards, state
from ui.processing import Document, structure_citation
from verdict import STATUS_LABEL, overall_status, status_rgba

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


_CARDS_CSS = (Path(__file__).parent / "style.css").read_text()


def render(doc: Document, lookup_results: list | None, doc_id: str,
           claims_by_ref: dict[int, list[dict]], busy: bool,
           tiles: list[dict] | None = None) -> dict | None:
    """The reference list + PDF, as one component.  Filtering, search, the
    page filter, opening cards, "Show in PDF" and PDF → card jumps run in
    the browser (no rerun).  Returns a server action
    ``{"action": "recheck" | "fulltext" | "remove", "ref_index": i}`` or None."""
    refs = doc.refs
    statuses = [overall_status(lookup_results[i] if lookup_results and i < len(lookup_results)
                               else None) for i in range(len(refs))]

    annots, items = [], []
    for i, r in enumerate(refs):
        lr = lookup_results[i] if lookup_results and i < len(lookup_results) else None
        status = statuses[i]
        label = f"[{i + 1}] {(r.get('title') or '')[:80]} · {STATUS_LABEL[status]}"
        pieces = r.get("rects") or ([{"page": r["page"], "rect": r["rect"]}]
                                    if r.get("rect") and r.get("page") else [])
        for pc in pieces:    # a reference split over columns / pages has several
            annots.append({"page": pc["page"], "rect": pc["rect"], "color": status_rgba(status),
                           "opacity": 1.0, "kind": "filled", "ref_index": i,
                           "label": label + " — click to open its card"})
        mine = claims_by_ref.get(i, [])
        actions = []
        if lr is not None and status != "match":
            actions.append({"act": "recheck", "label": "↻ Re-check", "disabled": busy})
        if (n := len(cards.escalatable(mine))):
            actions.append({"act": "fulltext", "label": f"📖 Full text ({n})", "disabled": busy})
        if r.get("_user_added"):
            actions.append({"act": "remove", "label": "🗑 Remove", "disabled": False})
        items.append({
            "i": i, "status": status,
            "page": r.get("page") if r.get("found") else None,
            "rect": r.get("rect"),
            "html": cards.card_html(i, r, lr, mine),
            "text": " ".join(str(r.get(k) or "") for k in ("title", "authors", "venue", "raw")).lower(),
            "actions": actions,
            # Extra filters behind the "Dead links" / "Claims contradicted" tiles.
            "dead": ((lr or {}).get("url_verify") or {}).get("status") == "dead_link",
            "bad_claims": any(v.get("verdict") == "contradicted" for v in mine),
        })

    event = pdf_selector(
        pdf_bytes=doc.pdf, annotations=annots, height=990 if tiles else 900, items=items,
        tiles=tiles,
        status_labels=STATUS_LABEL, status_order=["mismatch", "missing", "match", "pending"],
        cards_css=_CARDS_CSS,
        key=f"pdf_{doc_id}",     # stable per document: remounting re-sends the PDF
    )
    if not (isinstance(event, dict) and event.get("id")
            and event["id"] != st.session_state.get("last_pdf_event_id")):
        return None
    st.session_state.last_pdf_event_id = event["id"]
    kind = event.get("kind")
    if kind == "action" and isinstance(event.get("ref_index"), int):
        return {"action": event["action"], "ref_index": event["ref_index"]}
    if kind == "selection":
        text = (event.get("text") or "").strip()
        if text and add_citation(text, event.get("page"), event.get("rect")):
            st.rerun()
    return None


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
