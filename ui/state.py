"""
Per-session state for the "Check a PDF" page.

Everything that belongs to one uploaded document lives under keys reset
by ``bind_document`` whenever the file changes, so nothing — manual refs,
lookup results, claim verdicts, scroll / focus positions — leaks from one
PDF into the next.  Files are identified by content hash: (name, size)
treated two different PDFs of equal size as the same document.

Reference list layout (indices are shared with ``lookup_results``):

    [ parsed refs … | auto-detected refs … | manually added refs … ]
                                            ^ st.session_state.n_base
"""

from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st

# Keys reset whenever a different document is loaded.
_DOC_KEYS = (
    "manual_refs", "lookup_results", "claim_verifications",
    "selected_ref", "focus_nonce", "focused_ref", "pdf_current_page",
    "last_pdf_event_id", "processed_key", "n_base", "status_filter",
    "search", "page_only", "job", "auto_lookup_done",
)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:20]


def bind_document(doc_id: str) -> None:
    """Reset per-document state when ``doc_id`` differs from the last one."""
    if st.session_state.get("doc_id") == doc_id:
        return
    job = st.session_state.get("job")
    if job is not None:
        job.cancelled = True          # its result will be ignored
    for k in _DOC_KEYS:
        st.session_state.pop(k, None)
    st.session_state.doc_id = doc_id
    st.session_state.manual_refs = []


def manual_refs() -> list[dict[str, Any]]:
    return st.session_state.setdefault("manual_refs", [])


def append_manual_ref(new_ref: dict[str, Any]) -> None:
    """Add a user-supplied reference without wiping existing lookups: a
    ``None`` slot keeps ``lookup_results`` index-aligned and shows the new
    ref as pending until its lookup runs."""
    new_ref = {**new_ref, "_user_added": True}
    manual_refs().append(new_ref)
    lr = st.session_state.get("lookup_results")
    if isinstance(lr, list):
        lr.append(None)


def remove_manual_ref(i: int) -> None:
    """Remove manual ref ``i`` and its lookup slot (at ``n_base + i``)."""
    refs = manual_refs()
    if not 0 <= i < len(refs):
        return
    refs.pop(i)
    lr = st.session_state.get("lookup_results")
    slot = st.session_state.get("n_base", 0) + i
    if isinstance(lr, list) and 0 <= slot < len(lr):
        lr.pop(slot)
    # Claim verdicts never cite manual refs (claim_extractor skips them),
    # so they need no index fix-up.


def lookup_results() -> list | None:
    return st.session_state.get("lookup_results")
