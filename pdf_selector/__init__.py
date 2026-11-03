"""
Custom Streamlit component: PDF viewer with text-selection callbacks.

Renders a PDF in the browser via PDF.js, displays a text layer over the
canvas so users can select text natively, and streams the selected text
back to Python via Streamlit's component-value bridge.

Designed as a vanilla HTML+JS component — no React, no build pipeline.
Drops in as a replacement for streamlit_pdf_viewer when you need to capture
user interaction with the PDF.

Returned value
--------------
``pdf_selector(...)`` returns ``None`` when nothing has been selected, or a
dict ``{"text": str, "page": int, "rects": [[x0,y0,x1,y1], ...], "id": str}``
when the user clicks the "Add as reference" button in the viewer.  The ``id``
field is a monotonic token so consecutive selections of the same text still
trigger a Streamlit rerun (without it, the value wouldn't change and the
script wouldn't re-execute).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_NAME = "pdf_selector"
_FRONTEND_DIR = (Path(__file__).parent / "frontend").resolve()

_component_func = components.declare_component(
    _COMPONENT_NAME,
    path=str(_FRONTEND_DIR),
)


def pdf_selector(
    pdf_bytes: bytes,
    annotations: list[dict[str, Any]] | None = None,
    height: int = 900,
    scroll_to_page: int | None = None,
    emit_page_changes: bool = False,
    focus: dict[str, Any] | None = None,
    items: list[dict[str, Any]] | None = None,
    status_labels: dict[str, str] | None = None,
    status_order: list[str] | None = None,
    cards_css: str = "",
    tiles: list[dict[str, Any]] | None = None,
    key: str | None = None,
) -> dict[str, Any] | None:
    """
    Render a PDF and return user-selected text when the "Add" button is clicked.

    Parameters
    ----------
    pdf_bytes : bytes
        The PDF file content.
    annotations : list[dict] | None
        Optional list of rectangles to draw on the PDF.  Each entry:
        ``{"page": 1-based int, "rect": [x0, y0, x1, y1], "color": "#rrggbb",
        "opacity": float, "kind": "filled" | "outline", "label": str}``.
        Coordinates are in PDF points (PyMuPDF's native unit).
    height : int
        Viewer height in CSS pixels.
    scroll_to_page : int | None
        1-based page to scroll into view on render.
    focus : dict | None
        ``{"page", "rect", "nonce"}`` — scroll to that rectangle and flash
        it.  Runs once per new ``nonce`` (repeat clicks on the same
        reference need a fresh nonce).
    items : list[dict] | None
        When given, the component also shows the reference list (filters,
        search, page filter, cards) and handles it client-side.  Each item:
        ``{"i", "status", "page", "rect", "html", "text", "actions": [{"act",
        "label", "disabled"}]}``.  Server actions come back as
        ``{"kind": "action", "action", "ref_index", "id"}``.
    tiles : list[dict] | None
        Summary tiles shown above the list; clicking one filters it.  Each:
        ``{"label", "value", "filter", "help"}`` where ``filter`` is "all", a
        status key, "dead" or "bad_claims" (item flags of the same name).
    key : str | None
        Streamlit component key for state separation.

    Returns
    -------
    dict | None
        ``None`` until the user clicks the "Add" button.  Then a dict with the
        selection's text, page, bounding rects, and a unique id.
    """
    return _component_func(
        pdf_b64            = base64.b64encode(pdf_bytes).decode("ascii"),
        annotations        = annotations or [],
        height             = height,
        scroll_to_page     = scroll_to_page,
        emit_page_changes  = bool(emit_page_changes),
        focus              = focus,
        items              = items,
        status_labels      = status_labels or {},
        status_order       = status_order or [],
        cards_css          = cards_css,
        tiles              = tiles or [],
        default            = None,
        key                = key,
    )
