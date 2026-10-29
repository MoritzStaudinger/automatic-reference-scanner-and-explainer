"""
HTTP API reference page for ARES.

Intentionally minimal — the live OpenAPI/Swagger UI at /api/docs is the
source of truth for endpoints, request shapes, and response schemas.
This page exists to point users there and give one copy-pasteable curl.
"""

import os
import streamlit as st


st.set_page_config(
    page_title="API — ARES",
    layout="wide",
)


# Mirror the typography from pages/About.py so the sidebar of pages
# reads as one cohesive thing.
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter",
                 "Helvetica Neue", Arial, sans-serif;
}
h1, h2, h3, h4 { letter-spacing: -0.01em; color: #1f2933; }
.stApp h1 { font-weight: 600; font-size: 1.95rem; }
.stApp h2 { font-weight: 600; font-size: 1.35rem; margin-top: 2rem; }
.api-lead {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #343a40;
    max-width: 780px;
    margin-bottom: 1.2rem;
}
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.page_link("app.py",          label="Validate a PDF")
    st.page_link("pages/API.py",    label="API")
    st.page_link("pages/About.py",  label="About")
    st.divider()


# Resolve the public base URL from the request so the curl snippet
# works on whatever host the visitor reaches us at.
try:
    _hdrs   = st.context.headers       # Streamlit ≥ 1.40
    _scheme = _hdrs.get("x-forwarded-proto", "https")
    _host   = _hdrs.get("host", "reference-cleaner.ds-ifs.tuwien.ac.at")
    BASE = f"{_scheme}://{_host}"
except Exception:
    BASE = os.getenv("ARES_API_PUBLIC_BASE",
                     "https://reference-cleaner.ds-ifs.tuwien.ac.at")


st.title("HTTP API")
st.markdown(
    '<p class="api-lead">'
    "ARES exposes the same pipeline this UI uses over HTTP — "
    "single-PDF processing and async batch.  "
    f'The full interactive reference is at <a href="{BASE}/api/docs">'
    f"<code>{BASE}/api/docs</code></a>."
    '</p>',
    unsafe_allow_html=True,
)

st.markdown(
    "Authenticate with an `X-API-Key` header on every call to `/api/*`.  "
    "To request a key, email "
    "[moritz.staudinger@tuwien.ac.at](mailto:moritz.staudinger@tuwien.ac.at?subject=ARES%20API%20key%20request)"
    " with a one-line note about what you want to use it for.  "
    "`GET /healthz` is the only unauthenticated endpoint."
)

st.code(
    f'curl -fsS -X POST {BASE}/api/v1/process \\\n'
    f'     -H "X-API-Key: $ARES_API_KEY" \\\n'
    f'     -F file=@paper.pdf \\\n'
    f'     -F parser=grobid \\\n'
    f'     > report.json',
    language="bash",
)

st.markdown(
    "Pass `-F verify_claims=1` to also run per-citation claim "
    "verification.  Two modes:\n\n"
    "- `verify_depth=abstract` (default): fast, ~30-60 s per paper, "
    "  ~40% of refs verifiable (many don't have S2/OpenAlex abstracts).\n"
    "- `verify_depth=fulltext`: slow, minutes-to-hours, ~80% of refs "
    "  verifiable (fetches OA PDFs from arXiv / S2 / OpenAlex)."
)
