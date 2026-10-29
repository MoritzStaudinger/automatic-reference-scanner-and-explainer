"""
HTTP API reference page.

Deliberately short — the live OpenAPI UI at /api/docs is the source of
truth for endpoints and schemas.  This page points there and gives
copy-pasteable examples for whichever hostname the visitor used.
"""

import os

import streamlit as st

_KNOWN_HOSTS = {"reference-cleaner.ds-ifs.tuwien.ac.at", "arse.ds-ifs.tuwien.ac.at"}
_DEFAULT_BASE = os.getenv("ARES_API_PUBLIC_BASE", "https://reference-cleaner.ds-ifs.tuwien.ac.at")

# Examples use the host the visitor reached us at — only if it is one of
# ours, so a forged Host header can't put another URL on the page.
host = (st.context.headers.get("host") or "").split(":")[0].lower()
BASE = f"https://{host}" if host in _KNOWN_HOSTS else _DEFAULT_BASE

st.title("HTTP API")
st.markdown(
    "The pipeline behind this site is also available over HTTP: process one PDF "
    "synchronously, or submit a batch and collect the results later. "
    f"The interactive reference is at [{BASE}/api/docs]({BASE}/api/docs)."
)

with st.container(border=True):
    st.markdown(
        "**:material/key: Access** — every `/api/*` call needs an `X-API-Key` header. "
        "To request a key, email "
        "[moritz.staudinger@tuwien.ac.at](mailto:moritz.staudinger@tuwien.ac.at?subject=ARES%20API%20key%20request) "
        "with a line about what you want to use it for."
    )

st.subheader("Check one paper")
st.code(
    f'curl -fsS -X POST {BASE}/api/v1/process \\\n'
    f'     -H "X-API-Key: $ARES_API_KEY" \\\n'
    f'     -F file=@paper.pdf \\\n'
    f'     -F parser=hybrid \\\n'
    f'     -F use_llm=1 \\\n'
    f'     > report.json',
    language="bash",
)
st.markdown(
    "The response has a `summary` (counts, timings, LLM-repair and parser-fallback "
    "notes) and one entry per reference with its overall status, the reason, and "
    "what each source returned."
)

st.subheader("Batches")
st.code(
    f'job=$(curl -fsS -X POST {BASE}/api/v1/batch -H "X-API-Key: $ARES_API_KEY" \\\n'
    f'      -F files=@a.pdf -F files=@b.pdf -F parser=hybrid | jq -r .job_id)\n'
    f'curl -fsS {BASE}/api/v1/batch/$job -H "X-API-Key: $ARES_API_KEY"          # progress\n'
    f'curl -fsS {BASE}/api/v1/batch/$job/results.jsonl -H "X-API-Key: $ARES_API_KEY"',
    language="bash",
)

st.subheader("Citation context")
st.markdown(
    "Add `-F verify_claims=1` to also check whether each cited paper supports the "
    "sentence that cites it.\n\n"
    "- `verify_depth=abstract` (default) compares against the cited paper's abstract — "
    "fast, but only papers with an indexed abstract can be checked.\n"
    "- `verify_depth=fulltext` downloads open-access full texts first — slower, "
    "and covers more papers."
)

st.caption("Limits: one synchronous request is processed at a time per slot; when the "
           "server is busy you get `503` with `Retry-After`. Requests per IP are "
           "rate-limited (`429`).")
