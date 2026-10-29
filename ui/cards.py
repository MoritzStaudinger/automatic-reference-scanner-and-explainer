"""
HTML for reference cards.  Cards are plain ``<details>`` elements — cheap
to render for hundreds of references — styled by ``ui/style.css`` with
colours that work on both the light and dark theme.  All interpolated
text is escaped.
"""

from __future__ import annotations

import html
import urllib.parse
from typing import Any

from lookup import _is_preprint_venue
from verdict import SOURCE_NAMES, STATUS_LABEL, overall_status, status_reason

_ICON = {"match": "✓", "mismatch": "✕", "missing": "!", "pending": "…"}
_LABEL_DOT = {"match": "ok", "fuzzy": "warn", "mismatch": "bad"}
_SOURCE_ORDER = ("doi_org", "semantic_scholar", "acl_anthology", "dblp", "openalex",
                 "crossref", "arxiv", "openlibrary", "scholarly", "url_verify")

CLAIM_LABEL = {
    "supported": "supported", "contradicted": "contradicted",
    "not_addressed": "not addressed", "ambiguous": "ambiguous",
    "no_source": "no text available", "error": "check failed",
}
_CLAIM_TONE = {"supported": "ok", "contradicted": "bad", "not_addressed": "muted",
               "ambiguous": "warn", "no_source": "muted", "error": "muted"}

esc = html.escape


def _pct(x: float | None) -> str:
    return f"{x:.0%}" if isinstance(x, (int, float)) else ""


def _link(url: str, text: str) -> str:
    if not url.startswith(("http://", "https://")):
        return esc(text)
    return f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(text)}</a>'


def source_rows(lr: dict, ref: dict) -> str:
    """One row per source that found something, plus warnings."""
    rows: list[str] = []
    cited_venue = ref.get("venue") or ""
    for key in _SOURCE_ORDER:
        r = lr.get(key) or {}
        name = SOURCE_NAMES.get(key, key)
        status = r.get("status")
        if key == "url_verify" and status in ("dead_link", "error", "not_found"):
            note = r.get("note") or r.get("error") or status
            tone = "bad" if status == "dead_link" else "muted"
            url = r.get("url") or r.get("archived_url") or ""
            rows.append(f'<li><span class="dot {tone}"></span><b>{_link(url, name)}</b> '
                        f'<span class="muted">{esc(note)}</span></li>')
            continue
        if status != "found":
            continue
        url = r.get("url") or ""
        if not url and key == "scholarly" and ref.get("title"):
            url = "https://scholar.google.com/scholar?q=" + urllib.parse.quote_plus(ref["title"])
        label = r.get("label")
        found_t = r.get("found_title") or ""
        note = f' — <i>{esc(found_t)}</i>' if label != "match" and found_t else ""
        archived = (f' <span class="chip">archived copy</span>' if r.get("archived") else "")
        rows.append(f'<li><span class="dot {_LABEL_DOT.get(label, "muted")}"></span>'
                    f'<b>{_link(url, name)}</b> <span class="muted">{_pct(r.get("similarity"))}'
                    f'</span>{note}{archived}</li>')
        if r.get("wrong_doi"):
            rows.append(f'<li class="warnrow">Cited DOI <code>{esc(r["wrong_doi"])}</code> '
                        f'points to a different paper</li>')
        if r.get("wrong_arxiv_id"):
            rows.append(f'<li class="warnrow">Cited arXiv ID <code>{esc(r["wrong_arxiv_id"])}'
                        f'</code> points to a different paper</li>')
        if label == "fuzzy":
            continue     # a weak candidate: its author / venue details are noise
        fa, asim = r.get("found_authors") or "", r.get("authors_sim")
        if r.get("wrong_authors") and fa:
            overlap = f" ({_pct(asim)} overlap)" if asim is not None else ""
            rows.append(f'<li class="sub">Authors found: <i>{esc(fa[:300])}</i>{overlap}</li>')
        fv, vlabel = r.get("found_venue") or "", r.get("venue_label")
        if (fv and cited_venue and vlabel in ("fuzzy", "mismatch")
                and not _is_preprint_venue(cited_venue)):
            rows.append(f'<li class="sub">Venue found: <i>{esc(fv)}</i> '
                        f'{_pct(r.get("venue_sim"))}</li>')
    if lr.get("_error"):
        rows.append(f'<li class="warnrow">Lookup failed: {esc(lr["_error"])}</li>')
    if not rows:
        return ""
    return f'<div class="sect"><div class="sect-h">Sources</div><ul>{"".join(rows)}</ul></div>'


def fields_html(ref: dict) -> str:
    items = [(k.capitalize(), ref[k]) for k in ("authors", "year", "venue", "doi", "url")
             if ref.get(k)]
    if not items:
        return ""
    cells = "".join(f"<div><b>{esc(k)}</b> {esc(str(v))}</div>" for k, v in items)
    return f'<div class="fields">{cells}</div>'


def claims_html(verdicts: list[dict]) -> str:
    if not verdicts:
        return ""
    rows = []
    for v in verdicts:
        verdict = v.get("verdict") or "error"
        badge = f'<span class="chip {_CLAIM_TONE.get(verdict, "muted")}">{CLAIM_LABEL.get(verdict, verdict)}</span>'
        src = (' <span class="chip">abstract only</span>' if v.get("is_abstract_fallback")
               else (f' <span class="muted">via {esc(v["source"])}</span>' if v.get("source") else ""))
        conf = v.get("confidence")
        conf_s = (f' <span class="muted">{esc(conf)} confidence</span>'
                  if conf and verdict not in ("no_source", "error") else "")
        evidence = (f'<blockquote>“{esc((v.get("evidence_passage") or "")[:300])}”</blockquote>'
                    if v.get("evidence_passage") else "")
        expl = (f'<div class="muted">{esc((v.get("explanation") or "")[:300])}</div>'
                if v.get("explanation") else "")
        rows.append(f'<li>{badge}{conf_s}{src}<div class="claim">“{esc((v.get("claim_text") or "")[:400])}”'
                    f'</div>{evidence}{expl}</li>')
    return (f'<div class="sect"><div class="sect-h">Citation context ({len(verdicts)})</div>'
            f'<ul class="claims">{"".join(rows)}</ul></div>')


def card_html(i: int, ref: dict, lr: dict | None, verdicts: list[dict],
              expanded: bool = False) -> str:
    status = overall_status(lr)
    title = esc(ref.get("title") or (ref.get("raw") or "")[:110])
    page = ref.get("page")
    meta = [f"p. {page}" if page else "not located in PDF"]
    reason = status_reason(lr)
    badges = ""
    if ref.get("_auto_added"):
        badges += '<span class="chip info">auto-detected</span>'
    elif ref.get("_user_added"):
        badges += '<span class="chip info">added by you</span>'
    url_r = (lr or {}).get("url_verify") or {}
    if url_r.get("status") == "dead_link":
        badges += '<span class="chip bad">dead link</span>'
    n_bad_claims = sum(1 for v in verdicts if v.get("verdict") == "contradicted")
    if n_bad_claims:
        badges += f'<span class="chip bad">{n_bad_claims} claim{"s" if n_bad_claims > 1 else ""} contradicted</span>'
    body = (f'<div class="raw">{esc(ref.get("raw") or "").replace(chr(10), "<br>")}</div>'
            + fields_html(ref)
            + (source_rows(lr, ref) if lr else "")
            + claims_html(verdicts))
    return f"""<details class="refcard {status}"{' open' if expanded else ''}>
<summary><span class="st {status}" title="{STATUS_LABEL[status]}">{_ICON[status]}</span>
<span class="num">[{i + 1}]</span> <span class="ttl">{title}</span>{badges}
<div class="meta">{esc(' · '.join(meta))}{(' · ' + esc(reason)) if reason else ''}</div></summary>
<div class="body">{body}</div></details>""".replace("\n", " ")
    # One line: a blank line inside would end markdown's HTML block.


def escalatable(verdicts: list[dict]) -> list[dict]:
    """Verdicts a full-text check might change: inconclusive answers from
    the abstract, or no text found at all."""
    return [v for v in verdicts
            if (v.get("verdict") in ("not_addressed", "ambiguous") and v.get("is_abstract_fallback"))
            or v.get("verdict") == "no_source"]


def by_ref(verdicts: list[dict] | None) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for v in verdicts or []:
        out.setdefault(v.get("cited_ref_index", -1), []).append(v)
    return out


def counts(lookup_results: list | None, n_refs: int) -> dict[str, Any]:
    statuses = [overall_status(lookup_results[i] if lookup_results and i < len(lookup_results) else None)
                for i in range(n_refs)]
    dead = sum(1 for lr in (lookup_results or [])
               if lr and (lr.get("url_verify") or {}).get("status") == "dead_link")
    return {"statuses": statuses, "dead": dead,
            **{s: statuses.count(s) for s in ("match", "mismatch", "missing", "pending")}}
