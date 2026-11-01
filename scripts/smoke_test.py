"""
End-to-end smoke test for ARSE.

Exercises every code path that can run without a browser:
  - matching helpers (wrong_authors / wrong_venue / preprint exclusion)
  - URL metadata parser (citation_*, og:title, <title> fallback)
  - pipeline.process_pdf
  - HTTP API (health, auth, validation, sync process, async batch lifecycle)
  - batch CLI in local mode
  - (optional) one real-network URL verification

Exits 0 on full pass, 1 on any failure.  Streamlit UI behaviour
(highlight colours, sidebar add, manual refs) is covered by the
companion checklist in ``scripts/UI_CHECKLIST.md``.

Usage:
    python scripts/smoke_test.py [--pdf path/to/test.pdf] [--skip-network]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Make the project root importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[tuple[str, str, str]] = []


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    RESULTS.append((name, status, detail))
    line = f"  [{status}] {name}"
    if detail:
        line += f"  -- {detail[:200]}"
    print(line)
    return cond


# ---------------------------------------------------------------------------
# 1. Matching helpers
# ---------------------------------------------------------------------------

def test_finalize_match() -> None:
    section("Matching helpers (_finalize_match)")
    from lookup import _finalize_match

    r = _finalize_match(0.95, 0.10, None, "", "")
    check(
        "wrong_authors fires at tier-1 (tsim 0.95, asim 0.10)",
        r.get("wrong_authors") is True and r["label"] == "mismatch",
    )

    # Just-above-floor: titles are essentially the same, authors disagree.
    from lookup import WRONG_FLAG_TITLE_FLOOR
    r = _finalize_match(WRONG_FLAG_TITLE_FLOOR + 0.01, 0.10, None, "", "")
    check(
        "wrong_authors fires just above floor (asim 0.10)",
        r.get("wrong_authors") is True and r["label"] == "mismatch",
    )
    # A different paper with a similar title (0.86) is not "wrong authors".
    r = _finalize_match(0.86, 0.10, None, "", "")
    check("near-title (0.86) with other authors is not a mismatch",
          not r.get("wrong_authors") and r["label"] != "mismatch", f"{r}")

    # Just-below-floor: typical "nearest-titled paper in the index"
    # case — flagging would produce false positives (e.g. Anthropic
    # papers that S2 doesn't index returning irrelevant matches).
    r = _finalize_match(0.60, 0.10, None, "", "")
    check(
        "wrong_authors does NOT fire below WRONG_FLAG_TITLE_FLOOR (tsim 0.60)",
        not r.get("wrong_authors"),
    )

    r = _finalize_match(0.40, 0.10, None, "", "")
    check(
        "wrong_authors does NOT fire at low tsim (tsim 0.40)",
        not r.get("wrong_authors"),
    )

    r = _finalize_match(0.95, 0.80, None, "", "")
    check(
        "good asim keeps label='match'",
        not r.get("wrong_authors") and r["label"] == "match",
    )

    r = _finalize_match(0.95, None, None, "", "")
    check(
        "missing asim does NOT flag (we can't prove disagreement)",
        not r.get("wrong_authors") and r["label"] == "match",
    )

    r = _finalize_match(0.95, None, 0.10, "ACL", "NeurIPS")
    check(
        "wrong_venue fires when both venues present, low vsim, non-preprint",
        r.get("wrong_venue") is True and r["label"] == "mismatch",
    )

    r = _finalize_match(0.95, None, 0.10, "arXiv preprint", "NeurIPS")
    check(
        "wrong_venue suppressed when cited venue is a preprint marker",
        not r.get("wrong_venue") and r["label"] == "match",
    )

    r = _finalize_match(0.95, None, 0.10, "", "NeurIPS")
    check(
        "wrong_venue suppressed when cited venue is missing",
        not r.get("wrong_venue"),
    )


# ---------------------------------------------------------------------------
# 2. HTML metadata parser
# ---------------------------------------------------------------------------

def test_html_meta_parser() -> None:
    section("HTML metadata parser")
    from lookup import _parse_html_metadata

    citation_html = """
    <meta name="citation_title" content="Attention Is All You Need">
    <meta name="citation_author" content="Vaswani, Ashish">
    <meta name="citation_author" content="Shazeer, Noam">
    <meta name="citation_conference_title" content="NeurIPS">
    <meta name="citation_doi" content="10.5555/x.y">
    <meta name="citation_publication_date" content="2017/12/04">
    """
    m = _parse_html_metadata(citation_html)
    check(
        "citation_* meta extraction (title)",
        m.get("title") == "Attention Is All You Need",
    )
    check(
        "citation_* meta extraction (authors)",
        "Vaswani" in m.get("authors", "") and "Shazeer" in m.get("authors", ""),
    )
    check(
        "citation_* meta extraction (venue + doi + year)",
        m.get("venue") == "NeurIPS"
        and m.get("doi") == "10.5555/x.y"
        and m.get("year") == "2017",
    )

    og_only = '<html><head><meta property="og:title" content="Some Paper - Site Name"></head></html>'
    check(
        "og:title fallback (trims ' - Site Name' suffix)",
        _parse_html_metadata(og_only).get("title") == "Some Paper",
    )

    title_only = "<html><head><title>Just A Title | Site</title></head></html>"
    check(
        "<title> fallback (trims ' | Site' suffix)",
        _parse_html_metadata(title_only).get("title") == "Just A Title",
    )

    check(
        "empty HTML returns no title",
        not _parse_html_metadata("<html></html>").get("title"),
    )


# ---------------------------------------------------------------------------
# 3. Pipeline
# ---------------------------------------------------------------------------

def test_pipeline_local(pdf_path: Path) -> dict | None:
    section(f"Pipeline (local, pymupdf4llm) on {pdf_path.name}")
    from pipeline import PipelineOptions, process_pdf

    pdf_bytes = pdf_path.read_bytes()
    t0 = time.time()
    try:
        report = process_pdf(
            pdf_bytes,
            PipelineOptions(parser="pymupdf4llm", use_llm=False),
        )
    except Exception as exc:
        check("pipeline runs without crashing", False, str(exc))
        return None
    elapsed = time.time() - t0

    check("pipeline returns refs", report.n_refs > 0, f"{report.n_refs} refs")
    check(
        "match + mismatch + missing == total",
        (report.n_match + report.n_mismatch + report.n_missing) == report.n_refs,
        f"{report.n_match}+{report.n_mismatch}+{report.n_missing} vs {report.n_refs}",
    )
    out = report.to_dict()
    try:
        json.dumps(out)
        json_ok = True
    except (TypeError, ValueError) as exc:
        json_ok = False
        check("report is JSON-serialisable", False, str(exc))
    if json_ok:
        check("report is JSON-serialisable", True)

    check(
        "report carries new wrong_authors / wrong_venue fields on refs",
        all("wrong_authors" in r and "wrong_venue" in r for r in out["refs"]),
    )
    check(
        f"runtime under 5 min (was {elapsed:.0f}s)",
        elapsed < 300,
        f"{elapsed:.1f}s",
    )
    return out


# ---------------------------------------------------------------------------
# 4. HTTP API
# ---------------------------------------------------------------------------

def test_api(pdf_path: Path) -> None:
    section("HTTP API (FastAPI TestClient)")
    os.environ["ARES_API_KEY"] = "smoke-test-key"
    os.environ["ARES_JOBS_DB"] = str(
        Path(tempfile.gettempdir()) / "ares_smoke_jobs.sqlite"
    )
    # Drop any stale DB so the recent-jobs assertion isn't muddied
    try:
        Path(os.environ["ARES_JOBS_DB"]).unlink()
    except FileNotFoundError:
        pass

    import job_store
    importlib.reload(job_store)
    import api
    importlib.reload(api)
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    headers = {"X-API-Key": "smoke-test-key"}
    pdf = pdf_path.read_bytes()

    # Health
    r = client.get("/healthz")
    check("GET /healthz → 200", r.status_code == 200)

    # Auth
    r = client.post("/api/v1/process")
    check("POST /process without key → 401", r.status_code == 401)
    r = client.post("/api/v1/process", headers={"X-API-Key": "wrong"})
    check("POST /process with wrong key → 401", r.status_code == 401)

    # Validation: bad parser
    r = client.post(
        "/api/v1/process",
        headers=headers,
        files={"file": ("test.pdf", pdf, "application/pdf")},
        data={"parser": "bogus"},
    )
    check("bad parser → 400", r.status_code == 400)

    # Validation: non-PDF upload
    r = client.post(
        "/api/v1/process",
        headers=headers,
        files={"file": ("nope.txt", b"hello world", "text/plain")},
        data={"parser": "pymupdf4llm"},
    )
    check("non-PDF upload → 400", r.status_code == 400)

    # Sync process — real run
    t0 = time.time()
    r = client.post(
        "/api/v1/process",
        headers=headers,
        files={"file": ("test.pdf", pdf, "application/pdf")},
        data={"parser": "pymupdf4llm", "use_llm": "false"},
    )
    elapsed = time.time() - t0
    check(f"sync /process → 200 ({elapsed:.0f}s)", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        body = r.json()
        check("sync response has summary block", "summary" in body)
        check(
            "sync response has refs list",
            isinstance(body.get("refs"), list) and len(body["refs"]) > 0,
        )
        check(
            "sync refs include filename field on top",
            body.get("filename") == "test.pdf",
        )

    # Batch
    r = client.post(
        "/api/v1/batch",
        headers=headers,
        files=[("files", ("a.pdf", pdf, "application/pdf"))],
        data={"parser": "pymupdf4llm", "use_llm": "false"},
    )
    check("batch submit → 200", r.status_code == 200, r.text[:200])
    job_id = (r.json() or {}).get("job_id")
    check("batch returns job_id", bool(job_id))

    if job_id:
        # Poll for completion — single PDF, should finish in <120s with the cache warmed by the sync call
        final = None
        for _ in range(60):
            r = client.get(f"/api/v1/batch/{job_id}", headers=headers)
            final = r.json()
            if final.get("status") in ("done", "failed"):
                break
            time.sleep(2)
        check(
            f"batch reaches done (final={final.get('status')})",
            final and final.get("status") == "done",
        )

        r = client.get(f"/api/v1/batch/{job_id}/results", headers=headers)
        check("batch /results → 200", r.status_code == 200)
        if r.status_code == 200:
            body = r.json()
            check(
                "batch /results has at least one row",
                isinstance(body.get("results"), list) and len(body["results"]) >= 1,
            )
            if body.get("results"):
                row = body["results"][0]
                check("per-PDF row has filename", row.get("filename") == "a.pdf")
                check("per-PDF row has 'ok' status", row.get("status") == "ok")
                check(
                    "per-PDF payload has summary",
                    isinstance(row.get("payload"), dict) and "summary" in row["payload"],
                )

        r = client.get(f"/api/v1/batch/{job_id}/results.jsonl", headers=headers)
        check("batch /results.jsonl → 200", r.status_code == 200)
        lines = [ln for ln in r.text.strip().split("\n") if ln.strip()]
        check(
            "JSONL has meta header + ≥1 row",
            len(lines) >= 2 and "_meta" in json.loads(lines[0]),
        )

        r = client.get("/api/v1/batches", headers=headers)
        check(
            "/api/v1/batches lists our job",
            r.status_code == 200
            and any(b["job_id"] == job_id for b in r.json().get("batches", [])),
        )

    # Auth on a polling endpoint too
    r = client.get(f"/api/v1/batch/{job_id or 'fake'}")
    check("GET /batch/{id} without key → 401", r.status_code == 401)

    # Wire-format: the CLI sends use_llm as "1" / "0".  Confirm Pydantic
    # accepts that spelling so the CLI ↔ API contract stays valid even
    # if FastAPI tightens form coercion later.
    r = client.post(
        "/api/v1/process",
        headers=headers,
        files={"file": ("tiny.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")},
        data={"parser": "pymupdf4llm", "use_llm": "1"},
    )
    # A corrupt PDF must be a clean 400 ("not a readable PDF"), never a
    # 500; a 422 would mean use_llm="1" failed form coercion.
    check(
        "use_llm='1' accepted; corrupt PDF → 400 (not 500)",
        r.status_code == 400,
        f"status={r.status_code}",
    )

    # Stale-job sweeper:
    #   1. Inject a fake 'running' job (simulates one stranded by a crash).
    #   2. Trigger the lifespan startup hook by entering a fresh TestClient.
    #   3. Confirm the fake job moved to 'failed' and the previously-completed
    #      job is still 'done' (sweeper only touches non-terminal rows).
    from job_store import get_job_store as _gjs
    _gjs().create("smoke-stranded", n_total=1, parser="grobid", use_llm=False)
    _gjs().mark_running("smoke-stranded")
    with TestClient(api.app) as _restarted:
        _restarted.get("/healthz")
    stranded = _gjs().get("smoke-stranded")
    check(
        "lifespan sweeper marks stranded job 'failed'",
        stranded and stranded["status"] == "failed",
        f"status={stranded and stranded.get('status')}",
    )
    if job_id:
        meta = _gjs().get(job_id)
        check(
            "completed jobs survive lifespan sweep unchanged",
            meta is not None and meta["status"] == "done",
            f"status after restart={meta and meta.get('status')}",
        )


# ---------------------------------------------------------------------------
# 5. Batch CLI (local mode)
# ---------------------------------------------------------------------------

def test_batch_cli(pdf_path: Path) -> None:
    section("Batch CLI (local mode)")

    with tempfile.TemporaryDirectory() as tmp:
        in_dir  = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()
        shutil.copy(pdf_path, in_dir / pdf_path.name)

        proc = subprocess.run(
            [
                sys.executable, "batch_cli.py",
                str(in_dir), str(out_dir),
                "--parser", "pymupdf4llm",
                "--concurrency", "1",
            ],
            capture_output=True, text=True, timeout=400,
        )
        check(
            f"CLI exits 0 (rc={proc.returncode})",
            proc.returncode == 0,
            proc.stderr[:200] or proc.stdout[-200:],
        )
        json_path    = out_dir / (pdf_path.stem + ".json")
        summary_path = out_dir / "summary.csv"

        check("CLI wrote per-PDF JSON", json_path.exists())
        check("CLI wrote summary.csv", summary_path.exists())

        if json_path.exists():
            report = json.loads(json_path.read_text())
            check(
                "CLI per-PDF JSON has summary + refs",
                "summary" in report and isinstance(report.get("refs"), list),
            )

        if summary_path.exists():
            csv_text = summary_path.read_text()
            check(
                "summary.csv has header + ≥1 data row",
                len(csv_text.strip().split("\n")) >= 2,
            )


# ---------------------------------------------------------------------------
# 6. Live URL verification (optional)
# ---------------------------------------------------------------------------

def test_url_verify_live() -> None:
    section("URL verification (live network)")
    from lookup import _lookup_url

    ref = {
        "title":   "Attention Is All You Need",
        "authors": "Vaswani; Shazeer; Parmar",
        "venue":   "NeurIPS",
        "url":     "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
        "raw":     "Vaswani et al. Attention Is All You Need. NeurIPS 2017.",
    }
    try:
        r = _lookup_url(ref)
    except Exception as exc:
        check("URL verify completes", False, str(exc))
        return

    check(
        "URL verify against papers.nips.cc → found",
        r.get("status") == "found",
        f"got status={r.get('status')}",
    )
    if r.get("status") == "found":
        check(
            "title sim ≥ 0.95 against real authors",
            (r.get("similarity") or 0) >= 0.95,
            f"tsim={r.get('similarity')}",
        )
        check(
            "label = match",
            r.get("label") == "match",
            f"label={r.get('label')}",
        )


def test_website_lookups_live() -> None:
    section("Website lookups (GitHub, dead link, SSRF guard)")
    from lookup import _lookup_url

    r = _lookup_url({"title": "Transformers: State-of-the-art natural language processing",
                     "url": "https://github.com/huggingface/transformers"})
    check("GitHub repo → found/match", r.get("status") == "found" and r.get("label") == "match",
          f"status={r.get('status')} label={r.get('label')}")
    r = _lookup_url({"title": "A tool", "url": "https://arse-smoke-test-no-such-host.invalid/x"})
    check("unresolvable host → dead_link", r.get("status") == "dead_link", f"status={r.get('status')}")
    r = _lookup_url({"title": "internal", "url": "http://169.254.169.254/latest/meta-data/"})
    check("metadata IP blocked", r.get("status") == "error" and "blocked" in (r.get("error") or ""),
          f"{r}")


def test_llm_repair_live(pdf_path: Path, skip_llm: bool) -> None:
    section("LLM reference repair (live, every configured backend)")
    if skip_llm:
        print("  [SKIP] no LLM key configured")
        return
    from parsers import extract_references_pymupdf4llm
    from llm_parser import parse_references_with_llm
    refs = extract_references_pymupdf4llm(str(pdf_path))[:12]
    check("parser produced refs for LLM test", len(refs) > 0, f"n={len(refs)}")
    backends = [(b, k) for b, k in (("anthropic", "ANTHROPIC_API_KEY"),
                                    ("aqueduct", "AQUEDUCT_API_KEY")) if os.getenv(k)]
    for backend, _ in backends:
        stats: dict = {}
        out = parse_references_with_llm(refs, stats=stats, backend=backend)
        check(f"{backend}: no failed batches", stats.get("n_failed_refs") == 0,
              f"{stats.get('model')}: {stats.get('first_error')}")
        check(f"{backend}: same length, titles kept", len(out) == len(refs)
              and sum(1 for r in out if r.get("title")) >= 0.8 * len(refs))


# ---------------------------------------------------------------------------
# 7. Claim verification (extractor / abstract / fulltext / verifier / e2e)
# ---------------------------------------------------------------------------

def test_claim_extractor() -> None:
    section("Claim extractor")
    from claim_extractor import extract

    refs = [
        {"authors": "Vaswani, A., Shazeer, N.", "year": "2017", "title": "Attention"},
        {"authors": "Brown, T., Mann, B.",    "year": "2020", "title": "GPT-3"},
        {"authors": "Devlin, J., Chang, M.",  "year": "2019", "title": "BERT"},
    ]

    body = "We build on transformers [1]. Prior LM work [2, 3] shows scaling."
    cs = extract(body, refs)
    check("numeric [1] and [2, 3] → 3 claims",
          len(cs) == 3 and {c.cited_ref_index for c in cs} == {0, 1, 2})

    body = "Scaling matters (Brown et al., 2020; Devlin et al., 2019)."
    cs = extract(body, refs)
    check("paren multi-cite → 2 claims", len(cs) == 2)

    body = "Vaswani et al. (2017) introduced the transformer."
    cs = extract(body, refs)
    check("narrative form → 1 claim, ref 0",
          len(cs) == 1 and cs[0].cited_ref_index == 0)

    # No refs — no claims regardless of body
    check("no refs → no claims", extract("Anything with [1] cited.", []) == [])


def test_openalex_abstract_reconstruct() -> None:
    section("OpenAlex abstract inverted-index reconstruct")
    from lookup import _oa_reconstruct_abstract

    inv = {"attention": [0, 5], "is": [1], "all": [2], "you": [3], "need": [4]}
    r = _oa_reconstruct_abstract(inv)
    check("reconstructed positions in order", r == "attention is all you need attention",
          f"got {r!r}")
    check("None input → empty", _oa_reconstruct_abstract(None) == "")
    check("empty dict → empty",  _oa_reconstruct_abstract({}) == "")
    check("malformed positions ignored",
          _oa_reconstruct_abstract({"foo": "not-a-list"}) == "")


def test_fulltext_live() -> None:
    section("Full-text fetcher (live arxiv + abstract fallback)")
    from fulltext import fetch

    # arxiv path
    lookup_sources = {"arxiv": {"status": "found", "arxiv_id": "1706.03762",
                                "arxiv_url": "https://arxiv.org/abs/1706.03762"}}
    ft = fetch(lookup_sources)
    check(
        "arxiv fetch → non-empty full text",
        ft is not None and not ft.is_abstract_fallback
        and len(ft.text) > 5000 and "attention" in ft.text.lower(),
        f"got source={ft.source if ft else None} len={len(ft.text) if ft else 0}",
    )

    # abstract fallback path
    lookup_sources = {"semantic_scholar":
                      {"status": "found",
                       "abstract": "This paper introduces X and shows Y."}}
    ft2 = fetch(lookup_sources)
    check(
        "abstract fallback",
        ft2 is not None and ft2.is_abstract_fallback and "X" in ft2.text,
    )

    # nothing at all → None
    check("no sources → None", fetch({}) is None)


def test_claim_verifier_live(skip_llm: bool) -> None:
    section("Claim verifier (live LLM if configured)")
    from claim_extractor import Claim
    from fulltext         import FullText
    from claim_verifier   import verify, _parse_llm_response, _sanitize_verdict

    # Unit: response parser + sanitizer (no LLM needed)
    # The verifier batches claims, so the response is a JSON array.
    parsed = _parse_llm_response(
        '[{"verdict":"supported","evidence_passage":"q","confidence":"high","explanation":"e"}]')
    check("parser: strict JSON", parsed and parsed[0].get("verdict") == "supported")

    fenced = ('```json\n[{"verdict":"contradicted","evidence_passage":"…",'
              '"confidence":"medium","explanation":"…"}]\n```')
    parsed2 = _parse_llm_response(fenced)
    check("parser: fenced JSON", parsed2 and parsed2[0].get("verdict") == "contradicted")

    bad = _sanitize_verdict({"verdict": "bogus", "confidence": "super",
                             "evidence_passage": "", "explanation": ""})
    check("sanitizer: bad enums → ambiguous / low",
          bad["verdict"] == "ambiguous" and bad["confidence"] == "low")

    if skip_llm:
        return

    # Live: does the paper support the claim?
    claim = Claim(
        text="The transformer relies solely on attention without recurrence.",
        citation_marker="(Vaswani et al., 2017)",
        cited_ref_index=0, style="paren")
    ft = FullText(
        text=("We propose a new simple network architecture, the Transformer, "
              "based solely on attention mechanisms, dispensing with "
              "recurrence and convolutions entirely."),
        source="abstract_s2", is_abstract_fallback=True, url=None)
    try:
        v = verify(claim, ft, claim_index=0)
    except Exception as exc:
        check("live verifier completes", False, str(exc)[:120])
        return
    check("live verifier returns a recognised verdict",
          v.verdict in ("supported", "contradicted", "not_addressed",
                        "ambiguous", "error"),
          f"got {v.verdict}")
    if v.verdict == "supported":
        check("live verifier produced an evidence quote",
              bool(v.evidence_passage))


def test_pipeline_verify_claims(pdf_path: Path, skip_llm: bool) -> None:
    section("Pipeline: verify_claims=True (end-to-end)")
    if skip_llm:
        check("skipped — LLM backend not configured", True)
        return
    from pipeline import PipelineOptions, process_pdf
    try:
        report = process_pdf(
            pdf_path.read_bytes(),
            PipelineOptions(parser="pymupdf4llm", use_llm=False,
                            verify_claims=True),
        )
    except Exception as exc:
        check("pipeline runs", False, str(exc)[:200])
        return
    check("pipeline returns claim_verifications",
          hasattr(report, "claim_verifications"))
    check(f"n_claims > 0 (was {report.n_claims})", report.n_claims > 0)
    check(
        "n_claims == sum of per-verdict counts",
        report.n_claims == (report.n_supported + report.n_contradicted
                            + report.n_not_addressed + report.n_ambiguous
                            + report.n_verify_errors + report.n_verify_no_source),
    )
    check("report.to_dict() includes claim_verifications",
          isinstance(report.to_dict().get("claim_verifications"), list))
    # Runtime is dominated by LLM latency (5-30s per claim × ~50 claims);
    # 60 min is a soft ceiling that catches genuine hangs without flagging
    # normal Qwen/Aqueduct throughput as a failure.
    per_claim = report.verify_claims_seconds / max(1, report.n_claims)
    check(
        f"verify runtime under 60 min (was {report.verify_claims_seconds:.0f}s, "
        f"{per_claim:.1f}s/claim)",
        report.verify_claims_seconds < 3600,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pdf",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "benchmark" / "pdfs" / "2024.acl-long.560.pdf",
        help="Test PDF (default: benchmark/pdfs/2024.acl-long.560.pdf)",
    )
    p.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip tests that hit real external services.",
    )
    p.add_argument(
        "--skip-cli",
        action="store_true",
        help="Skip the batch CLI subprocess (~minute, requires running pipeline).",
    )
    p.add_argument(
        "--skip-verify-claims",
        action="store_true",
        help=(
            "Skip the pipeline-level claim-verification end-to-end test "
            "(slow: extracts inline citations, fetches cited papers, "
            "LLM-verdicts each claim — several minutes)."
        ),
    )
    args = p.parse_args()

    if not args.pdf.exists():
        print(f"Test PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    # Skip live LLM tests if no backend is configured — the check runs
    # regardless of --skip-network because it's cheap.
    from dotenv import load_dotenv as _load
    _load()
    llm_configured = bool(os.getenv("ANTHROPIC_API_KEY")
                          or os.getenv("AQUEDUCT_API_KEY"))

    test_finalize_match()
    test_html_meta_parser()
    test_claim_extractor()
    test_openalex_abstract_reconstruct()
    test_pipeline_local(args.pdf)
    test_api(args.pdf)
    if not args.skip_cli:
        test_batch_cli(args.pdf)
    if not args.skip_network:
        test_url_verify_live()
        test_website_lookups_live()
        test_llm_repair_live(args.pdf, skip_llm=not llm_configured)
        test_fulltext_live()
        test_claim_verifier_live(skip_llm=not llm_configured)
        if not args.skip_verify_claims:
            test_pipeline_verify_claims(args.pdf, skip_llm=not llm_configured)

    print()
    print("=" * 60)
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"  Smoke test summary: {n_pass} passed, {n_fail} failed")
    print("=" * 60)
    if n_fail:
        print("\nFailed checks:")
        for name, s, detail in RESULTS:
            if s == "FAIL":
                print(f"  - {name}  {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
