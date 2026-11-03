"""
Smoke test on N publications from the HalluCitation benchmark.

For each host paper:
  1. Extract refs with pymupdf4llm.
  2. Find which extracted ref corresponds to the labelled hallucination
     (title fuzzy-match against the bibliographic string in Table 7).
  3. Run lookup_all on the extracted refs.
  4. Report whether our pipeline flags the labelled ref as anything other than
     "match" (i.e., "missing" or "mismatch" → caught).

Usage:
    python benchmark/smoke_test.py --n 5
    python benchmark/smoke_test.py --ids 1,113,295
"""

from __future__ import annotations

import os as _os
for _k, _v in list(_os.environ.items()):   # old ARES_* settings
    if _k.startswith("ARES_"):
        _os.environ.setdefault("ARSE_" + _k[5:], _v)

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
PROJECT  = ROOT.parent
sys.path.insert(0, str(PROJECT))

from rapidfuzz import fuzz

from parsers import (
    extract_references_pymupdf4llm,
    extract_references_grobid,
    extract_references_hybrid,
    grobid_is_available,
    _parse_raw_reference,
)
from lookup import (
    lookup_all, reset_http_counts,
    _extract_doi, _cited_arxiv_id,
)
from report import (
    _overall_status,            # match / missing / mismatch / pending
    _categorise_suspicious,     # wrong_arxiv_id / title_authors_mismatch / …
    SUSPICIOUS_CATEGORIES,
)

try:
    from llm_parser import parse_references_with_llm, set_active_model, active_model
    _LLM_AVAILABLE = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("AQUEDUCT_API_KEY"))
except Exception:
    _LLM_AVAILABLE = False
    def set_active_model(*a, **k): return ("", "")
    def active_model(): return ("", "")

DATASET = ROOT / "hallucitation_table7.json"
PDF_DIR = ROOT / "pdfs"


def _extract_title(citation_text: str) -> str:
    """Reuse the parser's heuristic to pull a title out of a free-text citation."""
    parsed = _parse_raw_reference(citation_text)
    return parsed.get("title", "") or ""


def _find_target_ref(refs: list[dict], target_title: str, target_text: str) -> int | None:
    """Locate the index of the ref that matches the labelled hallucination.

    Uses the project's own `lookup.title_similarity` (which has the
    containment boost, subtitle handling, and normalisation we tuned for
    the rest of the system) and requires a strong title-level match.

    Crucially: NO raw-text fallback.  An earlier version used
    `fuzz.partial_ratio` on the full raw text as a secondary signal,
    but partial_ratio is generous — it found 0.70+ matches against
    completely-different real citations (e.g. Plate's "Holographic
    Reduced Representations" was picked as the supposed hallucination
    in paper 178 because partial_ratio matched on stop words and venue
    fragments).  The result: grobid was credited with parser-recall it
    didn't earn.

    If the labelled hallucination's title can't be matched to any
    extracted ref's title at ≥ 0.85, we report `None` — the parser
    truly missed it.  That excludes the case from the labelled-recall
    denominator and surfaces parser failures as a distinct metric.
    """
    if not refs or not target_title:
        return None

    from lookup import title_similarity

    best_i, best_score = None, 0.0
    for i, ref in enumerate(refs):
        ref_title = (ref.get("title") or "").strip()
        if not ref_title:
            continue
        s = title_similarity(ref_title, target_title)
        if s > best_score:
            best_score, best_i = s, i

    return best_i if best_score >= 0.85 else None


def _run_one(entry: dict, scholarly_budget: int = 0, parser: str = "pymupdf4llm",
             use_llm: bool = True, save_intermediates: bool = False) -> dict:
    pid = entry["paper_id"]
    pdf = PDF_DIR / f"{pid}.pdf"
    out = {"event_id": entry["event_id"], "paper_id": pid,
           "citation_key": entry["citation_key"]}

    if not pdf.exists():
        out["error"] = "pdf missing"
        return out

    t_paper = time.time()
    t0 = time.time()
    try:
        if parser == "grobid":
            refs = extract_references_grobid(str(pdf))
        elif parser == "hybrid":
            refs = extract_references_hybrid(str(pdf))
        else:
            refs = extract_references_pymupdf4llm(str(pdf))
    except Exception as e:
        out["error"] = f"parse fail: {e}"
        return out
    out["parse_s"]  = round(time.time() - t0, 1)
    out["n_refs"]   = len(refs)
    out["n_corrupted"] = sum(1 for r in refs if r.get("_corruption"))

    # Snapshot the raw parser output BEFORE LLM rewrites the fields in
    # place — needed for stage-by-stage error analysis.
    parser_refs_snapshot = [dict(r) for r in refs] if save_intermediates else None

    if use_llm and _LLM_AVAILABLE:
        t_llm = time.time()
        try:
            refs = parse_references_with_llm(refs)
        except Exception as e:
            out["llm_err"] = str(e)
        out["llm_repair_s"] = round(time.time() - t_llm, 1)
        out["n_non_citation"] = sum(1 for r in refs if r.get("_not_a_citation"))

    target_title = _extract_title(entry["hallucinated_citation"])
    target_idx = _find_target_ref(refs, target_title, entry["hallucinated_citation"])
    out["target_found_in_extract"] = target_idx is not None
    out["extracted_target_title"]  = (refs[target_idx].get("title", "")[:80]
                                       if target_idx is not None else "")
    out["labelled_target_title"]   = target_title[:80]

    # Cited-identifier distribution: for each ref, classify the
    # strongest identifier *the user actually cited* (not what the LLM
    # might have hallucinated into url/doi).  Lets us explain doi.org's
    # vs arxiv's vs title-search's share of the routing.
    id_dist = {"doi": 0, "arxiv": 0, "title_only": 0}
    for r in refs:
        if _cited_arxiv_id(r):
            id_dist["arxiv"] += 1
        elif _extract_doi(r):
            id_dist["doi"] += 1
        else:
            id_dist["title_only"] += 1
    out["cited_id_distribution"] = id_dist

    reset_http_counts()
    t1 = time.time()
    lookup_results = lookup_all(refs, scholarly_budget=scholarly_budget)
    out["lookup_s"] = round(time.time() - t1, 1)
    # Per-paper HTTP call breakdown for cost-at-scale analysis.
    out["http_calls"] = reset_http_counts()
    out["http_calls_total"] = sum(out["http_calls"].values())

    # Per-ref status counts (precision proxy)
    statuses = [_overall_status(lr) for lr in lookup_results]
    out["status_counts"] = {s: statuses.count(s) for s in
                            ("match", "missing", "mismatch", "pending")}

    # Per-ref suspicious-reason taxonomy.  None for clean refs; a
    # category tag (wrong_arxiv_id / title_authors_mismatch / …) for
    # flagged refs.  Aggregated into the summary table later.
    categories = [_categorise_suspicious(lr) for lr in lookup_results]
    out["suspicious_category_counts"] = {
        c: categories.count(c) for c in SUSPICIOUS_CATEGORIES
        if categories.count(c) > 0
    }
    # The labelled hallucination's category (alongside its status):
    if target_idx is not None:
        out["labelled_category"] = categories[target_idx]
    else:
        out["labelled_category"] = None

    # The labelled hallucination's status (recall on this single case)
    if target_idx is not None:
        out["labelled_status"] = _overall_status(lookup_results[target_idx])
        out["caught"] = out["labelled_status"] in ("missing", "mismatch")
    else:
        out["labelled_status"] = "n/a"
        out["caught"] = None

    out["paper_wall_s"] = round(time.time() - t_paper, 2)

    if save_intermediates:
        # Attach full per-stage state for offline analysis.  Main
        # caller writes this to a separate file per paper so the
        # top-level results JSON stays compact.
        out["_intermediates"] = {
            "parser_refs":     parser_refs_snapshot,
            "llm_refs":        [dict(r) for r in refs],
            "lookup_results":  lookup_results,
            "target_idx":      target_idx,
            "target_title":    target_title,
            "labelled_entry":  entry,   # the gold-label row from Table 7
        }
    return out


def _pct(p: float, q: float) -> float:
    return 100.0 * p / q if q else 0.0


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95%-CI for a binomial proportion k/n.  Honest small-sample CI."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    spread = z * (phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5
    return ((centre - spread) / denom, (centre + spread) / denom)


def _summarise_times(values: list[float]) -> dict[str, float]:
    """Median, mean, p95 — meaningful at any sample size; mean alone is skewed."""
    if not values:
        return {}
    vs = sorted(values)
    n = len(vs)
    def _q(p: float) -> float:
        # Linear-interpolation quantile
        if n == 1:
            return vs[0]
        k = p * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        return vs[f] + (vs[c] - vs[f]) * (k - f)
    return {
        "n":      n,
        "mean":   round(sum(vs) / n, 2),
        "median": round(_q(0.50), 2),
        "p95":    round(_q(0.95), 2),
        "max":    round(vs[-1], 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5,
                    help="Sample N papers spread across the dataset. Ignored if --all.")
    ap.add_argument("--all", action="store_true",
                    help="Run the full 295-paper corpus (overrides --n).")
    ap.add_argument("--ids", default="", help="Comma-separated event_ids")
    ap.add_argument("--scholarly", type=int, default=0,
                    help="Scholar lookups per paper (default 0 — off for batch). "
                         "Scholar is rate-limited and dominates wall time at scale.")
    ap.add_argument("--parser", choices=("pymupdf4llm", "grobid", "hybrid"), default="hybrid",
                    help="PDF parser backend. `grobid` uses GROBID's full-doc layout "
                         "analysis. `hybrid` (default) uses pymupdf to isolate the "
                         "References section then sends each candidate citation "
                         "through GROBID's /api/processCitationList — recovers refs "
                         "GROBID's layout analysis skips. Both require GROBID running.")
    ap.add_argument("--llm", dest="llm", action="store_true", default=None,
                    help="Force LLM repair on (overrides auto-detect).")
    ap.add_argument("--no-llm", dest="llm", action="store_false",
                    help="Force LLM repair off even if an API key is set.")
    ap.add_argument("--llm-backend", default="",
                    help="Override the LLM backend for this run "
                         "(anthropic | aqueduct).  Defaults to the env-var value.")
    ap.add_argument("--llm-model", default="",
                    help="Override the LLM model for this run "
                         "(e.g. qwen-3.8-27b, claude-haiku-5-5).  "
                         "Use this to compare models in side-by-side benchmark runs.")
    ap.add_argument("--cold", action="store_true",
                    help="Use a fresh empty SQLite cache for this run (ARSE_CACHE "
                         "redirected to a temp file).  Measures the realistic first-time "
                         "experience; warm-cache runs piggyback on previous lookups and "
                         "report misleadingly fast numbers.")
    ap.add_argument("--save-intermediates", action="store_true",
                    help="Dump full per-stage state (raw parser refs, LLM-repaired "
                         "refs, complete lookup_results dict) per paper into a "
                         "<out>_intermediates/ subdir.  Adds ~50-200 KB per paper "
                         "to disk but enables stage-by-stage error analysis.")
    ap.add_argument("--out", default="",
                    help="Output JSON path.  Auto-named per config when omitted.")
    args = ap.parse_args()

    # Cold-cache mode: redirect cache to a temp file, deleted on exit.
    # Must happen BEFORE any import of `lookup`/`cache` evaluates
    # `CACHE_PATH` — luckily we deferred those imports to module level,
    # so re-importing isn't safe; instead set env var before run.
    if args.cold:
        import tempfile
        cold_dir  = tempfile.mkdtemp(prefix="refcleaner_cold_")
        cold_path = Path(cold_dir) / "cache.sqlite"
        os.environ["ARSE_CACHE"] = str(cold_path)
        # Force a fresh cache singleton with the new path
        import cache as _cache_mod
        _cache_mod._cache_singleton = None
        _cache_mod.CACHE_PATH = cold_path
        print(f"❄️  Cold mode: cache → {cold_path}")

    if args.parser == "grobid" and not grobid_is_available():
        print("⚠️ GROBID not reachable at http://localhost:8070 — falling back "
              "to pymupdf4llm. Run `docker compose up -d grobid` to enable.")
        args.parser = "pymupdf4llm"

    # LLM toggle: explicit --llm / --no-llm wins; otherwise auto-on if key.
    if args.llm is None:
        use_llm = _LLM_AVAILABLE
    else:
        use_llm = bool(args.llm) and _LLM_AVAILABLE
        if args.llm and not _LLM_AVAILABLE:
            print("⚠️ --llm requested but no LLM API key set "
                  "(ANTHROPIC_API_KEY / AQUEDUCT_API_KEY). Running without LLM repair.")

    # Per-run LLM override.  Lets the harness do qwen-35B vs gemma-2B
    # comparisons in a single sequence.  Must happen BEFORE any
    # parse_references_with_llm() call.
    if use_llm and (args.llm_backend or args.llm_model):
        active_be, active_md = set_active_model(args.llm_backend or None,
                                                args.llm_model or None)
        print(f"🔧 LLM override active: backend={active_be} model={active_md}")
    elif use_llm:
        active_be, active_md = active_model()
        print(f"🔧 LLM active: backend={active_be} model={active_md}")

    entries = json.loads(DATASET.read_text())
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",")}
        sample = [e for e in entries if e["event_id"] in wanted]
    elif args.all:
        sample = list(entries)
    else:
        # Spread across the dataset
        n = args.n
        step = max(1, len(entries) // n)
        sample = [entries[i * step] for i in range(n)]

    # Include the LLM model in the tag when explicitly overridden so
    # qwen-vs-gemma runs don't clobber each other's outputs.  Sanitise
    # for path safety.
    def _slug(s: str) -> str:
        return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)
    llm_tag = ""
    if use_llm:
        if args.llm_model:
            llm_tag = f"_llm-{_slug(args.llm_model)}"
        else:
            llm_tag = "_llm"
    config_tag = (f"{args.parser}"
                  f"{llm_tag}"
                  f"{'_gs' + str(args.scholarly) if args.scholarly else ''}")
    print(f"Config: parser={args.parser} llm={use_llm} "
          f"scholarly={args.scholarly}  → tag={config_tag}")
    print(f"Running on {len(sample)} host papers\n")

    out_path = Path(args.out) if args.out else (ROOT / f"results_{config_tag}.json")
    inter_dir: Path | None = None
    if args.save_intermediates:
        inter_dir = out_path.parent / f"{out_path.stem}_intermediates"
        inter_dir.mkdir(parents=True, exist_ok=True)
        print(f"📦 Per-paper intermediates → {inter_dir}")

    results = []
    t_total = time.time()
    for i, entry in enumerate(sample, 1):
        print(f"--- [{i}/{len(sample)}] #{entry['event_id']}: {entry['paper_id']} ---")
        r = _run_one(entry, scholarly_budget=args.scholarly,
                     parser=args.parser, use_llm=use_llm,
                     save_intermediates=args.save_intermediates)
        # Persist intermediates to a side file per paper, then drop the
        # heavy payload from the in-memory result before appending so
        # the main results.json stays compact.
        if inter_dir is not None and "_intermediates" in r:
            inter_path = inter_dir / f"{r['paper_id']}.json"
            inter_path.write_text(
                json.dumps(r["_intermediates"], indent=2,
                           ensure_ascii=False, default=str))
            r.pop("_intermediates", None)
        results.append(r)
        print(f"  refs={r.get('n_refs','-'):>3} "
              f"target_found={'Y' if r.get('target_found_in_extract') else 'N'} "
              f"labelled_status={r.get('labelled_status','-'):<8} "
              f"caught={r.get('caught','-')!s:<5} "
              f"parse={r.get('parse_s','-')}s "
              f"lookup={r.get('lookup_s','-')}s")

    elapsed = time.time() - t_total
    print()
    print("=" * 70)
    print(f"SUMMARY — config={config_tag}  cache={'cold' if args.cold else 'warm'}  "
          f"({elapsed/60:.1f} min)")
    print("=" * 70)
    located    = [r for r in results if r.get("target_found_in_extract")]
    caught     = [r for r in results if r.get("caught") is True]
    parse_fail = [r for r in results if "error" in r]

    # Recall decomposition — separate parser-recall (did we even extract
    # the hallucinated ref from the PDF?) from lookup-recall (given we
    # found it, did the database vote flag it?).  Reviewers expect both
    # broken out; the joint number alone hides which stage is the
    # bottleneck.
    n_papers = len(results)
    n_loc    = len(located)        # parser FOUND the labelled hallucination
    n_caught = len(caught)         # ...AND the lookup flagged it
    parser_lo, parser_hi   = _wilson_ci(n_loc,    n_papers)
    cond_lo,   cond_hi     = _wilson_ci(n_caught, n_loc)
    joint_lo,  joint_hi    = _wilson_ci(n_caught, n_papers)

    print(f"Papers attempted:              {n_papers}")
    print(f"Parse failures:                {len(parse_fail)}")
    print()
    print(f"RECALL DECOMPOSITION (Wilson 95% CI):")
    print(f"  Parser-recall:    {n_loc}/{n_papers}  = "
          f"{_pct(n_loc, n_papers):.1f}% ({parser_lo*100:.1f}–{parser_hi*100:.1f}%)")
    print(f"  Lookup-recall:    {n_caught}/{n_loc} = "
          f"{_pct(n_caught, n_loc):.1f}% ({cond_lo*100:.1f}–{cond_hi*100:.1f}%) "
          f"  (conditional on parser-recall)")
    print(f"  Pipeline-recall:  {n_caught}/{n_papers}  = "
          f"{_pct(n_caught, n_papers):.1f}% ({joint_lo*100:.1f}–{joint_hi*100:.1f}%) "
          f"  (joint, end-to-end)")
    print(f"  ⚠️  Lower bound — HalluCitation labels one hallucination per paper.")
    # Backward-compat: keep the old label too in case anything parses it.
    lo, hi = cond_lo, cond_hi

    # Aggregate status breakdown across ALL refs
    agg = {"match": 0, "missing": 0, "mismatch": 0, "pending": 0}
    for r in results:
        for k, v in (r.get("status_counts") or {}).items():
            agg[k] = agg.get(k, 0) + v
    total_refs = sum(agg.values())
    flagged = agg.get("missing", 0) + agg.get("mismatch", 0)
    print(f"\nAcross all {total_refs} refs:")
    for k, v in agg.items():
        print(f"  {k:<9} {v:>6}  ({_pct(v, total_refs):5.1f}%)")
    print(f"\nAggregate suspicious rate: {flagged}/{total_refs} = "
          f"{_pct(flagged, total_refs):.1f}% of refs")
    print(f"  ⚠️  Not a precision/recall — mix of true positives the dataset")
    print(f"      didn't label AND our false positives.  Validate manually on a")
    print(f"      random sample of flagged refs to get a real precision number.")

    # Breakdown of WHY each flagged ref was flagged.  This is the failure-
    # mode taxonomy reviewers will ask for.
    cat_agg: dict[str, int] = {c: 0 for c in SUSPICIOUS_CATEGORIES}
    for r in results:
        for k, v in (r.get("suspicious_category_counts") or {}).items():
            cat_agg[k] = cat_agg.get(k, 0) + v
    cat_total = sum(cat_agg.values())
    print(f"\nSuspicious-ref breakdown by failure mode ({cat_total} flagged refs):")
    for cat in SUSPICIOUS_CATEGORIES:
        n = cat_agg.get(cat, 0)
        if not n:
            continue
        print(f"  {cat:<24} {n:>5}  ({_pct(n, cat_total):5.1f}% of flagged)")

    # Same breakdown but restricted to the LABELLED hallucinations —
    # tells us which failure modes our gold-label cases tend to fall
    # into.  Useful for the paper's discussion section.
    labelled_cats: dict[str, int] = {}
    for r in results:
        if r.get("caught") and r.get("labelled_category"):
            lc = r["labelled_category"]
            labelled_cats[lc] = labelled_cats.get(lc, 0) + 1
    if labelled_cats:
        print(f"\nHow we caught the LABELLED hallucinations ({sum(labelled_cats.values())} of {len(caught)}):")
        for cat in SUSPICIOUS_CATEGORIES:
            n = labelled_cats.get(cat, 0)
            if not n:
                continue
            print(f"  {cat:<24} {n:>5}")

    # HTTP cost summary — useful for "what does this cost to run at scale".
    http_agg: dict[str, int] = {}
    for r in results:
        for host, n in (r.get("http_calls") or {}).items():
            http_agg[host] = http_agg.get(host, 0) + n
    if http_agg:
        total_http = sum(http_agg.values())
        print(f"\nHTTP calls during lookup (cold cache, total {total_http}; "
              f"avg {total_http/max(1,n_papers):.0f}/paper):")
        for host, n in sorted(http_agg.items(), key=lambda kv: -kv[1]):
            print(f"  {host:<32} {n:>6}")

    # Cited-identifier distribution — explains routing share.  A corpus
    # heavily skewed to arxiv-ID-bearing refs will see ~0 doi.org calls
    # because the router goes ["arxiv"] only.
    id_agg = {"doi": 0, "arxiv": 0, "title_only": 0}
    for r in results:
        for k, v in (r.get("cited_id_distribution") or {}).items():
            id_agg[k] = id_agg.get(k, 0) + v
    id_total = sum(id_agg.values())
    if id_total:
        print(f"\nCited-identifier distribution across {id_total} refs:")
        print(f"  arXiv-ID:       {id_agg['arxiv']:>5}  ({_pct(id_agg['arxiv'], id_total):5.1f}%)"
              f"   → routes to ['arxiv'] only (skips doi.org)")
        print(f"  DOI:            {id_agg['doi']:>5}  ({_pct(id_agg['doi'], id_total):5.1f}%)"
              f"   → routes to ['doi_org', 'crossref', 's2', 'oa']")
        print(f"  Title-only:     {id_agg['title_only']:>5}  ({_pct(id_agg['title_only'], id_total):5.1f}%)"
              f"   → fanout title search across academic sources")

    # Timing — median + p95 + max, not just mean (which is skewed by outliers).
    def _times(key):
        return [r[key] for r in results if isinstance(r.get(key), (int, float))]
    timing = {
        "parse_s":      _summarise_times(_times("parse_s")),
        "llm_repair_s": _summarise_times(_times("llm_repair_s")),
        "lookup_s":     _summarise_times(_times("lookup_s")),
        "paper_wall_s": _summarise_times(_times("paper_wall_s")),
    }
    print(f"\nTiming (seconds, all stages):")
    print(f"  stage         n    mean   median    p95    max")
    for stage, t in timing.items():
        if not t:
            continue
        print(f"  {stage:<13} {t['n']:>4}  {t['mean']:>5.1f}   {t['median']:>5.1f}  "
              f"{t['p95']:>5.1f}  {t['max']:>5.1f}")

    _final_be, _final_md = active_model() if use_llm else ("", "")
    out_path.write_text(json.dumps({
        "config": {"parser": args.parser, "use_llm": use_llm,
                   "llm_backend": _final_be, "llm_model": _final_md,
                   "scholarly_budget": args.scholarly, "cold_cache": args.cold,
                   "save_intermediates": args.save_intermediates,
                   "intermediates_dir": (str(inter_dir) if inter_dir else None),
                   "n": len(sample)},
        "elapsed_sec": round(elapsed, 1),
        "summary": {
            "papers": len(results),
            "parse_failures": len(parse_fail),
            "hallucination_located": len(located),
            "caught": len(caught),
            "parser_recall_pct":   round(_pct(n_loc, n_papers), 2),
            "parser_recall_ci95":  [round(parser_lo*100,2), round(parser_hi*100,2)],
            "lookup_recall_pct":   round(_pct(n_caught, n_loc), 2),
            "lookup_recall_ci95":  [round(cond_lo*100,2), round(cond_hi*100,2)],
            "pipeline_recall_pct": round(_pct(n_caught, n_papers), 2),
            "pipeline_recall_ci95":[round(joint_lo*100,2), round(joint_hi*100,2)],
            "labelled_recall_pct": round(_pct(n_caught, n_loc), 2),     # alias for back-compat
            "labelled_recall_ci95": [round(lo*100,2), round(hi*100,2)],  # alias for back-compat
            "agg_status_counts": agg,
            "aggregate_suspicious_pct": round(_pct(flagged, total_refs), 2),
            "suspicious_breakdown": {c: cat_agg.get(c, 0)
                                     for c in SUSPICIOUS_CATEGORIES
                                     if cat_agg.get(c, 0) > 0},
            "labelled_caught_by_category": labelled_cats,
            "http_calls_by_host":  http_agg,
            "http_calls_total":    sum(http_agg.values()),
            "http_calls_per_paper":(round(sum(http_agg.values())/max(1,n_papers),1)),
            "cited_id_distribution": id_agg,
            "timing": timing,
        },
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nWritten {out_path}")

    # CSV companion — one row per paper, easy to plot or paste into a paper.
    import csv
    csv_path = out_path.with_suffix(".csv")
    fields = ["event_id", "paper_id", "n_refs", "target_found_in_extract",
              "labelled_status", "labelled_category", "caught",
              "parse_s", "llm_repair_s", "lookup_s", "paper_wall_s",
              "n_match", "n_missing", "n_mismatch",
              "n_refs_with_doi", "n_refs_with_arxiv_id", "n_refs_title_only",
              "n_wrong_arxiv_id", "n_wrong_doi",
              "n_title_authors_mismatch", "n_uncertain_match",
              "n_weak_title_match", "n_not_in_any_index", "error"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            sc  = r.get("status_counts") or {}
            cat = r.get("suspicious_category_counts") or {}
            counted_cols = {
                "n_match", "n_missing", "n_mismatch",
                "n_refs_with_doi", "n_refs_with_arxiv_id", "n_refs_title_only",
                "n_wrong_arxiv_id", "n_wrong_doi",
                "n_title_authors_mismatch", "n_uncertain_match",
                "n_weak_title_match", "n_not_in_any_index",
            }
            id_dist = r.get("cited_id_distribution") or {}
            w.writerow({
                **{k: r.get(k) for k in fields if k not in counted_cols},
                "n_match":    sc.get("match", 0),
                "n_missing":  sc.get("missing", 0),
                "n_mismatch": sc.get("mismatch", 0),
                "n_refs_with_doi":          id_dist.get("doi", 0),
                "n_refs_with_arxiv_id":     id_dist.get("arxiv", 0),
                "n_refs_title_only":        id_dist.get("title_only", 0),
                "n_wrong_arxiv_id":         cat.get("wrong_arxiv_id", 0),
                "n_wrong_doi":              cat.get("wrong_doi", 0),
                "n_title_authors_mismatch": cat.get("title_authors_mismatch", 0),
                "n_uncertain_match":        cat.get("uncertain_match", 0),
                "n_weak_title_match":       cat.get("weak_title_match", 0),
                "n_not_in_any_index":       cat.get("not_in_any_index", 0),
            })
    print(f"Written {csv_path}")


if __name__ == "__main__":
    main()
