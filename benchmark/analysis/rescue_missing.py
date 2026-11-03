"""
Re-lookup refs that the main pipeline flagged as `missing`, using two
broader-coverage sources in parallel:
  - **CORE** (api.core.ac.uk): open-access aggregator covering preprints,
    institutional repositories, theses, grey literature.  Requires
    CORE_API_KEY in env.  Free tier: 10 req/min.
  - **Google Scholar** (via the `scholarly` library): broadest coverage
    of all (TCT blog posts, books, workshop papers), but no API and
    will CAPTCHA-block after ~50-150 successful calls.

For every ref where `_overall_status(lookup_results) == "missing"` in any
of the configs under the given run directory, we dedupe by SQLite
`cache_key` and fire BOTH sources concurrently.  Results are written
incrementally to a JSON file so the run can be killed and resumed (the
SQLite cache + the resume logic handle both the API-cost and CAPTCHA
cases).

Usage:
    python benchmark/analysis/rescue_missing.py \\
        --from   benchmark/runs/latest \\
        --out    benchmark/runs/latest/scholar_core_rescue.json \\
        --limit  0           # 0 = all unique refs; e.g. 200 = pilot

Resume after interruption: pass the same --out path; refs already in
the file are skipped.

Output schema (one entry per unique ref):
    {
      "cache_key":         "...",
      "configs_missed_in": ["grobid", "hybrid", "hybrid_llm_qwen-3_6-35b"],
      "n_papers_affected": 3,
      "cited_title":   "...",
      "cited_authors": "...",
      "core":          { ... result schema ... | None },
      "scholarly":     { ... result schema ... | None },
      "rescued_by":    ["core" | "scholarly"],       # match/fuzzy only
      "new_status":    "match" | "mismatch" | "still_missing",
      "elapsed_s":     1.6,
    }

Then a summary block at the end with per-config rescue rates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from cache import cache_key as _cache_key
from lookup import (
    _lookup_core, _lookup_scholarly, ScholarBudget,
)
import lookup as _lookup_mod
from report import _overall_status


def _config_label_from_dir(name: str) -> str:
    """results_<cfg>_cold_intermediates → <cfg>"""
    n = name.replace("results_", "")
    for suffix in ("_cold_intermediates", "_intermediates", "_TAIL"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.rstrip("_")


def collect_missing(rundir: Path) -> dict[str, dict]:
    """
    Walk every intermediates directory in `rundir` (one per config) and
    collect refs whose `_overall_status` is "missing".  Dedupe by
    SQLite cache_key so identical refs across configs share one
    rescue call.

    Returns: cache_key → {
        "ref":              the ref dict,
        "configs_missed_in": set of config labels,
        "n_papers_affected": count of (paper_id, ref_index) pairs,
    }
    """
    by_key: dict[str, dict] = {}
    inter_dirs = list(rundir.glob("results_*_intermediates"))
    print(f"Scanning {len(inter_dirs)} intermediate directories:")
    for d in inter_dirs:
        print(f"  • {d.name}")

    for inter_dir in inter_dirs:
        cfg = _config_label_from_dir(inter_dir.name)
        n_missing_this_cfg = 0
        for paper_path in inter_dir.glob("*.json"):
            try:
                doc = json.loads(paper_path.read_text())
            except Exception:
                continue
            refs    = doc.get("llm_refs") or doc.get("parser_refs") or []
            lookups = doc.get("lookup_results") or []
            for ref, lr in zip(refs, lookups):
                if _overall_status(lr) != "missing":
                    continue
                key = _cache_key(ref)
                if not key:
                    # Fall back to a stable hash of raw text — better
                    # than dropping unkeyable refs.
                    raw = (ref.get("raw") or "")[:200]
                    if not raw:
                        continue
                    key = f"raw={raw}"
                if key not in by_key:
                    by_key[key] = {
                        "ref":               ref,
                        "configs_missed_in": set(),
                        "n_papers_affected": 0,
                    }
                by_key[key]["configs_missed_in"].add(cfg)
                by_key[key]["n_papers_affected"] += 1
                n_missing_this_cfg += 1
        print(f"    {cfg:<40} → {n_missing_this_cfg} missing refs")
    return by_key


def load_existing(out_path: Path) -> tuple[list[dict], set[str]]:
    """Resume support: load already-completed entries to skip on re-run."""
    if not out_path.exists():
        return [], set()
    try:
        data = json.loads(out_path.read_text())
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
        else:
            results = data
    except Exception:
        return [], set()
    keys = {r["cache_key"] for r in results if "cache_key" in r}
    return results, keys


def _rescue_status(core_r, gs_r) -> tuple[str, list[str]]:
    """Did EITHER source rescue this ref?  Returns (new_status, sources)."""
    rescued_by = []
    if core_r and core_r.get("status") == "found":
        if core_r.get("label") in ("match",):
            rescued_by.append("core:match")
        elif core_r.get("label") == "fuzzy":
            rescued_by.append("core:fuzzy")
    if gs_r and gs_r.get("status") == "found":
        if gs_r.get("label") in ("match",):
            rescued_by.append("scholarly:match")
        elif gs_r.get("label") == "fuzzy":
            rescued_by.append("scholarly:fuzzy")

    if any(t.endswith(":match") for t in rescued_by):
        return "match", rescued_by
    if rescued_by:
        return "fuzzy_rescue", rescued_by
    return "still_missing", []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_dir", required=True,
                    help="Run directory containing results_*_intermediates dirs")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap at first N unique refs (0 = all).  Use for pilots.")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="Write results to disk every N refs")
    ap.add_argument("--max-blocks", type=int, default=3,
                    help="How many consecutive Scholar CAPTCHA blocks to "
                         "tolerate before giving up on Scholar (CORE keeps "
                         "running).  Each block triggers a 20-min sleep.")
    ap.add_argument("--no-scholar", action="store_true",
                    help="Skip Google Scholar entirely; CORE-only rescue. "
                         "Use when the `scholarly` library is broken in the "
                         "environment (e.g. geckodriver/Firefox version "
                         "mismatch) so we don't waste ~7 s/ref on a Scholar "
                         "attempt that returns nothing.")
    args = ap.parse_args()

    rundir   = Path(args.from_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: gather the unique missing-ref set
    by_key = collect_missing(rundir)
    print(f"\nTotal unique missing refs: {len(by_key)}")

    # Step 2: resume support
    results, done_keys = load_existing(out_path)
    if done_keys:
        print(f"Resume: {len(done_keys)} refs already in {out_path.name}; skipping them")

    pending = [(k, v) for k, v in by_key.items() if k not in done_keys]
    if args.limit:
        pending = pending[: args.limit]
    print(f"To rescue this run: {len(pending)} refs")
    if not pending:
        print("Nothing to do.")
        return

    # Step 3: lift the Scholar budget so it doesn't refuse calls after the
    # default cap of 30 (designed for per-paper runs, not a rescue study).
    budget = ScholarBudget(len(pending) * 2)   # 2× headroom

    # Step 4: process refs serially.  Each ref fires CORE + Scholar
    # concurrently in a 2-thread pool.  Scholar's internal lock
    # serialises Scholar calls globally; we can't help that.
    print(f"\nStarting rescue (writing to {out_path})...\n")
    t_total = time.time()
    n_core_hits = n_scholar_hits = n_rescued_match = n_rescued_fuzzy = 0
    scholar_block_streak = 0
    scholar_disabled     = bool(args.no_scholar)
    if scholar_disabled:
        print("⚙️  Scholar disabled by --no-scholar flag; CORE-only rescue.\n")

    for i, (key, info) in enumerate(pending, 1):
        ref      = info["ref"]
        configs  = sorted(info["configs_missed_in"])
        n_papers = info["n_papers_affected"]
        t0       = time.time()

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_core = pool.submit(_lookup_core, ref)
            # Skip Scholar entirely if we've given up after too many blocks.
            if scholar_disabled:
                f_gs = None
                gs_r = {"status": "skipped", "reason": "captcha_disabled"}
            else:
                f_gs = pool.submit(_lookup_scholarly, ref, budget)
            core_r = f_core.result()
            if f_gs is not None:
                gs_r = f_gs.result()

        # Track Scholar block state — flip set inside _lookup_scholarly
        # when CAPTCHA hits.
        if not scholar_disabled and time.monotonic() < _lookup_mod._SCHOLARLY_BLOCKED_UNTIL:
            scholar_block_streak += 1
            print(f"  ⚠️  Scholar CAPTCHA hit (streak {scholar_block_streak}/{args.max_blocks})")
            if scholar_block_streak >= args.max_blocks:
                scholar_disabled = True
                print(f"  ✗  Scholar disabled for the remainder; CORE-only from here.")
            else:
                # Reset flag and sleep for 20 min before retrying
                _lookup_mod._SCHOLARLY_BLOCKED_UNTIL = 0.0
                print(f"  💤 sleeping 20 min before trying Scholar again …")
                time.sleep(20 * 60)

        new_status, rescued_by = _rescue_status(core_r, gs_r)

        if core_r.get("status") == "found":
            n_core_hits += 1
        if gs_r and gs_r.get("status") == "found":
            n_scholar_hits += 1
        if new_status == "match":
            n_rescued_match += 1
        elif new_status == "fuzzy_rescue":
            n_rescued_fuzzy += 1

        results.append({
            "cache_key":         key,
            "configs_missed_in": configs,
            "n_papers_affected": n_papers,
            "cited_title":       (ref.get("title") or "")[:240],
            "cited_authors":     (ref.get("authors") or "")[:240],
            "raw_excerpt":       (ref.get("raw") or "")[:400].replace("\n", " "),
            "core":              core_r,
            "scholarly":         gs_r,
            "rescued_by":        rescued_by,
            "new_status":        new_status,
            "elapsed_s":         round(time.time() - t0, 2),
        })

        if i % args.checkpoint_every == 0 or i == len(pending):
            out_path.write_text(json.dumps({
                "summary": {
                    "ran":               i,
                    "total_pending":     len(pending),
                    "total_unique":      len(by_key),
                    "core_hits":         n_core_hits,
                    "scholar_hits":      n_scholar_hits,
                    "scholar_disabled":  scholar_disabled,
                    "rescued_match":     n_rescued_match,
                    "rescued_fuzzy":     n_rescued_fuzzy,
                    "elapsed_s":         round(time.time() - t_total, 1),
                },
                "results": results,
            }, indent=2, ensure_ascii=False))
            print(f"  [{i}/{len(pending)}]  core={n_core_hits}  "
                  f"gs={n_scholar_hits}  rescued={n_rescued_match} match + "
                  f"{n_rescued_fuzzy} fuzzy   ({time.time() - t_total:.0f}s elapsed)")

    # Step 5: final summary block
    elapsed = time.time() - t_total
    print()
    print("=" * 70)
    print(f"RESCUE SUMMARY  ({elapsed/60:.1f} min, {len(pending)} refs)")
    print("=" * 70)
    print(f"CORE hits (any status):        {n_core_hits}/{len(pending)}  ({100*n_core_hits/len(pending):.1f}%)")
    print(f"Scholar hits (any status):     {n_scholar_hits}/{len(pending)}  ({100*n_scholar_hits/len(pending):.1f}%)")
    print(f"Scholar disabled by CAPTCHA:   {scholar_disabled}")
    print()
    print(f"Rescued to `match`:    {n_rescued_match}/{len(pending)}  ({100*n_rescued_match/len(pending):.1f}%)")
    print(f"Rescued to `fuzzy`:    {n_rescued_fuzzy}/{len(pending)}  ({100*n_rescued_fuzzy/len(pending):.1f}%)")
    print(f"Still missing:         {len(pending) - n_rescued_match - n_rescued_fuzzy}/{len(pending)}")

    # Per-config rescue rates: how many of each config's missing refs
    # get rescued?  (Useful for the paper's "missing != fabricated" claim.)
    print(f"\nPer-config rescue distribution:")
    by_cfg: dict[str, dict] = {}
    for r in results:
        for cfg in r.get("configs_missed_in", []):
            d = by_cfg.setdefault(cfg, {"n": 0, "rescued_match": 0, "rescued_fuzzy": 0})
            d["n"] += 1
            if r["new_status"] == "match":
                d["rescued_match"] += 1
            elif r["new_status"] == "fuzzy_rescue":
                d["rescued_fuzzy"] += 1
    for cfg, d in sorted(by_cfg.items()):
        rm, rf, n = d["rescued_match"], d["rescued_fuzzy"], d["n"]
        print(f"  {cfg:<40} {rm:>4} match + {rf:>4} fuzzy of {n:>4}  "
              f"({100*(rm+rf)/n:.1f}% rescued)")


if __name__ == "__main__":
    main()
