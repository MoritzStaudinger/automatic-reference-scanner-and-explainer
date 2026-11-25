#!/usr/bin/env python3
"""ARSE usage per day, as JSON for the vm-monitor agent (plugin).

    python scripts/usage_report.py [--since YYYY-MM-DD]

Reads the API job store and the lookup cache read-only and prints
{"usage": [{"site": "arse", "date", "metric", "value"}, ...]}.
Counts only — no file names, titles or keys.

The cache counts *new* entries per day: a reference looked up twice is
counted once (the second answer comes from the cache), and an entry that
was refreshed moves to the refresh day.  Good as a trend, not as an
exact request count — the nginx log (agent) has the request numbers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DB = Path(os.getenv("ARSE_CACHE", ROOT / "data" / "cache.sqlite"))
JOBS_DB = Path(os.getenv("ARSE_JOBS_DB", ROOT / "data" / "jobs.sqlite"))
SITE = "arse"

# cache source -> metric
CACHE_METRICS = {
    "claim_verify": "claim_checks",          # citation-context verdicts
    "claim_analyse": "claim_analyses",
    "fulltext": "fulltexts_fetched",
    "url_verify": "urls_checked",
}
LOOKUP_SOURCES = ("arxiv", "core", "crossref", "dblp", "doi_org", "openalex",
                  "openlibrary", "scholarly", "semantic_scholar")


def _ro(path: Path):
    if not path.exists():
        return None
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)


def report(since: dt.date) -> list[dict]:
    t0 = int(dt.datetime.combine(since, dt.time()).timestamp())
    rows = []

    def add(day, metric, value):
        rows.append({"site": SITE, "date": day, "metric": metric, "value": value})

    conn = _ro(CACHE_DB)
    if conn:
        with conn:
            q = ("SELECT date(ts, 'unixepoch') AS d, source, count(*) FROM cache "
                 "WHERE ts >= ? GROUP BY d, source")
            lookups: dict = {}
            for d, source, n in conn.execute(q, (t0,)):
                if source in CACHE_METRICS:
                    add(d, CACHE_METRICS[source], n)
                elif source in LOOKUP_SOURCES:
                    lookups[d] = lookups.get(d, 0) + n
            for d, n in sorted(lookups.items()):
                add(d, "lookups", n)
        conn.close()

    conn = _ro(JOBS_DB)
    if conn:
        with conn:
            q = ("SELECT date(created_at, 'unixepoch') AS d, count(*), sum(n_total) FROM jobs "
                 "WHERE created_at >= ? GROUP BY d")
            for d, n, papers in conn.execute(q, (t0,)):
                add(d, "api_jobs", n)
                add(d, "api_papers", papers or 0)
            q = ("SELECT date(ts, 'unixepoch') AS d, count(*) FROM results "
                 "WHERE ts >= ? AND status = 'error' GROUP BY d")
            for d, n in conn.execute(q, (t0,)):
                add(d, "api_papers_failed", n)
        conn.close()
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", type=dt.date.fromisoformat,
                    default=dt.date.today() - dt.timedelta(days=400))
    args = ap.parse_args(argv)
    json.dump({"usage": report(args.since)}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
