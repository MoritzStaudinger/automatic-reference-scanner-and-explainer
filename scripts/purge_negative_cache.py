"""
Delete negative (`not_found` / `not_in_anthology`) entries from the lookup
cache.  Run once after deploying the pre-warm cache-poisoning fix, and
whenever you want indexes to be re-queried for papers they now carry.

    python scripts/purge_negative_cache.py                 # all negatives
    python scripts/purge_negative_cache.py --older-than 7  # only > 7 days old
    python scripts/purge_negative_cache.py --source openalex

Inside Docker:
    docker compose exec ares python scripts/purge_negative_cache.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--older-than", type=float, default=0,
                    help="only delete entries older than N days (default: all)")
    ap.add_argument("--source", help="restrict to one source, e.g. openalex")
    args = ap.parse_args()

    c = cache.get_cache()
    before = c.stats()
    n = c.purge_negative(int(args.older_than * 86400), args.source)
    print(f"cache: {c.path}")
    print(f"deleted {n} negative entries")
    after = c.stats()
    for src in sorted(before):
        if before[src] != after.get(src, 0):
            print(f"  {src:18s} {before[src]:>7d} -> {after.get(src, 0)}")


if __name__ == "__main__":
    main()
