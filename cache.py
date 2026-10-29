"""
Persistent SQLite cache for external lookup responses.

Keys: (source, query_hash). Values: JSON-serialised result dicts.

We cache `found` results indefinitely — papers don't move.  Negative
results (`not_found`, `not_in_anthology`) expire after ARES_NEG_TTL_DAYS
(default 14) so papers that get indexed later are picked up again.  We do
NOT cache transient errors (`error`, `rate_limited`, `no_index`) so a
flaky run doesn't poison the cache for next time.

Clean up stale negatives with ``scripts/purge_negative_cache.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_PATH    = Path(os.getenv(
    "ARES_CACHE", _PROJECT_ROOT / "data" / "cache.sqlite"))

_CACHEABLE_STATUSES = {"found", "not_found", "not_in_anthology", "dead_link"}
_NEGATIVE_STATUSES  = {"not_found", "not_in_anthology", "dead_link"}
NEG_TTL_SECONDS     = int(float(os.getenv("ARES_NEG_TTL_DAYS", "14")) * 86400)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

class _LookupCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                     timeout=30)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                source TEXT NOT NULL,
                key    TEXT NOT NULL,
                value  TEXT NOT NULL,
                ts     INTEGER NOT NULL,
                PRIMARY KEY (source, key)
            )
        """)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.commit()

    def get(self, source: str, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, ts FROM cache WHERE source=? AND key=?",
                (source, key),
            ).fetchone()
        if not row:
            return None
        value = json.loads(row[0])
        if (isinstance(value, dict)
                and value.get("status") in _NEGATIVE_STATUSES
                and time.time() - row[1] > NEG_TTL_SECONDS):
            return None   # stale negative — re-query
        return value

    def set(self, source: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (source, key, value, ts) "
                "VALUES (?,?,?,?)",
                (source, key, json.dumps(value), int(time.time())),
            )
            self._conn.commit()

    def purge_negative(self, older_than_s: int = 0,
                       source: str | None = None) -> int:
        """Delete negative results older than ``older_than_s`` seconds."""
        cutoff = int(time.time()) - older_than_s
        sql = ("DELETE FROM cache WHERE ts <= ? AND "
               "json_extract(value, '$.status') IN ('not_found','not_in_anthology','dead_link')")
        params: list[Any] = [cutoff]
        if source:
            sql += " AND source = ?"
            params.append(source)
        with self._lock:
            n = self._conn.execute(sql, params).rowcount
            self._conn.commit()
        return n

    def forget(self, keys: list[str]) -> int:
        """Delete every source's entry for these keys (per-ref re-check)."""
        keys = [k for k in keys if k]
        if not keys:
            return 0
        with self._lock:
            n = self._conn.execute(
                f"DELETE FROM cache WHERE key IN ({','.join('?' * len(keys))})", keys
            ).rowcount
            self._conn.commit()
        return n

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT source, COUNT(*) FROM cache GROUP BY source"
            ).fetchall()
        return {src: n for src, n in rows}


_cache_singleton: _LookupCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> _LookupCache:
    global _cache_singleton
    if _cache_singleton is None:
        with _cache_lock:
            if _cache_singleton is None:
                _cache_singleton = _LookupCache(CACHE_PATH)
    return _cache_singleton


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

_NORM_RE = re.compile(r"[^a-z0-9]+")


def cache_key(ref: dict[str, Any]) -> str | None:
    """
    Deterministic key for a reference. Combines DOI, arXiv ID, normalized
    title, and year so refs with the same identity collide across runs but
    refs that happen to share a title fragment don't.
    """
    parts: list[str] = []

    doi = (ref.get("doi") or "").strip().lower().rstrip(".,)")
    if doi:
        parts.append(f"doi={doi}")

    url = ref.get("url") or ""
    if "arxiv" in url.lower():
        m = re.search(r"(\d{4}\.\d{4,5})", url)
        if m:
            parts.append(f"arxiv={m.group(1)}")

    title = (ref.get("title") or "").lower().strip()
    if title:
        norm = _NORM_RE.sub(" ", title).strip()
        if norm:
            parts.append(f"t={norm}")

    year = ref.get("year") or ""
    if year:
        parts.append(f"y={str(year)[:4]}")

    return "|".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def cached(source_name: str, postprocess: Callable | None = None,
           key_fn: Callable[[dict[str, Any]], str | None] | None = None,
           ) -> Callable:
    """
    Decorator that wraps a ``_lookup_<source>(ref) -> dict`` function with
    SQLite caching.  Cache miss → call wrapped function and (if cacheable)
    store the result.  Cache hit → skip the API call entirely.

    Pass ``postprocess`` to apply a (result -> result) transformation on
    every returned dict (whether the result came from a cache hit or a
    fresh call).  Use this to recompute fields whose logic may have
    changed since the result was cached — e.g. composite labels.

    Pass ``key_fn`` to override ``cache_key`` (e.g. to key on the URL).
    SQLite errors never fail a lookup — they degrade to an uncached call.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapped(ref: dict[str, Any], *args, **kwargs) -> dict:
            key = (key_fn or cache_key)(ref)
            if not key:
                r = fn(ref, *args, **kwargs)
                return postprocess(r) if postprocess else r

            try:
                hit = get_cache().get(source_name, key)
            except sqlite3.Error as e:
                log.warning("cache read failed (%s/%s): %s", source_name, key, e)
                hit = None
            if hit is not None:
                return postprocess(hit) if postprocess else hit

            result = fn(ref, *args, **kwargs) or {}
            status = result.get("status")
            if status in _CACHEABLE_STATUSES:
                try:
                    get_cache().set(source_name, key, result)
                except sqlite3.Error as e:
                    log.warning("cache write failed (%s/%s): %s", source_name, key, e)
            return postprocess(result) if postprocess else result

        return wrapped
    return decorator
