"""
SQLite-backed job store for the ARSE batch API.

Two tables:
  jobs     — one row per submitted batch (status, progress counters)
  results  — one row per PDF inside a batch (status, JSON report)

Why SQLite (not Redis / a queue) — batches are bounded (~hundreds of PDFs,
not millions), the API lives in one container, and SQLite + WAL handles
concurrent writes from the API thread and the background worker thread
without a separate service.  Same rationale as the lookup cache.

Path is configurable via env ``ARES_JOBS_DB`` (defaults next to the
lookup cache: ``./data/jobs.sqlite``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_PATH = _PROJECT_ROOT / "data" / "jobs.sqlite"
JOBS_DB_PATH = Path(os.getenv("ARES_JOBS_DB", _DEFAULT_PATH))


class JobStore:
    """Thread-safe job store.  One instance per process."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or JOBS_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                status      TEXT NOT NULL,         -- queued | running | done | failed
                n_total     INTEGER NOT NULL,
                n_done      INTEGER NOT NULL DEFAULT 0,
                n_error     INTEGER NOT NULL DEFAULT 0,
                parser      TEXT,
                use_llm     INTEGER NOT NULL DEFAULT 0,
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL,
                error       TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                job_id   TEXT NOT NULL,
                filename TEXT NOT NULL,
                status   TEXT NOT NULL,            -- ok | error
                payload  TEXT NOT NULL,            -- JSON: PipelineReport or {error}
                ts       INTEGER NOT NULL,
                PRIMARY KEY (job_id, filename),
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            )
        """)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    # ----- mutations ------------------------------------------------------

    def create(self, job_id: str, n_total: int, parser: str, use_llm: bool) -> None:
        now = int(time.time())
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, status, n_total, parser, use_llm, created_at, updated_at) "
                "VALUES (?, 'queued', ?, ?, ?, ?, ?)",
                (job_id, n_total, parser, int(bool(use_llm)), now, now),
            )
            self._conn.commit()

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='running', updated_at=? WHERE id=?",
                (int(time.time()), job_id),
            )
            self._conn.commit()

    def add_result(self, job_id: str, filename: str, report: dict[str, Any]) -> None:
        now = int(time.time())
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO results (job_id, filename, status, payload, ts) "
                "VALUES (?, ?, 'ok', ?, ?)",
                (job_id, filename, json.dumps(report), now),
            )
            self._conn.execute(
                "UPDATE jobs SET n_done=n_done+1, updated_at=? WHERE id=?",
                (now, job_id),
            )
            self._conn.commit()

    def add_error(self, job_id: str, filename: str, error: str) -> None:
        now = int(time.time())
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO results (job_id, filename, status, payload, ts) "
                "VALUES (?, ?, 'error', ?, ?)",
                (job_id, filename, json.dumps({"error": error[:500]}), now),
            )
            self._conn.execute(
                "UPDATE jobs SET n_error=n_error+1, updated_at=? WHERE id=?",
                (now, job_id),
            )
            self._conn.commit()

    def finish(self, job_id: str, error: Optional[str] = None) -> None:
        """Mark a job as fully complete (all PDFs processed or batch crashed)."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=?",
                ("failed" if error else "done", error, int(time.time()), job_id),
            )
            self._conn.commit()

    # ----- reads ---------------------------------------------------------

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, status, n_total, n_done, n_error, parser, use_llm, "
                "       created_at, updated_at, error "
                "FROM jobs WHERE id=?",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "job_id":     row[0],
            "status":     row[1],
            "n_total":    row[2],
            "n_done":     row[3],
            "n_error":    row[4],
            "parser":     row[5],
            "use_llm":    bool(row[6]),
            "created_at": row[7],
            "updated_at": row[8],
            "error":      row[9],
        }

    def get_results(self, job_id: str) -> Optional[dict[str, Any]]:
        meta = self.get(job_id)
        if not meta:
            return None
        with self._lock:
            rows = self._conn.execute(
                "SELECT filename, status, payload, ts FROM results "
                "WHERE job_id=? ORDER BY ts",
                (job_id,),
            ).fetchall()
        return {
            **meta,
            "results": [
                {
                    "filename": fname,
                    "status":   status,
                    "payload":  json.loads(payload),
                    "ts":       ts,
                }
                for fname, status, payload, ts in rows
            ],
        }

    def sweep_stale(self, reason: str = "container restarted mid-batch") -> int:
        """
        Mark any non-terminal job ('queued' or 'running') as 'failed'.

        Called once at API startup — if the container restarts while a
        batch is in flight, the in-memory ThreadPoolExecutor that owned
        those jobs is gone and they would otherwise sit in 'running'
        forever.  Returns the number of jobs swept (for logging).

        Assumes a single API container.  In a multi-replica deployment
        you'd need leases / heartbeats instead — out of scope for now.
        """
        now = int(time.time())
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='failed', error=?, updated_at=? "
                "WHERE status IN ('queued', 'running')",
                (reason, now),
            )
            n = cur.rowcount or 0
            self._conn.commit()
        return n

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, status, n_total, n_done, n_error, created_at "
                "FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"job_id": r[0], "status": r[1], "n_total": r[2],
             "n_done": r[3], "n_error": r[4], "created_at": r[5]}
            for r in rows
        ]


# Module-level singleton — same pattern as cache.py
_singleton: Optional[JobStore] = None
_singleton_lock = threading.Lock()


def get_job_store() -> JobStore:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = JobStore()
    return _singleton
