import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import usage_report  # noqa: E402


def _ts(day: str, hour: int = 12) -> int:
    return int(dt.datetime.fromisoformat(f"{day}T{hour:02d}:00:00").timestamp())


def test_report_counts_per_day(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache.sqlite"
    with sqlite3.connect(cache) as c:
        c.execute("CREATE TABLE cache (source TEXT, key TEXT, value TEXT, ts INTEGER)")
        c.executemany("INSERT INTO cache VALUES (?,?,?,?)", [
            ("crossref", "a", "{}", _ts("2026-11-20")),
            ("semantic_scholar", "b", "{}", _ts("2026-11-20")),
            ("claim_verify", "c", "{}", _ts("2026-11-20")),
            ("crossref", "old", "{}", _ts("2026-01-01")),
            ("unknown_source", "d", "{}", _ts("2026-11-20")),
        ])
    jobs = tmp_path / "jobs.sqlite"
    with sqlite3.connect(jobs) as c:
        c.execute("CREATE TABLE jobs (id TEXT, status TEXT, n_total INT, created_at INT)")
        c.execute("CREATE TABLE results (job_id TEXT, filename TEXT, status TEXT, payload TEXT, ts INT)")
        c.execute("INSERT INTO jobs VALUES ('j1', 'done', 3, ?)", (_ts("2026-11-21"),))
        c.execute("INSERT INTO results VALUES ('j1', 'secret-paper.pdf', 'error', '{}', ?)", (_ts("2026-11-21"),))
    monkeypatch.setattr(usage_report, "CACHE_DB", cache)
    monkeypatch.setattr(usage_report, "JOBS_DB", jobs)

    assert usage_report.main(["--since", "2026-11-01"]) == 0
    out = capsys.readouterr().out
    got = {(r["date"], r["metric"]): r["value"] for r in json.loads(out)["usage"]}
    assert got == {("2026-11-20", "lookups"): 2, ("2026-11-20", "claim_checks"): 1,
                   ("2026-11-21", "api_jobs"): 1, ("2026-11-21", "api_papers"): 3,
                   ("2026-11-21", "api_papers_failed"): 1}
    assert "secret-paper" not in out


def test_missing_databases_give_empty_report(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_report, "CACHE_DB", tmp_path / "nope.sqlite")
    monkeypatch.setattr(usage_report, "JOBS_DB", tmp_path / "nope2.sqlite")
    assert usage_report.report(dt.date(2026, 1, 1)) == []
    assert not (tmp_path / "nope.sqlite").exists()      # read-only: never creates a DB
