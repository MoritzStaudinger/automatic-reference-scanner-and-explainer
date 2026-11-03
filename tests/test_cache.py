import time

import cache


def test_negative_entries_expire(fresh_cache, monkeypatch):
    fresh_cache.set("openalex", "k", {"status": "not_found"})
    fresh_cache.set("openalex", "f", {"status": "found", "label": "match"})
    assert fresh_cache.get("openalex", "k") == {"status": "not_found"}
    monkeypatch.setattr(cache, "NEG_TTL_SECONDS", -1)
    assert fresh_cache.get("openalex", "k") is None          # stale negative
    assert fresh_cache.get("openalex", "f")["label"] == "match"  # found never expires


def test_purge_negative(fresh_cache):
    fresh_cache.set("crossref", "a", {"status": "not_found"})
    fresh_cache.set("crossref", "b", {"status": "found"})
    fresh_cache.set("dblp", "c", {"status": "not_in_anthology"})
    assert fresh_cache.purge_negative(source="crossref") == 1
    assert fresh_cache.purge_negative() == 1
    assert fresh_cache.stats() == {"crossref": 1}


def test_cached_survives_sqlite_errors(fresh_cache, monkeypatch):
    import sqlite3

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(fresh_cache, "get", boom)
    monkeypatch.setattr(fresh_cache, "set", boom)

    @cache.cached("x")
    def lookup(ref):
        return {"status": "found"}
    assert lookup({"title": "Some title"}) == {"status": "found"}


def test_custom_key_fn(fresh_cache):
    calls = []

    @cache.cached("url_verify", key_fn=lambda r: r.get("url"))
    def lookup(ref):
        calls.append(ref["url"])
        return {"status": "found"}
    lookup({"title": "T", "url": "https://a.example"})
    lookup({"title": "T", "url": "https://b.example"})
    lookup({"title": "T", "url": "https://a.example"})
    assert calls == ["https://a.example", "https://b.example"]


def test_forget(fresh_cache):
    fresh_cache.set("openalex", "k1", {"status": "found"})
    fresh_cache.set("crossref", "k1", {"status": "not_found"})
    fresh_cache.set("crossref", "k2", {"status": "found"})
    assert fresh_cache.forget(["k1", None]) == 2
    assert fresh_cache.stats() == {"crossref": 1}
