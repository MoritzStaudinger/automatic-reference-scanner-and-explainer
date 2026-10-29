"""Offline tests for lookup.py — every network call is mocked."""
import lookup


def test_failed_prewarm_chunk_does_not_poison_cache(fresh_cache, monkeypatch):
    def fail(*a, **k):
        raise TimeoutError("simulated")
    monkeypatch.setattr(lookup, "_http_get_retry", fail)
    refs = [{"title": "A real paper", "doi": "10.1234/abc"}]
    lookup._oa_batch_prewarm(refs)
    lookup._crossref_batch_prewarm(refs)
    assert fresh_cache.stats() == {}


def test_successful_prewarm_still_records_not_found(fresh_cache, monkeypatch):
    class R:
        def raise_for_status(self): pass
        def json(self): return {"results": [], "message": {"items": []}}
    monkeypatch.setattr(lookup, "_http_get_retry", lambda *a, **k: R())
    refs = [{"title": "A fabricated paper", "doi": "10.1234/nope"}]
    lookup._oa_batch_prewarm(refs)
    lookup._crossref_batch_prewarm(refs)
    assert fresh_cache.stats() == {"openalex": 1, "crossref": 1}


def test_lookup_all_isolates_per_ref_failures(fresh_cache, monkeypatch):
    for fn in ("_s2_batch_prewarm", "_oa_batch_prewarm",
               "_crossref_batch_prewarm", "_arxiv_batch_prewarm"):
        monkeypatch.setattr(lookup, fn, lambda refs: 0)

    def fake_lookup(ref, budget=None):
        if ref["title"] == "bad":
            raise RuntimeError("kaboom")
        return {"openalex": {"status": "not_found"}}
    monkeypatch.setattr(lookup, "lookup_reference", fake_lookup)
    out = lookup.lookup_all([{"title": "good"}, {"title": "bad"}], scholarly_budget=0)
    assert out[0]["openalex"]["status"] == "not_found"
    assert "kaboom" in out[1]["_error"]


def test_scholar_budget_is_per_call():
    a, b = lookup.ScholarBudget(1), lookup.ScholarBudget(2)
    assert a.take() and not a.take()
    assert b.take() and b.take() and not b.take()


def test_single_token_author_guard_is_whole_word():
    assert lookup.author_similarity("Li", "Oliver Smith, Ann Lin, Bo Chen") is None
    assert lookup.author_similarity("Li", "Wei Li, Ann Lin, Bo Chen") == 1.0
    assert lookup.author_similarity(
        "OpenAI", "OpenAI Josh Achiam, Steven Adler, Sandhini Agarwal") == 1.0


def test_refresh_label_keeps_wrong_venue_mismatch():
    r = {"status": "found", "similarity": 0.98, "authors_sim": 0.9,
         "venue_sim": 0.1, "found_venue": "ICML", "label": "mismatch",
         "wrong_venue": True}
    assert lookup._refresh_label(r)["label"] == "mismatch"
