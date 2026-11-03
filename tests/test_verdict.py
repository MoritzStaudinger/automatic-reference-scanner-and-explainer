from verdict import overall_status, status_reason, status_rgba


def _found(label, **kw):
    return {"status": "found", "label": label, **kw}


def test_pending_when_no_result():
    assert overall_status(None) == "pending"
    assert status_reason(None) == ""


def test_match_wins_over_mismatch():
    lr = {"crossref": _found("mismatch", wrong_doi=True), "openalex": _found("match")}
    assert overall_status(lr) == "match"
    assert status_reason(lr) == "confirmed by OpenAlex"


def test_mismatch_reason_priority():
    lr = {"crossref": _found("mismatch", wrong_venue=True),
          "openalex": _found("mismatch", wrong_doi=True)}
    assert overall_status(lr) == "mismatch"
    assert status_reason(lr) == "cited DOI points to a different paper"


def test_fuzzy_only_is_missing():
    assert overall_status({"dblp": _found("fuzzy")}) == "missing"
    assert overall_status({"dblp": {"status": "not_found"}}) == "missing"


def test_url_verify_counts_as_source():
    assert overall_status({"url_verify": _found("match")}) == "match"


def test_lookup_error_reason():
    assert "lookup failed" in status_reason({"_error": "boom"})


def test_rgba_format():
    assert status_rgba("match").startswith("rgba(") and status_rgba("nope") == status_rgba("pending")
