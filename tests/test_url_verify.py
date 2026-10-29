"""url_verify with every network call mocked."""
import lookup
import netguard


def _page(html, status=200, url="https://site.example/p", ctype="text/html; charset=utf-8"):
    return netguard.FetchResult(url=url, status_code=status,
                                headers={"content-type": ctype}, content=html.encode())


def _no_github(monkeypatch):
    monkeypatch.setattr(lookup, "_github_repo", lambda url: None)


def test_citation_meta_match(fresh_cache, monkeypatch):
    _no_github(monkeypatch)
    html = ('<meta name="citation_title" content="Deep Residual Learning for Image Recognition">'
            '<meta name="citation_author" content="He, Kaiming">'
            '<meta name="citation_doi" content="10.1109/CVPR.2016.90">')
    monkeypatch.setattr(lookup, "_fetch_page", lambda url: _page(html))
    r = lookup._lookup_url({"title": "Deep residual learning for image recognition",
                            "authors": "Kaiming He", "url": "https://site.example/p"})
    assert r["status"] == "found" and r["label"] == "match" and r["doi"] == "10.1109/CVPR.2016.90"


def test_404_without_archive_is_dead_link(fresh_cache, monkeypatch):
    _no_github(monkeypatch)
    monkeypatch.setattr(lookup, "_fetch_page", lambda url: _page("", status=404))
    monkeypatch.setattr(lookup, "_wayback_snapshot", lambda url: None)
    r = lookup._lookup_url({"title": "Gone", "url": "https://site.example/gone"})
    assert r["status"] == "dead_link"


def test_404_with_archive_scores_archived_copy(fresh_cache, monkeypatch):
    _no_github(monkeypatch)
    pages = {"https://site.example/gone": _page("", status=404),
             "https://web.archive.org/x": _page("<title>My Great Tool | Blog</title>")}
    monkeypatch.setattr(lookup, "_fetch_page", lambda url: pages[url])
    monkeypatch.setattr(lookup, "_wayback_snapshot",
                        lambda url: {"url": "https://web.archive.org/x", "date": "2021"})
    r = lookup._lookup_url({"title": "My Great Tool", "url": "https://site.example/gone"})
    assert r["status"] == "found" and r["archived"] and r["label"] == "match"


def test_403_is_error_not_dead(fresh_cache, monkeypatch):
    _no_github(monkeypatch)
    monkeypatch.setattr(lookup, "_fetch_page", lambda url: _page("", status=403))
    r = lookup._lookup_url({"title": "Paywalled", "url": "https://site.example/p"})
    assert r["status"] == "error"


def test_blocked_url_is_error(fresh_cache, monkeypatch):
    _no_github(monkeypatch)
    monkeypatch.setattr(lookup, "_fetch_page",
                        lambda url: netguard.FetchError("blocked", "non-public"))
    r = lookup._lookup_url({"title": "x", "url": "http://10.0.0.1/"})
    assert r["status"] == "error" and "blocked" in r["error"]


def test_github_repo_matches_by_name(fresh_cache, monkeypatch):
    monkeypatch.setattr(lookup, "_github_repo", lambda url: {
        "name": "stanford_alpaca", "full_name": "tatsu-lab/stanford_alpaca",
        "description": "Code and documentation to train Stanford's Alpaca models",
        "owner": {"login": "tatsu-lab"}, "html_url": url, "created_at": "2023-03-10"})
    r = lookup._lookup_url({"title": "Stanford Alpaca: An instruction-following LLaMA model",
                            "authors": "Rohan Taori, Ishaan Gulrajani",
                            "url": "https://github.com/tatsu-lab/stanford_alpaca"})
    assert r["status"] == "found" and r["label"] == "match" and "wrong_authors" not in r


def test_academic_urls_skipped():
    assert lookup._lookup_url({"title": "x", "url": "https://arxiv.org/abs/1706.03762"})["status"] == "skipped"


def test_cache_key_depends_on_url():
    a = lookup._url_cache_key({"title": "T", "url": "https://a.example"})
    b = lookup._url_cache_key({"title": "T", "url": "https://b.example"})
    assert a != b
