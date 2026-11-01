"""Headless Streamlit checks: every page renders without exceptions, and
card HTML escapes untrusted text."""
from streamlit.testing.v1 import AppTest

from ui import cards


def test_check_page_landing_renders():
    at = AppTest.from_file("ui/check.py", default_timeout=60).run()
    assert not at.exception
    assert any("Human review is required" in i.value for i in at.info)


def test_api_and_about_pages_render():
    for page in ("pages/API.py", "pages/About.py"):
        at = AppTest.from_file(page, default_timeout=30).run()
        assert not at.exception, page


def test_card_escapes_and_stays_single_line():
    ref = {"title": "<script>alert(1)</script>", "raw": "a\n\nb <b>x</b>", "page": 2}
    lr = {"url_verify": {"status": "dead_link", "note": "cited URL is dead (HTTP 404)",
                         "url": "javascript:alert(1)"},
          "crossref": {"status": "found", "label": "match", "similarity": 1.0,
                       "url": "https://doi.org/10.1/x", "found_title": "<i>t</i>"}}
    html = cards.card_html(0, ref, lr, [])
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "javascript:" not in html            # non-http links are not linked
    assert "\n" not in html
    assert "dead link" in html


def test_counts_include_dead_links():
    lrs = [{"url_verify": {"status": "dead_link"}}, None]
    c = cards.counts(lrs, 2)
    assert c["dead"] == 1 and c["pending"] == 1 and c["missing"] == 1


def test_orphans_only_from_bibliography_pages():
    from ui.processing import bibliography_orphans
    located = [{"found": True, "page": 2}] + [{"found": True, "page": 5}] * 3
    orphans = [{"page": 2, "text": "body"}, {"page": 5, "text": "ref"}, {"page": 6, "text": "ref2"}]
    assert [o["page"] for o in bibliography_orphans(located, orphans)] == [5, 6]
    assert bibliography_orphans([{"found": True, "page": 2}], orphans) == orphans
