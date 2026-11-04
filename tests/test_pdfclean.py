"""Margin line numbers of draft PDFs are removed; nothing else is."""

import fitz

from pdfclean import strip_line_numbers


def _pdf(pages: int = 2, line_numbers: bool = True, start: int = 1) -> bytes:
    doc = fitz.open()
    n = start
    for p in range(pages):
        page = doc.new_page(width=595, height=842)
        y = 80
        for i in range(48):
            if line_numbers:
                page.insert_text((40, y), str(n), fontsize=7)        # left margin
            text = (f"[{i + 1}] A. Author. A paper title number {i}. In Proc. of ACL 2023."
                    if p == pages - 1 else f"Body text line {i} mentions 2020 and 42 results.")
            page.insert_text((80, y), text, fontsize=9)
            y += 15
            n += 1
        page.insert_text((290, 820), str(p + 1), fontsize=9)         # page number
    return doc.tobytes()


def _text(pdf: bytes) -> str:
    with fitz.open(stream=pdf, filetype="pdf") as d:
        return "\n".join(p.get_text() for p in d)


def test_line_numbers_are_removed_and_content_kept():
    raw = _pdf(pages=3, start=1)
    clean, n = strip_line_numbers(raw)
    assert n == 3 * 48
    words = [w for line in _text(clean).splitlines() for w in line.split()]
    assert "97" not in words and "144" not in words           # line numbers gone
    t = _text(clean)
    assert "[12] A. Author. A paper title number 11. In Proc. of ACL 2023." in t
    assert "mentions 2020 and 42 results" in t                # numbers in prose stay
    assert "\n3\n" in "\n" + t + "\n" or t.strip().endswith("3")   # page number stays


def test_pdf_without_line_numbers_is_untouched():
    raw = _pdf(pages=2, line_numbers=False)
    clean, n = strip_line_numbers(raw)
    assert n == 0 and clean is raw


def test_short_last_page_uses_the_column_found_elsewhere():
    doc = fitz.open(stream=_pdf(pages=2), filetype="pdf")
    page = doc.new_page(width=595, height=842)      # last page: only 4 lines
    for i, y in enumerate(range(80, 140, 15)):
        page.insert_text((40, y), str(200 + i), fontsize=7)
        page.insert_text((80, y), f"[{60 + i}] Short reference {i}. 2024.", fontsize=9)
    clean, n = strip_line_numbers(doc.tobytes())
    assert n == 2 * 48 + 4
    assert "200" not in _text(clean).split()


def test_numbered_listing_in_the_body_is_kept():
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i in range(20):                             # algorithm lines 0..19, mid-page
        page.insert_text((250, 200 + 12 * i), str(i), fontsize=8)
        page.insert_text((270, 200 + 12 * i), f"step {i}", fontsize=8)
    raw = doc.tobytes()
    clean, n = strip_line_numbers(raw)
    assert n == 0 and clean is raw


def test_garbage_input_is_returned_unchanged():
    clean, n = strip_line_numbers(b"not a pdf")
    assert n == 0 and clean == b"not a pdf"
