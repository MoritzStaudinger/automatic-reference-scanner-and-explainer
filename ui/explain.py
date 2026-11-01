"""
How a verdict is reached — shown in the "How is this decided?" popover
and on the About page.  Numbers come from lookup.py, so the text can't
drift from what the code does.
"""

from lookup import (
    AUTHOR_MATCH_THRESHOLD, FUZZY_THRESHOLD, MATCH_THRESHOLD,
    WRONG_FLAG_TITLE_FLOOR, WRONG_ID_TITLE_FLOOR, WRONG_VENUE_THRESHOLD,
)


def _p(x: float) -> str:
    return f"{x:.0%}"


MATCHING_MD = f"""
**1. Where each reference is looked up**
- A **DOI** goes to doi.org first, with Crossref, Semantic Scholar and OpenAlex as a vote;
  an **arXiv ID** goes to arXiv.
- Without an identifier, the **title** is searched in Semantic Scholar, OpenAlex, Crossref
  and DBLP; books also in Open Library.
- A cited **web page or repository** is fetched and its own title compared
  (GitHub through its API; dead links through the Internet Archive).

**2. Three similarity scores per candidate** (0–100 %)
- **Title** — after lowercasing and removing accents and punctuation, a word-order-insensitive
  fuzzy comparison. A cited title that is fully contained in the found one (e.g. a missing
  subtitle) still scores high.
- **Authors** — surnames are compared, each with fuzzy matching (Müller = Mueller, small typos).
  If the citation lists only some authors (*et al.*), the score is the share of cited surnames
  found; otherwise the overlap of both lists.
- **Venue** — fuzzy, tolerant of "Proceedings of …" and years. Ignored when the citation
  names a preprint server (arXiv, etc.), since the published venue legitimately differs.

**3. What each source concludes**
- **Match** — title ≥ {_p(MATCH_THRESHOLD)} and the authors agree (≥ {_p(AUTHOR_MATCH_THRESHOLD)})
  or are unknown; or title {_p(FUZZY_THRESHOLD)}–{_p(MATCH_THRESHOLD)} *and* the authors agree.
- **Mismatch** — the title is practically identical (≥ {_p(WRONG_FLAG_TITLE_FLOOR)}) but the
  authors (< {_p(AUTHOR_MATCH_THRESHOLD)}) or the venue (< {_p(WRONG_VENUE_THRESHOLD)}) clearly
  disagree; or the cited DOI / arXiv ID leads to a paper whose title is less than
  {_p(WRONG_ID_TITLE_FLOOR)} similar.
- **Weak** — anything else: the nearest candidate in that index, which is no evidence either way.

**4. The overall verdict**
- **Verified** if any source reports a match.
- **Mismatch** if none matches but at least one reports a mismatch.
- **Not found** otherwise. A dead cited link is shown as a separate badge.

**Safeguards**: a "title" of one or two words (usually a parser fragment) never produces a
mismatch, and identifiers only count if they appear in the citation text itself, never when
the LLM repair step added them.
"""
