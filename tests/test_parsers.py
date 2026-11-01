"""Offline tests for the reference-section parsing helpers.

The cases come from real layouts that broke parsing: an ACM review
manuscript (margin line numbers, running headers), ACL papers (appendix
letter on its own line, page ranges ending in 19xx), and GROBID TEI with
identifiers nested under <analytic>.
"""

from parsers import (
    _better_title, _clean_doi, _distinct_refs, _locate_reference_section, _parse_tei_references,
    _split_numbered, _split_plaintext_bibliography, _strip_name_prefix,
    _strip_page_furniture, _trim_to_section_end, dedupe_references,
)
from lookup import _doi_in_citation


# ── Page furniture ──────────────────────────────────────────────────────────

def test_line_numbers_and_running_headers_are_removed():
    def page(n, body):
        return ([str(1500 + n * 10 + i) for i in range(6)]
                + ["Set in Stone: Reproducible Search", str(n)] + body
                + ["Manuscript submitted to ACM"])
    pages = [page(n, [f"body line {n}"]) for n in range(1, 7)]
    pages[3] = page(4, ["[12] Timo Breuer. A paper title. In", "Proceedings, 2021."])
    out = _strip_page_furniture(pages)
    assert out[3] == ["[12] Timo Breuer. A paper title. In", "Proceedings, 2021."]
    assert all("Manuscript" not in l and "Set in Stone" not in l for p in out for l in p)


# ── Section boundaries ──────────────────────────────────────────────────────

def test_wrapped_title_starting_with_A_does_not_end_the_section():
    section = ("[39] Wojciech Kusa. Automation of citation screening using neural networks:\n"
               "A replicability study. In Advances in IR, 2022.\n"
               "[40] Next Author. Another paper. 2023.\n")
    assert _trim_to_section_end(section) == section


def test_acl_appendix_letter_on_its_own_line_ends_the_section():
    section = ("Maria Vasardani. 2013. Locating place names. IJGIS, 27(12):2509–2532.\n"
               "A\nSystem Screen Shot\nFigure 3: The screen shot of the tool")
    assert _trim_to_section_end(section).rstrip().endswith("27(12):2509–2532.")


def test_appendix_headings_still_end_the_section():
    for heading in ("A Implementation Details", "Appendix B", "A. Prompts"):
        text = f"[1] A. Author. Title. 2020.\n{heading}\nmore text"
        assert heading not in _trim_to_section_end(text)


def test_bibliography_heading_is_chosen_by_content():
    bib = "".join(f"[{i}] A. Author. Title {i}. Venue, 20{10 + i}.\n" for i in range(1, 8))
    text = ("Contents\nReferences\n12\nIntro text …\n" + "x " * 300
            + "\nReferences\n" + bib
            # An appendix figure label must not win just because it comes last.
            + "Figure 9\nLiterature\nPainting\nMeme\n+7.4\n+6.1\n" + "z " * 300)
    assert _locate_reference_section(text).startswith("[1] A. Author")


# ── Splitting ───────────────────────────────────────────────────────────────

def test_numbered_split_ignores_lines_that_look_like_markers():
    section = ("[3] Avishek Anand. Temporal index sharding. In Proceedings of SIGIR, July\n"
               "2011. ACM. ISBN 978-1-4503-0757-4.\n"
               "[4] Avishek Anand. Index maintenance for time-travel text search. In\n"
               "Proceedings, 2012.\n"
               "[5] Peter G. Anick. Versioning a full-text system. 1992.\n")
    # The list starts at [3] here, so seed the sequence like a real list would.
    refs = _split_numbered("[1] X. A title here. 2001.\n[2] Y. B title here. 2002.\n" + section)
    assert len(refs) == 5
    assert refs[2].startswith("[3] Avishek Anand") and "2011. ACM." in refs[2]


def test_numbered_split_needs_a_sequence():
    assert _split_numbered("1. Introduction\nSome text\n3. Method\n") == []


def test_page_range_ending_in_19xx_is_not_a_year_anchor():
    section = ("Tom Brown, Amanda Askell, and others. 2020. Language models are few-shot "
               "learners. Advances in neural information processing systems, 33:1877–1901.\n"
               "Pei Chen, Haotian Xu, and Ruihong Huang. 2022. Crossroads, buildings and "
               "neighborhoods. In Proceedings of NAACL.\n"
               "Erik Tjong Kim Sang. 2003. Introduction to the CoNLL-2003 shared task.")
    refs = _split_plaintext_bibliography(section)
    assert any(r.startswith("Pei Chen") for r in refs)
    assert not any(r.startswith("Advances") for r in refs)


# ── Field clean-up ──────────────────────────────────────────────────────────

def test_clean_doi():
    assert _clean_doi("https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0163309") \
        == "10.1371/journal.pone.0163309"
    assert _clean_doi("10.1007/978-3-319-06028-6\\_97") == "10.1007/978-3-319-06028-6_97"
    assert _clean_doi("not a doi") == ""


def test_strip_name_prefix():
    assert _strip_name_prefix("Mateusz Lango, and Ondřej Dušek. Leak, cheat, repeat: Data "
                              "contamination") == "Leak, cheat, repeat: Data contamination"
    assert _strip_name_prefix("ESIP Data Preservation and Stewardship Committee. Data Citation "
                              "Guideline for Earth Science Data") \
        == "Data Citation Guideline for Earth Science Data"
    assert _strip_name_prefix("Alex Vaughan, and others. 2024. The llama 3 herd of models") \
        == "The llama 3 herd of models"
    for keep in ("Deep Learning. A Review of Methods and Uses",
                 "BERT. Pre-training of deep bidirectional transformers",
                 "Attention is all you need"):
        assert _strip_name_prefix(keep) == keep


def test_better_title():
    assert _better_title("Language models are few-shot learners. Advances in neural "
                         "information processing systems",
                         "Language models are few-shot learners") \
        == "Language models are few-shot learners"
    assert _better_title("A Toolkit for Domain-Specific Search Experiments",
                         "pybool_ir: A Toolkit for Domain-Specific Search Experiments") \
        == "pybool_ir: A Toolkit for Domain-Specific Search Experiments"
    assert _better_title("Attention is all you need", "Attention is all") == ""


_TEI = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><back><listBibl>
<biblStruct><analytic><title level="a" type="main">Refining Boolean queries</title>
<author><persName><forename>Amal</forename><surname>Alharbi</surname></persName></author>
<idno type="DOI">10.1093/jamia/ocaa148</idno></analytic>
<monogr><title level="j">JAMIA</title><imprint><date type="published" when="2020"/></imprint></monogr>
<note type="raw_reference">Amal Alharbi. Refining Boolean queries. JAMIA, 2020. doi: 10.1093/jamia/ocaa148.</note>
</biblStruct></listBibl></back></text></TEI>"""


def test_tei_doi_under_analytic_is_read():
    [ref] = _parse_tei_references(_TEI)
    assert ref["doi"] == "10.1093/jamia/ocaa148"
    assert "doi: 10.1093" in ref["raw"]


# ── Dedupe ──────────────────────────────────────────────────────────────────

def test_dedupe_keeps_works_that_differ_only_by_number():
    refs = [{"title": f"Clef {y} technology assisted reviews in empirical medicine overview",
             "year": str(y), "raw": f"Kanoulas. Clef {y} … {y}."} for y in (2017, 2018, 2019)]
    kept, merged = dedupe_references(refs)
    assert merged == 0 and len(kept) == 3


def test_dedupe_still_merges_true_duplicates():
    a = {"title": "Reproducible hybrid time-travel retrieval in evolving corpora",
         "year": "2024", "doi": "10.1145/3673791.3698421", "raw": "Staudinger. Reproducible …"}
    b = {**a, "title": "Reproducible Hybrid Time-Travel Retrieval in Evolving Corpora."}
    kept, merged = dedupe_references([a, b])
    assert merged == 1 and len(kept) == 1


# ── Wrong-DOI guard ─────────────────────────────────────────────────────────

def test_doi_in_citation():
    assert _doi_in_citation({"raw": "He. 2009. doi: 10.1145/1645953.\n1646008. URL"},
                            "10.1145/1645953.1646008")
    assert not _doi_in_citation({"raw": "Smith. 2020. A title. Venue."}, "10.1/abc")


def test_author_list_title_is_replaced():
    assert _better_title("Alan Schelten, Alex Vaughan, and others",
                         "The llama 3 herd of models") == "The llama 3 herd of models"


def test_more_garbage_title_patterns():
    assert _strip_name_prefix("others. 2021. Learning transferable visual models from "
                              "natural language supervision") \
        == "Learning transferable visual models from natural language supervision"
    assert _strip_name_prefix("Atnafu Lambebo Tonja, and 1 others. 2024. Cvqa: Culturally-"
                              "diverse multilingual visual question answering benchmark") \
        == "Cvqa: Culturally-diverse multilingual visual question answering benchmark"
    assert _strip_name_prefix("2019a. Sentencebert: Sentence embeddings using siamese "
                              "bert-networks").startswith("Sentencebert")
    assert _better_title("Proceedings of the 16th ACM Conference on Recommender Systems",
                         "Recommendation as language processing (rlp)") \
        == "Recommendation as language processing (rlp)"
    assert _better_title("", "BioCLIP: A vision foundation model") == "BioCLIP: A vision foundation model"


def test_hyphenation_number_loss_is_not_a_different_work():
    a = {"title": "Deepseek-r1-distill-qwen-32b", "raw": "DeepSeek-AI. 2024. Deepseek-r1-distill-qwen-32b."}
    b = {"title": "Deepseek-r1-distill-qwen32b", "raw": "DeepSeek-AI. 2024. Deepseek-r1-distill-qwen32b."}
    assert not _distinct_refs(a, b)
    assert _distinct_refs({"title": "Clef 2017 overview"}, {"title": "Clef 2018 overview"})
    assert _better_title("Yuxuan Lyu and 1 others", "Deepseek-r1: Incentivizing reasoning") \
        == "Deepseek-r1: Incentivizing reasoning"
