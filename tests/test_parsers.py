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


# ── Repair against the printed citation text ────────────────────────────────

from parsers import _repair_from_raw, _names_from_raw, _extend_title


def test_scrambled_grobid_authors_are_read_from_raw():
    ref = {"title": "Versioning a full-text information retrieval system",
           "authors": "G Peter, Rex Anick, Flynn",
           "raw": "[5] Peter G. Anick and Rex A. Flynn. Versioning a full-text information "
                  "retrieval system. In Proceedings of SIGIR, 1992."}
    assert _repair_from_raw(ref)["authors"] == "Peter G. Anick, Rex A. Flynn"


def test_acl_author_list_with_others_and_hyphenation():
    raw = ("Zaid Vayani, Zhong Ding, Yiyi Shen, and Radu Sori- cut. 2024b. All languages "
           "matter: Evaluating lmms on culturally diverse 100 languages.")
    assert _names_from_raw(raw, "All languages matter: Evaluating lmms") == \
        ["Zaid Vayani", "Zhong Ding", "Yiyi Shen", "Radu Soricut"]
    assert _names_from_raw("Tom Brown, Amanda Askell, and others. 2020. Language models are "
                           "few-shot learners.", "Language models are few-shot learners") == \
        ["Tom Brown", "Amanda Askell"]


def test_last_first_style_is_left_to_grobid():
    raw = "Smith, J., & Doe, K. (2020). A study of things. Journal of Stuff, 3(2), 1-10."
    assert _names_from_raw(raw, "A study of things") is None


def test_correct_grobid_authors_are_kept():
    ref = {"title": "Pyserini: A Python Toolkit", "authors": "Jimmy Lin, Xueguang Ma",
           "raw": "[44] Jimmy Lin and Xueguang Ma. Pyserini: A Python Toolkit. In SIGIR, 2021."}
    assert _repair_from_raw(ref) is ref


def test_title_cut_at_comma_is_completed():
    raw = ("[38] Jonathan B. Koffel and Melissa L. Rethlefsen. Reproducibility of Search "
           "Strategies Is Poor in Systematic Reviews Published in High-Impact Pediatrics, "
           "Cardiology and Surgery Journals: A Cross-Sectional Study. PLOS ONE, 11(9), 2016.")
    t = "Reproducibility of Search Strategies Is Poor in Systematic Reviews Published in High-Impact Pediatrics"
    assert _extend_title(raw, t).endswith("Cardiology and Surgery Journals: A Cross-Sectional Study")
    # A comma followed by the venue is not part of the title.
    raw2 = "A. Author. A good title, In Proceedings of ACL, pages 1-9, 2020."
    assert _extend_title(raw2, "A good title") == "A good title"
    # Repeated words in the title don't duplicate it.
    raw3 = "A. Author and B. Author. Data, models and data. Venue, 2020."
    assert _extend_title(raw3, "Data, models and data") == "Data, models and data"


def test_org_author_and_appended_publisher():
    ref = {"title": "Data Citation of Evolving Data. Research Data Alliance", "authors": "",
           "raw": "[52] Andreas Rauber and Ari Asmi. Data Citation of Evolving Data. Research "
                  "Data Alliance, 2015."}
    assert _repair_from_raw(ref)["title"] == "Data Citation of Evolving Data"
    iso = {"title": "Information and documentation -Guidelines for bibliographic references",
           "authors": "",
           "raw": "[28] International Organization for Standardization (ISO). Information and "
                  "documentation - Guidelines for bibliographic references, June 2010."}
    assert _repair_from_raw(iso)["authors"] == "International Organization for Standardization (ISO)"


def test_line_break_hyphens_use_document_evidence():
    from parsers import _line_break_hyphen_fixer
    doc = ("We compare zero-shot and few-shot prompting on many tasks.\n"
           "Tom Brown and others. 2020. Language models are few-\nshot learners.\n"
           "An evaluation of the method. Good evalu-\nation matters.\n"
           "Crossroads, buildings and neighbor-\nhoods: a dataset.\n")
    fix = _line_break_hyphen_fixer(doc)
    assert fix("Language models are fewshot learners") == "Language models are few-shot learners"
    assert fix("Good evaluation matters") == "Good evaluation matters"      # joined form occurs
    assert fix("Crossroads, buildings and neighborhoods") == \
        "Crossroads, buildings and neighborhoods"                          # "hoods" never alone


def test_no_date_entries_start_a_new_reference():
    sec = ("Edward J Hu and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large\nlanguage models. In ICLR.\n"
           "EuroBench. 2026. Wikipedia native topic\nclassification: sk. Hugging Face dataset\n"
           "EuroBench/eurobench-wiki-sk-topics, revision\n72d2dd53165c44acadab521946e36115ffdcb6e8.\n"
           "Slovak Wikipedia article introductions with category\nlabels; CC BY-SA 4.0.\n"
           "Google. n.d. Gemma-4-E2B-it: Pinned\nmodel card. Hugging Face model\n"
           "card google/gemma-4-E2B-it, revision\n3e22461f65e89153144f8adb70e3b8c2cc9845a7.\n"
           "Albert Q. Jiang and Others. 2023. Mistral 7B. arXiv preprint arXiv:2310.06825.\n")
    refs = _split_plaintext_bibliography(sec)
    assert [r.split(".")[0] for r in refs] == ["Edward J Hu and Weizhu Chen", "EuroBench", "Google", "Albert Q"]
    assert refs[1].endswith("CC BY-SA 4.0.")
    assert refs[2].startswith("Google. n.d. Gemma-4-E2B-it")


def test_orphan_block_with_two_author_year_entries_is_split():
    import fitz
    import highlighter
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    known = ["Edward J Hu and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In ICLR.",
             "Albert Q. Jiang and Others. 2023. Mistral 7B. arXiv preprint arXiv:2310.06825."]
    y = 80
    for t in known:
        page.insert_textbox(fitz.Rect(60, y, 300, y + 40), t, fontsize=8)
        y += 50
    page.insert_textbox(fitz.Rect(60, y, 300, y + 160), (
        "EuroBench. 2026. Wikipedia native topic classification: sk. Hugging Face dataset "
        "EuroBench/eurobench-wiki-sk-topics, revision 72d2dd53165c44acadab521946e36115ffdcb6e8. "
        "Slovak Wikipedia article introductions with category labels; CC BY-SA 4.0. "
        "Google. n.d. Gemma-4-E2B-it: Pinned model card. Hugging Face model card "
        "google/gemma-4-E2B-it, revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7."), fontsize=8)
    refs = [{"raw": t, "title": t.split(". ")[2]} for t in known]
    _, _, orphans = highlighter.highlight_references(doc.tobytes(), refs, ["match", "match"])
    texts = [o["text"] for o in orphans]
    assert len(texts) == 2
    assert texts[0].startswith("EuroBench. 2026.") and texts[1].startswith("Google. n.d.")



def test_tei_arxiv_prefix_is_stripped_from_url():
    tei = _TEI.replace('<idno type="DOI">10.1093/jamia/ocaa148</idno>',
                       '<idno type="arXiv">arXiv:2404.05961</idno>')
    [ref] = _parse_tei_references(tei)
    assert ref["url"] == "https://arxiv.org/abs/2404.05961"


# ── Author-year details found on EMNLP 2025 main.1070 ───────────────────────

def test_suffixed_years_pass_the_candidate_filter():
    from parsers import _looks_like_ref_candidate, _is_plausible_reference
    raw = "Yiming Zhao and Zhouhui Lian. 2024a. Anytext2: Visual text generation and editing. CoRR."
    assert _looks_like_ref_candidate(raw)
    ref = {"raw": raw, "title": "Anytext2"}
    assert _is_plausible_reference(ref) and ref["year"] == "2024"


def test_empty_grobid_entry_keeps_the_citation_list_aligned(monkeypatch):
    import parsers

    class R:
        text = ("<TEI xmlns='http://www.tei-c.org/ns/1.0'><text><back><listBibl>"
                "<biblStruct><analytic><title level='a' type='main'>First paper</title></analytic></biblStruct>"
                "<biblStruct/>"
                "<biblStruct><analytic><title level='a' type='main'>Third paper</title></analytic></biblStruct>"
                "</listBibl></back></text></TEI>")
    monkeypatch.setattr(parsers, "_grobid_post", lambda *a, **k: R())
    out = parsers._grobid_process_citation_list(["A. 2020. First paper.", "B. 2021. Recraft v3.",
                                                 "C. 2022. Third paper."])
    assert [o.get("title") for o in out] == ["First paper", None, "Third paper"]
    assert out[2]["raw"] == "C. 2022. Third paper."
