"""Offline tests for the passage-based citation-context check:
segmentation, retrieval, claim analysis, verdict parsing and aggregation."""

import json

import claim_analysis
import claim_verifier
import evidence
from claim_extractor import Claim
from fulltext import FullText

PAPER = """# Set in Stone

**Abstract**

We study reproducible search for systematic reviews. Dense retrieval recovers
substantially more relevant studies than BM25 on four collections.

## 1 Introduction

Systematic reviews depend on search. Prior work has used Boolean queries [3, 7].
Large language models have been used to write the query [1].

## 5 Results

The same ranked search repeated today keeps only 48 to 63 of its first hundred
documents. Dense retrieval saves substantially more screening effort than BM25.
==> picture [455 x 181] intentionally omitted <==

52 Manuscript submitted to ACM

## References

[1] A. Author. Boolean query writing with LLMs. 2023.
[2] B. Author. Another paper about dense retrieval and BM25. 2022.
"""


def _claim(text, marker="[4]"):
    return Claim(text=text, citation_marker=marker, cited_ref_index=0, style="numeric")


# ── Segmentation ────────────────────────────────────────────────────────────

def test_segment_tracks_sections_and_drops_references():
    ps = evidence.segment(PAPER)
    sections = {p.section for p in ps}
    assert "5 Results" in sections and "1 Introduction" in sections
    assert not any("Another paper about dense retrieval" in p.text for p in ps)
    assert not any("Manuscript submitted" in p.text or "picture" in p.text for p in ps)


def test_passages_stay_within_a_section():
    ps = evidence.segment(PAPER)
    for p in ps:
        if p.section == "5 Results":
            assert "Systematic reviews depend" not in p.text


def test_sentence_split_keeps_abbreviations():
    s = evidence.split_sentences("As shown by Smith et al. in Fig. 3, the gain is large. "
                                 "It holds for all datasets tested here.")
    assert len(s) == 2


# ── Retrieval ───────────────────────────────────────────────────────────────

def test_bm25_finds_the_matching_passage():
    ps = evidence.segment(PAPER)
    hits = evidence.retrieve("ranked search repeated today keeps 48 to 63 documents", ps, k=2)
    assert hits and "48 to 63" in hits[0].passage.text


class _FakeEmbedder:
    """Puts the passage mentioning 'screening' first, whatever the query."""
    name = "fake"

    def encode(self, texts, kind):
        import numpy as np
        if kind == "query":
            return np.array([[1.0, 0.0]])
        return np.array([[1.0, 0.0] if "screening" in t else [0.0, 1.0] for t in texts])


def test_rank_fusion_combines_bm25_and_dense(monkeypatch):
    monkeypatch.setattr(evidence, "passage_vectors",
                        lambda ps, emb: emb.encode([p.text for p in ps], "passage"))
    ps = evidence.segment(PAPER)
    hits = evidence.retrieve("unrelated words zebra", ps, k=1, embedder=_FakeEmbedder())
    assert "screening" in hits[0].passage.text          # dense-only match still found
    assert hits[0].dense_rank == 1 and hits[0].bm25_rank is None


def test_unknown_embedder_means_bm25_only(monkeypatch):
    monkeypatch.setenv("ARSE_EMBEDDER", "does-not-exist")
    assert evidence.get_embedder() is None


# ── Claim analysis (heuristic) ──────────────────────────────────────────────

def test_heuristic_isolates_the_clause_and_intent():
    cc = claim_analysis.heuristic(_claim(
        "Dense retrieval improves recall by 12% on MS MARCO [3], unlike sparse models [7].", "[3]"))
    assert cc.cited_claim == "Dense retrieval improves recall by 12% on MS MARCO"
    assert cc.intent == "result"
    assert claim_analysis.heuristic(_claim("We use the BERT-base model [5] here.", "[5]")).intent == "method"
    assert claim_analysis.heuristic(
        _claim("Several approaches have been proposed for expansion [1, 2, 4].", "[1, 2, 4]")).intent == "background"


def test_llm_analysis_output_is_coerced():
    fb = claim_analysis.CitedClaim("fallback", ["fallback"], "result")
    cc = claim_analysis._coerce({"cited_claim": "X shows Y", "subclaims": "X shows Y",
                                 "intent": "METHOD"}, fb)
    assert cc.subclaims == ["X shows Y"] and cc.intent == "method" and cc.source == "llm"
    assert claim_analysis._coerce({"cited_claim": ""}, fb) is fb
    assert claim_analysis._coerce({"cited_claim": "a", "intent": "weird"}, fb).intent == "result"


# ── Verdicts ────────────────────────────────────────────────────────────────

def test_aggregate():
    agg = claim_verifier.aggregate
    assert agg(["supported", "supported"]) == "supported"
    assert agg(["supported", "contradicted"]) == "partially_supported"   # part holds
    assert agg(["contradicted", "not_addressed"]) == "contradicted"
    assert agg(["supported", "not_addressed"]) == "partially_supported"
    assert agg(["not_addressed", "not_addressed"]) == "not_addressed"
    assert agg(["ambiguous", "not_addressed"]) == "ambiguous"
    assert agg([]) == "not_addressed"


def test_parse_passage_response_variants():
    p = claim_verifier._parse_passage_response
    body = {"claims": [{"verdict": "supported", "passage_ids": ["P1"]}], "confidence": "high"}
    assert p(json.dumps(body))["confidence"] == "high"
    assert p("```json\n" + json.dumps(body) + "\n```")["claims"][0]["verdict"] == "supported"
    assert p("Here you go: " + json.dumps(body) + " done")["claims"]
    assert p('[{"verdict": "contradicted", "passage_ids": []}]')["claims"][0]["verdict"] == "contradicted"
    assert p("no json") is None


def _ft():
    return FullText(text=PAPER, source="test", is_abstract_fallback=False, url=None)


def _fake_llm(monkeypatch, response):
    monkeypatch.setattr(claim_verifier, "_make_client", lambda backend=None: object())
    monkeypatch.setattr(claim_verifier, "active_model", lambda: ("anthropic", "m"))
    monkeypatch.setattr(claim_verifier, "_cached_get", lambda k: None)
    monkeypatch.setattr(claim_verifier, "_cached_set", lambda k, v: None)
    calls = []

    def call(client, backend, model, system, user, **kw):
        calls.append(user)
        return json.dumps(response)
    monkeypatch.setattr(claim_verifier, "_call_llm", call)
    return calls


def test_evidence_comes_from_passages_and_invented_ids_are_dropped(monkeypatch):
    monkeypatch.delenv("ARSE_CLAIM_METHOD", raising=False)
    monkeypatch.setenv("ARSE_EMBEDDER", "none")
    _fake_llm(monkeypatch, {"claims": [{"verdict": "contradicted",
                                        "passage_ids": ["P1", "P99"],
                                        "explanation": "says 48 to 63"}],
                            "confidence": "high"})
    cc = claim_analysis.CitedClaim("keeps only 20% of the top 100 documents",
                                   ["The same ranked search keeps only 20 of its first hundred documents."],
                                   "result")
    [v] = claim_verifier.verify_batch([_claim("… keeps only 20% [4].")], _ft(), cited=[cc])
    assert v.verdict == "contradicted" and v.mode == "passages"
    assert [e["id"] for e in v.evidence] == ["P1"]               # P99 doesn't exist
    assert any(v.evidence[0]["text"] == p.text for p in evidence.segment(PAPER))
    assert v.evidence_passage == v.evidence[0]["text"][:400]


def test_verdict_without_evidence_is_downgraded(monkeypatch):
    monkeypatch.setenv("ARSE_EMBEDDER", "none")
    _fake_llm(monkeypatch, {"claims": [{"verdict": "supported", "passage_ids": []}]})
    cc = claim_analysis.CitedClaim("dense beats BM25", ["Dense retrieval recovers more than BM25."])
    [v] = claim_verifier.verify_batch([_claim("Dense beats BM25 [4].")], _ft(), cited=[cc])
    assert v.verdict == "ambiguous"


def test_not_addressed_triggers_one_wider_search(monkeypatch):
    monkeypatch.setenv("ARSE_EMBEDDER", "none")
    monkeypatch.setattr(claim_verifier, "_K_PER_SUBCLAIM", 1)
    calls = _fake_llm(monkeypatch, {"claims": [{"verdict": "not_addressed", "passage_ids": []}]})
    cc = claim_analysis.CitedClaim("dense retrieval", ["dense retrieval BM25 screening"])
    [v] = claim_verifier.verify_batch([_claim("Dense [4].")], _ft(), cited=[cc])
    assert v.verdict == "not_addressed" and len(calls) == 2


def test_related_work_passages_are_flagged():
    sp = evidence.ScoredPassage(evidence.Passage(0, "Prior work used Boolean queries [3, 7].", "1 Intro"), 1.0)
    assert "[cites other work]" in claim_verifier._passage_block([sp])
    sp2 = evidence.ScoredPassage(evidence.Passage(1, "We find a 12% gain.", "5 Results"), 1.0)
    assert "[cites other work]" not in claim_verifier._passage_block([sp2])


def test_evidence_only_mode_without_llm(monkeypatch):
    monkeypatch.setenv("ARSE_EMBEDDER", "none")
    cc = claim_analysis.CitedClaim("dense beats BM25", ["Dense retrieval recovers more than BM25."])
    [v] = claim_verifier.verify_batch([_claim("Dense beats BM25 [4].")], _ft(), cited=[cc],
                                      use_llm=False)
    assert v.verdict == "evidence_only" and v.evidence and v.model == ""


def test_fulltext_method_still_selectable(monkeypatch):
    monkeypatch.setenv("ARSE_CLAIM_METHOD", "fulltext")
    seen = []
    monkeypatch.setattr(claim_verifier, "_verify_batch_fulltext",
                        lambda claims, ft, idx: seen.append(len(claims)) or [])
    claim_verifier.verify_batch([_claim("x [4].")], _ft())
    assert seen == [1]


def test_retry_answers_are_renumbered_and_merged():
    sp = lambda i, t: evidence.ScoredPassage(evidence.Passage(i, t, "s"), 1.0)
    shown = [sp(0, "first"), sp(1, "second")]
    more = [sp(5, "third"), sp(6, "fourth")]
    parsed = {"claims": [{"verdict": "supported", "passage_ids": ["P1"]},
                         {"verdict": "not_addressed", "passage_ids": []}]}
    p2 = {"claims": [{"verdict": "contradicted", "passage_ids": ["P2"]}]}
    merged, all_shown = claim_verifier._merge_retry(parsed, shown, p2, more, [1])
    assert merged["claims"][0]["passage_ids"] == ["P1"]
    assert merged["claims"][1] == {"verdict": "contradicted", "passage_ids": ["P4"]}
    assert all_shown[3].passage.text == "fourth"


def test_default_embedder_falls_back_to_bm25_without_fastembed(monkeypatch):
    monkeypatch.delenv("ARSE_EMBEDDER", raising=False)
    monkeypatch.setattr(evidence, "_loaded", {})
    monkeypatch.setattr(evidence, "_failed", {})

    def boom():
        raise ImportError("No module named 'fastembed'")
    monkeypatch.setitem(evidence._EMBEDDERS, "bge-small", boom)
    assert evidence.get_embedder() is None
    assert "bge-small" in evidence.embedder_status()


def test_bm25_matches_hyphenated_model_names():
    ps = [evidence.Passage(0, "We present Gemma 2, a family of open models.", ""),
          evidence.Passage(1, "Unrelated text about geoparsing of tweets here.", "")]
    hits = evidence.retrieve("The cited work introduces Gemma-2-10B.", ps, k=1)
    assert hits and "Gemma 2" in hits[0].passage.text


def test_short_text_without_matches_is_shown_whole():
    ps = [evidence.Passage(0, "Completely different words appear in this abstract.", "")]
    hits = evidence.retrieve("zebra quantum", ps, k=8)
    assert [h.passage.idx for h in hits] == [0]


# ── Fewer false conflicts (LLM2Vec review case) ─────────────────────────────

LLM2VEC = ("Prior work (Huang et al., 2024; BehnamGhader et al., 2024) has highlighted that LLMs "
           "generally underperform in encoder roles, although some modifications can improve "
           "their effectiveness.")


def test_heuristic_keeps_the_qualifier_and_marks_shared_citations():
    cc = claim_analysis.heuristic(_claim(LLM2VEC, "(Huang et al., 2024; BehnamGhader et al., 2024)"))
    assert "although some modifications can improve" in cc.cited_claim
    assert cc.shared is True
    assert claim_analysis.heuristic(_claim("X improves Y [3].", "[3]")).shared is False
    assert claim_analysis.heuristic(_claim("X improves Y [3, 7].", "[3, 7]")).shared is True


def test_llm_shared_flag_is_read():
    fb = claim_analysis.CitedClaim("fallback", ["fallback"], "result")
    assert claim_analysis._coerce({"cited_claim": "a b c", "shared": True}, fb).shared is True
    assert claim_analysis._coerce({"cited_claim": "a b c", "shared": "false"}, fb).shared is False
    fb_shared = claim_analysis.CitedClaim("f", ["f"], "result", shared=True)
    assert claim_analysis._coerce({"cited_claim": "a b c"}, fb_shared).shared is True


def test_low_confidence_conflict_becomes_unclear(monkeypatch):
    monkeypatch.setenv("ARSE_EMBEDDER", "none")
    _fake_llm(monkeypatch, {"claims": [{"verdict": "contradicted", "passage_ids": ["P1"]}],
                            "confidence": "low"})
    cc = claim_analysis.CitedClaim("dense beats BM25", ["Dense retrieval recovers more than BM25."])
    [v] = claim_verifier.verify_batch([_claim("Dense beats BM25 [4].")], _ft(), cited=[cc])
    assert v.verdict == "ambiguous"


def test_shared_claims_are_flagged_to_the_model(monkeypatch):
    monkeypatch.setenv("ARSE_EMBEDDER", "none")
    calls = _fake_llm(monkeypatch, {"claims": [{"verdict": "supported", "passage_ids": ["P1"]}]})
    cc = claim_analysis.CitedClaim("dense beats BM25", ["Dense retrieval recovers more than BM25."],
                                   shared=True)
    claim_verifier.verify_batch([_claim("Dense beats BM25 (A, 2020; B, 2021).")], _ft(), cited=[cc])
    assert "SHARED:" in calls[0]


def test_passages_drop_markup_and_join_broken_paragraphs():
    md = ("## 1 Introduction\nDecoder-only LLMs lag behind encoder-\n\n"
          "only counterparts.<sup>1</sup> During pre-training, they learn from all tokens.\n")
    [p] = evidence.segment(md)
    assert p.text.startswith("Decoder-only LLMs lag behind encoder-only counterparts.")
    assert "<sup>" not in p.text


# ── Citation → reference assignment (author-year) ───────────────────────────

def _ay_refs():
    return [
        {"authors": "Beichen Zhang, Pan Zhang, Xiaoyi Dong", "year": "2024",
         "raw": "Beichen Zhang, Pan Zhang, and Xiaoyi Dong. 2024a. Long-clip.", "title": "Long-clip"},
        {"authors": "Yiming Zhang, Zhouhui Lian", "year": "2024",
         "raw": "Yiming Zhang and Zhouhui Lian. 2024a. Brush your text.", "title": "Brush your text"},
        {"authors": "Yiming Zhang, Zhouhui Lian", "year": "2024",
         "raw": "Yiming Zhang and Zhouhui Lian. 2024b. Diffste.", "title": "Diffste"},
    ]


def test_year_suffix_and_coauthor_pick_the_right_reference():
    from claim_extractor import extract
    body = ("Brush Your Text (Zhang and Lian, 2024a) refines alignment. DiffSTE (Zhang and Lian, "
            "2024b) edits scene text. Long-CLIP (Zhang et al., 2024a) extends CLIP context.")
    got = {c.citation_marker: c.cited_ref_index for c in extract(body, _ay_refs())}
    assert got["(Zhang and Lian, 2024a)"] == 1
    assert got["(Zhang and Lian, 2024b)"] == 2
    assert got["(Zhang et al., 2024a)"] == 0


def test_bracket_numbers_ignored_in_author_year_papers():
    from claim_extractor import extract
    body = ("Tianyu Zhang[1], Xinyu Wang[2], Zhenghan Tai[3] wrote this. "
            "Brush Your Text (Zhang and Lian, 2024a) refines alignment. "
            "DiffSTE (Zhang and Lian, 2024b) edits text. Long-CLIP (Zhang et al., 2024a) helps.")
    styles = {c.style for c in extract(body, _ay_refs())}
    assert "numeric" not in styles
