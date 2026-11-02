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
    assert agg(["supported", "contradicted"]) == "contradicted"
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
