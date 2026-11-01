import json
from types import SimpleNamespace

import llm_parser


def test_numbers_and_lists_coerced_to_str():
    out = llm_parser._parse_response_text(json.dumps([
        {"is_citation": True, "title": "T", "year": 2020, "authors": ["A Smith", "B Jones"]},
        {"title": 12.0},
    ]), 2)
    assert out[0].year == "2020" and out[0].authors == "A Smith, B Jones"
    assert out[1].title == "12"


def test_malformed_item_does_not_fail_batch():
    out = llm_parser._parse_response_text(json.dumps([{"is_citation": {"x": 1}}, {"title": "ok"}]), 2)
    assert out[1].title == "ok"


def test_wrapped_object_and_padding():
    out = llm_parser._parse_response_text(json.dumps({"references": [{"title": "a"}]}), 3)
    assert [o.title for o in out] == ["a", None, None]


class _FakeClient:
    def __init__(self, blocks, stop="end_turn"):
        self.kwargs = None
        self._msg = SimpleNamespace(content=blocks, stop_reason=stop)
        self.messages = self

    def create(self, **kw):
        self.kwargs = kw
        return self._msg


def test_anthropic_skips_thinking_block_and_disables_thinking_on_haiku():
    blocks = [SimpleNamespace(type="thinking", thinking=""),
              SimpleNamespace(type="text", text='[{"title": "Fixed"}]')]
    c = _FakeClient(blocks)
    out = llm_parser._call_anthropic(c, [{"raw": "x"}], "claude-haiku-5-5")
    assert out[0].title == "Fixed"
    assert c.kwargs["thinking"] == {"type": "disabled"}
    c2 = _FakeClient(blocks)
    llm_parser._call_anthropic(c2, [{"raw": "x"}], "claude-opus-5-5")
    assert "thinking" not in c2.kwargs


def test_failed_batches_reported_in_stats(monkeypatch):
    monkeypatch.setattr(llm_parser, "_make_client", lambda backend=None: object())

    def boom(*a, **k):
        raise RuntimeError("credit balance too low")
    monkeypatch.setattr(llm_parser, "_call_llm", boom)
    stats = {}
    refs = [{"raw": "r", "title": "t"}]
    out = llm_parser.parse_references_with_llm(refs, stats=stats, backend="anthropic")
    assert out == refs
    assert stats["n_failed_refs"] == 1 and "credit" in stats["first_error"]


def test_resolve_backend(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AQUEDUCT_API_KEY", "x")
    assert llm_parser.resolve_backend() == "aqueduct"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    assert llm_parser.resolve_backend() == "anthropic"
    monkeypatch.setenv("LLM_BACKEND", "Aqueduct")
    assert llm_parser.resolve_backend() == "aqueduct"


def test_numeric_claims_survive_dedupe_and_manual_refs():
    from claim_extractor import extract
    from parsers import dedupe_references
    refs = [{"raw": "A. One. Alpha paper. 2020.", "title": "Alpha paper"},
            {"raw": "B. Two. Beta paper. 2021.", "title": "Beta paper"},
            {"raw": "B. Two. Beta paper. 2021", "title": "Beta paper."},   # dup of #2
            {"raw": "C. Three. Gamma paper. 2022.", "title": "Gamma paper"}]
    refs, n = dedupe_references(refs)
    assert n == 1 and len(refs) == 3
    refs.append({"raw": "manual", "title": "Manual", "_user_added": True})
    claims = extract("Gamma results are strong [4]. Beta is related [3].", refs)
    by_marker = {c.citation_marker: refs[c.cited_ref_index]["title"].rstrip(".") for c in claims}
    assert by_marker == {"[4]": "Gamma paper", "[3]": "Beta paper"}


def test_numeric_claims_use_raw_labels():
    from claim_extractor import extract
    refs = [{"raw": "[2] B. Beta.", "title": "Beta"}, {"raw": "[5] E. Eps.", "title": "Eps"}]
    claims = extract("As shown before [5].", refs)
    assert refs[claims[0].cited_ref_index]["title"] == "Eps"


def test_hybrid_raises_parser_error_when_grobid_down(monkeypatch, tmp_path):
    import parsers
    monkeypatch.setattr(parsers, "GROBID_URL", "http://127.0.0.1:9")   # closed port
    monkeypatch.setattr(parsers.time, "sleep", lambda s: None)
    import fitz
    pdf = tmp_path / "x.pdf"
    doc = fitz.open(); doc.new_page().insert_text((72, 72), "References\n[1] A. B. Title. 2020."); doc.save(pdf)
    import pytest
    with pytest.raises(parsers.ParserError):
        parsers.extract_references_hybrid(str(pdf))


def test_fallback_backend_recovers_failed_batches(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_BACKEND", "anthropic")
    monkeypatch.setattr(llm_parser, "_make_client", lambda backend=None: object())

    def call(client, batch, backend=None, model=None):
        if backend == "aqueduct":
            raise RuntimeError("500 connection error")
        return [llm_parser._ParsedRef(title="Fixed title of a paper") for _ in batch]
    monkeypatch.setattr(llm_parser, "_call_llm", call)
    stats = {}
    out = llm_parser.parse_references_with_llm(
        [{"raw": "A. Author. Fixed title of a paper. Venue, 2020.", "title": "t"}],
        stats=stats, backend="aqueduct")
    assert out[0]["title"] == "Fixed title of a paper"
    assert stats["n_failed_refs"] == 0 and "fallback anthropic" in stats["fallback"]


def test_no_fallback_without_opt_in(monkeypatch):
    monkeypatch.delenv("LLM_FALLBACK_BACKEND", raising=False)
    monkeypatch.setattr(llm_parser, "_make_client", lambda backend=None: object())
    monkeypatch.setattr(llm_parser, "_call_llm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    stats = {}
    llm_parser.parse_references_with_llm([{"raw": "r"}], stats=stats, backend="aqueduct")
    assert stats["n_failed_refs"] == 1 and stats["fallback"] is None


def test_circuit_breaker_skips_down_backend(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_BACKEND", "anthropic")
    monkeypatch.setattr(llm_parser, "_make_client", lambda backend=None: object())
    calls = []

    class InternalServerError(Exception):
        pass

    def call(client, batch, backend=None, model=None):
        calls.append(backend)
        if backend == "aqueduct":
            raise InternalServerError("500")
        return [llm_parser._ParsedRef(title="ok") for _ in batch]
    monkeypatch.setattr(llm_parser, "_call_llm", call)
    monkeypatch.setattr(llm_parser, "_DOWN_UNTIL", {})
    llm_parser.parse_references_with_llm([{"raw": "r"}], backend="aqueduct")
    assert llm_parser.is_down("aqueduct")
    calls.clear()
    llm_parser.parse_references_with_llm([{"raw": "r"}], backend="aqueduct")
    assert calls == ["anthropic"]          # primary skipped while down


# ── Grounding of repaired values ────────────────────────────────────────────

def test_repaired_values_must_appear_in_the_citation(monkeypatch):
    raw = ("[10] Peter G. Anick and Rex A. Flynn. Versioning a full-text information "
           "retrieval system. In Proceedings of SIGIR, pages 98-111, 1992.")
    monkeypatch.setattr(llm_parser, "_make_client", lambda backend=None: object())
    monkeypatch.setattr(llm_parser, "_call_llm", lambda c, batch, b=None, m=None: [
        llm_parser._ParsedRef(title="Versioning a full-text information retrieval system",
                              authors="Peter G. Anick, Rex A. Flynn",
                              venue="BMJ Open",                    # invented
                              doi="10.1145/133160.133183",         # not printed
                              year="1992")])
    out = llm_parser.parse_references_with_llm(
        [{"raw": raw, "title": "Versioning", "authors": "G Peter, Rex Anick, Flynn",
          "venue": "Proceedings of SIGIR"}], backend="anthropic")[0]
    assert out["authors"] == "Peter G. Anick, Rex A. Flynn"
    assert out["title"] == "Versioning a full-text information retrieval system"
    assert out["venue"] == "Proceedings of SIGIR"
    assert not out.get("doi")


def test_shortened_author_list_is_rejected():
    raw = "A. One, B. Two, C. Three, and D. Four. Title. 2020."
    assert not llm_parser._grounded("authors", "A. One, et al.", raw,
                                    "A. One, B. Two, C. Three, D. Four")
    assert llm_parser._grounded("authors", "A. One, B. Two, C. Three, D. Four", raw, "")
    assert llm_parser._grounded("url", "https://arxiv.org/abs/2311.12474",
                                "Kusa. Csmed. arXiv preprint arXiv:2311.12474, 2023.", "")
