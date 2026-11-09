"""
Hand-reviewed citation-context cases, re-run with the configured LLM.

Each line of regression_cases.jsonl: a citing sentence, its marker, the
cited paper's arXiv ID, and either "expect" (allowed verdicts) or "not"
(forbidden verdicts).  Needs network access (arXiv) and an LLM, so it is
not part of pytest.

    python benchmark/claims/run_regression.py            # all cases
    LLM_BACKEND=anthropic python benchmark/claims/run_regression.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import claim_analysis                        # noqa: E402
import claim_verifier                        # noqa: E402
from claim_extractor import Claim            # noqa: E402
from fulltext import FullText, _http_get_pdf, _pdf_bytes_to_text   # noqa: E402

_cache: dict[str, FullText] = {}


def _arxiv_fulltext(aid: str) -> FullText:
    if aid not in _cache:
        pdf = _http_get_pdf(f"https://arxiv.org/pdf/{aid}")
        text = _pdf_bytes_to_text(pdf) if pdf else None
        if not text:
            raise RuntimeError(f"could not fetch arXiv {aid}")
        _cache[aid] = FullText(text=text, source="arxiv", is_abstract_fallback=False,
                               url=f"https://arxiv.org/abs/{aid}")
    return _cache[aid]


def main() -> int:
    cases = [json.loads(l) for l in open(Path(__file__).with_name("regression_cases.jsonl"))
             if l.strip()]
    failed = 0
    for c in cases:
        claim = Claim(c["sentence"], c["marker"], 0, "paren")
        [cc] = claim_analysis.analyse([claim], [{"title": c.get("cited_title", "")}])
        [v] = claim_verifier.verify_batch([claim], _arxiv_fulltext(c["arxiv"]), cited=[cc])
        ok = (v.verdict in c["expect"]) if "expect" in c else (v.verdict not in c["not"])
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {c['id']:24s} verdict={v.verdict:20s} model={v.model}")
        print(f"      claim: {cc.cited_claim[:140]}  (intent {cc.intent}, shared {cc.shared})")
        print(f"      why:   {v.explanation[:220]}")
    print(f"{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
