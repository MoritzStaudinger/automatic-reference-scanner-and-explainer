"""
Evaluate the citation-context check on the citation-integrity corpus of
Sarol et al. (2024, Bioinformatics; github.com/ScienceNLP-Lab/Citation-Integrity,
MIT licence): citing sentences from PMC articles, each annotated against
the FULL TEXT of the cited article with evidence sentences and a label.

Data (not in git):
    git clone --depth 1 https://github.com/ScienceNLP-Lab/Citation-Integrity \\
        benchmark/claims/data/citation-integrity

Two evaluations:

  retrieval  Is a gold evidence sentence among the top-k passages?
             recall@1/3/5/8 per embedder.  No LLM, no cost.
                 python benchmark/claims/eval_citation_integrity.py retrieval \\
                     --embedders none,bge-small,scincl,specter2

  verdict    Old method (whole paper to the LLM) vs new (claim extraction +
             passages).  Binary: ACCURATE vs NOT_ACCURATE (contradict, not
             substantiate, oversimplify, misquote); citation-practice labels
             (indirect, etiquette) and claims without evidence are left out.
                 python benchmark/claims/eval_citation_integrity.py verdict \\
                     --methods fulltext,passages --limit 60

Results go to benchmark/claims/runs/<timestamp>_<mode>/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "benchmark/claims/data/citation-integrity/Data/multivers-format"
RUNS = ROOT / "benchmark/claims/runs"

NOT_ACCURATE = {"CONTRADICT", "NOT_SUBSTANTIATE", "OVERSIMPLIFY", "MISQUOTE"}
PRACTICE = {"INDIRECT", "INDIRECT_NOT_REVIEW", "ETIQUETTE", "IRRELEVANT"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(split: str):
    corpus = {}
    with open(DATA / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            corpus[int(d["doc_id"])] = d
    claims = []
    with open(DATA / f"claims-{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            claims.append(json.loads(line))
    return claims, corpus


def article_text(claim: dict, corpus: dict) -> str:
    """The cited article = its chunks in id order, one paragraph each."""
    paras = []
    for did in sorted(int(x) for x in claim.get("cited_doc_ids") or []):
        d = corpus.get(did)
        if d:
            title = (d.get("title") or "").strip()
            body = " ".join(s.strip() for s in d.get("abstract") or [])
            paras.append((f"## {title}\n" if title else "") + body)
    return "\n\n".join(paras)


def gold_sentences(claim: dict, corpus: dict) -> list[str]:
    out = []
    for did, evs in (claim.get("evidence") or {}).items():
        sents = (corpus.get(int(did)) or {}).get("abstract") or []
        for ev in evs:
            for i in ev.get("sentences") or []:
                if 0 <= i < len(sents):
                    out.append(sents[i])
    return out


def binary_label(claim: dict) -> str | None:
    """ACCURATE if every evidence label is ACCURATE, NOT_ACCURATE if any is a
    content error; None for no evidence or citation-practice labels only."""
    labels = {ev.get("label") for evs in (claim.get("evidence") or {}).values() for ev in evs}
    if labels & NOT_ACCURATE:
        return "NOT_ACCURATE"
    if labels == {"ACCURATE"}:
        return "ACCURATE"
    return None


_MARK_RE = re.compile(r"<\|(multi_)?cit\|>")


def citing_sentence(claim: dict) -> tuple[str, str]:
    """Replace the corpus's tags by numeric markers: <|cit|> / <|multi_cit|>
    cite the article under test, <|other_cit|> some other reference."""
    text = claim["claim"].replace("<|other_cit|>", " [9]")
    marker = "[1, 2]" if "<|multi_cit|>" in text else "[1]"
    text = _MARK_RE.sub(lambda m: " [1, 2]" if m.group(1) else " [1]", text, count=1)
    text = _MARK_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(), marker


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def run_retrieval(split: str, embedders: list[str], ks=(1, 3, 5, 8), limit: int = 0):
    import evidence
    from claim_analysis import heuristic
    from claim_extractor import Claim

    claims, corpus = load(split)
    claims = [c for c in claims if gold_sentences(c, corpus)]
    if limit:
        claims = claims[:limit]
    seg_cache: dict[str, list] = {}
    results = {}
    for name in embedders:
        emb = evidence.get_embedder(name) if name != "none" else None
        if name != "none" and emb is None:
            print(f"!! embedder {name} unavailable: {evidence.embedder_status()}")
            continue
        hits = Counter()
        t0 = time.time()
        for c in claims:
            text = article_text(c, corpus)
            if text not in seg_cache:
                seg_cache[text] = evidence.segment(text)
            passages = seg_cache[text]
            sent, marker = citing_sentence(c)
            query = heuristic(Claim(sent, marker, 0, "numeric")).cited_claim
            top = evidence.retrieve(query, passages, k=max(ks), embedder=emb)
            gold = [_norm(g) for g in gold_sentences(c, corpus)]
            for k in ks:
                window = " ".join(_norm(sp.passage.text) for sp in top[:k])
                if any(g and g[:120] in window for g in gold):
                    hits[k] += 1
        n = len(claims)
        results[name] = {f"recall@{k}": round(hits[k] / n, 3) for k in ks}
        results[name]["n"] = n
        results[name]["seconds"] = round(time.time() - t0, 1)
        print(name, results[name], flush=True)
    return results


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def run_verdict(split: str, methods: list[str], limit: int, embedder: str):
    import claim_verifier
    from claim_analysis import analyse
    from claim_extractor import Claim
    from fulltext import FullText

    os.environ["ARSE_EMBEDDER"] = embedder
    claims, corpus = load(split)
    items = [(c, binary_label(c)) for c in claims]
    items = [(c, y) for c, y in items if y]
    # Keep all NOT_ACCURATE cases (the rare class) and fill with ACCURATE.
    if limit:
        bad = [it for it in items if it[1] == "NOT_ACCURATE"]
        good = [it for it in items if it[1] == "ACCURATE"]
        n_bad = min(len(bad), limit // 2)
        items = bad[:n_bad] + good[:limit - n_bad]
    rows = []
    for method in methods:
        os.environ["ARSE_CLAIM_METHOD"] = method
        t0 = time.time()
        prompt_chars = 0
        for c, y in items:
            sent, marker = citing_sentence(c)
            claim = Claim(sent, marker, 0, "numeric")
            text = article_text(c, corpus)
            ft = FullText(text=text, source="citation-integrity", is_abstract_fallback=False, url=None)
            cited = analyse([claim], [{"title": ""}]) if method == "passages" else None
            t1 = time.time()
            [v] = claim_verifier.verify_batch([claim], ft, cited=cited)
            prompt_chars += (min(len(text), 120_000) if method == "fulltext"
                             else sum(len(e.get("text", "")) for e in v.evidence) + 4000)
            pred = "ACCURATE" if v.verdict == "supported" else (
                "ERROR" if v.verdict == "error" else "NOT_ACCURATE")
            rows.append({"id": c["id"], "method": method, "gold": y, "pred": pred,
                         "verdict": v.verdict, "intent": v.intent, "cited_claim": v.cited_claim,
                         "explanation": v.explanation, "seconds": round(time.time() - t1, 2),
                         "error": v.error or ""})
            print(f"{method:9s} {c['id']:5d} gold={y:12s} verdict={v.verdict}", flush=True)
        rows.append({"method": method, "_summary": True,
                     "seconds_total": round(time.time() - t0, 1),
                     "approx_prompt_tokens": prompt_chars // 4})
    return rows


def score(rows: list[dict]) -> dict:
    out = {}
    by = defaultdict(list)
    for r in rows:
        if not r.get("_summary"):
            by[r["method"]].append(r)
    for m, rs in by.items():
        rs_ok = [r for r in rs if r["pred"] != "ERROR"]
        cm = Counter((r["gold"], r["pred"]) for r in rs_ok)
        f1s = {}
        for lab in ("ACCURATE", "NOT_ACCURATE"):
            tp = cm[(lab, lab)]
            fp = sum(v for (g, p), v in cm.items() if p == lab and g != lab)
            fn = sum(v for (g, p), v in cm.items() if g == lab and p != lab)
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1s[lab] = {"precision": round(pr, 3), "recall": round(rc, 3),
                        "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0}
        acc = sum(cm[(l, l)] for l in ("ACCURATE", "NOT_ACCURATE")) / max(1, len(rs_ok))
        summ = next((r for r in rows if r.get("_summary") and r["method"] == m), {})
        out[m] = {"n": len(rs), "errors": len(rs) - len(rs_ok), "accuracy": round(acc, 3),
                  "macro_f1": round((f1s["ACCURATE"]["f1"] + f1s["NOT_ACCURATE"]["f1"]) / 2, 3),
                  "per_class": f1s, "confusion": {f"{g}->{p}": v for (g, p), v in cm.items()},
                  "verdicts": dict(Counter(r["verdict"] for r in rs)),
                  "seconds_total": summ.get("seconds_total"),
                  "approx_prompt_tokens": summ.get("approx_prompt_tokens")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("retrieval", "verdict"))
    ap.add_argument("--split", default="test", choices=("train", "dev", "test"))
    ap.add_argument("--embedders", default="none,bge-small,scincl,specter2")
    ap.add_argument("--embedder", default="none", help="dense retriever for the verdict run")
    ap.add_argument("--methods", default="fulltext,passages")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_dir = RUNS / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "retrieval":
        res = run_retrieval(args.split, [e.strip() for e in args.embedders.split(",")],
                            limit=args.limit)
        (out_dir / "summary.json").write_text(json.dumps(res, indent=1))
    else:
        rows = run_verdict(args.split, [m.strip() for m in args.methods.split(",")],
                           args.limit, args.embedder)
        with open(out_dir / "rows.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        res = score(rows)
        (out_dir / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("→", out_dir)


if __name__ == "__main__":
    main()
