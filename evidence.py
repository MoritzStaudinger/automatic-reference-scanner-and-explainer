"""
Evidence retrieval from a cited paper's full text.

The claim check used to hand the whole cited paper (up to 120 000
characters) to the LLM.  Here the paper is cut into short passages and
only the few that best match a claim are passed on, so the verdict step
sees ~2 000 tokens instead of ~30 000 and every piece of evidence it
cites is a real passage of the paper.

    passages = segment(fulltext)                 # section-tagged, refs dropped
    hits     = retrieve(claim, passages, key=…)  # BM25 ⊕ dense, fused by rank

Dense embedders are optional and pluggable (``ARSE_EMBEDDER``):

    specter2   allenai/specter2 with the ad-hoc query / proximity adapters
    scincl     malteos/scincl
    bge-small  BAAI/bge-small-en-v1.5 via fastembed (ONNX, no PyTorch)
    none       BM25 only

Missing packages or models fall back to BM25 instead of failing.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

# ---------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------

@dataclass
class Passage:
    idx:     int
    text:    str
    section: str            # nearest heading above, "" if none


@dataclass
class ScoredPassage:
    passage:    Passage
    score:      float                   # fused score (higher = better)
    bm25_rank:  Optional[int] = None    # 1-based, None = not in that list
    dense_rank: Optional[int] = None


# Headings in pymupdf4llm markdown: "## 3 Method", "**Abstract**", or a bare
# numbered line "4.2 Results".
_MD_HEADING_RE  = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_BOLD_LINE_RE   = re.compile(r"^\s*\*\*([^*]{2,90})\*\*\s*$")
_NUM_HEADING_RE = re.compile(r"^\s*((?:\d+|[A-Z])(?:\.\d+)*\.?\s+[A-Z][^.!?]{1,80})$")

# Sections of the cited paper that describe *other* work or no work at all.
_DROP_SECTION_RE = re.compile(
    r"^(?:[\dA-Z]+(?:\.\d+)*\.?\s+)?(references|bibliography|literature cited|works cited|"
    r"acknowledg|funding|author contributions|conflicts? of interest|competing interests)",
    re.I)

# Sentence end: ". " before a capital / digit / bracket, not after common
# abbreviations or single initials (same idea as claim_extractor._SENT_END_RE).
_SENT_END_RE = re.compile(
    r"(?<![A-Z])(?<!\bal)(?<!\be\.g)(?<!\bi\.e)(?<!\bcf)(?<!\bvs)(?<!\bFig)(?<!\bEq)"
    r"(?<!\bSec)(?<!\bTab)(?<!\bet)(?<!\bno)[.!?](?=\s+[A-Z\d\[\(])")

_WINDOW = 3          # sentences per passage
_STRIDE = 2          # overlap of one sentence
_MAX_PASSAGES = 1500


# pymupdf4llm placeholders for figures ("**==> picture [455 x 181] intentionally
# omitted <==**") and their extracted text blocks carry no prose.
_PLACEHOLDER_RE = re.compile(r"==>.*?<==|^\s*\*{0,2}-{3,}\s*(?:Start|End) of picture text.*$", re.I)
# Running headers / footers that pymupdf4llm keeps as separate lines.
_FURNITURE_RE = re.compile(r"Manuscript submitted to|^\W*(?:preprint|under review)\b|"
                           r"^\W*page \d+(?: of \d+)?\W*$", re.I)
# Line numbers of review manuscripts in front of headings: "928 6.1 Verification".
_LINE_NO_RE = re.compile(r"^\d{2,4}\s+(?=(?:\d+(?:\.\d+)*|[A-Z])\.?\s+[A-Z])")


def _heading(line: str) -> Optional[str]:
    for rx in (_MD_HEADING_RE, _BOLD_LINE_RE):
        m = rx.match(line)
        if m:
            return _LINE_NO_RE.sub("", re.sub(r"[*_`]", "", m.group(1))).strip()
    line = _LINE_NO_RE.sub("", line)
    m = _NUM_HEADING_RE.match(line)
    if m and len(line.split()) <= 10:
        return m.group(1).strip()
    return None


def _clean(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)          # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)        # links → label
    text = re.sub(r"(\*\*|__|`)", "", text)
    text = re.sub(r"(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)", "", text)
    text = re.sub(r"(\w)-\s+([a-z])", r"\1\2", text)            # line-break hyphens
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = _clean(text)
    if not text:
        return []
    out, start = [], 0
    for m in _SENT_END_RE.finditer(text):
        s = text[start:m.end()].strip()
        if s:
            out.append(s)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    # Very short fragments (page numbers, "Fig. 3") carry no evidence.
    return [s for s in out if len(s) >= 15 and len(s.split()) >= 3]


def _paragraphs_with_sections(text: str) -> list[tuple[str, str]]:
    """(section, paragraph text) pairs; drops tables and dropped sections."""
    out: list[tuple[str, str]] = []
    section, buf, dropping = "", [], False

    def flush():
        if buf and not dropping:
            out.append((section, " ".join(buf)))
        buf.clear()

    for line in text.splitlines():
        s = _PLACEHOLDER_RE.sub("", line).strip()
        if not s or s in ("**", "****"):
            if not line.strip():
                flush()
            continue
        if not s:
            flush()
            continue
        if _FURNITURE_RE.search(s):
            continue
        h = _heading(s)
        if h is not None:
            flush()
            section = h
            dropping = bool(_DROP_SECTION_RE.match(h))
            continue
        if s.startswith("|") or re.fullmatch(r"[-|:\s]+", s):   # markdown tables
            continue
        buf.append(s)
    flush()
    return out


def segment(text: str, is_abstract: bool = False) -> list[Passage]:
    """Cut a paper (pymupdf4llm markdown or plain text) into overlapping
    passages of ``_WINDOW`` sentences that never cross a section boundary."""
    window, stride = (2, 1) if is_abstract else (_WINDOW, _STRIDE)
    passages: list[Passage] = []
    # Merge consecutive paragraphs of one section, then window over sentences.
    by_section: list[tuple[str, list[str]]] = []
    for section, para in _paragraphs_with_sections(text or ""):
        sents = split_sentences(para)
        if not sents:
            continue
        if by_section and by_section[-1][0] == section:
            by_section[-1][1].extend(sents)
        else:
            by_section.append((section, sents))
    for section, sents in by_section:
        if len(sents) <= window:
            starts = [0]
        else:
            starts = list(range(0, len(sents) - window + 1, stride))
            if starts[-1] + window < len(sents):
                starts.append(len(sents) - window)
        for st in starts:
            passages.append(Passage(len(passages), " ".join(sents[st:st + window]), section))
            if len(passages) >= _MAX_PASSAGES:
                return passages
    return passages


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_STOP = frozenset("""a an and are as at be been but by can could did do does for from had has
have he her his how i if in into is it its may might more most no not of on or our she should
so such than that the their them then there these they this those to was we were what when
where which while who will with would you your also et al using use used based show shows
shown""".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())
            if t not in _STOP and len(t) > 1]


def bm25_scores(query: str, passages: Sequence[Passage],
                k1: float = 1.5, b: float = 0.75) -> list[float]:
    docs = [_tokens(p.text) for p in passages]
    if not docs:
        return []
    avgdl = sum(map(len, docs)) / len(docs) or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    q = _tokens(query)
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores


# ---------------------------------------------------------------------------
# Dense embedders
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str], kind: str) -> "list[list[float]]":
        """``kind`` is "query" (the claim) or "passage".  Vectors are
        L2-normalised so a dot product is the cosine similarity."""


def _model_dir() -> Optional[str]:
    return os.getenv("ARSE_MODEL_DIR") or None


class _HFEncoder:
    """CLS-pooled transformer encoder (SciNCL, SPECTER2 base)."""

    def __init__(self, model_name: str, max_len: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, cache_dir=_model_dir())
        self.model = AutoModel.from_pretrained(model_name, cache_dir=_model_dir()).eval()
        self.max_len = max_len

    def _encode(self, texts: list[str], batch: int = 32):
        import numpy as np
        torch = self._torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
                cls = self.model(**enc).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 768), dtype="float32")


class SciNCLEmbedder(_HFEncoder):
    name = "scincl"

    def __init__(self):
        super().__init__("malteos/scincl")

    def encode(self, texts, kind):
        return self._encode(texts)


class Specter2Embedder:
    """SPECTER2: the ad-hoc query adapter for claims, the proximity adapter
    for passages — the pairing AllenAI recommends for query → document."""
    name = "specter2"

    def __init__(self):
        import torch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
        self._torch = torch
        base = "allenai/specter2_base"
        self.tok = AutoTokenizer.from_pretrained(base, cache_dir=_model_dir())
        self.model = AutoAdapterModel.from_pretrained(base, cache_dir=_model_dir()).eval()
        self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
        self.model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query")

    def encode(self, texts, kind):
        import numpy as np
        torch = self._torch
        self.model.set_active_adapters("adhoc_query" if kind == "query" else "proximity")
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 32):
                enc = self.tok(texts[i:i + 32], padding=True, truncation=True,
                               max_length=256, return_tensors="pt")
                cls = self.model(**enc).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 768), dtype="float32")


class BgeSmallEmbedder:
    name = "bge-small"

    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=_model_dir())

    def encode(self, texts, kind):
        import numpy as np
        fn = self.model.query_embed if kind == "query" else self.model.passage_embed
        vecs = np.array(list(fn(texts)), dtype="float32")
        if len(vecs):
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs


_EMBEDDERS = {"specter2": Specter2Embedder, "scincl": SciNCLEmbedder, "bge-small": BgeSmallEmbedder}
_loaded: dict[str, object] = {}
_failed: dict[str, str] = {}
_load_lock = threading.Lock()


def get_embedder(name: Optional[str] = None) -> Optional[Embedder]:
    """The configured embedder, loaded once per process; None for BM25 only
    (``none``, unknown name, or packages / model files unavailable)."""
    name = (name or os.getenv("ARSE_EMBEDDER") or "none").strip().lower()
    if name in ("", "none", "bm25") or name not in _EMBEDDERS:
        return None
    with _load_lock:
        if name in _loaded:
            return _loaded[name]          # type: ignore[return-value]
        if name in _failed:
            return None
        try:
            _loaded[name] = _EMBEDDERS[name]()
        except Exception as exc:          # missing package, no network for the model…
            _failed[name] = f"{type(exc).__name__}: {exc}"[:300]
            import sys
            print(f"[evidence] embedder {name!r} unavailable, using BM25 only: "
                  f"{_failed[name]}", file=sys.stderr, flush=True)
            return None
        return _loaded[name]              # type: ignore[return-value]


def embedder_status() -> dict[str, str]:
    return {**{k: "loaded" for k in _loaded}, **_failed}


# Passage vectors are cached per (paper text, model): every claim citing the
# same paper reuses them.  Stored as base64 float16 in the JSON cache.

def _vec_key(passages: Sequence[Passage], model: str) -> str:
    h = hashlib.sha256(model.encode())
    for p in passages:
        h.update(b"\0" + p.text.encode("utf-8"))
    return h.hexdigest()[:32]


def passage_vectors(passages: Sequence[Passage], embedder: Embedder):
    import numpy as np
    key = _vec_key(passages, embedder.name)
    try:
        from cache import get_cache
        hit = get_cache().get("passage_emb", key)
    except Exception:
        hit = None
    if hit:
        arr = np.frombuffer(base64.b64decode(hit["v"]), dtype="float16").astype("float32")
        return arr.reshape(hit["n"], -1)
    vecs = np.asarray(embedder.encode([p.text for p in passages], "passage"), dtype="float32")
    try:
        from cache import get_cache
        get_cache().set("passage_emb", key, {
            "n": int(vecs.shape[0]),
            "v": base64.b64encode(vecs.astype("float16").tobytes()).decode("ascii")})
    except Exception:
        pass
    return vecs


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_RRF_K = 60


def _ranks(scores: Sequence[float], positive_only: bool) -> dict[int, int]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    if positive_only:
        order = [i for i in order if scores[i] > 0]
    return {i: r + 1 for r, i in enumerate(order)}


def retrieve(query: str, passages: Sequence[Passage], k: int = 8,
             embedder: Optional[Embedder] = None, skip: int = 0,
             candidates: int = 100) -> list[ScoredPassage]:
    """Top ``k`` passages for ``query`` (after skipping the first ``skip``),
    by reciprocal-rank fusion of BM25 and — when an embedder is given —
    dense cosine similarity."""
    if not passages or not query.strip():
        return []
    bm = _ranks(bm25_scores(query, passages), positive_only=True)
    dense: dict[int, int] = {}
    if embedder is not None:
        try:
            import numpy as np
            pv = passage_vectors(passages, embedder)
            qv = np.asarray(embedder.encode([query], "query"), dtype="float32")[0]
            dense = _ranks(list(pv @ qv), positive_only=False)
        except Exception as exc:
            import sys
            print(f"[evidence] dense retrieval failed, BM25 only: {exc}", file=sys.stderr)
    fused: dict[int, float] = {}
    for ranks in (bm, dense):
        for i, r in ranks.items():
            if r <= candidates:
                fused[i] = fused.get(i, 0.0) + 1.0 / (_RRF_K + r)
    order = sorted(fused, key=lambda i: -fused[i])[skip:skip + k]
    return [ScoredPassage(passages[i], fused[i], bm.get(i), dense.get(i)) for i in order]
