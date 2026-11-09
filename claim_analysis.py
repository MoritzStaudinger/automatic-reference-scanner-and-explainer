"""
What does a citing sentence claim about the cited work?

A citing sentence often mixes the authors' own statement with several
citations ("X improves Y [3], unlike Z [7]").  Checking the whole
sentence against paper [3] asks it to support things it never said.
This step isolates the part attributed to ONE reference, splits it into
atomic sub-claims, and classifies the citation's intent:

    result      the sentence reports a finding / property of the cited work
                → every sub-claim is checked strictly
    method      the citing paper uses a method, dataset or tool from it
                → only needs the cited work to describe that thing
    background  a pointer to related work ("e.g.", surveys, lists)
                → only needs the cited work to be about the topic

With an LLM: one batched call per citing paper (cached).  Without one,
or when the call fails: the clause around the citation marker and
cue-phrase rules.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from claim_extractor import Claim

INTENTS = ("result", "method", "background")


@dataclass
class CitedClaim:
    cited_claim: str                       # what the sentence says about this reference
    subclaims:   list[str] = field(default_factory=list)
    intent:      str = "result"
    source:      str = "heuristic"         # "llm" | "heuristic"
    # Several works cited together for one statement ("Prior work (A; B)
    # has shown X"): each needs only to be consistent with X.
    shared:      bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Heuristic analysis (no LLM)
# ---------------------------------------------------------------------------

_ANY_MARKER_RE = re.compile(
    r"\[\s*\d[\d\s,\-–]*\]"                                  # [3], [3, 7-9]
    r"|\((?:[^()]*?\b(?:19|20)\d{2}[a-z]?)(?:;[^()]*?\b(?:19|20)\d{2}[a-z]?)*\)"   # (Smith, 2020; …)
)
# Clause boundaries that separate statements about DIFFERENT works.
# "although" / "however" are not among them: they qualify the same
# statement ("LLMs underperform as encoders, although modifications help"),
# and dropping that half turned a fair summary into an overstatement.
_CLAUSE_SPLIT_RE = re.compile(
    r";\s+|,\s+(?:while|whereas|unlike|but|in contrast(?: to)?)\s+"
    r"|\s+(?:whereas|while)\s+")


def _is_shared_marker(marker: str) -> bool:
    """"(Huang et al., 2024; BehnamGhader et al., 2024)", "[3, 7]", "[2-4]"."""
    m = marker or ""
    return (";" in m and "(" in m) or bool(re.search(r"\[\s*\d+\s*[,\-–]", m))

_METHOD_RE = re.compile(
    r"\b(?:we|our)\b[^.;]{0,60}?\b(?:use[sd]?|using|adopt(?:ed|s)?|employ(?:ed|s)?|follow(?:ed|s|ing)?|"
    r"build(?:s)? on|built on|based on|implement(?:ed|s)?|appl(?:y|ied|ies)|fine-?tun(?:e|ed)|"
    r"initiali[sz]ed?|train(?:ed)? (?:on|with)|evaluate[ds]? on|borrow(?:ed)?)\b"
    r"|\b(?:following|as in|as proposed (?:by|in)|implemented (?:in|by)|available (?:at|from)|"
    r"we refer to)\b", re.I)
_BACKGROUND_RE = re.compile(
    r"\b(?:e\.g\.|i\.a\.|such as|for example|for instance|see\b|cf\.|"
    r"for (?:a |an )?(?:survey|overview|review)|have been (?:proposed|studied|explored|used)|"
    r"(?:several|many|numerous|various|prior|previous|recent|earlier|other) "
    r"(?:works?|studies|approaches|methods|papers|efforts))", re.I)


def _strip_markers(text: str) -> str:
    text = _ANY_MARKER_RE.sub("", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;")


def heuristic(claim: Claim) -> CitedClaim:
    sent = claim.text
    pos = sent.find(claim.citation_marker) if claim.citation_marker else -1
    clause = sent
    if pos >= 0:
        # The clause holding the marker, without the connective
        # ("X improves Y [3], unlike Z [7]" → "X improves Y").
        # Semicolons inside citation parentheses "(A, 2024; B, 2024)" are
        # not clause boundaries: mask them (same length, positions kept).
        masked = _ANY_MARKER_RE.sub(lambda m: m.group(0).replace(";", ","), sent)
        segs, start = [], 0
        for m in _CLAUSE_SPLIT_RE.finditer(masked):
            segs.append((start, m.start()))
            start = m.end()
        segs.append((start, len(sent)))
        for a, b in segs:
            if a <= pos < max(b, a + 1):
                clause = sent[a:b]
                break
    text = _strip_markers(clause)
    if len(text.split()) < 6:                 # too little left — keep the sentence
        text = _strip_markers(sent)
    n_markers = len(_ANY_MARKER_RE.findall(sent))
    if _METHOD_RE.search(clause):
        intent = "method"
    elif _BACKGROUND_RE.search(clause) or n_markers >= 3:
        intent = "background"
    else:
        intent = "result"
    return CitedClaim(cited_claim=text, subclaims=[text], intent=intent, source="heuristic",
                      shared=_is_shared_marker(claim.citation_marker))


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You read citing sentences from a scientific paper.  For each item you get
the sentence, the citation marker of interest, and the title of the work
that marker cites.  Determine what the sentence claims about THAT work.

Return a JSON array, one object per item, same order:
  {"cited_claim": "...", "subclaims": ["...", ...], "intent": "...", "shared": false}

- cited_claim: the part of the sentence attributed to the marked work,
  as one standalone statement about that work ("The cited work shows
  that …").  Leave out only what the sentence attributes to OTHER works
  or to the citing authors themselves.  Keep numbers, datasets and names
  exactly, and keep every qualifier: hedges ("generally", "often",
  "some", "can", "may") and concessive or limiting clauses ("although …",
  "but …", "however …") belong to the statement they qualify.
  "LLMs generally underperform as encoders, although some modifications
  improve them" is ONE statement — never reduce it to "LLMs underperform
  as encoders".
  If the sentence enumerates several works (models, datasets, tools,
  methods) each with its own citation, the claim about this work is only
  the item next to its marker: "We evaluate Mistral-7B [3], GPT-4 [5] and
  Gemma [8]" → for [5]: "The cited work introduces GPT-4." (intent method).
- subclaims: 1-3 statements ABOUT THE CITED WORK that together make up
  cited_claim and can be checked against its text.  A qualified statement
  stays in one subclaim together with its qualifier; split only
  independent facts.  Never about what
  the citing authors do: for "We use the X dataset [4]" the subclaim is
  "The cited work introduces or describes the X dataset."  One subclaim
  if the claim is already atomic.
- shared: true when the marker cites several works together for this one
  statement ("Prior work (A; B) has highlighted X", "[3, 7]"); then each
  work only needs to be consistent with X.
- intent:
    "result"     the sentence reports a finding, number, property or
                 conclusion of the cited work;
    "method"     the citing authors use / follow / build on a method,
                 model, dataset, metric or tool from the cited work;
    "background" a general pointer to related work (examples, lists,
                 surveys) without a specific claim.
Use only the sentence.  The cited title only tells you which work the
marker refers to: never rewrite the claim to fit the title, and never add
facts about the work from memory.  If the sentence attributes something to
the work that its title doesn't suggest, keep what the sentence says —
that mismatch is exactly what is being checked.
Return ONLY the JSON array.
"""

_MAX_PER_CALL = 15
_CACHE_SOURCE = "claim_analyse"


def _key(claim: Claim, title: str, model: str) -> str:
    h = hashlib.sha256(_SYSTEM_PROMPT.encode())      # a prompt change invalidates
    for part in (model, claim.text, claim.citation_marker, title):
        h.update(part.encode("utf-8") + b"\0")
    return h.hexdigest()[:32]


def _coerce(item: Any, fallback: CitedClaim) -> CitedClaim:
    if not isinstance(item, dict):
        return fallback
    cc = str(item.get("cited_claim") or "").strip()
    subs = item.get("subclaims") or []
    if isinstance(subs, str):
        subs = [subs]
    subs = [str(s).strip() for s in subs if str(s).strip()][:3]
    intent = str(item.get("intent") or "").strip().lower()
    if not cc:
        return fallback
    shared = item.get("shared")
    shared = fallback.shared if shared is None else (
        shared if isinstance(shared, bool) else str(shared).strip().lower() in ("true", "1", "yes"))
    return CitedClaim(cited_claim=cc[:600], subclaims=[s[:400] for s in subs] or [cc[:400]],
                      intent=intent if intent in INTENTS else fallback.intent, source="llm",
                      shared=shared)


def analyse(claims: list[Claim], refs: list[dict[str, Any]],
            use_llm: bool = True) -> list[CitedClaim]:
    """One :class:`CitedClaim` per claim, same order."""
    out = [heuristic(c) for c in claims]
    if not claims or not use_llm:
        return out
    from claim_verifier import _call_llm, _parse_llm_response
    from llm_parser import (_DEFAULT_MODELS, _make_client, active_model,
                            fallback_backend, is_down, mark_down)
    backend, model = active_model()
    if is_down(backend) and (fb := fallback_backend(backend)):
        backend = fb
        model = os.getenv("LLM_FALLBACK_MODEL") or _DEFAULT_MODELS.get(fb, model)

    try:
        from cache import get_cache
        cache = get_cache()
    except Exception:
        cache = None

    def title_of(c: Claim) -> str:
        i = c.cited_ref_index
        return (refs[i].get("title") or "") if 0 <= i < len(refs) else ""

    todo: list[int] = []
    for i, c in enumerate(claims):
        hit = cache.get(_CACHE_SOURCE, _key(c, title_of(c), model)) if cache else None
        if hit:
            out[i] = CitedClaim(**hit)
        else:
            todo.append(i)
    if not todo:
        return out
    client = _make_client(backend)
    if client is None:
        return out

    for k in range(0, len(todo), _MAX_PER_CALL):
        chunk = todo[k:k + _MAX_PER_CALL]
        payload = json.dumps([{"sentence": claims[i].text,
                               "marker": claims[i].citation_marker,
                               "cited_title": title_of(claims[i])} for i in chunk],
                             ensure_ascii=False)
        parsed: Optional[list] = None
        for attempt in range(3):
            try:
                text = _call_llm(client, backend, model, _SYSTEM_PROMPT, payload,
                                 n_claims=len(chunk))
                parsed = _parse_llm_response(text)
                if parsed is not None:
                    break
            except Exception as exc:
                mark_down(backend, exc)
                # Same opt-in backup as LLM repair and the verdict step.
                fb = fallback_backend(backend)
                fb_client = _make_client(fb) if fb else None
                if fb_client is None:
                    break
                client, backend = fb_client, fb
                model = os.getenv("LLM_FALLBACK_MODEL") or _DEFAULT_MODELS.get(fb, model)
        if not parsed:
            continue                       # keep the heuristic results
        for i, item in zip(chunk, parsed):
            cc = _coerce(item, out[i])
            out[i] = cc
            if cache is not None and cc.source == "llm":
                cache.set(_CACHE_SOURCE, _key(claims[i], title_of(claims[i]), model), cc.to_dict())
    return out
