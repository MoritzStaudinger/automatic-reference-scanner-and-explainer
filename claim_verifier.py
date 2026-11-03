"""
LLM-based verdict: does the cited paper support the claim?

Given a :class:`claim_extractor.Claim` and a :class:`fulltext.FullText`,
call the configured LLM and return a structured
:class:`ClaimVerification`.  Reuses the backend-agnostic client
construction from ``llm_parser`` so the same LLM_BACKEND / LLM_MODEL /
API-key env vars govern this step too.

Verdicts:
  supported       — the paper says the claim (evidence quote required)
  contradicted    — the paper disagrees with the claim
  not_addressed   — the paper doesn't discuss this
  ambiguous       — evidence is partial or context-dependent
  error           — LLM call failed; the caller decides how to display

Cached under source ``"claim_verify"`` in the shared SQLite cache; key
= sha256(claim || cited_paper_text[:10000]).  Same claim + same paper →
cache hit, so re-runs (batch reprocess, tweaking the UI) don't re-hit
the LLM.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from cache import get_cache
from claim_extractor import Claim
from fulltext     import FullText
from llm_parser   import (_DEFAULT_MODELS, _make_client, active_model,
                          fallback_backend, is_down, mark_down)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_VALID_VERDICTS    = {"supported", "contradicted", "not_addressed", "ambiguous"}
# Verdicts of the passage-based method on top of the above:
#   partially_supported — some sub-claims supported, none contradicted
#   evidence_only       — no LLM available: best-matching passages, no verdict
_VALID_CONFIDENCES = {"high", "medium", "low"}


@dataclass
class ClaimVerification:
    claim_index:      int
    cited_ref_index:  int
    claim_text:       str
    verdict:          str          # supported | contradicted | not_addressed | ambiguous | error
    evidence_passage: str          # exact quote from cited paper, or ""
    confidence:       str          # high | medium | low
    explanation:      str          # one sentence
    source:           str          # fulltext source tag (arxiv, s2_oa, ...)
    is_abstract_fallback: bool     # true when we only had the abstract
    model:            str          # LLM model used
    error:            Optional[str] = None    # populated when verdict == "error"
    # Passage-based method (ARSE_CLAIM_METHOD=passages, the default):
    cited_claim:      str = ""                # what the sentence attributes to this ref
    intent:           str = ""                # result | method | background
    subclaims:        list = field(default_factory=list)   # [{text, verdict, evidence_ids, explanation}]
    evidence:         list = field(default_factory=list)   # [{id, text, section, score}]
    mode:             str = "fulltext"        # fulltext | passages | evidence_only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert scientific reader.  Below is the full text of a CITED
PAPER, followed by one or more SENTENCES from a different paper that
all cite THIS cited paper.  For each sentence, judge whether the cited
paper actually supports what the sentence claims about it.

For each sentence, pick one verdict:

  supported     — the paper clearly says (or straightforwardly implies)
                  what the sentence claims.
  contradicted  — the paper explicitly disagrees with the sentence's
                  claim.
  not_addressed — the paper doesn't discuss the topic at all, or only
                  in passing without the substance the claim relies on.
  ambiguous     — partial support, or support only under conditions the
                  sentence doesn't mention.  Use this when a strict
                  yes/no would misrepresent the paper.

Rules:
- Base your verdicts ONLY on the cited paper text you're given.
- If the text is ABSTRACT-ONLY (marked as such), be strict — an abstract
  doesn't cover most cited claims; default to `not_addressed` unless the
  abstract itself makes the claim.
- Do not use outside knowledge.  If the paper doesn't say it, it's not
  supported, even if you know it's true.
- Cite ONE short exact quote (≤ 30 words) from the cited paper per
  supported / contradicted / ambiguous verdict.  For `not_addressed`,
  leave the evidence empty.

Return ONLY a JSON array of length N (one entry per sentence, in the
same order they appear below), with no prose before or after:

  [
    {"verdict": "...", "evidence_passage": "...", "confidence": "...", "explanation": "..."},
    {"verdict": "...", "evidence_passage": "...", "confidence": "...", "explanation": "..."},
    ...
  ]
"""


def _build_user_message(claims: list[Claim], ft: FullText) -> str:
    """
    Build a batched user message: one bundle per cited ref, listing every
    citing-paper sentence that references it.  The LLM returns one
    verdict per sentence in one shot — one call per ref instead of one
    call per (sentence, ref) pair.
    """
    fallback_marker = "  (ABSTRACT-ONLY)" if ft.is_abstract_fallback else ""
    numbered_sentences = "\n".join(
        f'  {i + 1}. "{c.text}"'
        for i, c in enumerate(claims)
    )
    return (
        f"CITED PAPER TEXT{fallback_marker}:\n"
        f"{ft.text}\n\n"
        f"SENTENCES FROM CITING PAPER (all cite the above):\n"
        f"{numbered_sentences}\n"
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_SOURCE = "claim_verify"


def _cache_key(claim: Claim, ft: FullText, model: str = "") -> str:
    """Per-sentence key so a re-run over a different bundle still hits
    cached verdicts on the sentences we already scored.  Includes the
    model so switching backends doesn't serve another model's verdicts."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(claim.text.encode("utf-8"))
    h.update(b"\0")
    h.update(ft.text[:10_000].encode("utf-8"))
    return h.hexdigest()[:32]


def _cached_get(key: str) -> Optional[dict]:
    return get_cache().get(_CACHE_SOURCE, key)


def _cached_set(key: str, verdict: dict) -> None:
    # Don't cache errors — they're usually transient (rate limit, timeout).
    if verdict.get("verdict") == "error":
        return
    get_cache().set(_CACHE_SOURCE, key, verdict)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

# Batched responses come back as a JSON array `[{...}, {...}, ...]`.
# The model may wrap it in ```json fences or add prose around it.
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]", re.MULTILINE)


def _parse_llm_response(text: str) -> Optional[list[dict]]:
    """
    Parse the model's batched response.  Returns a list of verdict
    dicts, or None if nothing parses.  Tolerant of leading/trailing
    prose and ```json fences.
    """
    if not text:
        return None
    stripped = text.strip()
    # Strip common code-fence wrappers before attempting parse.
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    # Best case: whole string is the JSON array.
    try:
        v = json.loads(stripped)
        return v if isinstance(v, list) else None
    except json.JSONDecodeError:
        pass
    # Fallback: greedy match for the outermost [...] block.
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    # Last-ditch: bracket-balanced extraction.
    start = text.find("[")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[start : i + 1])
                        return v if isinstance(v, list) else None
                    except json.JSONDecodeError:
                        return None
    return None


def _sanitize_verdict(raw: dict) -> dict:
    """Coerce free-form model output onto our schema, clamping enum values."""
    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        verdict = "ambiguous"
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in _VALID_CONFIDENCES:
        conf = "low"
    evidence = str(raw.get("evidence_passage", "")).strip()
    if len(evidence) > 400:
        evidence = evidence[:400].rsplit(" ", 1)[0] + "…"
    explanation = str(raw.get("explanation", "")).strip()
    if len(explanation) > 400:
        explanation = explanation[:400].rsplit(" ", 1)[0] + "…"
    return {
        "verdict":          verdict,
        "evidence_passage": evidence,
        "confidence":       conf,
        "explanation":      explanation,
    }


# ---------------------------------------------------------------------------
# LLM dispatch — reuse llm_parser's backend abstraction
# ---------------------------------------------------------------------------

# Per-batched-call timeout.  Aqueduct/Qwen on shared GPU can hang for
# tens of minutes under cluster contention — a hard timeout keeps one
# stuck ref from freezing the whole pipeline.  Env-configurable so the
# operator can tune per backend (Sonnet completes in <15s; a 60s ceiling
# on Aqueduct still lets healthy calls finish while cutting off zombies).
_LLM_TIMEOUT_S = float(os.getenv("LLM_VERIFY_TIMEOUT_S", "90"))


# Claims per LLM call.  Each verdict is ~150-250 output tokens; larger
# batches risked truncated JSON and an error verdict for every claim.
_MAX_CLAIMS_PER_CALL = 6


def _max_tokens_for(n_claims: int) -> int:
    return min(400 + 300 * n_claims, 8000)


def _is_auth_error(exc: Exception) -> bool:
    """401/403 from either SDK — retrying won't help."""
    return type(exc).__name__ in ("PermissionDeniedError", "AuthenticationError",
                                  "PermissionError")


def _call_llm(client, backend: str, model: str, system: str, user: str,
              timeout_s: Optional[float] = None, n_claims: int = 1) -> str:
    """
    Dispatch a batched-verify call, honouring a per-call timeout.  Both
    backends surface timeout as a native exception, which the retry
    logic in ``verify_batch`` catches and converts to an ``error``
    verdict rather than propagating.
    """
    t = timeout_s if timeout_s is not None else _LLM_TIMEOUT_S
    if backend == "aqueduct":
        # openai SDK exposes per-request timeouts via with_options —
        # more portable than passing timeout= on the create() call
        # (which some vLLM proxies choke on).
        scoped = client.with_options(timeout=t)
        kwargs: dict[str, Any] = {
            "model":      model,
            "max_tokens": _max_tokens_for(n_claims),
            "messages":   [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        if "qwen" in model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = scoped.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    # Default = anthropic.  `timeout` is a per-request kwarg supported by
    # the SDK directly.  Haiku 5.x thinks by default; a verdict per
    # sentence doesn't need it (Opus/Sonnet 5.5 reject "disabled").
    extra: dict[str, Any] = {}
    if model.startswith("claude-haiku-5"):
        extra["thinking"] = {"type": "disabled"}
    msg = client.messages.create(
        model=model,
        max_tokens=_max_tokens_for(n_claims),
        system=system,
        messages=[{"role": "user", "content": user}],
        timeout=t,
        **extra,
    )
    if msg.stop_reason == "refusal":
        raise RuntimeError("model declined the request (refusal)")
    return next((b.text for b in msg.content if b.type == "text"), "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _mk_error(claim: Claim, ft: FullText, model: str, err: str,
              claim_index: int) -> ClaimVerification:
    return ClaimVerification(
        claim_index=claim_index,
        cited_ref_index=claim.cited_ref_index,
        claim_text=claim.text,
        verdict="error",
        evidence_passage="",
        confidence="low",
        explanation="",
        source=ft.source,
        is_abstract_fallback=ft.is_abstract_fallback,
        model=model,
        error=err[:200],
    )


def _mk_from_cache(claim: Claim, ft: FullText, cached: dict,
                   model: str, claim_index: int) -> ClaimVerification:
    return ClaimVerification(
        claim_index=claim_index,
        cited_ref_index=claim.cited_ref_index,
        claim_text=claim.text,
        verdict=cached["verdict"],
        evidence_passage=cached["evidence_passage"],
        confidence=cached["confidence"],
        explanation=cached["explanation"],
        source=ft.source,
        is_abstract_fallback=ft.is_abstract_fallback,
        model=cached.get("model") or model,
    )


def _verify_batch_fulltext(
    claims:           list[Claim],
    ft:               FullText,
    claim_indices:    Optional[list[int]] = None,
) -> list[ClaimVerification]:
    """
    The original method (ARSE_CLAIM_METHOD=fulltext): verify one or more sentences (all citing the same paper) against
    that paper's full text — with ONE LLM call, not one per sentence.

    ``claims`` must all share the same ``cited_ref_index`` (asserted).
    ``claim_indices``, when provided, is a parallel list of pipeline-
    level claim indices used to populate ``ClaimVerification.claim_index``;
    defaults to positional (0, 1, 2, ...).

    Returns a list of :class:`ClaimVerification`, one per input claim,
    in the same order.  Per-sentence caching means re-running with a
    partial-overlap set of claims doesn't re-verify sentences we already
    scored.
    """
    if not claims:
        return []
    if claim_indices is None:
        claim_indices = list(range(len(claims)))
    assert len(claim_indices) == len(claims)
    if len(claims) > _MAX_CLAIMS_PER_CALL:
        out: list[ClaimVerification] = []
        for k in range(0, len(claims), _MAX_CLAIMS_PER_CALL):
            out.extend(_verify_batch_fulltext(claims[k:k + _MAX_CLAIMS_PER_CALL], ft,
                                              claim_indices[k:k + _MAX_CLAIMS_PER_CALL]))
        return out

    backend, model = active_model()
    # Primary known to be down and a fallback configured: use the fallback
    # directly rather than waiting out timeouts on every batch.
    if is_down(backend) and (fb := fallback_backend(backend)):
        backend = fb
        model = os.getenv("LLM_FALLBACK_MODEL") or _DEFAULT_MODELS.get(fb, model)

    # Cache lookup per sentence.  Any hits get pulled out; misses go to
    # the LLM in a single batched call.
    verdicts: list[Optional[ClaimVerification]] = [None] * len(claims)
    miss_indices: list[int] = []
    for i, c in enumerate(claims):
        key = _cache_key(c, ft, model)
        hit = _cached_get(key)
        if hit is not None:
            verdicts[i] = _mk_from_cache(c, ft, hit, model, claim_indices[i])
        else:
            miss_indices.append(i)

    if not miss_indices:
        return [v for v in verdicts if v is not None]

    client = _make_client(backend)
    if client is None:
        err = (f"LLM backend '{backend}' is not configured — set "
               f"ANTHROPIC_API_KEY (anthropic) or AQUEDUCT_API_KEY "
               f"(aqueduct).")
        for i in miss_indices:
            verdicts[i] = _mk_error(claims[i], ft, model, err, claim_indices[i])
        return [v for v in verdicts if v is not None]

    miss_claims = [claims[i] for i in miss_indices]
    user_msg = _build_user_message(miss_claims, ft)

    # Retry-once: if the first call times out or the response doesn't
    # parse, try one more time with a fresh client (helps with cluster
    # hiccups on Aqueduct + transient rate-limits on Anthropic).  We do
    # NOT retry on hard config errors (missing API key, 403) — those
    # won't fix themselves.
    raw_text: Optional[str] = None
    last_err: Optional[str]  = None
    for attempt in range(2):
        try:
            raw_text = _call_llm(client, backend, model, _SYSTEM_PROMPT, user_msg,
                                 n_claims=len(miss_claims))
            break
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            mark_down(backend, exc)
            if _is_auth_error(exc):
                break                                    # don't retry
            if attempt == 0:
                client = _make_client(backend) or client  # fresh client
                continue
    if raw_text is None:
        # Opt-in backup backend (same LLM_FALLBACK_BACKEND as LLM repair),
        # e.g. Anthropic while the Aqueduct endpoint is down.
        fb = fallback_backend(backend)
        fb_client = _make_client(fb) if fb else None
        if fb_client is not None:
            fb_model = os.getenv("LLM_FALLBACK_MODEL") or _DEFAULT_MODELS.get(fb, model)
            try:
                raw_text = _call_llm(fb_client, fb, fb_model, _SYSTEM_PROMPT, user_msg,
                                     n_claims=len(miss_claims))
                client, backend, model = fb_client, fb, fb_model
            except Exception as exc:
                last_err = f"{last_err}; fallback {fb}: {type(exc).__name__}: {exc}"
    if raw_text is None:
        for i in miss_indices:
            verdicts[i] = _mk_error(claims[i], ft, model,
                                    last_err or "LLM call failed",
                                    claim_indices[i])
        return [v for v in verdicts if v is not None]

    parsed = _parse_llm_response(raw_text)
    if not parsed or len(parsed) < len(miss_claims):
        # One more shot — sometimes the model produces valid JSON on
        # the second attempt (temperature quirks, mid-stream truncation).
        try:
            raw_text = _call_llm(client, backend, model,
                                 _SYSTEM_PROMPT, user_msg, n_claims=len(miss_claims))
            parsed = _parse_llm_response(raw_text)
        except Exception:
            parsed = None
    if not parsed or len(parsed) < len(miss_claims):
        err = (f"could not parse LLM response as a JSON array of "
               f"length {len(miss_claims)}; got {raw_text[:120]!r}")
        for i in miss_indices:
            verdicts[i] = _mk_error(claims[i], ft, model, err,
                                    claim_indices[i])
        return [v for v in verdicts if v is not None]

    # Zip the parsed array back onto the miss positions.  Extra entries
    # beyond len(miss_claims) are silently dropped.
    for slot, i in enumerate(miss_indices):
        raw = parsed[slot] if isinstance(parsed[slot], dict) else {}
        sanitized = _sanitize_verdict(raw)
        sanitized["model"] = model
        _cached_set(_cache_key(claims[i], ft, model), sanitized)
        verdicts[i] = ClaimVerification(
            claim_index=claim_indices[i],
            cited_ref_index=claims[i].cited_ref_index,
            claim_text=claims[i].text,
            verdict=sanitized["verdict"],
            evidence_passage=sanitized["evidence_passage"],
            confidence=sanitized["confidence"],
            explanation=sanitized["explanation"],
            source=ft.source,
            is_abstract_fallback=ft.is_abstract_fallback,
            model=model,
        )
    return [v for v in verdicts if v is not None]


# Convenience wrapper — the old single-claim API.  Kept for smoke tests
# and any external caller that still uses it.
def verify(claim: Claim, ft: FullText, claim_index: int = 0) -> ClaimVerification:
    return verify_batch([claim], ft, [claim_index])[0]


# ===========================================================================
# Passage-based verification (default)
# ===========================================================================
# claim_analysis isolates what the sentence attributes to the reference and
# splits it into sub-claims; evidence.retrieve picks the passages of the
# cited paper that best match each sub-claim; the LLM judges each sub-claim
# on those numbered passages and must name the ones it relies on.  Evidence
# shown to the user is therefore always real text of the cited paper.

_K_PER_SUBCLAIM = 6
_MAX_PASSAGES_PER_CALL = 14

_PASSAGE_PROMPT = """\
You check whether a cited paper supports what another paper says about it.

You get numbered PASSAGES from the cited paper (with their section) and
one or more CLAIMS about it, plus the citation INTENT:
  result     — the claim reports a finding of the cited paper: judge it
               strictly; numbers, datasets and conditions must match.
  method     — the citing authors use something from the cited paper:
               supported if the passages show the paper introduces or
               describes that method / model / dataset / tool.
  background — a pointer to related work: supported if the paper is
               clearly about the topic the claim names, even when the
               passages don't state the claim's details (the citation
               points the reader to relevant work, it doesn't attribute a
               finding); not_addressed only if the paper is about
               something else.

For each claim pick:
  supported     — the passages state it, or it is a faithful paraphrase or
                  summary of what they state (citing authors condense and
                  reword; that is fine as long as the meaning is kept);
  contradicted  — the passages say something incompatible with it;
  ambiguous     — the passages support it only in a weaker form: a
                  stronger effect, broader scope, missing condition or
                  different population than the claim states;
  not_addressed — the passages don't contain the information.
Don't demand the claim's exact wording; do check numbers, direction of
effects, populations (e.g. humans vs mice) and causal vs correlational
language.

Passages marked [cites other work] attribute statements to OTHER papers
(they contain citations).  For a result claim they are not the cited
paper's own finding: on their own they support at most "ambiguous".

Judge ONLY from the passages, never from what you know about the paper.
List the passage ids (e.g. "P3") your verdict relies on; none for
not_addressed.

Return ONLY JSON:
{"claims": [{"verdict": "...", "passage_ids": ["P1", ...], "explanation": "one sentence"}, ...],
 "confidence": "high|medium|low"}
"""


# A passage that cites other work: "[1, 70]", "(Smith et al., 2020)".
_CITES_OTHERS_RE = re.compile(r"\[\s*\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*\s*\]"
                              r"|\([A-Z][^()]{0,60}?(?:19|20)\d{2}[a-z]?\)")


def _passage_block(passages) -> str:
    return "\n".join(
        f"[P{n}] ({sp.passage.section or 'no section'})"
        f"{' [cites other work]' if _CITES_OTHERS_RE.search(sp.passage.text) else ''} "
        f"{sp.passage.text}"
        for n, sp in enumerate(passages, 1))


def _parse_passage_response(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        try:
            v = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(v, dict) and isinstance(v.get("claims"), list):
            return v
        if isinstance(v, list):
            return {"claims": v}
    return None


def aggregate(verdicts: list[str]) -> str:
    """Overall verdict from sub-claim verdicts."""
    vs = [v for v in verdicts if v]
    if not vs:
        return "not_addressed"
    if "contradicted" in vs:
        return "contradicted"
    if all(v == "supported" for v in vs):
        return "supported"
    if "supported" in vs:
        return "partially_supported"
    if all(v == "not_addressed" for v in vs):
        return "not_addressed"
    return "ambiguous"


def _gather(subclaims: list[str], passages, embedder, skip: int):
    """Union of the top passages per sub-claim, best first, capped."""
    import evidence
    seen, out = set(), []
    per = [evidence.retrieve(sc, passages, k=_K_PER_SUBCLAIM, embedder=embedder, skip=skip)
           for sc in subclaims]
    for rank in range(_K_PER_SUBCLAIM):          # interleave so each sub-claim gets some
        for hits in per:
            if rank < len(hits) and hits[rank].passage.idx not in seen:
                seen.add(hits[rank].passage.idx)
                out.append(hits[rank])
    return out[:_MAX_PASSAGES_PER_CALL]


def _passage_key(model: str, cc, shown) -> str:
    h = hashlib.sha256(_PASSAGE_PROMPT.encode())     # a prompt change invalidates
    h.update(f"{model}\0{cc.intent}\0".encode())
    for sc in cc.subclaims:
        h.update(sc.encode("utf-8") + b"\0")
    for sp in shown:
        h.update(sp.passage.text.encode("utf-8") + b"\0")
    return "p:" + h.hexdigest()[:30]


def _ask(client, backend, model, cc, shown) -> Optional[dict]:
    user = (f"INTENT: {cc.intent}\n\nPASSAGES:\n{_passage_block(shown)}\n\nCLAIMS:\n" +
            "\n".join(f"{i}. {sc}" for i, sc in enumerate(cc.subclaims, 1)))
    key = _passage_key(model, cc, shown)
    hit = _cached_get(key)
    if hit is not None:
        return hit
    for attempt in range(2):
        try:
            text = _call_llm(client, backend, model, _PASSAGE_PROMPT, user,
                             n_claims=len(cc.subclaims))
        except Exception as exc:
            mark_down(backend, exc)
            if _is_auth_error(exc) or attempt:
                raise
            continue
        parsed = _parse_passage_response(text)
        if parsed is not None:
            _cached_set(key, parsed)
            return parsed
    return None


def _sub_verdict(parsed: dict, i: int) -> str:
    items = parsed.get("claims") or []
    return (str(items[i].get("verdict", "")).strip().lower()
            if i < len(items) and isinstance(items[i], dict) else "")


def _merge_retry(parsed: dict, shown: list, p2: dict, more: list, missing: list[int]):
    """Fold a second-round answer for the ``missing`` sub-claims into the
    first one.  Passage ids of the second round are renumbered to follow
    the first round's, so every id still maps to exactly one passage."""
    offset = len(shown)
    shown = shown + more
    claims = [dict(c) if isinstance(c, dict) else {} for c in (parsed.get("claims") or [])]
    while len(claims) < max(missing) + 1:
        claims.append({})
    for j, i in enumerate(missing):
        item = (p2.get("claims") or [{}])[j] if j < len(p2.get("claims") or []) else {}
        if not isinstance(item, dict) or str(item.get("verdict", "")).lower() in ("", "not_addressed"):
            continue
        ids = []
        for x in item.get("passage_ids") or []:
            m = re.match(r"P?(\d+)$", str(x).strip(), re.I)
            if m:
                ids.append(f"P{int(m.group(1)) + offset}")
        claims[i] = {**item, "passage_ids": ids}
    return {**parsed, "claims": claims}, shown


def _verify_one_passages(claim: Claim, cc, ft: FullText, passages, embedder,
                         client, backend: str, model: str, claim_index: int) -> ClaimVerification:
    base = dict(claim_index=claim_index, cited_ref_index=claim.cited_ref_index,
                claim_text=claim.text, source=ft.source,
                is_abstract_fallback=ft.is_abstract_fallback,
                cited_claim=cc.cited_claim, intent=cc.intent)
    subclaims = cc.subclaims or [cc.cited_claim]
    if cc.intent == "background":
        # A pointer to related work only needs the cited paper to be on the
        # topic; judging the sentence's details would demand more than the
        # citation claims.
        subclaims = [f"The cited work addresses this topic: {cc.cited_claim}"]
        cc = type(cc)(cc.cited_claim, subclaims, cc.intent, cc.source)
    shown = _gather(subclaims, passages, embedder, skip=0)

    def ev(sps):
        return [{"id": f"P{n}", "text": sp.passage.text, "section": sp.passage.section,
                 "score": round(sp.score, 4)} for n, sp in enumerate(sps, 1)]

    if not shown:
        return ClaimVerification(**base, verdict="not_addressed", evidence_passage="",
                                 confidence="low", explanation="no passage of the cited paper "
                                 "matches the claim", model=model if client else "",
                                 subclaims=[{"text": t, "verdict": "not_addressed",
                                             "evidence_ids": [], "explanation": ""} for t in subclaims],
                                 mode="passages" if client else "evidence_only")
    if client is None:
        top = shown[:3]
        return ClaimVerification(**base, verdict="evidence_only",
                                 evidence_passage=top[0].passage.text[:400], confidence="low",
                                 explanation="No LLM configured: these are the passages of the "
                                             "cited paper that best match the claim.",
                                 model="", evidence=ev(top), mode="evidence_only",
                                 subclaims=[{"text": t, "verdict": "", "evidence_ids": [],
                                             "explanation": ""} for t in subclaims])
    try:
        parsed = _ask(client, backend, model, cc, shown)
        # Sub-claims not found in the first passages of a full text: search
        # further once for those, before concluding "not addressed".
        if parsed and not ft.is_abstract_fallback:
            missing = [i for i in range(len(subclaims))
                       if _sub_verdict(parsed, i) in ("not_addressed", "")]
            if missing:
                sub_cc = type(cc)(cc.cited_claim, [subclaims[i] for i in missing],
                                  cc.intent, cc.source)
                more = _gather(sub_cc.subclaims, passages, embedder, skip=_K_PER_SUBCLAIM)
                p2 = _ask(client, backend, model, sub_cc, more) if more else None
                if p2:
                    parsed, shown = _merge_retry(parsed, shown, p2, more, missing)
    except Exception as exc:
        return ClaimVerification(**base, verdict="error", evidence_passage="", confidence="low",
                                 explanation="", model=model, error=f"{type(exc).__name__}: {exc}"[:200],
                                 mode="passages")
    if parsed is None:
        return ClaimVerification(**base, verdict="error", evidence_passage="", confidence="low",
                                 explanation="", model=model, mode="passages",
                                 error="could not parse the LLM response")

    by_id = {f"P{n}": sp for n, sp in enumerate(shown, 1)}
    subs, used = [], []
    for i, text in enumerate(subclaims):
        item = parsed["claims"][i] if i < len(parsed["claims"]) and isinstance(parsed["claims"][i], dict) else {}
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in _VALID_VERDICTS:
            verdict = "ambiguous"
        ids = [str(x).strip().upper() for x in (item.get("passage_ids") or [])]
        ids = [x if x.startswith("P") else f"P{x}" for x in ids]
        ids = [x for x in dict.fromkeys(ids) if x in by_id]        # invented ids are dropped
        if verdict in ("supported", "contradicted") and not ids:
            verdict = "ambiguous"                                 # a verdict needs evidence
        used.extend(x for x in ids if x not in used)
        subs.append({"text": text, "verdict": verdict, "evidence_ids": ids,
                     "explanation": str(item.get("explanation", "")).strip()[:300]})
    overall = aggregate([s["verdict"] for s in subs])
    evidence = [{"id": x, "text": by_id[x].passage.text, "section": by_id[x].passage.section,
                 "score": round(by_id[x].score, 4)} for x in used]
    conf = str(parsed.get("confidence", "")).strip().lower()
    expl = "; ".join(s["explanation"] for s in subs if s["explanation"])[:400]
    return ClaimVerification(**base, verdict=overall,
                             evidence_passage=(evidence[0]["text"][:400] if evidence else ""),
                             confidence=conf if conf in _VALID_CONFIDENCES else "low",
                             explanation=expl, model=model, subclaims=subs, evidence=evidence,
                             mode="passages")


def claim_method() -> str:
    return (os.getenv("ARSE_CLAIM_METHOD") or "passages").strip().lower()


def verify_batch(
    claims:        list[Claim],
    ft:            FullText,
    claim_indices: Optional[list[int]] = None,
    cited:         Optional[list] = None,
    use_llm:       bool = True,
) -> list[ClaimVerification]:
    """Verify sentences that all cite the paper ``ft``.

    ``cited`` — one :class:`claim_analysis.CitedClaim` per claim (from
    ``claim_analysis.analyse``); computed heuristically when omitted.
    ``ARSE_CLAIM_METHOD=fulltext`` selects the original whole-paper method."""
    if not claims:
        return []
    if claim_indices is None:
        claim_indices = list(range(len(claims)))
    if claim_method() == "fulltext":
        return _verify_batch_fulltext(claims, ft, claim_indices)

    import evidence
    from claim_analysis import heuristic
    cited = cited or [heuristic(c) for c in claims]
    passages = evidence.segment(ft.text, is_abstract=ft.is_abstract_fallback)
    embedder = evidence.get_embedder()

    backend, model = active_model()
    if is_down(backend) and (fb := fallback_backend(backend)):
        backend = fb
        model = os.getenv("LLM_FALLBACK_MODEL") or _DEFAULT_MODELS.get(fb, model)
    client = _make_client(backend) if use_llm else None

    out = []
    for c, cc, ci in zip(claims, cited, claim_indices):
        v = _verify_one_passages(c, cc, ft, passages, embedder, client, backend, model, ci)
        if v.verdict == "error" and use_llm:
            fb = fallback_backend(backend)
            fb_client = _make_client(fb) if fb else None
            if fb_client is not None:
                fb_model = os.getenv("LLM_FALLBACK_MODEL") or _DEFAULT_MODELS.get(fb, model)
                v = _verify_one_passages(c, cc, ft, passages, embedder, fb_client, fb, fb_model, ci)
        out.append(v)
    return out
