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
from dataclasses import asdict, dataclass
from typing import Any, Optional

from cache import get_cache
from claim_extractor import Claim
from fulltext     import FullText
from llm_parser   import _make_client, active_model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_VALID_VERDICTS    = {"supported", "contradicted", "not_addressed", "ambiguous"}
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


def verify_batch(
    claims:           list[Claim],
    ft:               FullText,
    claim_indices:    Optional[list[int]] = None,
) -> list[ClaimVerification]:
    """
    Verify one or more sentences (all citing the same paper) against
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
            out.extend(verify_batch(claims[k:k + _MAX_CLAIMS_PER_CALL], ft,
                                    claim_indices[k:k + _MAX_CLAIMS_PER_CALL]))
        return out

    backend, model = active_model()

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

    client = _make_client()
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
            if _is_auth_error(exc):
                break                                    # don't retry
            if attempt == 0:
                client = _make_client() or client        # fresh client
                continue
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
