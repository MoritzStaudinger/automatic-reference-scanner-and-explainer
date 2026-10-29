"""
ARES HTTP API — drive the pipeline remotely and process PDFs in batch.

Endpoints
---------
GET  /healthz
    Liveness check.  Returns ``{"status": "ok"}``.  No auth.

POST /api/v1/process
    Sync single-PDF processing.  Multipart upload, returns JSON report.
    Fields: file (PDF), parser (str), use_llm (bool).
    Auth: X-API-Key header (when ARES_API_KEY is set).
    Use this when you have one PDF and want the answer in the response.

POST /api/v1/batch
    Async batch processing.  Multipart with multiple files.  Returns
    a job_id immediately; the actual processing runs on a background
    thread pool sized by ARES_API_WORKERS.
    Auth: X-API-Key header (when ARES_API_KEY is set).
    Use this for 10s–1000s of PDFs.

GET  /api/v1/batch/{job_id}
    Status snapshot for a batch — counts only, no per-PDF payloads.
    Useful for polling progress.

GET  /api/v1/batch/{job_id}/results
    Full per-PDF results for a batch.  Returned as the batch's status
    block plus a ``results`` array of {filename, status, payload}.

GET  /api/v1/batch/{job_id}/results.jsonl
    Same data as above, streamed as JSONL — one line per PDF.  Easier
    to consume from a shell pipeline (jq, awk) than a giant JSON blob.

GET  /api/v1/batches
    Recently submitted batches (default limit 50).

Auth
----
Set ``ARES_API_KEY=<secret>`` in the environment and pass it via the
``X-API-Key`` header on every request.  When the env var is unset the
API runs in dev mode (no auth required) — convenient for local use,
do not deploy that way.

Running locally
---------------
::

    uvicorn api:app --host 0.0.0.0 --port 8500 --reload

Inside the docker-compose deployment the ares-api service runs uvicorn
on port 8500 and nginx routes ``/api/*`` to it.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse

from job_store import get_job_store
from pipeline import PipelineOptions, process_pdf

_log = logging.getLogger("ares.api")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_API_KEY        = os.getenv("ARES_API_KEY", "").strip()
_KEYS_FILE      = Path(os.getenv(
    "ARES_API_KEYS_FILE",
    Path(__file__).resolve().parent / "data" / "api_keys.json",
))
_BATCH_WORKERS  = max(1, int(os.getenv("ARES_API_WORKERS", "2")))
_MAX_BATCH      = int(os.getenv("ARES_API_MAX_BATCH", "500"))
_VALID_PARSERS  = {"grobid", "hybrid", "pymupdf4llm"}


# In-memory cache of {key_value: metadata_dict} loaded from _KEYS_FILE.
# Reloaded whenever the file's mtime changes (cheap, file is small).
_KEYS_CACHE: dict[str, dict] = {}
_KEYS_MTIME: float = 0.0
_KEYS_LOCK = threading.Lock()


def _load_api_keys() -> dict[str, dict]:
    """
    Return the {key: metadata} mapping, lazily reloading from disk when
    the JSON file's mtime changes.  This lets a deployment operator add
    or revoke keys by editing the file without restarting the API.

    File schema (data/api_keys.json):

        {
          "<random-secret-key>": {
            "label":       "alice",
            "created_at":  "2026-06-19T14:23:00",
            "expires_at":  null         // or ISO 8601 timestamp
          },
          ...
        }

    Missing file → empty dict (dev mode falls back to ARES_API_KEY env).
    """
    global _KEYS_CACHE, _KEYS_MTIME
    try:
        mtime = _KEYS_FILE.stat().st_mtime
    except OSError:
        return _KEYS_CACHE
    if mtime == _KEYS_MTIME:
        return _KEYS_CACHE
    with _KEYS_LOCK:
        if mtime == _KEYS_MTIME:
            return _KEYS_CACHE
        try:
            data = json.loads(_KEYS_FILE.read_text() or "{}")
            if not isinstance(data, dict):
                raise ValueError(f"{_KEYS_FILE} root must be an object")
            _KEYS_CACHE = data
            _KEYS_MTIME = mtime
            _log.info("loaded %d API key(s) from %s", len(data), _KEYS_FILE)
        except Exception as exc:
            _log.error("failed to load API keys from %s: %s",
                       _KEYS_FILE, exc)
    return _KEYS_CACHE


# ThreadPool for background batch work.  One pool shared across all
# batches keeps total concurrency bounded — GROBID is the real
# bottleneck and exceeds the VM's CPU count quickly if we spawn a
# thread per PDF.
_workers = ThreadPoolExecutor(
    max_workers=_BATCH_WORKERS,
    thread_name_prefix="ares-batch",
)


# ---------------------------------------------------------------------------
# App + auth
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Startup: sweep any non-terminal jobs from a previous process.

    A batch's ThreadPoolExecutor lives in-memory, so if the API container
    restarts mid-batch the workers vanish but the SQLite rows stay
    'queued' / 'running' forever.  On every cold start we mark them
    'failed' with a clear reason so callers see the truth instead of
    polling a zombie indefinitely.
    """
    n = get_job_store().sweep_stale()
    if n:
        _log.warning(
            "swept %d stale job(s) from previous process to 'failed'", n)
    yield


app = FastAPI(
    title="ARES API",
    version="1.0",
    description=(
        "Automatic Reference Explainer and Scanner — programmatic interface "
        "for headless PDF reference validation and batch processing."
    ),
    lifespan=lifespan,
    # Serve the auto-generated docs under /api/* so nginx (which routes
    # /api/* to this container without stripping the prefix) lands users
    # on the right URL.  Without these overrides /docs would 404 from
    # the outside since nginx wouldn't forward it.
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """
    Auth dependency.

    Accepts a key if EITHER of these matches:
      1. ``ARES_API_KEY`` env var (single shared key — legacy path)
      2. A key entry in ``data/api_keys.json`` (multi-user with labels +
         optional expiry)

    When neither the env var nor the file has any keys, the API runs
    unauthenticated (dev mode).  This makes a fresh checkout usable
    without setting up a keys file first.
    """
    keys = _load_api_keys()
    if not _API_KEY and not keys:
        return
    if not x_api_key:
        raise HTTPException(status_code=401,
                            detail="invalid or missing X-API-Key")

    # Single-shared-key path (legacy).
    if _API_KEY and x_api_key == _API_KEY:
        return

    # File-backed path with metadata + optional expiry.
    meta = keys.get(x_api_key)
    if meta is None:
        raise HTTPException(status_code=401,
                            detail="invalid or missing X-API-Key")
    expires_at = (meta or {}).get("expires_at")
    if expires_at:
        try:
            exp = datetime.datetime.fromisoformat(str(expires_at))
            if exp < datetime.datetime.now():
                raise HTTPException(status_code=401, detail="API key expired")
        except ValueError:
            # Malformed expiry → ignore (don't lock the caller out over
            # an operator typo).  The error log will catch it on reload.
            pass


def _validate_parser(parser: str) -> str:
    if parser not in _VALID_PARSERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown parser {parser!r}; expected one of {sorted(_VALID_PARSERS)}",
        )
    return parser


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Sync single-PDF
# ---------------------------------------------------------------------------

_VALID_VERIFY_DEPTH = {"abstract", "fulltext"}


@app.post("/api/v1/process", dependencies=[Depends(require_api_key)])
async def process_one(
    file:          UploadFile = File(..., description="The PDF to process."),
    parser:        str        = Form("grobid"),
    use_llm:       bool       = Form(False),
    verify_claims: bool       = Form(False),
    verify_depth:  str        = Form("abstract"),
) -> dict:
    """Process one PDF synchronously.  Blocks until the report is ready.

    Set ``verify_claims=1`` to also run per-citation claim verification.
    ``verify_depth`` chooses the tradeoff:
      - ``abstract`` (default, fast, ~30-60s/paper): verify against the
        cited paper's abstract only.  Skips OA PDF download + parse.
      - ``fulltext`` (slow, high-recall, minutes-to-hours): fetch each
        cited paper's OA PDF and verify against the whole paper.
    """
    _validate_parser(parser)
    if verify_depth not in _VALID_VERIFY_DEPTH:
        raise HTTPException(
            status_code=400,
            detail=f"unknown verify_depth {verify_depth!r}; "
                   f"expected one of {sorted(_VALID_VERIFY_DEPTH)}",
        )
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty upload")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="upload does not look like a PDF (missing %PDF header)",
        )

    try:
        report = process_pdf(
            pdf_bytes,
            PipelineOptions(parser=parser, use_llm=use_llm,
                            verify_claims=verify_claims,
                            verify_depth=verify_depth),
        )
    except Exception as exc:    # parser/LLM/lookup failures
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}")

    return {"filename": file.filename, **report.to_dict()}


# ---------------------------------------------------------------------------
# Async batch
# ---------------------------------------------------------------------------

def _run_batch_job(
    job_id: str,
    payloads: list[tuple[str, bytes]],
    parser: str,
    use_llm: bool,
    verify_claims: bool = False,
    verify_depth:  str  = "abstract",
) -> None:
    """Background worker — process every PDF in a batch.

    Runs in the shared ThreadPoolExecutor.  One run-time exception per PDF
    is recorded against that PDF without halting the rest; a process-wide
    failure (DB lost, OOM) marks the whole job 'failed'.
    """
    store = get_job_store()
    store.mark_running(job_id)
    opts  = PipelineOptions(parser=parser, use_llm=use_llm,
                            verify_claims=verify_claims,
                            verify_depth=verify_depth)
    try:
        for filename, pdf_bytes in payloads:
            try:
                report = process_pdf(pdf_bytes, opts)
                store.add_result(job_id, filename, report.to_dict())
            except Exception as exc:
                store.add_error(job_id, filename, str(exc))
        store.finish(job_id)
    except Exception as exc:
        store.finish(job_id, error=str(exc))


@app.post("/api/v1/batch", dependencies=[Depends(require_api_key)])
async def batch_start(
    files:         list[UploadFile] = File(...),
    parser:        str              = Form("grobid"),
    use_llm:       bool             = Form(False),
    verify_claims: bool             = Form(False),
    verify_depth:  str              = Form("abstract"),
) -> dict:
    """Submit a batch of PDFs for background processing.

    Returns immediately with a ``job_id``; poll
    ``GET /api/v1/batch/{job_id}`` for progress, then
    ``GET /api/v1/batch/{job_id}/results`` once status is ``done``.
    """
    _validate_parser(parser)
    if verify_depth not in _VALID_VERIFY_DEPTH:
        raise HTTPException(
            status_code=400,
            detail=f"unknown verify_depth {verify_depth!r}; "
                   f"expected one of {sorted(_VALID_VERIFY_DEPTH)}",
        )
    if not files:
        raise HTTPException(status_code=400, detail="no files in batch")
    if len(files) > _MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"batch too large: got {len(files)} files, limit is {_MAX_BATCH}",
        )

    payloads: list[tuple[str, bytes]] = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        if not data.startswith(b"%PDF"):
            # Be tolerant — skip non-PDFs with an error row rather than
            # rejecting the whole batch.  Lets the caller submit a mixed
            # directory without pre-filtering.
            payloads.append((f.filename or "unnamed", data))   # will surface as parser error
        else:
            payloads.append((f.filename or f"file-{len(payloads)}.pdf", data))

    if not payloads:
        raise HTTPException(status_code=400, detail="no usable files in batch")

    job_id = uuid.uuid4().hex[:16]
    get_job_store().create(job_id, n_total=len(payloads), parser=parser, use_llm=use_llm)
    _workers.submit(_run_batch_job, job_id, payloads, parser, use_llm,
                    verify_claims, verify_depth)

    return {"job_id": job_id, "n_total": len(payloads), "status": "queued"}


@app.get("/api/v1/batch/{job_id}", dependencies=[Depends(require_api_key)])
def batch_status(job_id: str) -> dict:
    meta = get_job_store().get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="job not found")
    return meta


@app.get("/api/v1/batch/{job_id}/results", dependencies=[Depends(require_api_key)])
def batch_results(job_id: str) -> dict:
    """Full per-PDF results.  Can be large for big batches — prefer
    the JSONL variant for shell pipelines."""
    out = get_job_store().get_results(job_id)
    if not out:
        raise HTTPException(status_code=404, detail="job not found")
    return out


@app.get(
    "/api/v1/batch/{job_id}/results.jsonl",
    dependencies=[Depends(require_api_key)],
)
def batch_results_jsonl(job_id: str) -> StreamingResponse:
    """Same as /results, streamed as JSONL — one row per PDF.

    Stream rather than build-then-send so the caller can start parsing
    while we still read from SQLite (matters for batches with hundreds
    of large reports).
    """
    out = get_job_store().get_results(job_id)
    if not out:
        raise HTTPException(status_code=404, detail="job not found")

    def _iter():
        # First line: the batch metadata (status, counts) so the caller
        # can render progress without a second request.
        meta = {k: v for k, v in out.items() if k != "results"}
        yield json.dumps({"_meta": meta}) + "\n"
        for row in out["results"]:
            yield json.dumps(row) + "\n"

    return StreamingResponse(_iter(), media_type="application/x-ndjson")


@app.get("/api/v1/batches", dependencies=[Depends(require_api_key)])
def batches_list(limit: int = 50) -> dict:
    return {"batches": get_job_store().list_recent(limit=max(1, min(limit, 500)))}
