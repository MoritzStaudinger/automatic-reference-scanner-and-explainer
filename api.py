"""
ARSE HTTP API — drive the pipeline remotely and process PDFs in batch.

Endpoints
---------
GET  /healthz
    Liveness check.  Returns ``{"status": "ok"}``.  No auth.

POST /api/v1/process
    Sync single-PDF processing.  Multipart upload, returns JSON report.
    Fields: file (PDF), parser (str), use_llm (bool).
    Auth: X-API-Key header (when ARSE_API_KEY is set).
    Use this when you have one PDF and want the answer in the response.

POST /api/v1/batch
    Async batch processing.  Multipart with multiple files.  Returns
    a job_id immediately; the actual processing runs on a background
    thread pool sized by ARSE_API_WORKERS.
    Auth: X-API-Key header (when ARSE_API_KEY is set).
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
Set ``ARSE_API_KEY=<secret>`` in the environment and pass it via the
``X-API-Key`` header on every request.  When the env var is unset the
API runs in dev mode (no auth required) — convenient for local use,
do not deploy that way.

Running locally
---------------
::

    uvicorn api:app --host 0.0.0.0 --port 8500 --reload

Inside the docker-compose deployment the arse-api service runs uvicorn
on port 8500 and nginx routes ``/api/*`` to it.
"""

from __future__ import annotations

import arse_env  # noqa: F401  (accepts the old ARES_* settings)

import datetime
import hmac
import json
import logging
import os
import shutil
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

_log = logging.getLogger("arse.api")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_API_KEY        = os.getenv("ARSE_API_KEY", "").strip()
_KEYS_FILE      = Path(os.getenv(
    "ARSE_API_KEYS_FILE",
    Path(__file__).resolve().parent / "data" / "api_keys.json",
))
_BATCH_WORKERS  = max(1, int(os.getenv("ARSE_API_WORKERS", "2")))
# Without any key configured the API refuses requests (503) — the public
# deployment must never run it open by accident.  Local dev opts in.
_ALLOW_OPEN     = os.getenv("ARSE_API_ALLOW_UNAUTHENTICATED", "") == "1"
# Synchronous /process requests running at once; extra callers get 503.
_SYNC_SLOTS     = threading.BoundedSemaphore(int(os.getenv("ARSE_API_SYNC_CONCURRENCY", "2")))
_SYNC_WAIT_S    = float(os.getenv("ARSE_API_SYNC_WAIT_S", "30"))
_SPOOL_DIR      = Path(os.getenv("ARSE_API_SPOOL", Path(os.getenv(
    "ARSE_JOBS_DB", "data/jobs.sqlite")).parent / "batch_uploads"))
_MAX_BATCH      = int(os.getenv("ARSE_API_MAX_BATCH", "500"))
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

    Missing file → empty dict (dev mode falls back to ARSE_API_KEY env).
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
    thread_name_prefix="arse-batch",
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
    title="ARSE API",
    version="1.0",
    description=(
        "Automatic Reference Scanner and Explainer — programmatic interface "
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
      1. ``ARSE_API_KEY`` env var (single shared key — legacy path)
      2. A key entry in ``data/api_keys.json`` (multi-user with labels +
         optional expiry)

    When neither the env var nor the file has any keys, the API refuses
    every request with 503 — unless ARSE_API_ALLOW_UNAUTHENTICATED=1
    (local development only).  Keys are compared in constant time.
    """
    keys = _load_api_keys()
    if not _API_KEY and not keys:
        if _ALLOW_OPEN:
            return
        raise HTTPException(status_code=503,
                            detail="API access is not configured on this server")
    if not x_api_key:
        raise HTTPException(status_code=401,
                            detail="invalid or missing X-API-Key")
    given = x_api_key.encode()

    # Single-shared-key path (legacy).
    if _API_KEY and hmac.compare_digest(given, _API_KEY.encode()):
        return

    # File-backed path with metadata + optional expiry.
    meta = next((m for k, m in keys.items() if hmac.compare_digest(given, k.encode())), None)
    if meta is None:
        raise HTTPException(status_code=401,
                            detail="invalid or missing X-API-Key")
    exp = parse_expiry((meta or {}).get("expires_at"))
    if exp is not None and exp < datetime.datetime.now(datetime.timezone.utc):
        raise HTTPException(status_code=401, detail="API key expired")


def parse_expiry(value) -> Optional[datetime.datetime]:
    """ISO-8601 expiry as an aware UTC datetime.  Naive values are UTC.
    A malformed value counts as already expired — failing closed beats a
    key that silently never expires."""
    if not value:
        return None
    try:
        exp = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        _log.error("malformed expires_at %r — treating key as expired", value)
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=datetime.timezone.utc)
    return exp


def _check_pdf(data: bytes) -> Optional[str]:
    """Return an error message when ``data`` isn't a readable PDF."""
    if not data:
        return "empty upload"
    if not data.startswith(b"%PDF"):
        return "upload does not look like a PDF (missing %PDF header)"
    try:
        import fitz
        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.page_count == 0:
                return "PDF has no pages"
    except Exception:
        return "upload is not a readable PDF"
    return None


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
def process_one(
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
    # Plain `def`: FastAPI runs this in its threadpool, so a minutes-long
    # pipeline run no longer blocks the event loop (and /healthz with it).
    pdf_bytes = file.file.read()
    err = _check_pdf(pdf_bytes)
    if err:
        raise HTTPException(status_code=400, detail=err)

    if not _SYNC_SLOTS.acquire(timeout=_SYNC_WAIT_S):
        raise HTTPException(status_code=503,
                            detail="server busy — retry later or use /api/v1/batch",
                            headers={"Retry-After": "60"})
    try:
        report = process_pdf(
            pdf_bytes,
            PipelineOptions(parser=parser, use_llm=use_llm,
                            verify_claims=verify_claims,
                            verify_depth=verify_depth),
        )
    except Exception as exc:    # parser/LLM/lookup failures
        _log.exception("pipeline error for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"pipeline error: {type(exc).__name__}")
    finally:
        _SYNC_SLOTS.release()

    return {"filename": file.filename, **report.to_dict()}


# ---------------------------------------------------------------------------
# Async batch
# ---------------------------------------------------------------------------

def _run_batch_job(
    job_id: str,
    payloads: list[tuple[str, Path]],
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
        for filename, path in payloads:
            try:
                pdf_bytes = path.read_bytes()
                err = _check_pdf(pdf_bytes)
                if err:
                    raise ValueError(err)
                report = process_pdf(pdf_bytes, opts)
                store.add_result(job_id, filename, report.to_dict())
            except Exception as exc:
                store.add_error(job_id, filename, str(exc))
            finally:
                path.unlink(missing_ok=True)
        store.finish(job_id)
    except Exception as exc:
        store.finish(job_id, error=str(exc))
    finally:
        shutil.rmtree(_SPOOL_DIR / job_id, ignore_errors=True)


@app.post("/api/v1/batch", dependencies=[Depends(require_api_key)])
def batch_start(
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

    # Spool uploads to disk instead of holding up to 500 PDFs in memory
    # until the worker reaches them.  Non-PDFs get an error row in the
    # results rather than rejecting the whole batch.
    job_id = uuid.uuid4().hex[:16]
    job_dir = _SPOOL_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[tuple[str, Path]] = []
    for f in files:
        path = job_dir / f"{len(payloads):05d}.pdf"
        with path.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        if path.stat().st_size == 0:
            path.unlink()
            continue
        payloads.append((f.filename or f"file-{len(payloads)}.pdf", path))

    if not payloads:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="no usable files in batch")

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
