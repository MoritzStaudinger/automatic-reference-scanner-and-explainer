"""
ARSE batch CLI — process a directory of PDFs locally without the API.

Two modes:

Local (default) — call the pipeline directly, no HTTP.  Best when you're
running on the same host as the model + cache.

    python batch_cli.py <input_dir> <output_dir> [--parser grobid] [--use-llm]
                       [--concurrency 4] [--glob '*.pdf']

Remote — POST PDFs to a running ARSE API and collect results.  Best when
the API runs on a separate machine (e.g. the institutional VM).

    python batch_cli.py <input_dir> <output_dir> --remote https://host/api/v1
                       --api-key $ARES_API_KEY

Output: one ``<stem>.json`` per input PDF in the output directory, plus
a ``summary.csv`` rolling everything up (n_match / n_mismatch / n_missing
per paper).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Local mode — direct pipeline call
# ---------------------------------------------------------------------------

def _process_local(pdf_path: Path, parser: str, use_llm: bool) -> dict[str, Any]:
    """Import lazily so --remote runs don't pay the parser-import cost."""
    from pipeline import PipelineOptions, process_pdf
    pdf_bytes = pdf_path.read_bytes()
    report = process_pdf(pdf_bytes, PipelineOptions(parser=parser, use_llm=use_llm))
    return {"filename": pdf_path.name, **report.to_dict()}


# ---------------------------------------------------------------------------
# Remote mode — talk to the ARSE HTTP API
# ---------------------------------------------------------------------------

def _process_remote(
    pdf_path: Path,
    api_base: str,
    api_key: Optional[str],
    parser: str,
    use_llm: bool,
) -> dict[str, Any]:
    import requests
    headers = {"X-API-Key": api_key} if api_key else {}
    with pdf_path.open("rb") as f:
        r = requests.post(
            api_base.rstrip("/") + "/process",
            headers=headers,
            files={"file": (pdf_path.name, f, "application/pdf")},
            # `1`/`0` mirrors the env-var convention used elsewhere in the
            # project (ARES_PDF_SELECTOR=1, etc.).  FastAPI/Pydantic
            # accepts 1, 0, true, false, yes, no, on, off as boolean form
            # values — either spelling works, but staying consistent with
            # the shell-flag convention keeps the wire format predictable.
            data={"parser": parser, "use_llm": "1" if use_llm else "0"},
            timeout=600,
        )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="arse-batch",
        description="Batch-process PDFs through the ARSE pipeline.",
    )
    p.add_argument("input_dir",  type=Path, help="Directory of PDFs to process.")
    p.add_argument("output_dir", type=Path, help="Where to write per-PDF JSON and summary.csv.")
    p.add_argument("--parser",   default="grobid",
                   choices=["grobid", "hybrid", "pymupdf4llm"],
                   help="Parser backend.")
    p.add_argument("--use-llm",  action="store_true",
                   help="Run LLM-based field repair after parsing.")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Number of PDFs to process in parallel.")
    p.add_argument("--glob",     default="*.pdf",
                   help="File pattern in input_dir (default *.pdf).")
    p.add_argument("--remote",   default=None,
                   help=("Base URL of an ARSE API (e.g. "
                         "https://reference-cleaner.ds-ifs.tuwien.ac.at/api/v1). "
                         "When unset, runs the pipeline locally."))
    p.add_argument("--api-key",  default=os.getenv("ARES_API_KEY"),
                   help="X-API-Key for the remote ARSE API.  Defaults to $ARES_API_KEY.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing per-PDF JSONs in output_dir.")
    args = p.parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"input_dir {args.input_dir} is not a directory", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(args.input_dir.glob(args.glob))
    if not pdfs:
        print(f"no files matching {args.glob} in {args.input_dir}", file=sys.stderr)
        return 2

    # Skip already-done PDFs unless --overwrite
    todo: list[Path] = []
    for pdf in pdfs:
        out_json = args.output_dir / (pdf.stem + ".json")
        if out_json.exists() and not args.overwrite:
            continue
        todo.append(pdf)

    if not todo:
        print(f"all {len(pdfs)} PDFs already processed; pass --overwrite to redo")
        return 0

    mode = "remote" if args.remote else "local"
    print(f"Processing {len(todo)} of {len(pdfs)} PDFs ({mode}, "
          f"parser={args.parser}, use_llm={args.use_llm}, "
          f"concurrency={args.concurrency})")

    def _do_one(pdf: Path) -> tuple[Path, Optional[dict[str, Any]], Optional[str]]:
        try:
            if args.remote:
                report = _process_remote(
                    pdf, args.remote, args.api_key, args.parser, args.use_llm)
            else:
                report = _process_local(pdf, args.parser, args.use_llm)
            return pdf, report, None
        except Exception as exc:
            return pdf, None, str(exc)[:500]

    summary_rows: list[dict[str, Any]] = []
    n_ok = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(_do_one, pdf): pdf for pdf in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            pdf, report, error = fut.result()
            out_json = args.output_dir / (pdf.stem + ".json")
            if report is not None:
                out_json.write_text(json.dumps(report, indent=2))
                n_ok += 1
                s = report.get("summary", {}) or {}
                summary_rows.append({
                    "filename":   pdf.name,
                    "n_refs":     s.get("n_refs", 0),
                    "n_match":    s.get("n_match", 0),
                    "n_mismatch": s.get("n_mismatch", 0),
                    "n_missing":  s.get("n_missing", 0),
                    "parser":     s.get("parser", args.parser),
                    "used_llm":   s.get("used_llm", args.use_llm),
                })
                tag = "ok"
            else:
                out_json.write_text(json.dumps({"error": error}, indent=2))
                n_err += 1
                summary_rows.append({
                    "filename":   pdf.name,
                    "n_refs":     "",
                    "n_match":    "",
                    "n_mismatch": "",
                    "n_missing":  "",
                    "parser":     args.parser,
                    "used_llm":   args.use_llm,
                })
                tag = f"ERROR: {error[:80]}"
            print(f"  [{i}/{len(todo)}] {pdf.name}: {tag}")

    # Summary CSV — sorted by filename so re-runs produce stable diffs.
    summary_path = args.output_dir / "summary.csv"
    summary_rows.sort(key=lambda r: r["filename"])
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    elapsed = time.time() - t0
    print()
    print(f"Done: {n_ok} ok, {n_err} errors in {elapsed:.1f}s "
          f"({elapsed/len(todo):.1f}s/PDF)")
    print(f"Per-PDF JSON: {args.output_dir}/")
    print(f"Summary:      {summary_path}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
