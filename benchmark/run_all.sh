#!/usr/bin/env bash
# Sequential evaluation runs for the conference-demo benchmark.
#
# Each config runs in cold-cache mode (REFCLEANER_CACHE → temp file) and
# logs to its own file under benchmark/.  Individual failures do NOT
# abort the chain — failed runs leave their partial output and we move
# on.  Use the per-run logs + summary block at the end to triage.
#
# Usage:
#   bash benchmark/run_all.sh           # cold runs of configs 1-3
#   bash benchmark/run_all.sh --with-gs # also include the Google Scholar config (slow)
#   bash benchmark/run_all.sh --warm    # ALSO run a warm-cache pass for hybrid+llm
#
# Total wall time (cold, 295 papers):
#   grobid        : ~1.5-2 h
#   hybrid        : ~2-2.5 h
#   hybrid+llm    : ~3-4 h
#   hybrid+llm+gs : ~6-10 h
# Plan for 7-10 h for the three non-GS configs; double that if --with-gs.
#
# Run overnight with `nohup`:
#   nohup bash benchmark/run_all.sh > benchmark/run_all.log 2>&1 &
#   tail -f benchmark/run_all.log

# NOTE: `set +e` so individual run failures don't abort the rest.
set +e
set -u

cd "$(dirname "$0")/.."   # project root regardless of CWD
ROOT="$(pwd)"
BENCH="$ROOT/benchmark"

WITH_GS=0
WITH_WARM=0
WITH_GEMMA=0
SMOKE=0
NO_INTER=0
TAG=""
QWEN_MODEL="qwen-3.8-27b"
GEMMA_MODEL="gemma-4-e2b-it"
for arg in "$@"; do
  case "$arg" in
    --with-gs)     WITH_GS=1 ;;
    --warm)        WITH_WARM=1 ;;
    --smoke)       SMOKE=1 ;;
    --with-gemma)  WITH_GEMMA=1 ;;
    --gemma-model=*) GEMMA_MODEL="${arg#--gemma-model=}" ;;
    --qwen-model=*)  QWEN_MODEL="${arg#--qwen-model=}" ;;
    --no-intermediates) NO_INTER=1 ;;
    --tag=*)       TAG="${arg#--tag=}" ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *) echo "unknown flag: $arg" ; exit 2 ;;
  esac
done

# Scale + intermediate-storage selection.
# Smoke: small spread sample, fast (~minutes per config).  Useful to
# verify the whole sequence end-to-end before committing to 10+ hours.
if [[ $SMOKE -eq 1 ]]; then
  SCALE_ARGS="--n 5"
  SCALE_TAG="smoke"
else
  SCALE_ARGS="--all"
  SCALE_TAG="full"
fi
if [[ $NO_INTER -eq 1 ]]; then
  INTER_ARGS=""
else
  INTER_ARGS="--save-intermediates"
fi

# All artefacts from this sequence land under a timestamped directory so
# re-running doesn't clobber the previous results.  A `latest` symlink
# points to the most recent run for convenience.
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_NAME="${TIMESTAMP}_${SCALE_TAG}${TAG:+_$TAG}"
OUTDIR="$BENCH/runs/$RUN_NAME"
mkdir -p "$OUTDIR"
ln -sfn "$RUN_NAME" "$BENCH/runs/latest"

# Top-level orchestration log captures the banner + summary tails.
ORCH_LOG="$OUTDIR/run_all.log"
exec > >(tee -a "$ORCH_LOG") 2>&1

echo "Output directory: $OUTDIR"
echo "Convenience symlink: $BENCH/runs/latest -> $RUN_NAME"

# Helper: log + run one config.  $1 = friendly name, $2 = output json,
# remaining args go to smoke_test.py.
run_one () {
  local NAME="$1"; shift
  local OUT="$1";  shift
  local LOG="${OUT%.json}.log"
  local START_TS
  START_TS=$(date +%s)

  echo
  echo "============================================================"
  echo "▶ START $NAME"
  echo "  out:   $OUT"
  echo "  log:   $LOG"
  echo "  start: $(date)"
  echo "============================================================"

  # `python -u`: force unbuffered stdout/stderr.  Without this, output
  # redirected to a file gets fully-buffered (8 KB block) and per-paper
  # log lines only appear ~20 min later when the buffer flushes — looks
  # like nothing's happening.
  python -u "$BENCH/smoke_test.py" "$@" --out "$OUT" \
    > "$LOG" 2>&1
  local RC=$?

  local END_TS DUR
  END_TS=$(date +%s)
  DUR=$(( END_TS - START_TS ))

  if [[ $RC -ne 0 ]]; then
    echo "✗ FAILED $NAME  (rc=$RC, ${DUR}s)  — continuing with next config"
  else
    echo "✓ DONE   $NAME  (${DUR}s = $((DUR/60)) min)"
    # Tail the summary block from the log so you see the headline numbers
    # without having to open the log file.
    echo "--- Summary tail ---"
    awk '/^=+/{p=1} p' "$LOG" | tail -40
    echo "--------------------"
  fi
  echo
}

OVERALL_START=$(date +%s)
echo "Benchmark sequence start: $(date)"
echo "Mode:          $SCALE_TAG  (args: $SCALE_ARGS $INTER_ARGS)"
echo "Configs queued:"
echo "  1. grobid       (cold, no LLM)"
echo "  2. hybrid       (cold, no LLM)"
echo "  3. hybrid + LLM ($QWEN_MODEL, cold)"
[[ $WITH_GEMMA -eq 1 ]] && echo "  3b. hybrid + LLM ($GEMMA_MODEL, cold)  [side-by-side comparison]"
[[ $WITH_GS    -eq 1 ]] && echo "  4. hybrid + LLM + Google Scholar (cold)"
[[ $WITH_WARM  -eq 1 ]] && echo "  W. hybrid + LLM (warm-cache, for cold/warm comparison)"
echo

# Make sure GROBID is reachable for runs 1-4.
echo "→ Checking GROBID at http://localhost:8070 ..."
if ! curl --silent --max-time 5 http://localhost:8070/api/isalive > /dev/null; then
  echo "⚠️  GROBID not responding.  Trying 'docker compose up -d grobid' ..."
  (cd "$ROOT" && docker compose up -d grobid) || true
  sleep 10
  if ! curl --silent --max-time 5 http://localhost:8070/api/isalive > /dev/null; then
    echo "✗ Could not reach GROBID — grobid/hybrid configs will fall back to pymupdf4llm."
  fi
fi

# --- Run 1: grobid only --------------------------------------------------
run_one "grobid (cold, no LLM)" \
  "$OUTDIR/results_grobid_cold.json" \
  $SCALE_ARGS $INTER_ARGS --parser grobid --no-llm --cold

# --- Run 2: hybrid only --------------------------------------------------
run_one "hybrid (cold, no LLM)" \
  "$OUTDIR/results_hybrid_cold.json" \
  $SCALE_ARGS $INTER_ARGS --parser hybrid --no-llm --cold

# --- Run 3: hybrid + LLM (qwen) ------------------------------------------
run_one "hybrid + LLM ($QWEN_MODEL, cold)" \
  "$OUTDIR/results_hybrid_llm_${QWEN_MODEL//[^a-zA-Z0-9-]/_}_cold.json" \
  $SCALE_ARGS $INTER_ARGS --parser hybrid --llm \
  --llm-backend aqueduct --llm-model "$QWEN_MODEL" --cold

# --- Run 3b (optional): hybrid + LLM (gemma) for side-by-side ------------
if [[ $WITH_GEMMA -eq 1 ]]; then
  run_one "hybrid + LLM ($GEMMA_MODEL, cold)" \
    "$OUTDIR/results_hybrid_llm_${GEMMA_MODEL//[^a-zA-Z0-9-]/_}_cold.json" \
    $SCALE_ARGS $INTER_ARGS --parser hybrid --llm \
    --llm-backend aqueduct --llm-model "$GEMMA_MODEL" --cold
fi

# --- Run 4 (optional): hybrid + LLM + Google Scholar ---------------------
if [[ $WITH_GS -eq 1 ]]; then
  run_one "hybrid + LLM + Google Scholar (cold)" \
    "$OUTDIR/results_hybrid_llm_gs_cold.json" \
    $SCALE_ARGS $INTER_ARGS --parser hybrid --llm --scholarly 30 --cold
fi

# --- Run W (optional): warm-cache hybrid+LLM for cold/warm comparison ----
if [[ $WITH_WARM -eq 1 ]]; then
  run_one "hybrid + LLM ($QWEN_MODEL, warm cache)" \
    "$OUTDIR/results_hybrid_llm_${QWEN_MODEL//[^a-zA-Z0-9-]/_}_warm.json" \
    $SCALE_ARGS $INTER_ARGS --parser hybrid --llm \
    --llm-backend aqueduct --llm-model "$QWEN_MODEL"
fi

# --- Overall tally -------------------------------------------------------
OVERALL_END=$(date +%s)
OVERALL_DUR=$(( OVERALL_END - OVERALL_START ))
echo
echo "============================================================"
echo "ALL RUNS COMPLETE"
echo "  total elapsed: $((OVERALL_DUR/60)) min ($(($OVERALL_DUR/3600))h $(((OVERALL_DUR%3600)/60))m)"
echo "  finished at:   $(date)"
echo "============================================================"
echo
echo "All artefacts live under: $OUTDIR"
echo
echo "Files written:"
ls -lh "$OUTDIR"/ 2>/dev/null

# Side-by-side comparison table — extract the key numbers from each
# results_*.json via python and print one row per config.
echo
echo "------------------------------------------------------------"
echo "Cross-config comparison (from results_*.json summary blocks)"
echo "------------------------------------------------------------"
python - "$OUTDIR" <<'PY'
import json, sys
from pathlib import Path

outdir = Path(sys.argv[1])
rows = []
for path in sorted(outdir.glob("results_*.json")):
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        print(f"  {path.name}: unreadable ({e})")
        continue
    s = d.get("summary", {})
    c = d.get("config", {})
    rows.append({
        "config":    path.stem.replace("results_", ""),
        "parser":    c.get("parser"),
        "llm":       "Y" if c.get("use_llm") else "N",
        "gs":        c.get("scholarly_budget") or 0,
        "cold":      "Y" if c.get("cold_cache") else "N",
        "papers":    s.get("papers", 0),
        "located":   s.get("hallucination_located", 0),
        "caught":    s.get("caught", 0),
        "recall":    s.get("labelled_recall_pct", 0.0),
        "ci":        s.get("labelled_recall_ci95", [0.0, 0.0]),
        "median_wall_s": (s.get("timing", {}).get("paper_wall_s", {}) or {}).get("median"),
        "p95_wall_s":    (s.get("timing", {}).get("paper_wall_s", {}) or {}).get("p95"),
        "elapsed_min":   round(d.get("elapsed_sec", 0) / 60, 1),
    })

if not rows:
    print("  (no results files found)")
else:
    h = f"  {'config':<32} prs llm gs cold  recall (95% CI)        median  p95   total"
    print(h)
    print("  " + "-" * (len(h) - 2))
    for r in rows:
        lo, hi = r["ci"]
        print(f"  {r['config']:<32} {r['parser'][:3]:<3} {r['llm']:<3} "
              f"{str(r['gs']):<2} {r['cold']:<4} "
              f"{r['recall']:>5.1f}% ({lo:>4.1f}-{hi:>4.1f})  "
              f"{(str(r['median_wall_s'])+'s'):<7} "
              f"{(str(r['p95_wall_s'])+'s'):<5} "
              f"{r['elapsed_min']:>5.1f}m")
PY

# Reminder about temp cache cleanup
echo
echo "Cold-cache temp dirs (safe to delete now):"
ls -ld /tmp/refcleaner_cold_* 2>/dev/null | head
echo "  → cleanup: rm -rf /tmp/refcleaner_cold_*"
