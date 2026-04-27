#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Fast iteration metric for auto_optimize.py: runs ONLY the HipKitten-tagged
# pytest cases in test_gemm.py + test_grouped_gemm.py (where HIPKITTEN tests
# live) so each round finishes in tens of seconds, not minutes. Picks idle
# GPUs at runtime via rocm-smi so we never collide with other tenants.
#
# The broader "all 4 files green" DoD bar is the **final acceptance gate**
# (run with --full); the loop itself maximizes pass count over HipKitten
# tests only.
#
# Output: single integer score on stdout (default), or raw pytest output
# (--raw). Designed to be consumed by scripts/auto_optimize.py.
#
# Flags:
#   --full            Run the full DoD suite (4 files, no -k filter). Slow.
#                     Use this for final acceptance, not for the loop metric.
#   --deterministic   Add --deterministic-only to pytest.
#   --raw             Print full pytest output instead of score.
#   --max-workers N   Cap the number of xdist workers (default: # idle GPUs).
#
# Env overrides:
#   HIPKITTEN_PATH    Defaults to /workspace/code/HipKittens.
#   IDLE_GPUS         Override auto-detection ("0,2,4" style).
#   PYTEST_K          Override the -k expression (default: "hipkitten").
###############################################################################

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DETERMINISTIC=0
RAW=0
FULL=0
MAX_WORKERS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)          FULL=1; shift ;;
        --deterministic) DETERMINISTIC=1; shift ;;
        --raw)           RAW=1; shift ;;
        --max-workers)   MAX_WORKERS="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)
            echo "[run_dod_metric] unknown arg: $1" >&2; exit 2 ;;
    esac
done

HIPKITTEN_PATH="${HIPKITTEN_PATH:-/workspace/code/HipKittens}"

pick_idle_gpus() {
    # Returns comma-separated IDs of GPUs whose KFD process table shows
    # no PID using more than VRAM_BUSY_THRESHOLD bytes (default 100 MiB).
    # Honors HIPKITTEN_GPU_POOL='0,1,2,3' (or via env) to restrict to a subset.
    # Falls back to "0" if rocm-smi is unavailable / parse fails.
    python3 - <<'PYEOF' 2>/dev/null || echo 0
import os, re, subprocess, sys
THR = int(os.environ.get("VRAM_BUSY_THRESHOLD", 100 * 1024 * 1024))
pool_raw = os.environ.get("HIPKITTEN_GPU_POOL", "").strip()
pool = None
if pool_raw:
    pool = set()
    for tok in pool_raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            pool.add(int(tok))
        except ValueError:
            pass
    if not pool:
        pool = None
try:
    out = subprocess.check_output(
        ["rocm-smi", "--showuse", "--showpids"],
        stderr=subprocess.DEVNULL, text=True, timeout=10,
    )
except Exception:
    if pool:
        print(",".join(str(g) for g in sorted(pool)))
    else:
        print("0")
    sys.exit(0)
all_gpus = sorted({int(m) for m in re.findall(r"^GPU\[(\d+)\]", out, flags=re.M)})
if pool is not None:
    all_gpus = [g for g in all_gpus if g in pool]
busy = set()
in_kfd = False
for line in out.splitlines():
    if "KFD process information" in line:
        in_kfd = True
        continue
    if not in_kfd:
        continue
    if line.startswith("=") or "PROCESS NAME" in line:
        continue
    cols = line.split()
    if len(cols) < 4 or not cols[0].isdigit():
        continue
    try:
        vram = int(cols[3])
    except ValueError:
        continue
    if vram <= THR:
        continue
    for gid in re.findall(r"\d+", cols[2]):
        busy.add(int(gid))
idle = [g for g in all_gpus if g not in busy]
if not idle:
    # Last resort: use the pool itself, else first known GPU.
    idle = all_gpus[:1] or (sorted(pool)[:1] if pool else [0])
print(",".join(str(g) for g in idle))
PYEOF
}

if [[ -n "${IDLE_GPUS:-}" ]]; then
    DEVICES="$IDLE_GPUS"
else
    DEVICES="$(pick_idle_gpus)"
fi
NPROC=$(awk -F, '{print NF}' <<<"$DEVICES")
if [[ -n "$MAX_WORKERS" && "$NPROC" -gt "$MAX_WORKERS" ]]; then
    NPROC=$MAX_WORKERS
    DEVICES=$(awk -v n="$NPROC" -F, '{ for(i=1;i<=n;i++) printf "%s%s", $i, (i<n?",":"") }' <<<"$DEVICES")
fi

echo "[run_dod_metric] idle GPUs=${DEVICES} workers=${NPROC} deterministic=${DETERMINISTIC} full=${FULL}" >&2

if [[ "$FULL" -eq 1 ]]; then
    PYTEST_FILES=(
        "$REPO_ROOT/tests/pytorch/ops/test_gemm.py"
        "$REPO_ROOT/tests/pytorch/ops/test_gemm_fp8.py"
        "$REPO_ROOT/tests/pytorch/ops/test_grouped_gemm.py"
        "$REPO_ROOT/tests/pytorch/ops/test_grouped_gemm_fp8.py"
    )
else
    # Loop-mode: only the files that contain HIPKITTEN-tagged cases.
    PYTEST_FILES=(
        "$REPO_ROOT/tests/pytorch/ops/test_gemm.py"
        "$REPO_ROOT/tests/pytorch/ops/test_grouped_gemm.py"
    )
fi
KEXPR="${PYTEST_K:-hipkitten}"

PYTEST_CMD=(python3 -m pytest "${PYTEST_FILES[@]}" --tb=no -q)
if [[ "$FULL" -eq 0 ]]; then
    PYTEST_CMD+=( -k "$KEXPR" )
fi
# Cap parallelism by # of selected tests for the narrow loop run; xdist only
# helps if we have multiple cases per worker. For the HipKitten loop this
# rarely needs >2 workers.
if [[ "$NPROC" -gt 1 && "$FULL" -eq 1 ]]; then
    PYTEST_CMD+=( -n "$NPROC" )
elif [[ "$NPROC" -gt 1 && "$FULL" -eq 0 ]]; then
    LOOPN=$NPROC
    [[ "$LOOPN" -gt 2 ]] && LOOPN=2
    PYTEST_CMD+=( -n "$LOOPN" )
fi
if [[ "$DETERMINISTIC" -eq 1 ]]; then
    PYTEST_CMD+=( --deterministic-only )
fi

export PRIMUS_TURBO_HIPKITTEN_PATH="$HIPKITTEN_PATH"
export HIP_VISIBLE_DEVICES="$DEVICES"

cd "$REPO_ROOT"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
"${PYTEST_CMD[@]}" >"$TMP" 2>&1
RC=$?

if [[ "$RAW" -eq 1 ]]; then
    cat "$TMP"
    exit $RC
fi

# Extract "<N> passed" from the last summary line; if not found print 0 so
# the metric stays comparable.
PASS=$(grep -oE '[0-9]+ passed' "$TMP" | tail -1 | grep -oE '[0-9]+' || true)
FAIL=$(grep -oE '[0-9]+ failed' "$TMP" | tail -1 | grep -oE '[0-9]+' || true)
ERR=$(grep -oE '[0-9]+ errors?' "$TMP" | tail -1 | grep -oE '[0-9]+' || true)
PASS=${PASS:-0}
FAIL=${FAIL:-0}
ERR=${ERR:-0}

echo "[run_dod_metric] passed=${PASS} failed=${FAIL} errors=${ERR} pytest_rc=${RC}" >&2

# Penalize failures by subtracting them so any regression drops the metric.
# A run with N passed and 0 failed > a run with N passed and K failed.
SCORE=$(( PASS - 1000 * FAIL - 1000 * ERR ))
echo "$SCORE"
