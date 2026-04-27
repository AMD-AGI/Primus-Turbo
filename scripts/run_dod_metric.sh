#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Run the "all FP8/BF16 GEMM/grouped_GEMM tests must pass" DoD suite, picking
# idle GPUs at runtime so we never collide with other tenants on this box.
#
# Output: a single integer (pass count) on stdout when --metric is passed
# (default), or the raw pytest output when --raw is passed. Designed to be
# consumed by scripts/auto_optimize.py as its --metric-cmd / --deterministic-cmd.
#
# Flags:
#   --deterministic   Pass --deterministic-only to pytest (DoD second half).
#   --raw             Print full pytest output instead of just pass count.
#   --max-workers N   Cap the number of xdist workers (default: # idle GPUs).
#
# Env overrides:
#   HIPKITTEN_PATH    Defaults to /workspace/code/HipKittens.
#   IDLE_GPUS         Override auto-detection ("0,2,4" style).
###############################################################################

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DETERMINISTIC=0
RAW=0
MAX_WORKERS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
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
    # Falls back to "0" if rocm-smi is unavailable / parse fails.
    python3 - <<'PYEOF' 2>/dev/null || echo 0
import os, re, subprocess, sys
THR = int(os.environ.get("VRAM_BUSY_THRESHOLD", 100 * 1024 * 1024))
try:
    out = subprocess.check_output(
        ["rocm-smi", "--showuse", "--showpids"],
        stderr=subprocess.DEVNULL, text=True, timeout=10,
    )
except Exception:
    print("0")
    sys.exit(0)
# All GPU IDs visible to rocm-smi.
all_gpus = sorted({int(m) for m in re.findall(r"^GPU\[(\d+)\]", out, flags=re.M)})
# Parse KFD process table to find busy GPUs (those with a PID using > THR VRAM).
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
    # Pick the GPU with smallest VRAM usage among "all" as last resort.
    idle = all_gpus[:1] or [0]
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

echo "[run_dod_metric] idle GPUs=${DEVICES} workers=${NPROC} deterministic=${DETERMINISTIC}" >&2

PYTEST_FILES=(
    "$REPO_ROOT/tests/pytorch/ops/test_gemm.py"
    "$REPO_ROOT/tests/pytorch/ops/test_gemm_fp8.py"
    "$REPO_ROOT/tests/pytorch/ops/test_grouped_gemm.py"
    "$REPO_ROOT/tests/pytorch/ops/test_grouped_gemm_fp8.py"
)

PYTEST_CMD=(python3 -m pytest "${PYTEST_FILES[@]}" --tb=no -q)
if [[ "$NPROC" -gt 1 ]]; then
    PYTEST_CMD+=( -n "$NPROC" )
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
