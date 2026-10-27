#!/bin/bash
# ktrace.sh <ladder.json> <arm> -- per-kernel time of one arm: `python3 -m hkl.one <ladder> <arm> -n 5` under
# rocprofv3 --kernel-trace --stats through run_gpu.sh (HKL_GPU set). Output: logs/ktrace/<ladder>_<arm>/ (rocprofv3
# CSVs; *kernel_stats.csv has calls, total / mean ns per kernel). Light-weight (no thread trace); the warm calls
# are included in the stats.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,5p' "$0"; exit 2; }
LADDER=$1; ARM=$2; shift 2
HERE=$(readlink -f "$(dirname "$0")/..")
NAME=$(dirname "$LADDER")_$(basename "$LADDER" .json)_$ARM
OUT=$HERE/logs/ktrace/$NAME
rm -rf "$OUT"; mkdir -p "$OUT"
exec "$HERE/tools/run_gpu.sh" "ktrace_$NAME" 900 -- \
  rocprofv3 --kernel-trace --stats -d "$OUT" -o kt --output-format csv "$@" -- \
  python3 -m hkl.one "$LADDER" "$ARM" -n 5 --warm 2
