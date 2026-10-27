#!/bin/bash
# att.sh <ladder.json> <arm> <kernel-regex> [extra rocprofv3 args...] -- thread trace (ATT) of one arm's kernel
# Runs `python3 -m hkl.one <ladder> <arm> -n 1 --warm 2` under rocprofv3 --att through run_gpu.sh (HKL_GPU set),
# tracing the dispatches whose kernel name matches <kernel-regex> (e.g. 'k_fwd_f03', 'k_hk_bwd_g4', 'k_dkdv96ds')
# on one CU (--att-target-cu, default 1) with instruction detail. Output: logs/att/<ladder>_<arm>/ (rocprofv3's
# ui / json output; the a08-1 notes read it with tools like hk/prof/tools/att_*.py: cycles per loop trip, WMMA
# issue share, wait classes). The trace makes the kernel slower; never time anything under it.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '2,8p' "$0"; exit 2; }
LADDER=$1; ARM=$2; RE=$3; shift 3
HERE=$(readlink -f "$(dirname "$0")/..")
NAME=$(basename "$LADDER" .json)_$ARM
OUT=$HERE/logs/att/$(dirname "$LADDER")_$NAME
mkdir -p "$OUT"
exec "$HERE/tools/run_gpu.sh" "att_$(dirname "$LADDER")_$NAME" 900 -- \
  rocprofv3 --att --att-target-cu "${HKL_ATT_CU:-1}" --kernel-include-regex "$RE" -d "$OUT" "$@" -- \
  python3 -m hkl.one "$LADDER" "$ARM" -n 1 --warm 2
