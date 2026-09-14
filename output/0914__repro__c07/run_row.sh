#!/bin/bash
# Run one ladder row on one GPU, append a tagged JSON line to the ledger.
#   run_row.sh <gpu> <tag> <ledger> -- <tune_attention args...>
set -u
GPU=$1; TAG=$2; LEDGER=$3; shift 3; [ "${1:-}" = "--" ] && shift
R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
out=$(timeout 2400 docker exec -e GPU=$GPU -e PYTHONPATH=/home/lihuzhan/code/aiter-src \
        fa-repro bash -lc 'cd "$0" && exec python3 tools/gfx1250/tune_attention.py "$@"' "$R" "$@" 2>/tmp/err_${TAG}_g${GPU}.log | tail -1)
case "$out" in
  '{'*) echo "{\"tag\":\"$TAG\",\"gpu\":$GPU,\"r\":$out}" >> "$LEDGER"
        python3 -c "
import json;r=json.loads('''$out''')
print('%-34s gpu%s  fwd %7.3f  bwd %8.3f  total %8.3f ms  %7.1f TFLOP/s  %s' % (
 '$TAG', $GPU, r['fwd_ms'], r['bwd_ms'], r['total_ms'], r['total_tflops'],
 'OK' if r.get('correct') else 'SQNR-FAIL'),
      '| seen tune=%r bwd=%s' % (r.get('tune'), r['configs']['bwd']))" ;;
  *) echo "{\"tag\":\"$TAG\",\"gpu\":$GPU,\"r\":null}" >> "$LEDGER"
     echo "$TAG gpu$GPU  FAILED (see /tmp/err_${TAG}_g${GPU}.log)" ;;
esac
