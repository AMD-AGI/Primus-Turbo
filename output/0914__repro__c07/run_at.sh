#!/bin/bash
# run_at.sh <repo_dir> <gpu> <tag> <ledger> -- <tune_attention args...>
set -u
R=$1; GPU=$2; TAG=$3; LEDGER=$4; shift 4; [ "${1:-}" = "--" ] && shift
out=$(timeout 3600 docker exec -e GPU=$GPU -e PYTHONPATH=/home/lihuzhan/code/aiter-src \
        fa-repro bash -lc 'cd "$0" && exec python3 tools/gfx1250/tune_attention.py "$@"' "$R" "$@" 2>/tmp/err_$(echo $TAG|tr '|/' '__')_g${GPU}.log | tail -1)
case "$out" in
  '{'*) echo "{\"tag\":\"$TAG\",\"gpu\":$GPU,\"repo\":\"$R\",\"r\":$out}" >> "$LEDGER"
        python3 -c "
import json;r=json.loads('''$out''')
print('%-32s gpu%s  fwd %7.3f  bwd %8.3f  total %8.3f ms  %7.1f TF/s  %s' % (
 '$TAG', $GPU, r['fwd_ms'], r['bwd_ms'], r['total_ms'], r['total_tflops'],
 'OK' if r.get('correct') else 'SQNR-FAIL '+repr(r.get('sqnr_db'))),
      '| seen tune=%r bwd=%s' % (r.get('tune'), r['configs']['bwd']))" ;;
  *) echo "{\"tag\":\"$TAG\",\"gpu\":$GPU,\"repo\":\"$R\",\"r\":null}" >> "$LEDGER"
     echo "$TAG gpu$GPU  FAILED -> /tmp/err_$(echo $TAG|tr '|/' '__')_g${GPU}.log" ;;
esac
